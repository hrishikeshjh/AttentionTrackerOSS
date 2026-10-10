/**
 * Screen-facing pitch reference θ_ref (Annex A.1, B.1).
 *
 *  - θ_ref is the median pitch over a 5-second screen-facing reference.
 *  - The reference is re-estimated after the student returns from being
 *    absent, and after a camera/laptop move (sudden jump in the 1 s median
 *    pitch or face scale; thresholds are TBD on P1).
 *  - When PEB is enabled, a BaselineProvider supplies the reference instead.
 *
 * While a reference is being (re-)estimated, `calibrating` is true and the
 * running median of the samples collected so far is used.
 *
 * Sample gate (implementation choice): a frame contributes to the reference
 * only if the eyes are not looking down (g_t < γ_down or gaze unavailable),
 * so a student glancing at notes during calibration does not bias θ_ref.
 */
import type { DaceParams } from './params';

/** PEB hook (interface only in Stage 1). Return null to fall back to the internal estimate. */
export interface BaselineProvider {
  pitchReferenceDeg(): number | null;
  /** personal open-eye level (replaces β0 in the openness model; see eye-openness.ts) */
  openEyeLevel?(): number | null;
}

export type PitchReferenceReset = 'start' | 'absent_return' | 'camera_move';

export type PitchReferenceState = {
  refDeg: number | null;
  relPitchDeg: number | null;
  calibrating: boolean;
  /** set on the frame on which a reset happened */
  reset: PitchReferenceReset | null;
  source: 'internal' | 'peb';
};

export function median(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export type PitchReferenceInput = {
  t: number;
  /** null when no face */
  pitchDeg: number | null;
  /** any scale proxy (e.g. face box width or interocular distance); null if unknown */
  faceScale: number | null;
  gazeV: number | null;
};

export class PitchReference {
  private refDeg: number | null = null;
  private collecting = true;
  private collectStart: number | null = null;
  private samples: number[] = [];
  private missingRun = 0;
  private absentPending = false;
  private pendingReset: PitchReferenceReset | null = 'start';
  private block: { pitch: number[]; scale: number[] } = { pitch: [], scale: [] };
  private prevBlock: { pitch: number | null; scale: number | null } | null = null;
  private readonly framesPerBlock: number;
  private readonly absentFrames: number;

  constructor(
    private readonly params: DaceParams,
    private readonly baseline: BaselineProvider | null = null
  ) {
    this.framesPerBlock = Math.max(1, Math.round(params.fps)); // 1 s blocks
    this.absentFrames = Math.ceil(params.absentShare * params.windowFrames - 1e-9);
  }

  /** Explicit absent signal (e.g. the rule layer output `absent`). */
  markAbsent(): void {
    this.absentPending = true;
  }

  private startCollection(reason: PitchReferenceReset): void {
    this.collecting = true;
    this.collectStart = null;
    this.samples = [];
    this.pendingReset = reason;
    this.prevBlock = null;
    this.block = { pitch: [], scale: [] };
  }

  update(input: PitchReferenceInput): PitchReferenceState {
    const { t, pitchDeg, faceScale, gazeV } = input;
    let reset: PitchReferenceReset | null = null;

    if (pitchDeg === null) {
      this.missingRun++;
      if (this.missingRun >= this.absentFrames) this.absentPending = true;
      return this.state(null, null);
    }

    // Face is back after an absence -> re-estimate.
    if (this.absentPending) {
      this.absentPending = false;
      this.startCollection('absent_return');
    }
    this.missingRun = 0;

    // Camera-move detection on consecutive 1 s blocks (only once a reference exists).
    if (!this.collecting) {
      this.block.pitch.push(pitchDeg);
      if (faceScale !== null) this.block.scale.push(faceScale);
      if (this.block.pitch.length >= this.framesPerBlock) {
        const cur = { pitch: median(this.block.pitch), scale: median(this.block.scale) };
        if (this.prevBlock) {
          const pitchJump =
            cur.pitch !== null && this.prevBlock.pitch !== null &&
            Math.abs(cur.pitch - this.prevBlock.pitch) >= this.params.cameraMovePitchJumpDeg;
          const scaleJump =
            cur.scale !== null && this.prevBlock.scale !== null && this.prevBlock.scale > 0 &&
            Math.abs(cur.scale / this.prevBlock.scale - 1) >= this.params.cameraMoveFaceScaleJump;
          if (pitchJump || scaleJump) this.startCollection('camera_move');
        }
        if (!this.collecting) this.prevBlock = cur;
        this.block = { pitch: [], scale: [] };
      }
    }

    if (this.collecting) {
      if (this.collectStart === null) this.collectStart = t;
      const eyesDown = gazeV !== null && gazeV >= this.params.gammaDown;
      if (!eyesDown) this.samples.push(pitchDeg);
      if (t - this.collectStart >= this.params.referenceMs && this.samples.length > 0) {
        this.refDeg = median(this.samples);
        this.collecting = false;
        this.samples = [];
      }
    }

    if (this.pendingReset) {
      reset = this.pendingReset;
      this.pendingReset = null;
    }
    return this.state(pitchDeg, reset);
  }

  private state(pitchDeg: number | null, reset: PitchReferenceReset | null): PitchReferenceState {
    const peb = this.baseline?.pitchReferenceDeg() ?? null;
    if (peb !== null && Number.isFinite(peb)) {
      return {
        refDeg: peb,
        relPitchDeg: pitchDeg === null ? null : pitchDeg - peb,
        calibrating: false,
        reset,
        source: 'peb',
      };
    }
    const ref = this.collecting ? (median(this.samples) ?? this.refDeg) : this.refDeg;
    return {
      refDeg: ref,
      relPitchDeg: pitchDeg === null || ref === null ? null : pitchDeg - ref,
      calibrating: this.collecting,
      reset,
      source: 'internal',
    };
  }

  getReference(): number | null {
    return this.refDeg;
  }

  isCalibrating(): boolean {
    return this.collecting;
  }

  reset(): void {
    this.refDeg = null;
    this.missingRun = 0;
    this.absentPending = false;
    this.startCollection('start');
  }
}
