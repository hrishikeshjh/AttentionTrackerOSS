/** Test builders for DACE frames, hands, objects and window summaries. */
import type { DetectedObject, HandResult, NormalizedBBox } from '../../../types';
import type { DaceFace, DaceFrame, WindowSummary } from '../types';
import { makeParams, type DaceParams } from '../params';

export const P: DaceParams = makeParams();
export const DT = 200;

export const FACE_BOX: NormalizedBBox = { x: 0.5, y: 0.35, width: 0.25, height: 0.35 };

export function face(o: Partial<DaceFace> = {}): DaceFace {
  return {
    pitchDeg: 0,
    relPitchDeg: 0,
    yawDeg: 0,
    rollDeg: 0,
    gazeV: 0,
    gazeH: 0,
    gazeValid: true,
    ear: 0.3,
    residualOpenness: 1,
    eyeClosed: false,
    faceBox: FACE_BOX,
    ...o,
  };
}

/** Face looking down by head pitch (Δθ = 25°). */
export const down = (o: Partial<DaceFace> = {}) => face({ relPitchDeg: 25, ...o });

/**
 * 21-landmark hand. Wrist (0) at (cx, cy + 0.06), middle MCP (9) at (cx, cy):
 * hand length 0.06. Thumb tip (4) / index tip (8) separated vertically by
 * `tipGap` (pinch ratio = tipGap / 0.06). Other landmarks fill a
 * 0.06 x 0.08 box around the centre.
 */
export function hand(
  handedness: 'Left' | 'Right',
  cx: number,
  cy: number,
  opts: { pinch?: boolean; tipGap?: number } = {}
): HandResult {
  const gap = opts.tipGap ?? (opts.pinch ? 0.006 : 0.05);
  const lm = [];
  for (let i = 0; i < 21; i++) {
    const a = (i / 21) * 2 * Math.PI;
    lm.push({ x: cx + 0.03 * Math.cos(a), y: cy + 0.04 * Math.sin(a), z: 0 });
  }
  lm[0] = { x: cx, y: cy + 0.06, z: 0 };
  lm[9] = { x: cx, y: cy, z: 0 };
  lm[4] = { x: cx + 0.01, y: cy - 0.03, z: 0 };
  lm[8] = { x: cx + 0.01, y: cy - 0.03 - gap, z: 0 };
  return { landmarks: lm, handedness };
}

export function obj(cls: DetectedObject['class'], x: number, y: number, w = 0.06, h = 0.1, confidence = 0.9): DetectedObject {
  return { class: cls, confidence, bbox: { x, y, width: w, height: h } };
}

export function frame(i: number, f: DaceFace | null, hands: HandResult[] = [], objects: DetectedObject[] = []): DaceFrame {
  return { t: 1_000_000 + i * DT, face: f, hands, objects };
}

/** Writing hand: pinch + wrist oscillating by 0.003 (0.05 hand lengths) per frame. */
export function writingHand(handedness: 'Left' | 'Right', i: number, cx = 0.6, cy = 0.8): HandResult {
  return hand(handedness, cx, cy + (i % 2 ? 0.0015 : -0.0015), { pinch: true });
}

export function summary(o: Partial<WindowSummary> = {}): WindowSummary {
  return {
    t: 1_000_000,
    n: 15,
    noFaceCount: 0,
    lookingDownCount: 0,
    lookingDownMost: false,
    phoneInUseCount: 0,
    phoneDetectedCount: 0,
    phoneInUse: false,
    phonePresent: false,
    phoneHeldAny: false,
    bookEvidenceCount: 0,
    book: false,
    penGripMotion: {},
    writing: false,
    conflict: false,
    deskCueCount: 25,
    deskWindowN: 25,
    deskVisible: true,
    downEpisodeMs: 0,
    inDownEpisode: false,
    screenAttentive: true,
    ...o,
  };
}
