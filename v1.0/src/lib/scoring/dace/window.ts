/**
 * Sliding window W (15 frames = 3 s at 5 FPS), the 25-frame desk window, and
 * the downward-episode tracker used by rules 5/6 (Annex A.4).
 *
 * Downward episode: starts at the first frame with D_t = 1 and continues
 * while D_t holds, tolerating up to `episodeGapFrames` consecutive non-down
 * frames (single-frame landmark jitter must not split one long look at a
 * page into many "brief glances"). Duration = t_now − t_start + frame
 * interval, so 15 consecutive down frames = 3000 ms = τ_g (still a glance;
 * the 16th frame exceeds τ_g).
 */
import type { DaceParams } from './params';
import type { DaceFrame, FrameEvidence, WindowSummary } from './types';
import { computeFrameEvidence, summarizeEvidence } from './evidence';
import { isLookingDown } from './looking-down';

export class DaceWindow {
  private frames: FrameEvidence[] = [];
  private deskFrames: FrameEvidence[] = [];
  private episodeStart: number | null = null;
  private episodeGap = 0;
  private lastT: number | null = null;

  constructor(private readonly params: DaceParams) {}

  /** Add one tick. D_t is computed here from the frame's face measurement. */
  push(frame: DaceFrame): FrameEvidence {
    const f = frame.face;
    const d = f !== null && isLookingDown(f.relPitchDeg, f.gazeV, f.gazeValid, this.params);
    const prev = this.frames.length ? this.frames[this.frames.length - 1] : null;
    const ev = computeFrameEvidence(frame, d, prev, this.params);

    this.frames.push(ev);
    if (this.frames.length > this.params.windowFrames) this.frames.shift();
    this.deskFrames.push(ev);
    if (this.deskFrames.length > this.params.deskWindowFrames) this.deskFrames.shift();

    if (ev.lookingDown) {
      if (this.episodeStart === null) this.episodeStart = frame.t;
      this.episodeGap = 0;
    } else if (this.episodeStart !== null) {
      this.episodeGap++;
      if (this.episodeGap > this.params.episodeGapFrames) {
        this.episodeStart = null;
        this.episodeGap = 0;
      }
    }
    this.lastT = frame.t;
    return ev;
  }

  summary(): WindowSummary {
    const t = this.lastT ?? 0;
    const inDownEpisode = this.episodeStart !== null;
    return {
      t,
      ...summarizeEvidence(this.frames, this.deskFrames, this.params),
      inDownEpisode,
      downEpisodeMs: inDownEpisode ? t - (this.episodeStart as number) + this.params.frameIntervalMs : 0,
    };
  }

  /** Frames currently in W (oldest -> newest). */
  getFrames(): readonly FrameEvidence[] {
    return this.frames;
  }

  /** Timestamp of the oldest frame in W (for closure runs "within W"). */
  windowStartT(): number | null {
    return this.frames.length ? this.frames[0].t : null;
  }

  reset(): void {
    this.frames = [];
    this.deskFrames = [];
    this.episodeStart = null;
    this.episodeGap = 0;
    this.lastT = null;
  }
}
