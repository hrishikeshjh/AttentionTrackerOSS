/**
 * Gaze-corrected PERCLOS and longest continuous closure (Annex A.1, A.4 rule 2).
 *
 *   PERCLOS = share of closed frames (r_t < k) over the last 60 s, among
 *             frames with a valid openness value. Reported as a fraction 0-1.
 *   Continuous closure duration of a run of consecutive closed frames =
 *             t(last closed) − t(first closed) + nominal frame interval,
 *             so 10 consecutive closed frames at 5 FPS = 2000 ms.
 *   Frames without a valid openness value (no face, eyes not measurable)
 *   break a run.
 */
import type { DaceParams } from './params';

type Entry = { t: number; closed: boolean | null };

export class CorrectedPerclos {
  private hist: Entry[] = [];

  constructor(private readonly params: Pick<DaceParams, 'perclosWindowMs' | 'perclosMinSamples' | 'frameIntervalMs'>) {}

  push(t: number, closed: boolean | null): void {
    this.hist.push({ t, closed });
    const cutoff = t - this.params.perclosWindowMs;
    while (this.hist.length && this.hist[0].t <= cutoff) this.hist.shift();
  }

  /** PERCLOS (0-1) over the trailing window ending at the last push; 0 until enough samples. */
  value(): { perclos: number; n: number } {
    let n = 0;
    let c = 0;
    for (const e of this.hist) {
      if (e.closed === null) continue;
      n++;
      if (e.closed) c++;
    }
    if (n < this.params.perclosMinSamples) return { perclos: 0, n };
    return { perclos: c / n, n };
  }

  /** Longest closure (ms) among runs whose last frame is at or after `sinceT`; runs are counted in full. */
  longestClosureMs(sinceT: number): number {
    let best = 0;
    let runStart: number | null = null;
    let runLast: number | null = null;
    const close = () => {
      if (runStart !== null && runLast !== null && runLast >= sinceT) {
        best = Math.max(best, runLast - runStart + this.params.frameIntervalMs);
      }
      runStart = null;
      runLast = null;
    };
    for (const e of this.hist) {
      if (e.closed === true) {
        if (runStart === null) runStart = e.t;
        runLast = e.t;
      } else close();
    }
    close();
    return best;
  }

  reset(): void {
    this.hist = [];
  }
}
