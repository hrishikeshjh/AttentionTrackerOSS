/**
 * "Unsure" score-hold policy (FR7, plan section 7).
 *
 *  - Unsure: the EMA is NOT updated; the displayed score is the last smoothed
 *    value (`held = true`); the attention state is kept by the caller.
 *  - Unsure on the very first tick (no smoothed value yet): no score
 *    (`available = false`) -> show "-", record no data point.
 *  - Leaving Unsure: the EMA resumes from the held value, so there is no jump.
 *  - Never a penalty.
 */
import type { ExponentialMovingAverage } from '../ema';

export type HoldResult = {
  /** smoothed score to report (null when not available) */
  smoothed: number | null;
  held: boolean;
  available: boolean;
};

export function applyScoreHold(ema: ExponentialMovingAverage, rawScore: number, unsure: boolean): HoldResult {
  if (unsure) {
    const v = ema.getValue();
    return v === null ? { smoothed: null, held: false, available: false } : { smoothed: v, held: true, available: true };
  }
  return { smoothed: ema.update(rawScore), held: false, available: true };
}
