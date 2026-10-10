/**
 * STAGE 2 — eye kinematics K (Annex A.3, FR12): FEASIBILITY STUDY, not wired.
 *
 * Planned features over 3-6 s windows while D_t = 1 (iris track at up to 30 FPS):
 * return-sweep events (leftward jumps larger than half the window's horizontal
 * gaze range), horizontal drift rate, fixation dispersion. Go/no-go: dropped
 * if it adds < 2 macro-F1 points on reading vs phone under LOSO.
 *
 * Only the base-frame-rate FALLBACK features are sketched here (return-sweep
 * count and gaze dispersion), as pure functions with no runtime caller.
 */

export type KinematicsFeatures = {
  returnSweeps: number;
  /** standard deviation of horizontal gaze */
  dispersion: number;
};

/**
 * Leftward jumps larger than half the window's horizontal gaze range.
 * `gazeH` is in a frame where reading progresses toward +h (left-to-right
 * scripts); the sign convention must be checked on pilot data.
 */
export function countReturnSweeps(gazeH: readonly number[]): number {
  if (gazeH.length < 2) return 0;
  const range = Math.max(...gazeH) - Math.min(...gazeH);
  if (!(range > 0)) return 0;
  let n = 0;
  for (let i = 1; i < gazeH.length; i++) if (gazeH[i - 1] - gazeH[i] > range / 2) n++;
  return n;
}

export function fallbackKinematics(gazeH: readonly number[]): KinematicsFeatures | null {
  if (gazeH.length < 2) return null;
  const mean = gazeH.reduce((a, b) => a + b, 0) / gazeH.length;
  const dispersion = Math.sqrt(gazeH.reduce((a, b) => a + (b - mean) ** 2, 0) / gazeH.length);
  return { returnSweeps: countReturnSweeps(gazeH), dispersion };
}

/** Stage 1: kinematics is always off. */
export const KINEMATICS_AVAILABLE = false;
