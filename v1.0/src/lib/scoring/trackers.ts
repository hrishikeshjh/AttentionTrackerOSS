import { calculateVariance, clamp } from './geometry';
import {
  EAR_BLINK_THRESHOLD,
  PERCLOS_WINDOW_MS,
  NORMAL_BLINK_RATE_MIN,
  NORMAL_BLINK_RATE_MAX,
  FATIGUE_EAR_THRESHOLD,
} from './constants';
import type { HeadPose } from './pose/head-pose';

/**
 * Clock used for the blink-rate window.
 *   'wall'  (default): Date.now() at call time - the historical behaviour,
 *           frozen by the legacy golden fixture (which pins Date.now() to the
 *           frame timestamp with fake timers).
 *   'frame': the timestamp passed to detectBlink(), so offline replay or a
 *           delayed frame gives the same rate as live capture.
 */
export type BlinkClock = 'wall' | 'frame';

export class BlinkDetector {
  constructor(private readonly clock: BlinkClock = 'wall') {}

  private earHistory: Array<{ time: number; ear: number }> = [];
  private blinkEvents: number[] = [];
  private wasEyeClosed = false;
  private eyeClosureStart: number | null = null;
  private closedFrames: number[] = [];

  detectBlink(ear: number | null, timestamp: number): { blinkDetected: boolean; blinkRate: number; perclos: number } {
    if (ear === null) {
      return { blinkDetected: false, blinkRate: this.rateAt(timestamp), perclos: this.getPERCLOS(timestamp) };
    }

    this.earHistory.push({ time: timestamp, ear });
    const cutoff = timestamp - 120000;
    this.earHistory = this.earHistory.filter((e) => e.time > cutoff);
    this.blinkEvents = this.blinkEvents.filter((t) => t > cutoff);
    this.closedFrames = this.closedFrames.filter((t) => t > cutoff);

    const isEyeClosed = ear < EAR_BLINK_THRESHOLD;
    let blinkDetected = false;

    if (isEyeClosed) {
      this.closedFrames.push(timestamp);
      if (!this.wasEyeClosed) this.eyeClosureStart = timestamp;
    }

    if (this.wasEyeClosed && !isEyeClosed) {
      if (this.eyeClosureStart) {
        const blinkDuration = timestamp - this.eyeClosureStart;
        if (blinkDuration >= 50 && blinkDuration <= 500) {
          blinkDetected = true;
          this.blinkEvents.push(timestamp);
        }
      }
      this.eyeClosureStart = null;
    }

    this.wasEyeClosed = isEyeClosed;

    return { blinkDetected, blinkRate: this.rateAt(timestamp), perclos: this.getPERCLOS(timestamp) };
  }

  private rateAt(frameTimestamp: number): number {
    return this.getBlinkRate(this.clock === 'frame' ? frameTimestamp : undefined);
  }

  /**
   * Blinks per minute over the last 60 s.
   * @param now reference time in ms; defaults to Date.now() (historical behaviour)
   */
  getBlinkRate(now: number = Date.now()): number {
    if (this.blinkEvents.length < 2) return 0;
    const oneMinuteAgo = now - 60000;
    const recentBlinks = this.blinkEvents.filter((t) => t > oneMinuteAgo);
    const timeSpan = Math.min(60000, now - this.blinkEvents[0]);
    if (timeSpan < 5000) return 0;
    return (recentBlinks.length / timeSpan) * 60000;
  }

  private getPERCLOS(currentTime: number): number {
    const oneMinuteAgo = currentTime - PERCLOS_WINDOW_MS;
    const recentHistory = this.earHistory.filter((e) => e.time > oneMinuteAgo);
    if (recentHistory.length < 10) return 0;
    const closedCount = recentHistory.filter((e) => e.ear < EAR_BLINK_THRESHOLD).length;
    return (closedCount / recentHistory.length) * 100;
  }

  getEARStats(): { mean: number; variance: number } {
    if (this.earHistory.length === 0) return { mean: 0, variance: 0 };
    const ears = this.earHistory.map((e) => e.ear);
    const mean = ears.reduce((a, b) => a + b, 0) / ears.length;
    return { mean, variance: calculateVariance(ears) };
  }
}

/**
 * Tracks head-pose history for stability/fidget scoring.
 *
 * Fidget score is computed from an EMA-smoothed pose stream, not raw
 * frame-to-frame differences — the old implementation differenced raw
 * (unsmoothed) yaw/pitch/roll, so ordinary detector jitter (sub-pixel
 * landmark noise) inflated fidget score identically to genuine rapid head
 * movement. Smoothing first (alpha=0.4, fast enough to track real motion,
 * slow enough to reject single-frame jitter) fixes this without adding a
 * separate dependency (e.g. Savitzky-Golay). The smoothed series is
 * re-derived over the retained window in getFidgetScore(); no running
 * smoothed state is kept (the former smoothedYaw/Pitch/Roll fields were
 * written but never read and have been removed).
 */
export class HeadPoseTracker {
  private poseHistory: Array<{ time: number; yaw: number; pitch: number; roll: number }> = [];
  private static readonly FIDGET_SMOOTHING_ALPHA = 0.4;

  addPose(pose: HeadPose, timestamp: number): void {
    if (pose.yaw !== null && pose.pitch !== null && pose.roll !== null) {
      this.poseHistory.push({ time: timestamp, yaw: pose.yaw, pitch: pose.pitch, roll: pose.roll });
      const cutoff = timestamp - 30000;
      this.poseHistory = this.poseHistory.filter((p) => p.time > cutoff);
    }
  }

  getStability(): number {
    if (this.poseHistory.length < 5) return 100;
    const yaws = this.poseHistory.map((p) => p.yaw);
    const pitches = this.poseHistory.map((p) => p.pitch);
    const rolls = this.poseHistory.map((p) => p.roll);

    const combinedVar = (calculateVariance(yaws) + calculateVariance(pitches) + calculateVariance(rolls) * 0.5) / 2.5;
    return clamp(100 - combinedVar / 10, 0, 100);
  }

  getFidgetScore(): number {
    if (this.poseHistory.length < 10) return 0;

    // Re-derive a smoothed series over the retained window so the score
    // reflects genuine sustained movement, not raw per-frame jitter.
    const alpha = HeadPoseTracker.FIDGET_SMOOTHING_ALPHA;
    let sy = this.poseHistory[0].yaw;
    let sp = this.poseHistory[0].pitch;
    let sr = this.poseHistory[0].roll;
    const diffs: number[] = [];
    for (let i = 1; i < this.poseHistory.length; i++) {
      const prevY = sy, prevP = sp, prevR = sr;
      const cur = this.poseHistory[i];
      sy = alpha * cur.yaw + (1 - alpha) * sy;
      sp = alpha * cur.pitch + (1 - alpha) * sp;
      sr = alpha * cur.roll + (1 - alpha) * sr;
      diffs.push(Math.abs(sy - prevY) + Math.abs(sp - prevP) + Math.abs(sr - prevR));
    }

    const avgDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
    return clamp(avgDiff * 5, 0, 100);
  }

  getPoseVariance(): number {
    if (this.poseHistory.length < 2) return 0;
    const yaws = this.poseHistory.map((p) => p.yaw);
    const pitches = this.poseHistory.map((p) => p.pitch);
    const rolls = this.poseHistory.map((p) => p.roll);
    return (calculateVariance(yaws) + calculateVariance(pitches) + calculateVariance(rolls)) / 3;
  }
}

export class FatigueCalculator {
  private perclosHistory: Array<{ time: number; value: number }> = [];
  private blinkRateHistory: Array<{ time: number; value: number }> = [];
  private earMeanHistory: Array<{ time: number; value: number }> = [];

  update(perclos: number, blinkRate: number, earMean: number, timestamp: number): number {
    this.perclosHistory.push({ time: timestamp, value: perclos });
    this.blinkRateHistory.push({ time: timestamp, value: blinkRate });
    this.earMeanHistory.push({ time: timestamp, value: earMean });

    const cutoff = timestamp - 300000;
    this.perclosHistory = this.perclosHistory.filter((p) => p.time > cutoff);
    this.blinkRateHistory = this.blinkRateHistory.filter((p) => p.time > cutoff);
    this.earMeanHistory = this.earMeanHistory.filter((p) => p.time > cutoff);

    return this.calculateFatigueIndex();
  }

  private calculateFatigueIndex(): number {
    if (this.perclosHistory.length < 5) return 0;

    const avgPerclos = this.perclosHistory.reduce((a, b) => a + b.value, 0) / this.perclosHistory.length;
    const perclosScore = clamp(avgPerclos * 2, 0, 40);

    const avgBlinkRate = this.blinkRateHistory.reduce((a, b) => a + b.value, 0) / this.blinkRateHistory.length;
    let blinkScore = 0;
    if (avgBlinkRate < NORMAL_BLINK_RATE_MIN) {
      blinkScore = clamp((NORMAL_BLINK_RATE_MIN - avgBlinkRate) * 2, 0, 30);
    } else if (avgBlinkRate > NORMAL_BLINK_RATE_MAX) {
      blinkScore = clamp(avgBlinkRate - NORMAL_BLINK_RATE_MAX, 0, 30);
    }

    let earTrendScore = 0;
    if (this.earMeanHistory.length > 10) {
      const recentEars = this.earMeanHistory.slice(-10).map((e) => e.value);
      const avgEar = recentEars.reduce((a, b) => a + b, 0) / recentEars.length;
      if (avgEar < FATIGUE_EAR_THRESHOLD) {
        earTrendScore = clamp((FATIGUE_EAR_THRESHOLD - avgEar) * 100, 0, 30);
      }
    }

    return clamp(perclosScore + blinkScore + earTrendScore, 0, 100);
  }
}
