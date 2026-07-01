import { describe, it, expect } from 'vitest';
import { HeadPoseTracker, BlinkDetector } from '../trackers';

describe('HeadPoseTracker.getFidgetScore', () => {
  it('scores high-frequency detector jitter lower than genuine sustained movement of equal total amplitude', () => {
    const jitter = new HeadPoseTracker();
    const movement = new HeadPoseTracker();

    let t = 0;
    for (let i = 0; i < 30; i++) {
      // Jitter: alternates +/-2 around 0 — no real net motion, pure noise.
      const jitterYaw = i % 2 === 0 ? 2 : -2;
      jitter.addPose({ yaw: jitterYaw, pitch: 0, roll: 0 }, t);

      // Genuine movement: monotonically ramps, same per-step magnitude as jitter.
      movement.addPose({ yaw: i * 2, pitch: 0, roll: 0 }, t);

      t += 200;
    }

    const jitterScore = jitter.getFidgetScore();
    const movementScore = movement.getFidgetScore();

    // Both have the same raw per-frame delta magnitude, but jitter should be
    // suppressed relative to sustained motion once smoothed pre-differencing.
    expect(jitterScore).toBeLessThan(movementScore);
  });

  it('returns 0 with fewer than 10 samples', () => {
    const tracker = new HeadPoseTracker();
    tracker.addPose({ yaw: 5, pitch: 5, roll: 5 }, 0);
    expect(tracker.getFidgetScore()).toBe(0);
  });
});

describe('HeadPoseTracker.getStability', () => {
  it('returns 100 for a perfectly still pose history', () => {
    const tracker = new HeadPoseTracker();
    for (let i = 0; i < 10; i++) {
      tracker.addPose({ yaw: 5, pitch: -3, roll: 1 }, i * 1000);
    }
    expect(tracker.getStability()).toBe(100);
  });
});

describe('BlinkDetector', () => {
  it('counts a 200ms eye closure as a valid blink', () => {
    const detector = new BlinkDetector();
    detector.detectBlink(0.3, 0); // open
    detector.detectBlink(0.1, 100); // closed
    const result = detector.detectBlink(0.3, 300); // open again after 200ms
    expect(result.blinkDetected).toBe(true);
  });

  it('rejects an eye closure shorter than 50ms as noise', () => {
    const detector = new BlinkDetector();
    detector.detectBlink(0.3, 0);
    detector.detectBlink(0.1, 10);
    const result = detector.detectBlink(0.3, 20); // 10ms closure
    expect(result.blinkDetected).toBe(false);
  });

  it('rejects an eye closure longer than 500ms (likely not a blink)', () => {
    const detector = new BlinkDetector();
    detector.detectBlink(0.3, 0);
    detector.detectBlink(0.1, 100);
    const result = detector.detectBlink(0.3, 700); // 600ms closure
    expect(result.blinkDetected).toBe(false);
  });
});
