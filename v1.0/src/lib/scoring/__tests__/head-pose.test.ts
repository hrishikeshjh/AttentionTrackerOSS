import { describe, it, expect } from 'vitest';
import { calculateHeadPose } from '../pose/head-pose';
import type { Point2D } from '../geometry';

/** Builds a minimal 68-point landmark array with only the 4 points head-pose uses populated meaningfully. */
function buildLandmarks(overrides: { nose: Point2D; leftEye: Point2D; rightEye: Point2D; chin: Point2D }): Point2D[] {
  const landmarks: Point2D[] = Array.from({ length: 68 }, () => ({ x: 0, y: 0 }));
  landmarks[30] = overrides.nose;
  landmarks[36] = overrides.leftEye;
  landmarks[45] = overrides.rightEye;
  landmarks[8] = overrides.chin;
  return landmarks;
}

describe('calculateHeadPose', () => {
  it('returns ~0 yaw/pitch for a frontal face (nose centered between eyes, above chin)', () => {
    const landmarks = buildLandmarks({
      leftEye: { x: 40, y: 50 },
      rightEye: { x: 60, y: 50 },
      nose: { x: 50, y: 50 },
      chin: { x: 50, y: 90 },
    });
    const pose = calculateHeadPose(landmarks);
    expect(pose.yaw).toBeCloseTo(0, 5);
    expect(pose.pitch).toBeCloseTo(0, 5);
  });

  it('is invariant to uniform scaling of all landmarks (distance-from-camera invariance)', () => {
    const base = {
      leftEye: { x: 35, y: 48 },
      rightEye: { x: 62, y: 52 },
      nose: { x: 52, y: 60 },
      chin: { x: 51, y: 95 },
    };
    const scaled = {
      leftEye: { x: base.leftEye.x * 3, y: base.leftEye.y * 3 },
      rightEye: { x: base.rightEye.x * 3, y: base.rightEye.y * 3 },
      nose: { x: base.nose.x * 3, y: base.nose.y * 3 },
      chin: { x: base.chin.x * 3, y: base.chin.y * 3 },
    };

    const poseBase = calculateHeadPose(buildLandmarks(base));
    const poseScaled = calculateHeadPose(buildLandmarks(scaled));

    // Same physical angle at any distance from camera must produce the same
    // yaw/pitch — this is the literal regression test for the calibration fix.
    expect(poseScaled.yaw).toBeCloseTo(poseBase.yaw!, 6);
    expect(poseScaled.pitch).toBeCloseTo(poseBase.pitch!, 6);
  });

  it('flips yaw sign when the face is mirrored horizontally', () => {
    const landmarks = buildLandmarks({
      leftEye: { x: 30, y: 50 },
      rightEye: { x: 60, y: 50 },
      nose: { x: 55, y: 50 }, // nose shifted right of eye-midpoint (45)
      chin: { x: 55, y: 90 },
    });
    const mirrored = buildLandmarks({
      leftEye: { x: 40, y: 50 },
      rightEye: { x: 70, y: 50 },
      nose: { x: 45, y: 50 }, // nose shifted left of eye-midpoint (55)
      chin: { x: 45, y: 90 },
    });

    const pose = calculateHeadPose(landmarks);
    const mirroredPose = calculateHeadPose(mirrored);
    expect(pose.yaw).toBeGreaterThan(0);
    expect(mirroredPose.yaw).toBeLessThan(0);
  });

  it('returns nulls when fewer than 68 landmarks are provided', () => {
    const pose = calculateHeadPose([{ x: 0, y: 0 }]);
    expect(pose).toEqual({ yaw: null, pitch: null, roll: null });
  });

  it('uses the same normalizer for yaw and pitch (no nose-chin foreshortening bias)', () => {
    // Two faces with identical yaw/pitch ratio geometry but very different
    // nose-chin distances (simulating the old bug's sensitivity to chin
    // landmark foreshortening) must produce the same pitch.
    const shortChin = buildLandmarks({
      leftEye: { x: 40, y: 50 },
      rightEye: { x: 60, y: 50 },
      nose: { x: 50, y: 55 },
      chin: { x: 50, y: 60 }, // very short nose-chin distance
    });
    const longChin = buildLandmarks({
      leftEye: { x: 40, y: 50 },
      rightEye: { x: 60, y: 50 },
      nose: { x: 50, y: 55 },
      chin: { x: 50, y: 150 }, // very long nose-chin distance
    });

    const poseShort = calculateHeadPose(shortChin);
    const poseLong = calculateHeadPose(longChin);
    // Pitch depends only on eye distance now, not chin position at all.
    expect(poseShort.pitch).toBeCloseTo(poseLong.pitch!, 6);
  });
});
