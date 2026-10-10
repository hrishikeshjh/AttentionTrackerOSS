import { describe, it, expect } from 'vitest';
import {
  eulerFromMatrix,
  eulerFromRotation,
  mat4FromRotation,
  relativePitch,
  rotationFromEuler,
  rotationFromMat4,
  cvToGlRotation,
} from '../head-pose-matrix';
import { isLookingDown } from '../looking-down';
import { P } from './helpers';

describe('head pose from transformation matrix (FR1: +pitch = head down)', () => {
  for (const deg of [15, 30, 45, -15, -30, -45]) {
    it(`rotation of ${deg}° about +x (y-up) yields pitch ${deg}`, () => {
      const m = mat4FromRotation(rotationFromEuler({ pitch: deg, yaw: 0, roll: 0 }));
      const e = eulerFromMatrix(m)!;
      expect(e.pitch).toBeCloseTo(deg, 9);
      expect(e.yaw).toBeCloseTo(0, 9);
      expect(e.roll).toBeCloseTo(0, 9);
    });
  }

  it('head-down rotation tilts the face normal toward −y (down) in the y-up frame', () => {
    const r = rotationFromEuler({ pitch: 30, yaw: 0, roll: 0 });
    // face normal = third column
    expect(r[1][2]).toBeLessThan(0);
    expect(eulerFromRotation(r).pitch).toBeGreaterThan(0);
  });

  it('decomposes combined yaw/pitch/roll', () => {
    for (const e of [
      { pitch: 20, yaw: 30, roll: 10 },
      { pitch: -25, yaw: -40, roll: -15 },
      { pitch: 45, yaw: 15, roll: 5 },
    ]) {
      const back = eulerFromMatrix(mat4FromRotation(rotationFromEuler(e)))!;
      expect(back.pitch).toBeCloseTo(e.pitch, 9);
      expect(back.yaw).toBeCloseTo(e.yaw, 9);
      expect(back.roll).toBeCloseTo(e.roll, 9);
    }
  });

  it('reads the matrix as column-major and ignores translation and uniform scale', () => {
    const r = rotationFromEuler({ pitch: 30, yaw: 0, roll: 0 });
    const m = mat4FromRotation(r, [3, -7, -60], 2.5);
    expect(m[9]).toBeCloseTo(r[1][2] * 2.5, 12); // column 2, row 1
    expect(eulerFromMatrix(m)!.pitch).toBeCloseTo(30, 9);
    expect(rotationFromMat4([1, 2, 3])).toBeNull();
  });

  it('pitchSign can invert the sign if the pilot shows an inversion', () => {
    const m = mat4FromRotation(rotationFromEuler({ pitch: 20, yaw: 0, roll: 0 }));
    expect(eulerFromMatrix(m, -1)!.pitch).toBeCloseTo(-20, 9);
  });

  it('cvToGlRotation keeps +x rotations (pitch) and flips yaw', () => {
    const rx = rotationFromEuler({ pitch: 30, yaw: 0, roll: 0 });
    expect(eulerFromRotation(cvToGlRotation(rx)).pitch).toBeCloseTo(30, 9);
    const ry = rotationFromEuler({ pitch: 0, yaw: 20, roll: 0 });
    expect(eulerFromRotation(cvToGlRotation(ry)).yaw).toBeCloseTo(-20, 9);
  });

  it('relative pitch', () => {
    expect(relativePitch(20, 5)).toBe(15);
    expect(relativePitch(null, 5)).toBeNull();
    expect(relativePitch(3, null)).toBeNull();
  });
});

describe('looking down D_t (inclusive thresholds)', () => {
  it('head clause: Δθ ≥ 15', () => {
    expect(isLookingDown(15, 0, true, P)).toBe(true);
    expect(isLookingDown(14.999, 0, true, P)).toBe(false);
    expect(isLookingDown(-30, 0, true, P)).toBe(false);
  });
  it('eye clause: g ≥ 0.5', () => {
    expect(isLookingDown(0, 0.5, true, P)).toBe(true);
    expect(isLookingDown(0, 0.4999, true, P)).toBe(false);
  });
  it('invalid gaze falls back to head pitch only', () => {
    expect(isLookingDown(0, 0.9, false, P)).toBe(false);
    expect(isLookingDown(20, 0.9, false, P)).toBe(true);
    expect(isLookingDown(null, null, false, P)).toBe(false);
  });
});
