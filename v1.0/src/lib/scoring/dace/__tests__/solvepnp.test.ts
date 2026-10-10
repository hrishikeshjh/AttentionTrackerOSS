import { describe, it, expect } from 'vitest';
import { projectModel, solvePnP6, solvePnP68, PNP_MODEL_POINTS_MM } from '../solvepnp';
import { cvToGlRotation, rotationFromEuler } from '../head-pose-matrix';
import { mulberry32 } from '../../../__tests__/fixtures/synthetic-face';
import type { Point2D } from '../../geometry';

const K = { f: 640, cx: 320, cy: 240 };

function imagePoints(e: { pitch: number; yaw: number; roll: number }, t: [number, number, number] = [20, -10, 700]) {
  // rotationFromEuler builds a y-up rotation; F·R·F converts it to the OpenCV frame (involution).
  const Rcv = cvToGlRotation(rotationFromEuler(e));
  return projectModel(Rcv, t, K);
}

describe('solvePnP on the 68-point model (A\'\')', () => {
  const poses = [
    { pitch: 0, yaw: 0, roll: 0 },
    { pitch: 15, yaw: 0, roll: 0 },
    { pitch: 30, yaw: 0, roll: 0 },
    { pitch: 45, yaw: 0, roll: 0 },
    { pitch: -15, yaw: 0, roll: 0 },
    { pitch: -30, yaw: 0, roll: 0 },
    { pitch: 20, yaw: 25, roll: 8 },
    { pitch: -10, yaw: -30, roll: -12 },
  ];
  for (const e of poses) {
    it(`recovers ${JSON.stringify(e)} within 1°`, () => {
      const r = solvePnP6(imagePoints(e), K)!;
      expect(r).not.toBeNull();
      expect(Math.abs(r.euler.pitch - e.pitch)).toBeLessThan(1);
      expect(Math.abs(r.euler.yaw - e.yaw)).toBeLessThan(1);
      expect(Math.abs(r.euler.roll - e.roll)).toBeLessThan(1);
      expect(r.rmsPx).toBeLessThan(1e-3);
    });
  }

  it('stays within 1° of pitch with ±0.5 px landmark noise', () => {
    const rnd = mulberry32(7);
    for (const pitch of [-30, -15, 0, 15, 30]) {
      const pts = imagePoints({ pitch, yaw: 0, roll: 0 }).map((p) => ({
        x: p.x + (rnd() - 0.5),
        y: p.y + (rnd() - 0.5),
      }));
      const r = solvePnP6(pts, K)!;
      expect(Math.abs(r.euler.pitch - pitch)).toBeLessThan(1);
    }
  });

  it('solvePnP68 reads indices 30, 8, 36, 45, 48, 54 from a 68-point array', () => {
    const pts = imagePoints({ pitch: 25, yaw: 0, roll: 0 });
    const lm: Point2D[] = Array.from({ length: 68 }, () => ({ x: 0, y: 0 }));
    PNP_MODEL_POINTS_MM.forEach(({ idx }, i) => (lm[idx] = pts[i]));
    const r = solvePnP68(lm, 640, 480)!;
    expect(Math.abs(r.euler.pitch - 25)).toBeLessThan(1);
    expect(solvePnP68(lm.slice(0, 10), 640, 480)).toBeNull();
  });
});
