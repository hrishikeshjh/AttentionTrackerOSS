import { describe, it, expect } from 'vitest';
import { calculateGazeDirection } from '../pose/gaze';
import type { Point2D } from '../geometry';

function symmetricEyeLandmarks(): Point2D[] {
  const landmarks: Point2D[] = Array.from({ length: 68 }, () => ({ x: 0, y: 0 }));
  // Left eye: corners at 36/39, lids at 37/38/40/41 — centered iris approximation.
  landmarks[36] = { x: 0, y: 10 };
  landmarks[39] = { x: 20, y: 10 };
  landmarks[37] = { x: 8, y: 8 };
  landmarks[38] = { x: 12, y: 8 };
  landmarks[40] = { x: 12, y: 12 };
  landmarks[41] = { x: 8, y: 12 };
  // Right eye, mirrored.
  landmarks[42] = { x: 40, y: 10 };
  landmarks[45] = { x: 60, y: 10 };
  landmarks[43] = { x: 48, y: 8 };
  landmarks[44] = { x: 52, y: 8 };
  landmarks[46] = { x: 52, y: 12 };
  landmarks[47] = { x: 48, y: 12 };
  return landmarks;
}

describe('calculateGazeDirection', () => {
  it('returns ~0 for a centered, symmetric iris approximation', () => {
    const gaze = calculateGazeDirection(symmetricEyeLandmarks());
    expect(gaze.gazeX).toBeCloseTo(0, 5);
    expect(gaze.gazeY).toBeCloseTo(0, 5);
  });

  it('returns nulls for degenerate (zero-width) eyes', () => {
    const landmarks = symmetricEyeLandmarks();
    landmarks[36] = { x: 10, y: 10 };
    landmarks[39] = { x: 10, y: 10 }; // zero width
    const gaze = calculateGazeDirection(landmarks);
    expect(gaze.gazeX).toBeNull();
    expect(gaze.gazeY).toBeNull();
  });

  it('clamps output to [-1, 1]', () => {
    const landmarks = symmetricEyeLandmarks();
    // Push iris landmarks far outside the eye corners to force an extreme ratio.
    landmarks[37] = { x: 100, y: 8 };
    landmarks[38] = { x: 104, y: 8 };
    landmarks[40] = { x: 104, y: 12 };
    landmarks[41] = { x: 100, y: 12 };
    const gaze = calculateGazeDirection(landmarks);
    expect(gaze.gazeX).toBeLessThanOrEqual(1);
    expect(gaze.gazeX).toBeGreaterThanOrEqual(-1);
  });
});
