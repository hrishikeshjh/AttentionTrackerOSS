import { describe, it, expect } from 'vitest';
import { calculateEAR, calculateMAR } from '../ear-mar';
import type { Point2D } from '../geometry';

function buildEyeLandmarks(ear: number): Point2D[] {
  // Build both eyes with horizontal dist = 10, vertical dist chosen so
  // EAR = verticalDist / (2*horizontalDist) matches the target.
  const landmarks: Point2D[] = Array.from({ length: 68 }, () => ({ x: 0, y: 0 }));
  const verticalDist = ear * 10;
  const half = verticalDist / 2;

  const fillEye = (base: number, xOffset: number) => {
    landmarks[base] = { x: xOffset, y: 0 }; // p1 (left corner)
    landmarks[base + 1] = { x: xOffset + 3, y: -half }; // p2
    landmarks[base + 2] = { x: xOffset + 7, y: -half }; // p3
    landmarks[base + 3] = { x: xOffset + 10, y: 0 }; // p4 (right corner)
    landmarks[base + 4] = { x: xOffset + 7, y: half }; // p5
    landmarks[base + 5] = { x: xOffset + 3, y: half }; // p6
  };

  fillEye(36, 0);
  fillEye(42, 40);
  return landmarks;
}

describe('calculateEAR', () => {
  it('reproduces the target EAR value from constructed landmarks', () => {
    const landmarks = buildEyeLandmarks(0.3);
    expect(calculateEAR(landmarks)).toBeCloseTo(0.3, 5);
  });

  it('returns null with fewer than 68 landmarks', () => {
    expect(calculateEAR([{ x: 0, y: 0 }])).toBeNull();
  });
});

describe('calculateMAR', () => {
  it('returns null when mouth landmarks are collinear (zero horizontal distance)', () => {
    const landmarks: Point2D[] = Array.from({ length: 68 }, () => ({ x: 5, y: 5 }));
    expect(calculateMAR(landmarks)).toBeNull();
  });
});
