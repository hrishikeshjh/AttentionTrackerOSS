import { describe, it, expect, beforeEach } from 'vitest';
import { ContradictionDetector } from '../contradiction-detector';
import type { ExtendedFaceMetrics } from '../types';

function baseMetrics(overrides: Partial<ExtendedFaceMetrics> = {}): ExtendedFaceMetrics {
  return {
    ear: 0.3,
    mar: 0.1,
    yaw: 0,
    pitch: 0,
    roll: 0,
    blinkDetected: false,
    blinkRate: 16,
    perclos: 5,
    gazeX: 0,
    gazeY: 0,
    faceDistance: 0.5,
    faceArea: 0.1,
    fatigueIndex: 10,
    fidgetScore: 5,
    headPoseStability: 90,
    eyebrowRaise: 0.3,
    mouthOpenness: 0.05,
    earVariance: 0.001,
    marVariance: 0.001,
    poseVariance: 1,
    ...overrides,
  };
}

describe('ContradictionDetector', () => {
  let detector: ContradictionDetector;
  beforeEach(() => {
    detector = new ContradictionDetector();
  });

  it('reports a low contradiction index when all modalities agree (coherent, attentive)', () => {
    const result = detector.compute(baseMetrics());
    expect(result.contradictionIndex).toBeLessThan(0.25);
    expect(result.category).toBe('coherent');
  });

  it('flags fatigue_masking when gaze looks fine but posture and eyes say tired', () => {
    for (let i = 0; i < 5; i++) {
      var result = detector.compute(
        baseMetrics({ gazeX: 0, gazeY: 0, headPoseStability: 10, perclos: 40 })
      );
    }
    expect(result!.category).toBe('fatigue_masking');
    expect(result!.contradictionIndex).toBeGreaterThan(0.3);
  });

  it('flags prop_without_engagement when a book is detected but hands are not on it', () => {
    for (let i = 0; i < 5; i++) {
      var result = detector.compute(
        baseMetrics({
          detectedObjects: [{ class: 'book', confidence: 0.9, bbox: { x: 0.5, y: 0.5, width: 0.1, height: 0.1 } }],
          handObjectProximity: 'hands_not_visible',
        })
      );
    }
    expect(result!.category).toBe('prop_without_engagement');
  });

  it('flags context_conflict when gaze looks attentive but a phone is detected', () => {
    for (let i = 0; i < 5; i++) {
      var result = detector.compute(
        baseMetrics({
          gazeX: 0,
          gazeY: 0,
          detectedObjects: [{ class: 'phone', confidence: 0.9, bbox: { x: 0.5, y: 0.5, width: 0.1, height: 0.1 } }],
        })
      );
    }
    expect(result!.category).toBe('context_conflict');
  });
});
