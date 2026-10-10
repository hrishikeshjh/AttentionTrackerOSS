import { describe, it, expect } from 'vitest';
import {
  ABSENT_WINDOW_SHARE,
  TemporalBehaviorClassifier,
  deriveStudyContext,
  type TemporalFeatureFrame,
} from '../temporal-classifier';
import type { ExtendedFaceMetrics } from '../types';

function frame(overrides: Partial<ExtendedFaceMetrics> = {}): TemporalFeatureFrame {
  const metrics: ExtendedFaceMetrics = {
    ear: 0.3, mar: 0.1, yaw: 0, pitch: 0, roll: 0,
    blinkDetected: false, blinkRate: 16, perclos: 5,
    gazeX: 0, gazeY: 0, faceDistance: 0.5, faceArea: 0.1,
    fatigueIndex: 10, fidgetScore: 5, headPoseStability: 90,
    eyebrowRaise: 0.3, mouthOpenness: 0.05,
    earVariance: 0.001, marVariance: 0.001, poseVariance: 1,
    ...overrides,
  };
  return { metrics, detectedObjects: [], handProximity: 'hands_not_visible' };
}

describe('TemporalBehaviorClassifier absence', () => {
  it('updateAbsent() before the window is full returns absent', () => {
    const c = new TemporalBehaviorClassifier();
    const r = c.updateAbsent();
    expect(r.label).toBe('absent');
    expect(r.attentionScore).toBe(0);
  });

  it('the temporal rule layer returns absent once >= 80% of the window has no face', () => {
    const c = new TemporalBehaviorClassifier();
    for (let i = 0; i < 15; i++) c.update(frame());
    expect(c.getLastResult().label).toBe('attentive_screen');
    let r = c.getLastResult();
    const needed = Math.ceil(ABSENT_WINDOW_SHARE * 15); // 12
    for (let i = 0; i < needed - 1; i++) r = c.updateAbsent();
    expect(r.label).not.toBe('absent');
    r = c.updateAbsent();
    expect(r.label).toBe('absent');
    expect(r.source).toBe('temporal');
    // returning to the screen: absent share falls below 80% again
    for (let i = 0; i < 4; i++) r = c.update(frame());
    expect(r.label).not.toBe('absent');
  });

  it('windows built only from update() never return absent from the temporal layer (golden-fixture path)', () => {
    const c = new TemporalBehaviorClassifier();
    let r = c.getLastResult();
    for (let i = 0; i < 30; i++) r = c.update(frame({ faceArea: 0.001 }));
    expect(r.label).not.toBe('absent');
  });

  it('reset() clears absence', () => {
    const c = new TemporalBehaviorClassifier();
    for (let i = 0; i < 15; i++) c.updateAbsent();
    c.reset();
    for (let i = 0; i < 15; i++) c.update(frame());
    expect(c.getLastResult().label).toBe('attentive_screen');
  });
});

describe('deriveStudyContext', () => {
  it('maps behaviour labels and pen grip to a study context', () => {
    expect(deriveStudyContext('attentive_reading', 'holding_book')).toBe('reading');
    expect(deriveStudyContext('attentive_reading', 'holding_pen')).toBe('writing');
    expect(deriveStudyContext('using_phone', 'holding_phone')).toBe('using_phone');
    expect(deriveStudyContext('attentive_screen', 'hands_on_desk')).toBe('watching_screen');
    expect(deriveStudyContext('distracted', 'hands_on_desk')).toBe('unknown');
    expect(deriveStudyContext('absent', undefined)).toBe('unknown');
    expect(deriveStudyContext(undefined, undefined)).toBe('unknown');
  });
});
