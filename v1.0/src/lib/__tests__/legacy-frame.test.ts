import { describe, it, expect } from 'vitest';
import { AbsenceTracker, ABSENCE_GRACE_MS, withBehavior, withLegacyContext } from '../legacy-frame';
import { MLEngine } from '../ml-engine';
import type { ExtendedFaceMetrics } from '../types';

const base: ExtendedFaceMetrics = {
  ear: 0.3, mar: 0.1, yaw: 0, pitch: 0, roll: 0,
  blinkDetected: false, blinkRate: 16, perclos: 5,
  gazeX: 0, gazeY: 0, faceDistance: 0.5, faceArea: 0.1,
  fatigueIndex: 10, fidgetScore: 5, headPoseStability: 90,
  eyebrowRaise: 0.3, mouthOpenness: 0.05,
  earVariance: 0.001, marVariance: 0.001, poseVariance: 1,
};

describe('withLegacyContext', () => {
  it('omits detectedObjects when the detector is not ready -> MLEngine uses the legacy weights', () => {
    const m = withLegacyContext(base, { objectDetectorReady: false, detectedObjects: [], handObjectProximity: 'hands_not_visible' });
    expect('detectedObjects' in m).toBe(false);
    const p = new MLEngine().predict(m);
    expect(Object.keys(p.featureImportance)).toContain('pitchDeviation');
    expect(Object.keys(p.featureImportance)).not.toContain('contextAwarePitch');
  });

  it('includes detectedObjects when the detector is ready -> upgraded weights', () => {
    const m = withLegacyContext(base, { objectDetectorReady: true, detectedObjects: [], handObjectProximity: 'hands_on_desk' });
    expect(m.detectedObjects).toEqual([]);
    expect(Object.keys(new MLEngine().predict(m).featureImportance)).toContain('contextAwarePitch');
  });
});

describe('withBehavior', () => {
  it('sets behaviorResult and studyContext', () => {
    const m = withBehavior(
      { ...base, handObjectProximity: 'holding_pen' },
      { label: 'attentive_reading', attentionScore: 80, confidence: 0.8, source: 'temporal' }
    );
    expect(m.studyContext).toBe('writing');
    expect(m.behaviorResult?.label).toBe('attentive_reading');
    expect(withBehavior({ ...base }, undefined).studyContext).toBe('unknown');
  });
});

describe('AbsenceTracker', () => {
  it('reports absence only after the grace period and resets when the face returns', () => {
    const a = new AbsenceTracker();
    expect(a.noFace(1000)).toBe(false);
    expect(a.noFace(1000 + ABSENCE_GRACE_MS - 1)).toBe(false);
    expect(a.noFace(1000 + ABSENCE_GRACE_MS)).toBe(true);
    a.faceSeen();
    expect(a.noFace(10_000)).toBe(false);
  });
});
