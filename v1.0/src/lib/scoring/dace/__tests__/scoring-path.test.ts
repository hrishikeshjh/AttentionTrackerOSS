import { describe, it, expect } from 'vitest';
import { MLEngine } from '../../../ml-engine';
import { BEHAVIOR_ATTENTION_SCORES, TemporalBehaviorClassifier } from '../../../temporal-classifier';
import { SessionAnalyticsEngine } from '../../../analytics';
import type { ExtendedFaceMetrics, HandObjectProximity, SessionDataPoint } from '../../../types';
import { FEATURE_WEIGHTS_V1_UPGRADED, FEATURE_WEIGHTS_LEGACY } from '../../constants';
import { ExponentialMovingAverage } from '../../ema';
import { applyScoreHold } from '../score-hold';
import {
  CONTEXT_SCORES,
  DACE_LABEL_SCORES,
  mapContextFeatures,
  standardPitchScore,
  substituteMeasurement,
  type ContextMappingInput,
} from '../feature-mapping';
import { encode13 } from '../feature-vector';
import { summarizeUnsure, effectiveAbstainMode, UNSURE_REASON_TEXT } from '../abstain';
import { calibrateClassConditional, decideFromSet, predictionSet, softmaxT } from '../conformal';
import { countReturnSweeps, fallbackKinematics } from '../kinematics';
import { mulberry32 } from '../../../__tests__/fixtures/synthetic-face';
import { P } from './helpers';

function metrics(o: Partial<ExtendedFaceMetrics> = {}): ExtendedFaceMetrics {
  return {
    ear: 0.3, mar: 0.1, yaw: 2, pitch: -3, roll: 0,
    blinkDetected: false, blinkRate: 16, perclos: 5,
    gazeX: 0.05, gazeY: -0.02, faceDistance: 0.5, faceArea: 0.1,
    fatigueIndex: 10, fidgetScore: 5, headPoseStability: 90,
    eyebrowRaise: 0.3, mouthOpenness: 0.05,
    earVariance: 0.001, marVariance: 0.001, poseVariance: 1,
    ...o,
  };
}

describe('score hold (FR7)', () => {
  it('Unsure does not update the EMA; first-tick Unsure has no score', () => {
    const ema = new ExponentialMovingAverage(0.15);
    expect(applyScoreHold(ema, 80, true)).toEqual({ smoothed: null, held: false, available: false });
    expect(ema.getValue()).toBeNull();
    expect(applyScoreHold(ema, 80, false).smoothed).toBe(80);
    expect(applyScoreHold(ema, 10, true)).toEqual({ smoothed: 80, held: true, available: true });
    expect(ema.getValue()).toBe(80);
    expect(applyScoreHold(ema, 60, false).smoothed).toBeCloseTo(0.15 * 60 + 0.85 * 80, 12);
  });
});

describe('MLEngine DACE path', () => {
  it('no options == legacy path (same output for undefined and {} options)', () => {
    const a = new MLEngine(), b = new MLEngine(), c = new MLEngine();
    for (let i = 0; i < 5; i++) {
      const m = metrics({ ear: 0.25 + i * 0.01, detectedObjects: i % 2 ? [] : undefined });
      const pa = a.predict(m);
      expect(b.predict(m, undefined)).toEqual(pa);
      expect(c.predict(m, {})).toEqual(pa);
    }
  });

  it('holds score and state while Unsure, resumes without a jump', () => {
    const e = new MLEngine();
    const first = e.predict(metrics(), { dace: { unsure: true, unsureReason: 'U2' } });
    expect(first).toMatchObject({ unsure: true, scoreHeld: false, attentionState: 'Idle', rawScore: 0, unsureReason: 'U2' });
    const p1 = e.predict(metrics(), { dace: { label: 'attentive_screen', rule: 'screen' } });
    expect(p1).toMatchObject({ unsure: false, scoreHeld: false, scoringMode: 'dace', daceLabel: 'attentive_screen', daceRule: 'screen' });
    const held = e.predict(metrics({ ear: 0.05, perclos: 90 }), { dace: { unsure: true, unsureReason: 'U1' } });
    expect(held).toMatchObject({ unsure: true, scoreHeld: true, rawScore: p1.rawScore, attentionState: p1.attentionState, daceLabel: null });
    expect(held.contradictionIndex).toBeDefined(); // CMCV still updated
    const after = e.predict(metrics(), { dace: {} });
    expect(Math.abs(after.rawScore - p1.rawScore)).toBeLessThanOrEqual(1);
    // Unsure predictions are not in the stats history
    expect(e.getStats().avgScore).toBeCloseTo((p1.rawScore + after.rawScore) / 2, 9);
  });

  it('applies feature overrides and uses the scoring weights for contributions', () => {
    const e = new MLEngine();
    const p = e.predict(metrics({ detectedObjects: [] }), {
      dace: { weights: 'upgraded', featureOverrides: { contextAwarePitch: 85, objectContext: 80, handObjectProximity: 75, temporalPattern: 80, gazeDeviation: null } },
    });
    expect(p.featureImportance.contextAwarePitch).toBe(FEATURE_WEIGHTS_V1_UPGRADED.contextAwarePitch);
    expect(p.featureContributions.contextAwarePitch).toBeCloseTo((85 - 50) * 0.12, 12);
    expect(p.featureContributions.gazeDeviation).toBeUndefined();
    // legacy weights on the DACE path: ear contribution uses 0.15 (legacy path would use 0.10)
    const l = new MLEngine().predict(metrics({ ear: 0.3 }), { dace: { weights: 'legacy' } });
    const earScore = 85 + ((0.3 - 0.25) / 0.15) * 15;
    expect(l.featureContributions.ear).toBeCloseTo((earScore - 50) * FEATURE_WEIGHTS_LEGACY.ear, 12);
    const legacy = new MLEngine().predict(metrics({ ear: 0.3 }));
    expect(legacy.featureContributions.ear).toBeCloseTo((earScore - 50) * FEATURE_WEIGHTS_V1_UPGRADED.ear, 12);
  });
});

describe('feature mapping (plan section 6)', () => {
  const base: ContextMappingInput = {
    output: { label: 'attentive_screen', unsure: false, rule: 'screen' },
    evidence: { phoneInUse: false, book: false, writing: false },
    handsVisible: false,
    relPitchDeg: 0,
    lookingDown: false,
    gazeH: 0.1,
    gazeV: -0.2,
    gazeValid: true,
    previous: null,
  };

  it('contextAwarePitch by label; glance keeps the previous value; else the standard curve', () => {
    const m = (o: Partial<ContextMappingInput>) => mapContextFeatures({ ...base, ...o }, P).contextAwarePitch;
    expect(m({ output: { label: 'attentive_reading', unsure: false, rule: 'reading' } })).toBe(85);
    expect(m({ output: { label: 'using_phone', unsure: false, rule: 'phone' } })).toBe(20);
    expect(m({ output: { label: 'drowsy', unsure: false, rule: 'drowsy' } })).toBe(15);
    expect(m({ output: { label: 'attentive_screen', unsure: false, rule: 'glance' }, previous: { contextAwarePitch: 85 } })).toBe(85);
    expect(m({ relPitchDeg: 10 })).toBeCloseTo(100 - (10 / 15) * 20, 12);
    expect(m({ relPitchDeg: -30 })).toBe(50);
    expect(standardPitchScore(null)).toBe(50);
  });

  it('objectContext / handObjectProximity values; a phone merely present never lowers them', () => {
    const f = (ev: Partial<ContextMappingInput['evidence']>, handsVisible = false) =>
      mapContextFeatures({ ...base, evidence: { ...base.evidence, ...ev }, handsVisible }, P);
    expect(f({ phoneInUse: true })).toMatchObject({ objectContext: 25, handObjectProximity: 20 });
    expect(f({ book: true })).toMatchObject({ objectContext: 80, handObjectProximity: 80 });
    expect(f({ writing: true })).toMatchObject({ objectContext: 80, handObjectProximity: 75 });
    expect(f({})).toMatchObject({ objectContext: 65, handObjectProximity: 60 });
    expect(f({}, true).handObjectProximity).toBe(65);
    expect(CONTEXT_SCORES.object.neutral).toBe(65);
  });

  it('temporalPattern = label score; neutral 60 while Unsure; gaze only when not looking down', () => {
    expect(mapContextFeatures(base, P).temporalPattern).toBe(85);
    expect(mapContextFeatures({ ...base, output: { label: null, unsure: true, rule: 'U1' } }, P).temporalPattern).toBe(60);
    expect(mapContextFeatures(base, P).gazeDeviation).toBeCloseTo(0.3, 12);
    expect(mapContextFeatures({ ...base, lookingDown: true }, P).gazeDeviation).toBeNull();
    expect(mapContextFeatures({ ...base, gazeValid: false }, P).gazeDeviation).toBeNull();
  });

  it('DACE label scores equal the temporal classifier table', () => {
    expect(DACE_LABEL_SCORES).toEqual(BEHAVIOR_ATTENTION_SCORES);
  });

  it('substitution writes negated (+down -> legacy down-negative) pitch and gaze', () => {
    const m = substituteMeasurement(metrics(), { pitchDownDeg: 25, gazeV: 0.6, gazeH: 0.2, yawDeg: 10, residualOpenness: 1, correctedPerclos: 0.05 }, P);
    expect(m.pitch).toBe(-25); // so legacy `pitch < -15` fires for head down
    expect(m.gazeY).toBeCloseTo(-0.6, 12);
    expect(m.gazeX).toBeCloseTo(0.2, 12);
    expect(m.ear).toBeCloseTo(P.earScale, 12);
    expect(m.perclos).toBeCloseTo(5, 12);
    expect(m.yaw).toBe(10);
    // partial substitution (A'') leaves other fields untouched
    const a2 = substituteMeasurement(metrics(), { pitchDownDeg: 20 }, P);
    expect(a2.ear).toBe(0.3);
    expect(a2.gazeY).toBe(-0.02);
  });
});

describe('encode13 == TemporalBehaviorClassifier.extractFeatures', () => {
  it('matches on randomized inputs', () => {
    const tc = new TemporalBehaviorClassifier();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const extract = (tc as any).extractFeatures.bind(tc);
    const rnd = mulberry32(5);
    const prox: HandObjectProximity[] = ['holding_book', 'holding_pen', 'hands_on_desk', 'holding_phone', 'hands_not_visible'];
    const classes = ['book', 'notebook', 'phone', 'laptop', 'mouse', 'pen'] as const;
    for (let i = 0; i < 200; i++) {
      const m = metrics({
        gazeX: rnd() < 0.1 ? null : rnd() * 3 - 1.5,
        gazeY: rnd() < 0.1 ? null : rnd() * 3 - 1.5,
        pitch: rnd() < 0.1 ? null : rnd() * 200 - 100,
        yaw: rnd() * 200 - 100,
        ear: rnd() < 0.1 ? null : rnd() * 1.2,
        perclos: rnd() * 120,
        headPoseStability: rnd() * 120,
        fidgetScore: rnd() * 120,
      });
      const objs = classes.filter(() => rnd() < 0.3).map((c) => ({ class: c, confidence: 0.9, bbox: { x: 0.5, y: 0.5, width: 0.1, height: 0.1 } }));
      const f = { metrics: m, detectedObjects: objs, handProximity: prox[i % 5] };
      expect(encode13(f)).toEqual(extract(f));
    }
  });
});

describe('abstain / Unsure reporting', () => {
  it('share and dominant reason text', () => {
    const pts = [
      { unsure: false }, { unsure: true, unsureReason: 'U2' as const }, { unsure: true, unsureReason: 'U2' as const },
      { unsure: true, unsureReason: 'U1' as const }, { unsure: false },
    ];
    const s = summarizeUnsure(pts);
    expect(s.share).toBeCloseTo(0.6, 12);
    expect(s.dominantReason).toBe('U2');
    expect(s.text).toBe('CognEn could not determine the activity for 60% of the session (mostly: desk not visible)');
    expect(summarizeUnsure([{ unsure: false }]).text).toBeNull();
    expect(UNSURE_REASON_TEXT.U3).toBe('poor tracking');
  });
  it("'conformal' degrades to 'rules' in Stage 1", () => {
    expect(effectiveAbstainMode('conformal').mode).toBe('rules');
    expect(effectiveAbstainMode('conformal').warning).toMatch(/Stage 2/);
    expect(effectiveAbstainMode('off')).toEqual({ mode: 'off', warning: null });
  });
});

describe('analytics excludes Unsure points', () => {
  const pts = (withUnsure: boolean): SessionDataPoint[] =>
    Array.from({ length: 20 }, (_, i) => ({
      timestamp: 1000 * i,
      attentionScore: withUnsure && i % 4 === 0 ? 10 : 80,
      attentionState: 'Attentive' as const,
      ...(withUnsure ? { unsure: i % 4 === 0, ...(i % 4 === 0 ? { unsureReason: 'U2' as const } : {}) } : {}),
    }));
  it('legacy data: no unsureShare field', () => {
    const a = new SessionAnalyticsEngine();
    a.setData(pts(false));
    expect('unsureShare' in a.analyze()).toBe(false);
  });
  it('DACE data: Unsure points excluded from stats, share reported', () => {
    const a = new SessionAnalyticsEngine();
    a.setData(pts(true));
    const r = a.analyze();
    expect(r.unsureShare).toBeCloseTo(0.25, 12);
    expect(r.attentionVariability).toBe(0);
    expect(r.lowAttentionPeriods).toEqual([]);
    expect(a.getSummary()).toMatchObject({ totalDataPoints: 15, averageScore: 80, minScore: 80 });
  });
});

describe('Stage 2 stubs', () => {
  it('conformal: class-conditional thresholds, sets and U5', () => {
    const calib = Array.from({ length: 19 }, (_, i) => ({ probs: [1 - i * 0.02, i * 0.02], label: 0 }));
    const q = calibrateClassConditional(calib, 0.1, 2);
    expect(q[0]).toBeCloseTo(0.34, 12); // rank ceil((19+1)*0.9) = 18 -> 18th smallest of {0, 0.02, .., 0.36}
    expect(q[1]).toBe(Infinity);
    expect(predictionSet([0.9, 0.1], q)).toEqual([0, 1]);
    expect(decideFromSet([0], ['attentive_screen', 'distracted'])).toEqual({ kind: 'label', label: 'attentive_screen' });
    expect(decideFromSet([0, 1], ['attentive_screen', 'distracted'])).toMatchObject({ kind: 'unsure', reason: 'U5' });
    const p = softmaxT([2, 0], 2);
    expect(p[0] + p[1]).toBeCloseTo(1, 12);
  });
  it('kinematics fallback: return sweeps and dispersion', () => {
    expect(countReturnSweeps([0, 0.1, 0.2, 0.3, 0.4, 0, 0.1, 0.2, 0.4, 0])).toBe(2);
    expect(fallbackKinematics([0])).toBeNull();
  });
});
