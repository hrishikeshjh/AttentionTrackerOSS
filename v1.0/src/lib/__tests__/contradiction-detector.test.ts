import { describe, it, expect, beforeEach } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { ContradictionDetector, classifyContradiction } from '../contradiction-detector';
import { APP_ENGINE_OPTIONS, MLEngine } from '../ml-engine';
import { FEATURE_WEIGHTS_LEGACY } from '../scoring/constants';
import type { ExtendedFaceMetrics } from '../types';
import { runSequence, SEQUENCES, type CmcvFrame } from '../../../../shared/fixtures/gen_cmcv_fixture';

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
    let result;
    for (let i = 0; i < 5; i++) {
      result = detector.compute(
        baseMetrics({ gazeX: 0, gazeY: 0, headPoseStability: 10, perclos: 40 })
      );
    }
    expect(result!.category).toBe('fatigue_masking');
    expect(result!.contradictionIndex).toBeGreaterThan(0.3);
  });

  it('flags prop_without_engagement when a book is detected but hands are not on it', () => {
    let result;
    for (let i = 0; i < 5; i++) {
      result = detector.compute(
        baseMetrics({
          detectedObjects: [{ class: 'book', confidence: 0.9, bbox: { x: 0.5, y: 0.5, width: 0.1, height: 0.1 } }],
          handObjectProximity: 'hands_not_visible',
        })
      );
    }
    expect(result!.category).toBe('prop_without_engagement');
  });

  it('flags context_conflict when gaze looks attentive but a phone is detected', () => {
    let result;
    for (let i = 0; i < 5; i++) {
      result = detector.compute(
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

const BOOK = { class: 'book' as const, confidence: 0.9, bbox: { x: 0.5, y: 0.5, width: 0.1, height: 0.1 } };

describe('ContradictionDetector v2 taxonomy fixes', () => {
  it('does not flag prop_without_engagement when hands are on the desk, holding a pen or the book', () => {
    for (const p of ['hands_on_desk', 'holding_pen', 'holding_book'] as const) {
      const d = new ContradictionDetector();
      let r;
      for (let i = 0; i < 5; i++) r = d.compute(baseMetrics({ detectedObjects: [BOOK], handObjectProximity: p }));
      expect(r!.category, p).not.toBe('prop_without_engagement');
    }
  });

  it('does not flag prop_without_engagement when the behaviour label is attentive_reading', () => {
    const d = new ContradictionDetector();
    const r = d.compute(
      baseMetrics({
        detectedObjects: [BOOK],
        handObjectProximity: 'hands_not_visible',
        behaviorResult: { label: 'attentive_reading', attentionScore: 80, confidence: 0.8, source: 'rule_based' },
      })
    );
    expect(r.category).not.toBe('prop_without_engagement');
  });

  it('reports unclassified (not coherent) for a high index with no matching rule', () => {
    const d = new ContradictionDetector();
    let r;
    for (let i = 0; i < 6; i++) {
      r = d.compute(baseMetrics({ gazeX: 0.5, gazeY: 0.2, headPoseStability: 100, perclos: 0, handObjectProximity: 'holding_phone' }));
    }
    expect(r!.contradictionIndex).toBeGreaterThanOrEqual(0.25);
    expect(r!.category).toBe('unclassified');
  });

  it('classifyContradiction is pure and ordered (fatigue_masking wins over a book)', () => {
    expect(
      classifyContradiction({
        opinions: { gaze: 1, pose: 0.1, ocular: 0.1, object: 0.8, manual: 0.6 },
        contradictionIndex: 0.5,
        hasBook: true,
        hasPhone: false,
        handObjectProximity: 'hands_not_visible',
        behaviorLabel: undefined,
      })
    ).toBe('fatigue_masking');
  });

  it("'v1' taxonomy keeps the original rules (golden-fixture compatibility)", () => {
    const v1 = new ContradictionDetector('v1');
    expect(v1.compute(baseMetrics({ detectedObjects: [BOOK], handObjectProximity: 'hands_on_desk' })).category).toBe(
      'prop_without_engagement'
    );
    const v1b = new ContradictionDetector('v1');
    let r;
    for (let i = 0; i < 6; i++) {
      r = v1b.compute(baseMetrics({ gazeX: 0.5, gazeY: 0.2, headPoseStability: 100, perclos: 0, handObjectProximity: 'holding_phone' }));
    }
    expect(r!.category).toBe('coherent');
  });
});

describe('shared CMCV cross-language fixture (shared/fixtures/cmcv_cross_language.json)', () => {
  const path = fileURLToPath(new URL('../../../../shared/fixtures/cmcv_cross_language.json', import.meta.url));
  const fx = JSON.parse(readFileSync(path, 'utf8')) as {
    sequences: Array<{
      name: string;
      frames: CmcvFrame[];
      expected: Array<{ contradictionIndex: number; category: string; modalityOpinions: Record<string, number> }>;
    }>;
  };

  it('is up to date with the generator inputs (regenerate with npx tsx ../shared/fixtures/gen_cmcv_fixture.ts)', () => {
    expect(fx.sequences.map((s) => s.name)).toEqual(SEQUENCES.map((s) => s.name));
    expect(fx.sequences.map((s) => s.frames)).toEqual(SEQUENCES.map((s) => s.frames));
  });

  it('the shipped TS detector reproduces every expected frame', () => {
    for (const seq of fx.sequences) {
      const got = runSequence(seq.frames);
      got.forEach((g, i) => {
        const e = seq.expected[i];
        expect(g.category, `${seq.name}[${i}]`).toBe(e.category);
        expect(g.contradictionIndex).toBeCloseTo(e.contradictionIndex, 12);
        for (const k of Object.keys(e.modalityOpinions)) {
          expect(g.modalityOpinions[k]).toBeCloseTo(e.modalityOpinions[k], 12);
        }
      });
    }
  });

  it('covers every category', () => {
    const cats = new Set(fx.sequences.flatMap((s) => s.expected.map((e) => e.category)));
    expect([...cats].sort()).toEqual(
      ['coherent', 'context_conflict', 'fatigue_masking', 'prop_without_engagement', 'unclassified'].sort()
    );
  });
});

describe('MLEngine options (explanatory outputs only)', () => {
  const m = () =>
    baseMetrics({ ear: 0.25, perclos: 8, gazeX: 0.2, gazeY: 0.1, yaw: 12, pitch: 9, headPoseStability: 70, fidgetScore: 20 });

  it('APP_ENGINE_OPTIONS never changes score, state or confidence', () => {
    const a = new MLEngine();
    const b = new MLEngine(APP_ENGINE_OPTIONS);
    for (let i = 0; i < 5; i++) {
      const pa = a.predict(m());
      const pb = b.predict(m());
      expect(pb.rawScore).toBe(pa.rawScore);
      expect(pb.attentionState).toBe(pa.attentionState);
      expect(pb.confidence).toBe(pa.confidence);
    }
  });

  it("contributionWeights='used' weights legacy-path contributions with the legacy table", () => {
    const p = new MLEngine({ contributionWeights: 'used' }).predict(m());
    const h = new MLEngine().predict(m());
    // ear: legacy .15 vs upgraded .10 -> ratio 1.5
    expect(p.featureContributions.ear / h.featureContributions.ear).toBeCloseTo(
      FEATURE_WEIGHTS_LEGACY.ear / 0.1,
      9
    );
    expect(p.featureImportance.ear).toBe(FEATURE_WEIGHTS_LEGACY.ear);
  });

  it('predictAbsent produces Idle and lowers the smoothed score through the same EMA', () => {
    const e = new MLEngine(APP_ENGINE_OPTIONS);
    const first = e.predict(m());
    const idle = e.predictAbsent();
    expect(idle.attentionState).toBe('Idle');
    expect(idle.rawScore).toBeLessThan(first.rawScore);
    // alpha 0.15, absent score 0 (rawScore is rounded, so allow one point)
    expect(Math.abs(idle.rawScore - first.rawScore * 0.85)).toBeLessThanOrEqual(1);
    expect(e.getStats().stateDistribution.Idle).toBe(50);
  });
});
