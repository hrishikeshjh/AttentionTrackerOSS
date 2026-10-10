/**
 * Golden-fixture legacy equivalence test (DACE plan section 3/9).
 *
 * Runs deterministic synthetic 68-point landmark sequences plus object/hand
 * lists through the legacy pipeline exactly as webcam-panel.tsx wires it
 * (HandObjectProximityDetector -> FaceMetricsAggregator ->
 * TemporalBehaviorClassifier -> MLEngine.predict -> SessionAnalyticsEngine)
 * and snapshots the results. It was recorded BEFORE any DACE change; with
 * DACE off, the legacy path must reproduce it byte for byte. A SHA-256 over
 * the full per-frame trace guards every value; sampled frames are stored in
 * readable form to make a failing diff debuggable.
 *
 * Do NOT update the snapshot to make this pass unless a legacy-path change
 * has been explicitly approved.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { createHash } from 'node:crypto';
import { FaceMetricsAggregator, HandObjectProximityDetector } from '../scoring/aggregator';
import { TemporalBehaviorClassifier } from '../temporal-classifier';
import { MLEngine } from '../ml-engine';
import { SessionAnalyticsEngine } from '../analytics';
import type { DetectedObject, ExtendedFaceMetrics, HandResult, MLPrediction } from '../types';
import { DEFAULT_FACE, makeFace, makeHand, makeObject, mulberry32, type FaceParams } from './fixtures/synthetic-face';

type Segment = {
  name: string;
  frames: number;
  face: (k: number) => Partial<FaceParams>;
  hands: (k: number, rnd: () => number) => HandResult[];
  objects: (k: number) => DetectedObject[];
};

const SEGMENTS: Segment[] = [
  {
    name: 'attentive_screen',
    frames: 30,
    face: () => ({}),
    hands: () => [],
    objects: () => [],
  },
  {
    name: 'blinks_then_drowsy',
    frames: 40,
    // periodic blinks first, then mostly closed eyes
    face: (k) => ({ eyeOpen: k < 15 ? (k % 5 === 0 ? 0.2 : 1) : (k % 4 === 0 ? 0.8 : 0.35), noseShiftY: k >= 15 ? 0.05 : 0 }),
    hands: () => [],
    objects: () => [],
  },
  {
    name: 'book_nose_up',
    frames: 30,
    // nose raised above the eye line drives legacy pitch < -15
    face: () => ({ noseShiftY: -1.15 }),
    hands: (_k, rnd) => [makeHand(0.5, 0.8, 'Right', rnd)],
    objects: () => [makeObject('book', 0.52, 0.78)],
  },
  {
    name: 'phone_in_hand',
    frames: 30,
    face: (k) => ({ noseShiftY: k % 2 ? -1.2 : -0.7, lidShiftX: 0.03 }),
    hands: (_k, rnd) => [makeHand(0.3, 0.7, 'Left', rnd)],
    objects: () => [makeObject('phone', 0.32, 0.72, 0.9, 0.08, 0.12)],
  },
  {
    name: 'looking_away_fidget',
    frames: 30,
    face: (k) => ({ yawShift: 0.35 * Math.sin(k / 2), rollDeg: 8 * Math.sin(k / 3), lidShiftX: 0.12 * Math.cos(k / 4), mouthOpen: k % 6 < 3 ? 0.9 : 0.1 }),
    hands: (k, rnd) => (k % 3 === 0 ? [makeHand(0.8, 0.85, 'Right', rnd), makeHand(0.2, 0.85, 'Left', rnd)] : []),
    objects: (k) => (k % 2 ? [makeObject('laptop', 0.5, 0.9, 0.7, 0.5, 0.2)] : []),
  },
  {
    name: 'pen_and_unknown_down',
    frames: 30,
    face: (k) => ({ noseShiftY: -1.25, eyeOpen: k < 15 ? 1 : 0.6 }),
    hands: (_k, rnd) => [makeHand(0.6, 0.85, 'Right', rnd)],
    objects: (k) => (k < 15 ? [makeObject('pen', 0.62, 0.84, 0.6, 0.02, 0.1)] : []),
  },
  {
    name: 'tiny_face_absent',
    frames: 12,
    face: () => ({ scale: 4, noise: 0.05 }),
    hands: () => [],
    objects: () => [],
  },
  {
    name: 'back_to_screen',
    frames: 20,
    face: () => ({}),
    hands: () => [],
    objects: () => [],
  },
];

const T0 = 1_700_000_000_000;
const DT = 200; // 5 FPS
const W = 640;
const H = 480;

function runPipeline() {
  const rnd = mulberry32(20261010);
  const aggregator = new FaceMetricsAggregator();
  const proximity = new HandObjectProximityDetector();
  const temporal = new TemporalBehaviorClassifier();
  const upgradedEngine = new MLEngine();
  const legacyEngine = new MLEngine();
  const analytics = new SessionAnalyticsEngine();

  const trace: unknown[] = [];
  const sampled: unknown[] = [];
  let frame = 0;
  let lastUpgraded: MLPrediction | null = null;
  let lastLegacy: MLPrediction | null = null;

  for (const seg of SEGMENTS) {
    for (let k = 0; k < seg.frames; k++) {
      const t = T0 + frame * DT;
      vi.setSystemTime(t); // BlinkDetector.getBlinkRate reads Date.now()
      const landmarks = makeFace({ ...DEFAULT_FACE, ...seg.face(k) }, rnd);
      const hands = seg.hands(k, rnd);
      const detectedObjects = seg.objects(k);

      const handObjectProximity = proximity.compute(hands, detectedObjects);
      const baseMetrics = aggregator.calculateExtendedMetrics(landmarks, W, H, t);
      const extendedMetrics: ExtendedFaceMetrics = { ...baseMetrics, detectedObjects, handObjectProximity };
      extendedMetrics.behaviorResult = temporal.update({
        metrics: extendedMetrics,
        detectedObjects,
        handProximity: handObjectProximity,
      });

      const upgraded = upgradedEngine.predict(extendedMetrics);
      // Legacy weights path: context modules not loaded (detectedObjects undefined).
      const legacy = legacyEngine.predict({ ...baseMetrics });
      lastUpgraded = upgraded;
      lastLegacy = legacy;

      analytics.addDataPoint({ timestamp: t, attentionScore: upgraded.rawScore, attentionState: upgraded.attentionState });

      const record = { frame, seg: seg.name, handObjectProximity, baseMetrics, behavior: extendedMetrics.behaviorResult, upgraded, legacy };
      trace.push(record);
      if (frame % 15 === 0 || k === seg.frames - 1) sampled.push(record);
      frame++;
    }
  }

  return {
    frames: frame,
    sha256: createHash('sha256').update(JSON.stringify(trace)).digest('hex'),
    sampled,
    upgradedStats: upgradedEngine.getStats(),
    legacyStats: legacyEngine.getStats(),
    topInfluencers: {
      upgraded: lastUpgraded ? upgradedEngine.getTopInfluencers(lastUpgraded) : null,
      legacy: lastLegacy ? legacyEngine.getTopInfluencers(lastLegacy) : null,
    },
    analytics: analytics.analyze(),
    analyticsSummary: analytics.getSummary(),
    realtimeTrend: analytics.getRealtimeTrend(),
  };
}

describe('legacy pipeline golden fixture (DACE off must be byte-identical)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('is deterministic across two runs', () => {
    const a = runPipeline();
    const b = runPipeline();
    expect(a.sha256).toBe(b.sha256);
  });

  it('matches the recorded golden snapshot', () => {
    expect(runPipeline()).toMatchSnapshot();
  });

  it('exercises every legacy behaviour label and both weight sets', () => {
    const r = runPipeline();
    const labels = new Set(
      (r.sampled as Array<{ behavior?: { label: string } }>).map((s) => s.behavior?.label)
    );
    // sanity: fixture drives several distinct branches
    expect(labels.size).toBeGreaterThanOrEqual(3);
    const s0 = r.sampled[0] as { upgraded: MLPrediction; legacy: MLPrediction };
    expect(Object.keys(s0.upgraded.featureImportance)).toContain('contextAwarePitch');
    expect(Object.keys(s0.legacy.featureImportance)).toContain('pitchDeviation');
  });
});
