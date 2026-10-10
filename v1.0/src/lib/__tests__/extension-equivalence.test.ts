/**
 * Extension <-> web scoring equivalence.
 *
 * Loads the Chrome extension's plain-JS scoring core (extension/lib/ml-core.js)
 * in a node:vm sandbox and runs the same deterministic synthetic 68-point
 * landmark sequences through it and through the web app's TS scoring package
 * (src/lib/scoring/*) and legacy MLEngine (src/lib/ml-engine.ts, base metrics
 * without detectedObjects => FEATURE_WEIGHTS_LEGACY). Every formula and the
 * engine score/state must match.
 *
 * Date.now() is pinned to each frame timestamp with fake timers, because the
 * web BlinkDetector's default ('wall') blink-rate clock reads Date.now() while
 * the extension uses the frame timestamp.
 */
import { describe, it, expect, beforeAll, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import {
  calculateEAR,
  calculateMAR,
  calculateHeadPose,
  calculateGazeDirection,
  estimateFaceDistance,
  calculateFaceArea,
  calculateEyebrowRaise,
  calculateMouthOpenness,
  BlinkDetector,
  HeadPoseTracker,
  FatigueCalculator,
  FaceMetricsAggregator,
  PrimaryFaceSelector,
  ExponentialMovingAverage,
  THRESHOLDS,
  FEATURE_WEIGHTS_LEGACY,
  FEATURE_WEIGHTS_V1_UPGRADED,
  OPTIMAL_RANGES,
  type Point2D,
} from '@/lib/scoring';
import { MLEngine } from '@/lib/ml-engine';
import type { ExtendedFaceMetrics } from '@/lib/types';
import { DEFAULT_FACE, makeFace, mulberry32, type FaceParams } from './fixtures/synthetic-face';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Core = Record<string, any>;

const EXT_DIR = fileURLToPath(new URL('../../../extension/', import.meta.url));
const FRAME_W = 640;
const FRAME_H = 480;
const FRAME_MS = 200; // extension analysis tick (5 FPS)
const T0 = 1_700_000_000_000;

function loadCore(): Core {
  const code = readFileSync(EXT_DIR + 'lib/ml-core.js', 'utf8');
  const sandbox: Record<string, any> = { module: { exports: {} }, console, Date };
  sandbox.self = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(code, sandbox, { filename: 'ml-core.js' });
  return sandbox.module.exports as Core;
}

/** Deterministic multi-phase sequence: attentive, blinks, yaw sweep, drowsy, fidget, nose-up. */
function makeSequence(seed: number, frames: number): Point2D[][] {
  const rnd = mulberry32(seed);
  const out: Point2D[][] = [];
  for (let k = 0; k < frames; k++) {
    const phase = Math.floor(k / 40) % 6;
    let p: Partial<FaceParams> = {};
    if (phase === 1) p = { eyeOpen: k % 6 === 0 ? 0.15 : 1 }; // single-frame blinks
    if (phase === 2) p = { yawShift: Math.sin(k / 5) * 0.4, lidShiftX: 0.03 };
    if (phase === 3) p = { eyeOpen: k % 3 === 0 ? 0.9 : 0.3, noseShiftY: 0.1 }; // drowsy
    if (phase === 4) p = { rollDeg: Math.sin(k) * 12, yawShift: Math.cos(k) * 0.3, mouthOpen: 0.9 }; // fidget
    if (phase === 5) p = { noseShiftY: -1.15, scale: 70 };
    out.push(makeFace({ ...DEFAULT_FACE, ...p }, rnd));
  }
  return out;
}

const plain = (x: unknown) => JSON.parse(JSON.stringify(x));

let core: Core;
beforeAll(() => {
  core = loadCore();
});
afterEach(() => {
  vi.useRealTimers();
});

describe('extension ml-core loads in node', () => {
  it('exports the scoring API and the canonical constants', () => {
    expect(typeof core.MLEngine).toBe('function');
    expect(plain(core.THRESHOLDS)).toEqual(THRESHOLDS);
    expect(plain(core.FEATURE_WEIGHTS_LEGACY)).toEqual(FEATURE_WEIGHTS_LEGACY);
    expect(plain(core.FEATURE_WEIGHTS_V1_UPGRADED)).toEqual(FEATURE_WEIGHTS_V1_UPGRADED);
    expect(plain(core.OPTIMAL_RANGES)).toEqual(OPTIMAL_RANGES);
    expect(core.PREDICT_INTERVAL_MS).toBe(1000);
    expect(core.FRAME_INTERVAL_MS).toBe(200);
    expect(core.PROXIMITY_CODES_V1.hands_not_visible).toBe(0);
  });

  it('is idempotent when evaluated twice in the same realm', () => {
    const code = readFileSync(EXT_DIR + 'lib/ml-core.js', 'utf8');
    const sandbox: Record<string, any> = { module: { exports: {} }, console, Date };
    vm.createContext(sandbox);
    vm.runInContext(code, sandbox);
    const first = sandbox.module.exports;
    expect(() => vm.runInContext(code, sandbox)).not.toThrow();
    expect(sandbox.__cognen.mlCore).toBe(first);
  });
});

describe('pure formulas match the TS scoring package', () => {
  const seq = makeSequence(7, 240);
  const close = (a: number | null, b: number | null) => {
    if (a === null || b === null) return expect(a).toBe(b);
    expect(Math.abs(a - b)).toBeLessThan(1e-9);
  };

  it('EAR / MAR / head pose / gaze / basic metrics', () => {
    for (const lm of seq) {
      close(core.calculateEAR(lm), calculateEAR(lm));
      close(core.calculateMAR(lm), calculateMAR(lm));
      const a = core.calculateHeadPose(lm);
      const b = calculateHeadPose(lm);
      close(a.yaw, b.yaw);
      close(a.pitch, b.pitch);
      close(a.roll, b.roll);
      const ga = core.calculateGazeDirection(lm);
      const gb = calculateGazeDirection(lm);
      close(ga.gazeX, gb.gazeX);
      close(ga.gazeY, gb.gazeY);
      close(core.estimateFaceDistance(lm, FRAME_W), estimateFaceDistance(lm, FRAME_W));
      close(core.calculateFaceArea(lm, FRAME_W, FRAME_H), calculateFaceArea(lm, FRAME_W, FRAME_H));
      close(core.calculateEyebrowRaise(lm), calculateEyebrowRaise(lm));
      close(core.calculateMouthOpenness(lm), calculateMouthOpenness(lm));
    }
  });

  it('null handling for short landmark arrays', () => {
    const short = seq[0].slice(0, 30);
    expect(core.calculateEAR(short)).toBeNull();
    expect(plain(core.calculateHeadPose(short))).toEqual(calculateHeadPose(short));
    expect(plain(core.calculateGazeDirection(short))).toEqual(calculateGazeDirection(short));
  });
});

describe('stateful trackers match', () => {
  it('BlinkDetector (blinks, rate, PERCLOS, EAR stats) incl. 500 ms max closure', () => {
    vi.useFakeTimers();
    const web = new BlinkDetector();
    const ext = new core.BlinkDetector();
    // EAR stream at 50 ms steps with closures of 100, 450, 500, 550 and 900 ms.
    const closures = [100, 450, 500, 550, 900, 100, 150, 300];
    let t = T0;
    const step = 50;
    const run = (ear: number | null) => {
      vi.setSystemTime(t);
      const a = web.detectBlink(ear, t);
      const b = ext.detectBlink(ear, t);
      expect(b.blinkDetected).toBe(a.blinkDetected);
      expect(b.blinkRate).toBeCloseTo(a.blinkRate, 9);
      expect(b.perclos).toBeCloseTo(a.perclos, 9);
      t += step;
      return a.blinkDetected;
    };
    let blinks = 0;
    for (const c of closures) {
      for (let i = 0; i < 40; i++) run(0.3);
      for (let i = 0; i < c / step; i++) run(0.15);
      run(null);
      if (run(0.3)) blinks++;
    }
    expect(blinks).toBeGreaterThan(0);
    const sa = web.getEARStats();
    const sb = ext.getEARStats();
    expect(sb.mean).toBeCloseTo(sa.mean, 12);
    expect(sb.variance).toBeCloseTo(sa.variance, 12);
  });

  it('HeadPoseTracker (stability, fidget x5, pose variance) and FatigueCalculator', () => {
    const web = new HeadPoseTracker();
    const ext = new core.HeadPoseTracker();
    const fw = new FatigueCalculator();
    const fe = new core.FatigueCalculator();
    const rnd = mulberry32(3);
    for (let k = 0; k < 300; k++) {
      const t = T0 + k * FRAME_MS;
      const pose = { yaw: Math.sin(k / 3) * 20 + rnd(), pitch: Math.cos(k / 7) * 10, roll: rnd() * 5 };
      web.addPose(pose, t);
      ext.addPose(pose, t);
      expect(ext.getStability()).toBeCloseTo(web.getStability(), 9);
      expect(ext.getFidgetScore()).toBeCloseTo(web.getFidgetScore(), 9);
      expect(ext.getPoseVariance()).toBeCloseTo(web.getPoseVariance(), 9);

      const perclos = k > 150 ? 30 : rnd() * 5;
      const blinkRate = k % 50 < 25 ? 8 : 26;
      const earMean = k > 200 ? 0.18 : 0.3;
      expect(fe.update(perclos, blinkRate, earMean, t)).toBeCloseTo(fw.update(perclos, blinkRate, earMean, t), 9);
    }
  });

  it('PrimaryFaceSelector hysteresis picks the same candidate', () => {
    const web = new PrimaryFaceSelector();
    const ext = new core.PrimaryFaceSelector();
    const rnd = mulberry32(11);
    for (let k = 0; k < 200; k++) {
      const cands: { box: { x: number; y: number; width: number; height: number }; detection: number }[] = [];
      if (k % 37 < 30) cands.push({ box: { x: 200 + rnd() * 4, y: 120, width: 200, height: 220 }, detection: 0 });
      if (k > 40 && k % 13 < 9) cands.push({ box: { x: 20, y: 20, width: 260 + (k % 5) * 20, height: 300 }, detection: 1 });
      if (k % 7 === 0) cands.push({ box: { x: 500, y: 300, width: 60, height: 60 }, detection: 2 });
      const a = web.select(cands, FRAME_W, FRAME_H);
      const b = ext.select(cands, FRAME_W, FRAME_H);
      expect(b ? b.detection : null).toBe(a ? a.detection : null);
    }
  });

  it('EMA alpha 0.15 default', () => {
    const w = new ExponentialMovingAverage();
    const e = new core.ExponentialMovingAverage();
    for (const v of [50, 80, 20, 100, 0, 65]) expect(e.update(v)).toBeCloseTo(w.update(v), 12);
  });
});

describe('aggregator + legacy MLEngine end to end', () => {
  const METRIC_KEYS = [
    'ear', 'mar', 'yaw', 'pitch', 'roll', 'blinkDetected', 'blinkRate', 'perclos', 'gazeX', 'gazeY',
    'faceDistance', 'faceArea', 'fatigueIndex', 'fidgetScore', 'headPoseStability', 'eyebrowRaise',
    'mouthOpenness', 'earVariance', 'marVariance', 'poseVariance',
  ] as const;

  for (const seed of [1, 2, 3]) {
    it(`seed ${seed}: metrics every 200 ms tick and 1 Hz predictions agree`, () => {
      vi.useFakeTimers();
      const seq = makeSequence(seed, 600); // 120 s at 5 FPS
      const webAgg = new FaceMetricsAggregator();
      const extAgg = new core.FaceMetricsAggregator();
      const webEngine = new MLEngine();
      const extEngine = new core.MLEngine();
      const states = new Set<string>();
      let lastPredict = -Infinity;

      seq.forEach((lm, k) => {
        const t = T0 + k * FRAME_MS;
        vi.setSystemTime(t);
        const a = webAgg.calculateExtendedMetrics(lm, FRAME_W, FRAME_H, t);
        const b = extAgg.calculateExtendedMetrics(lm, FRAME_W, FRAME_H, t);
        for (const key of METRIC_KEYS) {
          const va = a[key] as number | boolean | null;
          const vb = b[key];
          if (typeof va === 'number' && typeof vb === 'number') expect(Math.abs(va - vb), key).toBeLessThan(1e-6);
          else expect(vb, key).toBe(va);
        }
        expect(Object.keys(b).sort()).toEqual(Object.keys(a).sort());

        if (t - lastPredict >= 1000) {
          lastPredict = t;
          expect((a as ExtendedFaceMetrics).detectedObjects).toBeUndefined();
          const pa = webEngine.predict(a);
          const pb = extEngine.predict(b);
          expect(Math.abs(pb.rawScore - pa.rawScore)).toBeLessThanOrEqual(0.5);
          expect(pb.attentionState).toBe(pa.attentionState);
          expect(pb.confidence).toBeCloseTo(pa.confidence, 9);
          expect(pb.scoringMode).toBe('legacy');
          expect(Object.keys(pb.featureContributions).sort()).toEqual(Object.keys(pa.featureContributions).sort());
          states.add(pa.attentionState);
        }
      });
      expect(states.size).toBeGreaterThan(0);
    });
  }

  it('engine uses legacy weights (pitchDeviation scored, no context features)', () => {
    const engine = new core.MLEngine();
    const metrics = {
      ear: 0.3, perclos: 2, blinkRate: 15, gazeX: 0.05, gazeY: 0.02, headPoseStability: 95, yaw: 3, pitch: 25,
      fatigueIndex: 5, fidgetScore: 3, eyebrowRaise: 0.4, mouthOpenness: 0.05, faceDistance: 0.5, faceArea: 0.1,
    };
    const p = engine.predict(metrics);
    expect(Object.keys(p.featureImportance)).toContain('pitchDeviation');
    expect(Object.keys(p.featureImportance)).not.toContain('contextAwarePitch');
    const web = new MLEngine().predict({ ...metrics, mar: 0.3, roll: 0, blinkDetected: false } as ExtendedFaceMetrics);
    expect(p.rawScore).toBe(web.rawScore);
    expect(p.attentionState).toBe(web.attentionState);
  });

  it('context features are scored with web semantics, not a constant 50', () => {
    const down = { pitch: -20, ear: 0.3, perclos: 0 };
    expect(core.computeContextAwarePitch({ ...down, detectedObjects: [{ class: 'phone' }] })).toBe(20);
    expect(core.computeContextAwarePitch({ ...down, handObjectProximity: 'holding_book', detectedObjects: [] })).toBe(85);
    expect(core.computeContextAwarePitch({ pitch: -20, ear: 0.2, perclos: 30 })).toBe(15);
    expect(core.computeContextAwarePitch({ pitch: 0 })).toBe(100);
    expect(core.computeObjectContext({ detectedObjects: [{ class: 'laptop' }], pitch: 0 })).toBe(80);
    expect(core.encodeHandProximity('holding_phone')).toBe(20);
    expect(core.encodeTemporalPattern({ attentionScore: 85 })).toBe(85);
    expect(core.scoreFeature('objectContext', 25).score).toBe(25);
    expect(core.scoreFeature('temporalPattern', 140).score).toBe(100);

    // Upgraded path (only if a caller attached detectedObjects) matches the web too.
    const m = {
      ear: 0.3, perclos: 2, blinkRate: 15, gazeX: 0.05, gazeY: 0.02, headPoseStability: 95, yaw: 3, pitch: -20,
      fatigueIndex: 5, fidgetScore: 3, eyebrowRaise: 0.4, mouthOpenness: 0.05, faceDistance: 0.5, faceArea: 0.1,
      mar: 0.3, roll: 0, blinkDetected: false,
      detectedObjects: [{ class: 'book', confidence: 0.9, bbox: { x: 0.5, y: 0.5, width: 0.1, height: 0.1 } }],
      handObjectProximity: 'holding_book',
    } as ExtendedFaceMetrics;
    const a = new MLEngine().predict(m);
    const b = new core.MLEngine().predict(m);
    expect(b.rawScore).toBe(a.rawScore);
    expect(b.scoringMode).toBe('upgraded');
  });
});

describe('extension surfaces use the 70/40 bands and alert threshold 40', () => {
  const read = (f: string) => readFileSync(EXT_DIR + f, 'utf8');

  it('background badge bands and default alert threshold', () => {
    const bg = read('background.js');
    expect(bg).toMatch(/SCORE_BANDS = Object\.freeze\(\{ ATTENTIVE_MIN: 70, DISTRACTED_MIN: 40 \}\)/);
    expect(bg).toMatch(/lowAttentionThreshold: 40/);
  });

  it('popup ring uses ml-core THRESHOLDS (fallback 70/40) and default threshold 40', () => {
    const popup = read('popup/popup.js');
    expect(popup).toMatch(/\{ ATTENTIVE_MIN: 70, DISTRACTED_MIN: 40 \}/);
    expect(popup).toMatch(/lowAttentionThreshold: 40/);
    expect(popup).not.toMatch(/>= 75|>= 45|\|\| 45/);
    expect(read('popup/popup.html')).toMatch(/lib\/ml-core\.js/);
  });

  it('content widget reads THRESHOLDS from ml-core; no stale 80/50 bands', () => {
    const content = read('content.js');
    expect(content).toMatch(/core\(\)\.THRESHOLDS/);
    expect(content).not.toMatch(/score < (50|80)/);
    expect(read('lib/ml-core.js')).toMatch(/ATTENTIVE_MIN: 70, DISTRACTED_MIN: 40/);
  });
});
