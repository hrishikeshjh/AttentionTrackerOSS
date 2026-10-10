import { describe, it, expect } from 'vitest';
import { DaceEngine, proximityFromEvidence } from '../engine';
import { EvalLogger } from '../eval-log';
import { MLEngine } from '../../ml-engine';
import { FEATURE_WEIGHTS_V1_UPGRADED } from '../../scoring/constants';
import { makeHand, makeObject, mulberry32 } from '../../__tests__/fixtures/synthetic-face';
import { MockDetector, MockHands, MockMeasurement, cfg, frameAt } from './mocks';
import type { DaceEngineDeps } from '../engine';

const CONTEXT_KEYS = ['contextAwarePitch', 'objectContext', 'handObjectProximity', 'temporalPattern'];

async function run(engine: DaceEngine, n: number) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(await engine.processFrame(frameAt(i)));
  return out;
}

function engineWith(o: Partial<DaceEngineDeps> & Pick<DaceEngineDeps, 'config' | 'measurement'>) {
  return new DaceEngine({ frameBudgetMs: 1e9, ...o });
}

describe('DaceEngine fallback (reliability requirement)', () => {
  it('a measurement throw returns { fallback: true } and the next tick recovers', async () => {
    const m = new MockMeasurement('mediapipe', () => 0);
    m.throwAt.add(2);
    const e = engineWith({ config: cfg('F'), measurement: m });
    const r = await run(e, 4);
    expect(r[2]).toMatchObject({ fallback: true, predict: null });
    expect(r[2].fallbackReason).toMatch(/^measure: mock measurement failure/);
    expect(r[3].fallback).toBe(false);
    expect(r[3].predict).not.toBeNull();
    expect(e.getFallbackCount()).toBe(1);
  });

  it('a frame over 2x budget falls back to the legacy path, with the legacy metrics to score', async () => {
    let clock = 0;
    const m = new MockMeasurement('mediapipe', () => 0);
    const e = new DaceEngine({ config: cfg('F'), measurement: m, frameBudgetMs: 200, now: () => (clock += 150) });
    const r = await e.processFrame(frameAt(0));
    expect(r.fallback).toBe(true);
    expect(r.fallbackReason).toMatch(/over_budget/);
    expect(r.predict).toBeNull();
    // legacy metrics: no DACE snapshot, legacy pitch formula
    expect(r.scoreMetrics?.dace).toBeUndefined();
    // the caller scores it with the unchanged legacy path
    const p = new MLEngine().predict(r.scoreMetrics!);
    expect(p.scoringMode).toBeUndefined();
  });

  it('the engine never throws, even when every component fails', async () => {
    const m = new MockMeasurement('mediapipe', () => 0);
    for (let i = 0; i < 10; i++) m.throwAt.add(i);
    const det = new MockDetector(true);
    det.failing = true;
    const e = engineWith({ config: cfg('F'), measurement: m, detector: det });
    const r = await run(e, 10);
    expect(r.every((x) => x.fallback)).toBe(true);
  });
});

describe('FR4: no empty-list upgraded weights in DACE mode', () => {
  it('detector unavailable: explicit upgraded weights with DACE-mapped context features, detectedObjects never set', async () => {
    const det = new MockDetector(false);
    const e = engineWith({ config: cfg('F'), measurement: new MockMeasurement('mediapipe', () => 0), detector: det, hands: new MockHands(true) });
    await e.selfTest();
    expect(e.isDetectorAvailable()).toBe(false);
    expect(e.getGate()).toMatchObject({ useObjects: false, useHands: true, weights: 'upgraded', mapContext: true, exposeObjects: false });
    const r = await run(e, 6);
    expect(det.calls).toHaveLength(0);
    for (const x of r) {
      expect(x.metrics?.detectedObjects).toBeUndefined();
      expect(x.predict?.weights).toBe('upgraded');
      for (const k of CONTEXT_KEYS) expect(x.predict?.featureOverrides?.[k]).toEqual(expect.any(Number));
    }
    // scoring through MLEngine never reaches computeObjectContext([]): objectContext is the mapped neutral 65
    const p = new MLEngine().predict(r[5].scoreMetrics!, { dace: r[5].predict! });
    expect(p.featureContributions.objectContext).toBeCloseTo((65 - 50) * FEATURE_WEIGHTS_V1_UPGRADED.objectContext, 9);
  });

  it('detector available: objects are exposed (filtered by c_min) and used as evidence', async () => {
    const det = new MockDetector(true, () => [makeObject('phone', 0.5, 0.8, 0.9), makeObject('book', 0.2, 0.8, 0.2)]);
    const e = engineWith({ config: cfg('F'), measurement: new MockMeasurement('mediapipe', () => 0), detector: det });
    await e.selfTest();
    const r = await run(e, 3);
    expect(r[2].metrics?.detectedObjects?.map((o) => o.class)).toEqual(['phone']);
  });

  it('evidence off (preset B): legacy weight table and no context overrides', async () => {
    const e = engineWith({ config: cfg('B'), measurement: new MockMeasurement('mediapipe', () => 0) });
    const r = await run(e, 3);
    expect(r[2].predict?.weights).toBe('legacy');
    expect(r[2].predict?.featureOverrides).toBeUndefined();
    expect(r[2].metrics?.deskEvidence).toBeUndefined();
    expect(r[2].metrics?.dace).toBeDefined();
  });
});

describe('staggered scheduling (detector i%3==0, hands i%3==1)', () => {
  it('runs each component on its own ticks and carries outputs forward', async () => {
    const det = new MockDetector(true, () => [makeObject('book', 0.5, 0.8)]);
    const hands = new MockHands(true, () => []);
    const e = engineWith({ config: cfg('F'), measurement: new MockMeasurement('mediapipe', () => 0), detector: det, hands });
    await e.selfTest();
    const r = await run(e, 9);
    expect(det.calls).toEqual([0, 3, 6].map((i) => frameAt(i).t));
    expect(hands.calls).toEqual([1, 4, 7].map((i) => frameAt(i).t));
    expect(r[4].step?.frame.objects).toHaveLength(1); // carried forward from tick 3
  });

  it('5 consecutive detector errors disable the detector (gate re-routes)', async () => {
    const det = new MockDetector(true);
    det.failing = true;
    const e = engineWith({ config: cfg('F'), measurement: new MockMeasurement('mediapipe', () => 0), detector: det });
    await e.selfTest();
    await run(e, 16);
    expect(det.calls).toHaveLength(5);
    expect(e.getGate().useObjects).toBe(false);
  });
});

describe('no-face frames reach the window (absent, U3)', () => {
  it('a full window without a face is labelled absent and records no score', async () => {
    const e = engineWith({ config: cfg('F'), measurement: new MockMeasurement('mediapipe', () => null) });
    const r = await run(e, 15);
    expect(r[14].step?.output?.label).toBe('absent');
    expect(r[14].scoreMetrics).toBeNull();
    expect(r[14].predict).toBeNull();
  });

  it('U3 (face missing in >= 50% of W) holds the score using the last face metrics', async () => {
    const e = engineWith({ config: cfg('F'), measurement: new MockMeasurement('mediapipe', (i) => (i < 30 ? 0 : null)) });
    const r = await run(e, 38);
    const last = r[37];
    expect(last.step?.output?.reason).toBe('U3');
    expect(last.metrics).toBeNull();
    expect(last.predict).toMatchObject({ unsure: true, unsureReason: 'U3' });
    expect(last.scoreMetrics).toBe(r[29].metrics);
  });
});

describe('measurement layer wiring', () => {
  it('pitch reference calibrates for 5 s, then relative pitch drives D_t and legacy pitch = -Δθ', async () => {
    const e = engineWith({ config: cfg('F'), measurement: new MockMeasurement('mediapipe', (i) => (i < 26 ? 10 : 35)) });
    const r = await run(e, 30);
    expect(r[0].calibrating).toBe(true);
    expect(r[0].reset).toBe('start');
    expect(r[29].calibrating).toBe(false);
    expect(r[29].metrics?.dace?.relPitchDeg).toBeCloseTo(25, 9);
    expect(r[29].metrics?.pitch).toBeCloseTo(-25, 9);
    expect(r[29].metrics?.dace?.lookingDown).toBe(true);
  });

  it('long downward episode without evidence -> Unsure U2 (desk not visible); score held in MLEngine', async () => {
    const e = engineWith({ config: cfg('F'), measurement: new MockMeasurement('mediapipe', (i) => (i < 26 ? 0 : 30)) });
    const r = await run(e, 50);
    const ml = new MLEngine();
    const preds = r.map((x) => (x.scoreMetrics && x.predict ? ml.predict(x.scoreMetrics, { dace: x.predict }) : null));
    const last = preds[49]!;
    expect(last).toMatchObject({ unsure: true, unsureReason: 'U2', scoreHeld: true });
    const before = preds[26]!;
    expect(before.unsure).toBe(false);
  });

  it('pnp68 kind substitutes pose only (absolute degrees); no gaze, no openness', async () => {
    const e = engineWith({ config: cfg("A''"), measurement: new MockMeasurement('pnp68', () => 20) });
    const r = await run(e, 2);
    expect(r[1].metrics?.pitch).toBeCloseTo(-20, 9);
    expect(r[1].metrics?.dace?.gazeValid).toBe(false);
    expect(r[1].predict?.weights).toBe('legacy');
  });
});

describe('negative control H: replay.shuffleObjects', () => {
  const truth = (call: number) =>
    call % 2 ? [makeObject('phone', 0.3, 0.7, 0.9)] : [makeObject('book', 0.7, 0.8, 0.9)];

  async function objectsSeen(shuffle: boolean) {
    const det = new MockDetector(true, truth);
    const e = engineWith({
      config: cfg('F', { replay: { shuffleObjects: shuffle } }),
      measurement: new MockMeasurement('mediapipe', () => 0),
      detector: det,
    });
    await e.selfTest();
    const r = await run(e, 60);
    return r.filter((x) => x.i % 3 === 0).map((x) => x.step!.frame.objects.map((o) => o.class).join(','));
  }

  it('off: the window sees the true detector output', async () => {
    const seen = await objectsSeen(false);
    expect(seen).toEqual(seen.map((_, k) => truth(k).map((o) => o.class).join(',')));
  });

  it('on: labels are decoupled from the current output, deterministically', async () => {
    const a = await objectsSeen(true);
    const b = await objectsSeen(true);
    expect(a).toEqual(b);
    const truthSeq = a.map((_, k) => truth(k).map((o) => o.class).join(','));
    expect(a.filter((x, k) => x !== truthSeq[k]).length).toBeGreaterThan(2);
  });
});

describe('eval logging from the engine', () => {
  it('logs a window record per tick (and frame records only at log=frames), all stamped', async () => {
    const logger = new EvalLogger({ level: 'frames', configHash: 'abc12345', sessionId: 'sid' });
    const rnd = mulberry32(1);
    const hands = new MockHands(true, () => [makeHand(0.5, 0.8, 'Right', rnd)]);
    const e = engineWith({ config: cfg('F', { log: 'frames' }), measurement: new MockMeasurement('mediapipe', () => 0), hands, logger });
    await e.selfTest();
    await run(e, 5);
    const recs = logger.getRecords();
    expect(recs.filter((r) => r.type === 'window')).toHaveLength(5);
    expect(recs.filter((r) => r.type === 'frame')).toHaveLength(5);
    for (const r of recs) expect(r).toMatchObject({ v: expect.any(String), cfg: 'abc12345', sid: 'sid' });
  });
});

describe('proximityFromEvidence', () => {
  it('maps evidence onto the legacy HandObjectProximity field', () => {
    const f = (o: Partial<{ phoneInUse: boolean; book: boolean; writing: boolean }>, hv = false) =>
      proximityFromEvidence({ phoneInUse: false, book: false, writing: false, ...o }, hv);
    expect(f({ phoneInUse: true, book: true })).toBe('holding_phone');
    expect(f({ book: true })).toBe('holding_book');
    expect(f({ writing: true })).toBe('holding_pen');
    expect(f({}, true)).toBe('hands_on_desk');
    expect(f({})).toBe('hands_not_visible');
  });
});

describe('offline replay reproduces the live rule outputs (FR8)', () => {
  it('RuleRunner over logged frames == live window records', async () => {
    const { RuleRunner } = await import('../rule-runner');
    const logger = new EvalLogger({ level: 'frames', configHash: 'h' });
    const det = new MockDetector(true, (c) => (c % 4 === 0 ? [makeObject('phone', 0.5, 0.8)] : [makeObject('book', 0.5, 0.8)]));
    const e = engineWith({
      config: cfg('F', { log: 'frames' }),
      measurement: new MockMeasurement('mediapipe', (i) => (i < 30 ? 0 : i < 90 ? 30 : i < 100 ? null : 0)),
      detector: det,
      logger,
    });
    await e.selfTest();
    await run(e, 120);
    const recs = logger.getRecords();
    const runner = new RuleRunner(e.params, { evidence: true, rules: true, abstain: 'rules', shuffleObjects: false });
    const frames = recs.filter((r) => r.type === 'frame') as unknown as Array<{ frame: import('../../scoring/dace/types').DaceFrame; objectsFresh: boolean }>;
    const offline = frames.map((f) => runner.push(JSON.parse(JSON.stringify(f.frame)), f.objectsFresh).effective);
    const live = recs.filter((r) => r.type === 'window').map((r) => ({ label: r.label, unsure: r.unsure, reason: r.reason, rule: r.rule }));
    expect(offline).toEqual(live);
  });
});
