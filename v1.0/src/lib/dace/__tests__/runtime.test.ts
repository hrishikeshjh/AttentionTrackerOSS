import { describe, it, expect, vi } from 'vitest';
import { LatencyProbe, percentile } from '../latency';
import { FixedTickScheduler } from '../scheduler';
import { EvalLogger, FORBIDDEN_KEYS, parseJsonl, randomSessionId } from '../eval-log';
import { DACE_VERSION } from '../../scoring/dace/VERSION';
import { RuleRunner, evidenceOnlyOutput } from '../rule-runner';
import { ObjectShuffler } from '../object-shuffle';
import { contextGate, buildSelfTestReport } from '../self-test';
import { frameTimes } from '../replay/frame-source';
import { P, summary, frame, down, obj, writingHand } from '../../scoring/dace/__tests__/helpers';

describe('latency probe', () => {
  it('nearest-rank p50/p95/max per component over a ring buffer', async () => {
    const lp = new LatencyProbe(100);
    for (let i = 1; i <= 100; i++) lp.record('measure', i);
    expect(lp.stats('measure')).toEqual({ n: 100, p50: 50, p95: 95, max: 100, last: 100 });
    for (let i = 0; i < 50; i++) lp.record('measure', 1000); // ring overwrites the oldest 50
    expect(lp.stats('measure')?.n).toBe(100);
    expect(lp.stats('measure')?.p50).toBe(100);
    expect(lp.stats('measure')?.max).toBe(1000);
    expect(percentile([], 95)).toBe(0);
  });

  it('time() records sync, async and throwing calls with an injected clock', async () => {
    let c = 0;
    const lp = new LatencyProbe(10, () => (c += 5));
    await lp.time('a', () => 1);
    await lp.time('b', async () => 2);
    await expect(lp.time('c', () => { throw new Error('x'); })).rejects.toThrow('x');
    expect(Object.keys(lp.summary()).sort()).toEqual(['a', 'b', 'c']);
    expect(lp.stats('a')?.last).toBe(5);
    expect(lp.lastValues()).toEqual({ a: 5, b: 5, c: 5 });
  });
});

describe('fixed 200 ms scheduler', () => {
  it('skips (does not queue) a tick that arrives while the previous one is busy', async () => {
    let release: () => void = () => {};
    const ran: number[] = [];
    const dropped: number[] = [];
    const s = new FixedTickScheduler(
      async (tick) => {
        ran.push(tick.i);
        if (tick.i === 0) await new Promise<void>((r) => (release = r));
      },
      { now: () => 0, onDropped: (d) => dropped.push(d.i), setInterval: () => 1, clearInterval: () => {} }
    );
    s.start();
    const first = s.onTick(); // busy
    await s.onTick(); // dropped
    await s.onTick(); // dropped
    release();
    await first;
    await s.onTick(); // runs
    expect(ran).toEqual([0, 3]);
    expect(dropped).toEqual([1, 2]);
    expect(s.stats()).toEqual({ ticks: 4, ran: 2, dropped: 2, errors: 0 });
    s.stop();
    expect(s.isRunning()).toBe(false);
  });

  it('uses a 200 ms interval by default and survives task errors', async () => {
    vi.useFakeTimers();
    const seen: number[] = [];
    const errors: unknown[] = [];
    const s = new FixedTickScheduler(
      (tick) => {
        seen.push(tick.i);
        if (tick.i === 1) throw new Error('boom');
      },
      { onError: (e) => errors.push(e) }
    );
    expect(s.intervalMs).toBe(200);
    s.start();
    await vi.advanceTimersByTimeAsync(1000);
    s.stop();
    vi.useRealTimers();
    expect(seen).toEqual([0, 1, 2, 3, 4]);
    expect(errors).toHaveLength(1);
    expect(s.stats().errors).toBe(1);
  });
});

describe('FR9 eval log', () => {
  it('stamps v (DACE_VERSION), cfg and a random sid on every record; no images or user ids', () => {
    const l = new EvalLogger({ level: 'windows', configHash: 'deadbeef' });
    expect(l.sessionId).toMatch(/^[0-9a-f]{32}$/);
    expect(randomSessionId()).not.toBe(randomSessionId());
    l.log('session', 0, { config: { enabled: true }, image: 'data:image/png;base64,xx', userId: 'u1', email: 'a@b' });
    l.log('window', 200, { label: 'attentive_screen', pixels: new Uint8Array(4), nested: { imageData: [1], ok: 1, bad: NaN } });
    l.log('frame', 200, { frame: {} }); // not captured at 'windows'
    const recs = parseJsonl(l.toJSONL());
    expect(recs).toHaveLength(2);
    for (const r of recs) {
      expect(r).toMatchObject({ v: DACE_VERSION, cfg: 'deadbeef', sid: l.sessionId });
      expect(typeof r.t).toBe('number');
      for (const k of FORBIDDEN_KEYS) expect(JSON.stringify(r)).not.toContain(`"${k}"`);
    }
    expect(recs[1]).toMatchObject({ type: 'window', label: 'attentive_screen', nested: { ok: 1, bad: null } });
    expect(recs[1]).not.toHaveProperty('pixels');
  });

  it('record fields cannot override the stamps; off logs nothing; frames level captures frames', () => {
    const l = new EvalLogger({ level: 'frames', configHash: 'h', sessionId: 's' });
    l.log('frame', 1, { v: 'forged', cfg: 'forged', sid: 'forged', type: 'session' });
    expect(l.getRecords()[0]).toMatchObject({ type: 'frame', v: DACE_VERSION, cfg: 'h', sid: 's' });
    const off = new EvalLogger({ level: 'off', configHash: 'h' });
    off.log('window', 1);
    expect(off.size()).toBe(0);
    expect(off.toJSONL()).toBe('');
    const capped = new EvalLogger({ level: 'windows', configHash: 'h', maxRecords: 2 });
    for (let i = 0; i < 5; i++) capped.log('window', i);
    expect(capped.size()).toBe(2);
    expect(capped.droppedRecords()).toBe(3);
  });
});

describe('rule runner (shared by the engine and the offline replay)', () => {
  it('preset C (evidence on, rules off) produces an evidence-only label', () => {
    expect(evidenceOnlyOutput(summary({ phoneInUse: true }))).toMatchObject({ label: 'using_phone', rule: 'evidence_only' });
    expect(evidenceOnlyOutput(summary({ lookingDownMost: true, book: true }))).toMatchObject({ label: 'attentive_reading' });
    expect(evidenceOnlyOutput(summary({ lookingDownMost: false, book: true })).label).toBeNull();
  });

  it('evidence off strips hands/objects; rules on produce reading from writing evidence', () => {
    const on = new RuleRunner(P, { evidence: true, rules: true, abstain: 'rules', shuffleObjects: false });
    const off = new RuleRunner(P, { evidence: false, rules: true, abstain: 'rules', shuffleObjects: false });
    let a, b;
    for (let i = 0; i < 20; i++) {
      const f = frame(i, down(), [writingHand('Right', i)], [obj('book', 0.6, 0.8)]);
      a = on.push(f);
      b = off.push(f);
    }
    expect(a!.effective).toMatchObject({ label: 'attentive_reading', rule: 'reading' });
    expect(b!.frame.hands).toHaveLength(0);
    expect(b!.effective.unsure).toBe(true);
  });

  it('shuffler: deterministic for a seed, only fresh outputs are drawn', () => {
    const s1 = new ObjectShuffler(3);
    const s2 = new ObjectShuffler(3);
    const lists = [[obj('book', 0.1, 0.1)], [obj('phone', 0.2, 0.2)], [], [obj('book', 0.3, 0.3)]];
    expect(lists.map((l) => s1.next(l))).toEqual(lists.map((l) => s2.next(l)));
  });
});

describe('FR4 context gate', () => {
  it('upgraded weights only with evidence on; objects only when the detector is available', () => {
    expect(contextGate(false, true, true)).toMatchObject({ weights: 'legacy', mapContext: false, useObjects: false });
    expect(contextGate(true, false, true)).toMatchObject({ weights: 'upgraded', mapContext: true, useObjects: false, useHands: true, exposeObjects: false });
    expect(contextGate(true, true, false)).toMatchObject({ useObjects: true, useHands: false, exposeObjects: true });
  });

  it('detectorAvailable requires loaded AND self-test passed', () => {
    const st = (o: object) => ({ kind: 'yolo' as const, loaded: true, selfTestOk: true, selfTestMs: 3, checksumVerified: null, error: null, ...o });
    expect(buildSelfTestReport({ measurement: null, detector: st({}), hands: null }).detectorAvailable).toBe(true);
    expect(buildSelfTestReport({ measurement: null, detector: st({ selfTestOk: false }), hands: null }).detectorAvailable).toBe(false);
    expect(buildSelfTestReport({ measurement: null, detector: st({ selfTestOk: null }), hands: null }).detectorAvailable).toBe(false);
    expect(buildSelfTestReport({ measurement: null, detector: st({ loaded: false }), hands: null }).detectorAvailable).toBe(false);
    const r = buildSelfTestReport({ measurement: null, detector: st({ selfTestOk: false, checksumVerified: false }), hands: null });
    expect(r.components[0].detail).toBe('checksum mismatch');
  });
});

describe('replay file stepping', () => {
  it('visits k * 200 ms strictly inside the duration', () => {
    expect(frameTimes(1000, 200)).toEqual([0, 200, 400, 600, 800]);
    expect(frameTimes(1001, 200)).toEqual([0, 200, 400, 600, 800, 1000]);
    expect(frameTimes(0, 200)).toEqual([]);
  });
});
