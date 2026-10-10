import { describe, it, expect } from 'vitest';
import { DaceRuleEngine, evaluateRules, initialRuleState, type RuleInput, type RuleOptions } from '../rules';
import { DaceWindow } from '../window';
import { CorrectedPerclos } from '../perclos';
import { P, down, face, frame, hand, obj, summary, writingHand } from './helpers';
import type { WindowSummary, DaceFrame } from '../types';

const ON: RuleOptions = { abstain: 'rules' };
const OFF: RuleOptions = { abstain: 'off' };

function evalOnce(s: Partial<WindowSummary>, extra: Partial<RuleInput> = {}, opts: RuleOptions = ON) {
  return evaluateRules({ summary: summary(s), perclos: 0, longestClosureMs: 0, ...extra }, initialRuleState(), P, opts).output;
}

describe('rule layer: each rule', () => {
  it('1 absent: no face in ≥ 80% of W (12/15)', () => {
    expect(evalOnce({ noFaceCount: 12 }).label).toBe('absent');
    expect(evalOnce({ noFaceCount: 11 }).rule).toBe('U3');
  });
  it('U3: face missing in ≥ 50% of W (8/15), not 7/15', () => {
    const o = evalOnce({ noFaceCount: 8 });
    expect(o).toMatchObject({ unsure: true, reason: 'U3', label: null });
    expect(evalOnce({ noFaceCount: 7 }).rule).toBe('screen');
  });
  it('2 drowsy: PERCLOS ≥ 0.15 (inclusive) or a closure ≥ 2 s', () => {
    expect(evalOnce({}, { perclos: 0.15 }).label).toBe('drowsy');
    expect(evalOnce({}, { perclos: 0.1499 }).label).not.toBe('drowsy');
    expect(evalOnce({}, { longestClosureMs: 2000 }).label).toBe('drowsy');
    expect(evalOnce({}, { longestClosureMs: 1999 }).label).not.toBe('drowsy');
  });
  it('U4: conflicting evidence', () => {
    expect(evalOnce({ conflict: true, phoneInUse: true })).toMatchObject({ unsure: true, reason: 'U4' });
  });
  it('3 phone in use', () => {
    expect(evalOnce({ phoneInUse: true, lookingDownMost: true, inDownEpisode: true, downEpisodeMs: 5000 })).toMatchObject({ label: 'using_phone', rule: 'phone' });
  });
  it('a phone merely present does not trigger the phone rule', () => {
    expect(evalOnce({ phonePresent: true }).label).toBe('attentive_screen');
  });
  it('4 reading: looking down in most of W + book, or + writing', () => {
    expect(evalOnce({ lookingDownMost: true, book: true }).label).toBe('attentive_reading');
    expect(evalOnce({ lookingDownMost: true, writing: true }).label).toBe('attentive_reading');
    expect(evalOnce({ lookingDownMost: false, book: true }).label).toBe('attentive_screen');
  });
  it('6 long downward episode without evidence -> Unsure U1 (desk visible) / U2 (desk not visible)', () => {
    expect(evalOnce({ lookingDownMost: true, inDownEpisode: true, downEpisodeMs: 3200, deskVisible: true })).toMatchObject({ unsure: true, reason: 'U1', rule: 'U1' });
    expect(evalOnce({ lookingDownMost: true, inDownEpisode: true, downEpisodeMs: 3200, deskVisible: false })).toMatchObject({ unsure: true, reason: 'U2' });
  });
  it('7 screen: attentive_screen or distracted', () => {
    expect(evalOnce({ screenAttentive: true }).label).toBe('attentive_screen');
    expect(evalOnce({ screenAttentive: false }).label).toBe('distracted');
  });
});

describe('rule layer: ordering conflicts', () => {
  it('absent beats U3 and everything else', () => {
    expect(evalOnce({ noFaceCount: 15, phoneInUse: true, conflict: true }, { perclos: 1 }).label).toBe('absent');
  });
  it('U3 beats drowsy', () => {
    expect(evalOnce({ noFaceCount: 9 }, { perclos: 0.5 }).rule).toBe('U3');
  });
  it('drowsy beats reading (dozing over a book, B.1) and phone', () => {
    expect(evalOnce({ lookingDownMost: true, book: true }, { perclos: 0.3 }).label).toBe('drowsy');
    expect(evalOnce({ phoneInUse: true }, { longestClosureMs: 2400 }).label).toBe('drowsy');
  });
  it('drowsy beats U4', () => {
    expect(evalOnce({ conflict: true, phoneInUse: true }, { perclos: 0.2 }).label).toBe('drowsy');
  });
  it('U4 beats phone', () => {
    expect(evalOnce({ conflict: true, phoneInUse: true }).rule).toBe('U4');
  });
  it('phone beats reading evidence (book visible while on the phone, segment 7)', () => {
    expect(evalOnce({ phoneInUse: true, lookingDownMost: true, book: true }).label).toBe('using_phone');
  });
  it('reading beats glance and U1/U2', () => {
    expect(evalOnce({ lookingDownMost: true, book: true, inDownEpisode: true, downEpisodeMs: 9000, deskVisible: false }).label).toBe('attentive_reading');
  });
  it('U2 takes precedence over U1', () => {
    expect(evalOnce({ inDownEpisode: true, downEpisodeMs: 4000, deskVisible: false }).reason).toBe('U2');
  });
});

describe('rule layer: state (glance, phone hold)', () => {
  it('5 glance ≤ τ_g keeps the previous output; > τ_g goes to rule 6', () => {
    const e = new DaceRuleEngine(P, ON);
    const prev = e.evaluate({ summary: summary({ t: 0, screenAttentive: false }), perclos: 0, longestClosureMs: 0 });
    expect(prev.label).toBe('distracted');
    const g = e.evaluate({ summary: summary({ t: 200, inDownEpisode: true, downEpisodeMs: 3000 }), perclos: 0, longestClosureMs: 0 });
    expect(g).toMatchObject({ label: 'distracted', rule: 'glance', keptFrom: 'screen' });
    const g2 = e.evaluate({ summary: summary({ t: 400, inDownEpisode: true, downEpisodeMs: 3000 }), perclos: 0, longestClosureMs: 0 });
    expect(g2.keptFrom).toBe('screen');
    const u = e.evaluate({ summary: summary({ t: 600, inDownEpisode: true, downEpisodeMs: 3200 }), perclos: 0, longestClosureMs: 0 });
    expect(u).toMatchObject({ unsure: true, reason: 'U1' });
  });

  it('a glance with no previous output falls through to screen scoring', () => {
    expect(evalOnce({ inDownEpisode: true, downEpisodeMs: 1000 }).rule).toBe('screen');
  });

  it('3 using_phone is held for 10 s after the last firing', () => {
    const e = new DaceRuleEngine(P, ON);
    const t0 = 1_000_000;
    const run = (dt: number, s: Partial<WindowSummary> = {}) =>
      e.evaluate({ summary: summary({ t: t0 + dt, ...s }), perclos: 0, longestClosureMs: 0 });
    expect(run(0, { phoneInUse: true }).rule).toBe('phone');
    expect(run(5000).rule).toBe('phone_hold');
    expect(run(9999).label).toBe('using_phone');
    expect(run(10000).rule).toBe('screen');
    // hold does not override higher-priority rules
    expect(run(0, { phoneInUse: true }).rule).toBe('phone');
    expect(run(2000, {}, ).rule).toBe('phone_hold');
    expect(e.evaluate({ summary: summary({ t: t0 + 2200 }), perclos: 0.5, longestClosureMs: 0 }).label).toBe('drowsy');
    // absent clears the hold
    expect(run(2400, { noFaceCount: 15 }).label).toBe('absent');
    expect(run(2600).rule).toBe('screen');
  });

  it('abstention off: U3/U4 skipped, rule 6 outputs distracted', () => {
    expect(evalOnce({ noFaceCount: 9 }, {}, OFF).rule).toBe('screen');
    expect(evalOnce({ conflict: true, phoneInUse: true }, {}, OFF).label).toBe('using_phone');
    expect(evalOnce({ inDownEpisode: true, downEpisodeMs: 4000, deskVisible: false }, {}, OFF)).toMatchObject({ label: 'distracted', rule: 'down_no_evidence', unsure: false });
  });
});

describe('rule layer end-to-end through the window', () => {
  function runFrames(frames: DaceFrame[]) {
    const w = new DaceWindow(P);
    const pc = new CorrectedPerclos(P);
    const e = new DaceRuleEngine(P, ON);
    let out = null;
    for (const f of frames) {
      w.push(f);
      pc.push(f.t, f.face?.eyeClosed ?? null);
      const start = w.windowStartT() ?? f.t;
      out = e.evaluate({ summary: w.summary(), perclos: pc.value().perclos, longestClosureMs: pc.longestClosureMs(start) });
    }
    return out!;
  }

  it('reading a held book (segment 3) -> attentive_reading', () => {
    const fs = Array.from({ length: 20 }, (_, i) => frame(i, down(), [hand('Right', 0.5, 0.8)], [obj('book', 0.5, 0.82, 0.2, 0.15)]));
    expect(runFrames(fs).label).toBe('attentive_reading');
  });
  it('writing notes (segment 2) -> attentive_reading', () => {
    const fs = Array.from({ length: 20 }, (_, i) => frame(i, down(), [writingHand('Left', i)]));
    expect(runFrames(fs).label).toBe('attentive_reading');
  });
  it('phone in hand (segment 4) -> using_phone', () => {
    const fs = Array.from({ length: 20 }, (_, i) => frame(i, down(), [hand('Right', 0.5, 0.8)], [obj('phone', 0.5, 0.81)]));
    expect(runFrames(fs).label).toBe('using_phone');
  });
  it('desk out of frame (segment 10) -> Unsure U2 after τ_g', () => {
    const fs = Array.from({ length: 30 }, (_, i) => frame(i, down()));
    expect(runFrames(fs)).toMatchObject({ unsure: true, reason: 'U2' });
  });
  it('loose paper, hands visible, no detectable object (U1)', () => {
    const fs = Array.from({ length: 30 }, (_, i) => frame(i, down(), [hand('Right', 0.5, 0.8)]));
    expect(runFrames(fs)).toMatchObject({ unsure: true, reason: 'U1' });
  });
  it('acted drowsiness while looking down (segment 8) -> drowsy', () => {
    const fs = Array.from({ length: 20 }, (_, i) => frame(i, down({ eyeClosed: true, residualOpenness: 0.3 }), [hand('Right', 0.5, 0.8)], [obj('book', 0.5, 0.82, 0.2, 0.15)]));
    expect(runFrames(fs).label).toBe('drowsy');
  });
  it('brief note glance (segment 9) keeps the screen label', () => {
    const fs = [
      ...Array.from({ length: 20 }, (_, i) => frame(i, face())),
      ...Array.from({ length: 10 }, (_, i) => frame(20 + i, down())),
    ];
    expect(runFrames(fs)).toMatchObject({ label: 'attentive_screen', rule: 'glance' });
  });
  it('student leaves -> absent', () => {
    const fs = Array.from({ length: 15 }, (_, i) => frame(i, null));
    expect(runFrames(fs).label).toBe('absent');
  });
});
