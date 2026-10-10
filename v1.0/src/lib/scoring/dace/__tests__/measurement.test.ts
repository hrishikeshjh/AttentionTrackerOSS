import { describe, it, expect } from 'vitest';
import { PitchReference, median } from '../pitch-reference';
import { estimateGaze, gazeFromBlendshapes, blendshapeMap, irisOffset } from '../gaze-blendshape';
import { OpennessModel, huberFitQuadratic, huberLocation, evalQuad } from '../eye-openness';
import { CorrectedPerclos } from '../perclos';
import { makeParams } from '../params';
import { mulberry32 } from '../../../__tests__/fixtures/synthetic-face';
import { EAR_BLINK_THRESHOLD } from '../../constants';
import { P } from './helpers';

const T0 = 1_000_000;

describe('pitch reference θ_ref (5 s median, resets)', () => {
  it('is the median of the first 5 s and then fixed; Δθ relative to it', () => {
    const ref = new PitchReference(P);
    const vals = [10, 12, 11, 30, 9]; // one outlier
    let st;
    for (let i = 0; i <= 25; i++) st = ref.update({ t: T0 + i * 200, pitchDeg: vals[i % 5], faceScale: 1, gazeV: 0 });
    expect(st!.calibrating).toBe(false);
    expect(st!.refDeg).toBe(11);
    const after = ref.update({ t: T0 + 26 * 200, pitchDeg: 31, faceScale: 1, gazeV: 0 });
    expect(after.relPitchDeg).toBe(20);
  });

  it('is calibrating (running median) for the first 5 s and reports the start reset', () => {
    const ref = new PitchReference(P);
    const s0 = ref.update({ t: T0, pitchDeg: 4, faceScale: 1, gazeV: 0 });
    expect(s0.calibrating).toBe(true);
    expect(s0.reset).toBe('start');
    expect(s0.relPitchDeg).toBe(0);
  });

  it('ignores eyes-down frames while estimating', () => {
    const ref = new PitchReference(P);
    let st;
    for (let i = 0; i <= 25; i++) {
      const eyesDown = i % 2 === 0;
      st = ref.update({ t: T0 + i * 200, pitchDeg: eyesDown ? 40 : 5, faceScale: 1, gazeV: eyesDown ? 0.8 : 0 });
    }
    expect(st!.refDeg).toBe(5);
  });

  it('re-estimates after the student returns from absence', () => {
    const ref = new PitchReference(P);
    let t = T0;
    for (let i = 0; i <= 25; i++) ref.update({ t: (t += 200), pitchDeg: 5, faceScale: 1, gazeV: 0 });
    for (let i = 0; i < 12; i++) ref.update({ t: (t += 200), pitchDeg: null, faceScale: null, gazeV: null });
    const back = ref.update({ t: (t += 200), pitchDeg: 20, faceScale: 1, gazeV: 0 });
    expect(back.reset).toBe('absent_return');
    expect(back.calibrating).toBe(true);
    let st;
    for (let i = 0; i <= 25; i++) st = ref.update({ t: (t += 200), pitchDeg: 20, faceScale: 1, gazeV: 0 });
    expect(st!.refDeg).toBe(20);
  });

  it('a short face loss (< 80% of W) does not reset', () => {
    const ref = new PitchReference(P);
    let t = T0;
    for (let i = 0; i <= 25; i++) ref.update({ t: (t += 200), pitchDeg: 5, faceScale: 1, gazeV: 0 });
    for (let i = 0; i < 5; i++) ref.update({ t: (t += 200), pitchDeg: null, faceScale: null, gazeV: null });
    const back = ref.update({ t: (t += 200), pitchDeg: 5, faceScale: 1, gazeV: 0 });
    expect(back.reset).toBeNull();
    expect(back.calibrating).toBe(false);
  });

  it('resets on a camera move (face-scale jump between 1 s medians)', () => {
    const ref = new PitchReference(P);
    let t = T0;
    for (let i = 0; i <= 30; i++) ref.update({ t: (t += 200), pitchDeg: 5, faceScale: 100, gazeV: 0 });
    let resetSeen = null;
    for (let i = 0; i < 10; i++) {
      const s = ref.update({ t: (t += 200), pitchDeg: 5, faceScale: 140, gazeV: 0 });
      resetSeen = resetSeen ?? s.reset;
    }
    expect(resetSeen).toBe('camera_move');
    expect(ref.isCalibrating()).toBe(true);
  });

  it('uses the PEB baseline when provided', () => {
    const ref = new PitchReference(P, { pitchReferenceDeg: () => -3 });
    const s = ref.update({ t: T0, pitchDeg: 12, faceScale: 1, gazeV: 0 });
    expect(s.source).toBe('peb');
    expect(s.relPitchDeg).toBe(15);
    expect(s.calibrating).toBe(false);
  });

  it('median helper', () => {
    expect(median([])).toBeNull();
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 2, 3])).toBe(2.5);
  });
});

describe('blendshape gaze (FR2)', () => {
  const base = {
    eyeLookDownLeft: 0, eyeLookDownRight: 0, eyeLookUpLeft: 0, eyeLookUpRight: 0,
    eyeLookInLeft: 0, eyeLookInRight: 0, eyeLookOutLeft: 0, eyeLookOutRight: 0,
  };
  it('g = ½(down_L + down_R) − ½(up_L + up_R)', () => {
    expect(gazeFromBlendshapes({ ...base, eyeLookDownLeft: 0.8, eyeLookDownRight: 0.6, eyeLookUpLeft: 0.1, eyeLookUpRight: 0.1 })!.v).toBeCloseTo(0.6, 12);
    expect(gazeFromBlendshapes({ ...base, eyeLookUpLeft: 1, eyeLookUpRight: 1 })!.v).toBe(-1);
  });
  it('horizontal gaze from in/out; magnitude symmetric', () => {
    const right = gazeFromBlendshapes({ ...base, eyeLookOutLeft: 0.6, eyeLookInRight: 0.6 })!;
    const left = gazeFromBlendshapes({ ...base, eyeLookInLeft: 0.6, eyeLookOutRight: 0.6 })!;
    expect(right.h).toBeCloseTo(0.6, 12);
    expect(left.h).toBeCloseTo(-0.6, 12);
  });
  it('missing keys -> null; categories -> map', () => {
    expect(gazeFromBlendshapes({ eyeLookDownLeft: 1 })).toBeNull();
    expect(blendshapeMap([{ categoryName: 'a', score: 0.5 }])).toEqual({ a: 0.5 });
  });

  function meshWithIris(dy: number) {
    const m = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5 }));
    m[33] = { x: 0.40, y: 0.40 }; m[133] = { x: 0.46, y: 0.40 }; m[468] = { x: 0.43, y: 0.40 + dy };
    m[362] = { x: 0.54, y: 0.40 }; m[263] = { x: 0.60, y: 0.40 }; m[473] = { x: 0.57, y: 0.40 + dy };
    return m;
  }
  it('iris offset (fraction of eye width, +down) and cross-check', () => {
    const p = makeParams({ frameAspect: 1 });
    expect(irisOffset(meshWithIris(0.012), 1)!.v).toBeCloseTo(0.2, 9);
    const bsDown = { ...base, eyeLookDownLeft: 0.8, eyeLookDownRight: 0.8 };
    expect(estimateGaze(bsDown, meshWithIris(0.012), p).gazeValid).toBe(true);
    // blendshapes say "down", iris clearly up -> inconsistent -> invalid
    expect(estimateGaze(bsDown, meshWithIris(-0.012), p).gazeValid).toBe(false);
    // no iris (e.g. reflective glasses) -> invalid, D_t uses head pitch only
    const noIris = estimateGaze(bsDown, null, p);
    expect(noIris.gazeValid).toBe(false);
    expect(noIris.gazeV).toBeCloseTo(0.8, 12);
    // implausible iris position
    expect(estimateGaze(bsDown, meshWithIris(0.06), p).irisValid).toBe(false);
  });
});

describe('gaze-corrected eye openness (FR3)', () => {
  it('Huber fit recovers the quadratic despite 15% gross outliers (blinks)', () => {
    const rnd = mulberry32(11);
    const truth = { b0: 0.3, b1: -0.1, b2: -0.05 };
    const s = [];
    for (let i = 0; i < 300; i++) {
      const g = -0.5 + 1.5 * rnd();
      let e = evalQuad(truth, g) + (rnd() - 0.5) * 0.004;
      if (i % 7 === 0) e = 0.05 + 0.05 * rnd(); // blink outliers
      s.push({ g, e });
    }
    const fit = huberFitQuadratic(s)!;
    expect(fit.b0).toBeCloseTo(truth.b0, 2);
    expect(fit.b1).toBeCloseTo(truth.b1, 2);
    expect(fit.b2).toBeCloseTo(truth.b2, 1);
    expect(huberLocation([1, 1, 1, 1, 50])).toBeCloseTo(1, 6);
  });

  it('uses the prior shape (level-only fit) when gaze range is too narrow', () => {
    const m = new OpennessModel(P);
    for (let i = 0; i < 60; i++) m.addSample(0.05 * (i % 3), 0.33);
    const fit = m.refit();
    expect(fit.kind).toBe('level');
    expect(fit.coef.b1 / fit.coef.b0).toBeCloseTo(P.earPrior.b1 / P.earPrior.b0, 12);
    expect(m.expected(0)).toBeCloseTo(fit.coef.b0, 12);
  });

  it('prior until enough samples; quadratic fit with wide gaze range', () => {
    const m = new OpennessModel(P);
    expect(m.refit().kind).toBe('prior');
    for (let i = 0; i < 100; i++) {
      const g = -0.4 + (i / 100) * 1.2;
      m.addSample(g, 0.3 - 0.1 * g);
    }
    const fit = m.refit();
    expect(fit.kind).toBe('quadratic');
    expect(fit.coef.b1).toBeCloseTo(-0.1, 6);
  });

  it('closed iff r < k (strict boundary)', () => {
    const m = new OpennessModel(P); // prior: Ê(0) = 0.3
    expect(m.residual(0.21, 0)).toBeCloseTo(0.7, 12);
    expect(m.isClosed(0.7)).toBe(false);
    expect(m.isClosed(0.6999)).toBe(true);
    expect(m.isClosed(null)).toBeNull();
  });

  it('a lid lowered by looking down is not counted as closed', () => {
    const m = new OpennessModel(P); // prior Ê(g) = 0.3 − 0.12 g
    const g = 0.8; // eyes down
    const ear = 0.2; // legacy EAR < 0.21 -> "closed"
    expect(ear < EAR_BLINK_THRESHOLD).toBe(true);
    const r = m.residual(ear, g)!; // 0.2 / 0.204
    expect(m.isClosed(r)).toBe(false);
  });

  it('PEB personal open-eye level scales Ê(g) multiplicatively', () => {
    const m = new OpennessModel(P);
    expect(m.expected(0, 0.36)).toBeCloseTo(0.36, 12);
    expect(m.expected(0.5, 0.36) / m.expected(0, 0.36)).toBeCloseTo(m.expected(0.5) / m.expected(0), 12);
  });
});

describe('corrected PERCLOS and continuous closure', () => {
  it('PERCLOS = closed / valid frames in 60 s; 0 below the minimum sample count', () => {
    const pc = new CorrectedPerclos(P);
    for (let i = 0; i < 9; i++) pc.push(T0 + i * 200, true);
    expect(pc.value().perclos).toBe(0);
    pc.push(T0 + 9 * 200, false);
    expect(pc.value().perclos).toBeCloseTo(0.9, 12);
    pc.push(T0 + 10 * 200, null); // invalid frames do not count
    expect(pc.value().n).toBe(10);
  });

  it('drops samples older than 60 s', () => {
    const pc = new CorrectedPerclos(P);
    for (let i = 0; i < 300; i++) pc.push(T0 + i * 200, true); // 60 s closed
    for (let i = 300; i < 600; i++) pc.push(T0 + i * 200, false); // 60 s open
    expect(pc.value().perclos).toBe(0);
  });

  it('regression: looking down at a page does not inflate corrected PERCLOS', () => {
    const m = new OpennessModel(P);
    const pc = new CorrectedPerclos(P);
    let legacyClosed = 0;
    for (let i = 0; i < 100; i++) {
      const g = 0.8;
      const ear = 0.2 + ((i % 5) - 2) * 0.002; // lowered lid while reading
      if (ear < EAR_BLINK_THRESHOLD) legacyClosed++;
      pc.push(T0 + i * 200, m.isClosed(m.residual(ear, g)));
    }
    expect(legacyClosed / 100).toBeGreaterThan(0.15); // legacy would call this drowsy
    expect(pc.value().perclos).toBeLessThan(0.15);
  });

  it('closure duration: 10 consecutive closed frames = 2000 ms; null breaks a run', () => {
    const pc = new CorrectedPerclos(P);
    for (let i = 0; i < 10; i++) pc.push(T0 + i * 200, true);
    pc.push(T0 + 10 * 200, false);
    expect(pc.longestClosureMs(T0)).toBe(2000);
    expect(pc.longestClosureMs(T0 + 10 * 200)).toBe(0); // run ended before sinceT
    const pc2 = new CorrectedPerclos(P);
    for (let i = 0; i < 10; i++) pc2.push(T0 + i * 200, i === 5 ? null : true);
    expect(pc2.longestClosureMs(T0)).toBe(1000);
  });
});
