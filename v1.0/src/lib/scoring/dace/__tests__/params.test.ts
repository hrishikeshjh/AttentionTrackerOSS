import { describe, it, expect } from 'vitest';
import {
  DEFAULT_DACE_PARAMS,
  PROSE_CONSTANTS,
  TABLE_A6,
  TBD_PARAM_KEYS,
  atLeastShare,
  makeParams,
  moreThanShare,
} from '../params';
import { DACE_VERSION } from '../VERSION';

describe('DACE params', () => {
  it('Table A.6 defaults are exactly the annex values', () => {
    // Frozen copy of Annex Table A.6. Changing a value requires a documented P1 tuning run.
    expect({ ...TABLE_A6 }).toEqual({
      thetaDownDeg: 15,
      gammaDown: 0.5,
      kOpen: 0.7,
      pDrowsy: 0.15,
      cMin: 0.35,
      rhoUse: 0.6,
      piMax: 0.35,
      mMin: 0.02,
      mMax: 0.15,
      tauGlanceMs: 3000,
      alphaConformal: 0.1,
    });
    for (const [k, v] of Object.entries(TABLE_A6)) {
      expect(DEFAULT_DACE_PARAMS[k as keyof typeof TABLE_A6]).toBe(v);
    }
  });

  it('prose constants match the annex text', () => {
    expect({ ...PROSE_CONSTANTS }).toEqual({
      fps: 5,
      frameIntervalMs: 200,
      windowFrames: 15,
      windowMs: 3000,
      referenceMs: 5000,
      absentShare: 0.8,
      u3MissingShare: 0.5,
      holdIoU: 0.1,
      nearDiagFactor: 0.5,
      phonePresentShare: 0.3,
      bookShare: 0.5,
      writingShare: 0.6,
      mostShare: 0.5,
      deskWindowFrames: 25,
      deskShare: 0.3,
      phoneHoldMs: 10000,
      closureMs: 2000,
      perclosWindowMs: 60000,
      detectorEveryNFrames: 3,
    });
    expect(PROSE_CONSTANTS.windowFrames * PROSE_CONSTANTS.frameIntervalMs).toBe(PROSE_CONSTANTS.windowMs);
  });

  it('lists every provisional (TBD) parameter explicitly', () => {
    expect(TBD_PARAM_KEYS).toEqual(
      expect.arrayContaining(['cameraMovePitchJumpDeg', 'cameraMoveFaceScaleJump', 'earPrior', 'earScale', 'gazeScale'])
    );
  });

  it('defaults are frozen; makeParams copies', () => {
    expect(Object.isFrozen(DEFAULT_DACE_PARAMS)).toBe(true);
    const p = makeParams({ thetaDownDeg: 20 });
    expect(p.thetaDownDeg).toBe(20);
    p.earPrior.b0 = 99;
    expect(DEFAULT_DACE_PARAMS.earPrior.b0).not.toBe(99);
  });

  it('share helpers are robust to floating point (ρ_use · 15 = 9)', () => {
    expect(atLeastShare(9, 15, 0.6)).toBe(true);
    expect(atLeastShare(8, 15, 0.6)).toBe(false);
    expect(atLeastShare(8, 25, 0.3)).toBe(true); // 7.5 -> 8
    expect(atLeastShare(7, 25, 0.3)).toBe(false);
    expect(moreThanShare(8, 15, 0.5)).toBe(true);
    expect(moreThanShare(7, 15, 0.5)).toBe(false);
  });

  it('has a version string', () => {
    expect(DACE_VERSION).toMatch(/^\d+\.\d+\.\d+$/);
  });
});
