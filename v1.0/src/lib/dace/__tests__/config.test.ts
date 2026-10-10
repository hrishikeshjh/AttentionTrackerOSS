import { describe, it, expect } from 'vitest';
import {
  DEFAULT_DACE_CONFIG,
  DACE_STORAGE_KEY,
  configHash,
  parseOverrideString,
  presetConfig,
  resolveDaceConfig,
} from '../config';
import { DACE_PRESETS, lookupPreset } from '../presets';
import { DACE_VERSION } from '@/lib/scoring/dace/VERSION';

const store = (v: string | null) => ({ getItem: (k: string) => (k === DACE_STORAGE_KEY ? v : null) });

describe('DACE config resolution', () => {
  it('defaults: DACE off (legacy path)', () => {
    const r = resolveDaceConfig({ env: {} });
    expect(r.config).toEqual(DEFAULT_DACE_CONFIG);
    expect(r.config.enabled).toBe(false);
    expect(r.version).toBe(DACE_VERSION);
    expect(r.layers).toEqual(['defaults']);
    expect(Object.isFrozen(r.config)).toBe(true);
    expect(Object.isFrozen(r.config.replay)).toBe(true);
  });

  it('env layer: preset first, then individual flags', () => {
    const r = resolveDaceConfig({ env: { NEXT_PUBLIC_DACE_PRESET: 'C', NEXT_PUBLIC_DACE_LOG: 'windows', NEXT_PUBLIC_DACE_EVIDENCE: 'false' } });
    expect(r.config).toMatchObject({ enabled: true, measurement: 'mediapipe', evidence: false, log: 'windows' });
  });

  it('research override needs NEXT_PUBLIC_DACE_RESEARCH=1; URL beats localStorage', () => {
    const blocked = resolveDaceConfig({ env: {}, search: '?dace=F' });
    expect(blocked.config.enabled).toBe(false);
    expect(blocked.warnings.join()).toMatch(/RESEARCH/);
    const env = { NEXT_PUBLIC_DACE_RESEARCH: '1' };
    expect(resolveDaceConfig({ env, search: '?dace=F' }).config).toMatchObject({ enabled: true, rules: true, abstain: 'rules' });
    expect(resolveDaceConfig({ env, storage: store('B') }).config.measurement).toBe('mediapipe');
    const both = resolveDaceConfig({ env, search: '?dace=A2', storage: store('B') });
    expect(both.config.measurement).toBe('pnp68');
    expect(both.layers).toContain('url');
  });

  it('explicit replay-runner object wins over everything', () => {
    const r = resolveDaceConfig({
      env: { NEXT_PUBLIC_DACE_PRESET: 'F', NEXT_PUBLIC_DACE_RESEARCH: '1' },
      search: '?dace=log=frames',
      explicit: { replay: { shuffleObjects: true }, log: 'windows' },
    });
    expect(r.config).toMatchObject({ enabled: true, log: 'windows', replay: { shuffleObjects: true } });
    expect(r.layers).toEqual(['defaults', 'env', 'url', 'explicit']);
  });

  it('invalid layers are skipped with a warning (never stop scoring)', () => {
    const r = resolveDaceConfig({ env: { NEXT_PUBLIC_DACE_MEASUREMENT: 'webgazer', NEXT_PUBLIC_DACE_ENABLED: 'true' } });
    expect(r.config.enabled).toBe(false);
    expect(r.warnings.some((w) => /invalid/.test(w))).toBe(true);
    const bad = resolveDaceConfig({ env: { NEXT_PUBLIC_DACE_RESEARCH: '1' }, search: '?dace={nope' });
    expect(bad.config.enabled).toBe(false);
    expect(resolveDaceConfig({ env: {}, explicit: { foo: 1 } as never }).warnings.join()).toMatch(/invalid/);
  });

  it("Stage 1 normalisation: conformal -> rules, kinematics forced off", () => {
    const r = resolveDaceConfig({ env: {}, explicit: { enabled: true, abstain: 'conformal', kinematics: 'on', rules: true, evidence: true } });
    expect(r.config.abstain).toBe('rules');
    expect(r.config.kinematics).toBe('off');
    expect(r.warnings.length).toBeGreaterThanOrEqual(2);
  });

  it('override string parsing', () => {
    expect(parseOverrideString('F,log=windows').layers).toHaveLength(2);
    expect(parseOverrideString('enabled=1;shuffleObjects=true').layers[0]).toEqual({ enabled: true, replay: { shuffleObjects: true } });
    expect(parseOverrideString('Z').error).toMatch(/unknown/);
    expect(parseOverrideString('{"enabled":true}').layers[0]).toEqual({ enabled: true });
  });

  it('config hash is stable, order-independent and changes with content', () => {
    const a = presetConfig('F')!;
    const b = { ...a, replay: { ...a.replay } };
    expect(configHash(a)).toBe(configHash(b));
    expect(configHash(a)).toMatch(/^[0-9a-f]{8}$/);
    expect(configHash(presetConfig('H')!)).not.toBe(configHash(a));
  });
});

describe('ablation presets (D.6, plan D12)', () => {
  it('A and A\' are DACE-off (legacy byte for byte); they differ only in the object model', () => {
    expect(presetConfig('A')).toEqual(DEFAULT_DACE_CONFIG);
    expect(presetConfig("A'")).toEqual(DEFAULT_DACE_CONFIG);
    expect(DACE_PRESETS.A.objectModel).toBe('absent');
    expect(DACE_PRESETS["A'"].objectModel).toBe('present');
  });
  it("A'' = solvePnP measurement only; B = MediaPipe only; C = B + E", () => {
    expect(presetConfig('A2')).toMatchObject({ enabled: true, measurement: 'pnp68', evidence: false, rules: false, abstain: 'off' });
    expect(presetConfig('B')).toMatchObject({ enabled: true, measurement: 'mediapipe', evidence: false, rules: false });
    expect(presetConfig('C')).toMatchObject({ measurement: 'mediapipe', evidence: true, rules: false, abstain: 'off' });
  });
  it('F (Stage 1) = C + rule T + rule U; H = F + shuffle; I = F on segment 10', () => {
    const F = presetConfig('F')!;
    expect(F).toMatchObject({ enabled: true, measurement: 'mediapipe', evidence: true, rules: true, abstain: 'rules' });
    expect(presetConfig('H')).toEqual({ ...F, replay: { shuffleObjects: true } });
    expect(presetConfig('I')).toEqual(F);
    expect(DACE_PRESETS.I.segments).toBe('desk_out_of_frame');
  });
  it('aliases', () => {
    expect(lookupPreset('a1')?.id).toBe("A'");
    expect(lookupPreset("A''")?.id).toBe("A''");
    expect(lookupPreset('nope')).toBeNull();
  });
});
