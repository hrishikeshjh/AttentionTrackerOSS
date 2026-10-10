/**
 * Stage 1 ablation presets (Annex D.6, plan section 4 / D12).
 *
 * Each preset is an override on top of DEFAULT_DACE_CONFIG (config.ts).
 * `objectModel` records whether the legacy object-detector model must be
 * installed for the run (A vs A'); it is an environment requirement, not a
 * DACE flag. Stage 1 "F" = C + rule T + rule U (plan D12).
 */
import type { DaceConfigOverride } from './config';

export type DacePresetId = 'A' | "A'" | "A''" | 'B' | 'C' | 'F' | 'H' | 'I';

export type DacePreset = {
  id: DacePresetId;
  title: string;
  question: string;
  objectModel: 'absent' | 'present';
  /** which recordings the preset is evaluated on */
  segments: 'all' | 'desk_out_of_frame';
  override: DaceConfigOverride;
};

const F_OVERRIDE: DaceConfigOverride = {
  enabled: true,
  measurement: 'mediapipe',
  evidence: true,
  rules: true,
  abstain: 'rules',
};

export const DACE_PRESETS: Readonly<Record<DacePresetId, DacePreset>> = Object.freeze({
  A: {
    id: 'A',
    title: 'CognEn as shipped',
    question: 'Reference, as users experience it',
    objectModel: 'absent',
    segments: 'all',
    override: { enabled: false },
  },
  "A'": {
    id: "A'",
    title: 'A with object detector installed',
    question: 'Fair baseline',
    objectModel: 'present',
    segments: 'all',
    override: { enabled: false },
  },
  "A''": {
    id: "A''",
    title: "A' with solvePnP head pose",
    question: 'Does a minimal fix suffice?',
    objectModel: 'present',
    segments: 'all',
    override: { enabled: true, measurement: 'pnp68', evidence: false, rules: false, abstain: 'off' },
  },
  B: {
    id: 'B',
    title: "A' + measurement layer M",
    question: 'RQ1',
    objectModel: 'present',
    segments: 'all',
    override: { enabled: true, measurement: 'mediapipe', evidence: false, rules: false, abstain: 'off' },
  },
  C: {
    id: 'C',
    title: 'B + desk-zone evidence E',
    question: 'RQ2',
    objectModel: 'present',
    segments: 'all',
    override: { enabled: true, measurement: 'mediapipe', evidence: true, rules: false, abstain: 'off' },
  },
  F: {
    id: 'F',
    title: 'C + rule-based temporal fusion T + rule-based abstention U (Stage 1)',
    question: 'RQ3 (pilot): does abstention reduce harmful errors?',
    objectModel: 'present',
    segments: 'all',
    override: F_OVERRIDE,
  },
  H: {
    id: 'H',
    title: 'F with object labels shuffled across windows (negative control)',
    question: 'Is the gain really from context?',
    objectModel: 'present',
    segments: 'all',
    override: { ...F_OVERRIDE, replay: { shuffleObjects: true } },
  },
  I: {
    id: 'I',
    title: 'F on desk-out-of-frame segments (stress test)',
    question: 'Does it abstain instead of guessing?',
    objectModel: 'present',
    segments: 'desk_out_of_frame',
    override: F_OVERRIDE,
  },
});

/** URL/env-safe aliases: A1 = A', A2 = A''. */
export const PRESET_ALIASES: Readonly<Record<string, DacePresetId>> = Object.freeze({
  A: 'A',
  "A'": "A'",
  A1: "A'",
  APRIME: "A'",
  "A''": "A''",
  A2: "A''",
  ADOUBLEPRIME: "A''",
  B: 'B',
  C: 'C',
  F: 'F',
  H: 'H',
  I: 'I',
});

export function lookupPreset(name: string): DacePreset | null {
  const id = PRESET_ALIASES[name.trim().toUpperCase()] ?? PRESET_ALIASES[name.trim()];
  return id ? DACE_PRESETS[id] : null;
}
