/**
 * DACE configuration (plan section 4, FR8).
 *
 * Resolution order (later layers override earlier ones):
 *   1. code defaults (DACE off: the legacy path runs byte for byte)
 *   2. NEXT_PUBLIC_DACE_* environment variables (NEXT_PUBLIC_DACE_PRESET first,
 *      then the individual flags)
 *   3. research-only override from `?dace=` or localStorage['cognen.dace'] —
 *      honoured ONLY when NEXT_PUBLIC_DACE_RESEARCH=1 (URL wins over storage)
 *   4. an explicit object from the replay runner
 *
 * Every layer is validated with zod; an invalid layer is skipped with a
 * warning (DACE must never stop CognEn from producing a score). The resolved
 * config is deep-frozen at session start; its hash and DACE_VERSION are
 * stamped on every evaluation-log record.
 *
 * Override string syntax (env PRESET, ?dace=, localStorage):
 *   - a preset name: A | A' (A1) | A'' (A2) | B | C | F | H | I
 *   - key=value pairs separated by ',' or ';', optionally starting with a
 *     preset: "F,log=windows" or "enabled=true,measurement=mediapipe"
 *   - a JSON object: {"enabled":true,"replay":{"shuffleObjects":true}}
 */
import { z } from 'zod';
import { DACE_VERSION } from '../scoring/dace/VERSION';
import { effectiveAbstainMode } from '../scoring/dace/abstain';
import { lookupPreset } from './presets';

export const DaceConfigSchema = z
  .object({
    enabled: z.boolean(),
    measurement: z.enum(['legacy', 'pnp68', 'mediapipe']),
    evidence: z.boolean(),
    rules: z.boolean(),
    abstain: z.enum(['off', 'rules', 'conformal']),
    detector: z.enum(['yolo', 'mediapipe']),
    kinematics: z.enum(['off', 'on']),
    temporalModel: z.enum(['rules', 'onnx']),
    baseline: z.enum(['internal', 'peb']),
    log: z.enum(['off', 'windows', 'frames']),
    replay: z.object({ shuffleObjects: z.boolean() }).strict(),
  })
  .strict();

export type DaceConfig = z.infer<typeof DaceConfigSchema>;

const OverrideSchema = DaceConfigSchema.deepPartial().strict();
export type DaceConfigOverride = z.infer<typeof OverrideSchema>;

export const DEFAULT_DACE_CONFIG: Readonly<DaceConfig> = deepFreeze({
  enabled: false,
  measurement: 'legacy',
  evidence: false,
  rules: false,
  abstain: 'off',
  detector: 'yolo',
  kinematics: 'off',
  temporalModel: 'rules',
  baseline: 'internal',
  log: 'off',
  replay: { shuffleObjects: false },
});

export const DACE_STORAGE_KEY = 'cognen.dace';

function deepFreeze<T>(o: T): T {
  if (o && typeof o === 'object') {
    Object.values(o as Record<string, unknown>).forEach(deepFreeze);
    Object.freeze(o);
  }
  return o;
}

function merge(base: DaceConfig, o: DaceConfigOverride): DaceConfig {
  const { replay, ...rest } = o;
  const clean = Object.fromEntries(Object.entries(rest).filter(([, v]) => v !== undefined));
  return {
    ...base,
    ...clean,
    replay: { ...base.replay, ...(replay ?? {}) },
  } as DaceConfig;
}

export function presetConfig(name: string): DaceConfig | null {
  const p = lookupPreset(name);
  return p ? merge(structuredCloneConfig(DEFAULT_DACE_CONFIG), p.override) : null;
}

function structuredCloneConfig(c: Readonly<DaceConfig>): DaceConfig {
  return { ...c, replay: { ...c.replay } };
}

const KEY_ALIASES: Record<string, string> = {
  shuffleobjects: 'replay.shuffleObjects',
  'replay.shuffleobjects': 'replay.shuffleObjects',
  temporalmodel: 'temporalModel',
};

function parseScalar(v: string): unknown {
  const s = v.trim();
  if (/^(true|1|yes|on)$/i.test(s)) return true;
  if (/^(false|0|no)$/i.test(s)) return false;
  return s;
}

const BOOL_KEYS = new Set(['enabled', 'evidence', 'rules', 'replay.shuffleObjects']);

/**
 * Parse an override string into a list of layers (a preset layer, then a
 * key/value layer). Returns null + error on malformed input.
 */
export function parseOverrideString(raw: string): { layers: unknown[]; error: string | null } {
  const s = raw.trim();
  if (!s) return { layers: [], error: null };
  if (s.startsWith('{')) {
    try {
      return { layers: [JSON.parse(s)], error: null };
    } catch {
      return { layers: [], error: `invalid JSON override: ${s}` };
    }
  }
  const layers: unknown[] = [];
  const kv: Record<string, unknown> = {};
  const parts = s.split(/[,;]/).map((p) => p.trim()).filter(Boolean);
  for (const part of parts) {
    const eq = part.indexOf('=');
    if (eq < 0) {
      const p = lookupPreset(part);
      if (!p) return { layers: [], error: `unknown DACE preset: ${part}` };
      layers.push(p.override);
      continue;
    }
    let key = part.slice(0, eq).trim();
    key = KEY_ALIASES[key.toLowerCase()] ?? key;
    const rawVal = part.slice(eq + 1);
    const val = BOOL_KEYS.has(key) ? parseScalar(rawVal) : rawVal.trim();
    if (key === 'replay.shuffleObjects') kv.replay = { shuffleObjects: val };
    else kv[key] = val;
  }
  if (Object.keys(kv).length) layers.push(kv);
  return { layers, error: null };
}

export type EnvLike = Record<string, string | undefined>;

/**
 * Read NEXT_PUBLIC_DACE_* with literal property accesses (Next.js only
 * inlines `process.env.NEXT_PUBLIC_X` written out in full).
 */
export function readDaceEnv(): EnvLike {
  if (typeof process === 'undefined' || !process.env) return {};
  return {
    NEXT_PUBLIC_DACE_PRESET: process.env.NEXT_PUBLIC_DACE_PRESET,
    NEXT_PUBLIC_DACE_ENABLED: process.env.NEXT_PUBLIC_DACE_ENABLED,
    NEXT_PUBLIC_DACE_MEASUREMENT: process.env.NEXT_PUBLIC_DACE_MEASUREMENT,
    NEXT_PUBLIC_DACE_EVIDENCE: process.env.NEXT_PUBLIC_DACE_EVIDENCE,
    NEXT_PUBLIC_DACE_RULES: process.env.NEXT_PUBLIC_DACE_RULES,
    NEXT_PUBLIC_DACE_ABSTAIN: process.env.NEXT_PUBLIC_DACE_ABSTAIN,
    NEXT_PUBLIC_DACE_DETECTOR: process.env.NEXT_PUBLIC_DACE_DETECTOR,
    NEXT_PUBLIC_DACE_KINEMATICS: process.env.NEXT_PUBLIC_DACE_KINEMATICS,
    NEXT_PUBLIC_DACE_TEMPORAL_MODEL: process.env.NEXT_PUBLIC_DACE_TEMPORAL_MODEL,
    NEXT_PUBLIC_DACE_BASELINE: process.env.NEXT_PUBLIC_DACE_BASELINE,
    NEXT_PUBLIC_DACE_LOG: process.env.NEXT_PUBLIC_DACE_LOG,
    NEXT_PUBLIC_DACE_SHUFFLE_OBJECTS: process.env.NEXT_PUBLIC_DACE_SHUFFLE_OBJECTS,
    NEXT_PUBLIC_DACE_RESEARCH: process.env.NEXT_PUBLIC_DACE_RESEARCH,
  };
}

const ENV_KEYS: Array<[string, string]> = [
  ['NEXT_PUBLIC_DACE_ENABLED', 'enabled'],
  ['NEXT_PUBLIC_DACE_MEASUREMENT', 'measurement'],
  ['NEXT_PUBLIC_DACE_EVIDENCE', 'evidence'],
  ['NEXT_PUBLIC_DACE_RULES', 'rules'],
  ['NEXT_PUBLIC_DACE_ABSTAIN', 'abstain'],
  ['NEXT_PUBLIC_DACE_DETECTOR', 'detector'],
  ['NEXT_PUBLIC_DACE_KINEMATICS', 'kinematics'],
  ['NEXT_PUBLIC_DACE_TEMPORAL_MODEL', 'temporalModel'],
  ['NEXT_PUBLIC_DACE_BASELINE', 'baseline'],
  ['NEXT_PUBLIC_DACE_LOG', 'log'],
  ['NEXT_PUBLIC_DACE_SHUFFLE_OBJECTS', 'replay.shuffleObjects'],
];

function envLayers(env: EnvLike, warnings: string[]): unknown[] {
  const layers: unknown[] = [];
  const preset = env.NEXT_PUBLIC_DACE_PRESET;
  if (preset && preset.trim()) {
    const p = lookupPreset(preset);
    if (p) layers.push(p.override);
    else warnings.push(`NEXT_PUBLIC_DACE_PRESET: unknown preset "${preset}" ignored`);
  }
  const kv: Record<string, unknown> = {};
  for (const [envKey, key] of ENV_KEYS) {
    const v = env[envKey];
    if (v === undefined || v.trim() === '') continue;
    const val = BOOL_KEYS.has(key) ? parseScalar(v) : v.trim();
    if (key === 'replay.shuffleObjects') kv.replay = { shuffleObjects: val };
    else kv[key] = val;
  }
  if (Object.keys(kv).length) layers.push(kv);
  return layers;
}

export type StorageLike = { getItem(key: string): string | null };

export type DaceConfigSources = {
  /** defaults to readDaceEnv() */
  env?: EnvLike;
  /** e.g. window.location.search */
  search?: string | null;
  storage?: StorageLike | null;
  /** replay runner */
  explicit?: DaceConfigOverride | null;
};

export type ResolvedDaceConfig = {
  config: Readonly<DaceConfig>;
  hash: string;
  version: string;
  warnings: string[];
  /** names of the layers that were applied, in order */
  layers: string[];
};

/** FNV-1a 32-bit over a canonical (sorted-key) JSON encoding. */
export function configHash(config: DaceConfig): string {
  const canon = (v: unknown): string => {
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      return `{${Object.keys(v as object)
        .sort()
        .map((k) => `${JSON.stringify(k)}:${canon((v as Record<string, unknown>)[k])}`)
        .join(',')}}`;
    }
    return JSON.stringify(v);
  };
  const s = canon(config);
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

function researchOverride(sources: DaceConfigSources, warnings: string[]): { raw: string; from: string } | null {
  let fromUrl: string | null = null;
  if (sources.search) {
    try {
      fromUrl = new URLSearchParams(sources.search).get('dace');
    } catch {
      warnings.push('could not parse URL search string');
    }
  }
  if (fromUrl) return { raw: fromUrl, from: 'url' };
  if (sources.storage) {
    try {
      const v = sources.storage.getItem(DACE_STORAGE_KEY);
      if (v) return { raw: v, from: 'localStorage' };
    } catch {
      warnings.push('localStorage unavailable');
    }
  }
  return null;
}

export function resolveDaceConfig(sources: DaceConfigSources = {}): ResolvedDaceConfig {
  const warnings: string[] = [];
  const applied: string[] = ['defaults'];
  let config = structuredCloneConfig(DEFAULT_DACE_CONFIG);

  const apply = (layer: unknown, name: string) => {
    const r = OverrideSchema.safeParse(layer);
    if (!r.success) {
      warnings.push(`${name}: invalid DACE config ignored (${r.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; ')})`);
      return;
    }
    config = merge(config, r.data);
    applied.push(name);
  };

  const env = sources.env ?? readDaceEnv();
  envLayers(env, warnings).forEach((l) => apply(l, 'env'));

  const research = researchOverride(sources, warnings);
  if (research) {
    if (env.NEXT_PUBLIC_DACE_RESEARCH === '1') {
      const parsed = parseOverrideString(research.raw);
      if (parsed.error) warnings.push(`${research.from}: ${parsed.error}`);
      else parsed.layers.forEach((l) => apply(l, research.from));
    } else {
      warnings.push(`${research.from} DACE override ignored (NEXT_PUBLIC_DACE_RESEARCH is not 1)`);
    }
  }

  if (sources.explicit) apply(sources.explicit, 'explicit');

  // Stage 1 normalisation.
  const ab = effectiveAbstainMode(config.abstain);
  if (ab.warning) {
    warnings.push(ab.warning);
    config.abstain = ab.mode;
  }
  if (config.kinematics === 'on') {
    warnings.push("dace.kinematics is a Stage 2 feasibility study; forced 'off' in Stage 1.");
    config.kinematics = 'off';
  }
  if (config.enabled) {
    if (config.temporalModel === 'onnx') {
      warnings.push("dace.temporalModel='onnx' is Stage 2 (loadModel contract fix pending, D10); the rule layer is the fallback.");
    }
    if (config.baseline === 'peb') warnings.push('dace.baseline=peb: PEB hooks are interface-only in Stage 1.');
    if (config.abstain !== 'off' && !config.rules) warnings.push('dace.abstain has no effect while dace.rules is false.');
    if (config.rules && !config.evidence) warnings.push('dace.rules without dace.evidence: no phone/book/writing evidence, long downward episodes will be Unsure.');
    if (config.measurement === 'legacy' && (config.evidence || config.rules)) {
      warnings.push('dace.measurement=legacy: legacy pitch sign/gaze are uncorrected; rules will see uncorrected looking-down signals.');
    }
  }

  const frozen = deepFreeze(DaceConfigSchema.parse(config));
  return { config: frozen, hash: configHash(frozen), version: DACE_VERSION, warnings, layers: applied };
}
