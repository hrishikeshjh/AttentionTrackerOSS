import { describe, it, expect } from 'vitest';
import { engineOptionsForPreset } from '../engine-options';
import { DACE_PRESETS, type DacePresetId } from '@/lib/dace/presets';
import { APP_ENGINE_OPTIONS } from '@/lib/ml-engine';

describe('replay engine options per preset', () => {
  it("A and A' (as shipped) use the default MLEngine options", () => {
    expect(engineOptionsForPreset('A')).toEqual({});
    expect(engineOptionsForPreset("A'")).toEqual({});
  });

  it("every DACE-enabled preset (A'' onward) uses APP_ENGINE_OPTIONS", () => {
    const dace = (Object.keys(DACE_PRESETS) as DacePresetId[]).filter((id) => DACE_PRESETS[id].override.enabled === true);
    expect(dace).toEqual(expect.arrayContaining(["A''", 'B', 'C', 'F', 'H', 'I']));
    for (const id of dace) expect(engineOptionsForPreset(id)).toBe(APP_ENGINE_OPTIONS);
  });

  it('scripts/dace/replay-features.ts never builds an MLEngine', async () => {
    const { readFileSync } = await import('node:fs');
    const { fileURLToPath } = await import('node:url');
    const src = readFileSync(fileURLToPath(new URL('../../../../scripts/dace/replay-features.ts', import.meta.url)), 'utf8');
    expect(src).not.toMatch(/MLEngine/);
  });
});
