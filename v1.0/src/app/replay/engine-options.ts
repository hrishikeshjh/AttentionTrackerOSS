/**
 * MLEngine options for a /replay preset.
 *
 * Presets A and A' (DACE off) are the ablation baselines "CognEn as shipped",
 * so they must run the frozen as-shipped engine: `new MLEngine()` with the
 * default options (historical contribution lookup, CMCV taxonomy 'v1'), the
 * same configuration the legacy golden fixture records. Presets that enable
 * DACE (A'' onward) run with the app's APP_ENGINE_OPTIONS. Neither choice
 * changes rawScore / attentionState / confidence; only the explanatory
 * outputs (featureContributions, contradictionCategory) differ.
 *
 * scripts/dace/replay-features.ts re-runs only the DACE rule/abstention
 * layers offline and never constructs an MLEngine, so it needs no selection.
 */
import { DACE_PRESETS, type DacePresetId } from '@/lib/dace/presets';
import { APP_ENGINE_OPTIONS, type MLEngineOptions } from '@/lib/ml-engine';

export function engineOptionsForPreset(preset: DacePresetId): MLEngineOptions {
  return DACE_PRESETS[preset].override.enabled === true ? APP_ENGINE_OPTIONS : {};
}
