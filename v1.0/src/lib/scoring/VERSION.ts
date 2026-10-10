/**
 * Bump this whenever a formula, threshold, or weight in this package changes.
 *
 * There is no automated cross-language check: v2.0/features/face_metrics.py
 * and the Chrome extension's extension/lib/ml-core.js are hand-maintained
 * copies of these formulas and must be updated manually in the same change
 * (the extension copy is currently known to diverge). DACE code under
 * scoring/dace/ is versioned separately (scoring/dace/VERSION.ts) and does
 * not change any formula here.
 */
export const SCORING_CORE_VERSION = '1.1.0';
