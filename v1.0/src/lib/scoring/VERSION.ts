/**
 * Bump this whenever a formula, threshold, or weight in this package changes.
 *
 * Copies of these formulas are hand-maintained in v2.0/features/face_metrics.py
 * and the Chrome extension's extension/lib/ml-core.js and must be updated in
 * the same change. They are guarded by tests, not by shared code: the
 * extension-equivalence vitest compares ml-core.js with this package for the
 * same inputs, and the generated fixtures in shared/fixtures (encode13,
 * encode15, cmcv) are asserted by both the vitest and pytest suites and
 * drift-checked in CI. The v2.0 legacy 68-point pitch formula is
 * intentionally unchanged (see the KNOWN DEFECT notes). DACE code under
 * scoring/dace/ is versioned separately (scoring/dace/VERSION.ts) and does
 * not change any formula here.
 */
export const SCORING_CORE_VERSION = '1.1.0';
