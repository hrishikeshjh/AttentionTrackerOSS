/**
 * DACE (Downward-Attention Context Engine) version.
 *
 * Independent of SCORING_CORE_VERSION: DACE adds new code under
 * scoring/dace/ and does not change any existing formula, so the legacy
 * scoring core stays at its own version. Bump this when any DACE formula,
 * rule, rule order or parameter default changes. It is stamped (with the
 * config hash) on every DACE evaluation-log record.
 */
export const DACE_VERSION = '0.1.0';
