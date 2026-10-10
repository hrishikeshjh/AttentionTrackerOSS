/**
 * Rule-based abstention ("Unsure") — Annex A.5 Stage 1, FR6/FR7.
 *
 * Reason codes:
 *   U1 looking down for longer than τ_g with no phone, book or writing evidence
 *   U2 looking down with the desk not visible (V = 0) — takes precedence over U1
 *   U3 low tracking quality: face not detected in ≥ 50% of W
 *   U4 conflicting evidence: phone in use in one hand and pen-grip motion in the other
 *   U5 model uncertain (Stage 2, conformal set with more than one class)
 *
 * Scoring policy: "Unsure" holds the engagement score (score-hold.ts) and is
 * reported as a share of session time with its dominant reason; it is never
 * converted into a penalty.
 */
import type { UnsureReason } from './types';

export type AbstainMode = 'off' | 'rules' | 'conformal';

export const UNSURE_REASON_TEXT: Readonly<Record<UnsureReason, string>> = Object.freeze({
  U1: 'no phone, book or writing evidence',
  U2: 'desk not visible',
  U3: 'poor tracking',
  U4: 'conflicting evidence',
  U5: 'model uncertain',
});

/** Reason for a long downward episode without evidence (rule 6): U2 beats U1. */
export function downwardUnsureReason(deskVisible: boolean): 'U1' | 'U2' {
  return deskVisible ? 'U1' : 'U2';
}

/** Stage 1 has no calibrated model: 'conformal' degrades to 'rules' with a warning. */
export function effectiveAbstainMode(mode: AbstainMode): { mode: 'off' | 'rules'; warning: string | null } {
  if (mode === 'conformal') {
    return { mode: 'rules', warning: "dace.abstain='conformal' is a Stage 2 feature; using 'rules' (Stage 1)." };
  }
  return { mode, warning: null };
}

export type UnsureSummary = {
  total: number;
  unsureCount: number;
  /** 0-1 */
  share: number;
  counts: Partial<Record<UnsureReason, number>>;
  dominantReason: UnsureReason | null;
  /** e.g. "CognEn could not determine the activity for 12% of the session (mostly: desk not visible)" */
  text: string | null;
};

const REASON_ORDER: UnsureReason[] = ['U2', 'U1', 'U3', 'U4', 'U5'];

export function summarizeUnsure(
  points: ReadonlyArray<{ unsure?: boolean; unsureReason?: UnsureReason }>
): UnsureSummary {
  const total = points.length;
  const counts: Partial<Record<UnsureReason, number>> = {};
  let unsureCount = 0;
  for (const p of points) {
    if (!p.unsure) continue;
    unsureCount++;
    if (p.unsureReason) counts[p.unsureReason] = (counts[p.unsureReason] ?? 0) + 1;
  }
  let dominantReason: UnsureReason | null = null;
  let best = 0;
  for (const r of REASON_ORDER) {
    const c = counts[r] ?? 0;
    if (c > best) {
      best = c;
      dominantReason = r;
    }
  }
  const share = total > 0 ? unsureCount / total : 0;
  const text =
    unsureCount > 0
      ? `CognEn could not determine the activity for ${Math.round(share * 100)}% of the session` +
        (dominantReason ? ` (mostly: ${UNSURE_REASON_TEXT[dominantReason]})` : '')
      : null;
  return { total, unsureCount, share, counts, dominantReason, text };
}
