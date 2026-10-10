/**
 * STAGE 2 — conformal abstention (Annex A.5, FR11). Not wired in Stage 1:
 * dace.abstain = 'conformal' degrades to 'rules' (abstain.ts).
 *
 * Small pure helpers so the Stage 2 work has a tested starting point:
 *   - temperature-scaled softmax;
 *   - class-conditional split-conformal thresholds at error level α, with
 *     non-conformity s = 1 − p(true class), q_k = the ⌈(n_k + 1)(1 − α)⌉-th
 *     smallest score of class k (∞ if n_k is too small);
 *   - prediction set {k : 1 − p_k ≤ q_k}; one class -> label, otherwise
 *     Unsure(U5) with the competing classes.
 * Calibration must use held-out calibration SUBJECTS only (A.5 step 1).
 */
import type { BehaviorLabel } from './types';

export function softmaxT(logits: readonly number[], temperature = 1): number[] {
  const z = logits.map((v) => v / temperature);
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function calibrateClassConditional(
  calib: ReadonlyArray<{ probs: readonly number[]; label: number }>,
  alpha: number,
  nClasses: number
): number[] {
  const q: number[] = [];
  for (let k = 0; k < nClasses; k++) {
    const scores = calib.filter((c) => c.label === k).map((c) => 1 - c.probs[k]).sort((a, b) => a - b);
    const n = scores.length;
    const rank = Math.ceil((n + 1) * (1 - alpha));
    q.push(n === 0 || rank > n ? Infinity : scores[rank - 1]);
  }
  return q;
}

export function predictionSet(probs: readonly number[], q: readonly number[]): number[] {
  const out: number[] = [];
  probs.forEach((p, k) => {
    if (1 - p <= q[k]) out.push(k);
  });
  return out;
}

export type ConformalDecision =
  | { kind: 'label'; label: BehaviorLabel }
  | { kind: 'unsure'; reason: 'U5'; competing: BehaviorLabel[] };

export function decideFromSet(set: readonly number[], classes: readonly BehaviorLabel[]): ConformalDecision {
  if (set.length === 1) return { kind: 'label', label: classes[set[0]] };
  return { kind: 'unsure', reason: 'U5', competing: set.map((k) => classes[k]) };
}
