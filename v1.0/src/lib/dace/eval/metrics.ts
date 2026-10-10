/**
 * Stage 1 pilot metrics (Annex D.5) from replay eval logs + CVAT windows.
 *
 *   FDR-DA          share of ground-truth attentive_reading windows labelled
 *                   distracted, drowsy or using_phone (primary). Reported over
 *                   all reading windows (Unsure counts as "not a false
 *                   distraction") and over answered windows only.
 *   phone recall    share of using_phone windows labelled using_phone (Unsure
 *                   and missing count as misses; guard >= 0.80); FNR = 1 - recall
 *   drowsy recall   same for drowsy (guard)
 *   unsure rate on desk-out-of-frame windows (segment 10; target >= 0.90)
 *   answer rate     share of windows that received a label (not Unsure)
 *   unsure/ambiguous agreement: share of Unsure windows that annotators
 *                   flagged ambiguous, and vice versa
 *   latency         p50/p95/max of per-tick total processing from the log
 *
 * Alignment: the prediction for a ground-truth window [start, end) is the
 * LAST window record with start <= t < end (a DACE window output summarises
 * the trailing 3 s). Windows without a record are counted as missing and
 * excluded from the denominators (reported separately).
 * All numbers are descriptive (pilot, no significance tests).
 */
import type { BehaviorLabel } from '../../types';
import type { GtWindow } from './cvat';
import { DESK_OUT_OF_FRAME_SEGMENT } from './cvat';
import { percentile } from '../latency';

export type PredRecord = {
  t: number;
  label: BehaviorLabel | null;
  unsure: boolean;
  reason?: string | null;
  latencyTotal?: number | null;
};

export type Pred = 'unsure' | 'none' | BehaviorLabel;

/** Extract window records (type 'window') from parsed eval-log records. */
export function windowRecords(records: ReadonlyArray<Record<string, unknown>>): PredRecord[] {
  return records
    .filter((r) => r.type === 'window' && typeof r.t === 'number')
    .map((r) => ({
      t: r.t as number,
      label: (r.label ?? null) as BehaviorLabel | null,
      unsure: r.unsure === true,
      reason: (r.reason ?? null) as string | null,
      latencyTotal: (r.latency as { total?: number } | undefined)?.total ?? null,
    }))
    .sort((a, b) => a.t - b.t);
}

export function predictionFor(win: Pick<GtWindow, 'start_ms' | 'end_ms'>, preds: readonly PredRecord[]): Pred | null {
  let last: PredRecord | null = null;
  for (const p of preds) {
    if (p.t >= win.start_ms && p.t < win.end_ms) last = p;
    if (p.t >= win.end_ms) break;
  }
  if (!last) return null;
  if (last.unsure) return 'unsure';
  return last.label ?? 'none';
}

const FALSE_DISTRACTION: ReadonlySet<Pred> = new Set<Pred>(['distracted', 'drowsy', 'using_phone']);
const answered = (p: Pred) => p !== 'unsure' && p !== 'none';

export type MetricBlock = {
  windows: number;
  missing: number;
  answerRate: number | null;
  fdrDa: number | null;
  fdrDaAnswered: number | null;
  readingWindows: number;
  phoneRecall: number | null;
  phoneFnr: number | null;
  phoneWindows: number;
  drowsyRecall: number | null;
  drowsyWindows: number;
  unsureRateDeskOut: number | null;
  deskOutWindows: number;
  unsureAmbiguousPrecision: number | null;
  unsureAmbiguousRecall: number | null;
  macroF1Answered: number | null;
};

const ratio = (a: number, b: number) => (b > 0 ? a / b : null);

export function computeMetrics(gt: readonly GtWindow[], preds: readonly PredRecord[]): MetricBlock {
  const pairs: Array<{ g: GtWindow; p: Pred }> = [];
  let missing = 0;
  for (const g of gt) {
    const p = predictionFor(g, preds);
    if (p === null) missing++;
    else pairs.push({ g, p });
  }
  const reading = pairs.filter((x) => x.g.label === 'attentive_reading');
  const readingAnswered = reading.filter((x) => answered(x.p));
  const fd = reading.filter((x) => FALSE_DISTRACTION.has(x.p)).length;
  const phone = pairs.filter((x) => x.g.label === 'using_phone');
  const drowsy = pairs.filter((x) => x.g.label === 'drowsy');
  const deskOut = pairs.filter((x) => x.g.segment === DESK_OUT_OF_FRAME_SEGMENT);
  const unsure = pairs.filter((x) => x.p === 'unsure');
  const ambiguous = pairs.filter((x) => x.g.ambiguous);

  // macro-F1 over the six classes on answered windows
  const ans = pairs.filter((x) => answered(x.p));
  const classes = Array.from(new Set(pairs.map((x) => x.g.label)));
  const f1s = classes.map((c) => {
    const tp = ans.filter((x) => x.g.label === c && x.p === c).length;
    const fp = ans.filter((x) => x.g.label !== c && x.p === c).length;
    const fn = ans.filter((x) => x.g.label === c && x.p !== c).length;
    return tp + fp + fn > 0 ? (2 * tp) / (2 * tp + fp + fn) : 0;
  });

  const phoneHit = phone.filter((x) => x.p === 'using_phone').length;
  return {
    windows: gt.length,
    missing,
    answerRate: ratio(ans.length, pairs.length),
    fdrDa: ratio(fd, reading.length),
    fdrDaAnswered: ratio(fd, readingAnswered.length),
    readingWindows: reading.length,
    phoneRecall: ratio(phoneHit, phone.length),
    phoneFnr: phone.length ? 1 - phoneHit / phone.length : null,
    phoneWindows: phone.length,
    drowsyRecall: ratio(drowsy.filter((x) => x.p === 'drowsy').length, drowsy.length),
    drowsyWindows: drowsy.length,
    unsureRateDeskOut: ratio(deskOut.filter((x) => x.p === 'unsure').length, deskOut.length),
    deskOutWindows: deskOut.length,
    unsureAmbiguousPrecision: ratio(unsure.filter((x) => x.g.ambiguous).length, unsure.length),
    unsureAmbiguousRecall: ratio(ambiguous.filter((x) => x.p === 'unsure').length, ambiguous.length),
    macroF1Answered: f1s.length ? f1s.reduce((a, b) => a + b, 0) / f1s.length : null,
  };
}

export function latencyStats(preds: readonly PredRecord[]): { n: number; p50: number; p95: number; max: number } | null {
  const v = preds.map((p) => p.latencyTotal).filter((x): x is number => typeof x === 'number').sort((a, b) => a - b);
  if (!v.length) return null;
  return { n: v.length, p50: percentile(v, 50), p95: percentile(v, 95), max: v[v.length - 1] };
}

/** Metrics overall, per person and per segment (Stage 1 reporting, D.5). */
export function metricsReport(gt: readonly GtWindow[], preds: readonly PredRecord[]) {
  const group = (key: (g: GtWindow) => string) => {
    const out: Record<string, MetricBlock> = {};
    for (const k of Array.from(new Set(gt.map(key))).sort()) out[k] = computeMetrics(gt.filter((g) => key(g) === k), preds);
    return out;
  };
  return {
    overall: computeMetrics(gt, preds),
    byPerson: group((g) => g.person),
    bySegment: group((g) => g.segment),
    latency: latencyStats(preds),
  };
}
