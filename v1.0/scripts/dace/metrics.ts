/**
 * Stage 1 pilot metrics (Annex D.5) for one preset.
 *
 *   npx tsx scripts/dace/metrics.ts --labels gt.jsonl \
 *       --log adarsh_setup1=adarsh_setup1__F.jsonl --log anushka_setup1=anushka_setup1__F.jsonl \
 *       [--preset F] [--out report_F.json]
 *
 * --labels: JSONL of ground-truth windows from scripts/dace/cvat-windows.ts
 *           (fields recording, person, setup, segment, start_ms, end_ms, label, ambiguous).
 * --log:    <recording>=<eval log> (repeatable). Logs come from the /replay page
 *           or `npm run dace:replay`; their t is media time, aligned with the labels.
 *
 * Prints FDR-DA (all / answered), phone recall + FNR, drowsy recall, Unsure
 * rate on desk-out-of-frame (segment 10), answer rate, Unsure-vs-ambiguous
 * agreement, macro-F1 (answered) and latency, overall / per person / per
 * segment. Descriptive only (pilot; no significance tests).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs, die } from './args';
import { parseJsonl } from '../../src/lib/dace/eval-log';
import type { GtWindow } from '../../src/lib/dace/eval/cvat';
import { metricsReport, windowRecords, type MetricBlock, type PredRecord } from '../../src/lib/dace/eval/metrics';

const OFFSET = 1e10; // separates recordings on one time axis

function main() {
  const args = parseArgs(process.argv.slice(2));
  const labelsPath = args.get('labels');
  const logs = args.all('log');
  if (!labelsPath || !logs.length) die('usage: metrics.ts --labels gt.jsonl --log <recording>=<log.jsonl> [--log ...] [--preset F] [--out report.json]');
  const gtAll = parseJsonl<GtWindow>(readFileSync(labelsPath, 'utf8'));

  const gt: GtWindow[] = [];
  const preds: PredRecord[] = [];
  logs.forEach((spec, k) => {
    const eq = spec.indexOf('=');
    if (eq < 1) die(`--log expects <recording>=<path>, got ${spec}`);
    const rec = spec.slice(0, eq);
    const path = spec.slice(eq + 1);
    const g = gtAll.filter((w) => w.recording === rec);
    if (!g.length) console.warn(`warning: no labelled windows for recording ${rec}`);
    const p = windowRecords(parseJsonl<Record<string, unknown>>(readFileSync(path, 'utf8')));
    if (!p.length) console.warn(`warning: ${path} has no 'window' records`);
    gt.push(...g.map((w) => ({ ...w, start_ms: w.start_ms + k * OFFSET, end_ms: w.end_ms + k * OFFSET })));
    preds.push(...p.map((r) => ({ ...r, t: r.t + k * OFFSET })));
  });
  preds.sort((a, b) => a.t - b.t);

  const report = metricsReport(gt, preds);
  const preset = args.get('preset') ?? 'unknown';
  const pct = (v: number | null) => (v === null ? '   n/a' : `${(100 * v).toFixed(1).padStart(5)}%`);
  const row = (name: string, m: MetricBlock) =>
    `${name.padEnd(14)} n=${String(m.windows).padStart(4)}  FDR-DA ${pct(m.fdrDa)} (ans ${pct(m.fdrDaAnswered)})  ` +
    `phoneR ${pct(m.phoneRecall)}  drowsyR ${pct(m.drowsyRecall)}  unsure@desk-out ${pct(m.unsureRateDeskOut)}  answer ${pct(m.answerRate)}  missing ${m.missing}`;
  console.log(`preset ${preset}`);
  console.log(row('overall', report.overall));
  for (const [k, m] of Object.entries(report.byPerson)) console.log(row(`person ${k}`, m));
  for (const [k, m] of Object.entries(report.bySegment)) console.log(row(`segment ${k}`, m));
  if (report.latency) {
    const l = report.latency;
    console.log(`latency total: n=${l.n} p50=${l.p50.toFixed(1)} ms p95=${l.p95.toFixed(1)} ms max=${l.max.toFixed(1)} ms (target p95 <= 200 ms)`);
  }
  const g = report.overall;
  if (g.phoneRecall !== null && g.phoneRecall < 0.8) console.log('GUARD FAILED: phone recall < 0.80');
  const out = args.get('out');
  if (out) {
    writeFileSync(out, JSON.stringify({ preset, labels: labelsPath, logs, ...report }, null, 2) + '\n');
    console.log(`-> ${out}`);
  }
}

main();
