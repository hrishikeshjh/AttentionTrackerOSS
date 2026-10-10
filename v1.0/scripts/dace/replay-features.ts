/**
 * Offline E/T/U re-run from a DACE frame log (FR8).
 *
 *   npm run dace:replay -- --frames rec1__F.jsonl --preset F --preset H --preset C --out out/
 *   npm run dace:replay -- --frames rec1__F.jsonl --preset F --params '{"rhoUse":0.5}'
 *
 * Input: an eval log recorded at log=frames (dashboard with ?dace=F,log=frames
 * in a research build, or the /replay page with "frames"). Each frame record
 * holds the measurement-layer output (pose, gaze, openness) plus the hands
 * and detector boxes BEFORE the rule runner, so the desk-evidence, rule and
 * abstention layers can be re-run here under any preset or parameter set
 * without the browser (threshold tuning on P1, controls H/I).
 *
 * The measurement layer M is NOT re-run (no video here): presets that change
 * M (A, A', A'') are rejected. Record frame logs with evidence on (C or F),
 * otherwise the detector/hands never ran and there is nothing to re-use.
 *
 * Output: one JSONL file per preset with 'window' records (same shape as the
 * browser log), stamped with DACE_VERSION, the preset's config hash and the
 * source session id.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { parseArgs, die } from './args';
import { parseJsonl, EvalLogger } from '../../src/lib/dace/eval-log';
import { resolveDaceConfig } from '../../src/lib/dace/config';
import { DACE_PRESETS, lookupPreset } from '../../src/lib/dace/presets';
import { RuleRunner } from '../../src/lib/dace/rule-runner';
import { makeParams, type DaceParams } from '../../src/lib/scoring/dace/params';
import type { DaceFrame } from '../../src/lib/scoring/dace/types';

type FrameRecord = { type: 'frame'; t: number; i: number; sid: string; objectsFresh?: boolean; frame: DaceFrame; v: string };

function main() {
  const args = parseArgs(process.argv.slice(2));
  const framesPath = args.get('frames') ?? args.positional[0];
  if (!framesPath) die('usage: dace:replay --frames <log.jsonl> --preset F [--preset H ...] [--params JSON] [--out dir]');
  const presets = args.all('preset').length ? args.all('preset') : ['F'];
  const outDir = args.get('out') ?? '.';
  let paramOverrides: Partial<DaceParams> = {};
  if (args.get('params')) {
    try {
      paramOverrides = JSON.parse(args.get('params')!);
    } catch {
      die('--params must be a JSON object');
    }
  }
  const params = makeParams(paramOverrides);

  const records = parseJsonl<Record<string, unknown>>(readFileSync(framesPath, 'utf8'));
  const frames = records.filter((r): r is FrameRecord => r.type === 'frame' && typeof r.t === 'number' && !!r.frame);
  if (!frames.length) die(`${framesPath}: no 'frame' records (record with log=frames)`);
  frames.sort((a, b) => a.t - b.t);
  const sid = String(frames[0].sid ?? 'unknown');
  mkdirSync(outDir, { recursive: true });

  for (const name of presets) {
    const preset = lookupPreset(name);
    if (!preset) die(`unknown preset ${name} (have: ${Object.keys(DACE_PRESETS).join(', ')})`);
    const o = preset.override;
    if (!o.enabled || o.measurement !== 'mediapipe') {
      console.warn(`skip ${preset.id}: it changes the measurement layer, which cannot be re-run offline`);
      continue;
    }
    const resolved = resolveDaceConfig({ env: {}, explicit: { ...o, log: 'windows' } });
    const cfg = resolved.config;
    const runner = new RuleRunner(params, {
      evidence: cfg.evidence,
      rules: cfg.rules,
      abstain: cfg.abstain === 'off' ? 'off' : 'rules',
      shuffleObjects: cfg.replay.shuffleObjects,
    });
    const logger = new EvalLogger({ level: 'windows', configHash: resolved.hash, sessionId: sid });
    logger.log('session', 0, { preset: preset.id, source: 'offline_replay', input: basename(framesPath), config: cfg, params });
    for (const fr of frames) {
      const step = runner.push(fr.frame, fr.objectsFresh ?? true);
      const s = step.summary;
      logger.log('window', fr.t, {
        i: fr.i,
        label: step.effective.label,
        unsure: step.effective.unsure,
        reason: step.effective.reason,
        rule: step.effective.rule,
        keptFrom: step.output?.keptFrom ?? null,
        face: fr.frame.face !== null,
        lookingDown: step.evidence.lookingDown,
        perclos: step.perclos,
        longestClosureMs: step.longestClosureMs,
        summary: {
          noFaceCount: s.noFaceCount,
          lookingDownCount: s.lookingDownCount,
          phoneInUse: s.phoneInUse,
          phonePresent: s.phonePresent,
          book: s.book,
          writing: s.writing,
          conflict: s.conflict,
          deskVisible: s.deskVisible,
          downEpisodeMs: s.downEpisodeMs,
        },
      });
    }
    const out = join(outDir, `${basename(framesPath).replace(/\.jsonl$/, '')}__offline_${preset.id.replace(/'/g, 'p')}.jsonl`);
    writeFileSync(out, logger.toJSONL());
    const n = logger.getRecords().filter((r) => r.type === 'window');
    const unsure = n.filter((r) => r.unsure).length;
    console.log(`${preset.id}: ${n.length} windows, unsure ${((100 * unsure) / Math.max(1, n.length)).toFixed(1)}% -> ${out}`);
    if (resolved.warnings.length) console.log(`  warnings: ${resolved.warnings.join(' | ')}`);
  }
}

main();
