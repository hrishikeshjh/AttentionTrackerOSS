/**
 * CVAT (for video 1.1 XML, frame tags) -> 3-second ground-truth windows.
 *
 *   npx tsx scripts/dace/cvat-windows.ts --xml annotations.xml --fps 30 \
 *       --recording adarsh_setup1 --person adarsh --setup setup1 [--append gt.jsonl | --out gt.jsonl]
 *
 * Tag each labelled frame with one of the six classes (aliases such as
 * "reading", "writing", "phone" are accepted), and add attributes
 * "ambiguous" (true/false) and "segment" (1-10; 10 = desk out of frame).
 * See src/lib/dace/eval/cvat.ts for the window rule.
 */
import { appendFileSync, readFileSync, writeFileSync } from 'node:fs';
import { parseArgs, die } from './args';
import { parseCvatTags, tagsToWindows } from '../../src/lib/dace/eval/cvat';

function main() {
  const a = parseArgs(process.argv.slice(2));
  const xml = a.get('xml');
  const fps = Number(a.get('fps'));
  const recording = a.get('recording');
  if (!xml || !(fps > 0) || !recording) die('usage: cvat-windows.ts --xml file.xml --fps 30 --recording id [--person p] [--setup s] [--window-ms 3000] [--out f | --append f]');
  const tags = parseCvatTags(readFileSync(xml, 'utf8'));
  const { windows, unknownLabels } = tagsToWindows(tags, {
    fps,
    windowMs: Number(a.get('window-ms') ?? 3000),
    recording,
    person: a.get('person') ?? recording,
    setup: a.get('setup') ?? 'unknown',
  });
  if (unknownLabels.length) console.warn(`ignored unknown labels: ${unknownLabels.join(', ')}`);
  const text = windows.map((w) => JSON.stringify(w)).join('\n') + (windows.length ? '\n' : '');
  const out = a.get('out');
  const append = a.get('append');
  if (append) appendFileSync(append, text);
  else if (out) writeFileSync(out, text);
  else process.stdout.write(text);
  const amb = windows.filter((w) => w.ambiguous).length;
  console.error(`${tags.length} tags -> ${windows.length} windows (${amb} ambiguous)`);
}

main();
