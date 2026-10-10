/**
 * CVAT tag annotations -> 3-second ground-truth windows (Annex D.1/D.2).
 *
 * Input: a "CVAT for video 1.1" XML export in which annotators placed frame
 * TAGS (one label per tagged frame) with optional attributes:
 *   <tag frame="123" label="attentive_reading" source="manual">
 *     <attribute name="ambiguous">true</attribute>
 *     <attribute name="segment">2</attribute>
 *   </tag>
 * Labels are the six DACE classes; common aliases are accepted (LABEL_ALIASES).
 * "ambiguous" is the annotators' "ambiguous even to a human" flag (the
 * reference for evaluating Unsure). "segment" is the scripted segment id 1-10
 * (10 = desk out of frame).
 *
 * Output window: [k*windowMs, (k+1)*windowMs) in media time. Label = majority
 * of the tagged frames inside; ambiguous if any tagged frame is ambiguous or
 * the majority holds less than half of the tagged frames; segment = majority
 * segment. Windows without tags are skipped.
 *
 * This is a new adapter for window labels; the existing classroom-JSON CVAT
 * converter in v2.0 is not reused (plan D15).
 */
import type { BehaviorLabel } from '../../types';

export const BEHAVIOR_LABELS: readonly BehaviorLabel[] = [
  'attentive_screen', 'attentive_reading', 'using_phone', 'distracted', 'drowsy', 'absent',
];

export const LABEL_ALIASES: Readonly<Record<string, BehaviorLabel>> = Object.freeze({
  screen: 'attentive_screen',
  attentive: 'attentive_screen',
  reading: 'attentive_reading',
  writing: 'attentive_reading',
  note_taking: 'attentive_reading',
  phone: 'using_phone',
  looking_away: 'distracted',
  sleepy: 'drowsy',
  away: 'absent',
});

export const DESK_OUT_OF_FRAME_SEGMENT = '10';

export function normalizeLabel(raw: string): BehaviorLabel | null {
  const k = raw.trim().toLowerCase().replace(/[\s-]+/g, '_');
  if ((BEHAVIOR_LABELS as readonly string[]).includes(k)) return k as BehaviorLabel;
  return LABEL_ALIASES[k] ?? null;
}

export type CvatTag = { frame: number; label: string; attributes: Record<string, string> };

const decode = (s: string) =>
  s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&amp;/g, '&');

function attr(tag: string, name: string): string | null {
  const m = new RegExp(`\\b${name}\\s*=\\s*"([^"]*)"`).exec(tag);
  return m ? decode(m[1]) : null;
}

/** Minimal parser for <tag> elements of a CVAT XML export (no external XML dependency). */
export function parseCvatTags(xml: string): CvatTag[] {
  const out: CvatTag[] = [];
  const re = /<tag\b([^>]*?)(\/>|>([\s\S]*?)<\/tag>)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(xml))) {
    const head = m[1];
    const frame = Number(attr(head, 'frame'));
    const label = attr(head, 'label');
    if (!Number.isFinite(frame) || label === null) continue;
    const attributes: Record<string, string> = {};
    const body = m[3] ?? '';
    const ar = /<attribute\b[^>]*name\s*=\s*"([^"]*)"[^>]*>([\s\S]*?)<\/attribute>/g;
    let a: RegExpExecArray | null;
    while ((a = ar.exec(body))) attributes[decode(a[1])] = decode(a[2].trim());
    out.push({ frame, label, attributes });
  }
  return out;
}

export type GtWindow = {
  recording: string;
  person: string;
  setup: string;
  segment: string;
  start_ms: number;
  end_ms: number;
  label: BehaviorLabel;
  ambiguous: boolean;
  /** tagged frames in the window */
  n: number;
};

const truthy = (v: string | undefined) => v !== undefined && /^(true|1|yes)$/i.test(v.trim());

function majority<T extends string>(values: T[]): { value: T; count: number } | null {
  const c = new Map<T, number>();
  for (const v of values) c.set(v, (c.get(v) ?? 0) + 1);
  let best: { value: T; count: number } | null = null;
  for (const [value, count] of c) if (!best || count > best.count) best = { value, count };
  return best;
}

export function tagsToWindows(
  tags: readonly CvatTag[],
  opts: { fps: number; windowMs?: number; recording: string; person: string; setup: string }
): { windows: GtWindow[]; unknownLabels: string[] } {
  const windowMs = opts.windowMs ?? 3000;
  const unknown = new Set<string>();
  const byWin = new Map<number, Array<{ label: BehaviorLabel; ambiguous: boolean; segment: string }>>();
  for (const t of tags) {
    const label = normalizeLabel(t.label);
    if (!label) {
      unknown.add(t.label);
      continue;
    }
    const ms = (t.frame / opts.fps) * 1000;
    const k = Math.floor(ms / windowMs + 1e-9);
    const arr = byWin.get(k) ?? [];
    arr.push({ label, ambiguous: truthy(t.attributes.ambiguous), segment: t.attributes.segment ?? 'unknown' });
    byWin.set(k, arr);
  }
  const windows: GtWindow[] = [...byWin.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([k, items]) => {
      const maj = majority(items.map((i) => i.label))!;
      const seg = majority(items.map((i) => i.segment))!;
      return {
        recording: opts.recording,
        person: opts.person,
        setup: opts.setup,
        segment: seg.value,
        start_ms: k * windowMs,
        end_ms: (k + 1) * windowMs,
        label: maj.value,
        ambiguous: items.some((i) => i.ambiguous) || maj.count * 2 < items.length,
        n: items.length,
      };
    });
  return { windows, unknownLabels: [...unknown] };
}
