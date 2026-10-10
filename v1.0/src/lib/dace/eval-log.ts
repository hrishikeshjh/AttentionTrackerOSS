/**
 * DACE evaluation log (FR9): per-window features, labels and reason codes as
 * JSON Lines.
 *
 * Privacy rules (proposal section 7, plan D5):
 *   - no images or pixel data, ever: only numbers, flags and landmark
 *     coordinates already computed in the browser;
 *   - no user id, e-mail or name: each log gets a RANDOM session id;
 *   - nothing is persisted or sent automatically: records live in memory and
 *     leave the browser only through an explicit, manual download (research
 *     builds only, see visuals-panel / replay page).
 * Every record carries DACE_VERSION (`v`) and the config hash (`cfg`).
 */
import type { DaceConfig } from './config';
import { DACE_VERSION } from '../scoring/dace/VERSION';

export type EvalRecordType = 'session' | 'frame' | 'window' | 'prediction' | 'event' | 'latency';

export type EvalRecord = {
  type: EvalRecordType;
  /** DACE_VERSION */
  v: string;
  /** config hash */
  cfg: string;
  /** random session id */
  sid: string;
  /** ms (session clock for webcam, media time for file replay) */
  t: number;
  [k: string]: unknown;
};

/** Fields that must never appear in a record (defence in depth). */
export const FORBIDDEN_KEYS = Object.freeze(['image', 'imageData', 'pixels', 'frameData', 'dataUrl', 'userId', 'uid', 'email', 'displayName']);

export function randomSessionId(): string {
  const bytes = new Uint8Array(16);
  const c = globalThis.crypto;
  if (c?.getRandomValues) c.getRandomValues(bytes);
  else for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

function scrub(v: unknown, depth = 0): unknown {
  if (depth > 8) return null;
  if (v === null || typeof v !== 'object') {
    if (typeof v === 'number' && !Number.isFinite(v)) return null;
    return v;
  }
  if (ArrayBuffer.isView(v) || v instanceof ArrayBuffer) return null;
  if (typeof ImageData !== 'undefined' && v instanceof ImageData) return null;
  if (Array.isArray(v)) return v.map((x) => scrub(x, depth + 1));
  const out: Record<string, unknown> = {};
  for (const [k, x] of Object.entries(v as Record<string, unknown>)) {
    if (FORBIDDEN_KEYS.includes(k)) continue;
    if (typeof x === 'function' || x === undefined) continue;
    out[k] = scrub(x, depth + 1);
  }
  return out;
}

export type EvalLoggerOptions = {
  level: DaceConfig['log'];
  configHash: string;
  sessionId?: string;
  maxRecords?: number;
};

export class EvalLogger {
  readonly sessionId: string;
  readonly level: DaceConfig['log'];
  readonly configHash: string;
  private records: EvalRecord[] = [];
  private readonly maxRecords: number;
  private dropped = 0;

  constructor(opts: EvalLoggerOptions) {
    this.level = opts.level;
    this.configHash = opts.configHash;
    this.sessionId = opts.sessionId ?? randomSessionId();
    this.maxRecords = opts.maxRecords ?? 500_000;
  }

  enabled(): boolean {
    return this.level !== 'off';
  }

  /** Whether a record type is captured at the configured level. */
  accepts(type: EvalRecordType): boolean {
    if (this.level === 'off') return false;
    if (type === 'frame') return this.level === 'frames';
    return true;
  }

  log(type: EvalRecordType, t: number, fields: Record<string, unknown> = {}): void {
    if (!this.accepts(type)) return;
    if (this.records.length >= this.maxRecords) {
      this.dropped++;
      return;
    }
    const body = scrub(fields) as Record<string, unknown>;
    this.records.push({ ...body, type, v: DACE_VERSION, cfg: this.configHash, sid: this.sessionId, t });
  }

  size(): number {
    return this.records.length;
  }

  droppedRecords(): number {
    return this.dropped;
  }

  getRecords(): readonly EvalRecord[] {
    return this.records;
  }

  toJSONL(): string {
    return this.records.map((r) => JSON.stringify(r)).join('\n') + (this.records.length ? '\n' : '');
  }

  clear(): void {
    this.records = [];
    this.dropped = 0;
  }
}

/** Parse JSONL text back into records (scripts and tests). Blank lines are skipped. */
export function parseJsonl<T = EvalRecord>(text: string): T[] {
  return text
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l) as T);
}

/** Manual download (browser only; call from a user click handler). */
export function downloadJsonl(logger: EvalLogger, filename = `dace-eval-${logger.sessionId.slice(0, 8)}.jsonl`): void {
  if (typeof document === 'undefined') return;
  const blob = new Blob([logger.toJSONL()], { type: 'application/x-ndjson' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
