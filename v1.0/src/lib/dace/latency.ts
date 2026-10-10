/**
 * Per-component latency probes (non-functional requirement: p95 total
 * processing <= 200 ms per frame at 5 FPS, reported per component).
 * p95 <= 200 ms is a TARGET; these probes are how it gets measured.
 *
 * Each component keeps a fixed-size ring buffer of recent durations;
 * summary() reports n, p50, p95 and max (nearest-rank percentiles).
 */
export type LatencyStats = { n: number; p50: number; p95: number; max: number; last: number };
export type LatencySummary = Record<string, LatencyStats>;

export const LATENCY_COMPONENTS = ['measure', 'detector', 'hands', 'evidence', 'rules', 'mapping', 'total'] as const;

class Ring {
  private buf: Float64Array;
  private n = 0;
  private i = 0;
  last = 0;
  constructor(size: number) {
    this.buf = new Float64Array(size);
  }
  push(v: number): void {
    this.buf[this.i] = v;
    this.i = (this.i + 1) % this.buf.length;
    if (this.n < this.buf.length) this.n++;
    this.last = v;
  }
  values(): number[] {
    return Array.from(this.buf.subarray(0, this.n));
  }
}

/** Nearest-rank percentile of an ascending array. */
export function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) return 0;
  const rank = Math.ceil((p / 100) * sorted.length);
  return sorted[Math.min(sorted.length, Math.max(1, rank)) - 1];
}

export class LatencyProbe {
  private rings = new Map<string, Ring>();

  constructor(
    private readonly size = 300,
    private readonly now: () => number = () => (typeof performance !== 'undefined' ? performance.now() : Date.now())
  ) {}

  record(component: string, ms: number): void {
    if (!Number.isFinite(ms)) return;
    let r = this.rings.get(component);
    if (!r) {
      r = new Ring(this.size);
      this.rings.set(component, r);
    }
    r.push(ms);
  }

  /** Time an async or sync function and record its duration (also on throw). */
  async time<T>(component: string, fn: () => T | Promise<T>): Promise<T> {
    const t0 = this.now();
    try {
      return await fn();
    } finally {
      this.record(component, this.now() - t0);
    }
  }

  stats(component: string): LatencyStats | null {
    const r = this.rings.get(component);
    if (!r) return null;
    const v = r.values().sort((a, b) => a - b);
    return { n: v.length, p50: percentile(v, 50), p95: percentile(v, 95), max: v[v.length - 1] ?? 0, last: r.last };
  }

  summary(): LatencySummary {
    const out: LatencySummary = {};
    for (const k of this.rings.keys()) out[k] = this.stats(k) as LatencyStats;
    return out;
  }

  /** last recorded duration per component (for per-window log records) */
  lastValues(): Record<string, number> {
    const out: Record<string, number> = {};
    for (const [k, r] of this.rings) out[k] = r.last;
    return out;
  }

  reset(): void {
    this.rings.clear();
  }
}
