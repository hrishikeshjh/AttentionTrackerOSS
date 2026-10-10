/**
 * Fixed-rate tick scheduler for DACE mode (plan D6).
 *
 * The legacy loop is requestAnimationFrame-bound with predictions at 1 Hz;
 * DACE needs a fixed 5 FPS base rate (200 ms) because its windows are counted
 * in frames. A tick that arrives while the previous one is still running is
 * SKIPPED (never queued) and counted as dropped, so a slow frame cannot build
 * a backlog. Dropped ticks are reported via onDropped and in stats().
 *
 * Timers are injectable for tests. Note: browsers throttle timers in
 * background tabs (plan section 11 risk); the dropped/elapsed counts make
 * that visible in the logs.
 */
export type Tick = { i: number; t: number };

export type SchedulerOptions = {
  intervalMs?: number;
  now?: () => number;
  setInterval?: (fn: () => void, ms: number) => unknown;
  clearInterval?: (h: unknown) => void;
  onDropped?: (info: { i: number; t: number; dropped: number }) => void;
  onError?: (e: unknown) => void;
};

export type SchedulerStats = { ticks: number; ran: number; dropped: number; errors: number };

export class FixedTickScheduler {
  private handle: unknown = null;
  private busy = false;
  private i = 0;
  private stats_: SchedulerStats = { ticks: 0, ran: 0, dropped: 0, errors: 0 };
  readonly intervalMs: number;
  private readonly now: () => number;
  private readonly setI: (fn: () => void, ms: number) => unknown;
  private readonly clearI: (h: unknown) => void;

  constructor(
    private readonly task: (tick: Tick) => Promise<void> | void,
    private readonly opts: SchedulerOptions = {}
  ) {
    this.intervalMs = opts.intervalMs ?? 200;
    this.now = opts.now ?? (() => (typeof performance !== 'undefined' ? performance.now() : Date.now()));
    this.setI = opts.setInterval ?? ((fn, ms) => setInterval(fn, ms));
    this.clearI = opts.clearInterval ?? ((h) => clearInterval(h as ReturnType<typeof setInterval>));
  }

  start(): void {
    if (this.handle !== null) return;
    this.handle = this.setI(() => void this.onTick(), this.intervalMs);
  }

  stop(): void {
    if (this.handle !== null) this.clearI(this.handle);
    this.handle = null;
  }

  isRunning(): boolean {
    return this.handle !== null;
  }

  /** Exposed for tests: one timer firing. */
  async onTick(): Promise<void> {
    const tick: Tick = { i: this.i++, t: this.now() };
    this.stats_.ticks++;
    if (this.busy) {
      this.stats_.dropped++;
      this.opts.onDropped?.({ i: tick.i, t: tick.t, dropped: this.stats_.dropped });
      return;
    }
    this.busy = true;
    try {
      await this.task(tick);
      this.stats_.ran++;
    } catch (e) {
      this.stats_.errors++;
      this.opts.onError?.(e);
    } finally {
      this.busy = false;
    }
  }

  stats(): SchedulerStats {
    return { ...this.stats_ };
  }
}
