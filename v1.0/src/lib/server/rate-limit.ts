/**
 * Minimal in-memory sliding-window rate limiter (per key, e.g. user id).
 *
 * Scope: one server instance. On a multi-instance deployment (App Hosting
 * with several instances) each instance enforces its own window, so the
 * effective limit is N x limit; a shared store (Firestore/Redis) would be
 * needed for a global limit. It still bounds Gemini cost per instance and
 * per authenticated user, which is the purpose here.
 */
export class SlidingWindowRateLimiter {
  private hits = new Map<string, number[]>();

  constructor(
    private readonly limit: number,
    private readonly windowMs: number,
    private readonly maxKeys = 10_000
  ) {}

  /** Record an attempt; returns whether it is allowed and when to retry. */
  check(key: string, nowMs: number = Date.now()): { allowed: boolean; retryAfterMs: number } {
    const cutoff = nowMs - this.windowMs;
    const list = (this.hits.get(key) ?? []).filter((t) => t > cutoff);
    if (list.length >= this.limit) {
      this.hits.set(key, list);
      return { allowed: false, retryAfterMs: list[0] + this.windowMs - nowMs };
    }
    list.push(nowMs);
    this.hits.set(key, list);
    if (this.hits.size > this.maxKeys) this.prune(cutoff);
    return { allowed: true, retryAfterMs: 0 };
  }

  private prune(cutoff: number): void {
    for (const [k, v] of this.hits) {
      if (v.every((t) => t <= cutoff)) this.hits.delete(k);
    }
  }

  reset(): void {
    this.hits.clear();
  }
}
