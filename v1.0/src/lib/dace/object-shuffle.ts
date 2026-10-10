/**
 * Negative control H (Annex D.6): "Full DACE with object labels shuffled
 * across windows" — is the gain really from context?
 *
 * Implementation: every detector output (one object list per detector run)
 * is added to a pool; the list actually fed to the window is drawn uniformly
 * from the pool so far with a seeded PRNG. Labels keep their boxes, so both
 * the class and its spatial relation to the hands are decoupled from the
 * current activity. Deterministic for a given seed and input sequence, so a
 * replay of the same recording reproduces the same control run.
 */
import type { DetectedObject } from '../types';

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const DEFAULT_SHUFFLE_SEED = 0x0dace;

export class ObjectShuffler {
  private pool: DetectedObject[][] = [];
  private rnd: () => number;

  constructor(seed = DEFAULT_SHUFFLE_SEED, private readonly maxPool = 2000) {
    this.rnd = mulberry32(seed);
  }

  /** Add the true detector output and return a shuffled replacement. */
  next(objects: readonly DetectedObject[]): DetectedObject[] {
    this.pool.push(objects.map((o) => ({ ...o, bbox: { ...o.bbox } })));
    if (this.pool.length > this.maxPool) this.pool.shift();
    const k = Math.floor(this.rnd() * this.pool.length);
    return this.pool[k].map((o) => ({ ...o, bbox: { ...o.bbox } }));
  }

  reset(seed = DEFAULT_SHUFFLE_SEED): void {
    this.pool = [];
    this.rnd = mulberry32(seed);
  }
}
