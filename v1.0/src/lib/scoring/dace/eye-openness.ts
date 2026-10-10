/**
 * Eye openness corrected for vertical gaze (Annex A.1, FR3).
 *
 *   Expected openness  Ê(g) = β0 + β1 g + β2 g², fitted with robust (Huber)
 *                      regression on attentive frames.
 *   Residual openness  r_t = EAR_t / Ê(g_t); the eye counts as closed when r_t < k.
 *
 * Until enough attentive samples exist, or while the sample's gaze range is
 * too narrow to identify β1/β2 (screen-facing frames only), the TBD prior
 * `earPrior` provides the gaze shape and only the level is fitted:
 *   Ê(g) = β0_fit · s(g),  s(g) = prior(g) / prior.b0.
 *
 * PEB: "the eye-closure threshold is k × (PEB's personal open-eye level) ×
 * (DACE's gaze correction)" (proposal §4). Implemented multiplicatively:
 *   Ê_PEB(g) = openEyeLevel · Ê(g) / Ê(0)
 * which equals replacing β0 when openEyeLevel = β0 (annex A.1 wording), and
 * keeps DACE's contribution a pure gaze-dependent factor.
 */
import type { DaceParams, EarPrior } from './params';
import type { MeshPoint } from './types';
import { MESH_LEFT_EYE_EAR, MESH_RIGHT_EYE_EAR } from './mesh-adapter';

/** 6-point EAR on mesh landmarks (normalized coords, aspect-corrected), averaged over both eyes. */
export function meshEAR(mesh: readonly MeshPoint[] | null | undefined, aspect = 1): number | null {
  if (!mesh || mesh.length < 468) return null;
  const d = (a: MeshPoint, b: MeshPoint) => Math.hypot((a.x - b.x) * aspect, a.y - b.y);
  const one = (idx: readonly number[]): number | null => {
    const p = idx.map((i) => mesh[i]);
    if (p.some((q) => !q)) return null;
    const horiz = d(p[0], p[3]);
    if (!(horiz > 0)) return null;
    return (d(p[1], p[5]) + d(p[2], p[4])) / (2 * horiz);
  };
  const r = one(MESH_RIGHT_EYE_EAR);
  const l = one(MESH_LEFT_EYE_EAR);
  if (r === null || l === null) return null;
  return (r + l) / 2;
}

export type QuadCoef = { b0: number; b1: number; b2: number };

export const evalQuad = (c: QuadCoef, g: number) => c.b0 + c.b1 * g + c.b2 * g * g;

function medianOf(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Robust scale: 1.4826 × MAD (falls back to a tiny positive value). */
function robustScale(res: number[]): number {
  const med = medianOf(res);
  const mad = medianOf(res.map((r) => Math.abs(r - med)));
  return Math.max(1.4826 * mad, 1e-9);
}

function solve3(a: number[][], b: number[]): number[] | null {
  // Gaussian elimination with partial pivoting.
  const m = a.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;
    if (Math.abs(m[p][c]) < 1e-12) return null;
    [m[c], m[p]] = [m[p], m[c]];
    for (let r = 0; r < 3; r++) {
      if (r === c) continue;
      const f = m[r][c] / m[c][c];
      for (let k = c; k < 4; k++) m[r][k] -= f * m[c][k];
    }
  }
  return [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
}

/** Huber IRLS fit of e = b0 + b1 g + b2 g². Returns null if the design is singular. */
export function huberFitQuadratic(
  samples: ReadonlyArray<{ g: number; e: number }>,
  delta = 1.345,
  maxIter = 100
): QuadCoef | null {
  if (samples.length < 3) return null;
  let w = samples.map(() => 1);
  let coef: number[] | null = null;
  for (let it = 0; it < maxIter; it++) {
    const A = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    const B = [0, 0, 0];
    samples.forEach((s, i) => {
      const x = [1, s.g, s.g * s.g];
      for (let r = 0; r < 3; r++) {
        B[r] += w[i] * x[r] * s.e;
        for (let c = 0; c < 3; c++) A[r][c] += w[i] * x[r] * x[c];
      }
    });
    const next = solve3(A, B);
    if (!next) return null;
    const res = samples.map((s) => s.e - (next[0] + next[1] * s.g + next[2] * s.g * s.g));
    const k = delta * robustScale(res);
    const nw = res.map((r) => (Math.abs(r) <= k ? 1 : k / Math.abs(r)));
    const converged = coef !== null && next.every((v, i) => Math.abs(v - coef![i]) < 1e-10);
    coef = next;
    w = nw;
    if (converged) break;
  }
  return coef ? { b0: coef[0], b1: coef[1], b2: coef[2] } : null;
}

/** Huber M-estimate of location (IRLS). */
export function huberLocation(values: readonly number[], delta = 1.345, maxIter = 100): number | null {
  if (values.length === 0) return null;
  let mu = medianOf([...values]);
  for (let it = 0; it < maxIter; it++) {
    const res = values.map((v) => v - mu);
    const k = delta * robustScale(res);
    let num = 0;
    let den = 0;
    for (const r of res) {
      const w = Math.abs(r) <= k ? 1 : k / Math.abs(r);
      num += w * (mu + r);
      den += w;
    }
    const next = num / den;
    if (Math.abs(next - mu) < 1e-12) {
      mu = next;
      break;
    }
    mu = next;
  }
  return mu;
}

export type OpennessFit = { coef: QuadCoef; kind: 'prior' | 'level' | 'quadratic'; n: number };

/**
 * Online openness model. Feed it (g, EAR) from attentive, eyes-open frames
 * (the caller decides which frames qualify); call refit() periodically.
 */
export class OpennessModel {
  private samples: Array<{ g: number; e: number }> = [];
  private fit: OpennessFit;

  constructor(
    private readonly params: Pick<DaceParams, 'earPrior' | 'opennessMinSamples' | 'opennessMinGazeRange' | 'huberDelta' | 'kOpen'>,
    private readonly maxSamples = 3000
  ) {
    this.fit = { coef: { ...params.earPrior }, kind: 'prior', n: 0 };
  }

  addSample(g: number | null, ear: number | null): void {
    if (ear === null || !Number.isFinite(ear) || ear <= 0) return;
    this.samples.push({ g: g ?? 0, e: ear });
    if (this.samples.length > this.maxSamples) this.samples.shift();
  }

  refit(): OpennessFit {
    const p = this.params;
    const n = this.samples.length;
    if (n < p.opennessMinSamples) {
      this.fit = { coef: { ...p.earPrior }, kind: 'prior', n };
      return this.fit;
    }
    const gs = this.samples.map((s) => s.g);
    const range = Math.max(...gs) - Math.min(...gs);
    if (range >= p.opennessMinGazeRange) {
      const q = huberFitQuadratic(this.samples, p.huberDelta);
      if (q && q.b0 > 0) {
        this.fit = { coef: q, kind: 'quadratic', n };
        return this.fit;
      }
    }
    const prior: EarPrior = p.earPrior;
    const shape = (g: number) => evalQuad(prior, g) / prior.b0;
    const level = huberLocation(this.samples.map((s) => s.e / shape(s.g)), p.huberDelta);
    if (level !== null && level > 0) {
      this.fit = {
        coef: { b0: level, b1: level * (prior.b1 / prior.b0), b2: level * (prior.b2 / prior.b0) },
        kind: 'level',
        n,
      };
    }
    return this.fit;
  }

  getFit(): OpennessFit {
    return this.fit;
  }

  /** Ê(g); with a PEB personal open-eye level, Ê_PEB(g) = level · Ê(g) / Ê(0). */
  expected(g: number | null, pebOpenEyeLevel: number | null = null): number {
    const gg = g ?? 0;
    const c = this.fit.coef;
    const e = evalQuad(c, gg);
    if (pebOpenEyeLevel !== null && Number.isFinite(pebOpenEyeLevel) && pebOpenEyeLevel > 0 && c.b0 > 0) {
      return (pebOpenEyeLevel * e) / c.b0;
    }
    return e;
  }

  /** r_t = EAR / Ê(g). Null when EAR is unknown or Ê(g) ≤ 0. */
  residual(ear: number | null, g: number | null, pebOpenEyeLevel: number | null = null): number | null {
    if (ear === null || !Number.isFinite(ear)) return null;
    const e = this.expected(g, pebOpenEyeLevel);
    return e > 0 ? ear / e : null;
  }

  /** closed iff r_t < k (strict). */
  isClosed(residual: number | null): boolean | null {
    return residual === null ? null : residual < this.params.kOpen;
  }

  reset(): void {
    this.samples = [];
    this.fit = { coef: { ...this.params.earPrior }, kind: 'prior', n: 0 };
  }
}
