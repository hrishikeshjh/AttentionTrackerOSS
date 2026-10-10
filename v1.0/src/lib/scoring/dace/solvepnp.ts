/**
 * solvePnP head pose on the existing 68 points (baseline A'', Annex D.5).
 *
 * Minimal-change baseline for the measurement layer: no new library, true
 * degrees, but no iris (so gaze is NOT fixed). Pure TypeScript
 * Levenberg-Marquardt over a 6-DoF pose (Rodrigues rotation + translation),
 * minimising reprojection error of the standard OpenCV 6-point head model:
 *
 *   point                      68-pt index   model (mm, x right, y UP, z toward viewer)
 *   nose tip                   30            (   0,    0,    0)
 *   chin                        8            (   0, −330,  −65)
 *   left eye left corner       36            (−225,  170, −135)
 *   right eye right corner     45            ( 225,  170, −135)
 *   left mouth corner          48            (−150, −150, −125)
 *   right mouth corner         54            ( 150, −150, −125)
 *
 * Camera: pinhole, f = frame width (px), principal point = frame centre, no
 * distortion. The solve runs in the OpenCV camera frame (x right, y down,
 * z forward) with the model expressed as (x, −y, −z), so a frontal face is
 * R = I. The result is converted to the y-up convention of
 * head-pose-matrix.ts, giving +pitch = head down.
 */
import type { Point2D } from '../geometry';
import type { EulerDeg, Mat3 } from './types';
import { cvToGlRotation, eulerFromRotation } from './head-pose-matrix';

export const PNP_MODEL_POINTS_MM: ReadonlyArray<{ idx: number; p: [number, number, number] }> = Object.freeze([
  { idx: 30, p: [0, 0, 0] },
  { idx: 8, p: [0, -330, -65] },
  { idx: 36, p: [-225, 170, -135] },
  { idx: 45, p: [225, 170, -135] },
  { idx: 48, p: [-150, -150, -125] },
  { idx: 54, p: [150, -150, -125] },
]);

/** Model points in the OpenCV camera convention (x, −y, −z). */
const MODEL_CV: Array<[number, number, number]> = PNP_MODEL_POINTS_MM.map(({ p }) => [p[0], -p[1], -p[2]]);

export function rodrigues(r: [number, number, number]): Mat3 {
  const th = Math.hypot(r[0], r[1], r[2]);
  if (th < 1e-12) {
    return [
      [1, -r[2], r[1]],
      [r[2], 1, -r[0]],
      [-r[1], r[0], 1],
    ];
  }
  const k = [r[0] / th, r[1] / th, r[2] / th];
  const c = Math.cos(th);
  const s = Math.sin(th);
  const v = 1 - c;
  return [
    [c + k[0] * k[0] * v, k[0] * k[1] * v - k[2] * s, k[0] * k[2] * v + k[1] * s],
    [k[1] * k[0] * v + k[2] * s, c + k[1] * k[1] * v, k[1] * k[2] * v - k[0] * s],
    [k[2] * k[0] * v - k[1] * s, k[2] * k[1] * v + k[0] * s, c + k[2] * k[2] * v],
  ];
}

export type Intrinsics = { f: number; cx: number; cy: number };

/** Project model points with pose (R in OpenCV frame, t in mm). */
export function projectModel(R: Mat3, t: [number, number, number], K: Intrinsics): Point2D[] {
  return MODEL_CV.map((m) => {
    const X = R[0][0] * m[0] + R[0][1] * m[1] + R[0][2] * m[2] + t[0];
    const Y = R[1][0] * m[0] + R[1][1] * m[1] + R[1][2] * m[2] + t[1];
    const Z = R[2][0] * m[0] + R[2][1] * m[1] + R[2][2] * m[2] + t[2];
    return { x: (K.f * X) / Z + K.cx, y: (K.f * Y) / Z + K.cy };
  });
}

function solveN(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const m = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;
    if (Math.abs(m[p][c]) < 1e-15) return null;
    [m[c], m[p]] = [m[p], m[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = m[r][c] / m[c][c];
      for (let k = c; k <= n; k++) m[r][k] -= f * m[c][k];
    }
  }
  return m.map((row, i) => row[n] / row[i]);
}

/** Rotation about the OpenCV x-axis (positive = head down for this model). */
function rvecForPitch(deg: number): [number, number, number] {
  return [(deg * Math.PI) / 180, 0, 0];
}

export type PnpResult = { euler: EulerDeg; rmsPx: number; rotationCv: Mat3; translation: [number, number, number] };

/** Solve pose from the 6 image points (same order as PNP_MODEL_POINTS_MM). */
export function solvePnP6(image: readonly Point2D[], K: Intrinsics, maxIter = 200): PnpResult | null {
  if (image.length !== 6 || image.some((p) => !p || !Number.isFinite(p.x) || !Number.isFinite(p.y))) return null;
  const eyeDist = Math.hypot(image[2].x - image[3].x, image[2].y - image[3].y);
  if (!(eyeDist > 0)) return null;
  const tz0 = (K.f * 450) / eyeDist;
  const t0: [number, number, number] = [((image[0].x - K.cx) * tz0) / K.f, ((image[0].y - K.cy) * tz0) / K.f, tz0];

  const residual = (q: number[]): number[] => {
    const pr = projectModel(rodrigues([q[0], q[1], q[2]]), [q[3], q[4], q[5]], K);
    const r: number[] = [];
    pr.forEach((p, i) => r.push(p.x - image[i].x, p.y - image[i].y));
    return r;
  };
  const sse = (r: number[]) => r.reduce((a, v) => a + v * v, 0);

  let best: { q: number[]; e: number } | null = null;
  for (const initPitch of [0, 30, -30]) {
    let q = [...rvecForPitch(initPitch), ...t0];
    let r = residual(q);
    let e = sse(r);
    let lambda = 1e-3;
    for (let it = 0; it < maxIter; it++) {
      // numeric Jacobian
      const J: number[][] = r.map(() => new Array(6).fill(0));
      for (let j = 0; j < 6; j++) {
        const h = j < 3 ? 1e-6 : Math.max(1e-6, Math.abs(q[j]) * 1e-6);
        const qp = [...q];
        qp[j] += h;
        const rp = residual(qp);
        for (let i = 0; i < r.length; i++) J[i][j] = (rp[i] - r[i]) / h;
      }
      const JtJ = Array.from({ length: 6 }, (_, a) =>
        Array.from({ length: 6 }, (_, b) => J.reduce((s, row) => s + row[a] * row[b], 0))
      );
      const Jtr = Array.from({ length: 6 }, (_, a) => J.reduce((s, row, i) => s + row[a] * r[i], 0));
      let improved = false;
      for (let tries = 0; tries < 10; tries++) {
        const A = JtJ.map((row, a) => row.map((v, b) => (a === b ? v * (1 + lambda) + 1e-12 : v)));
        const d = solveN(A, Jtr.map((v) => -v));
        if (!d) break;
        const qn = q.map((v, i) => v + d[i]);
        const rn = residual(qn);
        const en = sse(rn);
        if (en < e) {
          q = qn;
          r = rn;
          const rel = (e - en) / Math.max(e, 1e-30);
          e = en;
          lambda = Math.max(lambda / 10, 1e-12);
          improved = true;
          if (rel < 1e-14) it = maxIter;
          break;
        }
        lambda *= 10;
      }
      if (!improved || e < 1e-18) break;
    }
    if (q[5] > 0 && (best === null || e < best.e)) best = { q, e };
  }
  if (!best) return null;
  const Rcv = rodrigues([best.q[0], best.q[1], best.q[2]]);
  return {
    euler: eulerFromRotation(cvToGlRotation(Rcv)),
    rmsPx: Math.sqrt(best.e / 12),
    rotationCv: Rcv,
    translation: [best.q[3], best.q[4], best.q[5]],
  };
}

/** solvePnP on face-api 68-point landmarks (pixel coords). */
export function solvePnP68(landmarks: readonly Point2D[], frameWidth: number, frameHeight: number): PnpResult | null {
  if (!landmarks || landmarks.length < 68) return null;
  const pts = PNP_MODEL_POINTS_MM.map(({ idx }) => landmarks[idx]);
  return solvePnP6(pts, { f: frameWidth, cx: frameWidth / 2, cy: frameHeight / 2 });
}
