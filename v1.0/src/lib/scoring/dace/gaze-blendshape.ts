/**
 * Eye-in-head gaze (Annex A.1, FR2), kept separate from head pose.
 *
 *   vertical   g_t = ½(eyeLookDown_L + eyeLookDown_R) − ½(eyeLookUp_L + eyeLookUp_R)  ∈ [−1, 1], +down
 *   horizontal h_t built the same way from eyeLookIn / eyeLookOut, + = toward image right
 *              for an unmirrored camera: ½(eyeLookOut_L + eyeLookIn_R) − ½(eyeLookIn_L + eyeLookOut_R)
 *              (blendshape "Left"/"Right" = the subject's eyes, ARKit naming). Only |h| is used in
 *              scoring, so the horizontal sign convention does not affect any score.
 *
 * Iris cross-check: iris centre offset from the eye-corner midpoint divided by
 * eye width (y down => +down). If the iris is unavailable/implausible, or it
 * contradicts the blendshape direction, gaze is marked invalid and D_t falls
 * back to head pitch only (B.1, reflective glasses).
 */
import type { DaceParams } from './params';
import type { BlendshapeMap, MeshPoint } from './types';
import {
  MESH_LEFT_EYE_CORNERS,
  MESH_LEFT_IRIS_CENTER,
  MESH_RIGHT_EYE_CORNERS,
  MESH_RIGHT_IRIS_CENTER,
} from './mesh-adapter';

export type Gaze = { v: number; h: number };

const KEYS = [
  'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookUpLeft', 'eyeLookUpRight',
  'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
] as const;

const clamp1 = (v: number) => Math.max(-1, Math.min(1, v));

export function gazeFromBlendshapes(bs: BlendshapeMap | null | undefined): Gaze | null {
  if (!bs) return null;
  for (const k of KEYS) if (typeof bs[k] !== 'number' || !Number.isFinite(bs[k])) return null;
  const v = 0.5 * (bs.eyeLookDownLeft + bs.eyeLookDownRight) - 0.5 * (bs.eyeLookUpLeft + bs.eyeLookUpRight);
  const h = 0.5 * (bs.eyeLookOutLeft + bs.eyeLookInRight) - 0.5 * (bs.eyeLookInLeft + bs.eyeLookOutRight);
  return { v: clamp1(v), h: clamp1(h) };
}

/** Convert MediaPipe `faceBlendshapes[i].categories` into a map. */
export function blendshapeMap(categories: ReadonlyArray<{ categoryName: string; score: number }>): BlendshapeMap {
  const out: Record<string, number> = {};
  for (const c of categories) out[c.categoryName] = c.score;
  return out;
}

/** Iris offset (fractions of eye width), averaged over both eyes; aspect-corrected. */
export function irisOffset(mesh: readonly MeshPoint[] | null | undefined, aspect = 1): Gaze | null {
  if (!mesh || mesh.length < 478) return null;
  const one = (iris: number, corners: readonly number[]): Gaze | null => {
    const c = mesh[iris];
    const a = mesh[corners[0]];
    const b = mesh[corners[1]];
    if (!c || !a || !b) return null;
    const midX = (a.x + b.x) / 2;
    const midY = (a.y + b.y) / 2;
    const w = Math.hypot((a.x - b.x) * aspect, a.y - b.y);
    if (!(w > 0)) return null;
    return { h: ((c.x - midX) * aspect) / w, v: (c.y - midY) / w };
  };
  const r = one(MESH_RIGHT_IRIS_CENTER, MESH_RIGHT_EYE_CORNERS);
  const l = one(MESH_LEFT_IRIS_CENTER, MESH_LEFT_EYE_CORNERS);
  if (!r || !l) return null;
  return { h: (r.h + l.h) / 2, v: (r.v + l.v) / 2 };
}

export type GazeEstimate = {
  gazeV: number | null;
  gazeH: number | null;
  /** iris available and plausible */
  irisValid: boolean;
  /** usable for D_t (blendshapes present, iris valid and consistent) */
  gazeValid: boolean;
};

export function estimateGaze(
  bs: BlendshapeMap | null | undefined,
  mesh: readonly MeshPoint[] | null | undefined,
  params: Pick<DaceParams, 'gammaDown' | 'irisMaxOffset' | 'irisSignTol' | 'frameAspect'>
): GazeEstimate {
  const g = gazeFromBlendshapes(bs);
  const iris = irisOffset(mesh, params.frameAspect);
  const irisValid =
    iris !== null && Number.isFinite(iris.v) && Number.isFinite(iris.h) &&
    Math.abs(iris.v) <= params.irisMaxOffset && Math.abs(iris.h) <= params.irisMaxOffset;
  if (!g) return { gazeV: null, gazeH: null, irisValid, gazeValid: false };
  let consistent = true;
  if (irisValid && iris) {
    if (g.v >= params.gammaDown && iris.v < -params.irisSignTol) consistent = false;
    if (g.v <= -params.gammaDown && iris.v > params.irisSignTol) consistent = false;
  }
  return { gazeV: g.v, gazeH: g.h, irisValid, gazeValid: irisValid && consistent };
}
