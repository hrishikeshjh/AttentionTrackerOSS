/**
 * MediaPipe 478-point face mesh -> face-api-style 68-point subset.
 *
 * Used so the legacy aggregator (MAR, face area, eyebrow raise, mouth
 * openness, blink/fatigue trackers) can run on MediaPipe landmarks. This is
 * an independently audited table; it deliberately does NOT reuse v2.0's
 * `_MP_TO_68`. Audit rules (enforced in tests):
 *   - 68 unique indices in [0, 467] (no iris points);
 *   - every left/right 68-point pair maps to a MediaPipe mirror pair;
 *   - the eye 6-tuples are the standard mesh EAR points, so EAR on the
 *     adapted 68 points equals meshEAR() on the raw mesh;
 *   - the inner-lip points match the standard mesh MAR points (13/14/78/308).
 *
 * Index order follows the 68-point convention: point 0 is the jaw at the
 * subject's right (image left), 36-41 is the subject's right eye, etc.
 */
import type { Point2D } from '../geometry';
import type { MeshPoint } from './types';

export const MESH_TO_68: readonly number[] = Object.freeze([
  // 0-16 jaw (subject right -> chin 152 -> subject left)
  127, 234, 93, 132, 58, 172, 136, 150, 152, 379, 365, 397, 288, 361, 323, 454, 356,
  // 17-21 right brow (outer -> inner)
  70, 63, 105, 66, 107,
  // 22-26 left brow (inner -> outer)
  336, 296, 334, 293, 300,
  // 27-30 nose bridge -> tip (1 = pronasale)
  168, 197, 5, 1,
  // 31-35 lower nose (subject right -> left; 2 = subnasale)
  98, 97, 2, 326, 327,
  // 36-41 right eye: outer corner, upper outer, upper inner, inner corner, lower inner, lower outer
  33, 160, 158, 133, 153, 144,
  // 42-47 left eye: inner corner, upper inner, upper outer, outer corner, lower outer, lower inner
  362, 385, 387, 263, 373, 380,
  // 48-59 outer lips: right corner, upper, top centre (0), upper, left corner, lower, bottom centre (17), lower
  61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
  // 60-67 inner lips: right corner, upper, top centre (13), upper, left corner, lower, bottom centre (14), lower
  78, 81, 13, 311, 308, 402, 14, 178,
]);

/** 68-point index pairs that must map to mirror-symmetric mesh points. */
export const PAIRS_68: ReadonlyArray<[number, number]> = Object.freeze([
  [0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
  [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],
  [31, 35], [32, 34],
  [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
  [48, 54], [49, 53], [50, 52], [55, 59], [56, 58],
  [60, 64], [61, 63], [65, 67],
] as Array<[number, number]>);

/** Mirror pairs in the MediaPipe canonical face mesh for the indices used above. */
export const MESH_MIRROR: Readonly<Record<number, number>> = Object.freeze({
  127: 356, 234: 454, 93: 323, 132: 361, 58: 288, 172: 397, 136: 365, 150: 379,
  70: 300, 63: 293, 105: 334, 66: 296, 107: 336,
  98: 327, 97: 326,
  33: 263, 160: 387, 158: 385, 133: 362, 153: 380, 144: 373,
  61: 291, 40: 270, 37: 267, 321: 91, 314: 84,
  78: 308, 81: 311, 402: 178,
});

/** Standard mesh EAR points (p1..p6). */
export const MESH_RIGHT_EYE_EAR = Object.freeze([33, 160, 158, 133, 153, 144]);
export const MESH_LEFT_EYE_EAR = Object.freeze([362, 385, 387, 263, 373, 380]);
/** Iris centres (refined landmarks): 468 = subject right, 473 = subject left. */
export const MESH_RIGHT_IRIS_CENTER = 468;
export const MESH_LEFT_IRIS_CENTER = 473;
/** Eye corners (outer, inner). */
export const MESH_RIGHT_EYE_CORNERS = Object.freeze([33, 133]);
export const MESH_LEFT_EYE_CORNERS = Object.freeze([263, 362]);

/**
 * Convert normalized mesh landmarks to the 68-point pixel layout expected by
 * the legacy aggregator (which works in pixel units).
 */
export function meshTo68(mesh: readonly MeshPoint[], frameWidth: number, frameHeight: number): Point2D[] | null {
  if (!mesh || mesh.length < 468) return null;
  const out: Point2D[] = [];
  for (const idx of MESH_TO_68) {
    const p = mesh[idx];
    if (!p) return null;
    out.push({ x: p.x * frameWidth, y: p.y * frameHeight });
  }
  return out;
}

/** Face box (normalized, centre format) from mesh landmarks (iris excluded). */
export function meshFaceBox(mesh: readonly MeshPoint[]): { x: number; y: number; width: number; height: number } | null {
  if (!mesh || mesh.length < 468) return null;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < 468; i++) {
    const p = mesh[i];
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return { x: (minX + maxX) / 2, y: (minY + maxY) / 2, width: maxX - minX, height: maxY - minY };
}
