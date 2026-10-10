/**
 * Verifies the mesh -> 68 adapter's left/right table against the CANONICAL
 * MediaPipe face mesh, not just against itself (the Phase 1 audit flagged the
 * mirror table as unchecked):
 *   1. canonical-mesh-mirror.json: mirror pairs and sides derived from
 *      canonical_face_model.obj (exactly symmetric; residual 0);
 *   2. the connection tables shipped in the installed @mediapipe/tasks-vision
 *      package (FACE_LANDMARKS_LEFT_EYE / RIGHT_EYE / brows / iris / oval / lips).
 */
import { describe, it, expect } from 'vitest';
import { FaceLandmarker } from '@mediapipe/tasks-vision';
import canonical from './canonical-mesh-mirror.json';
import {
  MESH_LEFT_EYE_CORNERS,
  MESH_LEFT_EYE_EAR,
  MESH_LEFT_IRIS_CENTER,
  MESH_MIRROR,
  MESH_RIGHT_EYE_CORNERS,
  MESH_RIGHT_EYE_EAR,
  MESH_RIGHT_IRIS_CENTER,
  MESH_TO_68,
  PAIRS_68,
} from '../mesh-adapter';

type Conn = { start: number; end: number };
const mirror: number[] = canonical.mirror;
const sideX: number[] = canonical.sideX;

/** 68-point indices on the SUBJECT'S RIGHT (image left for an unmirrored camera). */
const RIGHT_68 = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 31, 32, 36, 37, 38, 39, 40, 41, 48, 49, 50, 58, 59, 60, 61, 67];
const MIDLINE_68 = [8, 27, 28, 29, 30, 33, 51, 57, 62, 66];

const nodes = (c: Conn[]) => new Set(c.flatMap((e) => [e.start, e.end]));

describe('canonical face model (canonical_face_model.obj)', () => {
  it('is exactly symmetric and the fixture covers the 468 face points', () => {
    expect(canonical.maxMirrorResidual).toBe(0);
    expect(mirror).toHaveLength(468);
    mirror.forEach((m, i) => expect(mirror[m]).toBe(i));
  });

  it('every MESH_MIRROR entry is a canonical mirror pair', () => {
    for (const [a, b] of Object.entries(MESH_MIRROR)) expect(mirror[Number(a)]).toBe(b);
  });

  it('every 68-point left/right pair maps to a canonical mirror pair; midline points are self-mirrored', () => {
    for (const [a, b] of PAIRS_68) expect(mirror[MESH_TO_68[a]]).toBe(MESH_TO_68[b]);
    for (const i of MIDLINE_68) {
      expect(mirror[MESH_TO_68[i]]).toBe(MESH_TO_68[i]);
      expect(sideX[MESH_TO_68[i]]).toBe(0);
    }
  });

  it("subject-right 68-point indices are on the canonical subject-right side (eye 33's side)", () => {
    const right = sideX[33];
    expect(right).toBe(-1);
    for (const i of RIGHT_68) expect(sideX[MESH_TO_68[i]]).toBe(right);
    const left = new Set(Array.from({ length: 68 }, (_, i) => i).filter((i) => !RIGHT_68.includes(i) && !MIDLINE_68.includes(i)));
    for (const i of left) expect(sideX[MESH_TO_68[i]]).toBe(-right);
  });
});

describe('connection tables of the installed @mediapipe/tasks-vision package', () => {
  it('eye EAR points and corners sit on the right/left eye contours (subject side)', () => {
    const re = nodes(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE);
    const le = nodes(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE);
    for (const i of [...MESH_RIGHT_EYE_EAR, ...MESH_RIGHT_EYE_CORNERS]) expect(re.has(i)).toBe(true);
    for (const i of [...MESH_LEFT_EYE_EAR, ...MESH_LEFT_EYE_CORNERS]) expect(le.has(i)).toBe(true);
  });

  it('left/right eye and brow contours are listed in mirror order (pairing by position reproduces the canonical mirror)', () => {
    const pairs = (l: Conn[], r: Conn[]) => {
      expect(l.length).toBe(r.length);
      l.forEach((e, k) => {
        expect(mirror[e.start]).toBe(r[k].start);
        expect(mirror[e.end]).toBe(r[k].end);
      });
    };
    pairs(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE);
    pairs(FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW);
  });

  it('brow, oval and lip points used by the adapter are on those contours', () => {
    const brows = new Set([...nodes(FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW), ...nodes(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW)]);
    for (const i of MESH_TO_68.slice(17, 27)) expect(brows.has(i)).toBe(true);
    const oval = nodes(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL);
    for (const i of MESH_TO_68.slice(0, 17)) expect(oval.has(i)).toBe(true);
    const lips = nodes(FaceLandmarker.FACE_LANDMARKS_LIPS);
    for (const i of MESH_TO_68.slice(48, 68)) expect(lips.has(i)).toBe(true);
  });

  it('iris centres: 468 belongs to the right iris, 473 to the left (rings 469-472 / 474-477)', () => {
    const ri = nodes(FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS);
    const li = nodes(FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS);
    expect([...ri].sort()).toEqual([469, 470, 471, 472]);
    expect([...li].sort()).toEqual([474, 475, 476, 477]);
    expect(MESH_RIGHT_IRIS_CENTER).toBe(468);
    expect(MESH_LEFT_IRIS_CENTER).toBe(473);
  });
});
