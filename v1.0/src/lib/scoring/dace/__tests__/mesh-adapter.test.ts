import { describe, it, expect } from 'vitest';
import {
  MESH_LEFT_EYE_EAR,
  MESH_MIRROR,
  MESH_RIGHT_EYE_EAR,
  MESH_TO_68,
  PAIRS_68,
  meshFaceBox,
  meshTo68,
} from '../mesh-adapter';
import { meshEAR } from '../eye-openness';
import { calculateEAR, calculateMAR } from '../../ear-mar';
import { mulberry32 } from '../../../__tests__/fixtures/synthetic-face';

function randomMesh(seed = 3) {
  const rnd = mulberry32(seed);
  return Array.from({ length: 478 }, () => ({ x: 0.3 + 0.4 * rnd(), y: 0.2 + 0.5 * rnd(), z: 0 }));
}

describe('mesh -> 68 adapter (audited subset)', () => {
  it('has 68 unique indices inside the 468 face points (no iris)', () => {
    expect(MESH_TO_68).toHaveLength(68);
    expect(new Set(MESH_TO_68).size).toBe(68);
    for (const i of MESH_TO_68) {
      expect(i).toBeGreaterThanOrEqual(0);
      expect(i).toBeLessThan(468);
    }
  });

  it('maps every left/right 68-point pair to a mesh mirror pair; centre points to self-mirrored midline points', () => {
    const mirror = (i: number) => MESH_MIRROR[i] ?? Object.entries(MESH_MIRROR).find(([, v]) => v === i)?.[0];
    for (const [a, b] of PAIRS_68) {
      expect(String(mirror(MESH_TO_68[a]))).toBe(String(MESH_TO_68[b]));
    }
    // midline: chin 152, nose bridge/tip, subnasale, lip centres
    for (const i of [8, 27, 28, 29, 30, 33, 51, 57, 62, 66]) {
      expect(MESH_MIRROR[MESH_TO_68[i]]).toBeUndefined();
    }
  });

  it('eye points are the standard mesh EAR points, so EAR(adapted 68) == meshEAR(mesh)', () => {
    expect(MESH_TO_68.slice(36, 42)).toEqual([...MESH_RIGHT_EYE_EAR]);
    expect(MESH_TO_68.slice(42, 48)).toEqual([...MESH_LEFT_EYE_EAR]);
    const mesh = randomMesh();
    const W = 640, H = 480;
    const pts = meshTo68(mesh, W, H)!;
    // meshEAR with aspect W/H on normalized coords == pixel EAR
    expect(calculateEAR(pts)).toBeCloseTo(meshEAR(mesh, W / H)!, 12);
  });

  it('inner-lip points are the standard MAR points 13/14/78/308', () => {
    expect([MESH_TO_68[62], MESH_TO_68[66], MESH_TO_68[60], MESH_TO_68[64]]).toEqual([13, 14, 78, 308]);
    expect(calculateMAR(meshTo68(randomMesh(), 640, 480)!)).not.toBeNull();
  });

  it('returns null for short meshes; face box spans the mesh', () => {
    expect(meshTo68([{ x: 0, y: 0 }], 640, 480)).toBeNull();
    const box = meshFaceBox(randomMesh())!;
    expect(box.width).toBeGreaterThan(0);
    expect(box.x).toBeGreaterThan(0.3);
  });
});
