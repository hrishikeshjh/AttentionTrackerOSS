/** Pure result-mapping functions of the tasks-vision wrappers (no WASM). */
import { describe, it, expect } from 'vitest';
import { candidatesFromResult } from '../measurement/face-landmarker';
import { handResultsFromLandmarker } from '../hands/hand-landmarker';
import { mapDetections } from '../detectors/mediapipe-object-detector';
import { mat4FromRotation, rotationFromEuler } from '../../scoring/dace/head-pose-matrix';
import { MESH_TO_68 } from '../../scoring/dace/mesh-adapter';
import { mulberry32 } from '../../__tests__/fixtures/synthetic-face';

function mesh(seed = 1) {
  const rnd = mulberry32(seed);
  return Array.from({ length: 478 }, () => ({ x: 0.4 + 0.2 * rnd(), y: 0.3 + 0.3 * rnd(), z: 0, visibility: 0 }));
}

describe('FaceLandmarker result -> candidates', () => {
  it('pose from the matrix (+down), 68-pt subset in pixels, pixel top-left box, blendshape map', () => {
    const m = mesh();
    const matrix = mat4FromRotation(rotationFromEuler({ pitch: 20, yaw: -10, roll: 5 }));
    const res = {
      faceLandmarks: [m],
      faceBlendshapes: [{ categories: [{ categoryName: 'eyeLookDownLeft', score: 0.7, index: 0, displayName: '' }], headIndex: 0, headName: '' }],
      facialTransformationMatrixes: [{ rows: 4, columns: 4, data: matrix }],
    };
    const [c] = candidatesFromResult(res, 640, 480);
    expect(c.detection.pose!.pitch).toBeCloseTo(20, 9);
    expect(c.detection.pose!.yaw).toBeCloseTo(-10, 9);
    expect(c.detection.landmarks68[36]).toEqual({ x: m[MESH_TO_68[36]].x * 640, y: m[MESH_TO_68[36]].y * 480 });
    expect(c.box.x).toBeCloseTo((c.detection.faceBox.x - c.detection.faceBox.width / 2) * 640, 9);
    expect(c.detection.blendshapes).toEqual({ eyeLookDownLeft: 0.7 });
    expect(c.detection.matrix).toHaveLength(16);
    const neg = candidatesFromResult(res, 640, 480, -1)[0];
    expect(neg.detection.pose!.pitch).toBeCloseTo(-20, 9);
  });

  it('missing matrix/blendshapes are tolerated; short meshes dropped', () => {
    const out = candidatesFromResult({ faceLandmarks: [mesh(), mesh().slice(0, 10)], faceBlendshapes: [], facialTransformationMatrixes: [] }, 640, 480);
    expect(out).toHaveLength(1);
    expect(out[0].detection.pose).toBeNull();
    expect(out[0].detection.blendshapes).toBeNull();
  });
});

describe('HandLandmarker result -> HandResult', () => {
  it('keeps x/y/z and maps handedness (default Left)', () => {
    const lm = Array.from({ length: 21 }, (_, i) => ({ x: i / 21, y: 0.5, z: 0.1, visibility: 0 }));
    const r = handResultsFromLandmarker({
      landmarks: [lm, lm],
      handedness: [[{ categoryName: 'Right', score: 0.9, index: 1, displayName: '' }], []],
    });
    expect(r[0].handedness).toBe('Right');
    expect(r[1].handedness).toBe('Left');
    expect(r[0].landmarks[3]).toEqual({ x: 3 / 21, y: 0.5, z: 0.1 });
  });
});

describe('MediaPipe EfficientDet detections -> DetectedObject (R2)', () => {
  it("maps 'cell phone' -> phone and 'book' -> book with normalized centre boxes; drops others and < minConf", () => {
    const det = (name: string, score: number) => ({
      categories: [{ categoryName: name, score, index: 0, displayName: '' }],
      boundingBox: { originX: 64, originY: 48, width: 128, height: 96, angle: 0 },
    });
    const out = mapDetections([det('cell phone', 0.6), det('book', 0.5), det('person', 0.9), det('book', 0.2)], 640, 480, 0.35);
    expect(out.map((o) => o.class)).toEqual(['phone', 'book']);
    expect(out[0]).toEqual({ class: 'phone', confidence: 0.6, bbox: { x: 0.2, y: 0.2, width: 0.2, height: 0.2 } });
  });
});
