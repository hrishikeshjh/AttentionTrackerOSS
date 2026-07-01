import { describe, it, expect } from 'vitest';
import { PrimaryFaceSelector, type FaceCandidate } from '../face-selection';

const FRAME_W = 640;
const FRAME_H = 480;

function candidate(box: { x: number; y: number; width: number; height: number }): FaceCandidate<string> {
  return { box, detection: `${box.x},${box.y}` };
}

describe('PrimaryFaceSelector', () => {
  it('selects the largest, most-centered face when no lock exists', () => {
    const selector = new PrimaryFaceSelector();
    const small = candidate({ x: 10, y: 10, width: 50, height: 50 });
    const large = candidate({ x: 270, y: 190, width: 100, height: 100 }); // centered + larger
    const result = selector.select([small, large], FRAME_W, FRAME_H);
    expect(result).toBe(large);
  });

  it('maintains identity continuity across frames instead of flickering between near-equal candidates', () => {
    const selector = new PrimaryFaceSelector();
    const faceA = candidate({ x: 100, y: 100, width: 80, height: 80 });
    const faceB = candidate({ x: 400, y: 100, width: 82, height: 82 }); // marginally bigger

    const first = selector.select([faceA, faceB], FRAME_W, FRAME_H);
    // Same two boxes next frame — selector should stick with whichever it locked onto.
    const second = selector.select([faceA, faceB], FRAME_W, FRAME_H);
    expect(second?.detection).toBe(first?.detection);
  });

  it('unlocks after the locked face disappears for the configured number of frames', () => {
    const selector = new PrimaryFaceSelector();
    const face = candidate({ x: 270, y: 190, width: 100, height: 100 });
    selector.select([face], FRAME_W, FRAME_H);

    for (let i = 0; i < 9; i++) {
      expect(selector.select([], FRAME_W, FRAME_H)).toBeNull();
    }
    // 10th consecutive empty frame crosses the disappearance threshold.
    expect(selector.select([], FRAME_W, FRAME_H)).toBeNull();

    const newFace = candidate({ x: 50, y: 50, width: 60, height: 60 });
    expect(selector.select([newFace], FRAME_W, FRAME_H)).toBe(newFace);
  });

  it('returns null when no candidates are present and nothing was ever locked', () => {
    const selector = new PrimaryFaceSelector();
    expect(selector.select([], FRAME_W, FRAME_H)).toBeNull();
  });
});
