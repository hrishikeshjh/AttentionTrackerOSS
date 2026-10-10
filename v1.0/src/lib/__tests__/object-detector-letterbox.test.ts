import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { computeLetterbox, imageDataToLetterboxTensor, unletterboxBox } from '../object-detector';

describe('object detector letterbox', () => {
  it('640x480 -> scale 1, 80 px bars top and bottom', () => {
    const lb = computeLetterbox(640, 480);
    expect(lb).toMatchObject({ scale: 1, padX: 0, padY: 80, drawW: 640, drawH: 480 });
  });

  it('1280x720 -> scale 0.5, 140 px bars, aspect ratio kept', () => {
    const lb = computeLetterbox(1280, 720);
    expect(lb.scale).toBe(0.5);
    expect(lb.drawW / lb.drawH).toBeCloseTo(1280 / 720, 2);
    expect(lb.padY).toBe(140);
  });

  it('missing source size falls back to identity (square)', () => {
    expect(computeLetterbox(0, 0)).toMatchObject({ scale: 1, padX: 0, padY: 0, drawW: 640, drawH: 640 });
  });

  it('round-trips a source-frame box through the letterbox', () => {
    const lb = computeLetterbox(1280, 720);
    // a box centred at (0.25, 0.75) of the source, 0.1 x 0.2 in size
    const srcCx = 0.25 * 1280, srcCy = 0.75 * 720, srcW = 0.1 * 1280, srcH = 0.2 * 720;
    const inCx = srcCx * lb.scale + lb.padX;
    const inCy = srcCy * lb.scale + lb.padY;
    const b = unletterboxBox(inCx, inCy, srcW * lb.scale, srcH * lb.scale, lb);
    expect(b.x).toBeCloseTo(0.25, 9);
    expect(b.y).toBeCloseTo(0.75, 9);
    expect(b.width).toBeCloseTo(0.1, 9);
    expect(b.height).toBeCloseTo(0.2, 9);
  });

  it('a stretched-frame bbox would be wrong: the letterboxed y differs from cy/640', () => {
    const lb = computeLetterbox(640, 480);
    const b = unletterboxBox(320, 400, 64, 64, lb);
    expect(b.y).toBeCloseTo((400 - 80) / 480, 9);
    expect(b.y).not.toBeCloseTo(400 / 640, 3);
  });

  it('tensor: content pixels normalised, padding filled with 114/255', () => {
    const size = 4;
    const lb = { scale: 1, padX: 0, padY: 1, drawW: 4, drawH: 2, srcW: 4, srcH: 2 };
    const data = new Uint8ClampedArray(size * size * 4).fill(255);
    const t = imageDataToLetterboxTensor({ data, width: size, height: size } as unknown as ImageData, lb);
    const n = size * size;
    expect(t[0]).toBeCloseTo(114 / 255, 6); // row 0 = padding
    expect(t[1 * size]).toBe(1); // row 1 = content
    expect(t[2 * size + n]).toBe(1); // row 2 G = content
    expect(t[3 * size + 2 * n]).toBeCloseTo(114 / 255, 6); // row 3 B = padding
  });
});

describe('ObjectDetector.detect letterboxes the source frame', () => {
  const draws: number[][] = [];
  beforeEach(() => {
    vi.stubGlobal('OffscreenCanvas', class {
      getContext() {
        return {
          drawImage: (_s: unknown, ...a: number[]) => draws.push(a),
          getImageData: () => ({ data: new Uint8ClampedArray(640 * 640 * 4), width: 640, height: 640 }),
        };
      }
    });
    vi.doMock('onnxruntime-web', () => {
      const N = 8400;
      const data = new Float32Array(84 * N);
      data[0] = 320; data[N] = 400; data[2 * N] = 64; data[3 * N] = 48;
      data[(4 + 73) * N] = 0.9; // COCO 73 = book
      return {
        env: { wasm: {} },
        Tensor: class { constructor(public type: string, public data: Float32Array, public dims: number[]) {} },
        InferenceSession: { create: async () => ({ run: async () => ({ output0: { data, dims: [1, 84, N] } }) }) },
      };
    });
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.doUnmock('onnxruntime-web');
    vi.resetModules();
  });

  it('draws with aspect-preserving geometry and maps boxes back to the source frame', async () => {
    const { ObjectDetector } = await import('../object-detector');
    const od = new ObjectDetector();
    await od.load('/m.onnx');
    const video = { videoWidth: 640, videoHeight: 480 } as HTMLVideoElement;
    const r = await od.detect(video, { force: true });
    expect(draws.at(-1)).toEqual([0, 80, 640, 480]);
    expect(r).toHaveLength(1);
    expect(r[0].class).toBe('book');
    expect(r[0].bbox.x).toBeCloseTo(0.5, 9);
    expect(r[0].bbox.y).toBeCloseTo(320 / 480, 9);
    expect(r[0].bbox.width).toBeCloseTo(0.1, 9);
    expect(r[0].bbox.height).toBeCloseTo(48 / 480, 9);
  });
});
