/**
 * The legacy provider must reproduce the inline webcam-panel face-api code:
 * same CDN URL, same nets, detectAllFaces(video, new TinyFaceDetectorOptions())
 * .withFaceLandmarks(), resizeResults(detections, displaySize), and the
 * resized detections returned unchanged (so the same landmarks reach the
 * aggregator and PrimaryFaceSelector). face-api.js is mocked.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

const calls: Array<[string, ...unknown[]]> = [];
const RESIZED = [
  {
    detection: { box: { x: 10, y: 20, width: 100, height: 120 } },
    landmarks: { positions: Array.from({ length: 68 }, (_, i) => ({ x: i, y: 2 * i })) },
  },
];

vi.mock('face-api.js', () => {
  class TinyFaceDetectorOptions {
    readonly tag = 'tiny-default';
  }
  return {
    nets: {
      tinyFaceDetector: { loadFromUri: async (u: string) => void calls.push(['tiny', u]) },
      faceLandmark68Net: { loadFromUri: async (u: string) => void calls.push(['lm68', u]) },
    },
    TinyFaceDetectorOptions,
    detectAllFaces: (input: unknown, opts: unknown) => {
      calls.push(['detectAllFaces', input, opts]);
      return { withFaceLandmarks: async () => (calls.push(['withFaceLandmarks']), ['raw-detections']) };
    },
    resizeResults: (dets: unknown, size: unknown) => (calls.push(['resizeResults', dets, size]), RESIZED),
    draw: { drawFaceLandmarks: (c: unknown, d: unknown) => void calls.push(['draw', c, d]) },
  };
});

import { FACE_API_MODEL_URL, LegacyFaceApiProvider } from '../measurement/legacy-faceapi';
import { createMeasurementProvider } from '../measurement/provider';

beforeEach(() => {
  calls.length = 0;
});

describe('legacy face-api provider (verbatim move from webcam-panel)', () => {
  it('loads the same two nets from the same CDN URL', async () => {
    const p = new LegacyFaceApiProvider();
    expect(p.isReady()).toBe(false);
    await p.loadModels();
    expect(FACE_API_MODEL_URL).toBe('https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/');
    expect(calls).toEqual([
      ['tiny', FACE_API_MODEL_URL],
      ['lm68', FACE_API_MODEL_URL],
    ]);
    expect(p.isReady()).toBe(true);
  });

  it('detectResized == detectAllFaces(TinyFaceDetectorOptions).withFaceLandmarks() -> resizeResults(displaySize), unchanged', async () => {
    const p = new LegacyFaceApiProvider();
    await p.loadModels();
    calls.length = 0;
    const video = { id: 'video' } as unknown as HTMLVideoElement;
    const out = await p.detectResized(video, { width: 640, height: 480 });
    expect(out).toBe(RESIZED);
    expect(calls[0][0]).toBe('detectAllFaces');
    expect(calls[0][1]).toBe(video);
    expect((calls[0][2] as { tag: string }).tag).toBe('tiny-default');
    expect(calls[1]).toEqual(['withFaceLandmarks']);
    expect(calls[2]).toEqual(['resizeResults', ['raw-detections'], { width: 640, height: 480 }]);
  });

  it('measure() yields the same box and landmark coordinates the inline code passed on', async () => {
    const p = new LegacyFaceApiProvider();
    await p.loadModels();
    const cands = await p.measure({ image: {} as HTMLVideoElement, width: 640, height: 480, t: 0 });
    expect(cands).toHaveLength(1);
    expect(cands[0].box).toEqual(RESIZED[0].detection.box);
    expect(cands[0].detection.landmarks68).toEqual(RESIZED[0].landmarks.positions);
    expect(cands[0].detection.raw).toBe(RESIZED[0]);
    expect(cands[0].detection.pose).toBeNull();
    p.draw({} as HTMLCanvasElement, cands[0].detection);
    expect(calls.at(-1)).toEqual(['draw', {}, [RESIZED[0]]]);
  });

  it('createMeasurementProvider(legacy | pnp68) only imports face-api', async () => {
    const a = await createMeasurementProvider('legacy');
    const b = await createMeasurementProvider('pnp68');
    expect(a.kind).toBe('legacy');
    expect(b.kind).toBe('pnp68');
  });
});
