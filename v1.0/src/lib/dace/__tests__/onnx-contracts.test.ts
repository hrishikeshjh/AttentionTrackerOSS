/**
 * onnxruntime-web is mocked: tests the temporal-classifier loadModel contract
 * (plan D10 / FR10) and the YOLO detector self-test / checksum / cadence (FR4).
 */
import { readFileSync } from 'node:fs';
import { describe, it, expect, vi, beforeEach } from 'vitest';

type RunFn = (feeds: Record<string, { data: Float32Array; dims: number[] }>) => Promise<Record<string, { data: ArrayLike<number>; dims: number[] }>>;

const ortState: {
  inputNames: string[];
  outputNames: string[];
  run: RunFn;
  created: unknown[];
  createThrows: boolean;
} = { inputNames: ['input'], outputNames: ['logits'], run: async () => ({}), created: [], createThrows: false };

const env = { wasm: { wasmPaths: '' } };

vi.mock('onnxruntime-web', () => {
  class Tensor {
    constructor(public type: string, public data: Float32Array, public dims: number[]) {}
  }
  return {
    env,
    Tensor,
    InferenceSession: {
      create: async (src: unknown) => {
        if (ortState.createThrows) throw new Error('no model');
        ortState.created.push(src);
        return { inputNames: ortState.inputNames, outputNames: ortState.outputNames, run: (f: never) => ortState.run(f) };
      },
    },
  };
});

import {
  MODEL_CLASSES,
  PROXIMITY_CODES_V1,
  PROXIMITY_CODES_V2,
  TemporalBehaviorClassifier,
  packWindow,
  softmax,
} from '../../temporal-classifier';
import { PROXIMITY_CODES_V1 as FV_CODES_V1, encode13 } from '../../scoring/dace/feature-vector';
import { ObjectDetector, YOLO_OUTPUT_DIMS, sha256Hex } from '../../object-detector';
import type { ExtendedFaceMetrics, HandObjectProximity } from '../../types';

function metrics(o: Partial<ExtendedFaceMetrics> = {}): ExtendedFaceMetrics {
  return {
    ear: 0.3, mar: 0.1, yaw: 2, pitch: -3, roll: 0,
    blinkDetected: false, blinkRate: 16, perclos: 5,
    gazeX: 0.05, gazeY: -0.02, faceDistance: 0.5, faceArea: 0.1,
    fatigueIndex: 10, fidgetScore: 5, headPoseStability: 90,
    eyebrowRaise: 0.3, mouthOpenness: 0.05,
    earVariance: 0.001, marVariance: 0.001, poseVariance: 1,
    ...o,
  };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

beforeEach(() => {
  ortState.inputNames = ['input'];
  ortState.outputNames = ['logits'];
  ortState.created = [];
  ortState.createThrows = false;
  ortState.run = async () => ({});
  env.wasm.wasmPaths = '';
});

describe('temporal classifier loadModel contract (D10)', () => {
  it('defaults: input "input", channels-first [1,13,15], output "logits" + softmax, v2 proximity codes, /onnx/ wasm', async () => {
    let feed: { data: Float32Array; dims: number[] } | null = null;
    ortState.run = async (f) => {
      feed = f.input;
      return { logits: { data: [0, 3, 0, 0, 0, 0], dims: [1, 6] } };
    };
    const tc = new TemporalBehaviorClassifier();
    expect(await tc.loadModel('/models/behavior.onnx')).toBe(true);
    expect(tc.isModelLoaded()).toBe(true);
    expect(env.wasm.wasmPaths).toBe('/onnx/');
    const prox: HandObjectProximity = 'holding_phone';
    for (let i = 0; i < 15; i++) tc.update({ metrics: metrics({ ear: 0.2 + i * 0.001 }), detectedObjects: [], handProximity: prox });
    await flush();
    expect(feed!.dims).toEqual([1, 13, 15]);
    // channels-first: feature f, time t at f*15 + t; ear (f=4) at t=14
    expect(feed!.data[4 * 15 + 14]).toBeCloseTo(0.214, 6);
    // proximity column re-encoded with the v2.0 code set (holding_phone = 1.0, v1 would be 0.2)
    expect(feed!.data[12 * 15 + 0]).toBe(PROXIMITY_CODES_V2.holding_phone);
    const r = tc.getLastResult();
    expect(r.label).toBe('attentive_reading');
    expect(r.source).toBe('temporal');
    expect(r.confidence).toBeCloseTo(softmax([0, 3, 0, 0, 0, 0])[1], 6);
    expect(tc.lastError).toBeNull();
  });

  it('channels_last + probabilities + v1 codes are options', async () => {
    let feed: { data: Float32Array; dims: number[] } | null = null;
    ortState.inputNames = ['x'];
    ortState.outputNames = ['probs'];
    ortState.run = async (f) => ((feed = f.x), { probs: { data: [0.1, 0.1, 0.1, 0.1, 0.5, 0.1], dims: [1, 6] } });
    const tc = new TemporalBehaviorClassifier();
    await tc.loadModel('/m.onnx', { layout: 'channels_last', inputName: 'x', outputName: 'probs', outputIsLogits: false, proximityCodes: 'v1' });
    for (let i = 0; i < 15; i++) tc.update({ metrics: metrics(), detectedObjects: [], handProximity: 'holding_phone' });
    await flush();
    expect(feed!.dims).toEqual([1, 15, 13]);
    expect(feed!.data[12]).toBeCloseTo(PROXIMITY_CODES_V1.holding_phone, 6);
    expect(tc.getLastResult()).toMatchObject({ label: 'drowsy', confidence: 0.5 });
  });

  it('featureDim 15 (v2.0 v3 model): [1,15,15], first 13 channels unchanged, posture padded per encode15_v2.json', async () => {
    const fx = JSON.parse(
      readFileSync(new URL('../../../../../shared/fixtures/encode15_v2.json', import.meta.url), 'utf8')
    ) as { feature_dim: number; v1_load_model_feature_dim_15_padding: number[] };
    const feeds: Array<{ data: Float32Array; dims: number[] }> = [];
    ortState.run = async (f) => {
      feeds.push(f.input);
      return { logits: { data: [0, 0, 0, 0, 0, 0], dims: [1, 6] } };
    };
    const run = async (featureDim?: 13 | 15) => {
      const tc = new TemporalBehaviorClassifier();
      expect(await tc.loadModel('/m.onnx', featureDim ? { featureDim } : {})).toBe(true);
      for (let i = 0; i < 15; i++) tc.update({ metrics: metrics({ ear: 0.2 + i * 0.001 }), detectedObjects: [], handProximity: 'holding_book' });
      await flush();
      return feeds[feeds.length - 1];
    };
    const f13 = await run();
    const f15 = await run(15);
    expect(f13.dims).toEqual([1, 13, 15]);
    expect(f15.dims).toEqual([1, fx.feature_dim, 15]);
    expect(Array.from(f15.data.slice(0, 13 * 15))).toEqual(Array.from(f13.data));
    for (let t = 0; t < 15; t++) {
      expect(f15.data[13 * 15 + t]).toBe(fx.v1_load_model_feature_dim_15_padding[0]);
      expect(f15.data[14 * 15 + t]).toBe(fx.v1_load_model_feature_dim_15_padding[1]);
    }
    const bad = new TemporalBehaviorClassifier();
    expect(await bad.loadModel('/m.onnx', { featureDim: 14 as never })).toBe(false);
    expect(bad.lastError).toMatch(/featureDim/);
  });

  it('wrong input/output names: load fails, lastError set, rule layer used', async () => {
    ortState.outputNames = ['output'];
    const tc = new TemporalBehaviorClassifier();
    expect(await tc.loadModel('/m.onnx')).toBe(false);
    expect(tc.isModelLoaded()).toBe(false);
    expect(tc.lastError).toMatch(/logits/);
    let r;
    for (let i = 0; i < 15; i++) r = tc.update({ metrics: metrics(), detectedObjects: [], handProximity: 'hands_not_visible' });
    expect(r!.source).toBe('temporal'); // rule-based temporal path (15 frames)
  });

  it('inference failure unloads the model and falls back to rules', async () => {
    ortState.run = async () => ({ logits: { data: [1, 2], dims: [1, 2] } }); // wrong size
    const tc = new TemporalBehaviorClassifier();
    await tc.loadModel('/m.onnx');
    for (let i = 0; i < 15; i++) tc.update({ metrics: metrics(), detectedObjects: [], handProximity: 'hands_not_visible' });
    await flush();
    expect(tc.isModelLoaded()).toBe(false);
    expect(tc.lastError).toMatch(/unexpected 'logits' size 2/);
    const plain = new TemporalBehaviorClassifier();
    let a, b;
    for (let i = 0; i < 3; i++) {
      a = tc.update({ metrics: metrics(), detectedObjects: [], handProximity: 'hands_not_visible' });
    }
    for (let i = 0; i < 18; i++) b = plain.update({ metrics: metrics(), detectedObjects: [], handProximity: 'hands_not_visible' });
    expect(a).toEqual(b);
  });

  it('missing onnxruntime model: load returns false without throwing', async () => {
    ortState.createThrows = true;
    const tc = new TemporalBehaviorClassifier();
    expect(await tc.loadModel('/missing.onnx')).toBe(false);
    expect(tc.lastError).toBe('no model');
  });

  it('class order and code sets', () => {
    expect(MODEL_CLASSES).toEqual(['attentive_screen', 'attentive_reading', 'using_phone', 'distracted', 'drowsy', 'absent']);
    expect(PROXIMITY_CODES_V1).toEqual(FV_CODES_V1);
    expect(PROXIMITY_CODES_V2).toEqual({ hands_not_visible: 0, hands_on_desk: 0.25, holding_pen: 0.5, holding_book: 0.75, holding_phone: 1 });
    const buf = [[1, 2], [3, 4], [5, 6]];
    expect(Array.from(packWindow(buf, 'channels_first'))).toEqual([1, 3, 5, 2, 4, 6]);
    expect(Array.from(packWindow(buf, 'channels_last'))).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('encode13 === extractFeatures (rule path encoding unchanged)', () => {
    const tc = new TemporalBehaviorClassifier();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const extract = (tc as any).extractFeatures.bind(tc);
    const f = { metrics: metrics({ pitch: -40, gazeY: -0.6 }), detectedObjects: [], handProximity: 'holding_pen' as const };
    expect(encode13(f)).toEqual(extract(f));
  });
});

describe('YOLO ObjectDetector DACE additions (FR4)', () => {
  beforeEach(() => {
    // node has no OffscreenCanvas; the browser path is unchanged
    vi.stubGlobal('OffscreenCanvas', class {
      getContext() {
        return { drawImage() {}, getImageData: () => ({ data: new Uint8ClampedArray(640 * 640 * 4), width: 640, height: 640 }) };
      }
    });
  });

  it('self-test: zero [1,3,640,640] tensor, output0 dims must be [1,84,8400]', async () => {
    let feed: { data: Float32Array; dims: number[] } | null = null;
    ortState.run = async (f) => ((feed = f.images), { output0: { data: new Float32Array(84 * 8400), dims: [1, 84, 8400] } });
    const od = new ObjectDetector();
    await od.load('/models/yolo.onnx', { selfTest: true });
    expect(feed!.dims).toEqual([1, 3, 640, 640]);
    expect(feed!.data.every((v) => v === 0)).toBe(true);
    expect(od.getSelfTest()).toMatchObject({ ok: true, dims: [...YOLO_OUTPUT_DIMS] });
    expect(od.isReady()).toBe(true);
  });

  it('self-test failure (wrong dims) makes isReady() false', async () => {
    ortState.run = async () => ({ output0: { data: new Float32Array(10), dims: [1, 85, 8400] } });
    const od = new ObjectDetector();
    await od.load('/m.onnx', { selfTest: true });
    expect(od.isLoaded()).toBe(true);
    expect(od.isReady()).toBe(false);
    expect(od.getSelfTest()?.error).toMatch(/1,85,8400/);
  });

  it('SHA-256 check: mismatch throws before a session is created; match loads from the verified bytes', async () => {
    const bytes = new TextEncoder().encode('fake model');
    vi.stubGlobal('fetch', async () => ({ ok: true, status: 200, arrayBuffer: async () => bytes.buffer }));
    const good = await sha256Hex(bytes);
    const od = new ObjectDetector();
    await expect(od.load('/m.onnx', { sha256: '00'.repeat(32) })).rejects.toThrow(/checksum mismatch/);
    expect(od.getChecksumVerified()).toBe(false);
    expect(ortState.created).toHaveLength(0);
    const ok = new ObjectDetector();
    await ok.load('/m.onnx', { sha256: good.toUpperCase() });
    expect(ok.getChecksumVerified()).toBe(true);
    expect(ortState.created[0]).toBeInstanceOf(Uint8Array);
    vi.unstubAllGlobals();
  });

  it('detect(): default cadence unchanged (every 3rd call); force runs every call; minConf filters', async () => {
    const N = 8400;
    const data = new Float32Array(84 * N);
    // one phone (COCO 67) box with score 0.3 at anchor 0
    data[0] = 320; data[N] = 320; data[2 * N] = 64; data[3 * N] = 64;
    data[(4 + 67) * N] = 0.3;
    let runs = 0;
    ortState.run = async () => (runs++, { output0: { data, dims: [1, 84, N] } });
    const od = new ObjectDetector();
    await od.load('/m.onnx'); // legacy load path
    const src = {} as HTMLCanvasElement;
    expect(await od.detect(src)).toEqual([]);
    expect(await od.detect(src)).toEqual([]);
    const third = await od.detect(src);
    expect(runs).toBe(1);
    expect(third).toHaveLength(1);
    expect(third[0]).toMatchObject({ class: 'phone', bbox: { x: 0.5, y: 0.5, width: 0.1, height: 0.1 } });
    expect(await od.detect(src, { force: true, minConf: 0.35 })).toEqual([]);
    expect(await od.detect(src, { force: true, minConf: 0.25 })).toHaveLength(1);
    expect(runs).toBe(3);
  });
});
