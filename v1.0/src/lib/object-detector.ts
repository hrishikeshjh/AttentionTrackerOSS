'use client';

/**
 * ObjectDetector — YOLOv8-nano ONNX wrapper for browser
 *
 * Detects study-relevant objects (book, phone, laptop, keyboard, mouse) from a
 * video frame. Uses YOLOv8-nano running via ONNX Runtime Web (WebGPU → WASM
 * fallback). Runs every 3rd frame to keep CPU usage low (ARCHITECTURE.md §2.1).
 *
 * Model:  public/models/yolov8n-objects.onnx
 *         YOLOv8-nano fine-tuned on COCO classes of interest.
 * Input:  [1, 3, 640, 640]  float32, normalized [0, 1], NCHW, LETTERBOXED
 *         (aspect ratio kept, padding value 114/255 as in Ultralytics); boxes
 *         are mapped back to the source frame. Frames used to be stretched
 *         to 640x640, which distorts objects in a 4:3 / 16:9 webcam frame.
 * Output: [1, 84, 8400]     float32  (cx cy w h + 80 COCO class scores)
 *
 * New dependency: onnxruntime-web
 *   npm install onnxruntime-web
 *   Copy WASM binaries to public/onnx/ (from node_modules/onnxruntime-web/dist/)
 *
 * Licence: YOLOv8 weights are AGPL-3.0 (DACE decision R2: research use only
 * until approved). The Apache-2.0 alternative is
 * src/lib/dace/detectors/mediapipe-object-detector.ts.
 *
 * DACE additions (all opt-in; the legacy call sequence load(url) /
 * detect(source) behaves exactly as before):
 *   - load(url, { sha256, selfTest }) verifies the model bytes with
 *     crypto.subtle before creating the session, and can run the self-test;
 *   - selfTest(): zero [1,3,640,640] tensor, asserts output0 dims [1,84,8400];
 *   - isReady(): session loaded and self-test not failed (FR4);
 *   - detect(source, { minConf, force }): confidence floor and forced cadence
 *     (the DACE engine schedules the detector itself, every 3rd tick).
 */

import type { DetectedObject, ObjectClass } from './types';

// ==================== CONSTANTS ====================

const INPUT_SIZE = 640;

/** Minimum class confidence to consider a detection valid. */
const CONFIDENCE_THRESHOLD = 0.25;

/** IoU threshold for Non-Maximum Suppression. */
const IOU_THRESHOLD = 0.45;

/** Letterbox padding value (Ultralytics default grey 114). */
const PAD_VALUE = 114 / 255;

/**
 * COCO 80-class indices (0-indexed) mapped to CognEn object classes.
 * Only these classes are extracted during post-processing.
 * Reference: https://tech.amikelive.com/node-718/
 *
 * KNOWN LIMITATION - 'pen', 'notebook' and 'tablet' are ObjectClass values
 * that this detector can NEVER emit: COCO has no pen/pencil or notebook
 * class, and a tablet is detected (if at all) as 'cell phone' or 'laptop'.
 * Consequently, with this stock-COCO model, HandObjectProximityDetector
 * never returns 'holding_pen', and the rule branches that test 'notebook'
 * only fire for the 'book' class. Producing them needs a fine-tuned model
 * (v2.0 open task: fine-tune on an Open Images subset) or the DACE pen-grip
 * evidence (scoring/dace/hand-object.ts), which infers writing from the hand
 * pose rather than from a pen box.
 */
const COCO_CLASS_MAP: Record<number, ObjectClass> = {
  63: 'laptop',
  64: 'mouse',
  66: 'keyboard',
  67: 'phone',    // "cell phone" in COCO
  73: 'book',
};

// ==================== INTERNAL TYPES ====================

interface RawDetection {
  cx: number;
  cy: number;
  w: number;
  h: number;
  classId: number;
  score: number;
}

// ==================== NMS UTILITIES ====================

function computeIoU(a: RawDetection, b: RawDetection): number {
  const ax1 = a.cx - a.w / 2;
  const ay1 = a.cy - a.h / 2;
  const ax2 = a.cx + a.w / 2;
  const ay2 = a.cy + a.h / 2;

  const bx1 = b.cx - b.w / 2;
  const by1 = b.cy - b.h / 2;
  const bx2 = b.cx + b.w / 2;
  const by2 = b.cy + b.h / 2;

  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);

  const interArea = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const aArea = a.w * a.h;
  const bArea = b.w * b.h;

  return interArea / (aArea + bArea - interArea);
}

/**
 * Class-aware NMS: suppresses overlapping boxes only within the same class.
 */
function nms(detections: RawDetection[]): RawDetection[] {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const keep: RawDetection[] = [];

  for (const det of sorted) {
    const suppressed = keep.some(
      (k) => k.classId === det.classId && computeIoU(det, k) > IOU_THRESHOLD
    );
    if (!suppressed) keep.push(det);
  }

  return keep;
}

// ==================== LETTERBOX ====================

/** Scale and padding used to fit a srcW x srcH frame into size x size. */
export type Letterbox = {
  scale: number;
  padX: number;
  padY: number;
  /** drawn content size inside the square input */
  drawW: number;
  drawH: number;
  srcW: number;
  srcH: number;
};

/**
 * Letterbox geometry: uniform scale = min(size/srcW, size/srcH), content
 * centred, padding split evenly (integer pixels). Missing or zero source
 * dimensions fall back to a square size x size source (identity).
 */
export function computeLetterbox(srcW: number, srcH: number, size: number = INPUT_SIZE): Letterbox {
  const w = srcW > 0 ? srcW : size;
  const h = srcH > 0 ? srcH : size;
  const scale = Math.min(size / w, size / h);
  const drawW = Math.round(w * scale);
  const drawH = Math.round(h * scale);
  const padX = Math.floor((size - drawW) / 2);
  const padY = Math.floor((size - drawH) / 2);
  return { scale, padX, padY, drawW, drawH, srcW: w, srcH: h };
}

/**
 * Map a box predicted in letterboxed input pixels (cx, cy, w, h) back to the
 * source frame, normalised to [0, 1] (centre x/y + width/height).
 */
export function unletterboxBox(
  cx: number,
  cy: number,
  w: number,
  h: number,
  lb: Letterbox
): { x: number; y: number; width: number; height: number } {
  return {
    x: (cx - lb.padX) / lb.scale / lb.srcW,
    y: (cy - lb.padY) / lb.scale / lb.srcH,
    width: w / lb.scale / lb.srcW,
    height: h / lb.scale / lb.srcH,
  };
}

/**
 * RGBA ImageData of the square input -> CHW float32 tensor in [0, 1]; pixels
 * outside the drawn content region are set to the padding value, so stale
 * content from a previous frame never leaks into the padding.
 */
export function imageDataToLetterboxTensor(imageData: ImageData, lb: Letterbox): Float32Array {
  const { data, width, height } = imageData;
  const numPixels = width * height;
  const tensor = new Float32Array(3 * numPixels);
  const x0 = lb.padX;
  const x1 = lb.padX + lb.drawW;
  const y0 = lb.padY;
  const y1 = lb.padY + lb.drawH;
  for (let y = 0; y < height; y++) {
    const inY = y >= y0 && y < y1;
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (inY && x >= x0 && x < x1) {
        tensor[i] = data[i * 4] / 255.0;
        tensor[numPixels + i] = data[i * 4 + 1] / 255.0;
        tensor[2 * numPixels + i] = data[i * 4 + 2] / 255.0;
      } else {
        tensor[i] = PAD_VALUE;
        tensor[numPixels + i] = PAD_VALUE;
        tensor[2 * numPixels + i] = PAD_VALUE;
      }
    }
  }
  return tensor;
}

function sourceSize(source: HTMLVideoElement | HTMLCanvasElement): { w: number; h: number } {
  const v = source as Partial<HTMLVideoElement>;
  if (typeof v.videoWidth === 'number' && v.videoWidth > 0) return { w: v.videoWidth, h: v.videoHeight ?? 0 };
  const c = source as Partial<HTMLCanvasElement>;
  return { w: typeof c.width === 'number' ? c.width : 0, h: typeof c.height === 'number' ? c.height : 0 };
}

// ==================== OBJECT DETECTOR ====================

/** Expected YOLOv8 output0 shape (cx, cy, w, h + 80 COCO scores, 8400 anchors). */
export const YOLO_OUTPUT_DIMS = Object.freeze([1, 84, 8400]);

export type DetectorSelfTest = {
  ok: boolean;
  /** milliseconds for one zero-tensor inference */
  ms: number | null;
  dims: number[] | null;
  error?: string;
};

export type DetectorLoadOptions = {
  /** expected lowercase hex SHA-256 of the model file; mismatch -> load() throws */
  sha256?: string;
  /** run selfTest() after loading */
  selfTest?: boolean;
};

export type DetectOptions = {
  /** confidence floor (replaces the default 0.25 threshold) */
  minConf?: number;
  /** run inference on this call regardless of the internal every-3rd-call cadence */
  force?: boolean;
};

/** Lowercase hex SHA-256 via Web Crypto (browser and Node 18+). */
export async function sha256Hex(bytes: ArrayBuffer | Uint8Array): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error('crypto.subtle unavailable');
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const digest = await subtle.digest('SHA-256', view as BufferSource);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

export class ObjectDetector {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private session: any = null;
  private offscreenCanvas: OffscreenCanvas | null = null;
  private offscreenCtx: OffscreenCanvasRenderingContext2D | null = null;
  private frameCounter = 0;
  private lastResults: DetectedObject[] = [];
  private selfTestResult: DetectorSelfTest | null = null;
  private checksumVerified: boolean | null = null;

  /**
   * Load the ONNX model. Must be called before detect().
   * @param modelUrl  Path relative to /public, e.g. '/models/yolov8n-objects.onnx'
   * @param opts      DACE options (checksum, self-test); omit for the legacy behaviour
   */
  async load(modelUrl: string, opts?: DetectorLoadOptions): Promise<void> {
    if (opts) {
      await this.loadVerified(modelUrl, opts);
      return;
    }
    // Use an indirect import so TS resolves this as Promise<any>.
    // Install the real package with: npm install onnxruntime-web
    // Copy WASM binaries to public/onnx/ from node_modules/onnxruntime-web/dist/
    const pkg = 'onnxruntime-web';
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ort: any = await import(/* webpackMode: "eager" */ pkg as string);

    // Point WASM runtime to /public/onnx/ binaries
    ort.env.wasm.wasmPaths = '/onnx/';

    this.session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['webgpu', 'wasm'],
    });

    this.offscreenCanvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
    this.offscreenCtx = this.offscreenCanvas.getContext(
      '2d'
    ) as OffscreenCanvasRenderingContext2D;
  }

  isLoaded(): boolean {
    return this.session !== null;
  }

  /** DACE load path: optional checksum before session creation, optional self-test after. */
  private async loadVerified(modelUrl: string, opts: DetectorLoadOptions): Promise<void> {
    const pkg = 'onnxruntime-web';
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ort: any = await import(/* webpackMode: "eager" */ pkg as string);
    ort.env.wasm.wasmPaths = '/onnx/';

    let modelSource: string | Uint8Array = modelUrl;
    if (opts.sha256) {
      const res = await fetch(modelUrl);
      if (!res.ok) throw new Error(`model fetch failed: HTTP ${res.status}`);
      const bytes = new Uint8Array(await res.arrayBuffer());
      const actual = await sha256Hex(bytes);
      this.checksumVerified = actual === opts.sha256.trim().toLowerCase();
      if (!this.checksumVerified) {
        throw new Error(`model checksum mismatch: expected ${opts.sha256}, got ${actual}`);
      }
      modelSource = bytes;
    }

    this.session = await ort.InferenceSession.create(modelSource, {
      executionProviders: ['webgpu', 'wasm'],
    });
    if (typeof OffscreenCanvas !== 'undefined') {
      this.offscreenCanvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
      this.offscreenCtx = this.offscreenCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
    }
    if (opts.selfTest) await this.selfTest();
  }

  /**
   * FR4 start-up self-test: run a zero [1,3,640,640] tensor and assert that
   * output0 has dims [1,84,8400]. The result gates isReady().
   */
  async selfTest(): Promise<DetectorSelfTest> {
    if (!this.session) {
      this.selfTestResult = { ok: false, ms: null, dims: null, error: 'session not loaded' };
      return this.selfTestResult;
    }
    const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
    try {
      const pkg = 'onnxruntime-web';
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ort: any = await import(pkg as string);
      const zero = new ort.Tensor('float32', new Float32Array(3 * INPUT_SIZE * INPUT_SIZE), [1, 3, INPUT_SIZE, INPUT_SIZE]);
      const out = await this.session.run({ images: zero });
      const dims: number[] = Array.from((out?.['output0']?.dims ?? []) as ArrayLike<number>).map(Number);
      const ms = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
      const ok = dims.length === YOLO_OUTPUT_DIMS.length && dims.every((d, i) => d === YOLO_OUTPUT_DIMS[i]);
      this.selfTestResult = ok
        ? { ok, ms, dims }
        : { ok, ms, dims, error: `output0 dims [${dims.join(',')}] != [${YOLO_OUTPUT_DIMS.join(',')}]` };
    } catch (e) {
      this.selfTestResult = { ok: false, ms: null, dims: null, error: e instanceof Error ? e.message : String(e) };
    }
    return this.selfTestResult;
  }

  getSelfTest(): DetectorSelfTest | null {
    return this.selfTestResult;
  }

  /** null = not checked, true = verified, false = mismatch */
  getChecksumVerified(): boolean | null {
    return this.checksumVerified;
  }

  /** Loaded and the self-test (if run) passed. DACE uses object evidence only when true (FR4). */
  isReady(): boolean {
    return this.session !== null && this.selfTestResult?.ok !== false;
  }

  /**
   * Detect objects in a single video / canvas frame.
   * Internally throttled: only processes every 3rd call; returns cached results
   * on skipped frames (ARCHITECTURE.md §2.1).
   */
  async detect(
    source: HTMLVideoElement | HTMLCanvasElement,
    opts?: DetectOptions
  ): Promise<DetectedObject[]> {
    this.frameCounter++;
    if (!opts?.force && this.frameCounter % 3 !== 0) return this.lastResults;

    if (!this.session || !this.offscreenCanvas || !this.offscreenCtx) {
      return [];
    }

    // ── Pre-processing (letterbox: keep aspect ratio, pad with grey) ──
    const { w: srcW, h: srcH } = sourceSize(source);
    const lb = computeLetterbox(srcW, srcH);
    this.offscreenCtx.drawImage(source, lb.padX, lb.padY, lb.drawW, lb.drawH);
    const imageData = this.offscreenCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const inputTensor = imageDataToLetterboxTensor(imageData, lb);

    // ── Inference ───────────────────────────────────────────────
    const pkg2 = 'onnxruntime-web';
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ort: any = await import(pkg2 as string);
    const ortTensor = new ort.Tensor('float32', inputTensor, [
      1,
      3,
      INPUT_SIZE,
      INPUT_SIZE,
    ]);

    const results = await this.session.run({ images: ortTensor });
    const outputData: Float32Array = results['output0'].data;
    const outputDims: number[] = results['output0'].dims;

    // ── Post-processing ─────────────────────────────────────────
    this.lastResults = this.postprocess(outputData, outputDims, opts?.minConf ?? CONFIDENCE_THRESHOLD, lb);
    return this.lastResults;
  }

  /**
   * Parse YOLOv8 ONNX output tensor → NMS-filtered DetectedObject[].
   *
   * Output layout: [1, 84, 8400]
   *   - Rows 0–3   : cx, cy, w, h  (pixel coords at INPUT_SIZE scale)
   *   - Rows 4–83  : per-class confidence scores (sigmoid already applied)
   * Boxes are mapped from letterboxed input pixels back to the source frame.
   */
  private postprocess(
    outputData: Float32Array,
    dims: number[],
    threshold: number = CONFIDENCE_THRESHOLD,
    lb: Letterbox = computeLetterbox(INPUT_SIZE, INPUT_SIZE)
  ): DetectedObject[] {
    const numPreds = dims[2]; // 8400
    const raw: RawDetection[] = [];
    const targetClassIds = Object.keys(COCO_CLASS_MAP).map(Number);

    for (let i = 0; i < numPreds; i++) {
      const cx = outputData[0 * numPreds + i];
      const cy = outputData[1 * numPreds + i];
      const w = outputData[2 * numPreds + i];
      const h = outputData[3 * numPreds + i];

      // Only scan the class indices we care about (faster than iterating all 80)
      let maxScore = 0;
      let maxClassId = -1;

      for (const classId of targetClassIds) {
        const score = outputData[(4 + classId) * numPreds + i];
        if (score > maxScore) {
          maxScore = score;
          maxClassId = classId;
        }
      }

      if (maxScore >= threshold && maxClassId >= 0) {
        raw.push({ cx, cy, w, h, classId: maxClassId, score: maxScore });
      }
    }

    return nms(raw).map((d) => ({
      class: COCO_CLASS_MAP[d.classId],
      confidence: d.score,
      bbox: unletterboxBox(d.cx, d.cy, d.w, d.h, lb),
    }));
  }
}
