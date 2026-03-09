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
 * Input:  [1, 3, 640, 640]  float32, normalized [0, 1], NCHW
 * Output: [1, 84, 8400]     float32  (cx cy w h + 80 COCO class scores)
 *
 * New dependency: onnxruntime-web
 *   npm install onnxruntime-web
 *   Copy WASM binaries to public/onnx/ (from node_modules/onnxruntime-web/dist/)
 */

import type { DetectedObject, ObjectClass } from './types';

// ==================== CONSTANTS ====================

const INPUT_SIZE = 640;

/** Minimum class confidence to consider a detection valid. */
const CONFIDENCE_THRESHOLD = 0.25;

/** IoU threshold for Non-Maximum Suppression. */
const IOU_THRESHOLD = 0.45;

/**
 * COCO 80-class indices (0-indexed) mapped to CognEn object classes.
 * Only these classes are extracted during post-processing.
 * Reference: https://tech.amikelive.com/node-718/
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

// ==================== OBJECT DETECTOR ====================

export class ObjectDetector {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private session: any = null;
  private offscreenCanvas: OffscreenCanvas | null = null;
  private offscreenCtx: OffscreenCanvasRenderingContext2D | null = null;
  private frameCounter = 0;
  private lastResults: DetectedObject[] = [];

  /**
   * Load the ONNX model. Must be called before detect().
   * @param modelUrl  Path relative to /public, e.g. '/models/yolov8n-objects.onnx'
   */
  async load(modelUrl: string): Promise<void> {
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

  /**
   * Detect objects in a single video / canvas frame.
   * Internally throttled: only processes every 3rd call; returns cached results
   * on skipped frames (ARCHITECTURE.md §2.1).
   */
  async detect(
    source: HTMLVideoElement | HTMLCanvasElement
  ): Promise<DetectedObject[]> {
    this.frameCounter++;
    if (this.frameCounter % 3 !== 0) return this.lastResults;

    if (!this.session || !this.offscreenCanvas || !this.offscreenCtx) {
      return [];
    }

    // ── Pre-processing ──────────────────────────────────────────
    this.offscreenCtx.drawImage(source, 0, 0, INPUT_SIZE, INPUT_SIZE);
    const imageData = this.offscreenCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const inputTensor = this.imageDataToTensor(imageData);

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
    this.lastResults = this.postprocess(outputData, outputDims);
    return this.lastResults;
  }

  /** Convert RGBA ImageData → CHW float32 tensor, values in [0, 1]. */
  private imageDataToTensor(imageData: ImageData): Float32Array {
    const { data, width, height } = imageData;
    const numPixels = width * height;
    const tensor = new Float32Array(3 * numPixels);

    for (let i = 0; i < numPixels; i++) {
      tensor[i] = data[i * 4] / 255.0;                    // R channel
      tensor[numPixels + i] = data[i * 4 + 1] / 255.0;   // G channel
      tensor[2 * numPixels + i] = data[i * 4 + 2] / 255.0; // B channel
    }

    return tensor;
  }

  /**
   * Parse YOLOv8 ONNX output tensor → NMS-filtered DetectedObject[].
   *
   * Output layout: [1, 84, 8400]
   *   - Rows 0–3   : cx, cy, w, h  (pixel coords at INPUT_SIZE scale)
   *   - Rows 4–83  : per-class confidence scores (sigmoid already applied)
   */
  private postprocess(
    outputData: Float32Array,
    dims: number[]
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

      if (maxScore >= CONFIDENCE_THRESHOLD && maxClassId >= 0) {
        raw.push({ cx, cy, w, h, classId: maxClassId, score: maxScore });
      }
    }

    return nms(raw).map((d) => ({
      class: COCO_CLASS_MAP[d.classId],
      confidence: d.score,
      bbox: {
        x: d.cx / INPUT_SIZE,
        y: d.cy / INPUT_SIZE,
        width: d.w / INPUT_SIZE,
        height: d.h / INPUT_SIZE,
      },
    }));
  }
}
