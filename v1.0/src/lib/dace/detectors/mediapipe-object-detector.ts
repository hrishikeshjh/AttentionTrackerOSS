/**
 * Apache-2.0 object detector option for decision R2: MediaPipe Object
 * Detector with EfficientDet-Lite0 (COCO labels), via @mediapipe/tasks-vision.
 * Model: /models/efficientdet_lite0.tflite (downloaded and checksum-verified
 * by scripts/setup-models.sh). Stage 1 compares it with YOLOv8n on the pilot
 * set (expected: weaker recall on small phones; that is what R2 measures).
 *
 * COCO 'cell phone' -> 'phone', 'book' -> 'book' (the classes DACE evidence
 * uses), plus 'laptop' / 'mouse' / 'keyboard' for parity with the YOLO map.
 * Bounding boxes come back in pixels (top-left); they are converted to the
 * normalized centre boxes of DetectedObject.
 */
import type { DetectedObject, ObjectClass } from '../../types';
import {
  EFFICIENTDET_MODEL_URL,
  MonotonicClock,
  blankCanvas,
  createWithDelegateFallback,
  loadTasksVision,
  loadVisionFileset,
  nowMs,
  type Delegate,
} from '../mediapipe-runtime';
import type { FrameImage } from '../measurement/provider';
import type { DaceObjectDetector, DetectorStatus } from './types';

type ObjectDetectorT = import('@mediapipe/tasks-vision').ObjectDetector;
type DetectionT = import('@mediapipe/tasks-vision').Detection;

export const COCO_NAME_TO_CLASS: Readonly<Record<string, ObjectClass>> = Object.freeze({
  'cell phone': 'phone',
  book: 'book',
  laptop: 'laptop',
  mouse: 'mouse',
  keyboard: 'keyboard',
});

/** Pure mapping of tasks-vision detections onto DetectedObject (exported for tests). */
export function mapDetections(
  detections: ReadonlyArray<Pick<DetectionT, 'categories' | 'boundingBox'>>,
  width: number,
  height: number,
  minConf: number
): DetectedObject[] {
  const out: DetectedObject[] = [];
  for (const d of detections) {
    const bb = d.boundingBox;
    if (!bb || !(width > 0) || !(height > 0)) continue;
    // best mapped category of this detection
    let best: { cls: ObjectClass; score: number } | null = null;
    for (const c of d.categories ?? []) {
      const cls = COCO_NAME_TO_CLASS[c.categoryName];
      if (cls && c.score >= minConf && (!best || c.score > best.score)) best = { cls, score: c.score };
    }
    if (!best) continue;
    out.push({
      class: best.cls,
      confidence: best.score,
      bbox: {
        x: (bb.originX + bb.width / 2) / width,
        y: (bb.originY + bb.height / 2) / height,
        width: bb.width / width,
        height: bb.height / height,
      },
    });
  }
  return out;
}

function frameSize(image: FrameImage): { width: number; height: number } {
  const v = image as HTMLVideoElement;
  if (typeof v.videoWidth === 'number' && v.videoWidth > 0) return { width: v.videoWidth, height: v.videoHeight };
  const c = image as HTMLCanvasElement;
  return { width: c.width, height: c.height };
}

export class MediaPipeObjectDetector implements DaceObjectDetector {
  readonly kind = 'mediapipe' as const;
  private detector: ObjectDetectorT | null = null;
  private clock = new MonotonicClock();
  private selfTestOk: boolean | null = null;
  private selfTestMs: number | null = null;
  private error: string | null = null;
  delegate: Delegate | null = null;

  constructor(
    private readonly modelUrl = EFFICIENTDET_MODEL_URL,
    private readonly scoreThreshold = 0.2
  ) {}

  /** Load + blank-canvas self-test. Never throws: failures leave isReady() false. */
  static async create(opts: { modelUrl?: string; scoreThreshold?: number } = {}): Promise<MediaPipeObjectDetector> {
    const d = new MediaPipeObjectDetector(opts.modelUrl, opts.scoreThreshold);
    try {
      await d.init();
      d.runSelfTest();
    } catch (e) {
      d.error = e instanceof Error ? e.message : String(e);
    }
    return d;
  }

  async init(): Promise<void> {
    const vision = await loadTasksVision();
    const fileset = await loadVisionFileset();
    const r = await createWithDelegateFallback((delegate) =>
      vision.ObjectDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: this.modelUrl, delegate },
        runningMode: 'VIDEO',
        scoreThreshold: this.scoreThreshold,
        categoryAllowlist: Object.keys(COCO_NAME_TO_CLASS),
      })
    );
    this.detector = r.task;
    this.delegate = r.delegate;
  }

  private runSelfTest(): void {
    const c = blankCanvas();
    if (!this.detector || !c) {
      this.selfTestOk = false;
      return;
    }
    const t0 = nowMs();
    try {
      const r = this.detector.detectForVideo(c, this.clock.next(nowMs()));
      this.selfTestOk = Array.isArray(r?.detections);
    } catch (e) {
      this.selfTestOk = false;
      this.error = e instanceof Error ? e.message : String(e);
    }
    this.selfTestMs = nowMs() - t0;
  }

  isReady(): boolean {
    return this.detector !== null && this.selfTestOk === true;
  }

  async detect(image: FrameImage, t: number, opts: { minConf: number }): Promise<DetectedObject[]> {
    if (!this.detector) return [];
    const { width, height } = frameSize(image);
    const r = this.detector.detectForVideo(image, this.clock.next(t));
    return mapDetections(r.detections ?? [], width, height, opts.minConf);
  }

  status(): DetectorStatus {
    return {
      kind: this.kind,
      loaded: this.detector !== null,
      selfTestOk: this.selfTestOk,
      selfTestMs: this.selfTestMs,
      checksumVerified: null,
      error: this.error,
    };
  }

  dispose(): void {
    try {
      this.detector?.close();
    } catch {
      /* ignore */
    }
    this.detector = null;
  }
}
