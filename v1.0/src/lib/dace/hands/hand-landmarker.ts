/**
 * MediaPipe Hand Landmarker (tasks-vision) for DACE desk-zone evidence E
 * (Annex A.2). Replaces the unmaintained @mediapipe/hands package in DACE
 * mode only; the legacy HandTracker (src/lib/hand-tracker.ts) is unchanged.
 *
 * VIDEO mode, up to 2 hands, GPU delegate with CPU fallback. Returns the
 * existing HandResult shape (21 normalized landmarks + handedness).
 *
 * Handedness note: MediaPipe labels handedness assuming a mirrored (selfie)
 * image. CognEn feeds the unmirrored webcam frame, so the label may be the
 * anatomical opposite. DACE uses handedness only as a stable per-hand id and
 * checks both hands symmetrically (Annex B.1, left-handed writers), so this
 * does not change any rule outcome.
 *
 * No cadence logic here: the DaceEngine staggers hands on ticks i % 3 == 1.
 */
import type { HandResult } from '../../types';
import {
  HAND_LANDMARKER_MODEL_URL,
  MonotonicClock,
  blankCanvas,
  createWithDelegateFallback,
  loadTasksVision,
  loadVisionFileset,
  nowMs,
  type Delegate,
} from '../mediapipe-runtime';
import type { FrameImage, ProviderSelfTest } from '../measurement/provider';

type HandLandmarkerT = import('@mediapipe/tasks-vision').HandLandmarker;
type HandLandmarkerResultT = import('@mediapipe/tasks-vision').HandLandmarkerResult;

export const HAND_LANDMARKER_OPTIONS = Object.freeze({
  runningMode: 'VIDEO' as const,
  numHands: 2,
});

/** Pure conversion into the existing HandResult shape (exported for tests). */
export function handResultsFromLandmarker(res: Pick<HandLandmarkerResultT, 'landmarks' | 'handedness'>): HandResult[] {
  return (res.landmarks ?? []).map((lms, i) => {
    const name = res.handedness?.[i]?.[0]?.categoryName;
    return {
      landmarks: lms.map((p) => ({ x: p.x, y: p.y, z: p.z })),
      handedness: name === 'Right' ? 'Right' : 'Left',
    };
  });
}

export interface DaceHandSource {
  isReady(): boolean;
  detect(image: FrameImage, t: number): Promise<HandResult[]>;
  selfTest?(): Promise<ProviderSelfTest>;
  dispose(): void;
}

export class MediaPipeHandLandmarker implements DaceHandSource {
  private landmarker: HandLandmarkerT | null = null;
  private clock = new MonotonicClock();
  delegate: Delegate | null = null;
  gpuError: string | null = null;

  constructor(private readonly modelUrl = HAND_LANDMARKER_MODEL_URL) {}

  async init(): Promise<void> {
    const vision = await loadTasksVision();
    const fileset = await loadVisionFileset();
    const r = await createWithDelegateFallback((delegate) =>
      vision.HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: this.modelUrl, delegate },
        ...HAND_LANDMARKER_OPTIONS,
      })
    );
    this.landmarker = r.task;
    this.delegate = r.delegate;
    this.gpuError = r.gpuError;
  }

  isReady(): boolean {
    return this.landmarker !== null;
  }

  async detect(image: FrameImage, t: number): Promise<HandResult[]> {
    if (!this.landmarker) return [];
    return handResultsFromLandmarker(this.landmarker.detectForVideo(image, this.clock.next(t)));
  }

  async selfTest(): Promise<ProviderSelfTest> {
    if (!this.landmarker) return { ok: false, ms: null, error: 'not initialised' };
    const c = blankCanvas();
    if (!c) return { ok: false, ms: null, error: 'no document' };
    const t0 = nowMs();
    try {
      this.landmarker.detectForVideo(c, this.clock.next(nowMs()));
      return { ok: true, ms: nowMs() - t0 };
    } catch (e) {
      return { ok: false, ms: nowMs() - t0, error: e instanceof Error ? e.message : String(e) };
    }
  }

  dispose(): void {
    try {
      this.landmarker?.close();
    } catch {
      /* ignore */
    }
    this.landmarker = null;
  }
}
