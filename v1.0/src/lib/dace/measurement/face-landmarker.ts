/**
 * MediaPipe Face Landmarker measurement provider (DACE measurement layer M,
 * Annex A.1, decision R1 — behind dace.measurement = 'mediapipe').
 *
 * VIDEO running mode, up to 2 faces (the existing PrimaryFaceSelector keeps
 * the original student, Annex B.1), blendshapes and facial transformation
 * matrixes enabled, GPU delegate with CPU fallback. WASM from
 * /mediapipe/wasm, model from /models/face_landmarker.task.
 *
 * Each face yields:
 *   - the 478-point normalized mesh (iris included);
 *   - pose in degrees from the transformation matrix (+pitch = head down);
 *   - blendshape scores (eye gaze);
 *   - the audited 68-point subset in pixels for the legacy aggregator
 *     (MAR, blink/fatigue trackers, face area, ...).
 */
import { blendshapeMap } from '../../scoring/dace/gaze-blendshape';
import { eulerFromMatrix } from '../../scoring/dace/head-pose-matrix';
import { meshFaceBox, meshTo68 } from '../../scoring/dace/mesh-adapter';
import type { FaceCandidate } from '../../scoring/face-selection';
import type { BlendshapeMap, MeshPoint } from '../../scoring/dace/types';
import {
  FACE_LANDMARKER_MODEL_URL,
  MonotonicClock,
  blankCanvas,
  createWithDelegateFallback,
  loadTasksVision,
  loadVisionFileset,
  nowMs,
  type Delegate,
} from '../mediapipe-runtime';
import type { FaceMeasurement, FrameInput, MeasurementProvider, ProviderSelfTest } from './provider';
import { MEASUREMENT_LABELS, pixelFromNormalizedBox } from './provider';

type FaceLandmarkerT = import('@mediapipe/tasks-vision').FaceLandmarker;
type FaceLandmarkerResultT = import('@mediapipe/tasks-vision').FaceLandmarkerResult;

export const FACE_LANDMARKER_OPTIONS = Object.freeze({
  runningMode: 'VIDEO' as const,
  numFaces: 2,
  outputFaceBlendshapes: true,
  outputFacialTransformationMatrixes: true,
});

/** Pure conversion of one FaceLandmarker result into candidates (exported for tests). */
export function candidatesFromResult(
  res: Pick<FaceLandmarkerResultT, 'faceLandmarks' | 'faceBlendshapes' | 'facialTransformationMatrixes'>,
  width: number,
  height: number,
  pitchSign = 1
): Array<FaceCandidate<FaceMeasurement>> {
  const out: Array<FaceCandidate<FaceMeasurement>> = [];
  (res.faceLandmarks ?? []).forEach((lm, i) => {
    const mesh: MeshPoint[] = lm.map((p) => ({ x: p.x, y: p.y, z: p.z }));
    const landmarks68 = meshTo68(mesh, width, height);
    const faceBox = meshFaceBox(mesh);
    if (!landmarks68 || !faceBox) return;
    const cats = res.faceBlendshapes?.[i]?.categories;
    const blendshapes: BlendshapeMap | null = cats ? blendshapeMap(cats) : null;
    const m = res.facialTransformationMatrixes?.[i];
    const matrix = m && Array.isArray(m.data) && m.data.length >= 16 ? [...m.data] : null;
    out.push({
      box: pixelFromNormalizedBox(faceBox, width, height),
      detection: {
        landmarks68,
        faceBox,
        pose: matrix ? eulerFromMatrix(matrix, pitchSign) : null,
        mesh,
        blendshapes,
        matrix,
        raw: null,
      },
    });
  });
  return out;
}

export class FaceLandmarkerProvider implements MeasurementProvider {
  readonly kind = 'mediapipe' as const;
  readonly label = MEASUREMENT_LABELS.mediapipe;
  private landmarker: FaceLandmarkerT | null = null;
  private clock = new MonotonicClock();
  delegate: Delegate | null = null;
  gpuError: string | null = null;

  constructor(
    private readonly modelUrl = FACE_LANDMARKER_MODEL_URL,
    private readonly pitchSign = 1
  ) {}

  async init(): Promise<void> {
    const vision = await loadTasksVision();
    const fileset = await loadVisionFileset();
    const r = await createWithDelegateFallback((delegate) =>
      vision.FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: this.modelUrl, delegate },
        ...FACE_LANDMARKER_OPTIONS,
      })
    );
    this.landmarker = r.task;
    this.delegate = r.delegate;
    this.gpuError = r.gpuError;
  }

  isReady(): boolean {
    return this.landmarker !== null;
  }

  async measure(frame: FrameInput): Promise<Array<FaceCandidate<FaceMeasurement>>> {
    if (!this.landmarker) throw new Error('FaceLandmarker not initialised');
    const res = this.landmarker.detectForVideo(frame.image, this.clock.next(frame.t));
    return candidatesFromResult(res, frame.width, frame.height, this.pitchSign);
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

  draw(canvas: HTMLCanvasElement, face: FaceMeasurement): void {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    // Draw the 68-point subset as dots (same visual density as the legacy overlay).
    ctx.fillStyle = 'rgba(0, 255, 153, 0.9)';
    for (const p of face.landmarks68) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 1.5, 0, 2 * Math.PI);
      ctx.fill();
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
