/**
 * Legacy measurement provider: face-api.js TinyFaceDetector + 68-point
 * landmarks.
 *
 * The model-loading and detection code below is moved VERBATIM from
 * src/components/dashboard/webcam-panel.tsx (same CDN URL, same detector
 * options, same resizeResults call). The only change is that face-api.js is
 * loaded with a dynamic import, so a session that selects the MediaPipe
 * provider never downloads it. With DACE off the dashboard uses
 * loadModels() / detectResized() / drawDetection() exactly where the inline
 * code used to run; legacy-provider.test.ts checks the calls and the returned
 * landmarks against the inline sequence.
 */
import type { Point2D } from '../../scoring/geometry';
import type { FaceCandidate } from '../../scoring/face-selection';
import type { FaceMeasurement, FrameInput, MeasurementProvider } from './provider';
import { MEASUREMENT_LABELS, normalizedFromPixelBox } from './provider';

type FaceApi = typeof import('face-api.js');

/** The resized face-api detection type (detection + 68 landmarks). */
export type FaceApiDetection = import('face-api.js').WithFaceLandmarks<
  { detection: import('face-api.js').FaceDetection },
  import('face-api.js').FaceLandmarks68
>;

/** Same CDN as the original inline code. */
export const FACE_API_MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/';

let faceApiPromise: Promise<FaceApi> | null = null;

/** Dynamic import of face-api.js (cached). */
export function loadFaceApi(): Promise<FaceApi> {
  if (!faceApiPromise) {
    faceApiPromise = import('face-api.js').catch((e) => {
      faceApiPromise = null;
      throw e;
    });
  }
  return faceApiPromise;
}

export class LegacyFaceApiProvider implements MeasurementProvider {
  readonly kind: MeasurementProvider['kind'] = 'legacy';
  readonly label: string = MEASUREMENT_LABELS.legacy;
  protected faceapi: FaceApi | null = null;
  private ready = false;

  /** Verbatim: load TinyFaceDetector + 68-point landmark nets from the CDN. */
  async loadModels(): Promise<void> {
    const faceapi = await loadFaceApi();
    const MODEL_URL = FACE_API_MODEL_URL;
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    ]);
    this.faceapi = faceapi;
    this.ready = true;
  }

  init(): Promise<void> {
    return this.loadModels();
  }

  isReady(): boolean {
    return this.ready;
  }

  /**
   * Verbatim: detect all faces with landmarks, then resize to the display size.
   * Returns face-api's resized detections unchanged.
   */
  async detectResized(
    video: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement,
    displaySize: { width: number; height: number }
  ): Promise<FaceApiDetection[]> {
    const faceapi = this.faceapi ?? (await loadFaceApi());
    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    return resizedDetections;
  }

  /** Verbatim: draw the 68 landmarks of one detection. */
  drawDetection(canvas: HTMLCanvasElement, detection: FaceApiDetection): void {
    this.faceapi?.draw.drawFaceLandmarks(canvas, [detection]);
  }

  /** Candidates in the same shape the inline code passed to PrimaryFaceSelector. */
  async measure(frame: FrameInput): Promise<Array<FaceCandidate<FaceMeasurement>>> {
    if (
      (typeof ImageBitmap !== 'undefined' && frame.image instanceof ImageBitmap) ||
      (typeof OffscreenCanvas !== 'undefined' && frame.image instanceof OffscreenCanvas)
    ) {
      throw new Error('legacy face-api provider needs a video, canvas or image element');
    }
    const dets = await this.detectResized(frame.image as HTMLVideoElement, { width: frame.width, height: frame.height });
    return dets.map((d) => {
      const box = d.detection.box;
      const landmarks68: Point2D[] = d.landmarks.positions.map((p) => ({ x: p.x, y: p.y }));
      const plainBox = { x: box.x, y: box.y, width: box.width, height: box.height };
      return {
        box: plainBox,
        detection: {
          landmarks68,
          faceBox: normalizedFromPixelBox(plainBox, frame.width, frame.height),
          pose: null,
          mesh: null,
          blendshapes: null,
          matrix: null,
          raw: d,
        },
      };
    });
  }

  draw(canvas: HTMLCanvasElement, face: FaceMeasurement): void {
    if (face.raw) this.drawDetection(canvas, face.raw as FaceApiDetection);
  }

  dispose(): void {
    // face-api nets are global singletons; nothing to release.
  }
}
