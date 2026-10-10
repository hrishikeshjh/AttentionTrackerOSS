/**
 * Measurement provider abstraction (DACE plan section 3).
 *
 * A provider turns one video frame into zero or more face candidates. The
 * candidate box is in PIXELS, top-left format, exactly what
 * PrimaryFaceSelector.select() expects, so primary-face selection is the same
 * for every provider. Each measurement carries the 68-point landmarks the
 * legacy FaceMetricsAggregator needs (face-api output, or the audited
 * mesh->68 subset for MediaPipe) plus whatever extra signals the provider
 * measures (pose in degrees, mesh, blendshapes, transformation matrix).
 *
 * Providers load their library with a dynamic import, so only the selected
 * library is downloaded (face-api.js for legacy/pnp68, @mediapipe/tasks-vision
 * for mediapipe).
 */
import type { Point2D } from '../../scoring/geometry';
import type { FaceBox, FaceCandidate } from '../../scoring/face-selection';
import type { BlendshapeMap, EulerDeg, MeshPoint } from '../../scoring/dace/types';
import type { NormalizedBBox } from '../../types';

export type MeasurementKind = 'legacy' | 'pnp68' | 'mediapipe';

/** Anything the browser libraries accept as a frame. */
export type FrameImage = HTMLVideoElement | HTMLCanvasElement | HTMLImageElement | ImageBitmap | OffscreenCanvas;

export type FrameInput = {
  image: FrameImage;
  width: number;
  height: number;
  /** monotonically increasing timestamp (ms) */
  t: number;
};

export type FaceMeasurement = {
  /** 68-point landmarks in PIXELS (legacy aggregator input) */
  landmarks68: Point2D[];
  /** face box, normalized centre format */
  faceBox: NormalizedBBox;
  /** head pose in degrees (+pitch = head down), when the provider measures it */
  pose: EulerDeg | null;
  /** 478-point normalized mesh (MediaPipe only) */
  mesh: MeshPoint[] | null;
  blendshapes: BlendshapeMap | null;
  /** 4x4 column-major facial transformation matrix (MediaPipe only) */
  matrix: number[] | null;
  /** provider-specific object used for drawing (e.g. the face-api detection) */
  raw: unknown;
};

export type ProviderSelfTest = { ok: boolean; ms: number | null; error?: string };

export interface MeasurementProvider {
  readonly kind: MeasurementKind;
  /** short human-readable label shown on the dashboard */
  readonly label: string;
  init(): Promise<void>;
  isReady(): boolean;
  measure(frame: FrameInput): Promise<Array<FaceCandidate<FaceMeasurement>>>;
  /** draw the measured face on an overlay canvas (sized to the frame) */
  draw(canvas: HTMLCanvasElement, face: FaceMeasurement): void;
  selfTest?(): Promise<ProviderSelfTest>;
  dispose(): void;
}

/** Normalized centre box from a pixel top-left box. */
export function normalizedFromPixelBox(box: FaceBox, width: number, height: number): NormalizedBBox {
  return {
    x: (box.x + box.width / 2) / width,
    y: (box.y + box.height / 2) / height,
    width: box.width / width,
    height: box.height / height,
  };
}

/** Pixel top-left box from a normalized centre box. */
export function pixelFromNormalizedBox(b: NormalizedBBox, width: number, height: number): FaceBox {
  return {
    x: (b.x - b.width / 2) * width,
    y: (b.y - b.height / 2) * height,
    width: b.width * width,
    height: b.height * height,
  };
}

export const MEASUREMENT_LABELS: Readonly<Record<MeasurementKind, string>> = Object.freeze({
  legacy: 'face-api.js 68-pt (legacy formulas)',
  pnp68: 'face-api.js 68-pt + solvePnP (A″)',
  mediapipe: 'MediaPipe Face Landmarker (478-pt, matrix pose)',
});

/** Create the provider for a measurement kind (dynamic import: only that library loads). */
export async function createMeasurementProvider(
  kind: MeasurementKind,
  opts: { pitchSign?: number } = {}
): Promise<MeasurementProvider> {
  if (kind === 'mediapipe') {
    const { FaceLandmarkerProvider } = await import('./face-landmarker');
    return new FaceLandmarkerProvider(undefined, opts.pitchSign ?? 1);
  }
  if (kind === 'pnp68') {
    const { Pnp68Provider } = await import('./pnp68');
    return new Pnp68Provider();
  }
  const { LegacyFaceApiProvider } = await import('./legacy-faceapi');
  return new LegacyFaceApiProvider();
}
