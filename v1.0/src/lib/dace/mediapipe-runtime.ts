/**
 * Shared @mediapipe/tasks-vision runtime helpers (browser only).
 *
 * WASM is served from /mediapipe/wasm (copied from node_modules by
 * scripts/setup-models.sh) and models from /models (downloaded and
 * checksum-verified by the same script). Nothing is fetched from a CDN.
 */
export const MEDIAPIPE_WASM_PATH = '/mediapipe/wasm';
export const FACE_LANDMARKER_MODEL_URL = '/models/face_landmarker.task';
export const HAND_LANDMARKER_MODEL_URL = '/models/hand_landmarker.task';
export const EFFICIENTDET_MODEL_URL = '/models/efficientdet_lite0.tflite';

type TasksVision = typeof import('@mediapipe/tasks-vision');
type WasmFileset = Awaited<ReturnType<TasksVision['FilesetResolver']['forVisionTasks']>>;

let visionPromise: Promise<TasksVision> | null = null;
let filesetPromise: Promise<WasmFileset> | null = null;

export function loadTasksVision(): Promise<TasksVision> {
  if (!visionPromise) {
    visionPromise = import('@mediapipe/tasks-vision').catch((e) => {
      visionPromise = null;
      throw e;
    });
  }
  return visionPromise;
}

export function loadVisionFileset(): Promise<WasmFileset> {
  if (!filesetPromise) {
    filesetPromise = loadTasksVision()
      .then((v) => v.FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_PATH))
      .catch((e) => {
        filesetPromise = null;
        throw e;
      });
  }
  return filesetPromise;
}

export type Delegate = 'GPU' | 'CPU';

/**
 * Create a task with the GPU delegate, falling back to CPU if GPU creation
 * fails (no WebGL2, blocked context, driver issue).
 */
export async function createWithDelegateFallback<T>(
  create: (delegate: Delegate) => Promise<T>
): Promise<{ task: T; delegate: Delegate; gpuError: string | null }> {
  try {
    return { task: await create('GPU'), delegate: 'GPU', gpuError: null };
  } catch (e) {
    const gpuError = e instanceof Error ? e.message : String(e);
    return { task: await create('CPU'), delegate: 'CPU', gpuError };
  }
}

/** tasks-vision VIDEO mode needs strictly increasing timestamps per task instance. */
export class MonotonicClock {
  private last = -Infinity;
  next(t: number): number {
    const v = Number.isFinite(t) && t > this.last ? t : this.last + 1;
    this.last = v;
    return v;
  }
}

/** Small blank canvas for start-up self-tests (browser only). */
export function blankCanvas(size = 64): HTMLCanvasElement | null {
  if (typeof document === 'undefined') return null;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');
  if (ctx) {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, size, size);
  }
  return c;
}

export const nowMs = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());
