/** Mock providers for DaceEngine tests (no browser, no MediaPipe). */
import type { FaceCandidate } from '../../scoring/face-selection';
import type { DetectedObject, HandResult } from '../../types';
import type { FaceMeasurement, FrameInput, MeasurementProvider, MeasurementKind } from '../measurement/provider';
import { normalizedFromPixelBox } from '../measurement/provider';
import type { DaceObjectDetector, DetectorStatus } from '../detectors/types';
import type { DaceHandSource } from '../hands/hand-landmarker';
import { DEFAULT_FACE, makeFace, mulberry32 } from '../../__tests__/fixtures/synthetic-face';
import { presetConfig, type DaceConfig } from '../config';

export const W = 640;
export const H = 480;
export const DT = 200;

/** Per tick: null = no face; number = head pitch in degrees (+down). */
export type FaceScript = (i: number) => number | null;

export class MockMeasurement implements MeasurementProvider {
  readonly label = 'mock';
  calls = 0;
  throwAt = new Set<number>();
  private rnd = mulberry32(7);
  constructor(
    public readonly kind: MeasurementKind,
    private readonly script: FaceScript
  ) {}
  async init(): Promise<void> {}
  isReady(): boolean {
    return true;
  }
  async measure(frame: FrameInput): Promise<Array<FaceCandidate<FaceMeasurement>>> {
    const i = this.calls++;
    if (this.throwAt.has(i)) throw new Error(`mock measurement failure at ${i}`);
    const pitch = this.script(i);
    if (pitch === null) return [];
    const landmarks68 = makeFace({ ...DEFAULT_FACE, noise: 0 }, this.rnd);
    const box = { x: 220, y: 120, width: 200, height: 240 };
    return [
      {
        box,
        detection: {
          landmarks68,
          faceBox: normalizedFromPixelBox(box, frame.width, frame.height),
          pose: { pitch, yaw: 0, roll: 0 },
          mesh: null,
          blendshapes: null,
          matrix: null,
          raw: null,
        },
      },
    ];
  }
  draw(): void {}
  dispose(): void {}
}

export class MockDetector implements DaceObjectDetector {
  readonly kind = 'yolo' as const;
  calls: number[] = [];
  failing = false;
  constructor(
    private readonly ready: boolean,
    private readonly script: (call: number) => DetectedObject[] = () => []
  ) {}
  isReady(): boolean {
    return this.ready;
  }
  async detect(_img: unknown, t: number): Promise<DetectedObject[]> {
    this.calls.push(t);
    if (this.failing) throw new Error('detector boom');
    return this.script(this.calls.length - 1);
  }
  status(): DetectorStatus {
    return { kind: 'yolo', loaded: this.ready, selfTestOk: this.ready, selfTestMs: 1, checksumVerified: null, error: null };
  }
  dispose(): void {}
}

export class MockHands implements DaceHandSource {
  calls: number[] = [];
  constructor(
    private readonly ready: boolean,
    private readonly script: (call: number) => HandResult[] = () => []
  ) {}
  isReady(): boolean {
    return this.ready;
  }
  async detect(_img: unknown, t: number): Promise<HandResult[]> {
    this.calls.push(t);
    return this.script(this.calls.length - 1);
  }
  async selfTest() {
    return { ok: this.ready, ms: 1 };
  }
  dispose(): void {}
}

export function cfg(preset: string, o: Partial<DaceConfig> = {}): DaceConfig {
  const c = presetConfig(preset);
  if (!c) throw new Error(`no preset ${preset}`);
  return { ...c, ...o, replay: { ...c.replay, ...(o.replay ?? {}) } };
}

export const frameAt = (i: number): FrameInput => ({
  image: {} as HTMLCanvasElement,
  width: W,
  height: H,
  t: 1_000_000 + i * DT,
});
