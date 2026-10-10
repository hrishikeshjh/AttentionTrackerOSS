/**
 * DACE adapter around the existing YOLOv8n ObjectDetector
 * (src/lib/object-detector.ts): checksum-verified load, FR4 self-test, forced
 * cadence (the engine schedules it every 3rd tick) and the c_min floor.
 *
 * Licence: YOLOv8 is AGPL-3.0 (decision R2). Research use only until the
 * supervisors approve it or a commercial licence is confirmed.
 */
import type { DetectedObject } from '../../types';
import type { ObjectDetector } from '../../object-detector';
import type { FrameImage } from '../measurement/provider';
import type { DaceObjectDetector, DetectorStatus } from './types';

export const YOLO_MODEL_URL = '/models/yolov8n-objects.onnx';

export class YoloDaceDetector implements DaceObjectDetector {
  readonly kind = 'yolo' as const;
  private error: string | null = null;

  constructor(private readonly od: ObjectDetector) {}

  /** Load with optional checksum, then self-test. Never throws: failures leave isReady() false. */
  static async create(opts: { modelUrl?: string; sha256?: string } = {}): Promise<YoloDaceDetector> {
    const { ObjectDetector: OD } = await import('../../object-detector');
    const det = new YoloDaceDetector(new OD());
    try {
      await det.od.load(opts.modelUrl ?? YOLO_MODEL_URL, { sha256: opts.sha256, selfTest: true });
      const st = det.od.getSelfTest();
      if (st && !st.ok) det.error = st.error ?? 'self-test failed';
    } catch (e) {
      det.error = e instanceof Error ? e.message : String(e);
    }
    return det;
  }

  isReady(): boolean {
    return this.od.isReady() && this.od.getSelfTest()?.ok === true;
  }

  async detect(image: FrameImage, _t: number, opts: { minConf: number }): Promise<DetectedObject[]> {
    if (!this.isReady()) return [];
    return this.od.detect(image as HTMLVideoElement | HTMLCanvasElement, { force: true, minConf: opts.minConf });
  }

  status(): DetectorStatus {
    const st = this.od.getSelfTest();
    return {
      kind: this.kind,
      loaded: this.od.isLoaded(),
      selfTestOk: st ? st.ok : null,
      selfTestMs: st?.ms ?? null,
      checksumVerified: this.od.getChecksumVerified(),
      error: this.error,
    };
  }

  dispose(): void {
    /* onnxruntime sessions are released with the page */
  }
}
