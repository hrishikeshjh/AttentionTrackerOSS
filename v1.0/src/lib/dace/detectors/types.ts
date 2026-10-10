/**
 * Common interface for the two DACE object detectors (decision R2):
 *   - 'yolo'      YOLOv8n ONNX (AGPL-3.0, research use pending approval)
 *   - 'mediapipe' MediaPipe Object Detector, EfficientDet-Lite0 (Apache-2.0)
 * Both map COCO classes onto the existing DetectedObject shape
 * (normalized centre boxes), so evidence code does not care which ran.
 */
import type { DetectedObject } from '../../types';
import type { FrameImage } from '../measurement/provider';

export type DetectorKind = 'yolo' | 'mediapipe';

export type DetectorStatus = {
  kind: DetectorKind;
  loaded: boolean;
  selfTestOk: boolean | null;
  selfTestMs: number | null;
  checksumVerified: boolean | null;
  error: string | null;
};

export interface DaceObjectDetector {
  readonly kind: DetectorKind;
  /** loaded AND self-test passed: only then is object evidence used (FR4) */
  isReady(): boolean;
  /** runs inference now (no internal cadence); detections below minConf are dropped */
  detect(image: FrameImage, t: number, opts: { minConf: number }): Promise<DetectedObject[]>;
  status(): DetectorStatus;
  dispose(): void;
}
