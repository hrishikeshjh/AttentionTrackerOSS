/**
 * 13-dim per-frame feature vector, identical to
 * TemporalBehaviorClassifier.extractFeatures() (temporal-classifier.ts).
 * Kept here as a pure function so replay/offline tooling and the Stage 2
 * learned layer can encode windows without the browser classifier. A test
 * asserts equality with the classifier on randomized inputs.
 *
 * NOTE: the v1.0 proximity codes below differ from v2.0's
 * (hands_not_visible 0, hands_on_desk 0.25, holding_pen 0.5, holding_book
 * 0.75, holding_phone 1.0). A v2.0-trained model therefore cannot be loaded
 * as-is (plan D10); harmonisation is a Stage 2 task.
 */
import type { DetectedObject, ExtendedFaceMetrics, HandObjectProximity } from '../../types';
import { clamp } from '../geometry';

export const FEATURE_NAMES_13 = Object.freeze([
  'gaze_x', 'gaze_y', 'pitch_norm', 'yaw_norm', 'ear', 'perclos',
  'head_pose_stability', 'obj_book', 'obj_phone', 'obj_laptop', 'obj_none',
  'fidget_score', 'hand_proximity_code',
] as const);

export const PROXIMITY_CODES_V1: Readonly<Record<HandObjectProximity, number>> = Object.freeze({
  holding_book: 1.0,
  holding_pen: 0.8,
  hands_on_desk: 0.5,
  holding_phone: 0.2,
  hands_not_visible: 0.0,
});

export type Encode13Input = {
  metrics: Pick<ExtendedFaceMetrics, 'gazeX' | 'gazeY' | 'pitch' | 'yaw' | 'ear' | 'perclos' | 'headPoseStability' | 'fidgetScore'>;
  detectedObjects: readonly DetectedObject[];
  handProximity: HandObjectProximity;
};

export function encode13(frame: Encode13Input): number[] {
  const { metrics, detectedObjects, handProximity } = frame;
  const hasBook = detectedObjects.some((o) => o.class === 'book' || o.class === 'notebook') ? 1 : 0;
  const hasPhone = detectedObjects.some((o) => o.class === 'phone') ? 1 : 0;
  const hasLaptop = detectedObjects.some((o) => o.class === 'laptop') ? 1 : 0;
  const hasNone = !hasBook && !hasPhone && !hasLaptop ? 1 : 0;
  return [
    clamp(metrics.gazeX ?? 0, -1, 1),
    clamp(metrics.gazeY ?? 0, -1, 1),
    clamp((metrics.pitch ?? 0) / 90, -1, 1),
    clamp((metrics.yaw ?? 0) / 90, -1, 1),
    clamp(metrics.ear ?? 0.3, 0, 1),
    clamp(metrics.perclos / 100, 0, 1),
    clamp(metrics.headPoseStability / 100, 0, 1),
    hasBook,
    hasPhone,
    hasLaptop,
    hasNone,
    clamp(metrics.fidgetScore / 100, 0, 1),
    PROXIMITY_CODES_V1[handProximity] ?? 0,
  ];
}
