/**
 * Pure helpers for the legacy (DACE-off) webcam loop in
 * components/dashboard/webcam-panel.tsx, kept here so they can be unit-tested
 * in node. They change WHEN the unchanged scoring functions are called, never
 * what they compute.
 */
import type { DetectedObject, ExtendedFaceMetrics, HandObjectProximity } from './types';
import { deriveStudyContext } from './temporal-classifier';
import type { BehaviorResult } from './types';

/**
 * Fixed legacy frame interval (5 FPS). The loop used to run once per
 * requestAnimationFrame (30-60 Hz, display-dependent), so the temporal
 * classifier's "15 frames = 3 s" window and the every-3rd-frame detector
 * throttles had no fixed time meaning; at 200 ms they match ARCHITECTURE.md
 * §2.3 and the 5 FPS legacy golden fixture.
 */
export const LEGACY_FRAME_INTERVAL_MS = 200;

/** No face for this long -> the user is treated as absent (Idle, scored 0 through the EMA). */
export const ABSENCE_GRACE_MS = 3000;

/** Tracks how long the face has been missing (brief detector drop-outs are ignored). */
export class AbsenceTracker {
  private since: number | null = null;

  faceSeen(): void {
    this.since = null;
  }

  /** Record a no-face frame at time t; true once the face has been missing for >= graceMs. */
  noFace(t: number, graceMs: number = ABSENCE_GRACE_MS): boolean {
    if (this.since === null) this.since = t;
    return t - this.since >= graceMs;
  }

  reset(): void {
    this.since = null;
  }
}

/**
 * Attach the context fields to the base metrics.
 *
 * `detectedObjects` is set ONLY when the object detector actually loaded.
 * MLEngine selects the UPGRADED weight set when `detectedObjects !== undefined`;
 * the panel used to always pass an array (even when the YOLO model failed to
 * load, which is the default because the model is not shipped), so the
 * upgraded weights always ran with neutral context values and the legacy
 * weights were unreachable.
 */
export function withLegacyContext(
  base: ExtendedFaceMetrics,
  ctx: {
    objectDetectorReady: boolean;
    detectedObjects: DetectedObject[];
    handObjectProximity: HandObjectProximity;
  }
): ExtendedFaceMetrics {
  return {
    ...base,
    ...(ctx.objectDetectorReady ? { detectedObjects: ctx.detectedObjects } : {}),
    handObjectProximity: ctx.handObjectProximity,
  };
}

/** Set behaviourResult and the derived studyContext on the metrics (mutates and returns them). */
export function withBehavior(m: ExtendedFaceMetrics, behavior: BehaviorResult | undefined): ExtendedFaceMetrics {
  if (behavior) m.behaviorResult = behavior;
  m.studyContext = deriveStudyContext(behavior?.label, m.handObjectProximity);
  return m;
}
