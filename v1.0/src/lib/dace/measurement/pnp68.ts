/**
 * Baseline A'' measurement provider: the legacy face-api 68 landmarks with
 * head pose from solvePnP (scoring/dace/solvepnp.ts) in true degrees,
 * +pitch = head down. No iris points, so gaze is NOT fixed by this provider
 * (Annex D.5: "does a minimal fix suffice?").
 */
import { solvePnP68 } from '../../scoring/dace/solvepnp';
import type { FaceCandidate } from '../../scoring/face-selection';
import { LegacyFaceApiProvider } from './legacy-faceapi';
import type { FaceMeasurement, FrameInput, MeasurementProvider } from './provider';
import { MEASUREMENT_LABELS } from './provider';

/** Pose for one measurement from its 68 points (pure; exported for tests). */
export function withPnpPose(m: FaceMeasurement, width: number, height: number): FaceMeasurement {
  const r = solvePnP68(m.landmarks68, width, height);
  return { ...m, pose: r ? r.euler : null };
}

export class Pnp68Provider extends LegacyFaceApiProvider implements MeasurementProvider {
  override readonly kind: MeasurementProvider['kind'] = 'pnp68';
  override readonly label: string = MEASUREMENT_LABELS.pnp68;

  override async measure(frame: FrameInput): Promise<Array<FaceCandidate<FaceMeasurement>>> {
    const cands = await super.measure(frame);
    return cands.map((c) => ({ box: c.box, detection: withPnpPose(c.detection, frame.width, frame.height) }));
  }
}
