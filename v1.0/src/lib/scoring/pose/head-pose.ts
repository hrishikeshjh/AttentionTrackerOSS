import type { Point2D } from '../geometry';
import { distance } from '../geometry';
import { HEAD_POSE_LANDMARKS, HEAD_POSE_CALIBRATION } from '../constants';

export type HeadPose = { yaw: number | null; pitch: number | null; roll: number | null };

/**
 * Head-pose estimation from 68-point landmarks.
 *
 * Both yaw and pitch use the SAME normalizer (interocular distance) rather
 * than the old formula's yaw/eyeDist + pitch/noseChinDist split. Rationale:
 * nose-chin distance foreshortens as the head pitches (perspective), which
 * made the old pitch signal bias against itself at extreme angles — using
 * interocular distance for both keeps the normalizer stable as pitch varies,
 * since eye-to-eye distance is largely invariant to head pitch (roll and yaw
 * do affect it somewhat, but far less than nose-chin distance is affected by
 * pitch). This does not require camera calibration or a 3D model — it is a
 * geometric-ratio fix, not a full solvePnP solution (tracked separately as a
 * stretch goal).
 *
 * Gain constants (HEAD_POSE_CALIBRATION) are chosen to keep output magnitude
 * compatible with existing downstream thresholds (e.g. the "pitch < -15"
 * branch in ml-engine.ts's context-aware scoring) rather than representing
 * true degrees — this is a known limitation. A future solvePnP-based upgrade
 * would replace HEAD_POSE_CALIBRATION with a real camera-intrinsics-derived
 * projection.
 */
export const calculateHeadPose = (landmarks: Point2D[]): HeadPose => {
  if (!landmarks || landmarks.length < 68) return { yaw: null, pitch: null, roll: null };

  const nose = landmarks[HEAD_POSE_LANDMARKS.nose];
  const leftEye = landmarks[HEAD_POSE_LANDMARKS.leftEyeCorner];
  const rightEye = landmarks[HEAD_POSE_LANDMARKS.rightEyeCorner];
  const chin = landmarks[HEAD_POSE_LANDMARKS.chin];
  if (!nose || !leftEye || !rightEye || !chin) return { yaw: null, pitch: null, roll: null };

  const eyeMidpoint = { x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2 };
  const eyeDist = distance(leftEye, rightEye);
  if (eyeDist === 0) return { yaw: null, pitch: null, roll: null };

  const yawRatio = (nose.x - eyeMidpoint.x) / eyeDist;
  const pitchRatio = (nose.y - eyeMidpoint.y) / eyeDist; // same normalizer as yaw
  const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

  return {
    yaw: yawRatio * HEAD_POSE_CALIBRATION.YAW_GAIN,
    pitch: pitchRatio * HEAD_POSE_CALIBRATION.PITCH_GAIN,
    roll: roll * (180 / Math.PI),
  };
};
