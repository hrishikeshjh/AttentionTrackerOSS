import type { Point2D } from '../geometry';
import { distance, clamp } from '../geometry';

export type GazeDirection = { gazeX: number | null; gazeY: number | null };

/**
 * TODO(gaze-v2): this approximates iris position as the centroid of eyelid
 * landmarks (face-api.js's 68-point model has no dedicated iris landmarks),
 * so it cannot detect genuine eyes-only glances without head movement. A
 * real fix means migrating to MediaPipe FaceMesh with iris refinement
 * (landmark indices 468-477), which is a separate ~2-3MB WASM+model payload
 * and a bigger change than a correctness pass — tracked as a follow-up, not
 * silently folded in here. This unifies what was previously two diverged
 * copies (web app used a 2x multiplier, the Chrome extension used 3x) into
 * one canonical formula.
 */
export const calculateGazeDirection = (landmarks: Point2D[]): GazeDirection => {
  if (!landmarks || landmarks.length < 68) return { gazeX: null, gazeY: null };

  const leftEyeLeft = landmarks[36];
  const leftEyeRight = landmarks[39];
  const rightEyeLeft = landmarks[42];
  const rightEyeRight = landmarks[45];
  if (!leftEyeLeft || !leftEyeRight || !rightEyeLeft || !rightEyeRight) {
    return { gazeX: null, gazeY: null };
  }

  const leftIrisCenter = {
    x: (landmarks[37].x + landmarks[38].x + landmarks[40].x + landmarks[41].x) / 4,
    y: (landmarks[37].y + landmarks[38].y + landmarks[40].y + landmarks[41].y) / 4,
  };
  const rightIrisCenter = {
    x: (landmarks[43].x + landmarks[44].x + landmarks[46].x + landmarks[47].x) / 4,
    y: (landmarks[43].y + landmarks[44].y + landmarks[46].y + landmarks[47].y) / 4,
  };

  const leftEyeWidth = distance(leftEyeLeft, leftEyeRight);
  const rightEyeWidth = distance(rightEyeLeft, rightEyeRight);
  if (leftEyeWidth === 0 || rightEyeWidth === 0) return { gazeX: null, gazeY: null };

  // Normalize iris position within eye to [-1, 1].
  const leftGazeX = ((leftIrisCenter.x - leftEyeLeft.x) / leftEyeWidth - 0.5) * 2;
  const rightGazeX = ((rightIrisCenter.x - rightEyeLeft.x) / rightEyeWidth - 0.5) * 2;

  const leftEyeCenterY = (leftEyeLeft.y + leftEyeRight.y + landmarks[37].y + landmarks[41].y) / 4;
  const rightEyeCenterY = (rightEyeLeft.y + rightEyeRight.y + landmarks[43].y + landmarks[47].y) / 4;
  const leftEyeHeight = Math.abs(landmarks[37].y - landmarks[41].y);
  const rightEyeHeight = Math.abs(landmarks[43].y - landmarks[47].y);

  const leftGazeY = leftEyeHeight > 0 ? (leftEyeCenterY - leftIrisCenter.y) / leftEyeHeight : 0;
  const rightGazeY = rightEyeHeight > 0 ? (rightEyeCenterY - rightIrisCenter.y) / rightEyeHeight : 0;

  return {
    gazeX: clamp((leftGazeX + rightGazeX) / 2, -1, 1),
    gazeY: clamp((leftGazeY + rightGazeY) / 2, -1, 1),
  };
};
