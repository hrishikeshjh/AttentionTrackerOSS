import type { Point2D } from './geometry';
import { distance } from './geometry';
import { LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS, MOUTH_LANDMARKS } from './constants';

const calculateSingleEAR = (landmarks: Point2D[], eyeIndices: number[]): number | null => {
  const eyePoints = eyeIndices.map((i) => landmarks[i]);
  if (eyePoints.some((p) => p === undefined)) return null;
  const [p1, p2, p3, p4, p5, p6] = eyePoints;
  if (!p1 || !p2 || !p3 || !p4 || !p5 || !p6) return null;

  const verticalDist = distance(p2, p6) + distance(p3, p5);
  const horizontalDist = distance(p1, p4);
  if (horizontalDist === 0) return null;

  return verticalDist / (2 * horizontalDist);
};

export const calculateEAR = (landmarks: Point2D[]): number | null => {
  if (!landmarks || landmarks.length < 68) return null;
  const leftEAR = calculateSingleEAR(landmarks, LEFT_EYE_LANDMARKS);
  const rightEAR = calculateSingleEAR(landmarks, RIGHT_EYE_LANDMARKS);
  if (leftEAR === null || rightEAR === null) return null;
  return (leftEAR + rightEAR) / 2;
};

export const calculateMAR = (landmarks: Point2D[]): number | null => {
  if (!landmarks || landmarks.length < 68) return null;
  const mouthPoints = MOUTH_LANDMARKS.map((i) => landmarks[i]);
  if (mouthPoints.some((p) => p === undefined)) return null;
  const [pLeft, pRight, pTop, pBottom] = mouthPoints;
  if (!pLeft || !pRight || !pTop || !pBottom) return null;

  const verticalDist = distance(pTop, pBottom);
  const horizontalDist = distance(pLeft, pRight);
  if (horizontalDist === 0) return null;

  return verticalDist / horizontalDist;
};
