import type { Point2D } from './geometry';
import { distance, clamp } from './geometry';
import { LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS, LEFT_EYEBROW, RIGHT_EYEBROW } from './constants';

export const estimateFaceDistance = (landmarks: Point2D[], frameWidth: number): number | null => {
  if (!landmarks || landmarks.length < 68) return null;
  const leftEye = landmarks[36];
  const rightEye = landmarks[45];
  if (!leftEye || !rightEye) return null;

  const ipd = distance(leftEye, rightEye);
  const normalizedIPD = ipd / frameWidth;
  return clamp((normalizedIPD - 0.1) / 0.3, 0, 1);
};

export const calculateFaceArea = (landmarks: Point2D[], frameWidth: number, frameHeight: number): number | null => {
  if (!landmarks || landmarks.length < 68) return null;
  const xs = landmarks.map((p) => p.x);
  const ys = landmarks.map((p) => p.y);
  const faceArea = (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
  return faceArea / (frameWidth * frameHeight);
};

export const calculateEyebrowRaise = (landmarks: Point2D[]): number | null => {
  if (!landmarks || landmarks.length < 68) return null;

  const leftEyebrowY = LEFT_EYEBROW.map((i) => landmarks[i]?.y).filter((y) => y !== undefined) as number[];
  const rightEyebrowY = RIGHT_EYEBROW.map((i) => landmarks[i]?.y).filter((y) => y !== undefined) as number[];
  if (leftEyebrowY.length === 0 || rightEyebrowY.length === 0) return null;
  const avgEyebrowY = [...leftEyebrowY, ...rightEyebrowY].reduce((a, b) => a + b, 0) / (leftEyebrowY.length + rightEyebrowY.length);

  const leftEyeY = LEFT_EYE_LANDMARKS.map((i) => landmarks[i]?.y).filter((y) => y !== undefined) as number[];
  const rightEyeY = RIGHT_EYE_LANDMARKS.map((i) => landmarks[i]?.y).filter((y) => y !== undefined) as number[];
  if (leftEyeY.length === 0 || rightEyeY.length === 0) return null;
  const avgEyeY = [...leftEyeY, ...rightEyeY].reduce((a, b) => a + b, 0) / (leftEyeY.length + rightEyeY.length);

  const eyeWidth = Math.abs(landmarks[36].x - landmarks[45].x);
  if (eyeWidth === 0) return null;

  const eyebrowEyeDistance = (avgEyeY - avgEyebrowY) / eyeWidth;
  return clamp((eyebrowEyeDistance - 0.2) / 0.3, 0, 1);
};

export const calculateMouthOpenness = (landmarks: Point2D[]): number | null => {
  if (!landmarks || landmarks.length < 68) return null;
  const topLip = landmarks[62];
  const bottomLip = landmarks[66];
  const leftCorner = landmarks[60];
  const rightCorner = landmarks[64];
  if (!topLip || !bottomLip || !leftCorner || !rightCorner) return null;

  const mouthHeight = distance(topLip, bottomLip);
  const mouthWidth = distance(leftCorner, rightCorner);
  if (mouthWidth === 0) return null;

  return clamp(mouthHeight / mouthWidth, 0, 1);
};
