import type { Point } from 'face-api.js';

// face-api.js landmark indices
// https://github.com/justadudewhohacks/face-api.js/blob/master/src/classes/FaceLandmarks68.ts

// Calculate Euclidean distance between two points
export const distance = (p1: Point, p2: Point): number => {
  return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
};

// Eye landmarks from face-api.js 68 point model
const LEFT_EYE_LANDMARKS = [36, 37, 38, 39, 40, 41];
const RIGHT_EYE_LANDMARKS = [42, 43, 44, 45, 46, 47];
export const LEFT_IRIS_LANDMARKS = [37, 38, 40, 41]; // Using eye corners and lids to approximate iris area
export const RIGHT_IRIS_LANDMARKS = [43, 44, 46, 47]; // Using eye corners and lids to approximate iris area


// Mouth landmarks for MAR
const MOUTH_LANDMARKS = [48, 54, 51, 57]; // Left, Right, Top, Bottom of inner lip

// Head pose landmarks
const HEAD_POSE_LANDMARKS = {
  nose: 30,
  leftEyeCorner: 36, 
  rightEyeCorner: 45,
  chin: 8,
};

// Calculate Eye Aspect Ratio (EAR) for one eye
const calculateSingleEAR = (landmarks: Point[], eyeIndices: number[]): number | null => {
  const eyePoints = eyeIndices.map(i => landmarks[i]);
  if (eyePoints.some(p => p === undefined)) return null;

  const p1 = eyePoints[0], p2 = eyePoints[1], p3 = eyePoints[2], p4 = eyePoints[3], p5 = eyePoints[4], p6 = eyePoints[5];
  if (!p1 || !p2 || !p3 || !p4 || !p5 || !p6) return null;

  const verticalDist = distance(p2, p6) + distance(p3, p5);
  const horizontalDist = distance(p1, p4);

  if (horizontalDist === 0) return null;
  
  return verticalDist / (2 * horizontalDist);
};

// Calculate average EAR for both eyes
export const calculateEAR = (landmarks: Point[]): number | null => {
  if (!landmarks || landmarks.length < 68) return null;
  const leftEAR = calculateSingleEAR(landmarks, LEFT_EYE_LANDMARKS);
  const rightEAR = calculateSingleEAR(landmarks, RIGHT_EYE_LANDMARKS);
  
  if (leftEAR === null || rightEAR === null) return null;
  
  return (leftEAR + rightEAR) / 2;
};

// Calculate Mouth Aspect Ratio (MAR)
export const calculateMAR = (landmarks: Point[]): number | null => {
    if (!landmarks || landmarks.length < 68) return null;
    const mouthPoints = MOUTH_LANDMARKS.map(i => landmarks[i]);
    if (mouthPoints.some(p => p === undefined)) return null;
  
    const pLeft = mouthPoints[0], pRight = mouthPoints[1], pTop = mouthPoints[2], pBottom = mouthPoints[3];
    if (!pLeft || !pRight || !pTop || !pBottom) return null;
    
    const verticalDist = distance(pTop, pBottom);
    const horizontalDist = distance(pLeft, pRight);
    
    if (horizontalDist === 0) return null;
    
    return verticalDist / horizontalDist;
};

// Simplified head pose calculation
export const calculateHeadPose = (landmarks: Point[]): { yaw: number | null, pitch: number | null, roll: number | null } => {
  if (!landmarks || landmarks.length < 68) return { yaw: null, pitch: null, roll: null };
  
  const lmPoints = Object.values(HEAD_POSE_LANDMARKS).map(i => landmarks[i]);
  if (lmPoints.some(p => p === undefined)) {
    return { yaw: null, pitch: null, roll: null };
  }

  const [nose, leftEye, rightEye, chin] = lmPoints;
  if(!nose || !leftEye || !rightEye || !chin) return { yaw: null, pitch: null, roll: null };

  const eyeMidpoint = {
    x: (leftEye.x + rightEye.x) / 2,
    y: (leftEye.y + rightEye.y) / 2,
  };

  const eyeDist = distance(leftEye, rightEye);
  const noseChinDist = distance(nose, chin);

  if (eyeDist === 0 || noseChinDist === 0) {
      return { yaw: null, pitch: null, roll: null };
  }

  // Yaw (simplified): horizontal deviation of nose from eye midpoint
  const yaw = (nose.x - eyeMidpoint.x) / eyeDist;

  // Pitch (simplified): vertical deviation of nose from eye midpoint
  const pitch = (nose.y - eyeMidpoint.y) / noseChinDist;

  // Roll: angle of the line connecting the eyes
  const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

  return {
    yaw: yaw * 50, // scale for better range
    pitch: pitch * 50, // scale for better range
    roll: roll * (180 / Math.PI), // convert to degrees
  };
};