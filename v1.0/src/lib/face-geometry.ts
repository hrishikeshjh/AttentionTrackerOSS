import type { Point } from 'face-api.js';
import type {
  ExtendedFaceMetrics,
  FaceMetrics,
  DetectedObject,
  HandResult,
  HandObjectProximity,
} from './types';
import { HandTracker } from './hand-tracker';

// face-api.js landmark indices
// https://github.com/justadudewhohacks/face-api.js/blob/master/src/classes/FaceLandmarks68.ts

// ==================== CONSTANTS ====================

// Eye landmarks from face-api.js 68 point model
const LEFT_EYE_LANDMARKS = [36, 37, 38, 39, 40, 41];
const RIGHT_EYE_LANDMARKS = [42, 43, 44, 45, 46, 47];
export const LEFT_IRIS_LANDMARKS = [37, 38, 40, 41];
export const RIGHT_IRIS_LANDMARKS = [43, 44, 46, 47];

// Mouth landmarks
const MOUTH_OUTER = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59];
const MOUTH_INNER = [60, 61, 62, 63, 64, 65, 66, 67];
const MOUTH_LANDMARKS = [48, 54, 51, 57]; // Left, Right, Top, Bottom

// Eyebrow landmarks
const LEFT_EYEBROW = [17, 18, 19, 20, 21];
const RIGHT_EYEBROW = [22, 23, 24, 25, 26];

// Head pose landmarks
const HEAD_POSE_LANDMARKS = {
  nose: 30,
  leftEyeCorner: 36,
  rightEyeCorner: 45,
  chin: 8,
};

// Face boundary landmarks for distance estimation
const FACE_BOUNDARY = {
  leftCheek: 0,
  rightCheek: 16,
  topHead: 27, // nose bridge top
  chin: 8,
};

// Thresholds for ML calculations
const EAR_BLINK_THRESHOLD = 0.21; // EAR below this = eye closed
const PERCLOS_WINDOW_MS = 60000; // 1 minute window for PERCLOS
const NORMAL_BLINK_RATE_MIN = 15; // blinks per minute
const NORMAL_BLINK_RATE_MAX = 20;
const FATIGUE_EAR_THRESHOLD = 0.25; // EAR consistently low indicates fatigue

// ==================== UTILITY FUNCTIONS ====================

// Calculate Euclidean distance between two points
export const distance = (p1: Point, p2: Point): number => {
  return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
};

// Calculate variance of an array
export const calculateVariance = (values: number[]): number => {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
};

// Calculate standard deviation
export const calculateStdDev = (values: number[]): number => {
  return Math.sqrt(calculateVariance(values));
};

// Clamp value between min and max
export const clamp = (value: number, min: number, max: number): number => {
  return Math.max(min, Math.min(max, value));
};

// ==================== BASIC METRICS (EXISTING) ====================

// Calculate Eye Aspect Ratio (EAR) for one eye
const calculateSingleEAR = (landmarks: Point[], eyeIndices: number[]): number | null => {
  const eyePoints = eyeIndices.map(i => landmarks[i]);
  if (eyePoints.some(p => p === undefined)) return null;

  const [p1, p2, p3, p4, p5, p6] = eyePoints;
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

  const [pLeft, pRight, pTop, pBottom] = mouthPoints;
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
  if (!nose || !leftEye || !rightEye || !chin) return { yaw: null, pitch: null, roll: null };

  const eyeMidpoint = {
    x: (leftEye.x + rightEye.x) / 2,
    y: (leftEye.y + rightEye.y) / 2,
  };

  const eyeDist = distance(leftEye, rightEye);
  const noseChinDist = distance(nose, chin);

  if (eyeDist === 0 || noseChinDist === 0) {
    return { yaw: null, pitch: null, roll: null };
  }

  const yaw = (nose.x - eyeMidpoint.x) / eyeDist;
  const pitch = (nose.y - eyeMidpoint.y) / noseChinDist;
  const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

  return {
    yaw: yaw * 50,
    pitch: pitch * 50,
    roll: roll * (180 / Math.PI),
  };
};

// ==================== ADVANCED ML METRICS ====================

// Blink detection state tracker
export class BlinkDetector {
  private earHistory: Array<{ time: number; ear: number }> = [];
  private blinkEvents: number[] = [];
  private wasEyeClosed = false;
  private eyeClosureStart: number | null = null;
  private closedFrames: number[] = []; // timestamps when eyes were closed

  detectBlink(ear: number | null, timestamp: number): { blinkDetected: boolean; blinkRate: number; perclos: number } {
    if (ear === null) {
      return { blinkDetected: false, blinkRate: this.getBlinkRate(), perclos: this.getPERCLOS(timestamp) };
    }

    // Update EAR history
    this.earHistory.push({ time: timestamp, ear });

    // Keep only last 2 minutes of data
    const cutoff = timestamp - 120000;
    this.earHistory = this.earHistory.filter(e => e.time > cutoff);
    this.blinkEvents = this.blinkEvents.filter(t => t > cutoff);
    this.closedFrames = this.closedFrames.filter(t => t > cutoff);

    const isEyeClosed = ear < EAR_BLINK_THRESHOLD;
    let blinkDetected = false;

    // Track eye closure for PERCLOS
    if (isEyeClosed) {
      this.closedFrames.push(timestamp);
      if (!this.wasEyeClosed) {
        this.eyeClosureStart = timestamp;
      }
    }

    // Detect blink: transition from closed to open
    if (this.wasEyeClosed && !isEyeClosed) {
      // Blink duration should be 100-400ms to count as intentional blink
      if (this.eyeClosureStart) {
        const blinkDuration = timestamp - this.eyeClosureStart;
        if (blinkDuration >= 50 && blinkDuration <= 500) {
          blinkDetected = true;
          this.blinkEvents.push(timestamp);
        }
      }
      this.eyeClosureStart = null;
    }

    this.wasEyeClosed = isEyeClosed;

    return {
      blinkDetected,
      blinkRate: this.getBlinkRate(),
      perclos: this.getPERCLOS(timestamp),
    };
  }

  private getBlinkRate(): number {
    if (this.blinkEvents.length < 2) return 0;

    const now = Date.now();
    const oneMinuteAgo = now - 60000;
    const recentBlinks = this.blinkEvents.filter(t => t > oneMinuteAgo);

    // Calculate blinks per minute
    const timeSpan = Math.min(60000, now - this.blinkEvents[0]);
    if (timeSpan < 5000) return 0; // Need at least 5 seconds of data

    return (recentBlinks.length / timeSpan) * 60000;
  }

  private getPERCLOS(currentTime: number): number {
    // PERCLOS: percentage of time eyes were closed in the last minute
    const oneMinuteAgo = currentTime - PERCLOS_WINDOW_MS;
    const recentHistory = this.earHistory.filter(e => e.time > oneMinuteAgo);

    if (recentHistory.length < 10) return 0;

    const closedCount = recentHistory.filter(e => e.ear < EAR_BLINK_THRESHOLD).length;
    return (closedCount / recentHistory.length) * 100;
  }

  getEARStats(): { mean: number; variance: number } {
    if (this.earHistory.length === 0) return { mean: 0, variance: 0 };

    const ears = this.earHistory.map(e => e.ear);
    const mean = ears.reduce((a, b) => a + b, 0) / ears.length;
    const variance = calculateVariance(ears);

    return { mean, variance };
  }
}

// Gaze direction estimation
export const calculateGazeDirection = (landmarks: Point[]): { gazeX: number | null; gazeY: number | null } => {
  if (!landmarks || landmarks.length < 68) return { gazeX: null, gazeY: null };

  // Get eye corners and approximate iris centers
  const leftEyeLeft = landmarks[36];
  const leftEyeRight = landmarks[39];
  const rightEyeLeft = landmarks[42];
  const rightEyeRight = landmarks[45];

  // Approximate iris center from eye landmarks (average of upper and lower lids)
  const leftIrisCenter = {
    x: (landmarks[37].x + landmarks[38].x + landmarks[40].x + landmarks[41].x) / 4,
    y: (landmarks[37].y + landmarks[38].y + landmarks[40].y + landmarks[41].y) / 4,
  };
  const rightIrisCenter = {
    x: (landmarks[43].x + landmarks[44].x + landmarks[46].x + landmarks[47].x) / 4,
    y: (landmarks[43].y + landmarks[44].y + landmarks[46].y + landmarks[47].y) / 4,
  };

  if (!leftEyeLeft || !leftEyeRight || !rightEyeLeft || !rightEyeRight) {
    return { gazeX: null, gazeY: null };
  }

  // Calculate gaze for each eye
  const leftEyeWidth = distance(leftEyeLeft, leftEyeRight);
  const rightEyeWidth = distance(rightEyeLeft, rightEyeRight);

  if (leftEyeWidth === 0 || rightEyeWidth === 0) return { gazeX: null, gazeY: null };

  // Normalize iris position within eye (-1 to 1)
  const leftGazeX = ((leftIrisCenter.x - leftEyeLeft.x) / leftEyeWidth - 0.5) * 2;
  const rightGazeX = ((rightIrisCenter.x - rightEyeLeft.x) / rightEyeWidth - 0.5) * 2;

  // Vertical gaze (using vertical position relative to eye center)
  const leftEyeCenterY = (leftEyeLeft.y + leftEyeRight.y + landmarks[37].y + landmarks[41].y) / 4;
  const rightEyeCenterY = (rightEyeLeft.y + rightEyeRight.y + landmarks[43].y + landmarks[47].y) / 4;

  const leftEyeHeight = Math.abs(landmarks[37].y - landmarks[41].y);
  const rightEyeHeight = Math.abs(landmarks[43].y - landmarks[47].y);

  const leftGazeY = leftEyeHeight > 0 ? (leftEyeCenterY - leftIrisCenter.y) / leftEyeHeight : 0;
  const rightGazeY = rightEyeHeight > 0 ? (rightEyeCenterY - rightIrisCenter.y) / rightEyeHeight : 0;

  // Average both eyes
  return {
    gazeX: clamp((leftGazeX + rightGazeX) / 2, -1, 1),
    gazeY: clamp((leftGazeY + rightGazeY) / 2, -1, 1),
  };
};

// Face distance estimation based on inter-pupillary distance
export const estimateFaceDistance = (landmarks: Point[], frameWidth: number): number | null => {
  if (!landmarks || landmarks.length < 68) return null;

  const leftEye = landmarks[36];
  const rightEye = landmarks[45];

  if (!leftEye || !rightEye) return null;

  const ipd = distance(leftEye, rightEye);

  // Normalize by frame width (typical IPD is ~6.3cm, appears ~1/4 of frame when at optimal distance)
  const normalizedIPD = ipd / frameWidth;

  // Map to 0-1 range (0 = very far, 1 = very close)
  // Typical range: 0.1 (far) to 0.4 (close)
  return clamp((normalizedIPD - 0.1) / 0.3, 0, 1);
};

// Calculate face bounding box area for presence detection
export const calculateFaceArea = (landmarks: Point[], frameWidth: number, frameHeight: number): number | null => {
  if (!landmarks || landmarks.length < 68) return null;

  const xs = landmarks.map(p => p.x);
  const ys = landmarks.map(p => p.y);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const faceArea = (maxX - minX) * (maxY - minY);
  const frameArea = frameWidth * frameHeight;

  return faceArea / frameArea;
};

// Eyebrow raise detection (surprise/attention indicator)
export const calculateEyebrowRaise = (landmarks: Point[]): number | null => {
  if (!landmarks || landmarks.length < 68) return null;

  // Compare eyebrow height to eye position
  const leftEyebrowY = LEFT_EYEBROW.map(i => landmarks[i]?.y).filter(y => y !== undefined);
  const rightEyebrowY = RIGHT_EYEBROW.map(i => landmarks[i]?.y).filter(y => y !== undefined);

  if (leftEyebrowY.length === 0 || rightEyebrowY.length === 0) return null;

  const avgEyebrowY = [...leftEyebrowY, ...rightEyebrowY].reduce((a, b) => a + b, 0) / (leftEyebrowY.length + rightEyebrowY.length);

  // Get eye center Y
  const leftEyeY = LEFT_EYE_LANDMARKS.map(i => landmarks[i]?.y).filter(y => y !== undefined);
  const rightEyeY = RIGHT_EYE_LANDMARKS.map(i => landmarks[i]?.y).filter(y => y !== undefined);

  if (leftEyeY.length === 0 || rightEyeY.length === 0) return null;

  const avgEyeY = [...leftEyeY, ...rightEyeY].reduce((a, b) => a + b, 0) / (leftEyeY.length + rightEyeY.length);

  // Distance between eyebrow and eye (normalized)
  const eyeWidth = Math.abs(landmarks[36].x - landmarks[45].x);
  if (eyeWidth === 0) return null;

  const eyebrowEyeDistance = (avgEyeY - avgEyebrowY) / eyeWidth;

  // Normalize to 0-1 range (typical range is 0.2-0.5)
  return clamp((eyebrowEyeDistance - 0.2) / 0.3, 0, 1);
};

// Mouth openness for speaking/yawning detection
export const calculateMouthOpenness = (landmarks: Point[]): number | null => {
  if (!landmarks || landmarks.length < 68) return null;

  // Use inner mouth landmarks for more accurate measurement
  const topLip = landmarks[62]; // Top of inner upper lip
  const bottomLip = landmarks[66]; // Bottom of inner lower lip
  const leftCorner = landmarks[60];
  const rightCorner = landmarks[64];

  if (!topLip || !bottomLip || !leftCorner || !rightCorner) return null;

  const mouthHeight = distance(topLip, bottomLip);
  const mouthWidth = distance(leftCorner, rightCorner);

  if (mouthWidth === 0) return null;

  // Normalize (typical range 0-0.5)
  return clamp(mouthHeight / mouthWidth, 0, 1);
};

// Head pose stability tracker
export class HeadPoseTracker {
  private poseHistory: Array<{ time: number; yaw: number; pitch: number; roll: number }> = [];

  addPose(pose: { yaw: number | null; pitch: number | null; roll: number | null }, timestamp: number): void {
    if (pose.yaw !== null && pose.pitch !== null && pose.roll !== null) {
      this.poseHistory.push({ time: timestamp, ...pose as { yaw: number; pitch: number; roll: number } });

      // Keep last 30 seconds
      const cutoff = timestamp - 30000;
      this.poseHistory = this.poseHistory.filter(p => p.time > cutoff);
    }
  }

  getStability(): number {
    if (this.poseHistory.length < 5) return 100;

    const yaws = this.poseHistory.map(p => p.yaw);
    const pitches = this.poseHistory.map(p => p.pitch);
    const rolls = this.poseHistory.map(p => p.roll);

    const yawVar = calculateVariance(yaws);
    const pitchVar = calculateVariance(pitches);
    const rollVar = calculateVariance(rolls);

    // Combined variance (weighted)
    const combinedVar = (yawVar + pitchVar + rollVar * 0.5) / 2.5;

    // Convert to stability score (high variance = low stability)
    // Typical variance range: 0-1000
    return clamp(100 - (combinedVar / 10), 0, 100);
  }

  getFidgetScore(): number {
    // Fidget = rapid, small movements (high frequency, low amplitude)
    if (this.poseHistory.length < 10) return 0;

    // Calculate frame-to-frame differences
    const diffs: number[] = [];
    for (let i = 1; i < this.poseHistory.length; i++) {
      const prev = this.poseHistory[i - 1];
      const curr = this.poseHistory[i];
      const diff = Math.abs(curr.yaw - prev.yaw) + Math.abs(curr.pitch - prev.pitch) + Math.abs(curr.roll - prev.roll);
      diffs.push(diff);
    }

    // High average diff with low variance = fidgeting
    const avgDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;

    // Normalize to 0-100
    return clamp(avgDiff * 5, 0, 100);
  }

  getPoseVariance(): number {
    if (this.poseHistory.length < 2) return 0;

    const yaws = this.poseHistory.map(p => p.yaw);
    const pitches = this.poseHistory.map(p => p.pitch);
    const rolls = this.poseHistory.map(p => p.roll);

    return (calculateVariance(yaws) + calculateVariance(pitches) + calculateVariance(rolls)) / 3;
  }
}

// Fatigue Index Calculator
export class FatigueCalculator {
  private perclosHistory: Array<{ time: number; value: number }> = [];
  private blinkRateHistory: Array<{ time: number; value: number }> = [];
  private earMeanHistory: Array<{ time: number; value: number }> = [];

  update(perclos: number, blinkRate: number, earMean: number, timestamp: number): number {
    this.perclosHistory.push({ time: timestamp, value: perclos });
    this.blinkRateHistory.push({ time: timestamp, value: blinkRate });
    this.earMeanHistory.push({ time: timestamp, value: earMean });

    // Keep last 5 minutes
    const cutoff = timestamp - 300000;
    this.perclosHistory = this.perclosHistory.filter(p => p.time > cutoff);
    this.blinkRateHistory = this.blinkRateHistory.filter(p => p.time > cutoff);
    this.earMeanHistory = this.earMeanHistory.filter(p => p.time > cutoff);

    return this.calculateFatigueIndex();
  }

  private calculateFatigueIndex(): number {
    if (this.perclosHistory.length < 5) return 0;

    // PERCLOS contribution (high PERCLOS = fatigue)
    const avgPerclos = this.perclosHistory.reduce((a, b) => a + b.value, 0) / this.perclosHistory.length;
    const perclosScore = clamp(avgPerclos * 2, 0, 40); // Max 40 points

    // Blink rate contribution (very low or very high = fatigue)
    const avgBlinkRate = this.blinkRateHistory.reduce((a, b) => a + b.value, 0) / this.blinkRateHistory.length;
    let blinkScore = 0;
    if (avgBlinkRate < NORMAL_BLINK_RATE_MIN) {
      blinkScore = clamp((NORMAL_BLINK_RATE_MIN - avgBlinkRate) * 2, 0, 30);
    } else if (avgBlinkRate > NORMAL_BLINK_RATE_MAX) {
      blinkScore = clamp((avgBlinkRate - NORMAL_BLINK_RATE_MAX), 0, 30);
    }

    // EAR trend contribution (decreasing EAR = fatigue)
    let earTrendScore = 0;
    if (this.earMeanHistory.length > 10) {
      const recentEars = this.earMeanHistory.slice(-10).map(e => e.value);
      const earVariance = calculateVariance(recentEars);
      const avgEar = recentEars.reduce((a, b) => a + b, 0) / recentEars.length;

      if (avgEar < FATIGUE_EAR_THRESHOLD) {
        earTrendScore = clamp((FATIGUE_EAR_THRESHOLD - avgEar) * 100, 0, 30);
      }
    }

    return clamp(perclosScore + blinkScore + earTrendScore, 0, 100);
  }
}

// ==================== HAND-OBJECT PROXIMITY DETECTOR ====================

/**
 * HandObjectProximityDetector
 *
 * Fuses detected hand centroids with detected object bounding boxes to produce
 * the hand-object proximity signal required by ARCHITECTURE.md §2.4.
 *
 * Returns a HandObjectProximity label that the MLEngine uses for context-aware
 * pitch scoring (book → attentive, phone → distracted).
 */
export class HandObjectProximityDetector {
  /**
   * Normalized distance threshold: a hand is "near" an object when the
   * Euclidean distance (in [0,1] space) between the hand centroid and the
   * object centre falls below this value.
   */
  private static readonly PROXIMITY_THRESHOLD = 0.20;

  /**
   * @param handResults    Output of HandTracker.detect()
   * @param detectedObjects Output of ObjectDetector.detect()
   */
  compute(
    handResults: HandResult[],
    detectedObjects: DetectedObject[]
  ): HandObjectProximity {
    if (handResults.length === 0) return 'hands_not_visible';
    if (detectedObjects.length === 0) return 'hands_on_desk';

    const handCenters = handResults.map((h) =>
      HandTracker.getHandCenter(h.landmarks)
    );

    // Priority order matches ARCHITECTURE.md §2.2: phone > book > pen
    for (const priority of ['phone', 'book', 'notebook', 'pen'] as const) {
      const obj = detectedObjects.find((o) => o.class === priority);
      if (!obj) continue;

      const objCenter = { x: obj.bbox.x, y: obj.bbox.y };

      for (const hand of handCenters) {
        const dist = HandTracker.distance2D(hand, objCenter);
        if (dist < HandObjectProximityDetector.PROXIMITY_THRESHOLD) {
          if (priority === 'phone') return 'holding_phone';
          if (priority === 'book' || priority === 'notebook') return 'holding_book';
          if (priority === 'pen') return 'holding_pen';
        }
      }
    }

    return 'hands_on_desk';
  }
}

// ==================== MAIN METRICS AGGREGATOR ====================

export class FaceMetricsAggregator {
  private blinkDetector = new BlinkDetector();
  private headPoseTracker = new HeadPoseTracker();
  private fatigueCalculator = new FatigueCalculator();
  private marHistory: number[] = [];

  calculateExtendedMetrics(
    landmarks: Point[],
    frameWidth: number,
    frameHeight: number,
    timestamp: number
  ): ExtendedFaceMetrics {
    // Basic metrics
    const ear = calculateEAR(landmarks);
    const mar = calculateMAR(landmarks);
    const headPose = calculateHeadPose(landmarks);

    // Update MAR history
    if (mar !== null) {
      this.marHistory.push(mar);
      if (this.marHistory.length > 100) this.marHistory.shift();
    }

    // Blink detection
    const { blinkDetected, blinkRate, perclos } = this.blinkDetector.detectBlink(ear, timestamp);
    const earStats = this.blinkDetector.getEARStats();

    // Head pose tracking
    this.headPoseTracker.addPose(headPose, timestamp);
    const headPoseStability = this.headPoseTracker.getStability();
    const fidgetScore = this.headPoseTracker.getFidgetScore();
    const poseVariance = this.headPoseTracker.getPoseVariance();

    // Gaze
    const gaze = calculateGazeDirection(landmarks);

    // Face position
    const faceDistance = estimateFaceDistance(landmarks, frameWidth);
    const faceArea = calculateFaceArea(landmarks, frameWidth, frameHeight);

    // Micro-expressions
    const eyebrowRaise = calculateEyebrowRaise(landmarks);
    const mouthOpenness = calculateMouthOpenness(landmarks);

    // Fatigue
    const fatigueIndex = this.fatigueCalculator.update(perclos, blinkRate, earStats.mean, timestamp);

    return {
      // Basic
      ear,
      mar,
      yaw: headPose.yaw,
      pitch: headPose.pitch,
      roll: headPose.roll,

      // Blink
      blinkDetected,
      blinkRate,
      perclos,

      // Gaze
      gazeX: gaze.gazeX,
      gazeY: gaze.gazeY,

      // Distance
      faceDistance,
      faceArea,

      // Fatigue
      fatigueIndex,
      fidgetScore,
      headPoseStability,

      // Micro-expressions
      eyebrowRaise,
      mouthOpenness,

      // Variance
      earVariance: earStats.variance,
      marVariance: calculateVariance(this.marHistory),
      poseVariance,
    };
  }

  reset(): void {
    this.blinkDetector = new BlinkDetector();
    this.headPoseTracker = new HeadPoseTracker();
    this.fatigueCalculator = new FatigueCalculator();
    this.marHistory = [];
  }
}

// Export basic metrics function for backward compatibility
export const calculateBasicMetrics = (landmarks: Point[]): FaceMetrics => {
  const ear = calculateEAR(landmarks);
  const mar = calculateMAR(landmarks);
  const headPose = calculateHeadPose(landmarks);

  return {
    ear,
    mar,
    yaw: headPose.yaw,
    pitch: headPose.pitch,
    roll: headPose.roll,
  };
};