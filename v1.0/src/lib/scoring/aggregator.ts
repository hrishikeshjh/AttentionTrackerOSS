import type { Point2D } from './geometry';
import { calculateVariance } from './geometry';
import { calculateEAR, calculateMAR } from './ear-mar';
import { calculateHeadPose } from './pose/head-pose';
import { calculateGazeDirection } from './pose/gaze';
import {
  estimateFaceDistance,
  calculateFaceArea,
  calculateEyebrowRaise,
  calculateMouthOpenness,
} from './basic-metrics';
import { BlinkDetector, HeadPoseTracker, FatigueCalculator } from './trackers';
import { HandTracker } from '../hand-tracker';
import type {
  ExtendedFaceMetrics,
  FaceMetrics,
  DetectedObject,
  HandResult,
  HandObjectProximity,
} from '../types';

export class HandObjectProximityDetector {
  private static readonly PROXIMITY_THRESHOLD = 0.2;

  compute(handResults: HandResult[], detectedObjects: DetectedObject[]): HandObjectProximity {
    if (handResults.length === 0) return 'hands_not_visible';
    if (detectedObjects.length === 0) return 'hands_on_desk';

    const handCenters = handResults.map((h) => HandTracker.getHandCenter(h.landmarks));

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

export class FaceMetricsAggregator {
  private blinkDetector = new BlinkDetector();
  private headPoseTracker = new HeadPoseTracker();
  private fatigueCalculator = new FatigueCalculator();
  private marHistory: number[] = [];

  calculateExtendedMetrics(
    landmarks: Point2D[],
    frameWidth: number,
    frameHeight: number,
    timestamp: number
  ): ExtendedFaceMetrics {
    const ear = calculateEAR(landmarks);
    const mar = calculateMAR(landmarks);
    const headPose = calculateHeadPose(landmarks);

    if (mar !== null) {
      this.marHistory.push(mar);
      if (this.marHistory.length > 100) this.marHistory.shift();
    }

    const { blinkDetected, blinkRate, perclos } = this.blinkDetector.detectBlink(ear, timestamp);
    const earStats = this.blinkDetector.getEARStats();

    this.headPoseTracker.addPose(headPose, timestamp);
    const headPoseStability = this.headPoseTracker.getStability();
    const fidgetScore = this.headPoseTracker.getFidgetScore();
    const poseVariance = this.headPoseTracker.getPoseVariance();

    const gaze = calculateGazeDirection(landmarks);
    const faceDistance = estimateFaceDistance(landmarks, frameWidth);
    const faceArea = calculateFaceArea(landmarks, frameWidth, frameHeight);
    const eyebrowRaise = calculateEyebrowRaise(landmarks);
    const mouthOpenness = calculateMouthOpenness(landmarks);
    const fatigueIndex = this.fatigueCalculator.update(perclos, blinkRate, earStats.mean, timestamp);

    return {
      ear,
      mar,
      yaw: headPose.yaw,
      pitch: headPose.pitch,
      roll: headPose.roll,
      blinkDetected,
      blinkRate,
      perclos,
      gazeX: gaze.gazeX,
      gazeY: gaze.gazeY,
      faceDistance,
      faceArea,
      fatigueIndex,
      fidgetScore,
      headPoseStability,
      eyebrowRaise,
      mouthOpenness,
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

export const calculateBasicMetrics = (landmarks: Point2D[]): FaceMetrics => {
  const ear = calculateEAR(landmarks);
  const mar = calculateMAR(landmarks);
  const headPose = calculateHeadPose(landmarks);
  return { ear, mar, yaw: headPose.yaw, pitch: headPose.pitch, roll: headPose.roll };
};
