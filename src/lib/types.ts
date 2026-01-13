export type AttentionState = 'Attentive' | 'Distracted' | 'Multitasking' | 'Idle';

export type SessionDataPoint = {
  timestamp: number;
  attentionScore: number;
  attentionState: AttentionState;
};

export type SessionSummary = {
  averageAttention: number;
  multitaskEvents: number;
  engagementLevel: 'High' | 'Moderate' | 'Low';
  duration: string;
};

export type FaceMetrics = {
  ear: number | null;
  mar: number | null;
  yaw: number | null;
  pitch: number | null;
  roll: number | null;
};

// Extended metrics with ML-style feature engineering
export type ExtendedFaceMetrics = FaceMetrics & {
  // Blink detection
  blinkDetected: boolean;
  blinkRate: number; // blinks per minute
  perclos: number; // Percentage of Eye Closure (0-100)

  // Gaze estimation
  gazeX: number | null; // -1 (left) to 1 (right)
  gazeY: number | null; // -1 (down) to 1 (up)

  // Distance and position
  faceDistance: number | null; // relative distance score (0-1)
  faceArea: number | null; // bounding box area normalized

  // Fatigue indicators
  fatigueIndex: number; // 0-100
  fidgetScore: number; // 0-100 (head movement variance)
  headPoseStability: number; // 0-100

  // Micro-expressions
  eyebrowRaise: number | null; // 0-1
  mouthOpenness: number | null; // 0-1

  // Temporal features
  earVariance: number; // variance over sliding window
  marVariance: number;
  poseVariance: number; // combined yaw/pitch/roll variance
};

// ML Prediction with confidence and feature importance
export type MLPrediction = {
  attentionState: AttentionState;
  confidence: number; // 0-1
  rawScore: number; // 0-100 before classification
  featureImportance: Record<string, number>; // feature name -> importance weight
  featureContributions: Record<string, number>; // feature name -> contribution to score
};

// Session-level analytics
export type SessionAnalytics = {
  attentionTrend: 'improving' | 'declining' | 'stable';
  trendSlope: number; // positive = improving
  peakAttentionPeriods: Array<{ start: number; end: number; avgScore: number }>;
  lowAttentionPeriods: Array<{ start: number; end: number; avgScore: number }>;
  blinkAnomalies: number; // count of abnormal blink patterns
  fatigueOnset: number | null; // timestamp when fatigue detected
  optimalEngagementWindow: { start: number; end: number } | null;
  attentionVariability: number; // standard deviation of scores
};

// Enhanced session data point
export type EnhancedSessionDataPoint = SessionDataPoint & {
  confidence: number;
  fatigueIndex: number;
  perclos: number;
  prediction: MLPrediction;
};
