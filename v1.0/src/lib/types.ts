export type AttentionState = 'Attentive' | 'Distracted' | 'Multitasking' | 'Idle';

// ==================== OBJECT DETECTION TYPES ====================

export type ObjectClass =
  | 'book'
  | 'notebook'
  | 'phone'
  | 'laptop'
  | 'mouse'
  | 'keyboard'
  | 'pen'
  | 'tablet';

/** Bounding box in normalized [0–1] coordinates (center x/y + w/h). */
export type NormalizedBBox = {
  x: number;      // center x
  y: number;      // center y
  width: number;
  height: number;
};

export type DetectedObject = {
  class: ObjectClass;
  confidence: number; // 0–1
  bbox: NormalizedBBox;
};

// ==================== HAND + CONTEXT TYPES ====================

/** Single MediaPipe hand landmark, normalized [0–1]. */
export type HandLandmark = { x: number; y: number; z: number };

export type HandResult = {
  landmarks: HandLandmark[]; // 21 landmarks
  handedness: 'Left' | 'Right';
};

/**
 * Proximity signal: what the user's hand is closest to.
 * Computed by HandObjectProximityDetector in face-geometry.ts.
 */
export type HandObjectProximity =
  | 'holding_book'
  | 'holding_phone'
  | 'holding_pen'
  | 'hands_on_desk'
  | 'hands_not_visible';

/** High-level interpretation of what the user is currently doing. */
export type StudyContext =
  | 'reading'
  | 'using_phone'
  | 'watching_screen'
  | 'writing'
  | 'unknown';

// ==================== BEHAVIOR CLASSIFICATION TYPES ====================

/**
 * Fine-grained behavior labels used by the temporal classifier.
 * Maps to ARCHITECTURE.md §2.3 class definitions.
 */
export type BehaviorLabel =
  | 'attentive_screen'   // watching lecture / screen
  | 'attentive_reading'  // reading book / notes
  | 'using_phone'        // phone distraction
  | 'distracted'         // looking around, talking
  | 'drowsy'             // head down + low EAR
  | 'absent';            // face not in frame

export type BehaviorResult = {
  label: BehaviorLabel;
  attentionScore: number;  // 0–100 mapped from label
  confidence: number;      // 0–1 classifier confidence
  /** 'temporal' = full 15-frame window used; 'rule_based' = fallback */
  source: 'temporal' | 'rule_based';
};

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

  // ── v1.0 Upgrade: context-aware features (optional until modules load) ──
  detectedObjects?: DetectedObject[];
  handObjectProximity?: HandObjectProximity;
  studyContext?: StudyContext;
  behaviorResult?: BehaviorResult;
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
