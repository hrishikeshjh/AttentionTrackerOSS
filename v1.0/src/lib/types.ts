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
  /** 'temporal' = full 15-frame window used; 'rule_based' = fallback;
   *  'dace_rules' = DACE ordered rule layer (Annex A.4) */
  source: 'temporal' | 'rule_based' | 'dace_rules';
};

// ==================== DACE (all optional; absent when DACE is off) ====================

/**
 * "Unsure" reason codes (Annex A.5). "Unsure" is NOT an AttentionState or a
 * BehaviorLabel; it is carried as flags next to them.
 *   U1 looking down > τ_g with no phone/book/writing evidence
 *   U2 looking down with the desk not visible (takes precedence over U1)
 *   U3 low tracking quality (face missing in ≥ 50% of the window)
 *   U4 conflicting evidence (phone in use in one hand, pen-grip motion in the other)
 *   U5 model uncertain (Stage 2, conformal set with > 1 class)
 */
export type UnsureReason = 'U1' | 'U2' | 'U3' | 'U4' | 'U5';

/** Which scoring path produced an MLPrediction. */
export type ScoringMode = 'legacy' | 'upgraded' | 'dace';

/** DACE measurement-layer values for one frame. Angles in degrees, +pitch = head down. */
export type DaceMeasurementSnapshot = {
  pitchDeg: number | null;
  /** Δθ = θ − θ_ref (+down) */
  relPitchDeg: number | null;
  yawDeg: number | null;
  rollDeg: number | null;
  /** vertical eye-in-head gaze g_t in [−1, 1], +down */
  gazeV: number | null;
  /** horizontal eye-in-head gaze in [−1, 1] */
  gazeH: number | null;
  gazeValid: boolean;
  /** r_t = EAR_t / Ê(g_t) */
  residualOpenness: number | null;
  /** gaze-corrected PERCLOS over 60 s, fraction 0-1 */
  correctedPerclos: number | null;
  /** D_t */
  lookingDown: boolean;
  /** pitch reference still being estimated */
  calibrating: boolean;
};

/** Window-level desk-zone evidence (Annex A.2). */
export type DeskEvidenceSummary = {
  phoneInUse: boolean;
  phonePresent: boolean;
  book: boolean;
  writing: boolean;
  deskVisible: boolean;
  /** U4 condition: phone in use and pen-grip motion in the other hand */
  conflict: boolean;
};

/** Compact DACE window output attached to metrics/predictions. */
export type DaceOutputSummary = {
  label: BehaviorLabel | null;
  unsure: boolean;
  reason: UnsureReason | null;
  /** id of the rule that fired (see scoring/dace/rules.ts RuleId) */
  rule: string;
};

export type SessionDataPoint = {
  timestamp: number;
  attentionScore: number;
  attentionState: AttentionState;
  /** DACE: score was held because the window was "Unsure" (excluded from averages/trends) */
  unsure?: boolean;
  unsureReason?: UnsureReason;
  /** DACE behaviour label for this point (absent when DACE is off or the point is Unsure) */
  behaviorLabel?: BehaviorLabel;
};

export type SessionSummary = {
  averageAttention: number;
  multitaskEvents: number;
  engagementLevel: 'High' | 'Moderate' | 'Low';
  duration: string;
  /** DACE: share (0-1) of data points that were Unsure (only set when the data carries DACE flags) */
  unsureShare?: number;
  /** DACE: Unsure point counts per reason code */
  unsureReasons?: Partial<Record<UnsureReason, number>>;
  /** DACE: e.g. "CognEn could not determine the activity for 12% of the session (mostly: desk not visible)" */
  unsureText?: string | null;
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

  // ── DACE (optional; only set when DACE is enabled) ──
  dace?: DaceMeasurementSnapshot;
  deskEvidence?: DeskEvidenceSummary;
  daceOutput?: DaceOutputSummary;
};

// ML Prediction with confidence and feature importance
export type MLPrediction = {
  attentionState: AttentionState;
  confidence: number; // 0-1
  rawScore: number; // 0-100 before classification
  featureImportance: Record<string, number>; // feature name -> importance weight
  featureContributions: Record<string, number>; // feature name -> contribution to score

  // ── Cross-Modal Contradiction Vector (see contradiction-detector.ts) ──
  // Exposed as a SIBLING signal, never folded into rawScore/confidence —
  // downstream consumers decide what to do with disagreement, it isn't
  // averaged away before it reaches them.
  contradictionIndex?: number; // 0-1, EMA-smoothed
  contradictionCategory?: 'coherent' | 'fatigue_masking' | 'prop_without_engagement' | 'context_conflict' | 'unclassified';

  // ── DACE (optional; only set on the DACE scoring path) ──
  scoringMode?: ScoringMode;
  /** true when the score was held at the last smoothed value because the window was Unsure */
  scoreHeld?: boolean;
  /** window was Unsure. If unsure && !scoreHeld there is no score yet (show "-", record no point) */
  unsure?: boolean;
  unsureReason?: UnsureReason;
  /** DACE behaviour label (null when Unsure) */
  daceLabel?: BehaviorLabel | null;
  daceRule?: string;
};

// Session-level analytics
export type SessionAnalytics = {
  attentionTrend: 'improving' | 'declining' | 'stable';
  trendSlope: number; // positive = improving
  peakAttentionPeriods: Array<{ start: number; end: number; avgScore: number }>;
  lowAttentionPeriods: Array<{ start: number; end: number; avgScore: number }>;
  /**
   * @deprecated Misnamed: this is the number of sharp attention-score drops
   * (> 30 points between consecutive points), not a blink measure. Kept under
   * the old name because the legacy golden fixture freezes analyze()'s output;
   * use SessionAnalyticsEngine.getAnomalies().scoreDropCount.
   */
  blinkAnomalies: number;
  fatigueOnset: number | null; // timestamp when fatigue detected
  optimalEngagementWindow: { start: number; end: number } | null;
  attentionVariability: number; // standard deviation of scores
  /** DACE: share (0-1) of data points that were Unsure; present only when the data carries DACE flags */
  unsureShare?: number;
};

// Enhanced session data point
export type EnhancedSessionDataPoint = SessionDataPoint & {
  confidence: number;
  fatigueIndex: number;
  perclos: number;
  prediction: MLPrediction;
};
