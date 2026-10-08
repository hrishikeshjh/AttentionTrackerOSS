// face-api.js 68-point landmark indices.
// https://github.com/justadudewhohacks/face-api.js/blob/master/src/classes/FaceLandmarks68.ts

export const LEFT_EYE_LANDMARKS = [36, 37, 38, 39, 40, 41];
export const RIGHT_EYE_LANDMARKS = [42, 43, 44, 45, 46, 47];
export const LEFT_IRIS_LANDMARKS = [37, 38, 40, 41];
export const RIGHT_IRIS_LANDMARKS = [43, 44, 46, 47];

export const MOUTH_LANDMARKS = [48, 54, 51, 57]; // Left, Right, Top, Bottom

export const LEFT_EYEBROW = [17, 18, 19, 20, 21];
export const RIGHT_EYEBROW = [22, 23, 24, 25, 26];

export const HEAD_POSE_LANDMARKS = {
  nose: 30,
  leftEyeCorner: 36,
  rightEyeCorner: 45,
  chin: 8,
};

// ==================== THRESHOLDS (canonical — web app values) ====================
// The Chrome extension previously diverged to ATTENTIVE_MIN=80/DISTRACTED_MIN=50
// and its own weight tables. This package is the single source of truth; any
// extension-specific tuning must be an explicit SCORING_PROFILE override, not
// silent drift.

export const EAR_BLINK_THRESHOLD = 0.21;
export const PERCLOS_WINDOW_MS = 60000;
export const NORMAL_BLINK_RATE_MIN = 15;
export const NORMAL_BLINK_RATE_MAX = 20;
export const FATIGUE_EAR_THRESHOLD = 0.25;

export const THRESHOLDS = {
  ATTENTIVE_MIN: 70,
  DISTRACTED_MIN: 40,
};

export const OPTIMAL_RANGES = {
  ear: { min: 0.25, max: 0.4 },
  perclos: { max: 15 },
  blinkRate: { min: 12, max: 20 },
  gaze: { maxDeviation: 0.3 },
  yaw: { maxDeviation: 15 },
  pitch: { maxDeviation: 15 },
  fatigueIndex: { max: 30 },
  fidgetScore: { max: 25 },
  faceDistance: { min: 0.3, max: 0.7 },
};

/**
 * Original (v1.0) weights — used when no context features are present.
 * Preserved for legacy code paths and Chrome extension fallback.
 */
export const FEATURE_WEIGHTS_LEGACY: Record<string, number> = {
  ear: 0.15,
  perclos: 0.12,
  blinkRate: 0.08,
  gazeDeviation: 0.1,
  headPoseStability: 0.12,
  yawDeviation: 0.08,
  pitchDeviation: 0.06,
  fatigueIndex: 0.1,
  fidgetScore: 0.07,
  eyebrowRaise: 0.04,
  mouthOpenness: 0.03,
  faceDistance: 0.03,
  facePresence: 0.02,
};

/**
 * Upgraded (v1.0+) weights — used when context features are available.
 * pitchDeviation is replaced by contextAwarePitch + objectContext +
 * handObjectProximity + temporalPattern (ARCHITECTURE.md §2).
 */
export const FEATURE_WEIGHTS_V1_UPGRADED: Record<string, number> = {
  ear: 0.1,
  perclos: 0.08,
  blinkRate: 0.05,
  gazeDeviation: 0.12,
  headPoseStability: 0.1,
  yawDeviation: 0.08,
  fatigueIndex: 0.07,
  fidgetScore: 0.04,
  mouthOpenness: 0.02,
  faceDistance: 0.02,
  facePresence: 0.02,
  contextAwarePitch: 0.12,
  objectContext: 0.1,
  handObjectProximity: 0.05,
  temporalPattern: 0.03,
};

/**
 * Head-pose calibration constants (see pose/head-pose.ts).
 * Both yaw and pitch are normalized by the SAME reference distance
 * (interocular distance) rather than yaw using eye-distance and pitch
 * using nose-chin distance — the old two-normalizer approach let pitch's
 * own normalizer (nose-chin distance) foreshorten as pitch increased,
 * biasing the pitch signal against itself. Constants below are chosen to
 * keep output magnitude compatible with existing thresholds
 * (e.g. ml-engine.ts's "pitch < -15" context-aware branch) while fixing
 * the normalizer inconsistency.
 */
export const HEAD_POSE_CALIBRATION = {
  YAW_GAIN: 50,
  PITCH_GAIN: 33.3, // 50 / 1.5 — 1.5 ≈ typical noseChinDist:eyeDist ratio
};
