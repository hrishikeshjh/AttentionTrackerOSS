/**
 * DACE parameters (Technical Annex, Table A.6 + prose constants).
 *
 * Three groups:
 *   1. TABLE_A6       — exact Table A.6 initial values (tested against a
 *                       frozen snapshot; change only with a documented tuning
 *                       run on the pilot set P1).
 *   2. PROSE          — constants stated in the annex prose (A.1-A.5, B.1).
 *   3. TBD            — values the annex leaves to be fitted on P1, or that the
 *                       implementation had to choose. Each carries a
 *                       PROVISIONAL default so the code runs; none of them is a
 *                       claim. They are listed in TBD_PARAM_KEYS.
 *
 * All shares are fractions of the window length |W| (or of the desk window),
 * compared inclusively ("at least") unless the annex says otherwise.
 */

export type EarPrior = { b0: number; b1: number; b2: number };

export type DaceParams = {
  // ── Table A.6 ────────────────────────────────────────────────────────
  /** θ_down: relative head pitch (deg, +down) counted as looking down. Inclusive. */
  thetaDownDeg: number;
  /** γ_down: eye-down blendshape level counted as looking down. Inclusive. */
  gammaDown: number;
  /** k: residual openness below which the eye counts as closed. Strict (<). */
  kOpen: number;
  /** p_drowsy: gaze-corrected PERCLOS (60 s, fraction 0-1) treated as drowsy. Inclusive. */
  pDrowsy: number;
  /** c_min: minimum detector confidence (detections below are ignored). */
  cMin: number;
  /** ρ_use: share of W with phone-in-use evidence P_t. Inclusive. */
  rhoUse: number;
  /** π_max: pinch ratio for a pen grip. Strict (<). */
  piMax: number;
  /** m_min: writing wrist motion lower bound (hand lengths / frame). Inclusive. */
  mMin: number;
  /** m_max: writing wrist motion upper bound (hand lengths / frame). Inclusive. */
  mMax: number;
  /** τ_g: longest downward glance treated as brief (ms). Inclusive (≤). */
  tauGlanceMs: number;
  /** α: conformal error level (Stage 2). */
  alphaConformal: number;

  // ── Prose constants ──────────────────────────────────────────────────
  /** base frame rate (A.0 notation: t is the frame index at 5 FPS) */
  fps: number;
  /** nominal frame interval (ms) = 1000 / fps */
  frameIntervalMs: number;
  /** |W|: window length in frames (3 s at 5 FPS) */
  windowFrames: number;
  windowMs: number;
  /** screen-facing pitch reference duration (A.1, B.1) */
  referenceMs: number;
  /** rule 1: no face in at least this share of W -> absent */
  absentShare: number;
  /** U3: face not detected in at least this share of W */
  u3MissingShare: number;
  /** holding(h,b): IoU(box(h), b) ≥ this */
  holdIoU: number;
  /** near(h,b): ‖centre(h) − centre(b)‖ < this · diag(b) */
  nearDiagFactor: number;
  /** phone detected in at least this share of W but not in use -> "present" */
  phonePresentShare: number;
  /** book detected and held/near in at least this share of W */
  bookShare: number;
  /** pinch_t and move_t each in at least this share of W */
  writingShare: number;
  /** "most" (rule 4 looking down; writing's D_t clause): strictly more than this share */
  mostShare: number;
  /** desk-visibility window (frames, 5 s) */
  deskWindowFrames: number;
  /** desk visible if cue present in at least this share of the desk window */
  deskShare: number;
  /** rule 3: using_phone label held for this long (ms) */
  phoneHoldMs: number;
  /** rule 2: one continuous closure of at least this long (ms) -> drowsy */
  closureMs: number;
  /** PERCLOS window (ms) */
  perclosWindowMs: number;
  /** object detector runs every Nth frame (outputs carried forward between runs) */
  detectorEveryNFrames: number;

  // ── TBD / implementation choices (provisional; fitted on P1) ─────────
  /** camera-move reset: jump in 1 s median pitch between consecutive seconds (deg) */
  cameraMovePitchJumpDeg: number;
  /** camera-move reset: relative jump in 1 s median face scale */
  cameraMoveFaceScaleJump: number;
  /** quadratic openness prior Ê(g) = b0 + b1 g + b2 g² used until the Huber fit is identifiable */
  earPrior: EarPrior;
  /** ear feature = residual openness × earScale (maps r≈1 into the legacy EAR range) */
  earScale: number;
  /** gaze feature scale (blendshape units -> legacy gaze units) */
  gazeScale: number;
  /** pitch sign multiplier, kept +1 unless the pilot up/down segments show an inversion */
  pitchSign: number;
  /** minimum samples before a Huber openness fit replaces the prior */
  opennessMinSamples: number;
  /** minimum spread of g in the fit sample before β1/β2 are fitted (else intercept-only) */
  opennessMinGazeRange: number;
  /** Huber tuning constant (× robust scale) */
  huberDelta: number;
  /** minimum valid samples before PERCLOS is reported (legacy BlinkDetector uses 10) */
  perclosMinSamples: number;
  /** downward-episode tracker tolerates this many non-down frames inside an episode */
  episodeGapFrames: number;
  /** iris validity: |iris offset| / eye width must not exceed this (iris inside the eye) */
  irisMaxOffset: number;
  /** iris/blendshape cross-check: blendshape says |g| ≥ γ_down but the iris offset points the other way by more than this -> gaze invalid */
  irisSignTol: number;
  /** rule 7 screen label: max |horizontal gaze| (legacy OPTIMAL_RANGES.gaze.maxDeviation) */
  screenGazeMax: number;
  /** rule 7 screen label: max |yaw| deg (legacy OPTIMAL_RANGES.yaw.maxDeviation) */
  screenYawMaxDeg: number;
  /** rule 7 screen label: max upward relative pitch deg (legacy pitch maxDeviation) */
  screenPitchUpMaxDeg: number;
  /** frame aspect ratio (width / height) used to make normalized distances isotropic */
  frameAspect: number;
  /** a hand belongs to the student if its wrist is within this many face widths of the face box */
  handFaceMaxDistFaceWidths: number;

  // ── Stage 2 (K, conformal) — not used in Stage 1 ─────────────────────
  kinematicsFps: number;
  kinematicsWindowMinMs: number;
  kinematicsWindowMaxMs: number;
  /** K is dropped if it adds less than this many macro-F1 points (LOSO) */
  kinematicsGoNoGoF1Points: number;
};

/** Exact Table A.6 values. */
export const TABLE_A6 = Object.freeze({
  thetaDownDeg: 15,
  gammaDown: 0.5,
  kOpen: 0.7,
  pDrowsy: 0.15,
  cMin: 0.35,
  rhoUse: 0.6,
  piMax: 0.35,
  mMin: 0.02,
  mMax: 0.15,
  tauGlanceMs: 3000,
  alphaConformal: 0.1,
} as const);

export const PROSE_CONSTANTS = Object.freeze({
  fps: 5,
  frameIntervalMs: 200,
  windowFrames: 15,
  windowMs: 3000,
  referenceMs: 5000,
  absentShare: 0.8,
  u3MissingShare: 0.5,
  holdIoU: 0.1,
  nearDiagFactor: 0.5,
  phonePresentShare: 0.3,
  bookShare: 0.5,
  writingShare: 0.6,
  mostShare: 0.5,
  deskWindowFrames: 25,
  deskShare: 0.3,
  phoneHoldMs: 10000,
  closureMs: 2000,
  perclosWindowMs: 60000,
  detectorEveryNFrames: 3,
} as const);

/** PROVISIONAL values — every one of these is to be fitted or confirmed on P1. */
export const TBD_DEFAULTS = Object.freeze({
  cameraMovePitchJumpDeg: 25,
  cameraMoveFaceScaleJump: 0.3,
  earPrior: Object.freeze({ b0: 0.3, b1: -0.12, b2: 0 }) as EarPrior,
  earScale: 0.3,
  gazeScale: 1,
  pitchSign: 1,
  opennessMinSamples: 50,
  opennessMinGazeRange: 0.3,
  huberDelta: 1.345,
  perclosMinSamples: 10,
  episodeGapFrames: 1,
  irisMaxOffset: 0.5,
  irisSignTol: 0.05,
  screenGazeMax: 0.3,
  screenYawMaxDeg: 15,
  screenPitchUpMaxDeg: 15,
  frameAspect: 640 / 480,
  handFaceMaxDistFaceWidths: 3,
});

export const STAGE2_CONSTANTS = Object.freeze({
  kinematicsFps: 30,
  kinematicsWindowMinMs: 3000,
  kinematicsWindowMaxMs: 6000,
  kinematicsGoNoGoF1Points: 2,
} as const);

export const TBD_PARAM_KEYS = Object.freeze(Object.keys(TBD_DEFAULTS)) as ReadonlyArray<keyof typeof TBD_DEFAULTS>;

export const DEFAULT_DACE_PARAMS: Readonly<DaceParams> = Object.freeze({
  ...TABLE_A6,
  ...PROSE_CONSTANTS,
  ...TBD_DEFAULTS,
  ...STAGE2_CONSTANTS,
});

export function makeParams(overrides: Partial<DaceParams> = {}): DaceParams {
  return { ...DEFAULT_DACE_PARAMS, ...overrides, earPrior: { ...(overrides.earPrior ?? DEFAULT_DACE_PARAMS.earPrior) } };
}

/** count/|W| ≥ share, robust to binary floating point (e.g. 0.6 × 15). */
export function atLeastShare(count: number, total: number, share: number): boolean {
  return count >= share * total - 1e-9;
}

/** count/|W| > share (strict "most"). */
export function moreThanShare(count: number, total: number, share: number): boolean {
  return count > share * total + 1e-9;
}
