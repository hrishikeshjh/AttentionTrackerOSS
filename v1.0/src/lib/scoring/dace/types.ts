/**
 * Pure DACE types (no DOM, no MediaPipe imports).
 *
 * Coordinate conventions used throughout scoring/dace/:
 *   - Image-normalized coordinates: x right, y DOWN, both in [0, 1].
 *   - Boxes (objects, faces) use the existing NormalizedBBox: CENTRE x/y + width/height.
 *   - Angles in degrees. Pitch: POSITIVE = HEAD DOWN (FR1). Legacy fields
 *     (`ExtendedFaceMetrics.pitch`, `gazeY`) keep the legacy convention
 *     (down = negative); feature-mapping.ts writes negated values into them.
 */
import type {
  BehaviorLabel,
  DetectedObject,
  HandResult,
  NormalizedBBox,
  UnsureReason,
} from '../../types';

export type { BehaviorLabel, DetectedObject, HandResult, NormalizedBBox, UnsureReason };

/** 4x4 matrix, COLUMN-MAJOR (MediaPipe `facialTransformationMatrixes[i].data`). */
export type Mat4 = ArrayLike<number>;

/** Row-major 3x3 rotation matrix. */
export type Mat3 = [[number, number, number], [number, number, number], [number, number, number]];

export type EulerDeg = {
  /** + = head down */
  pitch: number;
  /** + = face turned toward the camera's +x (image right) */
  yaw: number;
  /** + = head tilted so the face's up vector leans toward image right */
  roll: number;
};

/** MediaPipe blendshape scores keyed by categoryName (e.g. 'eyeLookDownLeft'). */
export type BlendshapeMap = Readonly<Record<string, number>>;

export type MeshPoint = { x: number; y: number; z?: number };

/** Per-frame face measurement after the M layer. */
export type DaceFace = {
  /** absolute pitch (deg, +down) */
  pitchDeg: number | null;
  /** Δθ = θ − θ_ref */
  relPitchDeg: number | null;
  yawDeg: number | null;
  rollDeg: number | null;
  /** vertical eye-in-head gaze g_t, +down */
  gazeV: number | null;
  /** horizontal eye-in-head gaze */
  gazeH: number | null;
  /** false when the iris cross-check fails or gaze is unavailable: D_t then uses head pitch only */
  gazeValid: boolean;
  /** raw openness (mesh EAR) */
  ear: number | null;
  /** r_t = EAR / Ê(g) */
  residualOpenness: number | null;
  /** r_t < k (null if openness unavailable) */
  eyeClosed: boolean | null;
  /** face box, normalized centre format */
  faceBox: NormalizedBBox | null;
};

/** One 200 ms tick. `face === null` means no face (still fed to the window: needed for absent/U3). */
export type DaceFrame = {
  t: number;
  face: DaceFace | null;
  /** hand landmarks for this frame (already filtered to the primary student if needed) */
  hands: HandResult[];
  /** most recent detector output, carried forward between detector runs (every 3rd frame) */
  objects: DetectedObject[];
};

export type HandId = 'Left' | 'Right';

/** Per-hand per-frame features. */
export type HandFrameFeatures = {
  id: HandId;
  pinch: boolean;
  pinchRatio: number | null;
  wrist: { x: number; y: number };
  /** ‖wrist − middle_MCP‖ in isotropic normalized units */
  handLength: number;
  holdingPhone: boolean;
  holdingBook: boolean;
  nearBook: boolean;
};

/** Per-frame derived evidence (inputs to the window). */
export type FrameEvidence = {
  t: number;
  faceVisible: boolean;
  /** D_t */
  lookingDown: boolean;
  eyeClosed: boolean | null;
  phoneDetected: boolean;
  bookDetected: boolean;
  /** P_t = phone ∧ holding(h, phone) ∧ D_t */
  phoneInUseFrame: boolean;
  /** book detected and held or near a hand */
  bookEvidenceFrame: boolean;
  /** desk cue: hand wrist below face box bottom, or book/phone detected */
  deskCue: boolean;
  hands: HandFrameFeatures[];
  /** per hand: wrist motion in hand lengths since previous frame within [m_min, m_max] */
  move: Partial<Record<HandId, boolean>>;
  moveAmount: Partial<Record<HandId, number | null>>;
  /** screen-attention inputs (rule 7) */
  relPitchDeg: number | null;
  yawDeg: number | null;
  gazeH: number | null;
};

/** Window-level summary over W (and the 25-frame desk window). */
export type WindowSummary = {
  t: number;
  n: number;
  noFaceCount: number;
  lookingDownCount: number;
  lookingDownMost: boolean;
  phoneInUseCount: number;
  phoneDetectedCount: number;
  phoneInUse: boolean;
  phonePresent: boolean;
  phoneHeldAny: boolean;
  bookEvidenceCount: number;
  book: boolean;
  /** per-hand pinch∧move shares meet writingShare */
  penGripMotion: Partial<Record<HandId, boolean>>;
  writing: boolean;
  /** U4: phone in use and pen-grip motion in a hand other than the phone hand */
  conflict: boolean;
  deskCueCount: number;
  deskWindowN: number;
  deskVisible: boolean;
  /** current downward episode (ms), 0 if not in an episode */
  downEpisodeMs: number;
  inDownEpisode: boolean;
  /** rule 7 */
  screenAttentive: boolean;
};

export type RuleId =
  | 'absent'
  | 'U3'
  | 'drowsy'
  | 'U4'
  | 'phone'
  | 'phone_hold'
  | 'reading'
  | 'glance'
  | 'U1'
  | 'U2'
  /** rule 6 with abstention off: long downward episode, no evidence -> distracted (legacy behaviour) */
  | 'down_no_evidence'
  | 'screen';

export type DaceWindowOutput = {
  t: number;
  /** null when unsure */
  label: BehaviorLabel | null;
  unsure: boolean;
  reason: UnsureReason | null;
  rule: RuleId;
  /** for rule 'glance': the rule whose output was kept */
  keptFrom?: RuleId;
  /** U5 (Stage 2): competing classes in the conformal set */
  competing?: BehaviorLabel[];
  summary: WindowSummary;
};
