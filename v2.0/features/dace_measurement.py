"""
dace_measurement.py -- v2.0 port of v1.0 DACE measurement (M), per tracked person.

Ported from v1.0/src/lib/scoring/dace/ (Technical Annex A.1; ARCHITECTURE.md §11.8):
  head-pose-matrix.ts  facial transformation matrix -> Euler angles (+pitch = head DOWN)
  pitch-reference.ts   per-track screen-facing pitch reference (median), reset rules
  gaze-blendshape.ts   eye-in-head gaze from blendshapes (+v = eyes DOWN) + iris cross-check
  eye-openness.ts      gaze-corrected eye openness (Huber quadratic Ê(g)), residual r = EAR / Ê(g)
  perclos.ts           gaze-corrected PERCLOS (closed iff r < k) over 60 s
  looking-down.ts      D_t = 1[Δθ ≥ θ_down or (gaze valid and g ≥ γ_down)]

Why: the legacy 68-point head pose (features/face_metrics.py::calculate_head_pose)
is not in degrees, has a ~+12.6 neutral offset and RISES when the head tilts
down (tests/test_pitch_characterisation.py), and the legacy "gaze" is the
eyelid centroid (~0 for every gaze direction). Every `pitch < -15` rule and
the synthetic training distributions assumed "down = negative", so the
reading/phone branches were unreachable on real input.

Conventions
-----------
Measurement layer (this module, DACE):   pitch / relative pitch in degrees, POSITIVE = head down;
                                           gaze_v in [-1, 1], POSITIVE = eyes down.
Feature / rule layer (FaceMetricsResult.pitch, .gaze_y, the 13/15-dim vector, behavior rules):
                                           DOWN = NEGATIVE. ``to_feature_pitch`` / ``to_feature_gaze_y``
                                           apply the single negation. This is the same as v1.0 DACE's
                                           legacy-field substitution (plan D7, ARCHITECTURE.md §11.8),
                                           so a model trained here sees the same signs as a model loaded
                                           by v1.0 ``TemporalBehaviorClassifier.loadModel`` in DACE mode.
                                           See features/feature_vector.py for the full contract.

Matrix layout (verified on mediapipe 0.10.31): the Python Tasks API returns
``facial_transformation_matrixes[i]`` as a numpy (4, 4) array in standard
row/column order (translation in column 3, e.g. [-5, 23, -70]); JavaScript
returns the same matrix flattened column-major. ``rotation_from_matrix``
accepts both. The +down sign was checked on a real face image warped by the
homography K·Rx(φ)·K⁻¹ (head tilted down by φ in the OpenCV camera frame):
pitch rose monotonically with φ (scratch probe, not a committed test; the
committed tests use synthetic matrices).

Fallbacks (labelled in ``FaceMeasurement.pitch_source`` / ``gaze_source``):
  pitch  "matrix"      MediaPipe transformation matrix (preferred)
         "legacy68"    LEGACY: 68-point formula, passed through the same per-track
                       reference and sign flip. Not degrees; saturates ~+3 units
                       below the reference for real downward tilts, so it can
                       NOT reach the 15-degree threshold. Used only when no matrix
                       is available (e.g. a landmark-only detector).
  gaze   "blendshape"  blendshapes, valid only when the iris cross-check agrees
         "legacy68"    LEGACY eyelid-centroid gaze (no iris points; ~0)
         None          gaze invalid -> looking-down falls back to head pitch only
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from features.face_metrics import calculate_ear, calculate_gaze, calculate_head_pose

# ── Parameters (v1.0 scoring/dace/params.ts: Table A.6, prose, TBD defaults) ──

THETA_DOWN_DEG = 15.0          # Table A.6 θ_down (relative pitch, +down, inclusive)
GAMMA_DOWN = 0.5               # Table A.6 γ_down (blendshape eye-down level, inclusive)
K_OPEN = 0.7                   # Table A.6 k (residual openness; closed iff r < k)
P_DROWSY = 0.15                # Table A.6 p_drowsy (gaze-corrected PERCLOS fraction)
REFERENCE_MS = 5_000           # prose: 5 s screen-facing reference
ABSENT_FRAMES = 12             # prose: ceil(0.8 * 15) missing frames -> absent -> re-estimate
PERCLOS_WINDOW_MS = 60_000
CLOSURE_FRAME_MS = 200         # nominal frame interval used for the closure duration
# TBD / provisional (to be fitted on pilot data; not claims)
CAMERA_MOVE_PITCH_JUMP_DEG = 25.0
CAMERA_MOVE_FACE_SCALE_JUMP = 0.3
EAR_PRIOR = (0.3, -0.12, 0.0)  # Ê(g) = b0 + b1 g + b2 g² prior
EAR_SCALE = 0.3                # ear feature = residual openness × EAR_SCALE (legacy EAR range)
OPENNESS_MIN_SAMPLES = 50
OPENNESS_MIN_GAZE_RANGE = 0.3
OPENNESS_REFIT_EVERY = 25      # frames (v1.0 refits every deskWindowFrames)
HUBER_DELTA = 1.345
PERCLOS_MIN_SAMPLES = 10
IRIS_MAX_OFFSET = 0.5
IRIS_SIGN_TOL = 0.05
BLOCK_MS = 1_000               # camera-move detection block (v1.0: fps frames = 1 s)

# MediaPipe 478-point mesh indices (v1.0 scoring/dace/mesh-adapter.ts)
MESH_RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
MESH_LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)
MESH_RIGHT_IRIS_CENTER = 468
MESH_LEFT_IRIS_CENTER = 473
MESH_RIGHT_EYE_CORNERS = (33, 133)
MESH_LEFT_EYE_CORNERS = (263, 362)

_RAD = 180.0 / math.pi


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _xy(p) -> tuple[float, float]:
    if hasattr(p, "x"):
        return float(p.x), float(p.y)
    return float(p[0]), float(p[1])


# ── Sign convention helpers (the ONE place the measurement sign is flipped) ──

def to_feature_pitch(rel_pitch_down_deg: float | None) -> float | None:
    """Measurement (+down) -> feature/rule convention (down = negative)."""
    return None if rel_pitch_down_deg is None else -rel_pitch_down_deg


def to_feature_gaze_y(gaze_v_down: float | None) -> float | None:
    """Blendshape gaze (+down) -> feature/rule convention (down = negative)."""
    return None if gaze_v_down is None else -gaze_v_down


# ── Head pose from the facial transformation matrix (head-pose-matrix.ts) ────

def rotation_from_matrix(m) -> np.ndarray | None:
    """
    Scale-normalised 3x3 rotation from a facial transformation matrix:
    a numpy/nested (4, 4) matrix (Python Tasks API) or a flat 16-vector in
    column-major order (JavaScript ``.data``). None if degenerate.
    """
    if m is None:
        return None
    a = np.asarray(m, dtype=np.float64)
    if a.shape == (4, 4):
        r = a[:3, :3].copy()
    elif a.size == 16 and a.ndim == 1:
        r = a.reshape(4, 4).T[:3, :3].copy()   # column-major flat -> matrix
    else:
        return None
    norms = np.linalg.norm(r, axis=0)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        return None
    return r / norms


def euler_from_rotation(r: np.ndarray, pitch_sign: float = 1.0) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees from a y-up rotation; +pitch = head down.
    R = Ry(yaw)·Rx(pitch)·Rz(roll): pitch = asin(-r12), yaw = atan2(r02, r22), roll = atan2(r10, r11)."""
    pitch = math.asin(_clamp(-float(r[1, 2]), -1.0, 1.0)) * _RAD
    yaw = math.atan2(float(r[0, 2]), float(r[2, 2])) * _RAD
    roll = math.atan2(float(r[1, 0]), float(r[1, 1])) * _RAD
    return pitch_sign * pitch, yaw, roll


def euler_from_matrix(m, pitch_sign: float = 1.0) -> tuple[float, float, float] | None:
    r = rotation_from_matrix(m)
    return None if r is None else euler_from_rotation(r, pitch_sign)


def rotation_from_euler(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Inverse of euler_from_rotation (degrees), y-up frame. Used by tests."""
    p, y, w = (math.radians(v) for v in (pitch, yaw, roll))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(w), -math.sin(w), 0], [math.sin(w), math.cos(w), 0], [0, 0, 1]])
    return ry @ rx @ rz


def matrix_from_euler(pitch: float, yaw: float, roll: float,
                      t: tuple[float, float, float] = (0.0, 0.0, -50.0), scale: float = 1.0) -> np.ndarray:
    """(4, 4) transform as the Python Tasks API returns it. Used by tests/fixtures."""
    out = np.eye(4)
    out[:3, :3] = rotation_from_euler(pitch, yaw, roll) * scale
    out[:3, 3] = t
    return out


# ── Gaze (gaze-blendshape.ts) ──────────────────────────────────────────────────

_GAZE_KEYS = (
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
)


def gaze_from_blendshapes(bs: Mapping[str, float] | None) -> tuple[float, float] | None:
    """(v, h): v = ½(down_L + down_R) − ½(up_L + up_R) (+down);
    h = ½(out_L + in_R) − ½(in_L + out_R) (+ toward image right, unmirrored camera)."""
    if not bs:
        return None
    for k in _GAZE_KEYS:
        v = bs.get(k)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            return None
    v = 0.5 * (bs["eyeLookDownLeft"] + bs["eyeLookDownRight"]) - 0.5 * (bs["eyeLookUpLeft"] + bs["eyeLookUpRight"])
    h = 0.5 * (bs["eyeLookOutLeft"] + bs["eyeLookInRight"]) - 0.5 * (bs["eyeLookInLeft"] + bs["eyeLookOutRight"])
    return _clamp(v, -1, 1), _clamp(h, -1, 1)


def iris_offset(mesh: Sequence | None) -> tuple[float, float] | None:
    """(v, h) iris-centre offset from the eye-corner midpoint / eye width, both
    eyes averaged. *mesh* in isotropic (pixel) coordinates, y down => +v down."""
    if mesh is None or len(mesh) < 478:
        return None

    def one(iris: int, corners: tuple[int, int]):
        cx, cy = _xy(mesh[iris])
        ax, ay = _xy(mesh[corners[0]])
        bx, by = _xy(mesh[corners[1]])
        w = math.hypot(ax - bx, ay - by)
        if not w > 0:
            return None
        return (cy - (ay + by) / 2) / w, (cx - (ax + bx) / 2) / w

    r = one(MESH_RIGHT_IRIS_CENTER, MESH_RIGHT_EYE_CORNERS)
    left = one(MESH_LEFT_IRIS_CENTER, MESH_LEFT_EYE_CORNERS)
    if r is None or left is None:
        return None
    return (r[0] + left[0]) / 2, (r[1] + left[1]) / 2


@dataclass
class GazeEstimate:
    gaze_v: float | None
    gaze_h: float | None
    iris_valid: bool
    gaze_valid: bool


def estimate_gaze(bs: Mapping[str, float] | None, mesh: Sequence | None) -> GazeEstimate:
    g = gaze_from_blendshapes(bs)
    iris = iris_offset(mesh)
    iris_valid = (
        iris is not None and all(math.isfinite(x) for x in iris)
        and abs(iris[0]) <= IRIS_MAX_OFFSET and abs(iris[1]) <= IRIS_MAX_OFFSET
    )
    if g is None:
        return GazeEstimate(None, None, iris_valid, False)
    consistent = True
    if iris_valid and iris is not None:
        if g[0] >= GAMMA_DOWN and iris[0] < -IRIS_SIGN_TOL:
            consistent = False
        if g[0] <= -GAMMA_DOWN and iris[0] > IRIS_SIGN_TOL:
            consistent = False
    return GazeEstimate(g[0], g[1], iris_valid, iris_valid and consistent)


def is_looking_down(rel_pitch_down_deg: float | None, gaze_v: float | None, gaze_valid: bool) -> bool:
    """D_t (looking-down.ts), both clauses inclusive; measurement convention (+down)."""
    head = rel_pitch_down_deg is not None and rel_pitch_down_deg >= THETA_DOWN_DEG
    eyes = gaze_valid and gaze_v is not None and gaze_v >= GAMMA_DOWN
    return head or eyes


# ── Eye openness (eye-openness.ts) ────────────────────────────────────────────

def mesh_ear(mesh: Sequence | None) -> float | None:
    """6-point EAR on mesh landmarks (isotropic coordinates), both eyes averaged."""
    if mesh is None or len(mesh) < 468:
        return None

    def one(idx):
        p = [_xy(mesh[i]) for i in idx]
        d = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])  # noqa: E731
        horiz = d(p[0], p[3])
        if not horiz > 0:
            return None
        return (d(p[1], p[5]) + d(p[2], p[4])) / (2 * horiz)

    r, left = one(MESH_RIGHT_EYE_EAR), one(MESH_LEFT_EYE_EAR)
    if r is None or left is None:
        return None
    return (r + left) / 2


def _robust_scale(res: np.ndarray) -> float:
    med = float(np.median(res))
    mad = float(np.median(np.abs(res - med)))
    return max(1.4826 * mad, 1e-9)


def huber_fit_quadratic(gs: Sequence[float] | np.ndarray, es: Sequence[float] | np.ndarray,
                        delta: float = HUBER_DELTA,
                        max_iter: int = 100) -> tuple[float, float, float] | None:
    """Huber IRLS fit of e = b0 + b1 g + b2 g². None if singular / < 3 samples."""
    g = np.asarray(gs, dtype=np.float64)
    e = np.asarray(es, dtype=np.float64)
    if len(g) < 3:
        return None
    x = np.stack([np.ones_like(g), g, g * g], axis=1)
    w = np.ones_like(g)
    coef = None
    for _ in range(max_iter):
        a = (x * w[:, None]).T @ x
        b = (x * w[:, None]).T @ e
        if abs(np.linalg.det(a)) < 1e-12:
            return None
        nxt = np.linalg.solve(a, b)
        res = e - x @ nxt
        k = delta * _robust_scale(res)
        w = np.where(np.abs(res) <= k, 1.0, k / np.maximum(np.abs(res), 1e-300))
        converged = coef is not None and np.all(np.abs(nxt - coef) < 1e-10)
        coef = nxt
        if converged:
            break
    return None if coef is None else (float(coef[0]), float(coef[1]), float(coef[2]))


def huber_location(values: Sequence[float], delta: float = HUBER_DELTA, max_iter: int = 100) -> float | None:
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return None
    mu = float(np.median(v))
    for _ in range(max_iter):
        res = v - mu
        k = delta * _robust_scale(res)
        w = np.where(np.abs(res) <= k, 1.0, k / np.maximum(np.abs(res), 1e-300))
        nxt = float(np.sum(w * v) / np.sum(w))
        if abs(nxt - mu) < 1e-12:
            return nxt
        mu = nxt
    return mu


class OpennessModel:
    """Online Ê(g) model. Feed (g, EAR) from attentive, eyes-open frames; refit() periodically."""

    def __init__(self, max_samples: int = 3000):
        self._samples: deque[tuple[float, float]] = deque(maxlen=max_samples)
        self.coef: tuple[float, float, float] = EAR_PRIOR
        self.kind = "prior"

    def add_sample(self, g: float | None, ear: float | None) -> None:
        if ear is None or not math.isfinite(ear) or ear <= 0:
            return
        self._samples.append((0.0 if g is None else g, ear))

    def refit(self) -> str:
        n = len(self._samples)
        if n < OPENNESS_MIN_SAMPLES:
            self.coef, self.kind = EAR_PRIOR, "prior"
            return self.kind
        gs = [s[0] for s in self._samples]
        es = [s[1] for s in self._samples]
        if max(gs) - min(gs) >= OPENNESS_MIN_GAZE_RANGE:
            q = huber_fit_quadratic(gs, es)
            if q is not None and q[0] > 0:
                self.coef, self.kind = q, "quadratic"
                return self.kind
        b0, b1, b2 = EAR_PRIOR
        level = huber_location([e / ((b0 + b1 * g + b2 * g * g) / b0) for g, e in zip(gs, es)])
        if level is not None and level > 0:
            self.coef = (level, level * b1 / b0, level * b2 / b0)
            self.kind = "level"
        return self.kind

    def expected(self, g: float | None) -> float:
        gg = 0.0 if g is None else g
        b0, b1, b2 = self.coef
        return b0 + b1 * gg + b2 * gg * gg

    def residual(self, ear: float | None, g: float | None) -> float | None:
        if ear is None or not math.isfinite(ear):
            return None
        e = self.expected(g)
        return ear / e if e > 0 else None

    @staticmethod
    def is_closed(residual: float | None) -> bool | None:
        return None if residual is None else residual < K_OPEN


class CorrectedPerclos:
    """Share of closed frames (r < k) over the trailing 60 s; frames with no valid openness are skipped."""

    def __init__(self, window_ms: int = PERCLOS_WINDOW_MS, min_samples: int = PERCLOS_MIN_SAMPLES):
        self._hist: deque[tuple[int, bool | None]] = deque()
        self._window_ms = window_ms
        self._min = min_samples

    def push(self, t_ms: int, closed: bool | None) -> None:
        self._hist.append((t_ms, closed))
        cutoff = t_ms - self._window_ms
        while self._hist and self._hist[0][0] <= cutoff:
            self._hist.popleft()

    def value(self) -> tuple[float, int]:
        valid = [c for _, c in self._hist if c is not None]
        n = len(valid)
        if n < self._min:
            return 0.0, n
        return sum(1 for c in valid if c) / n, n

    def longest_closure_ms(self, since_ms: int) -> int:
        best, start, last = 0, None, None
        for t, c in list(self._hist) + [(None, None)]:
            if c is True:
                start = t if start is None else start
                last = t
                continue
            if start is not None and last is not None and last >= since_ms:
                best = max(best, last - start + CLOSURE_FRAME_MS)
            start = last = None
        return best


# ── Pitch reference (pitch-reference.ts) ───────────────────────────────────────

def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if len(values) else None


@dataclass
class PitchReferenceState:
    ref_deg: float | None
    rel_pitch_deg: float | None
    calibrating: bool
    reset: str | None


class PitchReference:
    """
    θ_ref = median pitch over the first REFERENCE_MS of screen-facing frames
    (frames whose valid gaze is already looking down are excluded). Re-estimated
    after ABSENT_FRAMES consecutive no-face frames and after a camera move
    (jump between consecutive 1 s block medians of pitch or face scale). While
    (re-)estimating, the running median of the collected samples is used.
    Time-based blocks (v1.0 uses fps frames) because v2.0's analysis rate varies.
    """

    def __init__(self) -> None:
        self.ref_deg: float | None = None
        self._collecting = True
        self._collect_start: int | None = None
        self._samples: list[float] = []
        self._missing = 0
        self._absent_pending = False
        self._pending_reset: str | None = "start"
        self._block_start: int | None = None
        self._block_pitch: list[float] = []
        self._block_scale: list[float] = []
        self._prev_block: tuple[float | None, float | None] | None = None

    def _start(self, reason: str) -> None:
        self._collecting = True
        self._collect_start = None
        self._samples = []
        self._pending_reset = reason
        self._prev_block = None
        self._block_start = None
        self._block_pitch, self._block_scale = [], []

    def mark_absent(self) -> None:
        self._absent_pending = True

    @property
    def calibrating(self) -> bool:
        return self._collecting

    def update(self, t_ms: int, pitch_deg: float | None, face_scale: float | None,
               gaze_v: float | None) -> PitchReferenceState:
        if pitch_deg is None:
            self._missing += 1
            if self._missing >= ABSENT_FRAMES:
                self._absent_pending = True
            return self._state(None, None)

        if self._absent_pending:
            self._absent_pending = False
            self._start("absent_return")
        self._missing = 0

        if not self._collecting:
            if self._block_start is None:
                self._block_start = t_ms
            self._block_pitch.append(pitch_deg)
            if face_scale is not None:
                self._block_scale.append(face_scale)
            if t_ms - self._block_start >= BLOCK_MS:
                cur = (_median(self._block_pitch), _median(self._block_scale))
                if self._prev_block is not None:
                    pp, ps = self._prev_block
                    pitch_jump = cur[0] is not None and pp is not None and \
                        abs(cur[0] - pp) >= CAMERA_MOVE_PITCH_JUMP_DEG
                    scale_jump = cur[1] is not None and ps is not None and ps > 0 and \
                        abs(cur[1] / ps - 1) >= CAMERA_MOVE_FACE_SCALE_JUMP
                    if pitch_jump or scale_jump:
                        self._start("camera_move")
                if not self._collecting:
                    self._prev_block = cur
                self._block_start = None
                self._block_pitch, self._block_scale = [], []

        if self._collecting:
            if self._collect_start is None:
                self._collect_start = t_ms
            eyes_down = gaze_v is not None and gaze_v >= GAMMA_DOWN
            if not eyes_down:
                self._samples.append(pitch_deg)
            if t_ms - self._collect_start >= REFERENCE_MS and self._samples:
                self.ref_deg = _median(self._samples)
                self._collecting = False
                self._samples = []

        reset, self._pending_reset = self._pending_reset, None
        return self._state(pitch_deg, reset)

    def _state(self, pitch_deg: float | None, reset: str | None) -> PitchReferenceState:
        ref = (_median(self._samples) if self._samples else self.ref_deg) if self._collecting else self.ref_deg
        rel = None if pitch_deg is None or ref is None else pitch_deg - ref
        return PitchReferenceState(ref, rel, self._collecting, reset)


# ── Per-track measurement ─────────────────────────────────────────────────────

@dataclass
class FaceObservation:
    """One face from perception/face_landmarks.py::FaceLandmarksDetector.detect_full.
    landmarks68 / mesh in FRAME pixel coordinates (isotropic)."""
    landmarks68: list
    mesh: list | None = None
    matrix: np.ndarray | None = None
    blendshapes: dict[str, float] | None = None


@dataclass
class FaceMeasurement:
    """Per-frame output. Measurement convention (+down) unless named *_feature."""
    face_present: bool
    pitch_abs_deg: float | None = None
    rel_pitch_down_deg: float | None = None
    yaw_deg: float | None = None
    roll_deg: float | None = None
    gaze_v: float | None = None
    gaze_h: float | None = None
    gaze_valid: bool = False
    ear_raw: float | None = None
    residual_openness: float | None = None
    eye_closed: bool | None = None
    perclos: float = 0.0                 # gaze-corrected, fraction 0-1
    longest_closure_ms: int = 0
    looking_down: bool = False
    calibrating: bool = True
    pitch_source: str | None = None      # "matrix" | "legacy68" | None
    gaze_source: str | None = None       # "blendshape" | "legacy68" | None
    # feature/rule convention (down = negative) -- what FaceMetricsResult carries
    pitch_feature: float | None = None
    gaze_x_feature: float | None = None
    gaze_y_feature: float | None = None
    ear_feature: float | None = None     # residual × EAR_SCALE (gaze-corrected), else raw EAR


@dataclass
class DaceTrackMeasurer:
    """Stateful DACE measurement for ONE tracked person (pitch reference,
    openness model, corrected PERCLOS)."""
    pitch_ref: PitchReference = field(default_factory=PitchReference)
    openness: OpennessModel = field(default_factory=OpennessModel)
    perclos: CorrectedPerclos = field(default_factory=CorrectedPerclos)
    _frames: int = 0

    def update(self, t_ms: int, obs: FaceObservation | None, *, attentive: bool = False) -> FaceMeasurement:
        self._frames += 1
        if obs is None or not obs.landmarks68:
            st = self.pitch_ref.update(t_ms, None, None, None)
            self.perclos.push(t_ms, None)
            p, _ = self.perclos.value()
            return FaceMeasurement(face_present=False, perclos=p, calibrating=st.calibrating,
                                   longest_closure_ms=self.perclos.longest_closure_ms(t_ms - 3_000))

        lm = obs.landmarks68
        # Head pose: matrix (preferred) or the clearly-labelled legacy 68-point fallback.
        euler = euler_from_matrix(obs.matrix) if obs.matrix is not None else None
        pitch: float | None
        yaw: float | None
        roll: float | None
        pitch_source: str | None
        if euler is not None:
            pitch, yaw, roll = euler
            pitch_source = "matrix"
        else:
            # LEGACY fallback: not degrees and rises when the head goes down
            # (features/face_metrics.py docstring). Same reference + flip as the
            # matrix path; cannot reach THETA_DOWN_DEG for real tilts.
            lyaw, lpitch, lroll = calculate_head_pose(lm)
            pitch, yaw, roll = lpitch, lyaw, lroll
            pitch_source = "legacy68" if lpitch is not None else None

        # Gaze: blendshapes + iris cross-check (preferred) or legacy eyelid centroid.
        gaze_source = None
        gz = estimate_gaze(obs.blendshapes, obs.mesh)
        gaze_v = gaze_h = None
        gaze_valid = False
        gaze_x_feat = gaze_y_feat = None
        if obs.blendshapes is not None:
            if gz.gaze_valid:
                gaze_v, gaze_h, gaze_valid = gz.gaze_v, gz.gaze_h, True
                gaze_source = "blendshape"
                gaze_x_feat, gaze_y_feat = gaze_h, to_feature_gaze_y(gaze_v)
        else:
            lgx, lgy = calculate_gaze(lm)
            if lgx is not None:
                gaze_source = "legacy68"
                gaze_x_feat, gaze_y_feat = lgx, lgy   # eyelid-centroid gaze_y is already down = negative

        # Eye openness and gaze-corrected PERCLOS.
        ear = mesh_ear(obs.mesh) if obs.mesh is not None else calculate_ear(lm)
        g_for_openness = gaze_v if gaze_valid else None
        residual = self.openness.residual(ear, g_for_openness)
        closed = OpennessModel.is_closed(residual)
        self.perclos.push(t_ms, closed)

        face_scale = None
        if obs.mesh is not None and len(obs.mesh) >= 468:
            ax, ay = _xy(obs.mesh[33])
            bx, by = _xy(obs.mesh[263])
            face_scale = math.hypot(ax - bx, ay - by)
        elif len(lm) >= 46:
            ax, ay = _xy(lm[36])
            bx, by = _xy(lm[45])
            face_scale = math.hypot(ax - bx, ay - by)

        st = self.pitch_ref.update(t_ms, pitch, face_scale, g_for_openness)
        rel = st.rel_pitch_deg
        looking_down = is_looking_down(rel, gaze_v, gaze_valid)

        if ear is not None and attentive and closed is not True:
            self.openness.add_sample(g_for_openness, ear)
        if self._frames % OPENNESS_REFIT_EVERY == 0:
            self.openness.refit()

        perc, _ = self.perclos.value()
        return FaceMeasurement(
            face_present=True,
            pitch_abs_deg=pitch, rel_pitch_down_deg=rel, yaw_deg=yaw, roll_deg=roll,
            gaze_v=gaze_v, gaze_h=gaze_h, gaze_valid=gaze_valid,
            ear_raw=ear, residual_openness=residual, eye_closed=closed,
            perclos=perc, longest_closure_ms=self.perclos.longest_closure_ms(t_ms - 3_000),
            looking_down=looking_down, calibrating=st.calibrating,
            pitch_source=pitch_source, gaze_source=gaze_source,
            pitch_feature=to_feature_pitch(rel),
            gaze_x_feature=gaze_x_feat, gaze_y_feature=gaze_y_feat,
            ear_feature=(residual * EAR_SCALE) if residual is not None else ear,
        )
