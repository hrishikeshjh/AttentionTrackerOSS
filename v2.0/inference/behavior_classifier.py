"""
BehaviorClassifier — 15-frame temporal behavior classifier.
(ARCHITECTURE.md §3 Behavior Inference Layer + §5.3)

Python counterpart of v1.0's temporal-classifier.ts (v1.0 legacy path, DACE
off) for the v2.0 server. Rule-based by default; plugs in ONNX 1D-CNN when
model file is present. The rules are similar but not identical to the TS
rules, and the two surfaces use different proximity codes in their rule
layers (see _PROX_CODE below).

Sign convention (features/feature_vector.py): metrics.pitch is RELATIVE
pitch in degrees and metrics.gaze_y vertical gaze, both DOWN = NEGATIVE.
RoomAnalyzer fills them from features/dace_measurement.py (MediaPipe
transformation matrix, per-track screen-facing reference, blendshape gaze
with iris cross-check), which flips DACE's +down measurement exactly once.
"Looking down" here is DACE's D_t in this convention:
    pitch <= -THETA_DOWN_DEG (15)  or  gaze_y <= -GAMMA_DOWN (0.5)
(inclusive, like v1.0 looking-down.ts). Before this change pitch came from
the legacy 68-point formula, which rises when the head tilts down, so every
`pitch < -15` branch was unreachable (tests/test_pitch_characterisation.py).

Context evidence (window shares, DACE Annex A.2 / v1.0 evidence.ts):
    phone-in-use frame = phone detected and (holding_phone or looking down)
    book-evidence frame = book detected and (holding_book or looking down)
    phone in use  : share >= RHO_USE (0.6) of the window
    book          : share >= BOOK_SHARE (0.5) of the window, phone in < 10%
The hand_proximity labels come from features/hand_object.py (HandLandmarker,
IoU holding / near) or features/hand_proximity.py (pose wrists), selected by
COGNEN_HAND_PROXIMITY (default "pose").

Feature vector: FEATURE_VECTOR_VERSION 2 (15 dims = 13 v1.0-compatible +
posture) from features.feature_vector.encode_frame_v2. A 13-channel model
(pre-v3) is fed the first 13 channels; the channel count is read from the
ONNX session's input shape.

"Unsure": v1.0 DACE can abstain ("Unsure" + reason code U1-U4, score held).
v2.0 has no abstention; BehaviorResult always carries one of BEHAVIOR_LABELS.

"absent": a frame whose metrics carry ``face_detected is False`` (the face
landmarker ran on this person and found no face) is classified "absent" by
the single-frame rules; once the window is full, "absent" is returned when at
least ABSENT_MIN_NO_FACE_FRAMES of the last WINDOW_SIZE frames had no face
(both the rule path and the ONNX path). ``face_detected is None`` (unknown,
e.g. landmarker unavailable) never counts as "no face". Previously "absent"
was effectively unreachable on the server (face_area is the person-box
fraction, never < 0.005 for a tracked person, and the temporal rules had no
absent branch).

ONNX sessions: one InferenceSession per model file is shared by every
classifier (load_onnx_session caches it). RoomAnalyzer used to build one
session per tracked person.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from features.feature_vector import FEATURE_DIM as _FV_FEATURE_DIM
from features.feature_vector import FEATURE_DIM_V1, encode_frame_v2

logger = logging.getLogger("cognen.behavior")

BEHAVIOR_LABELS = [
    "attentive_screen",
    "attentive_reading",
    "using_phone",
    "distracted",
    "drowsy",
    "absent",
]

ATTENTION_SCORES: dict[str, int] = {
    "attentive_screen":  85,
    "attentive_reading": 80,
    "distracted":        45,
    "using_phone":       25,
    "drowsy":            20,
    "absent":             0,
}

WINDOW_SIZE  = 15
FEATURE_DIM  = _FV_FEATURE_DIM          # 15 (feature_vector.FEATURE_DIM_V2)

# DACE thresholds in the feature convention (down = negative)
THETA_DOWN_DEG = 15.0    # Table A.6 θ_down
GAMMA_DOWN     = 0.5     # Table A.6 γ_down
RHO_USE        = 0.6     # Table A.6 ρ_use: phone-in-use share of W
BOOK_SHARE     = 0.5     # prose: book evidence share of W
MOST_SHARE     = 0.5     # prose: "most" = strictly more than half of W

# Sustained no-face threshold for "absent" once the window is full
# (10 of 15 frames = 2/3 of the window).
ABSENT_MIN_NO_FACE_FRAMES = 10

# Proximity codes. Same table as features/feature_vector.py::_PROXIMITY_CODE
# (used for training) and as v1.0 temporal-classifier.ts PROXIMITY_CODES_V2,
# which v1.0 uses only when feeding a loaded ONNX model (loadModel default
# proximityCodes='v2'). v1.0's own rule layer and its 13-dim vector
# (encode13 / extractFeatures) use DIFFERENT codes (PROXIMITY_CODES_V1:
# book 1.0, pen 0.8, desk 0.5, phone 0.2, not visible 0.0). Checked against
# the generated fixture shared/fixtures/encode13_cross_language.json by
# tests/test_cross_language_contract.py.
_PROX_CODE: dict[str, float] = {
    "hands_not_visible": 0.00,
    "hands_on_desk":     0.25,
    "holding_pen":       0.50,
    "holding_book":      0.75,
    "holding_phone":     1.00,
}

# Class indices
_IDX = {lbl: i for i, lbl in enumerate(BEHAVIOR_LABELS)}


@dataclass
class BehaviorResult:
    label: str
    attention_score: int
    confidence: float
    source: str   # "temporal" or "rule_based"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (max-subtracted; large logits used to
    overflow np.exp to inf and yield NaN probabilities)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


_SESSION_CACHE: dict[str, object] = {}
_SESSION_LOCK = threading.Lock()


def load_onnx_session(model_path: Path | str | None):
    """
    Return a shared onnxruntime.InferenceSession for *model_path*, or None if
    the path is missing or onnxruntime / the model cannot be loaded (logged).
    Sessions are cached per resolved path: InferenceSession.run is
    thread-safe, and the model is stateless across calls.
    """
    if not model_path or not Path(model_path).exists():
        return None
    key = str(Path(model_path).resolve())
    with _SESSION_LOCK:
        if key in _SESSION_CACHE:
            return _SESSION_CACHE[key]
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(key)
        except Exception as e:
            logger.error("[BehaviorClassifier] ONNX load failed for %s: %s", key, e)
            return None
        _SESSION_CACHE[key] = sess
        return sess


def _face_missing(m) -> bool:
    """True only when the landmarker ran and found no face (None = unknown)."""
    return getattr(m, "face_detected", None) is False


def _session_feature_dim(session) -> int:
    """Input channel count of an ONNX session ((B, F, W) layout); FEATURE_DIM if unknown."""
    try:
        dim = session.get_inputs()[0].shape[1]
        return int(dim) if isinstance(dim, (int, np.integer)) and int(dim) in (FEATURE_DIM_V1, FEATURE_DIM) \
            else FEATURE_DIM
    except Exception:
        return FEATURE_DIM


def looking_down(pitch_deg: float, gaze_y: float) -> bool:
    """D_t in the feature convention (down = negative), both clauses inclusive."""
    return pitch_deg <= -THETA_DOWN_DEG or gaze_y <= -GAMMA_DOWN


class BehaviorClassifier:
    def __init__(self, model_path: Path | None = None, session=None):
        """
        model_path: ONNX model; loaded through the shared load_onnx_session
                    cache. Ignored when *session* is given.
        session:    an existing InferenceSession-like object (``run``) to
                    share between classifiers.
        """
        self._buffer: deque[list[float]] = deque(maxlen=WINDOW_SIZE)
        self._no_face: deque[bool] = deque(maxlen=WINDOW_SIZE)
        self._session = session if session is not None else load_onnx_session(model_path)
        self._model_dim = _session_feature_dim(self._session) if self._session is not None else FEATURE_DIM
        self._onnx_error_logged = False
        self._last = BehaviorResult("distracted", 45, 0.0, "rule_based")

    def reset(self) -> None:
        """Clear the temporal window and reset state (call at session start)."""
        self._buffer.clear()
        self._no_face.clear()
        self._last = BehaviorResult("distracted", 45, 0.0, "rule_based")

    def update(self, frame: dict) -> BehaviorResult:
        """
        frame keys: metrics (FaceMetricsResult), detected_objects (list),
                    hand_proximity (str), posture (PostureResult | None)
        """
        self._buffer.append(self._extract(frame))
        self._no_face.append(_face_missing(frame["metrics"]))
        if len(self._buffer) < WINDOW_SIZE:
            self._last = self._single_frame(frame)
        elif sum(self._no_face) >= ABSENT_MIN_NO_FACE_FRAMES:
            self._last = self._make("absent", sum(self._no_face) / len(self._no_face))
        elif self._session:
            self._last = self._onnx_classify(frame)
        else:
            self._last = self._temporal_rule()
        return self._last

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract(self, frame: dict) -> list[float]:
        # The shared encoder (features/feature_vector.py, version 2, 15 dims).
        # EAR None -> 0.3, EAR 0.0 stays 0.0 (D-EAR0, like v1.0 encode13).
        return encode_frame_v2(
            frame["metrics"],
            frame.get("detected_objects", []) or [],
            frame.get("hand_proximity", "hands_not_visible"),
            frame.get("posture"),
        ).tolist()

    # ── Window evidence ──────────────────────────────────────────────────────

    def _evidence(self) -> dict[str, float]:
        """Per-window shares (DACE evidence.ts, feature convention)."""
        buf = list(self._buffer)
        n = max(len(buf), 1)
        down = [looking_down(f[2] * 90.0, f[1]) for f in buf]
        phone_use = sum(1 for f, d in zip(buf, down) if f[8] > 0.5 and (f[12] >= 0.9 or d))
        phone_hold = sum(1 for f in buf if f[8] > 0.5 and f[12] >= 0.9)
        book_ev = sum(1 for f, d in zip(buf, down) if f[7] > 0.5 and (0.7 <= f[12] < 0.9 or d))
        phone_det = sum(1 for f in buf if f[8] > 0.5)
        return {
            "down": sum(down) / n,
            "phone_use": phone_use / n,
            "phone_hold": phone_hold / n,
            "book": book_ev / n,
            "phone_detected": phone_det / n,
        }

    # ── Rule-based classifiers ────────────────────────────────────────────────

    def _single_frame(self, frame: dict) -> BehaviorResult:
        m    = frame["metrics"]
        objs = frame.get("detected_objects", [])
        prox = frame.get("hand_proximity", "hands_not_visible")

        if _face_missing(m) or not m.face_area or m.face_area < 0.005:
            return self._make("absent", 1.0)
        if (0.3 if m.ear is None else m.ear) < 0.22 and m.perclos > 20:
            return self._make("drowsy", 0.85)

        pitch = m.pitch or 0
        down  = looking_down(pitch, m.gaze_y or 0)
        phone = any(o.cls == "phone" for o in objs)
        book  = any(o.cls in ("book","notebook") for o in objs)

        if phone and (prox == "holding_phone" or down):
            return self._make("using_phone", 0.80)
        if book and (prox == "holding_book" or down):
            return self._make("attentive_reading", 0.75)

        gaze_dev = ((m.gaze_x or 0)**2 + (m.gaze_y or 0)**2)**0.5
        if gaze_dev < 0.3 and abs(pitch) < THETA_DOWN_DEG:
            return self._make("attentive_screen", 0.70)

        return self._make("distracted", 0.60)

    def _temporal_rule(self) -> BehaviorResult:
        arr = np.mean(list(self._buffer), axis=0)
        gx, gy, pn, _, ear, perclos, stab = arr[:7]
        pitch = pn * 90
        ev = self._evidence()

        if ear < 0.22 and perclos > 0.2:
            return self._make("drowsy", 0.88)
        if ev["phone_use"] >= RHO_USE - 1e-9:
            return self._make("using_phone", 0.90 if ev["phone_hold"] >= RHO_USE - 1e-9 else 0.83)
        if ev["book"] >= BOOK_SHARE - 1e-9 and ev["phone_detected"] < 0.1:
            return self._make("attentive_reading", 0.88 if ev["down"] > MOST_SHARE else 0.82)
        gaze_dev = (gx**2 + gy**2)**0.5
        if ev["down"] <= MOST_SHARE and gaze_dev < 0.3 and abs(pitch) < THETA_DOWN_DEG and stab > 0.6:
            return self._make("attentive_screen", 0.78)
        return self._make("distracted", 0.65)

    def _onnx_classify(self, frame: dict | None = None) -> BehaviorResult:
        # shape: (1, FEATURE_DIM, WINDOW_SIZE) — channel-first for Conv1D
        win = np.array(list(self._buffer), dtype=np.float32)[:, : self._model_dim]  # (W, F)
        inp = np.ascontiguousarray(win.T[np.newaxis, :, :])                          # (1, F, W)
        try:
            logits = self._session.run(None, {"input": inp})[0][0]   # (6,)
            probs  = _softmax(logits)
            probs  = self._context_gate(probs, frame)
            idx    = int(np.argmax(probs))
            return self._make(BEHAVIOR_LABELS[idx], float(probs[idx]), source="temporal")
        except Exception as e:
            if not self._onnx_error_logged:
                logger.error("[BehaviorClassifier] ONNX inference failed, using rules (logged once): %s", e)
                self._onnx_error_logged = True
            return self._temporal_rule()

    # ── Context-gate override (proposed in the RETRACTED archive/experiment_report.md) ─

    def _context_gate(
        self,
        probs: np.ndarray,
        frame: dict | None,
        gate_threshold: float = 0.80,
    ) -> np.ndarray:
        """
        Post-process CNN probabilities with hard object-context overrides.

        When the object detector reports a clear phone or book signal, the CNN
        should not be able to override it with low confidence.  This gate
        boosts the contextually correct class and suppresses the opposite class
        when the current frame's evidence is unambiguous.

        Its effect has never been measured on real data (the "+10-15pp phone
        precision / +8-12pp reading recall" estimate came from the RETRACTED
        evaluation/reports/archive/experiment_report.md §4.1). evaluate.py
        now reports the deployed path (rules + gate) next to the raw ONNX
        model so the gate's effect is measurable once real data exists.
        """
        if frame is None:
            return probs

        objs = frame.get("detected_objects", [])
        prox = frame.get("hand_proximity", "hands_not_visible")

        has_phone = any(getattr(o, "cls", None) == "phone" for o in objs)
        has_book  = any(getattr(o, "cls", None) in ("book", "notebook") for o in objs)
        prox_code = _PROX_CODE.get(prox, 0.0)
        ev = self._evidence() if self._buffer else {"down": 0.0, "phone_use": 0.0, "book": 0.0,
                                                     "phone_detected": 0.0, "phone_hold": 0.0}

        p = probs.copy()

        # Hard phone override: this frame phone + holding_phone, or phone in use
        # (phone + held/looking down) in >= RHO_USE of the window.
        if (has_phone and prox_code >= 0.9) or ev["phone_use"] >= RHO_USE - 1e-9:
            if p[_IDX["using_phone"]] < gate_threshold:
                p[_IDX["using_phone"]]      = max(p[_IDX["using_phone"]], 0.85)
                p[_IDX["attentive_reading"]] = min(p[_IDX["attentive_reading"]], 0.05)
                p /= p.sum()

        # Hard reading override: this frame book + holding_book (no phone), or
        # book evidence (held/near or looking down) in >= BOOK_SHARE of W with
        # no phone in the window.
        elif ((has_book and 0.7 <= prox_code < 0.9 and not has_phone)
              or (ev["book"] >= BOOK_SHARE - 1e-9 and ev["phone_detected"] < 0.1 and not has_phone)):
            if p[_IDX["attentive_reading"]] < gate_threshold:
                p[_IDX["attentive_reading"]] = max(p[_IDX["attentive_reading"]], 0.80)
                p[_IDX["using_phone"]]       = min(p[_IDX["using_phone"]], 0.08)
                p /= p.sum()

        # Ambiguity flag: looking down in most of W with no object context ->
        # reduce confidence (DACE would say "Unsure" U1/U2; v2.0 has no abstention).
        elif not has_phone and not has_book and ev["down"] > MOST_SHARE:
            p = p * 0.7 + (1.0 / len(p)) * 0.3  # pull toward uniform, argmax unchanged
            p /= p.sum()

        return p

    def _make(self, label: str, conf: float, source: str = "") -> BehaviorResult:
        s = source or ("temporal" if len(self._buffer) >= WINDOW_SIZE else "rule_based")
        return BehaviorResult(label, ATTENTION_SCORES[label], conf, s)
