"""
BehaviorClassifier — 15-frame temporal behavior classifier.
(ARCHITECTURE.md §3 Behavior Inference Layer + §5.3)

Mirrors temporal-classifier.ts but for Python / v2.0 server context.
Rule-based by default; plugs in ONNX 1D-CNN when model file is present.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass
from pathlib import Path

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
FEATURE_DIM  = 13


@dataclass
class BehaviorResult:
    label: str
    attention_score: int
    confidence: float
    source: str   # "temporal" or "rule_based"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class BehaviorClassifier:
    def __init__(self, model_path: Path | None = None):
        self._buffer: deque[list[float]] = deque(maxlen=WINDOW_SIZE)
        self._session = None
        self._last = BehaviorResult("distracted", 45, 0.0, "rule_based")

        if model_path and Path(model_path).exists():
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(str(model_path))
            except Exception as e:
                import warnings
                warnings.warn(f"[BehaviorClassifier] ONNX load failed: {e}")

    def update(self, frame: dict) -> BehaviorResult:
        """
        frame keys: metrics (FaceMetricsResult), detected_objects (list),
                    hand_proximity (str), posture (PostureResult | None)
        """
        self._buffer.append(self._extract(frame))
        if len(self._buffer) < WINDOW_SIZE:
            self._last = self._single_frame(frame)
        elif self._session:
            self._last = self._onnx_classify()
        else:
            self._last = self._temporal_rule()
        return self._last

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract(self, frame: dict) -> list[float]:
        m   = frame["metrics"]
        objs = frame.get("detected_objects", [])
        prox = frame.get("hand_proximity", "hands_not_visible")

        has_book  = 1 if any(o.cls in ("book","notebook") for o in objs) else 0
        has_phone = 1 if any(o.cls == "phone" for o in objs) else 0
        has_lap   = 1 if any(o.cls == "laptop" for o in objs) else 0
        has_none  = 0 if (has_book or has_phone or has_lap) else 1

        prox_map = {"holding_book":1.0,"holding_pen":0.8,"hands_on_desk":0.5,
                    "holding_phone":0.2,"hands_not_visible":0.0}

        return [
            _clamp(m.gaze_x or 0, -1, 1),
            _clamp(m.gaze_y or 0, -1, 1),
            _clamp((m.pitch or 0) / 90, -1, 1),
            _clamp((m.yaw   or 0) / 90, -1, 1),
            _clamp(m.ear or 0.3, 0, 1),
            _clamp(m.perclos / 100, 0, 1),
            _clamp(m.head_pose_stability / 100, 0, 1),
            has_book, has_phone, has_lap, has_none,
            _clamp(m.fidget_score / 100, 0, 1),
            prox_map.get(prox, 0.0),
        ]

    # ── Rule-based classifiers ────────────────────────────────────────────────

    def _single_frame(self, frame: dict) -> BehaviorResult:
        m    = frame["metrics"]
        objs = frame.get("detected_objects", [])
        prox = frame.get("hand_proximity", "hands_not_visible")

        if not m.face_area or m.face_area < 0.005:
            return self._make("absent", 1.0)
        if (m.ear or 0.3) < 0.22 and m.perclos > 20:
            return self._make("drowsy", 0.85)

        pitch = m.pitch or 0
        phone = any(o.cls == "phone" for o in objs)
        book  = any(o.cls in ("book","notebook") for o in objs)

        if phone and (prox == "holding_phone" or pitch < -15):
            return self._make("using_phone", 0.80)
        if book and (prox == "holding_book" or pitch < -15):
            return self._make("attentive_reading", 0.75)

        gaze_dev = ((m.gaze_x or 0)**2 + (m.gaze_y or 0)**2)**0.5
        if gaze_dev < 0.3 and abs(pitch) < 15:
            return self._make("attentive_screen", 0.70)

        return self._make("distracted", 0.60)

    def _temporal_rule(self) -> BehaviorResult:
        arr = np.mean(list(self._buffer), axis=0)
        gx, gy, pn, _, ear, perclos, stab, book, phone, _, _, _, prox = arr
        pitch = pn * 90

        if ear < 0.22 and perclos > 0.2:       return self._make("drowsy", 0.88)
        if phone > 0.5 and prox < 0.3:         return self._make("using_phone", 0.83)
        if book  > 0.5 and pitch < -15:        return self._make("attentive_reading", 0.82)
        gaze_dev = (gx**2 + gy**2)**0.5
        if gaze_dev < 0.3 and abs(pitch) < 15 and stab > 0.6:
            return self._make("attentive_screen", 0.78)
        return self._make("distracted", 0.65)

    def _onnx_classify(self) -> BehaviorResult:
        flat = np.array(list(self._buffer), dtype=np.float32).reshape(1, WINDOW_SIZE, FEATURE_DIM)
        try:
            probs = self._session.run(None, {"input": flat})[0][0]
            idx   = int(np.argmax(probs))
            return self._make(BEHAVIOR_LABELS[idx], float(probs[idx]), source="temporal")
        except Exception:
            return self._temporal_rule()

    def _make(self, label: str, conf: float, source: str = "") -> BehaviorResult:
        s = source or ("temporal" if len(self._buffer) >= WINDOW_SIZE else "rule_based")
        return BehaviorResult(label, ATTENTION_SCORES[label], conf, s)
