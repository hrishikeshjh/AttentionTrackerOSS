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

# Proximity codes (matches feature_extractor.py and temporal-classifier.ts)
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

    def reset(self) -> None:
        """Clear the temporal window and reset state (call at session start)."""
        self._buffer.clear()
        self._last = BehaviorResult("distracted", 45, 0.0, "rule_based")

    def update(self, frame: dict) -> BehaviorResult:
        """
        frame keys: metrics (FaceMetricsResult), detected_objects (list),
                    hand_proximity (str), posture (PostureResult | None)
        """
        self._buffer.append(self._extract(frame))
        if len(self._buffer) < WINDOW_SIZE:
            self._last = self._single_frame(frame)
        elif self._session:
            self._last = self._onnx_classify(frame)
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
            _PROX_CODE.get(prox, 0.0),
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

        if ear < 0.22 and perclos > 0.2:                return self._make("drowsy", 0.88)
        # prox >= 0.9 means holding_phone (1.0 in new scale)
        if phone > 0.5 and prox >= 0.9:                 return self._make("using_phone", 0.90)
        if phone > 0.5 and pitch < -15:                 return self._make("using_phone", 0.83)
        # prox >= 0.7 means holding_book (0.75 in new scale)
        if book > 0.5 and prox >= 0.7 and phone < 0.1: return self._make("attentive_reading", 0.88)
        if book > 0.5 and pitch < -15:                  return self._make("attentive_reading", 0.82)
        gaze_dev = (gx**2 + gy**2)**0.5
        if gaze_dev < 0.3 and abs(pitch) < 15 and stab > 0.6:
            return self._make("attentive_screen", 0.78)
        return self._make("distracted", 0.65)

    def _onnx_classify(self, frame: dict | None = None) -> BehaviorResult:
        # shape: (1, FEATURE_DIM, WINDOW_SIZE) — channel-first for Conv1D
        win = np.array(list(self._buffer), dtype=np.float32)  # (W, F)
        inp = win.T[np.newaxis, :, :]                          # (1, F, W)
        try:
            logits = self._session.run(None, {"input": inp})[0][0]   # (6,)
            probs  = np.exp(logits) / np.exp(logits).sum()           # softmax
            probs  = self._context_gate(probs, frame)
            idx    = int(np.argmax(probs))
            return self._make(BEHAVIOR_LABELS[idx], float(probs[idx]), source="temporal")
        except Exception:
            return self._temporal_rule()

    # ── Context-gate override (P1 improvement from experiment_report.md) ─────

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

        Expected improvement (from experiment_report.md §4.1):
          phone precision  +10–15pp
          reading recall   +8–12pp
        """
        if frame is None:
            return probs

        objs = frame.get("detected_objects", [])
        prox = frame.get("hand_proximity", "hands_not_visible")

        has_phone = any(getattr(o, "cls", None) == "phone" for o in objs)
        has_book  = any(getattr(o, "cls", None) in ("book", "notebook") for o in objs)
        prox_code = _PROX_CODE.get(prox, 0.0)

        p = probs.copy()

        # Hard phone override: phone detected + holding_phone proximity
        if has_phone and prox_code >= 0.9:
            if p[_IDX["using_phone"]] < gate_threshold:
                p[_IDX["using_phone"]]      = max(p[_IDX["using_phone"]], 0.85)
                p[_IDX["attentive_reading"]] = min(p[_IDX["attentive_reading"]], 0.05)
                p /= p.sum()

        # Hard reading override: book detected + holding_book, no phone
        elif has_book and prox_code >= 0.7 and not has_phone:
            if p[_IDX["attentive_reading"]] < gate_threshold:
                p[_IDX["attentive_reading"]] = max(p[_IDX["attentive_reading"]], 0.80)
                p[_IDX["using_phone"]]       = min(p[_IDX["using_phone"]], 0.08)
                p /= p.sum()

        # Ambiguity flag: downward gaze with no object context → reduce confidence
        elif (not has_phone and not has_book
              and len(list(self._buffer)) > 0):
            avg_pitch = float(np.mean([f[2] for f in self._buffer]))
            if avg_pitch < -0.15:   # pitch_norm < -0.15 ≈ pitch < −14°
                # Don't change the argmax, just compress confidence
                p = p * 0.7 + (1.0 / len(p)) * 0.3  # pull toward uniform
                p /= p.sum()

        return p

    def _make(self, label: str, conf: float, source: str = "") -> BehaviorResult:
        s = source or ("temporal" if len(self._buffer) >= WINDOW_SIZE else "rule_based")
        return BehaviorResult(label, ATTENTION_SCORES[label], conf, s)
