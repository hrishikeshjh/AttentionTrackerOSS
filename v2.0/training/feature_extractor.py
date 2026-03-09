"""
feature_extractor.py — Convert FaceMetricsResult + context into the 13-dim
feature vector used by the 1D-CNN and temporal-classifier.ts.

This mirrors the feature extraction logic in temporal-classifier.ts so that
models trained here are directly compatible with the inference runtime.

Feature vector (FEATURE_DIM = 13, all values in [0, 1]):
    0  gaze_x           — horizontal gaze deviation (0=left, 0.5=center, 1=right)
    1  gaze_y           — vertical gaze deviation   (0=up,   0.5=center, 1=down)
    2  pitch_norm       — head pitch, normalised to [-1, 1] → [0, 1]
    3  yaw_norm         — head yaw,   normalised to [-1, 1] → [0, 1]
    4  ear              — eye aspect ratio [0, 1]
    5  perclos          — PERCLOS % / 100 → [0, 1]
    6  stability        — head pose stability / 100 → [0, 1]
    7  has_book         — 1.0 if book/notebook detected
    8  has_phone        — 1.0 if phone detected
    9  has_laptop       — 1.0 if laptop detected
    10 has_none         — 1.0 if no study object detected
    11 fidget           — fidget score / 100 → [0, 1]
    12 proximity_code   — encoded hand-object proximity [0, 1]
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.face_metrics import FaceMetricsResult

# Proximity encoding (same as classroom_loader.py and temporal-classifier.ts)
_PROXIMITY_CODE: dict[str, float] = {
    "hands_not_visible": 0.00,
    "hands_on_desk":     0.25,
    "holding_pen":       0.50,
    "holding_book":      0.75,
    "holding_phone":     1.00,
}

FEATURE_DIM = 13


def extract_feature_vector(
    metrics: FaceMetricsResult,
    detected_objects: list | None = None,
    hand_proximity: str = "hands_not_visible",
) -> np.ndarray:
    """
    Build the 13-dim feature vector from a FaceMetricsResult.

    Args:
        metrics:          FaceMetricsResult from face_metrics.py.
        detected_objects: List of detected object dicts with `.cls` attribute.
        hand_proximity:   String code for hand-object proximity.

    Returns:
        np.ndarray of shape (13,), dtype float32, all values in [0, 1].
    """
    if detected_objects is None:
        detected_objects = []

    def _safe(val: float | None, lo: float = 0.0, hi: float = 1.0, default: float = 0.5) -> float:
        if val is None:
            return default
        return float(np.clip(val, lo, hi))

    # Gaze: raw deviation stored as signed [-1, 1]; remap to [0, 1]
    gaze_x = _safe(metrics.gaze_x, -1.0, 1.0, 0.0) * 0.5 + 0.5
    gaze_y = _safe(metrics.gaze_y, -1.0, 1.0, 0.0) * 0.5 + 0.5

    # Head pose: pitch/yaw in degrees [-90, 90]; normalise to [0, 1]
    pitch_norm = (_safe(metrics.pitch, -90.0, 90.0, 0.0) / 90.0) * 0.5 + 0.5
    yaw_norm   = (_safe(metrics.yaw,   -90.0, 90.0, 0.0) / 90.0) * 0.5 + 0.5

    ear       = _safe(metrics.ear,                0.0, 0.5,  0.3)   / 0.5  # norm 0-1
    perclos   = _safe(metrics.perclos,            0.0, 100.0, 5.0)  / 100.0
    stability = _safe(metrics.head_pose_stability, 0.0, 100.0, 90.0) / 100.0
    fidget    = _safe(metrics.fidget_score,        0.0, 100.0, 5.0)  / 100.0

    # Object one-hots
    obj_classes = {getattr(o, "cls", o) for o in detected_objects}
    has_book   = float("book" in obj_classes or "notebook" in obj_classes)
    has_phone  = float("phone" in obj_classes)
    has_laptop = float("laptop" in obj_classes)
    has_none   = float(not (has_book or has_phone or has_laptop))

    proximity_code = _PROXIMITY_CODE.get(hand_proximity, 0.0)

    return np.array(
        [gaze_x, gaze_y, pitch_norm, yaw_norm, ear, perclos, stability,
         has_book, has_phone, has_laptop, has_none, fidget, proximity_code],
        dtype=np.float32,
    )


def build_input_tensor(
    feature_window: np.ndarray,
    device: str = "cpu",
):
    """
    Convert a (WINDOW_SIZE, FEATURE_DIM) numpy window to a PyTorch tensor
    in (1, FEATURE_DIM, WINDOW_SIZE) channel-first format.
    """
    import torch
    x = torch.from_numpy(feature_window).float()  # (W, F)
    x = x.T.unsqueeze(0)                          # (1, F, W)
    return x.to(device)
