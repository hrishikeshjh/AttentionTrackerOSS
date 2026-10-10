"""
feature_vector.py — canonical 13-dim feature vector construction.

Consolidates what used to be THREE independently hand-reimplemented,
partially-inconsistent feature-extraction functions:
  - training/feature_extractor.py::extract_feature_vector
  - training/datasets/daisee_loader.py::_extract_frame_features
  - training/datasets/engagenet_loader.py::_frame_features
  - training/datasets/classroom_loader.py::_extract_features_from_crop
(each reimplemented its own crude EAR/pitch math instead of importing
calculate_ear/calculate_head_pose/calculate_gaze from features/face_metrics.py,
and disagreed with each other on units -- see the encoding bug below).

ENCODING BUG THIS FIXES: training/feature_extractor.py and
daisee_loader.py's gaze_x/gaze_y both remapped [-1,1] -> [0,1] via
`* 0.5 + 0.5`, while training/datasets/realistic_synthetic_loader.py (the
synthetic AR(1) generator actually used for training) and
inference/behavior_classifier.py::_extract (the actual inference-time
consumer) both expect gaze_x/gaze_y/pitch_norm/yaw_norm in [-1, 1]. This
was a live train/inference distribution-mismatch bug: a model trained
partly on [0,1]-encoded real-data windows and partly on [-1,1]-encoded
synthetic windows would learn a broken, inconsistent decision boundary.

Canonical feature vector (FEATURE_DIM = 13, index order fixed):
    0  gaze_x           [-1, 1]   horizontal gaze deviation (0=center)
    1  gaze_y           [-1, 1]   vertical gaze deviation   (0=center)
    2  pitch_norm       [-1, 1]   pitch / 90
    3  yaw_norm         [-1, 1]   yaw / 90
    4  ear              [ 0, 1]   eye aspect ratio (already ~0-0.5, clipped)
    5  perclos          [ 0, 1]   PERCLOS % / 100
    6  stability        [ 0, 1]   head pose stability / 100
    7  has_book         {0, 1}
    8  has_phone        {0, 1}
    9  has_laptop       {0, 1}
    10 has_none         {0, 1}
    11 fidget           [ 0, 1]   fidget score / 100
    12 proximity_code   {0.00, 0.25, 0.50, 0.75, 1.00}

Cross-language: indices 0-11 equal v1.0 encode13
(v1.0/src/lib/scoring/dace/feature-vector.ts). Index 12 equals v1.0's ONNX
input path (temporal-classifier.ts PROXIMITY_CODES_V2), NOT v1.0's rule-layer
codes (PROXIMITY_CODES_V1). Checked by tests/test_cross_language_contract.py
against shared/fixtures/encode13_cross_language.json (generated from the
TypeScript).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from features.face_metrics import calculate_ear, calculate_gaze, calculate_head_pose

FEATURE_DIM = 13

_PROXIMITY_CODE: dict[str, float] = {
    "hands_not_visible": 0.00,
    "hands_on_desk":     0.25,
    "holding_pen":       0.50,
    "holding_book":      0.75,
    "holding_phone":     1.00,
}


def _safe(val: float | None, lo: float, hi: float, default: float) -> float:
    if val is None:
        return default
    return float(np.clip(val, lo, hi))


def _object_one_hots(detected_objects: Sequence) -> tuple[float, float, float, float]:
    obj_classes = {getattr(o, "cls", o) for o in detected_objects}
    has_book   = float("book" in obj_classes or "notebook" in obj_classes)
    has_phone  = float("phone" in obj_classes)
    has_laptop = float("laptop" in obj_classes)
    has_none   = float(not (has_book or has_phone or has_laptop))
    return has_book, has_phone, has_laptop, has_none


def encode_feature_vector(
    *,
    gaze_x: float | None = 0.0,
    gaze_y: float | None = 0.0,
    pitch: float | None = 0.0,
    yaw: float | None = 0.0,
    ear: float | None = 0.3,
    perclos: float | None = 5.0,
    stability: float | None = 90.0,
    fidget: float | None = 5.0,
    detected_objects: Sequence = (),
    hand_proximity: str = "hands_not_visible",
) -> np.ndarray:
    """
    Assemble the canonical 13-dim vector from already-computed metrics
    (raw units: pitch/yaw in degrees, perclos/stability/fidget in 0-100).
    All normalization/unit-conversion happens HERE, once, so no caller
    hand-rolls its own (possibly inconsistent) remap.
    """
    gx = _safe(gaze_x, -1.0, 1.0, 0.0)
    gy = _safe(gaze_y, -1.0, 1.0, 0.0)
    pitch_norm = _safe(pitch, -90.0, 90.0, 0.0) / 90.0
    yaw_norm   = _safe(yaw,   -90.0, 90.0, 0.0) / 90.0

    ear_v       = _safe(ear, 0.0, 1.0, 0.3)
    perclos_v   = _safe(perclos, 0.0, 100.0, 5.0) / 100.0
    stability_v = _safe(stability, 0.0, 100.0, 90.0) / 100.0
    fidget_v    = _safe(fidget, 0.0, 100.0, 5.0) / 100.0

    has_book, has_phone, has_laptop, has_none = _object_one_hots(detected_objects)
    proximity_code = _PROXIMITY_CODE.get(hand_proximity, 0.0)

    return np.array(
        [gx, gy, pitch_norm, yaw_norm, ear_v, perclos_v, stability_v,
         has_book, has_phone, has_laptop, has_none, fidget_v, proximity_code],
        dtype=np.float32,
    )


def extract_feature_vector_from_landmarks(
    landmarks,
    detected_objects: Sequence = (),
    hand_proximity: str = "hands_not_visible",
) -> np.ndarray:
    """
    Canonical per-frame feature extraction from raw 68-point landmarks.
    Reuses the corrected calculate_ear/calculate_head_pose/calculate_gaze
    from features/face_metrics.py instead of a hand-rolled formula --
    every dataset loader should call this rather than reimplementing its
    own EAR/pitch/gaze math.
    """
    if not landmarks or len(landmarks) < 68:
        return encode_feature_vector(
            gaze_x=None, gaze_y=None, pitch=None, yaw=None, ear=None,
            detected_objects=detected_objects, hand_proximity=hand_proximity,
        )

    ear = calculate_ear(landmarks)
    yaw, pitch, _roll = calculate_head_pose(landmarks)
    gaze_x, gaze_y = calculate_gaze(landmarks)

    return encode_feature_vector(
        gaze_x=gaze_x, gaze_y=gaze_y, pitch=pitch, yaw=yaw, ear=ear,
        detected_objects=detected_objects, hand_proximity=hand_proximity,
    )
