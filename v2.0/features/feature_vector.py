"""
feature_vector.py — canonical, versioned per-frame feature vectors.

ONE encoder for training (dataset loaders, synthetic generator via the same
index layout), inference (inference/behavior_classifier.py) and the
cross-language contract (tests/test_cross_language_contract.py,
tests/test_feature_vector_v2_contract.py).

Versions
--------
FEATURE_VECTOR_VERSION = 2

  v1 (FEATURE_DIM_V1 = 13)  ``encode_feature_vector`` -- v1.0-browser
      compatible. Indices 0-11 equal v1.0 encode13
      (v1.0/src/lib/scoring/dace/feature-vector.ts); index 12 equals v1.0's
      ONNX input path (temporal-classifier.ts PROXIMITY_CODES_V2), not
      v1.0's rule-layer codes. Checked against the TypeScript-generated
      shared/fixtures/encode13_cross_language.json.

  v2 (FEATURE_DIM_V2 = 15)  ``encode_feature_vector_v2`` -- what the v2.0
      1D-CNN consumes (training/model.py, export_onnx.py,
      models/behavior_classifier_v3.onnx). First 13 = v1 exactly, plus
      posture from YOLOv8-pose (perception/pose_estimator.py::analyze_posture):
          13 posture_upright    {0, 1}   lean_angle < 15 deg
          14 posture_lean_norm  [0, 1]   lean_angle / 90 (degrees from vertical)
      Unknown posture (no pose matched, or a browser without pose) encodes as
      POSTURE_UNKNOWN = (1.0, 0.0), the same values analyze_posture returns
      when keypoints are missing. v1.0 loadModel(featureDim: 15) pads with
      these values (shared/fixtures/encode15_v2.json).

  FEATURE_DIM = FEATURE_DIM_V2 (the dimension the v2 model is trained on).
  Older 13-channel models (behavior_classifier.onnx, _v2.onnx) still load:
  the classifier feeds them the first 13 channels.

Layout (index order fixed):
    0  gaze_x           [-1, 1]   horizontal gaze (+ = toward image right)
    1  gaze_y           [-1, 1]   vertical gaze, EYES DOWN = NEGATIVE
    2  pitch_norm       [-1, 1]   relative pitch / 90, HEAD DOWN = NEGATIVE
    3  yaw_norm         [-1, 1]   yaw / 90
    4  ear              [ 0, 1]   eye openness (gaze-corrected residual x 0.3 on the server)
    5  perclos          [ 0, 1]   PERCLOS % / 100 (gaze-corrected on the server)
    6  stability        [ 0, 1]   head pose stability / 100
    7  has_book         {0, 1}
    8  has_phone        {0, 1}
    9  has_laptop       {0, 1}
    10 has_none         {0, 1}
    11 fidget           [ 0, 1]   fidget score / 100
    12 proximity_code   {0.00, 0.25, 0.50, 0.75, 1.00}
    13 posture_upright  {0, 1}            (v2 only)
    14 posture_lean     [ 0, 1]           (v2 only)

SIGN CONVENTION (single convention for vector, rules and training data)
-----------------------------------------------------------------------
Head down and eyes down are NEGATIVE in the vector and in the behavior rules
(inference/behavior_classifier.py: looking down <=> pitch <= -THETA_DOWN_DEG
or gaze_y <= -GAMMA_DOWN). The DACE measurement layer
(features/dace_measurement.py) works in +down degrees / blendshape units and
flips the sign exactly once (``to_feature_pitch`` / ``to_feature_gaze_y``).
``pitch`` is RELATIVE to a per-track screen-facing reference, so 0 = the
person's own screen-facing pose. This matches v1.0 DACE's legacy-field
substitution (plan D7: legacy ``pitch``/``gazeY`` receive the negated DACE
values), so v1.0 loadModel in DACE mode feeds the same signs. v1.0's LEGACY
measurement path (DACE off) feeds the uncorrected 68-point formula and is
NOT consistent with this convention (ARCHITECTURE.md §11.1).
The synthetic generator (training/datasets/realistic_synthetic_loader.py)
uses negative pitch/gaze_y means for reading and phone use.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from features.face_metrics import (
    calculate_ear,
    calculate_gaze,
    calculate_head_pose,
    legacy_pitch_to_feature,
)

FEATURE_VECTOR_VERSION = 2
FEATURE_DIM_V1 = 13
FEATURE_DIM_V2 = 15
FEATURE_DIM = FEATURE_DIM_V2

FEATURE_NAMES_V1: tuple[str, ...] = (
    "gaze_x", "gaze_y", "pitch_norm", "yaw_norm", "ear", "perclos",
    "head_pose_stability", "obj_book", "obj_phone", "obj_laptop", "obj_none",
    "fidget_score", "hand_proximity_code",
)
FEATURE_NAMES_V2: tuple[str, ...] = FEATURE_NAMES_V1 + ("posture_upright", "posture_lean_norm")

# Unknown posture: same values PoseEstimator.analyze_posture returns without keypoints.
POSTURE_UNKNOWN: tuple[float, float] = (1.0, 0.0)
POSTURE_UPRIGHT_MAX_LEAN_DEG = 15.0   # pose_estimator.analyze_posture is_upright threshold

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
    v1 (13-dim) vector from already-computed metrics (raw units: pitch/yaw in
    degrees with head down = negative, perclos/stability/fidget in 0-100).
    All normalization happens HERE, once.
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


def encode_posture(
    posture: Any = None,
    *,
    is_upright: bool | float | None = None,
    lean_angle: float | None = None,
) -> tuple[float, float]:
    """(posture_upright, posture_lean_norm) from a PostureResult-like object
    (``is_upright``, ``lean_angle`` in degrees) or explicit values.
    Missing -> POSTURE_UNKNOWN."""
    if posture is not None:
        is_upright = getattr(posture, "is_upright", is_upright)
        lean_angle = getattr(posture, "lean_angle", lean_angle)
    if is_upright is None and lean_angle is None:
        return POSTURE_UNKNOWN
    if lean_angle is None or not np.isfinite(lean_angle):
        lean = POSTURE_UNKNOWN[1]
    else:
        lean = float(np.clip(abs(lean_angle), 0.0, 90.0)) / 90.0
    if is_upright is None:
        upright = float(lean * 90.0 < POSTURE_UPRIGHT_MAX_LEAN_DEG)
    else:
        upright = float(bool(is_upright))
    return upright, lean


def encode_feature_vector_v2(
    *,
    posture: Any = None,
    is_upright: bool | float | None = None,
    lean_angle: float | None = None,
    **v1_kwargs: Any,
) -> np.ndarray:
    """v2 (15-dim) vector: encode_feature_vector(**v1_kwargs) + posture."""
    v1 = encode_feature_vector(**v1_kwargs)
    up, lean = encode_posture(posture, is_upright=is_upright, lean_angle=lean_angle)
    return np.concatenate([v1, np.array([up, lean], dtype=np.float32)])


def encode_frame_v2(
    metrics: Any,
    detected_objects: Sequence = (),
    hand_proximity: str = "hands_not_visible",
    posture: Any = None,
) -> np.ndarray:
    """v2 vector from a FaceMetricsResult-like object (the inference path)."""
    return encode_feature_vector_v2(
        gaze_x=metrics.gaze_x, gaze_y=metrics.gaze_y,
        pitch=metrics.pitch, yaw=metrics.yaw, ear=metrics.ear,
        perclos=metrics.perclos, stability=metrics.head_pose_stability,
        fidget=metrics.fidget_score,
        detected_objects=detected_objects, hand_proximity=hand_proximity,
        posture=posture,
    )


def extract_feature_vector_from_landmarks(
    landmarks,
    detected_objects: Sequence = (),
    hand_proximity: str = "hands_not_visible",
) -> np.ndarray:
    """
    v1 (13-dim) vector from raw 68-point landmarks only (dataset loaders; no
    transformation matrix, blendshapes or per-track reference available).
    Uses the LEGACY formulas, converted to the feature convention:
    pitch = -(legacy - LEGACY_PITCH_NEUTRAL) (face_metrics.legacy_pitch_to_feature;
    still saturating, not degrees), gaze_y = eyelid-centroid gaze (down = negative).
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
        gaze_x=gaze_x, gaze_y=gaze_y, pitch=legacy_pitch_to_feature(pitch), yaw=yaw, ear=ear,
        detected_objects=detected_objects, hand_proximity=hand_proximity,
    )


def extract_feature_vector_v2_from_landmarks(
    landmarks,
    detected_objects: Sequence = (),
    hand_proximity: str = "hands_not_visible",
    posture: Any = None,
) -> np.ndarray:
    """v2 (15-dim) counterpart of extract_feature_vector_from_landmarks."""
    v1 = extract_feature_vector_from_landmarks(landmarks, detected_objects, hand_proximity)
    up, lean = encode_posture(posture)
    return np.concatenate([v1, np.array([up, lean], dtype=np.float32)])
