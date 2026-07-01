"""
feature_extractor.py — Convert FaceMetricsResult + context into the 13-dim
feature vector used by the 1D-CNN and temporal-classifier.ts.

This is now a thin shim over features/feature_vector.py, the single
canonical implementation (see that module's docstring for full rationale).
Previously this file's gaze_x/gaze_y/pitch_norm/yaw_norm were remapped
from [-1,1] to [0,1] -- inconsistent with
training/datasets/realistic_synthetic_loader.py (the synthetic generator
actually used for training) and inference/behavior_classifier.py::_extract
(the actual inference-time consumer), both of which use [-1,1]. That was a
live train/inference distribution-mismatch bug, now fixed by delegating to
the canonical encoder.

Feature vector (FEATURE_DIM = 13):
    0  gaze_x           [-1, 1]
    1  gaze_y           [-1, 1]
    2  pitch_norm       [-1, 1]   pitch / 90
    3  yaw_norm         [-1, 1]   yaw / 90
    4  ear              [ 0, 1]
    5  perclos          [ 0, 1]   PERCLOS % / 100
    6  stability        [ 0, 1]   head pose stability / 100
    7  has_book         {0, 1}
    8  has_phone        {0, 1}
    9  has_laptop       {0, 1}
    10 has_none         {0, 1}
    11 fidget           [ 0, 1]   fidget score / 100
    12 proximity_code   {0.00, 0.25, 0.50, 0.75, 1.00}
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.face_metrics import FaceMetricsResult
from features.feature_vector import encode_feature_vector, FEATURE_DIM

__all__ = ["extract_feature_vector", "build_input_tensor", "FEATURE_DIM"]


def extract_feature_vector(
    metrics: FaceMetricsResult,
    detected_objects: list | None = None,
    hand_proximity: str = "hands_not_visible",
) -> np.ndarray:
    """
    Build the canonical 13-dim feature vector from a FaceMetricsResult.

    Args:
        metrics:          FaceMetricsResult from face_metrics.py.
        detected_objects: List of detected object dicts with `.cls` attribute.
        hand_proximity:   String code for hand-object proximity.

    Returns:
        np.ndarray of shape (13,), dtype float32. gaze/pitch/yaw in [-1,1],
        remaining continuous features in [0,1] -- see module docstring.
    """
    return encode_feature_vector(
        gaze_x=metrics.gaze_x,
        gaze_y=metrics.gaze_y,
        pitch=metrics.pitch,
        yaw=metrics.yaw,
        ear=metrics.ear,
        perclos=metrics.perclos,
        stability=metrics.head_pose_stability,
        fidget=metrics.fidget_score,
        detected_objects=detected_objects or [],
        hand_proximity=hand_proximity,
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
