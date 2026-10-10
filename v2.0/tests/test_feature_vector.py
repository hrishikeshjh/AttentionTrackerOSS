"""
test_feature_vector.py — regression tests for the canonical feature encoder
(features/feature_vector.py) and the train/inference encoding-consistency
bug it fixes.

No torch/cv2/mediapipe needed: features/feature_vector.py only depends on
numpy + features/face_metrics.py (stdlib math).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from features.face_metrics import Point2D
from features.feature_vector import FEATURE_DIM_V1 as FEATURE_DIM
from features.feature_vector import encode_feature_vector, extract_feature_vector_from_landmarks


def _synthetic_landmarks(nose=(50, 55), l_eye=(40, 50), r_eye=(60, 50), chin=(50, 90)):
    lm = [Point2D(0, 0) for _ in range(68)]
    lm[30] = Point2D(*nose)
    lm[36] = Point2D(*l_eye)
    lm[45] = Point2D(*r_eye)
    lm[8] = Point2D(*chin)
    # minimal eye-shape landmarks so calculate_ear doesn't hit degenerate geometry
    for i in range(36, 48):
        if lm[i] == Point2D(0, 0):
            lm[i] = Point2D(45 + i * 0.1, 50 + (1 if i % 2 else -1))
    return lm


class TestEncodeFeatureVector:
    def test_shape_and_dtype(self):
        v = encode_feature_vector()
        assert v.shape == (FEATURE_DIM,)
        assert v.dtype == np.float32

    def test_gaze_pitch_yaw_are_signed_range(self):
        """The literal regression test for the encoding-mismatch bug:
        gaze_x/gaze_y/pitch_norm/yaw_norm (indices 0-3) must stay in
        [-1, 1] -- NOT remapped to [0, 1] like the old, buggy
        feature_extractor.py/daisee_loader.py did."""
        v = encode_feature_vector(gaze_x=-0.8, gaze_y=0.9, pitch=-45.0, yaw=80.0)
        assert v[0] == pytest.approx(-0.8)
        assert v[1] == pytest.approx(0.9)
        assert v[2] == pytest.approx(-0.5)   # -45/90
        assert v[3] == pytest.approx(80 / 90)

    def test_remaining_features_are_unsigned_range(self):
        v = encode_feature_vector(perclos=80.0, stability=10.0, fidget=90.0)
        assert np.all(v[4:] >= 0.0) and np.all(v[4:] <= 1.0)

    def test_object_one_hots(self):
        class _Obj:
            cls = "phone"
        v = encode_feature_vector(detected_objects=[_Obj()])
        assert v[8] == 1.0  # has_phone
        assert v[10] == 0.0  # has_none

    def test_proximity_code(self):
        v_phone = encode_feature_vector(hand_proximity="holding_phone")
        v_book = encode_feature_vector(hand_proximity="holding_book")
        assert v_phone[12] == 1.0
        assert v_book[12] == 0.75

    def test_none_values_fall_back_to_neutral_defaults(self):
        v = encode_feature_vector(gaze_x=None, pitch=None, ear=None)
        assert v[0] == 0.0
        assert v[2] == 0.0
        assert v[4] == pytest.approx(0.3)


class TestExtractFromLandmarks:
    def test_frontal_face_yields_near_zero_pose(self):
        v = extract_feature_vector_from_landmarks(_synthetic_landmarks())
        assert abs(v[2]) < 0.1  # pitch_norm
        assert abs(v[3]) < 0.1  # yaw_norm

    def test_empty_landmarks_falls_back_to_defaults_not_crash(self):
        v = extract_feature_vector_from_landmarks([])
        assert v.shape == (FEATURE_DIM,)


class TestCrossModuleConsistency:
    """Asserts the real bug is fixed: training-time loaders, the synthetic
    generator, and inference-time extraction must all agree on units."""

    def test_feature_extractor_shim_matches_canonical_encoder(self):
        from features.face_metrics import FaceMetricsResult
        from training.feature_extractor import extract_feature_vector

        metrics = FaceMetricsResult(
            ear=0.30, mar=0.02, pitch=-45.0, yaw=30.0, roll=0.0,
            gaze_x=-0.6, gaze_y=0.4,
            blink_detected=False, blink_rate=15.0, perclos=80.0,
            fatigue_index=10.0, fidget_score=5.0, head_pose_stability=90.0,
            face_distance=0.5, face_area=0.05, mouth_openness=0.05,
            ear_variance=0.001, pose_variance=0.5,
        )
        v = extract_feature_vector(metrics)
        # gaze/pitch/yaw must NOT be remapped to [0,1] -- this is the
        # exact bug: old code did `* 0.5 + 0.5` here.
        assert v[0] == pytest.approx(-0.6)
        assert v[1] == pytest.approx(0.4)
        assert v[2] == pytest.approx(-0.5)
        assert v[3] == pytest.approx(30 / 90)
