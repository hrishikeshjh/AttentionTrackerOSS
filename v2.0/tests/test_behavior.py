"""
Tests for BehaviorClassifier.
Run: pytest v2.0/tests/test_behavior.py
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from features.face_metrics import FaceMetricsResult
from inference.behavior_classifier import BehaviorClassifier


def _metrics(**kwargs) -> FaceMetricsResult:
    defaults = dict(
        ear=0.30, mar=0.02, pitch=0.0, yaw=0.0, roll=0.0,
        gaze_x=0.0, gaze_y=0.0,
        blink_detected=False, blink_rate=15.0, perclos=5.0,
        fatigue_index=10.0, fidget_score=5.0, head_pose_stability=90.0,
        face_distance=0.5, face_area=0.05, mouth_openness=0.05,
        ear_variance=0.001, pose_variance=0.5,
    )
    defaults.update(kwargs)
    return FaceMetricsResult(**defaults)


def _frame(metrics, objects=None, hand_proximity="hands_not_visible"):
    class Obj:
        def __init__(self, cls):
            self.cls = cls
            self.bbox_xyxy = (0, 0, 10, 10)

    return {
        "metrics": metrics,
        "detected_objects": [Obj(c) for c in (objects or [])],
        "hand_proximity": hand_proximity,
        "posture": None,
    }


# ── Single-frame rule-based tests ────────────────────────────────────────────

def test_absent_detection():
    clf = BehaviorClassifier()
    r = clf.update(_frame(_metrics(face_area=0.001)))
    assert r.label == "absent"


def test_drowsy_detection():
    clf = BehaviorClassifier()
    r = clf.update(_frame(_metrics(ear=0.19, perclos=25.0)))
    assert r.label == "drowsy"


def test_phone_detection():
    clf = BehaviorClassifier()
    r = clf.update(_frame(_metrics(pitch=-20.0), objects=["phone"], hand_proximity="holding_phone"))
    assert r.label == "using_phone"


def test_reading_detection():
    clf = BehaviorClassifier()
    r = clf.update(_frame(_metrics(pitch=-25.0), objects=["book"], hand_proximity="holding_book"))
    assert r.label == "attentive_reading"


def test_screen_attention():
    clf = BehaviorClassifier()
    r = clf.update(_frame(_metrics(gaze_x=0.1, gaze_y=0.0, pitch=-5.0)))
    assert r.label == "attentive_screen"


# ── Temporal window stabilisation test ───────────────────────────────────────

def test_temporal_window_fills():
    clf = BehaviorClassifier()
    frame = _frame(_metrics(gaze_x=0.05, pitch=-3.0))
    results = [clf.update(frame) for _ in range(15)]
    # After 15 frames the source should switch to temporal
    assert results[-1].source == "temporal"
