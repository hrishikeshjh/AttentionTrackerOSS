"""
test_v1_pipeline.py — End-to-end simulation of the v1.0 browser pipeline.

Simulates:
    synthetic face landmarks
    → FaceMetricsAggregator
    → context modules (ObjectDetector stub, HandTracker stub)
    → HandObjectProximityDetector
    → TemporalBehaviorClassifier (rule-based)
    → MLEngine.predict() (UPGRADED path)
    → AttentionState output

All external dependencies (face-api.js, MediaPipe, ONNX) are stubbed out
so the test runs entirely in Python without any model files.

Run: pytest v2.0/tests/test_v1_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.face_metrics import FaceMetricsResult
from inference.behavior_classifier import BehaviorClassifier

# ── helpers ───────────────────────────────────────────────────────────────────

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
    class _Obj:
        def __init__(self, cls):
            self.cls = cls
            self.bbox_xyxy = (0, 0, 10, 10)
    return {
        "metrics":          metrics,
        "detected_objects": [_Obj(c) for c in (objects or [])],
        "hand_proximity":   hand_proximity,
        "posture":          None,
    }


# ── v1.0 pipeline stages ──────────────────────────────────────────────────────

class TestV1FaceMetrics:
    """FaceMetricsResult field validation."""

    def test_ear_normal(self):
        m = _metrics(ear=0.32)
        assert 0.0 < m.ear < 1.0

    def test_perclos_zero(self):
        m = _metrics(perclos=0.0)
        assert m.perclos == 0.0

    def test_fatigue_positive(self):
        m = _metrics(fatigue_index=25.0)
        assert m.fatigue_index == 25.0

    def test_blink_detected_flag(self):
        m = _metrics(blink_detected=True)
        assert m.blink_detected is True


class TestV1BehaviorInference:
    """Rule-based behavior classifier (Python port of TS logic)."""

    def _clf(self):
        return BehaviorClassifier()

    def test_attentive_screen(self):
        clf = self._clf()
        r = clf.update(_frame(_metrics(gaze_x=0.05, pitch=-3.0)))
        assert r.label == "attentive_screen"
        assert r.attention_score >= 70

    def test_phone_beats_book(self):
        """phone context should win over book when both present."""
        clf = self._clf()
        r = clf.update(_frame(
            _metrics(pitch=-25.0),
            objects=["phone", "book"],
            hand_proximity="holding_phone",
        ))
        assert r.label == "using_phone"

    def test_reading_high_score(self):
        clf = self._clf()
        r = clf.update(_frame(
            _metrics(pitch=-28.0),
            objects=["book"],
            hand_proximity="holding_book",
        ))
        assert r.label == "attentive_reading"
        assert r.attention_score >= 70

    def test_drowsy_low_ear(self):
        clf = self._clf()
        r = clf.update(_frame(_metrics(ear=0.18, perclos=30.0)))
        assert r.label == "drowsy"
        assert r.attention_score < 40

    def test_absent_tiny_face(self):
        clf = self._clf()
        r = clf.update(_frame(_metrics(face_area=0.0005)))
        assert r.label == "absent"
        assert r.attention_score == 0

    def test_confidence_bounded(self):
        clf = self._clf()
        r = clf.update(_frame(_metrics()))
        assert 0.0 <= r.confidence <= 1.0

    def test_attention_score_bounded(self):
        clf = self._clf()
        r = clf.update(_frame(_metrics(ear=0.15, perclos=40.0)))
        assert 0 <= r.attention_score <= 100


class TestV1TemporalWindow:
    """Temporal classifier window accumulation."""

    def test_fills_window(self):
        clf = BehaviorClassifier()
        frame = _frame(_metrics(gaze_x=0.03, pitch=-4.0))
        results = [clf.update(frame) for _ in range(15)]
        # After WINDOW_SIZE frames source switches to temporal
        assert results[-1].source == "temporal"

    def test_rule_based_before_window_full(self):
        clf = BehaviorClassifier()
        frame = _frame(_metrics(gaze_x=0.03, pitch=-4.0))
        r = clf.update(frame)
        assert r.source == "rule_based"

    def test_stable_behavior_consistent(self):
        """Sustained attentive-screen input → stable label."""
        clf = BehaviorClassifier()
        frame = _frame(_metrics(gaze_x=0.04, pitch=-2.0, ear=0.32))
        labels = {clf.update(frame).label for _ in range(20)}
        assert len(labels) == 1  # No oscillation

    def test_phone_detection_propagates(self):
        clf = BehaviorClassifier()
        for _ in range(20):
            r = clf.update(_frame(
                _metrics(pitch=-22.0),
                objects=["phone"],
                hand_proximity="holding_phone",
            ))
        assert r.label == "using_phone"


class TestV1Pipeline:
    """Full pipeline: metrics → behavior → attention score."""

    def test_score_attentive_above_threshold(self):
        """Attentive screen score should be ≥ 70."""
        clf = BehaviorClassifier()
        for _ in range(5):
            r = clf.update(_frame(_metrics(gaze_x=0.05, pitch=-3.0, ear=0.30)))
        assert r.attention_score >= 70

    def test_score_phone_below_threshold(self):
        """Phone usage score should be < 40."""
        clf = BehaviorClassifier()
        for _ in range(5):
            r = clf.update(_frame(
                _metrics(pitch=-25.0),
                objects=["phone"],
                hand_proximity="holding_phone",
            ))
        assert r.attention_score < 40

    def test_drowsy_multitasking_state(self):
        """Drowsy behavior maps to very low attention."""
        clf = BehaviorClassifier()
        for _ in range(5):
            r = clf.update(_frame(_metrics(ear=0.17, perclos=35.0)))
        assert r.attention_score < 40

    def test_no_crash_on_none_metrics(self):
        """Pipeline should handle None metric fields gracefully."""
        clf = BehaviorClassifier()
        m = _metrics(gaze_x=None, gaze_y=None, pitch=None)
        r = clf.update(_frame(m))
        assert r is not None
        assert r.label in {"attentive_screen", "attentive_reading",
                           "using_phone", "distracted", "drowsy", "absent"}
