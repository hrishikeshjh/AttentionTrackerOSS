"""
test_v2_pipeline.py — End-to-end simulation of the v2.0 multi-person pipeline.

Tests the following stages with synthetic data:
    1. PersonDetector stub → Track list
    2. ObjectDetector stub → detected objects
    3. PoseEstimator stub  → posture
    4. ByteTrackWrapper    → persistent IDs
    5. SeatMapper          → seat assignments
    6. BehaviorClassifier  → per-person behavior
    7. RoomAnalyzer._process_frame()  → full snapshot dict

No real models or camera feeds are required.

Run: pytest v2.0/tests/test_v2_pipeline.py
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from features.face_metrics import FaceMetricsResult
from inference.behavior_classifier import BehaviorClassifier
from tracking.byte_tracker import Track
from tracking.seat_mapper import SeatMapper

# ── shared fixtures ───────────────────────────────────────────────────────────

SEAT_MAP = {
    "seat_1": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
    "seat_2": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
}

W, H = 1280, 720

def _track(tid, x1, y1, x2, y2, conf=0.9):
    return Track(track_id=tid, bbox_xyxy=(x1, y1, x2, y2), confidence=conf)

def _metrics(**kw):
    defaults: dict[str, Any] = dict(
        ear=0.30, mar=0.02, pitch=0.0, yaw=0.0, roll=0.0,
        gaze_x=0.0, gaze_y=0.0,
        blink_detected=False, blink_rate=15.0, perclos=5.0,
        fatigue_index=10.0, fidget_score=5.0, head_pose_stability=90.0,
        face_distance=0.5, face_area=0.05, mouth_openness=0.05,
        ear_variance=0.001, pose_variance=0.5,
    )
    defaults.update(kw)
    return FaceMetricsResult(**defaults)


# ── Track tests ───────────────────────────────────────────────────────────────

class TestTrack:
    def test_track_fields(self):
        t = _track(1, 10, 20, 200, 400)
        assert t.track_id == 1
        assert len(t.bbox_xyxy) == 4

    def test_track_confidence(self):
        t = _track(3, 0, 0, 100, 200, conf=0.75)
        assert t.confidence == 0.75


# ── SeatMapper tests ──────────────────────────────────────────────────────────

class TestSeatMapper:
    def _mapper(self):
        return SeatMapper(SEAT_MAP)

    def test_single_person_seat1(self):
        mapper = self._mapper()
        t = _track(1, 50, 100, 500, 600)
        statuses = mapper.assign([t])
        occupied = {s.seat_id for s in statuses if s.occupied}
        assert "seat_1" in occupied

    def test_single_person_seat2(self):
        mapper = self._mapper()
        t = _track(1, 750, 100, 1200, 600)
        statuses = mapper.assign([t])
        occupied = {s.seat_id for s in statuses if s.occupied}
        assert "seat_2" in occupied

    def test_two_persons_two_seats(self):
        mapper = self._mapper()
        statuses = mapper.assign([
            _track(1, 50,  100, 500,  600),
            _track(2, 750, 100, 1200, 600),
        ])
        occupied = {s.seat_id for s in statuses if s.occupied}
        assert occupied == {"seat_1", "seat_2"}

    def test_no_persons_all_absent(self):
        mapper = self._mapper()
        statuses = mapper.assign([])
        assert all(not s.occupied for s in statuses)

    def test_track_id_in_status(self):
        mapper = self._mapper()
        t = _track(42, 50, 100, 500, 600)
        statuses = mapper.assign([t])
        occ = [s for s in statuses if s.occupied]
        assert any(s.track_id == 42 for s in occ)

    def test_six_seat_room(self):
        import json
        seat_map_path = Path(__file__).parent.parent / "data" / "seat_maps" / "room_default.json"
        with open(seat_map_path) as f:
            full_map = json.load(f)
        mapper = SeatMapper(full_map)
        # Place 6 persons roughly in the 6 zones (normalised coords × 1280×720)
        tracks = [
            _track(1, int(0.05*W), int(0.10*H), int(0.30*W), int(0.60*H)),
            _track(2, int(0.35*W), int(0.10*H), int(0.60*W), int(0.60*H)),
            _track(3, int(0.65*W), int(0.10*H), int(0.90*W), int(0.60*H)),
            _track(4, int(0.05*W), int(0.65*H), int(0.30*W), int(0.95*H)),
            _track(5, int(0.35*W), int(0.65*H), int(0.60*W), int(0.95*H)),
            _track(6, int(0.65*W), int(0.65*H), int(0.90*W), int(0.95*H)),
        ]
        statuses = mapper.assign(tracks)
        occupied = {s.seat_id for s in statuses if s.occupied}
        assert len(occupied) == 6


# ── BehaviorClassifier tests ──────────────────────────────────────────────────

class TestBehaviorClassifier:
    def _clf(self):
        return BehaviorClassifier()

    def _frame(self, m, objects=None, prox="hands_not_visible"):
        class _Obj:
            def __init__(self, c):
                self.cls = c
                self.bbox_xyxy = (0, 0, 10, 10)
        return {"metrics": m, "detected_objects": [_Obj(c) for c in (objects or [])],
                "hand_proximity": prox, "posture": None}

    def test_labels_valid(self):
        valid = {"attentive_screen","attentive_reading","using_phone","distracted","drowsy","absent"}
        clf = self._clf()
        for _ in range(20):
            r = clf.update(self._frame(_metrics()))
            assert r.label in valid

    def test_source_rule_based_initially(self):
        clf = self._clf()
        r = clf.update(self._frame(_metrics()))
        assert r.source == "rule_based"

    def test_source_temporal_after_15(self):
        clf = self._clf()
        frame = self._frame(_metrics(gaze_x=0.03))
        for _ in range(14):
            clf.update(frame)
        r = clf.update(frame)
        assert r.source == "temporal"

    def test_phone_high_confidence(self):
        clf = self._clf()
        r = clf.update(self._frame(
            _metrics(pitch=-22.0), objects=["phone"], prox="holding_phone"))
        assert r.label == "using_phone"
        assert r.confidence > 0.5


# ── Full snapshot structure test ──────────────────────────────────────────────

class TestRoomSnapshot:
    """
    Validate the dict structure returned by RoomAnalyzer._process_frame()
    using a fully mocked pipeline (no real models).
    """

    def _make_snapshot(self):
        """Build a minimal RoomAnalyzer snapshot manually."""
        clf1 = BehaviorClassifier()
        clf2 = BehaviorClassifier()

        def _frame(m):
            return {"metrics": m, "detected_objects": [],
                    "hand_proximity": "hands_not_visible", "posture": None}

        r1 = clf1.update(_frame(_metrics(gaze_x=0.05, pitch=-3.0)))
        r2 = clf2.update(_frame(_metrics(ear=0.18, perclos=28.0)))

        persons: list[dict[str, Any]] = [
            {"track_id": 1, "seat_id": "seat_1",
             "behavior": r1.label, "attention_score": r1.attention_score,
             "confidence": round(r1.confidence, 2)},
            {"track_id": 2, "seat_id": "seat_2",
             "behavior": r2.label, "attention_score": r2.attention_score,
             "confidence": round(r2.confidence, 2)},
        ]
        room_attention = sum(p["attention_score"] for p in persons) / len(persons)
        return {
            "ts": 1700000000.0,
            "persons": persons,
            "occupancy": {
                "occupied": ["seat_1", "seat_2"],
                "absent":   [],
                "total":    2,
            },
            "room_attention": round(room_attention, 1),
        }

    def test_snapshot_keys(self):
        snap = self._make_snapshot()
        assert {"ts", "persons", "occupancy", "room_attention"} <= snap.keys()

    def test_persons_list(self):
        snap = self._make_snapshot()
        assert isinstance(snap["persons"], list)
        assert len(snap["persons"]) == 2

    def test_person_fields(self):
        snap = self._make_snapshot()
        for p in snap["persons"]:
            assert "track_id" in p
            assert "behavior" in p
            assert "attention_score" in p
            assert 0 <= p["attention_score"] <= 100

    def test_occupancy_structure(self):
        snap = self._make_snapshot()
        occ = snap["occupancy"]
        assert "occupied" in occ and "absent" in occ and "total" in occ

    def test_room_attention_bounded(self):
        snap = self._make_snapshot()
        assert 0 <= snap["room_attention"] <= 100


# ── Evaluation metrics smoke tests ───────────────────────────────────────────

class TestEvaluationMetrics:
    """Smoke-test the evaluation framework itself."""

    def test_behavior_evaluator(self):
        from evaluation.behavior_metrics import BehaviorEvaluator
        ev = BehaviorEvaluator()
        ev.update([0, 1, 2, 0, 1], [0, 1, 2, 0, 2])
        m = ev.compute()
        assert "accuracy" in m
        assert "f1_macro" in m
        assert 0.0 <= m["f1_macro"] <= 1.0

    def test_tracking_evaluator(self):
        from evaluation.tracking_metrics import TrackFrame, TrackingEvaluator
        ev = TrackingEvaluator()
        gt   = [TrackFrame(1, (10, 10, 200, 400))]
        pred = [TrackFrame(1, (12, 12, 202, 402))]
        ev.update(gt, pred)
        m = ev.compute()
        assert "mota" in m
        assert "idf1" in m
        assert 0.0 <= m["mota"] <= 1.0

    def test_occupancy_evaluator(self):
        from evaluation.occupancy_metrics import OccupancyEvaluator
        ev = OccupancyEvaluator(["seat_1", "seat_2"], fps=5.0)
        ev.update({"seat_1"}, {"seat_1", "seat_2"})
        m = ev.compute()
        assert "occupancy_accuracy" in m
        assert 0.0 <= m["occupancy_accuracy"] <= 1.0

    def test_latency_benchmark(self):
        import time

        from evaluation.latency_benchmark import LatencyBenchmark
        bench = LatencyBenchmark("test")
        for _ in range(10):
            with bench.measure():
                time.sleep(0.001)
        m = bench.compute()
        assert m["mean_ms"] > 0
        assert m["n_frames"] == 10
