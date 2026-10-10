"""
Evaluation correctness: real IDF1 (global identity assignment) and unclipped
MOTA against hand-computed examples, occupancy absence events, behavior
metrics that refuse to report unmeasured targets, deployed-vs-raw behavior
evaluation, report data_source / claims_allowed, full-pipeline latency.

Run: pytest v2.0/tests/test_evaluation_metrics.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

V2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V2_ROOT))

pytest.importorskip("scipy")

from evaluation.behavior_metrics import BehaviorEvaluator  # noqa: E402
from evaluation.occupancy_metrics import OccupancyEvaluator  # noqa: E402
from evaluation.tracking_metrics import TrackFrame, TrackingEvaluator, _match_iou  # noqa: E402

BOX_A = (0, 0, 10, 10)
BOX_B = (100, 0, 110, 10)


def _tf(i, box):
    return TrackFrame(i, box)


# ── IDF1 / MOTA (hand-computed) ───────────────────────────────────────────────

def test_identity_switch_halves_idf1_and_costs_mota():
    """GT A for 4 frames; prediction id 1 for frames 0-1, id 2 for frames 2-3.
    Frame matching: TP 4, FP 0, FN 0, IDSW 1 -> MOTA 1 - 1/4 = 0.75.
    Identity: m[A,1] = 2, m[A,2] = 2 -> IDTP 2, IDFP 2, IDFN 2 -> IDF1 4/8 = 0.5.
    (The old per-frame 'IDF1' said 1.0.)"""
    ev = TrackingEvaluator(fps=5)
    for f in range(4):
        ev.update([_tf("A", BOX_A)], [_tf(1 if f < 2 else 2, BOX_A)])
    m = ev.compute()
    assert m["mota"] == pytest.approx(0.75)
    assert (m["idtp"], m["idfp"], m["idfn"]) == (2, 2, 2)
    assert m["idf1"] == pytest.approx(0.5)
    assert m["id_switches"] == 1


def test_swapped_identities():
    """Two people, prediction ids swap after frame 1: IDSW 2, MOTA 0.75, IDF1 0.5."""
    ev = TrackingEvaluator(fps=5)
    for f in range(4):
        pa, pb = (1, 2) if f < 2 else (2, 1)
        ev.update([_tf("A", BOX_A), _tf("B", BOX_B)], [_tf(pa, BOX_A), _tf(pb, BOX_B)])
    m = ev.compute()
    assert m["id_switches"] == 2
    assert m["mota"] == pytest.approx(1 - 2 / 8)
    assert m["idf1"] == pytest.approx(0.5)


def test_partial_identity_overlap():
    """GT A 10 frames; pred 1 present 10 frames but on A only in 6 -> IDF1 12/20."""
    ev = TrackingEvaluator(fps=5)
    for f in range(10):
        ev.update([_tf("A", BOX_A)], [_tf(1, BOX_A if f < 6 else BOX_B)])
    m = ev.compute()
    assert (m["idtp"], m["idfp"], m["idfn"]) == (6, 4, 4)
    assert m["idf1"] == pytest.approx(0.6)
    assert m["mota"] == pytest.approx(1 - (4 + 4) / 10)


def test_mota_is_not_clipped():
    ev = TrackingEvaluator(fps=5)
    for _ in range(2):
        ev.update([_tf("A", BOX_A)], [_tf(i, (200 + 20 * i, 0, 210 + 20 * i, 10)) for i in range(3)])
    m = ev.compute()
    assert m["mota"] == pytest.approx(1 - (2 + 6) / 2)          # -3.0
    assert m["targets_met"]["mota"] is False


def test_hungarian_frame_matching_beats_greedy():
    """Greedy takes g1-p1 (IoU .82) and leaves g2 unmatched; the optimal
    one-to-one matching is g1-p2 (.67) + g2-p1 (.67)."""
    g1, g2 = TrackFrame("g1", (0, 0, 10, 10)), TrackFrame("g2", (3, 0, 13, 10))
    p1, p2 = TrackFrame(1, (1, 0, 11, 10)), TrackFrame(2, (-2, 0, 8, 10))
    matches, ug, up = _match_iou([g1, g2], [p1, p2], 0.5)
    assert len(matches) == 2 and not ug and not up


def test_sequences_scope_ids():
    """Track id 1 in clip A and clip B are different identities."""
    ev = TrackingEvaluator(fps=5)
    ev.start_sequence("clipA")
    for _ in range(3):
        ev.update([_tf(1, BOX_A)], [_tf(1, BOX_A)])
    ev.start_sequence("clipB")
    for _ in range(3):
        ev.update([_tf(1, BOX_A)], [_tf(7, BOX_A)])        # new clip, new pred id: not a switch
    m = ev.compute()
    assert m["id_switches"] == 0 and m["idf1"] == 1.0 and m["n_gt_identities"] == 2


def test_no_ground_truth_is_not_evaluable():
    ev = TrackingEvaluator()
    ev.update([], [_tf(1, BOX_A)])
    m = ev.compute()
    assert m["mota"] is None and m["idf1"] is None
    assert set(m["targets_met"].values()) == {None}
    assert m["status"].startswith("insufficient_data")


def test_empty_gt_frames_count_false_positives():
    ev = TrackingEvaluator()
    ev.update([_tf("A", BOX_A)], [_tf(1, BOX_A)])
    ev.update([], [_tf(1, BOX_A)])                          # annotated frame, nobody there
    assert ev.compute()["total_fp"] == 1


# ── Occupancy ─────────────────────────────────────────────────────────────────

def test_empty_seat_does_not_record_zero_latency_every_frame():
    ev = OccupancyEvaluator(["s1"], fps=5)
    for _ in range(50):
        ev.update(set(), set())
    m = ev.compute()
    assert m["n_latency_events"] == 0 and m["mean_absence_latency_s"] is None
    assert m["targets_met"]["absence_detection_latency_s"] is None
    assert m["occupancy_accuracy"] == 1.0


def test_absence_event_latency_recorded_once():
    ev = OccupancyEvaluator(["s1"], fps=5)
    gt = [True] * 5 + [False] * 10
    pred = [True] * 8 + [False] * 7
    for g, p in zip(gt, pred):
        ev.update({"s1"} if p else set(), {"s1"} if g else set())
    m = ev.compute()
    assert m["n_absence_events"] == 1 and m["n_latency_events"] == 1
    assert m["mean_absence_latency_s"] == pytest.approx(3 / 5)
    assert m["targets_met"]["absence_detection_latency_s"] is True


def test_missed_absence_fails_latency_target():
    ev = OccupancyEvaluator(["s1"], fps=5)
    for g in [True, True, False, False, True]:
        ev.update({"s1"}, {"s1"} if g else set())
    m = ev.compute()
    assert m["n_missed_events"] == 1 and m["n_latency_events"] == 0
    assert m["targets_met"]["absence_detection_latency_s"] is False


# ── Behavior metrics ──────────────────────────────────────────────────────────

def test_missing_classes_are_not_evaluated():
    ev = BehaviorEvaluator()
    for t in range(10):
        ev.update(0 if t < 8 else 3, 0, person_id="p")
    for _ in range(5):
        ev.update(3, 3, person_id="q")
    m = ev.compute()
    assert m["f1_macro_classes"] == ["attentive_screen", "distracted"]
    assert m["per_class"]["using_phone"]["precision"] is None
    assert m["targets_met"]["precision_using_phone"] is None
    assert m["targets_met"]["recall_attentive_reading"] is None
    assert m["temporal_consistency"] is not None and m["temporal_consistency_status"] == "ok"


# ── evaluate.py ───────────────────────────────────────────────────────────────

class _FakeSession:
    """Raw model: always predicts 'distracted'."""

    def get_inputs(self):
        class _I:
            shape = [None, 15, 15]
        return [_I()]

    def run(self, _names, feeds):
        x = feeds["input"]
        out = np.zeros((x.shape[0], 6), dtype=np.float32)
        out[:, 3] = 5.0
        return [out]


def _fake_sequences():
    from training.datasets.base_dataset import CLASS_TO_IDX
    from training.datasets.sequence_encoder import LabelledSequence, TrackSequenceEncoder
    seqs = []
    for k in range(2):
        enc = TrackSequenceEncoder()   # rule-based: neutral face -> attentive_screen
        s = LabelledSequence(f"clip:{k}", f"g{k}")
        for i in range(25):
            s.frames.append(enc.step(i * 200, None))
            s.labels.append(CLASS_TO_IDX["absent"])
        seqs.append(s)
    return seqs


def test_evaluate_behavior_reports_deployed_and_raw():
    from evaluation.evaluate import evaluate_behavior
    res = evaluate_behavior(_fake_sequences(), _FakeSession())
    assert res["status"] == "ok" and res["data_source"] == "real" and res["primary"] == "deployed"
    dep, raw = res["deployed"], res["raw_onnx"]
    assert dep["n_samples"] == raw["n_samples"] == 2 * (25 - 14)
    # deployed path's absent override fires on no-face windows; the raw model cannot
    assert dep["accuracy"] == 1.0 and raw["accuracy"] == 0.0
    assert dep["temporal_consistency_status"] == "ok" and dep["n_sequences"] == 2


def test_report_without_real_data_allows_no_claims(tmp_path):
    from evaluation.evaluate import _skip, build_report, write_report
    rep = build_report({"behavior": _skip("x"), "tracking": _skip("y")})
    assert rep["data_source"] == "none" and rep["claims_allowed"] is False
    p = write_report(rep, tmp_path / "r.json")
    assert json.loads(p.read_text())["claims_allowed"] is False
    real = build_report({"tracking": {"data_source": "real", "targets_met": {"mota": False, "idf1": None}}})
    assert real["data_source"] == "real" and real["claims_allowed"] is True
    assert real["summary"]["n_evaluated"] == 1 and real["summary"]["n_not_evaluable"] == 1
    none_eval = build_report({"tracking": {"data_source": "real", "targets_met": {"mota": None}}})
    assert none_eval["claims_allowed"] is False


def test_report_never_contains_nan(tmp_path):
    from evaluation.evaluate import write_report
    with pytest.raises(ValueError):
        write_report({"x": float("nan")}, tmp_path / "n.json")
    with pytest.raises(ValueError):
        write_report({"x": np.float64("nan")}, tmp_path / "m.json")
    p = write_report({"y": np.int64(3), "z": np.bool_(True), "w": np.float32(0.5)}, tmp_path / "m.json")
    assert json.loads(p.read_text()) == {"y": 3, "z": True, "w": 0.5}


def test_latency_skipped_when_models_missing(monkeypatch):
    from evaluation import evaluate as E
    monkeypatch.setattr(E, "missing_pipeline_models", lambda: ["models/yolov8m.pt"])
    args = E._parse_args(["--classroom", "/nonexistent"])
    res = E._eval_latency(args)
    assert res["status"] == "SKIPPED" and "missing" in res["reason"]
    assert E._eval_latency(E._parse_args(["--skip_latency"]))["status"] == "SKIPPED"


def test_main_without_data_writes_no_claims_report(tmp_path, monkeypatch):
    from evaluation import evaluate as E
    out = tmp_path / "rep.json"
    rep = E.main(E._parse_args(["--skip_latency", "--out", str(out)]))
    assert rep["data_source"] == "none" and rep["claims_allowed"] is False
    assert json.loads(out.read_text())["summary"]["n_evaluated"] == 0


# ── Full-pipeline latency (fakes) ─────────────────────────────────────────────

def _fake_analyzer(with_person=True):
    from features.dace_measurement import FaceObservation
    from inference import room_analyzer as ra
    from tests.test_training_data_pipeline import _face_68
    from tracking.byte_tracker import Track

    class _E:
        def detect(self, f):
            return []

    class _T:
        def update(self, dets, frame):
            return [Track(1, (100, 50, 300, 450), 0.9)] if with_person else []

    class _S:
        def assign(self, tracks):
            return []

    class _F:
        available = True
        status = "ready"

        def detect_full(self, crop, offset=(0, 0)):
            return [FaceObservation(_face_68(offset[0], offset[1]))]

        def detect(self, crop, offset=(0, 0)):
            return [o.landmarks68 for o in self.detect_full(crop, offset)]

    return ra.RoomAnalyzer("lat", {}, person_detector=_E(), object_detector=_E(), pose_estimator=_E(),
                           face_landmarks=_F(), tracker=_T(), seat_mapper=_S(), behavior_session=None,
                           analyze_posture=lambda p: None, hand_proximity_source="off")


def test_stage_benchmark_times_every_stage():
    from evaluation.latency_benchmark import benchmark_pipeline_stages
    from inference import room_analyzer as ra
    frames = [np.zeros((ra.INPUT_HEIGHT, ra.INPUT_WIDTH, 3), dtype=np.uint8)] * 12
    res = benchmark_pipeline_stages(_fake_analyzer(), frames, n_warmup=2)
    assert res["status"] == "ok" and res["n_frames"] == 10 and res["frames_with_persons"] == 10
    for st in ("person_detect", "object_detect", "pose_detect", "track", "face_landmarks", "classify"):
        assert st in res["stages"], st
        assert res["stages"][st]["p95_ms"] >= res["stages"][st]["p50_ms"] >= 0
    assert res["total"]["p95_ms"] >= res["stages"]["classify"]["p50_ms"]
    assert res["stages_not_run"] == ["hand_landmarks"]
    assert isinstance(res["target_met"], bool)


def test_stage_benchmark_without_persons_is_incomplete():
    from evaluation.latency_benchmark import benchmark_pipeline_stages
    from inference import room_analyzer as ra
    frames = [np.zeros((ra.INPUT_HEIGHT, ra.INPUT_WIDTH, 3), dtype=np.uint8)] * 8
    res = benchmark_pipeline_stages(_fake_analyzer(with_person=False), frames, n_warmup=2)
    assert res["status"] == "INCOMPLETE" and res["target_met"] is None
    assert "classify" in res["stages_not_run"]
    assert benchmark_pipeline_stages(_fake_analyzer(), frames[:2], n_warmup=2)["status"] == "SKIPPED"


# ── Tracking / occupancy end to end on a fake clip (cv2) ──────────────────────

def test_tracking_occupancy_on_test_split_only(tmp_path):
    pytest.importorskip("cv2")
    from evaluation.evaluate import evaluate_tracking_occupancy
    from server.config import INPUT_HEIGHT, INPUT_WIDTH
    from tests.test_training_data_pipeline import _classroom_root
    from tracking.byte_tracker import Track

    root = _classroom_root(tmp_path / "ds")                 # 90 frames @ 15 fps, split test
    sx, sy = INPUT_WIDTH / 320, INPUT_HEIGHT / 240
    box = (10 * sx, 10 * sy, 150 * sx, 230 * sy)

    class _Det:
        def detect(self, f):
            return []

    class _Trk:
        calls = 0

        def update(self, dets, frame):
            _Trk.calls += 1
            assert frame.shape[:2] == (INPUT_HEIGHT, INPUT_WIDTH)
            return [Track(42, box, 0.9)]

    class _Seat:
        def __init__(self, seat_id, occupied):
            self.seat_id, self.occupied = seat_id, occupied

    class _Mapper:
        def __init__(self, seat_map):
            pass

        def assign(self, tracks):
            return [_Seat("seat_1", bool(tracks)), _Seat("seat_2", False)]

    seat_map = {"seat_1": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]], "seat_2": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]}
    t, o = evaluate_tracking_occupancy(root, seat_map, 5.0, _Det(), _Trk, _Mapper)
    assert _Trk.calls == 30                                 # 6 s at the 5 Hz analysis cadence
    assert t["status"] == "ok" and t["mota"] == 1.0 and t["idf1"] == 1.0 and t["n_frames"] == 30
    assert t["data_source"] == "real" and t["split"] == "test"
    assert o["occupancy_accuracy"] == 1.0 and o["absence_latency_status"].startswith("insufficient_data")
    assert o["targets_met"]["absence_detection_latency_s"] is None

    train_root = _classroom_root(tmp_path / "tr", split="train")
    t2, o2 = evaluate_tracking_occupancy(train_root, seat_map, 5.0, _Det(), _Trk, _Mapper)
    assert t2["status"] == "SKIPPED" and o2["status"] == "SKIPPED"


def test_tcc_is_nan_without_person_ids():
    ev = BehaviorEvaluator()
    ev.update([0, 0, 0], [0, 0, 0])
    assert math.isnan(ev.temporal_consistency())
