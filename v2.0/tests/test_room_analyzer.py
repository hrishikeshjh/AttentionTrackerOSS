"""
RoomAnalyzer / BehaviorClassifier runtime fixes, tested with fakes (no cv2,
ultralytics, shapely or mediapipe needed; runs in the light CI tier).

Covers: track_id 0 losing its seat, one ONNX session shared by all tracks,
pruning of stale per-person state, bbox clipping + crop->frame landmark
offsets, face_detected / "absent" reachability, stable softmax, EAR 0.0,
_process_frame off the event loop, shutdown aborting the source.

Run: pytest v2.0/tests/test_room_analyzer.py
"""

import asyncio
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

V2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V2_ROOT))

from features.face_metrics import FaceMetricsResult  # noqa: E402
from inference import behavior_classifier as bc  # noqa: E402
from inference import room_analyzer as ra  # noqa: E402
from inference.behavior_classifier import (  # noqa: E402
    ABSENT_MIN_NO_FACE_FRAMES,
    WINDOW_SIZE,
    BehaviorClassifier,
)
from perception.face_landmarks import Point2D  # noqa: E402
from tracking.byte_tracker import Track  # noqa: E402

W, H = ra.INPUT_WIDTH, ra.INPUT_HEIGHT


# ── Fakes ─────────────────────────────────────────────────────────────────────

@dataclass
class _Seat:
    seat_id: str
    occupied: bool
    track_id: int | None = None


class _Empty:
    def detect(self, frame):
        return []


class _FakeTracker:
    def __init__(self, frames):
        self.frames = list(frames)

    def update(self, dets, frame):
        return self.frames.pop(0) if self.frames else []


class _FakeSeatMapper:
    """Puts every track in seat_<track_id>."""

    def assign(self, tracks):
        return [_Seat(f"seat_{t.track_id}", True, t.track_id) for t in tracks]


def _face_68(offset=(0.0, 0.0)):
    """A plausible open-eyed 68-point face in crop coordinates."""
    pts = [Point2D(50.0, 50.0) for _ in range(68)]
    eye = {36: (20, 40), 37: (25, 37), 38: (30, 37), 39: (35, 40), 40: (30, 43), 41: (25, 43),
           42: (65, 40), 43: (70, 37), 44: (75, 37), 45: (80, 40), 46: (75, 43), 47: (70, 43),
           30: (50, 55), 48: (35, 70), 54: (65, 70), 51: (50, 66), 57: (50, 76)}
    for i, (x, y) in eye.items():
        pts[i] = Point2D(float(x), float(y))
    return [Point2D(p.x + offset[0], p.y + offset[1]) for p in pts]


class _FakeFaceDet:
    status = "ready"
    init_error = None

    def __init__(self, available=True, find=True):
        self.available = available
        self.find = find
        self.calls = []

    def detect(self, crop, offset=(0, 0)):
        self.calls.append((crop.shape, offset))
        return [_face_68(offset)] if self.find else []


class _CountingSession:
    def __init__(self):
        self.calls = 0

    def run(self, _names, feeds):
        self.calls += 1
        return [np.array([[5.0, 0, 0, 0, 0, 0]], dtype=np.float32)]


def _analyzer(track_frames, face=None, session=None):
    return ra.RoomAnalyzer(
        "room_t", {},
        person_detector=_Empty(), object_detector=_Empty(), pose_estimator=_Empty(),
        face_landmarks=face or _FakeFaceDet(), tracker=_FakeTracker(track_frames),
        seat_mapper=_FakeSeatMapper(), behavior_session=session or _CountingSession(),
        analyze_posture=lambda pose: None,
    )


FRAME = np.zeros((H, W, 3), dtype=np.uint8)


# ── track_id 0 ────────────────────────────────────────────────────────────────

def test_track_id_zero_keeps_its_seat():
    a = _analyzer([[Track(0, (100, 100, 300, 500), 0.9), Track(1, (600, 100, 800, 500), 0.9)]])
    snap = a._process_frame(FRAME)
    seats = {p["track_id"]: p["seat_id"] for p in snap["persons"]}
    assert seats == {0: "seat_0", 1: "seat_1"}   # 0 used to map to None


# ── Shared ONNX session + pruning ─────────────────────────────────────────────

def test_one_onnx_session_shared_by_all_tracks():
    sess = _CountingSession()
    a = _analyzer([[Track(i, (100 * i, 100, 100 * i + 90, 400), 0.9) for i in range(1, 4)]], session=sess)
    a._process_frame(FRAME)
    clfs = [st.behavior_clf for st in a._person_states.values()]
    assert len(clfs) == 3
    assert all(c._session is sess for c in clfs)


def test_stale_person_states_are_pruned(monkeypatch):
    t1 = Track(1, (100, 100, 300, 500), 0.9)
    t2 = Track(2, (600, 100, 800, 500), 0.9)
    a = _analyzer([[t1, t2], [t2], [t2]])
    clock = iter([1000.0, 1000.0 + ra.PERSON_STATE_TTL_MS / 2000, 1000.0 + ra.PERSON_STATE_TTL_MS / 1000 + 1])
    monkeypatch.setattr(ra.time, "time", lambda: next(clock))
    a._process_frame(FRAME)
    assert set(a._person_states) == {1, 2}
    a._process_frame(FRAME)
    assert set(a._person_states) == {1, 2}      # 1 still within TTL
    a._process_frame(FRAME)
    assert set(a._person_states) == {2}


def test_load_onnx_session_is_cached():
    ort = pytest.importorskip("onnxruntime")  # noqa: F841
    path = V2_ROOT / "models" / "behavior_classifier.onnx"
    if not path.exists():
        pytest.skip("model not present")
    s1 = bc.load_onnx_session(path)
    s2 = bc.load_onnx_session(str(path))
    assert s1 is not None and s1 is s2
    assert BehaviorClassifier(path)._session is s1
    assert bc.load_onnx_session(V2_ROOT / "models" / "missing.onnx") is None


# ── Clipping, ROI and crop -> frame offsets ───────────────────────────────────

def test_clip_bbox_and_face_roi():
    assert ra.clip_bbox((-20, -5, 100.4, 50.2), 640, 480) == (0, 0, 101, 51)
    assert ra.clip_bbox((700, 10, 800, 50), 640, 480) is None
    assert ra.clip_bbox((10, 10, 10, 50), 640, 480) is None
    # tall box -> upper square; wide box -> whole (clipped) box
    assert ra.face_roi((100, 50, 200, 400), 640, 480) == (100, 50, 200, 150)
    assert ra.face_roi((0, 0, 300, 100), 640, 480) == (0, 0, 300, 100)


def test_negative_bbox_is_clipped_and_offset_passed():
    face = _FakeFaceDet()
    a = _analyzer([[Track(3, (-40, -30, 160, 470), 0.9)]], face=face)
    a._process_frame(FRAME)
    (shape, offset), = face.calls
    assert offset == (0, 0)
    assert shape[:2] == (160, 160)    # upper square of the clipped 160x470 box


def test_landmarks_returned_in_frame_coordinates():
    a = _analyzer([[]])
    lms = a._get_landmarks(FRAME, (300, 200, 400, 300))
    assert lms[36].x == pytest.approx(320.0) and lms[36].y == pytest.approx(240.0)


# ── face_detected / absent ────────────────────────────────────────────────────

def _metrics(**kw):
    base: dict[str, Any] = dict(
        ear=0.3, mar=0.3, pitch=0.0, yaw=0.0, roll=0.0, gaze_x=0.0, gaze_y=0.0,
        blink_detected=False, blink_rate=15.0, perclos=5.0, fatigue_index=10.0,
        fidget_score=5.0, head_pose_stability=90.0, face_distance=None, face_area=0.05,
        mouth_openness=None, ear_variance=0.0, pose_variance=0.0,
    )
    base.update(kw)
    return FaceMetricsResult(**base)


def _frame(m):
    return {"metrics": m, "detected_objects": [], "hand_proximity": "hands_not_visible", "posture": None}


def test_single_frame_no_face_is_absent_but_unknown_is_not():
    assert BehaviorClassifier().update(_frame(_metrics(face_detected=False))).label == "absent"
    assert BehaviorClassifier().update(_frame(_metrics(face_detected=None))).label != "absent"
    assert BehaviorClassifier().update(_frame(_metrics(face_detected=True))).label != "absent"


@pytest.mark.parametrize("with_session", [False, True])
def test_window_absent_after_sustained_no_face(with_session):
    clf = BehaviorClassifier(session=_CountingSession() if with_session else None)
    for _ in range(WINDOW_SIZE - ABSENT_MIN_NO_FACE_FRAMES + 1):
        clf.update(_frame(_metrics(face_detected=True)))
    for _ in range(ABSENT_MIN_NO_FACE_FRAMES - 1):
        clf.update(_frame(_metrics(face_detected=False)))
    assert len(clf._buffer) == WINDOW_SIZE
    r = clf._last
    assert r.label != "absent"                  # one short of the threshold
    r = clf.update(_frame(_metrics(face_detected=False)))
    assert r.label == "absent" and r.attention_score == 0 and r.source == "temporal"


def test_analyzer_marks_face_detected_flags():
    a_found = _analyzer([[Track(1, (100, 100, 300, 500), 0.9)]], face=_FakeFaceDet(find=True))
    a_none = _analyzer([[Track(1, (100, 100, 300, 500), 0.9)]], face=_FakeFaceDet(find=False))
    a_off = _analyzer([[Track(1, (100, 100, 300, 500), 0.9)]], face=_FakeFaceDet(available=False))
    assert a_found._process_frame(FRAME)["persons"][0]["behavior"] != "absent"
    assert a_none._process_frame(FRAME)["persons"][0]["behavior"] == "absent"
    # landmarker unavailable: unknown, not "absent" (historical output)
    assert a_off._process_frame(FRAME)["persons"][0]["behavior"] != "absent"


# ── Classifier numerics ───────────────────────────────────────────────────────

def test_softmax_is_stable_for_large_logits():
    p = bc._softmax(np.array([1000.0, 999.0, -1000.0, 0, 0, 0]))
    assert np.all(np.isfinite(p)) and p.sum() == pytest.approx(1.0)
    assert int(np.argmax(p)) == 0
    clf = BehaviorClassifier()
    clf._buffer.extend([[0.0] * 13] * WINDOW_SIZE)

    class _Big:
        def run(self, *_):
            return [np.array([[900.0, 0, 0, 0, 0, 0]], dtype=np.float32)]

    clf._session = _Big()
    r = clf._onnx_classify(frame=None)
    assert r.label == "attentive_screen" and r.source == "temporal"
    assert r.confidence == pytest.approx(1.0)


def test_ear_zero_is_kept_and_counts_as_closed():
    clf = BehaviorClassifier()
    assert clf._extract(_frame(_metrics(ear=0.0)))[4] == 0.0
    assert clf._extract(_frame(_metrics(ear=None)))[4] == pytest.approx(0.3)
    # EAR 0.0 with high PERCLOS is drowsy (used to be read as 0.3 = open)
    assert clf.update(_frame(_metrics(ear=0.0, perclos=40.0))).label == "drowsy"


# ── Event loop / shutdown ─────────────────────────────────────────────────────

class _FakeSource:
    name = "fake"
    fps = 1000.0

    def __init__(self, n):
        self.n = n
        self.stopped = False
        self.released = False
        self.open_thread = None

    def open(self):
        self.open_thread = threading.get_ident()
        return True

    def read(self):
        if self.n <= 0:
            return False, None
        self.n -= 1
        return True, FRAME

    def release(self):
        self.released = True

    def request_stop(self):
        self.stopped = True


def test_process_frame_runs_off_the_event_loop(monkeypatch):
    fake_cv2 = types.SimpleNamespace(resize=lambda f, size: np.zeros((size[1], size[0], 3), np.uint8))
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    a = _analyzer([[Track(1, (100, 100, 300, 500), 0.9)]] * 3)
    seen = []
    real = a._process_frame

    def _spy(frame):
        seen.append(threading.get_ident())
        return real(frame)
    monkeypatch.setattr(a, "_process_frame", _spy)
    src = _FakeSource(3)

    async def main():
        loop_thread = threading.get_ident()
        await a.run_source(src, process_every_n=1)
        return loop_thread

    loop_thread = asyncio.run(main())
    assert len(seen) == 3 and all(t != loop_thread for t in seen)
    assert src.open_thread != loop_thread
    assert src.released and a.get_snapshot()["persons"]


def test_shutdown_requests_source_stop():
    a = _analyzer([])
    src = _FakeSource(0)
    a._source = src
    a.shutdown()
    assert src.stopped and not a._running
