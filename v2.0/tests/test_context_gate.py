"""
Context gate end to end on constructed phone / book scenes (fakes only; light tier).

Before: RoomAnalyzer passed a hardcoded "hands_not_visible" (and later the
pose-wrist method was opt-in), so behavior_classifier._context_gate never saw
a holding_* label and the `pitch < -15` rules were unreachable. Now the
default COGNEN_HAND_PROXIMITY is "pose" and "hands" uses HandLandmarker boxes
(features/hand_object.py, DACE Annex A.2) with wrist fallback.

Covers: hand_object relations; default source; phone and book scenes firing
the gate against a CNN that prefers "distracted"; "off" opt-out; "hands"
mode incl. fallback; temporal rules using the corrected down = negative
convention; the looking-down ambiguity branch being reachable.

Run: pytest v2.0/tests/test_context_gate.py
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

V2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V2_ROOT))

from features import hand_object as ho  # noqa: E402
from features.face_metrics import FaceMetricsResult  # noqa: E402
from inference import room_analyzer as ra  # noqa: E402
from inference.behavior_classifier import BEHAVIOR_LABELS, WINDOW_SIZE, BehaviorClassifier  # noqa: E402
from perception.face_landmarks import Point2D  # noqa: E402
from tracking.byte_tracker import Track  # noqa: E402

W, H = ra.INPUT_WIDTH, ra.INPUT_HEIGHT
FRAME = np.zeros((H, W, 3), dtype=np.uint8)
PERSON = (400.0, 100.0, 700.0, 700.0)
PHONE_BOX = (520.0, 450.0, 580.0, 520.0)
BOOK_BOX = (470.0, 480.0, 630.0, 600.0)


# ── hand_object relations ────────────────────────────────────────────────────

def _hand(cx, cy, size=40.0, pinch=False):
    """21 landmarks spread in a box around (cx, cy); thumb/index tips together if pinch."""
    pts = [Point2D(cx + size * (i % 5 - 2) / 4, cy + size * (i // 5 - 2) / 4) for i in range(21)]
    pts[ho.WRIST] = Point2D(cx, cy + size / 2)
    pts[ho.MIDDLE_MCP] = Point2D(cx, cy)
    if pinch:
        pts[ho.THUMB_TIP] = Point2D(cx + 2, cy - size / 2)
        pts[ho.INDEX_TIP] = Point2D(cx + 4, cy - size / 2)
    else:
        pts[ho.THUMB_TIP] = Point2D(cx - size / 2, cy - size / 2)
        pts[ho.INDEX_TIP] = Point2D(cx + size / 2, cy - size / 2)
    return pts


@dataclass
class _Obj:
    cls: str
    bbox_xyxy: tuple
    confidence: float = 0.9


def test_iou_holding_near_pinch():
    assert ho.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert ho.holding(_hand(550, 485), PHONE_BOX)                 # overlaps the phone
    assert not ho.holding(_hand(200, 200), PHONE_BOX)
    assert ho.near(_hand(560, 560, size=10), BOOK_BOX)             # inside the book's 0.5-diag radius
    assert ho.is_pinch(_hand(0, 0, pinch=True)) and not ho.is_pinch(_hand(0, 0))


def test_proximity_from_hands_priority():
    phone, book = _Obj("phone", PHONE_BOX), _Obj("book", BOOK_BOX)
    assert ho.proximity_from_hands([], [phone]) is None
    assert ho.proximity_from_hands([_hand(550, 485)], [phone, book]) == "holding_phone"
    assert ho.proximity_from_hands([_hand(560, 560)], [book]) == "holding_book"
    assert ho.proximity_from_hands([_hand(100, 100, pinch=True)], []) == "holding_pen"
    assert ho.proximity_from_hands([_hand(100, 100)], [phone]) == "hands_on_desk"


# ── Fakes for RoomAnalyzer ───────────────────────────────────────────────────

class _Det:
    def __init__(self, out):
        self.out = out

    def detect(self, frame):
        return list(self.out)


@dataclass
class _Kp:
    x: float
    y: float
    conf: float


@dataclass
class _Pose:
    keypoints: list = field(default_factory=list)
    bbox_xyxy: tuple = PERSON


def _pose_with_wrist(x, y):
    kps = [_Kp(550.0, 200.0, 0.9) for _ in range(17)]
    kps[9] = _Kp(x, y, 0.9)
    kps[10] = _Kp(x + 5, y, 0.9)
    return _Pose(kps)


class _Tracker:
    def update(self, dets, frame):
        return [Track(1, PERSON, 0.9)]


class _Seats:
    def assign(self, tracks):
        return []


class _NoFace:
    status, init_error, available = "unavailable", None, False

    def detect(self, crop, offset=(0, 0)):
        return []


class _Session:
    """A CNN that is confident the person is 'distracted'."""

    def get_inputs(self):
        class _I:
            shape = ["batch", 15, WINDOW_SIZE]
        return [_I()]

    def run(self, _names, feeds):
        logits = np.zeros(6, dtype=np.float32)
        logits[BEHAVIOR_LABELS.index("distracted")] = 2.0
        return [logits[np.newaxis, :]]


class _FakeHands:
    status, init_error, available = "ready", None, True

    def __init__(self, hands):
        self.hands = hands
        self.calls = 0

    def detect(self, crop, offset=(0, 0)):
        self.calls += 1
        from perception.hand_landmarks import HandObservation
        return [HandObservation(landmarks=h) for h in self.hands]


def _analyzer(objects, pose=None, source=None, hands=None):
    return ra.RoomAnalyzer(
        "room_gate", {},
        person_detector=_Det([]), object_detector=_Det(objects),
        pose_estimator=_Det([pose] if pose else []), face_landmarks=_NoFace(),
        tracker=_Tracker(), seat_mapper=_Seats(), behavior_session=_Session(),
        analyze_posture=lambda p: None, hand_landmarks=hands, hand_proximity_source=source,
    )


def _run(a, n=WINDOW_SIZE + 2):
    snap = None
    for _ in range(n):
        snap = a._process_frame(FRAME)
    return snap["persons"][0]


def test_default_source_is_pose():
    assert ra.HAND_PROXIMITY_SOURCE == "pose"
    assert _analyzer([])._hand_source == "pose"


def test_phone_scene_fires_gate_with_default_pose_wrists():
    a = _analyzer([_Obj("phone", PHONE_BOX)], pose=_pose_with_wrist(550, 485))
    p = _run(a)
    assert p["behavior"] == "using_phone"
    assert a._person_states[1].behavior_clf._last.source == "temporal"   # ONNX path + gate


def test_book_scene_fires_gate():
    a = _analyzer([_Obj("book", BOOK_BOX)], pose=_pose_with_wrist(550, 540))
    assert _run(a)["behavior"] == "attentive_reading"


def test_off_is_an_explicit_opt_out():
    a = _analyzer([_Obj("phone", PHONE_BOX)], pose=_pose_with_wrist(550, 485), source="off")
    assert _run(a)["behavior"] == "distracted"


def test_hands_mode_uses_hand_boxes_and_falls_back_to_wrists():
    hands = _FakeHands([_hand(550, 485)])
    a = _analyzer([_Obj("phone", PHONE_BOX)], pose=None, source="hands", hands=hands)
    assert _run(a)["behavior"] == "using_phone" and hands.calls > 0
    # no hand found -> pose wrists decide
    a2 = _analyzer([_Obj("book", BOOK_BOX)], pose=_pose_with_wrist(550, 540), source="hands",
                   hands=_FakeHands([]))
    assert _run(a2)["behavior"] == "attentive_reading"


def test_phone_lying_unused_on_desk_does_not_fire():
    """Wrist far from the phone -> hands_on_desk; the CNN's answer stands."""
    a = _analyzer([_Obj("phone", PHONE_BOX)], pose=_pose_with_wrist(420, 680))
    assert _run(a)["behavior"] == "distracted"


# ── Rules use the down = negative convention ─────────────────────────────────

def _m(pitch=0.0, gaze_y=0.0, **kw):
    base = dict(
        ear=0.3, mar=0.3, pitch=pitch, yaw=0.0, roll=0.0, gaze_x=0.0, gaze_y=gaze_y,
        blink_detected=False, blink_rate=15.0, perclos=2.0, fatigue_index=0.0,
        fidget_score=5.0, head_pose_stability=90.0, face_distance=None, face_area=0.05,
        mouth_openness=None, ear_variance=0.0, pose_variance=0.0, face_detected=True,
    )
    base.update(kw)
    return FaceMetricsResult(**base)


def _feed(clf, metrics, objs, prox="hands_not_visible", n=WINDOW_SIZE):
    r = None
    for _ in range(n):
        r = clf.update({"metrics": metrics, "detected_objects": objs, "hand_proximity": prox, "posture": None})
    return r


@pytest.mark.parametrize("pitch,gaze_y", [(-25.0, 0.0), (0.0, -0.7)])
def test_rules_looking_down_at_book_is_reading(pitch, gaze_y):
    """Head 25 deg down (relative), or eyes down, with a book and no grip seen."""
    assert _feed(BehaviorClassifier(), _m(pitch, gaze_y), [_Obj("book", BOOK_BOX)]).label == "attentive_reading"


def test_rules_head_up_is_not_down():
    """+25 (head UP in this convention) must not count as looking down."""
    assert _feed(BehaviorClassifier(), _m(25.0), [_Obj("book", BOOK_BOX)]).label == "distracted"


def test_rules_looking_down_at_phone_is_phone():
    assert _feed(BehaviorClassifier(), _m(-30.0, -0.6), [_Obj("phone", PHONE_BOX)]).label == "using_phone"


def test_single_frame_rule_uses_convention():
    clf = BehaviorClassifier()
    assert _feed(clf, _m(-20.0), [_Obj("phone", PHONE_BOX)], n=1).label == "using_phone"
    assert _feed(BehaviorClassifier(), _m(20.0), [_Obj("phone", PHONE_BOX)], n=1).label == "distracted"


def test_ambiguity_branch_reachable_when_looking_down_without_objects():
    clf = BehaviorClassifier(session=_Session())
    for _ in range(WINDOW_SIZE):
        clf._buffer.append(clf._extract({"metrics": _m(-30.0), "detected_objects": []}))
    probs = np.zeros(6)
    probs[3] = 1.0
    gated = clf._context_gate(probs, {"detected_objects": [], "hand_proximity": "hands_not_visible"})
    assert gated[3] < 1.0 and int(np.argmax(gated)) == 3


# ── CMCV wiring (room_analyzer passes the behavior label) ────────────────────

def test_room_analyzer_passes_behavior_label_to_cmcv():
    from inference.contradiction_detector import CATEGORIES
    a = _analyzer([_Obj("book", BOOK_BOX)], pose=_pose_with_wrist(550, 540))
    seen = []
    a._process_frame(FRAME)
    det = a._person_states[1].contradiction_detector
    orig = det.compute

    def spy(*args, **kw):
        seen.append(args[3] if len(args) > 3 else kw.get("behavior_label"))
        return orig(*args, **kw)

    det.compute = spy
    p = _run(a)
    assert seen and seen[-1] == p["behavior"] == "attentive_reading"
    assert p["contradiction_category"] in CATEGORIES


def test_reading_label_exempts_book_without_grip_from_prop_category():
    from inference.contradiction_detector import ContradictionDetector
    m = _m(-25.0, -0.6)
    book = [_Obj("book", BOOK_BOX)]
    plain = ContradictionDetector().compute(m, book, "hands_not_visible")
    reading = ContradictionDetector().compute(m, book, "hands_not_visible", "attentive_reading")
    assert plain.category == "prop_without_engagement"
    assert reading.category != "prop_without_engagement"
    assert reading.category in ("coherent", "fatigue_masking", "context_conflict", "unclassified")
