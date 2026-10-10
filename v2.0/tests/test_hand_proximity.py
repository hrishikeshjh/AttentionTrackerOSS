"""
Tests for features/hand_proximity.py (opt-in wrist-based hand/object proximity).

Run: pytest v2.0/tests/test_hand_proximity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.hand_proximity import (
    COCO_L_WRIST,
    COCO_R_WRIST,
    estimate_hand_proximity,
    parse_hand_proximity_source,
    visible_wrists,
)
from inference.behavior_classifier import _PROX_CODE
from inference.contradiction_detector import _encode_hand_proximity

PERSON = (100.0, 100.0, 300.0, 500.0)   # 200 x 400 px


class _Obj:
    def __init__(self, cls, xyxy):
        self.cls = cls
        self.bbox_xyxy = xyxy


def _kps(left=None, right=None, conf=0.9):
    kps = [(0.0, 0.0, 0.0)] * 17
    kps = list(kps)
    if left is not None:
        kps[COCO_L_WRIST] = (left[0], left[1], conf)
    if right is not None:
        kps[COCO_R_WRIST] = (right[0], right[1], conf)
    return kps


def _box_at(cx, cy, half=10):
    return (cx - half, cy - half, cx + half, cy + half)


def test_no_pose_or_low_conf_wrists_is_not_visible():
    assert estimate_hand_proximity(None, PERSON, []) == "hands_not_visible"
    assert estimate_hand_proximity([], PERSON, []) == "hands_not_visible"
    assert estimate_hand_proximity(_kps(left=(200, 400), conf=0.1), PERSON, []) == "hands_not_visible"


def test_visible_wrists_without_objects_is_hands_on_desk():
    # mirrors v1.0 HandObjectProximityDetector: hands visible, no objects -> hands_on_desk
    assert estimate_hand_proximity(_kps(left=(200, 400)), PERSON, []) == "hands_on_desk"


def test_wrist_near_phone_is_holding_phone():
    objs = [_Obj("phone", _box_at(210, 410))]
    assert estimate_hand_proximity(_kps(right=(200, 400)), PERSON, objs) == "holding_phone"


def test_wrist_near_book_and_notebook_is_holding_book():
    for cls in ("book", "notebook"):
        objs = [_Obj(cls, _box_at(200, 420))]
        assert estimate_hand_proximity(_kps(left=(200, 400)), PERSON, objs) == "holding_book"


def test_phone_takes_priority_over_book():
    objs = [_Obj("book", _box_at(200, 400)), _Obj("phone", _box_at(205, 405))]
    assert estimate_hand_proximity(_kps(left=(200, 400)), PERSON, objs) == "holding_phone"


def test_far_object_is_hands_on_desk_and_threshold_is_strict():
    # 0.2 of the person box: 40 px horizontally (width 200), 80 px vertically (height 400)
    wrist = (150.0, 300.0)
    just_inside = [_Obj("phone", _box_at(150.0 + 39.9, 300.0))]
    at_threshold = [_Obj("phone", _box_at(150.0 + 40.0, 300.0))]
    vertical_inside = [_Obj("phone", _box_at(150.0, 300.0 + 79.0))]
    assert estimate_hand_proximity(_kps(left=wrist), PERSON, just_inside) == "holding_phone"
    assert estimate_hand_proximity(_kps(left=wrist), PERSON, at_threshold) == "hands_on_desk"
    assert estimate_hand_proximity(_kps(left=wrist), PERSON, vertical_inside) == "holding_phone"


def test_laptop_and_unknown_classes_are_ignored():
    objs = [_Obj("laptop", _box_at(200, 400)), _Obj("cup", _box_at(200, 400))]
    assert estimate_hand_proximity(_kps(left=(200, 400)), PERSON, objs) == "hands_on_desk"


def test_degenerate_person_box():
    assert estimate_hand_proximity(_kps(left=(200, 400)), (10, 10, 10, 50), []) == "hands_not_visible"


def test_keypoint_objects_are_accepted():
    class KP:
        def __init__(self, x, y, conf):
            self.x, self.y, self.conf = x, y, conf
    kps = [KP(0, 0, 0.0) for _ in range(17)]
    kps[COCO_R_WRIST] = KP(200, 400, 0.8)
    assert visible_wrists(kps) == [(200.0, 400.0)]
    objs = [_Obj("book", _box_at(200, 400))]
    assert estimate_hand_proximity(kps, PERSON, objs) == "holding_book"


def test_never_produces_holding_pen_and_labels_are_known_codes():
    labels = set()
    for objs in ([], [_Obj("phone", _box_at(200, 400))], [_Obj("book", _box_at(200, 400))]):
        for kps in (None, _kps(left=(200, 400))):
            labels.add(estimate_hand_proximity(kps, PERSON, objs))
    assert "holding_pen" not in labels
    for lbl in labels:
        assert lbl in _PROX_CODE
        _encode_hand_proximity(lbl)  # known to the CMCV encoder


def test_parse_source_defaults_to_off():
    assert parse_hand_proximity_source(None) == "off"
    assert parse_hand_proximity_source("") == "off"
    assert parse_hand_proximity_source(" POSE ") == "pose"
    assert parse_hand_proximity_source("off") == "off"
    assert parse_hand_proximity_source("hands") == "off"
