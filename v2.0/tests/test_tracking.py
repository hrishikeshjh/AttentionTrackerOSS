"""
Tests for SeatMapper.
Run: pytest v2.0/tests/test_tracking.py
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from tracking.byte_tracker import Track
from tracking.seat_mapper import SeatMapper

SEAT_MAP = {
    "seat_1": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
    "seat_2": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
}

# Frame dimensions used in config
W, H = 1280, 720


def test_single_person_assigned():
    mapper = SeatMapper(SEAT_MAP)
    track = Track(track_id=1, bbox_xyxy=(50, 100, 500, 600), confidence=0.9)
    statuses = mapper.assign([track])
    occupied = {s.seat_id for s in statuses if s.occupied}
    assert "seat_1" in occupied


def test_absent_when_no_tracks():
    mapper = SeatMapper(SEAT_MAP)
    statuses = mapper.assign([])
    assert all(not s.occupied for s in statuses)


def test_two_persons_two_seats():
    mapper = SeatMapper(SEAT_MAP)
    t1 = Track(track_id=1, bbox_xyxy=(50, 100, 500, 600), confidence=0.9)
    t2 = Track(track_id=2, bbox_xyxy=(700, 100, 1200, 600), confidence=0.9)
    statuses = mapper.assign([t1, t2])
    occupied = {s.seat_id for s in statuses if s.occupied}
    assert occupied == {"seat_1", "seat_2"}
