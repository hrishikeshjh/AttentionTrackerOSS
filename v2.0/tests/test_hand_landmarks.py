"""
HandLandmarker wrapper (perception/hand_landmarks.py): result conversion with
fakes (light tier) and real initialisation with the pinned mediapipe
(heavy tier, skipped when mediapipe / the model file is absent).

Run: pytest v2.0/tests/test_hand_landmarks.py
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from perception import hand_landmarks as hl  # noqa: E402


def test_hands_from_result_offsets_and_handedness():
    lms = [types.SimpleNamespace(x=i / 21, y=0.25) for i in range(21)]
    res = types.SimpleNamespace(
        hand_landmarks=[lms, lms[:5]],
        handedness=[[types.SimpleNamespace(category_name="Left", score=0.9)], []],
    )
    hands = hl.hands_from_result(res, 210, 40, offset=(100, 50))
    assert len(hands) == 1                       # the 5-point "hand" is dropped
    h = hands[0]
    assert h.handedness == "Left" and h.score == pytest.approx(0.9)
    assert h.landmarks[1].x == pytest.approx(110) and h.landmarks[1].y == pytest.approx(60)


def test_missing_mediapipe_is_reported_not_raised(monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks", None)
    det = hl.HandLandmarksDetector()
    assert not det.available and det.status == hl.STATUS_MEDIAPIPE_MISSING
    assert det.detect(object()) == []


def test_real_hand_landmarker_initialises():
    pytest.importorskip("mediapipe")
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    if not hl._MODEL_FILE.exists():
        pytest.skip(f"{hl._MODEL_FILE} not present")
    det = hl.HandLandmarksDetector()
    try:
        assert det.status == hl.STATUS_READY, det.init_error
        assert det.detect(np.zeros((240, 320, 3), dtype=np.uint8)) == []
    finally:
        det.close()
