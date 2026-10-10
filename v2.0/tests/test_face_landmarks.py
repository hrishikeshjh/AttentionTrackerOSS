"""
Tests for perception/face_landmarks.py: the MediaPipe -> ibug-68 mapping, the
FaceLandmarker options/initialisation (previously never initialised because
of a wrong option name), IMAGE running mode, and crop -> frame coordinates.

The geometry checks use a subset of MediaPipe's canonical face model
(tests/fixtures/mediapipe_canonical_subset.json, Apache-2.0, from
mediapipe/modules/face_geometry/data/canonical_face_model.obj). Light tier:
no mediapipe needed (fakes); the last test runs the real FaceLandmarker when
mediapipe + cv2 + the model file are present.

Run: pytest v2.0/tests/test_face_landmarks.py
"""

import json
import logging
import sys
import types
from dataclasses import dataclass, fields
from pathlib import Path

import pytest

V2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V2_ROOT))

from features.face_metrics import (  # noqa: E402
    LEFT_EYE,
    RIGHT_EYE,
    calculate_ear,
    calculate_head_pose,
    calculate_mar,
)
from perception import face_landmarks as fl  # noqa: E402

CANON = json.loads((V2_ROOT / "tests" / "fixtures" / "mediapipe_canonical_subset.json").read_text())
VERTS = {int(k): v for k, v in CANON["vertices"].items()}


def _canon_68_image():
    """68 points of the canonical face in IMAGE coordinates (y down)."""
    pts = []
    for mp_idx in fl._MP_TO_68:
        x, y, _z = VERTS[mp_idx]
        pts.append(fl.Point2D(x=x, y=-y))
    return pts


# ── Mapping table ─────────────────────────────────────────────────────────────

def test_mapping_shape():
    assert len(fl._MP_TO_68) == 68
    assert len(set(fl._MP_TO_68)) == 68
    assert all(0 <= i < 468 for i in fl._MP_TO_68)


def test_mapping_key_points_used_by_face_metrics():
    m = fl._MP_TO_68
    # previously 30 -> 195 (mid-bridge), 54 -> 267, 57 -> 409, jaw from forehead (10)
    assert m[30] == 4          # nose tip = most anterior canonical vertex
    assert (m[48], m[51], m[54], m[57]) == (61, 0, 291, 17)
    assert m[8] == 152         # chin
    assert 10 not in m         # forehead top is not a jaw point
    assert [m[i] for i in RIGHT_EYE] == [362, 385, 387, 263, 373, 380]
    assert [m[i] for i in LEFT_EYE] == [33, 160, 158, 133, 153, 144]


def test_nose_tip_is_most_anterior():
    tip_z = VERTS[fl._MP_TO_68[30]][2]
    assert all(tip_z >= v[2] for v in VERTS.values())


def test_canonical_geometry_matches_ibug_layout():
    lm = _canon_68_image()
    jaw = lm[0:17]
    # chin is the lowest jaw point; jaw runs image-left -> image-right
    assert max(range(17), key=lambda i: jaw[i].y) == 8
    assert jaw[0].x < jaw[8].x < jaw[16].x
    # mirror symmetry: point i and 16-i
    for i in range(8):
        assert jaw[i].x == pytest.approx(-jaw[16 - i].x, abs=1e-3)
        assert jaw[i].y == pytest.approx(jaw[16 - i].y, abs=1e-3)
    # eyes: 36 outer < 39 inner < 42 inner < 45 outer (image x)
    assert lm[36].x < lm[39].x < lm[42].x < lm[45].x
    # upper lids above lower lids
    for up, low in ((37, 41), (38, 40), (43, 47), (44, 46)):
        assert lm[up].y < lm[low].y
    # nose line top -> tip goes down
    assert lm[27].y < lm[28].y < lm[29].y < lm[30].y
    # mouth corners are the extreme x of the outer lip; 51 top, 57 bottom
    outer = lm[48:60]
    assert lm[48].x == min(p.x for p in outer)
    assert lm[54].x == max(p.x for p in outer)
    # 51 = upper-lip centre (the cupid's-bow dip, so 50/52 sit slightly higher)
    assert lm[51].x == pytest.approx(0.0) and lm[51].y < min(p.y for p in lm[55:60])
    assert lm[57].x == pytest.approx(0.0) and lm[57].y == max(p.y for p in outer)
    # brows above eyes
    assert max(p.y for p in lm[17:27]) < min(p.y for p in lm[36:48])


def test_face_metrics_on_canonical_face_are_plausible():
    lm = _canon_68_image()
    ear = calculate_ear(lm)
    assert ear is not None and 0.2 < ear < 0.4                     # open eyes
    mar = calculate_mar(lm)
    assert mar is not None and 0.2 < mar < 0.6                     # closed mouth, outer-lip MAR
    yaw, _pitch, roll = calculate_head_pose(lm)
    assert yaw is not None and roll is not None
    assert abs(yaw) < 1e-6 and abs(roll) < 1e-6   # frontal, symmetric


def test_old_mapping_made_mar_invalid():
    """Evidence for the fix: the previous table's 48/54/51/57 were 61/267/39/409."""
    pts = {i: fl.Point2D(VERTS[i][0], -VERTS[i][1]) for i in (61, 267, 39, 409, 291, 0, 17)}
    old_h = abs(pts[267].x - pts[61].x)        # corner -> upper-lip point, not corner
    new_h = abs(pts[291].x - pts[61].x)
    assert old_h < 0.7 * new_h


# ── landmarks_to_68: crop -> frame coordinates ───────────────────────────────

@dataclass
class _NL:
    x: float
    y: float


def test_landmarks_to_68_applies_crop_size_and_offset():
    face = [_NL(0.5, 0.25) for _ in range(478)]
    out = fl.landmarks_to_68([face], crop_w=200, crop_h=400, offset=(30, 50))
    assert len(out) == 1 and len(out[0]) == 68
    assert (out[0][0].x, out[0][0].y) == (130.0, 150.0)


def test_landmarks_to_68_skips_short_faces():
    assert fl.landmarks_to_68([[_NL(0, 0)] * 10], 10, 10) == []
    assert fl.landmarks_to_68(None, 10, 10) == []


# ── Initialisation with a fake mediapipe (real option names) ─────────────────

@dataclass
class _FakeOptions:
    """Same field names as mediapipe 0.10.31 FaceLandmarkerOptions."""
    base_options: object = None
    running_mode: object = None
    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    output_face_blendshapes: bool = False
    output_facial_transformation_matrixes: bool = False
    result_callback: object = None


def _install_fake_mediapipe(monkeypatch, create=None):
    created = {}

    class _Landmarker:
        @staticmethod
        def create_from_options(options):
            created["options"] = options
            if create is not None:
                return create(options)
            return types.SimpleNamespace(detect=lambda img: None, close=lambda: None)

    vision = types.SimpleNamespace(
        FaceLandmarkerOptions=_FakeOptions,
        FaceLandmarker=_Landmarker,
        RunningMode=types.SimpleNamespace(IMAGE="IMAGE", VIDEO="VIDEO"),
    )
    python = types.SimpleNamespace(BaseOptions=lambda model_asset_path: ("base", model_asset_path))
    tasks = types.ModuleType("mediapipe.tasks")
    setattr(tasks, "python", python)
    python_mod = types.ModuleType("mediapipe.tasks.python")
    setattr(python_mod, "vision", vision)
    setattr(python_mod, "BaseOptions", python.BaseOptions)
    mp = types.ModuleType("mediapipe")
    setattr(mp, "tasks", tasks)
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks", tasks)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python", python_mod)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python.vision", vision)
    return created


def test_options_use_real_field_names_and_image_mode(monkeypatch, tmp_path):
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"x")
    created = _install_fake_mediapipe(monkeypatch)
    det = fl.FaceLandmarksDetector(model_path=model)
    assert det.available and det.status == fl.STATUS_READY and det.init_error is None
    opts = created["options"]
    assert opts.running_mode == "IMAGE"
    assert opts.min_face_presence_confidence == 0.5
    assert "min_face_presence_score" not in {f.name for f in fields(_FakeOptions)}
    # DACE measurement needs the transformation matrix and blendshapes
    assert opts.output_facial_transformation_matrixes is True
    assert opts.output_face_blendshapes is True


def test_observations_from_result_carries_matrix_blendshapes_and_mesh():
    import numpy as np
    pts = [types.SimpleNamespace(x=i / 478, y=0.5) for i in range(478)]
    cat = types.SimpleNamespace(category_name="eyeLookDownLeft", score=0.4)
    res = types.SimpleNamespace(face_landmarks=[pts], facial_transformation_matrixes=[np.eye(4)],
                                face_blendshapes=[[cat]])
    (obs,) = fl.observations_from_result(res, 200, 100, offset=(10, 20))
    assert len(obs.mesh) == 478 and len(obs.landmarks68) == 68
    assert obs.mesh[1].x == pytest.approx(200 / 478 + 10) and obs.mesh[1].y == pytest.approx(70)
    assert obs.landmarks68[30] is obs.mesh[fl._MP_TO_68[30]]
    assert obs.matrix.shape == (4, 4) and obs.blendshapes == {"eyeLookDownLeft": 0.4}
    bare = types.SimpleNamespace(face_landmarks=[pts])          # outputs not requested
    (o2,) = fl.observations_from_result(bare, 10, 10)
    assert o2.matrix is None and o2.blendshapes is None


def test_init_failure_is_loud_and_exposed(monkeypatch, tmp_path, caplog):
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"x")

    def boom(_opts):
        raise RuntimeError("bad model bundle")

    _install_fake_mediapipe(monkeypatch, create=boom)
    with caplog.at_level(logging.ERROR, logger="cognen.face_landmarks"):
        det = fl.FaceLandmarksDetector(model_path=model)
    assert not det.available
    assert det.status == fl.STATUS_INIT_FAILED
    assert det.init_error is not None and "bad model bundle" in det.init_error
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "bad model bundle" in errors[0].getMessage()
    import numpy as np
    assert det.detect(np.zeros((4, 4, 3), np.uint8)) == []


def test_mediapipe_missing_status(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks", None)
    with caplog.at_level(logging.ERROR, logger="cognen.face_landmarks"):
        det = fl.FaceLandmarksDetector()
    assert det.status == fl.STATUS_MEDIAPIPE_MISSING and not det.available
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_model_missing_status(monkeypatch, tmp_path):
    _install_fake_mediapipe(monkeypatch)
    det = fl.FaceLandmarksDetector(model_path=tmp_path / "nope.task")
    assert det.status == fl.STATUS_MODEL_MISSING and not det.available


# ── Real MediaPipe (heavy tier) ──────────────────────────────────────────────

def test_real_face_landmarker_initialises():
    pytest.importorskip("mediapipe")
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    if not fl._MODEL_FILE.exists():
        pytest.skip(f"{fl._MODEL_FILE} not present")
    det = fl.FaceLandmarksDetector()
    try:
        assert det.status == fl.STATUS_READY, det.init_error
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        assert det.detect(blank) == []
        # non-contiguous crop view is accepted
        assert det.detect(cv2.flip(blank, 1)[10:200, 20:300]) == []
        assert det.detect_full(blank) == []
    finally:
        det.close()
