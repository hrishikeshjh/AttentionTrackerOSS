"""
DACE measurement port (features/dace_measurement.py): matrix -> Euler (+down),
per-track pitch reference, blendshape gaze + iris cross-check, gaze-corrected
openness / PERCLOS, and the single sign flip into the feature convention
(down = negative). numpy only -> light CI tier.

Run: pytest v2.0/tests/test_dace_measurement.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from features import dace_measurement as dm  # noqa: E402
from features.dace_measurement import (  # noqa: E402
    CorrectedPerclos,
    DaceTrackMeasurer,
    FaceObservation,
    OpennessModel,
    PitchReference,
    estimate_gaze,
    euler_from_matrix,
    gaze_from_blendshapes,
    huber_fit_quadratic,
    is_looking_down,
    matrix_from_euler,
    rotation_from_euler,
    to_feature_gaze_y,
    to_feature_pitch,
)
from perception.face_landmarks import Point2D  # noqa: E402

# ── Head pose from the transformation matrix ─────────────────────────────────


@pytest.mark.parametrize("pyr", [(0, 0, 0), (20, 0, 0), (-15, 10, 5), (35, -25, -10), (5, 40, 20)])
def test_euler_round_trip(pyr):
    pitch, yaw, roll = pyr
    got = euler_from_matrix(matrix_from_euler(pitch, yaw, roll, scale=3.7))   # uniform scale is removed
    assert got is not None
    np.testing.assert_allclose(got, pyr, atol=1e-6)


def test_column_major_flat_equals_matrix():
    m = matrix_from_euler(25, -10, 4, t=(1, 2, -60))
    flat_col_major = m.T.reshape(-1)   # JavaScript .data layout
    from_flat, from_matrix = euler_from_matrix(flat_col_major), euler_from_matrix(m)
    assert from_flat is not None and from_matrix is not None
    np.testing.assert_allclose(from_flat, from_matrix, atol=1e-9)


def test_positive_pitch_is_head_down():
    """+x rotation tilts the face normal (0,0,1) to (0,-sin,cos) in the y-up
    camera frame, i.e. DOWN, and the extracted pitch is positive."""
    r = rotation_from_euler(30, 0, 0)
    normal = r @ np.array([0, 0, 1.0])
    assert normal[1] < 0
    euler = euler_from_matrix(matrix_from_euler(30, 0, 0))
    assert euler is not None and euler[0] == pytest.approx(30)


def test_degenerate_matrix_is_none():
    assert euler_from_matrix(np.zeros((4, 4))) is None
    assert euler_from_matrix(None) is None


def test_sign_flip_into_feature_convention():
    assert to_feature_pitch(20.0) == -20.0
    assert to_feature_gaze_y(0.6) == -0.6
    assert to_feature_pitch(None) is None


# ── Gaze ─────────────────────────────────────────────────────────────────────

def _bs(down=0.0, up=0.0, inn=0.0, out=0.0):
    return {
        "eyeLookDownLeft": down, "eyeLookDownRight": down, "eyeLookUpLeft": up, "eyeLookUpRight": up,
        "eyeLookInLeft": inn, "eyeLookInRight": inn, "eyeLookOutLeft": out, "eyeLookOutRight": out,
    }


def _mesh(iris_dy=0.0):
    """478 points; eyes 40 px wide, iris offset iris_dy * eye width (y down)."""
    pts = [Point2D(100.0, 100.0) for _ in range(478)]
    pts[33], pts[133] = Point2D(60, 100), Point2D(100, 100)       # right eye corners
    pts[263], pts[362] = Point2D(180, 100), Point2D(140, 100)     # left eye corners
    pts[468] = Point2D(80, 100 + iris_dy * 40)
    pts[473] = Point2D(160, 100 + iris_dy * 40)
    # EAR points: open eye, height 12 / width 40 = 0.3
    for idx, cx in ((dm.MESH_RIGHT_EYE_EAR, 80), (dm.MESH_LEFT_EYE_EAR, 160)):
        o, u1, u2, i, l2, l1 = idx
        pts[o], pts[i] = Point2D(cx - 20, 100), Point2D(cx + 20, 100)
        pts[u1], pts[u2] = Point2D(cx - 7, 94), Point2D(cx + 7, 94)
        pts[l1], pts[l2] = Point2D(cx - 7, 106), Point2D(cx + 7, 106)
    return pts


def test_blendshape_gaze_sign():
    gaze = gaze_from_blendshapes(_bs(down=0.7, up=0.05))
    assert gaze is not None
    v, h = gaze
    assert v == pytest.approx(0.65)
    assert gaze_from_blendshapes({"eyeLookDownLeft": 0.5}) is None


def test_iris_cross_check():
    good = estimate_gaze(_bs(down=0.7), _mesh(iris_dy=0.1))
    assert good.gaze_valid and good.gaze_v == pytest.approx(0.7)
    contradicting = estimate_gaze(_bs(down=0.7), _mesh(iris_dy=-0.2))   # iris points UP
    assert contradicting.iris_valid and not contradicting.gaze_valid
    no_iris = estimate_gaze(_bs(down=0.7), None)
    assert not no_iris.gaze_valid


def test_looking_down_clauses():
    assert is_looking_down(15.0, None, False)            # inclusive
    assert not is_looking_down(14.9, 0.9, False)         # gaze ignored when invalid
    assert is_looking_down(0.0, 0.5, True)


# ── Eye openness / PERCLOS ───────────────────────────────────────────────────

def test_huber_fit_recovers_quadratic_with_outliers():
    rng = np.random.default_rng(0)
    g = rng.uniform(-0.2, 0.9, 300)
    e = 0.31 - 0.10 * g - 0.04 * g * g + rng.normal(0, 0.003, 300)
    e[:15] = 0.05   # blinks
    fit = huber_fit_quadratic(g, e)
    assert fit is not None
    b0, b1, b2 = fit
    assert b0 == pytest.approx(0.31, abs=0.01) and b1 == pytest.approx(-0.10, abs=0.03)


def test_looking_down_is_not_closed_but_closure_is():
    """Prior Ê(g)=0.3-0.12g: EAR 0.21 while eyes are down (g=0.8) is open
    (r=0.21/0.204>0.7); the same EAR at g=0 is closed (r=0.7 is not < 0.7 but
    0.20 is)."""
    m = OpennessModel()
    assert not m.is_closed(m.residual(0.16, 0.8))
    assert m.is_closed(m.residual(0.20, 0.0))
    assert m.is_closed(m.residual(0.05, 0.8))


def test_openness_refit_level_and_quadratic():
    m = OpennessModel()
    for _ in range(60):
        m.add_sample(0.0, 0.26)
    assert m.refit() == "level" and m.expected(0.0) == pytest.approx(0.26, abs=1e-6)
    for g in np.linspace(-0.2, 0.8, 120):
        m.add_sample(float(g), 0.26 - 0.08 * float(g))
    assert m.refit() == "quadratic"


def test_corrected_perclos_and_closure():
    p = CorrectedPerclos()
    for i in range(20):
        p.push(i * 200, i >= 10)        # last 10 frames closed (2 s)
    assert p.value()[0] == pytest.approx(0.5)
    assert p.longest_closure_ms(0) == 2000
    p.push(4000, None)                  # no valid openness: ignored
    assert p.value()[1] == 20


# ── Pitch reference ──────────────────────────────────────────────────────────

def test_reference_is_median_of_first_5s_and_excludes_eyes_down():
    r = PitchReference()
    for i in range(30):                   # 6 s at 5 FPS, head at +12 deg
        st = r.update(i * 200, 12.0 if i % 7 else 40.0, 50.0, 0.9 if i % 7 == 0 else 0.0)
    assert not st.calibrating and r.ref_deg == pytest.approx(12.0)
    assert r.update(6200, 37.0, 50.0, 0.0).rel_pitch_deg == pytest.approx(25.0)


def test_reference_reset_after_absence_and_camera_move():
    r = PitchReference()
    t = 0
    for _ in range(30):
        r.update(t, 10.0, 50.0, 0.0)
        t += 200
    for _ in range(dm.ABSENT_FRAMES):
        r.update(t, None, None, None)
        t += 200
    st = r.update(t, 30.0, 50.0, 0.0)
    assert st.reset == "absent_return" and st.calibrating and st.rel_pitch_deg == pytest.approx(0.0)
    for _ in range(30):
        t += 200
        r.update(t, 30.0, 50.0, 0.0)
    assert r.ref_deg == pytest.approx(30.0)
    resets = []
    for _ in range(15):                   # face scale doubles: camera moved
        t += 200
        resets.append(r.update(t, 30.0, 100.0, 0.0).reset)
    assert "camera_move" in resets


# ── Per-track measurer ───────────────────────────────────────────────────────

def _lm68():
    lm = [Point2D(50.0, 50.0) for _ in range(68)]
    pts = {36: (20, 40), 37: (25, 37), 38: (30, 37), 39: (35, 40), 40: (30, 43), 41: (25, 43),
           42: (65, 40), 43: (70, 37), 44: (75, 37), 45: (80, 40), 46: (75, 43), 47: (70, 43), 30: (50, 55)}
    for i, (x, y) in pts.items():
        lm[i] = Point2D(float(x), float(y))
    return lm


def test_measurer_relative_pitch_and_feature_sign():
    m = DaceTrackMeasurer()
    t = 0
    for _ in range(30):   # 6 s screen-facing at an absolute +8 deg (camera above the screen)
        out = m.update(t, FaceObservation(_lm68(), _mesh(), matrix_from_euler(8, 0, 0), _bs()))
        t += 200
    assert out.pitch_source == "matrix" and out.gaze_source == "blendshape"
    assert out.pitch_feature is not None
    assert not out.calibrating and abs(out.pitch_feature) < 1e-6
    down = m.update(t, FaceObservation(_lm68(), _mesh(0.1), matrix_from_euler(33, 0, 0), _bs(down=0.7)))
    assert down.rel_pitch_down_deg == pytest.approx(25.0)
    assert down.pitch_feature == pytest.approx(-25.0)        # feature convention: down = negative
    assert down.gaze_y_feature == pytest.approx(-0.7)
    assert down.looking_down


def test_reading_posture_does_not_raise_perclos():
    """Eyes down at a book narrow the lids (EAR 0.17 at g=0.7); the corrected
    PERCLOS stays 0 while raw EAR < 0.21 would have counted every frame."""
    m = DaceTrackMeasurer()
    mesh = _mesh(0.1)
    # squeeze lids to EAR 0.17
    for idx in (dm.MESH_RIGHT_EYE_EAR, dm.MESH_LEFT_EYE_EAR):
        o, u1, u2, i, l2, l1 = idx
        for k in (u1, u2):
            mesh[k] = Point2D(mesh[k].x, 100 - 3.4)
        for k in (l1, l2):
            mesh[k] = Point2D(mesh[k].x, 100 + 3.4)
    out = None
    for i in range(40):
        out = m.update(i * 200, FaceObservation(_lm68(), mesh, matrix_from_euler(30, 0, 0), _bs(down=0.7)))
    assert out is not None
    assert out.ear_raw == pytest.approx(0.17, abs=1e-3)
    assert out.eye_closed is False and out.perclos == 0.0
    assert out.ear_feature is not None and out.ear_feature > 0.21


def test_legacy_fallback_is_labelled():
    m = DaceTrackMeasurer()
    out = m.update(0, FaceObservation(_lm68()))
    assert out.pitch_source == "legacy68" and out.gaze_source == "legacy68"
    assert out.gaze_valid is False and out.pitch_feature == pytest.approx(0.0)   # its own reference
    none = m.update(200, None)
    assert none.face_present is False and none.pitch_feature is None


def test_measurer_angles_are_degrees():
    m = DaceTrackMeasurer()
    out = m.update(0, FaceObservation(_lm68(), _mesh(), matrix_from_euler(0, 30, -12), _bs()))
    assert out.yaw_deg == pytest.approx(30) and out.roll_deg == pytest.approx(-12)
    assert out.ear_raw is not None and math.isfinite(out.ear_raw)
