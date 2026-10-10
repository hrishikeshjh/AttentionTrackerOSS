"""
Legacy head-pitch characterisation for v2.0 (DACE proposal section 1).

v2.0's calculate_head_pose is a hand copy of v1.0's legacy head-pose.ts.
v1.0's pitch-characterisation test emits the TypeScript constants and
results to docs/proposals/evidence/legacy_pitch_constants.json, and
docs/proposals/evidence/pitch_sim.py projects a 3D head model under known
tilts. This test feeds those projections through the PYTHON formula and
checks that:

  1. it reproduces the TypeScript results (cross-language numerical check);
  2. the known defect is present: tilting down makes pitch RISE and it never
     goes below -15, so behavior_classifier's `pitch < -15` reading/phone
     branches are unreachable by looking down.

(2) documents the defect; it is deliberately not fixed in v2.0 (DACE is
v1.0-only, ARCHITECTURE.md §11.13). If v2.0 adopts a corrected pitch, this
test must be updated together with the rule thresholds and the synthetic
training data (pitch_mu) that assume down = negative.

Run: pytest v2.0/tests/test_pitch_characterisation.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

V2_ROOT = Path(__file__).parent.parent
EVIDENCE = V2_ROOT.parent / "docs" / "proposals" / "evidence"
sys.path.insert(0, str(V2_ROOT))

from features import face_metrics as fm
from features.face_metrics import Point2D, calculate_head_pose


@pytest.fixture(scope="module")
def sim():
    path = EVIDENCE / "pitch_sim.py"
    if not path.exists() or not (EVIDENCE / "legacy_pitch_constants.json").exists():
        pytest.skip("DACE pitch evidence files not present")
    spec = importlib.util.spec_from_file_location("dace_pitch_sim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def consts():
    path = EVIDENCE / "legacy_pitch_constants.json"
    if not path.exists():
        pytest.skip("legacy_pitch_constants.json not present")
    return json.loads(path.read_text())


def _landmarks(img: dict, c: dict) -> list:
    lm = [Point2D(0.0, 0.0) for _ in range(68)]
    idx = c["landmarks"]
    for key, name in (("nose", "noseTip"), ("leftEyeCorner", "leftEyeLeftCorner"),
                      ("rightEyeCorner", "rightEyeRightCorner"), ("chin", "chin")):
        x, y = img[name]
        lm[idx[key]] = Point2D(float(x), float(y))
    return lm


def _python_pitches(sim, c) -> list[float]:
    pts = sim.model_points(c)
    out = []
    for deg in c["tiltsDeg"]:
        img = sim.project(pts, sim.rotation_head_down(deg), None)
        _yaw, pitch, _roll = calculate_head_pose(_landmarks(img, c))
        out.append(pitch)
    return out


def test_landmark_indices_and_gains_match_typescript(consts):
    assert consts["landmarks"]["nose"] == 30
    assert consts["landmarks"]["leftEyeCorner"] == 36
    assert consts["landmarks"]["rightEyeCorner"] == 45
    assert fm.HEAD_POSE_PITCH_GAIN == consts["PITCH_GAIN"]
    assert fm.HEAD_POSE_YAW_GAIN == consts["YAW_GAIN"]


def test_python_formula_reproduces_typescript_results(sim, consts):
    ts = consts["typescriptResults"]
    py = _python_pitches(sim, consts)
    assert [r["trueTiltDeg"] for r in ts] == consts["tiltsDeg"]
    for r, p in zip(ts, py):
        # JSON values are rounded to 4 decimals by the TS test
        assert p == pytest.approx(r["legacyPitch"], abs=1e-3), r["trueTiltDeg"]


def test_known_defect_down_tilt_never_reaches_minus_15(sim, consts):
    py = _python_pitches(sim, consts)
    thr = consts["contextAwarePitchDownThreshold"]
    assert thr == -15
    # pitch rises monotonically as the head tilts down ...
    assert all(b > a for a, b in zip(py, py[1:]))
    # ... has a large positive offset at a level head ...
    level = py[consts["tiltsDeg"].index(0)]
    assert level > 10
    # ... and never crosses the `pitch < -15` threshold used by behavior_classifier.
    assert all(p > thr for p in py)
