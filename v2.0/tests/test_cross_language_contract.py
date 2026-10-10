"""
Cross-language contract: v1.0 TypeScript 13-dim encoder / loadModel() vs v2.0.

The fixture shared/fixtures/encode13_cross_language.json is GENERATED from
the shipped v1.0 TypeScript (shared/fixtures/gen_encode13_fixture.ts; run
`cd v1.0 && npx tsx ../shared/fixtures/gen_encode13_fixture.ts`). These
tests compare it with the v2.0 Python encoders and the ONNX export/inference
contract. Intentional or known divergences are asserted explicitly, not
hidden:

  D-PROX  v1.0's rule layer and encode13 use PROXIMITY_CODES_V1
          (book 1.0, pen 0.8, desk 0.5, phone 0.2, none 0); v2.0 uses
          (none 0, desk .25, pen .5, book .75, phone 1). v1.0's ONNX path
          re-encodes index 12 with the v2.0 codes, so the MODEL input agrees.
  D-EAR0  inference/behavior_classifier.py::_extract maps EAR == 0.0 to 0.3
          (`m.ear or 0.3`); v1.0 encode13 and features/feature_vector.py keep 0.0.

Not covered here (documented elsewhere): pitch/gaze_y semantics. Both
surfaces use the same 68-point formula on the legacy path
(test_pitch_characterisation.py), but v1.0 DACE mode feeds negated DACE
degrees instead (ARCHITECTURE.md §11.8).

Run: pytest v2.0/tests/test_cross_language_contract.py
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

V2_ROOT = Path(__file__).parent.parent
REPO_ROOT = V2_ROOT.parent
sys.path.insert(0, str(V2_ROOT))

from features.face_metrics import FaceMetricsResult
from features.feature_vector import _PROXIMITY_CODE, FEATURE_DIM, encode_feature_vector
from inference.behavior_classifier import _PROX_CODE, BEHAVIOR_LABELS, WINDOW_SIZE, BehaviorClassifier

FIXTURE = REPO_ROOT / "shared" / "fixtures" / "encode13_cross_language.json"
ATOL = 1e-6


@pytest.fixture(scope="module")
def fx() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"missing {FIXTURE}")
    return json.loads(FIXTURE.read_text())


class _Obj:
    def __init__(self, cls: str):
        self.cls = cls
        self.bbox_xyxy = (0, 0, 10, 10)


def _py_vector(case: dict) -> np.ndarray:
    m = case["metrics"]
    return encode_feature_vector(
        gaze_x=m["gazeX"], gaze_y=m["gazeY"], pitch=m["pitch"], yaw=m["yaw"], ear=m["ear"],
        perclos=m["perclos"], stability=m["headPoseStability"], fidget=m["fidgetScore"],
        detected_objects=[_Obj(c) for c in case["objects"]],
        hand_proximity=case["handProximity"],
    )


def _metrics_from_case(case: dict) -> FaceMetricsResult:
    m = case["metrics"]
    return FaceMetricsResult(
        ear=m["ear"], mar=0.02, pitch=m["pitch"], yaw=m["yaw"], roll=0.0,
        gaze_x=m["gazeX"], gaze_y=m["gazeY"],
        blink_detected=False, blink_rate=15.0, perclos=m["perclos"],
        fatigue_index=0.0, fidget_score=m["fidgetScore"], head_pose_stability=m["headPoseStability"],
        face_distance=None, face_area=0.05, mouth_openness=None,
        ear_variance=0.0, pose_variance=0.0,
    )


def _frame(case: dict) -> dict:
    return {
        "metrics": _metrics_from_case(case),
        "detected_objects": [_Obj(c) for c in case["objects"]],
        "hand_proximity": case["handProximity"],
        "posture": None,
    }


# ── Tables ────────────────────────────────────────────────────────────────────

def test_feature_dim_and_class_order_match(fx):
    assert FEATURE_DIM == len(fx["feature_names"]) == 13
    assert list(BEHAVIOR_LABELS) == fx["model_classes"]


def test_v2_proximity_tables_match_v1_model_codes(fx):
    assert _PROXIMITY_CODE == fx["proximity_codes_v2_model"]
    assert _PROX_CODE == fx["proximity_codes_v2_model"]


def test_v1_rule_codes_differ_from_v2_codes(fx):
    """D-PROX: documented divergence. Only hands_not_visible (0.0) agrees."""
    v1, v2 = fx["proximity_codes_v1_rules"], fx["proximity_codes_v2_model"]
    assert set(v1) == set(v2)
    same = {k for k in v1 if v1[k] == v2[k]}
    assert same == {"hands_not_visible"}


# ── Per-frame vectors ─────────────────────────────────────────────────────────

def test_encode_feature_vector_equals_v1_model_input(fx):
    """The vector a v1.0-loaded ONNX model sees equals v2.0's training encoder."""
    for case in fx["cases"]:
        np.testing.assert_allclose(_py_vector(case), case["v1_model_input"], atol=ATOL, err_msg=case["name"])


def test_encode_feature_vector_vs_v1_encode13(fx):
    """Indices 0-11 agree with v1.0 encode13; index 12 differs only by D-PROX."""
    v1_codes, v2_codes = fx["proximity_codes_v1_rules"], fx["proximity_codes_v2_model"]
    for case in fx["cases"]:
        py, ts = _py_vector(case), np.asarray(case["v1_encode13"])
        np.testing.assert_allclose(py[:12], ts[:12], atol=ATOL, err_msg=case["name"])
        p = case["handProximity"]
        assert ts[12] == pytest.approx(v1_codes[p])
        assert py[12] == pytest.approx(v2_codes[p])


def test_inference_extract_matches_except_ear_zero(fx):
    """BehaviorClassifier._extract (inference) vs v1 model input; D-EAR0 pinned."""
    clf = BehaviorClassifier()
    for case in fx["cases"]:
        got = np.asarray(clf._extract(_frame(case)), dtype=np.float64)
        want = np.asarray(case["v1_model_input"], dtype=np.float64)
        if case["name"] == "ear_exactly_zero":
            assert want[4] == 0.0
            assert got[4] == pytest.approx(0.3)   # D-EAR0: known v2.0 inference divergence
            got[4] = want[4]
        np.testing.assert_allclose(got, want, atol=ATOL, err_msg=case["name"])


# ── ONNX contract (layout, names, softmax) ────────────────────────────────────

class _FakeSession:
    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)
        self.calls = []

    def run(self, output_names, feeds):
        self.calls.append((output_names, feeds))
        return [self.logits[np.newaxis, :]]


def test_onnx_input_layout_name_and_softmax(fx):
    w = fx["window"]
    assert w["dims_channels_first"] == [1, FEATURE_DIM, WINDOW_SIZE]

    clf = BehaviorClassifier()
    clf._buffer.extend(w["frames"])
    sess = _FakeSession(fx["softmax"]["logits"])
    clf._session = sess

    res = clf._onnx_classify(frame=None)

    (_, feeds), = sess.calls
    assert list(feeds) == [fx["load_model_defaults_documented"]["inputName"]] == ["input"]
    inp = feeds["input"]
    assert inp.shape == (1, FEATURE_DIM, WINDOW_SIZE)
    np.testing.assert_allclose(inp.ravel(), w["packed_channels_first"], atol=1e-5)

    probs = np.asarray(fx["softmax"]["probs"])
    idx = int(np.argmax(probs))
    assert res.label == fx["model_classes"][idx]
    assert res.confidence == pytest.approx(probs[idx], abs=1e-6)
    assert res.source == "temporal"


def test_export_script_declares_the_v1_load_model_contract(fx):
    """training/export_onnx.py (needs torch to run) must keep 'input'/'logits'
    and a (1, FEATURE_DIM, WINDOW_SIZE) dummy; checked statically."""
    tree = ast.parse((V2_ROOT / "training" / "export_onnx.py").read_text())
    kw = {}
    zeros_args = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for k in node.keywords:
                if k.arg in ("input_names", "output_names"):
                    kw[k.arg] = ast.literal_eval(k.value)
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "zeros" and getattr(f.value, "id", "") == "torch":
                zeros_args = [ast.unparse(a) for a in node.args]
    d = fx["load_model_defaults_documented"]
    assert kw["input_names"] == [d["inputName"]]
    assert kw["output_names"] == [d["outputName"]]
    assert d["layout"] == "channels_first"
    assert zeros_args == ["1", "FEATURE_DIM", "WINDOW_SIZE"]


@pytest.mark.parametrize("name", ["behavior_classifier.onnx", "behavior_classifier_v2.onnx"])
def test_committed_onnx_models_match_contract(fx, name):
    ort = pytest.importorskip("onnxruntime")
    path = V2_ROOT / "models" / name
    if not path.exists():
        pytest.skip(f"{path} not present")
    sess = ort.InferenceSession(str(path))
    (inp,), (out,) = sess.get_inputs(), sess.get_outputs()
    assert inp.name == "input" and out.name == "logits"
    assert list(inp.shape[1:]) == [FEATURE_DIM, WINDOW_SIZE]
    assert out.shape[-1] == len(fx["model_classes"])
