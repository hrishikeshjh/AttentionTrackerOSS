"""
Cross-language CMCV check: v2.0 inference/contradiction_detector.py vs the
v1.0 TypeScript detector (taxonomy 'v2').

The fixture shared/fixtures/cmcv_cross_language.json is GENERATED from the
shipped TS module (`cd v1.0 && npx tsx ../shared/fixtures/gen_cmcv_fixture.ts`);
the TS side asserts the same file in
v1.0/src/lib/__tests__/contradiction-detector.test.ts. Light tier: no ML
dependencies (pure Python).

Run: pytest v2.0/tests/test_contradiction_detector.py
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

V2_ROOT = Path(__file__).parent.parent
REPO_ROOT = V2_ROOT.parent
sys.path.insert(0, str(V2_ROOT))

from inference.contradiction_detector import (
    CATEGORIES,
    COHERENT_MAX,
    ContradictionDetector,
    classify_category,
)

FIXTURE = REPO_ROOT / "shared" / "fixtures" / "cmcv_cross_language.json"


class _Obj:
    def __init__(self, cls: str):
        self.cls = cls
        self.bbox_xyxy = (0, 0, 10, 10)


def _metrics(frame: dict) -> SimpleNamespace:
    return SimpleNamespace(
        gaze_x=frame["gazeX"],
        gaze_y=frame["gazeY"],
        head_pose_stability=frame["headPoseStability"],
        perclos=frame["perclos"],
    )


@pytest.fixture(scope="module")
def fx() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"missing {FIXTURE}")
    return json.loads(FIXTURE.read_text())


def test_python_matches_typescript_fixture(fx):
    for seq in fx["sequences"]:
        det = ContradictionDetector()
        for i, (frame, exp) in enumerate(zip(seq["frames"], seq["expected"], strict=True)):
            r = det.compute(
                _metrics(frame),
                [_Obj(c) for c in frame["objects"]],
                frame["handProximity"],
                frame["behaviorLabel"],
            )
            where = f"{seq['name']}[{i}]"
            assert r.category == exp["category"], where
            assert r.contradiction_index == pytest.approx(exp["contradictionIndex"], abs=1e-12), where
            for k, v in exp["modalityOpinions"].items():
                assert r.modality_opinions[k] == pytest.approx(v, abs=1e-12), (where, k)


def test_fixture_covers_every_category(fx):
    cats = {e["category"] for s in fx["sequences"] for e in s["expected"]}
    assert cats == set(CATEGORIES)


def _ops(gaze=1.0, pose=1.0, ocular=1.0):
    return {"gaze": gaze, "pose": pose, "ocular": ocular, "object": 0.8, "manual": 0.6}


@pytest.mark.parametrize("prox", ["hands_on_desk", "holding_pen", "holding_book"])
def test_book_with_engaged_hands_is_not_prop(prox):
    assert classify_category(_ops(), 0.1, True, False, prox) != "prop_without_engagement"


def test_book_with_reading_label_is_not_prop():
    assert classify_category(_ops(), 0.1, True, False, "hands_not_visible", "attentive_reading") != (
        "prop_without_engagement"
    )


def test_book_without_engagement_is_prop():
    assert classify_category(_ops(), 0.1, True, False, "hands_not_visible") == "prop_without_engagement"
    assert classify_category(_ops(), 0.1, True, False, None) == "prop_without_engagement"


def test_high_index_without_rule_is_unclassified_not_coherent():
    assert classify_category(_ops(gaze=0.0), COHERENT_MAX, False, False, "hands_on_desk") == "unclassified"
    assert classify_category(_ops(gaze=0.0), COHERENT_MAX - 1e-9, False, False, "hands_on_desk") == "coherent"


def test_zero_stability_is_not_replaced_by_default():
    det = ContradictionDetector()
    r = det.compute(
        SimpleNamespace(gaze_x=None, gaze_y=None, head_pose_stability=0.0, perclos=0.0),
        [],
        "hands_not_visible",
    )
    assert r.modality_opinions["pose"] == 0.0
