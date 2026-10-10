"""
Cross-Modal Contradiction Vector (CMCV) -- Python port.

Canonical spec + rationale: v1.0/src/lib/contradiction-detector.ts. Ported
here rather than re-derived so both surfaces produce the same taxonomy for
the same underlying signals. This port implements the corrected ("v2")
taxonomy only (the TS 'v1' taxonomy exists solely for v1.0's frozen golden
fixture). Cross-language check: shared/fixtures/cmcv_cross_language.json is
GENERATED from the TS module (shared/fixtures/gen_cmcv_fixture.ts) and
asserted by both v1.0/src/lib/__tests__/contradiction-detector.test.ts and
v2.0/tests/test_contradiction_detector.py.
Note: v1.0's opt-in DACE mode feeds its scoring with DACE-substituted
measurements (ARCHITECTURE.md §11.9); this port corresponds to the v1.0
legacy path (DACE off) only.

Proximity codes differ between modules on purpose (do not unify):
  CMCV "manual" opinion (this file == TS):  book .8, pen .75, desk .65, not visible .6, phone .2
  v1.0 rule layer (PROXIMITY_CODES_V1):     book 1.0, pen .8, desk .5, phone .2, not visible 0
  v2.0 model input (feature_vector.py):     not visible 0, desk .25, pen .5, book .75, phone 1.0
The CMCV value is an opinion on a 0-1 "attentive" scale; the other two are
categorical model-input codes.

Categories (first match wins):
  fatigue_masking          gaze >= HIGH, pose <= LOW, ocular <= LOW
  prop_without_engagement  book visible, hands not holding it / writing / on the
                           desk, and behaviour label != attentive_reading
  context_conflict         gaze >= HIGH and a phone visible
  coherent                 smoothed index < COHERENT_MAX
  unclassified             otherwise (disagreement without a named pattern)

Additive: called from RoomAnalyzer._analyze_person alongside the behavior
classifier, exposed as sibling fields on the per-person result dict, never
folded into attention_score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_ALPHA = 0.35  # faster than the temporal classifier's own smoothing
HIGH = 0.65
LOW = 0.35
COHERENT_MAX = 0.25

# Hand states counted as manual engagement with a visible book.
_ENGAGED_PROXIMITY = frozenset({"holding_book", "holding_pen", "hands_on_desk"})

CATEGORIES = (
    "coherent",
    "fatigue_masking",
    "prop_without_engagement",
    "context_conflict",
    "unclassified",
)

PAIR_WEIGHTS = {
    ("gaze", "pose"): 1.5,
    ("gaze", "ocular"): 1.2,
    ("pose", "ocular"): 1.0,
    ("gaze", "object"): 1.3,
    ("gaze", "manual"): 1.0,
    ("pose", "object"): 0.8,
    ("pose", "manual"): 0.8,
    ("ocular", "object"): 0.8,
    ("ocular", "manual"): 0.8,
    ("object", "manual"): 0.6,
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _encode_hand_proximity(proximity: str | None) -> float:
    return {
        "holding_book": 0.8,
        "holding_pen": 0.75,
        "hands_on_desk": 0.65,
        "hands_not_visible": 0.6,
        "holding_phone": 0.2,
    }.get(proximity, 0.6) if proximity is not None else 0.6


def classify_category(
    opinions: dict,
    contradiction_index: float,
    has_book: bool,
    has_phone: bool,
    hand_proximity: str | None,
    behavior_label: str | None = None,
) -> str:
    """Pure category rule; same semantics as TS classifyContradiction."""
    if opinions["gaze"] >= HIGH and opinions["pose"] <= LOW and opinions["ocular"] <= LOW:
        return "fatigue_masking"
    manual_engagement = hand_proximity in _ENGAGED_PROXIMITY
    reading_engagement = behavior_label == "attentive_reading"
    if has_book and not manual_engagement and not reading_engagement:
        return "prop_without_engagement"
    if opinions["gaze"] >= HIGH and has_phone:
        return "context_conflict"
    if contradiction_index < COHERENT_MAX:
        return "coherent"
    return "unclassified"


def _num(value, default: float) -> float:
    """None -> default; keeps 0.0 (TS `??` semantics, unlike Python `or`)."""
    return default if value is None else value


@dataclass
class ContradictionResult:
    contradiction_index: float
    category: str
    modality_opinions: dict = field(default_factory=dict)


class ContradictionDetector:
    def __init__(self):
        self._ema: float | None = None

    def reset(self) -> None:
        self._ema = None

    def compute(
        self,
        metrics,
        detected_objects,
        hand_proximity: str | None,
        behavior_label: str | None = None,
    ) -> ContradictionResult:
        # TS: deviation 0 unless BOTH gaze components are present.
        if metrics.gaze_x is not None and metrics.gaze_y is not None:
            gaze_deviation = (metrics.gaze_x**2 + metrics.gaze_y**2) ** 0.5
        else:
            gaze_deviation = 0.0

        has_book = any(o.cls in ("book", "notebook") for o in detected_objects)
        has_phone = any(o.cls == "phone" for o in detected_objects)

        opinions = {
            "gaze": 1 - _clamp(gaze_deviation / 0.3, 0, 1),
            "pose": _clamp(_num(metrics.head_pose_stability, 50) / 100, 0, 1),
            "ocular": 1 - _clamp(_num(metrics.perclos, 0) / 20, 0, 1),
            "object": 0.25 if has_phone else (0.8 if has_book else 0.65),
            "manual": _encode_hand_proximity(hand_proximity),
        }

        names = list(opinions.keys())
        weighted_sum = 0.0
        weight_total = 0.0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = (names[i], names[j])
                weight = PAIR_WEIGHTS.get(key, 1.0)
                weighted_sum += weight * abs(opinions[names[i]] - opinions[names[j]])
                weight_total += weight
        raw_ci = weighted_sum / weight_total if weight_total > 0 else 0.0

        self._ema = raw_ci if self._ema is None else _ALPHA * raw_ci + (1 - _ALPHA) * self._ema
        ci = self._ema

        category = classify_category(opinions, ci, has_book, has_phone, hand_proximity, behavior_label)

        return ContradictionResult(contradiction_index=ci, category=category, modality_opinions=opinions)
