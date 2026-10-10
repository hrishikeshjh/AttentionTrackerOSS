"""
Cross-Modal Contradiction Vector (CMCV) -- Python port.

Canonical spec + rationale: v1.0/src/lib/contradiction-detector.ts. Ported
here rather than re-derived so both surfaces produce the same taxonomy for
the same underlying signals; keep them in sync by hand. There is no
automated cross-language check for this module (no EQUIVALENCE.md exists).
Note: v1.0's opt-in DACE mode feeds its scoring with DACE-substituted
measurements (ARCHITECTURE.md §11.9); this port corresponds to the v1.0
legacy path (DACE off) only.

Additive: called from RoomAnalyzer._analyze_person alongside the behavior
classifier, exposed as sibling fields on the per-person result dict, never
folded into attention_score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_ALPHA = 0.35  # faster than the temporal classifier's own smoothing
HIGH = 0.65
LOW = 0.35

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
    }.get(proximity, 0.6)


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

    def compute(self, metrics, detected_objects, hand_proximity: str | None) -> ContradictionResult:
        gaze_x = metrics.gaze_x or 0
        gaze_y = metrics.gaze_y or 0
        gaze_deviation = (gaze_x**2 + gaze_y**2) ** 0.5

        has_book = any(o.cls in ("book", "notebook") for o in detected_objects)
        has_phone = any(o.cls == "phone" for o in detected_objects)

        opinions = {
            "gaze": 1 - _clamp(gaze_deviation / 0.3, 0, 1),
            "pose": _clamp((metrics.head_pose_stability or 50) / 100, 0, 1),
            "ocular": 1 - _clamp((metrics.perclos or 0) / 20, 0, 1),
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

        category = "coherent"
        if opinions["gaze"] >= HIGH and opinions["pose"] <= LOW and opinions["ocular"] <= LOW:
            category = "fatigue_masking"
        elif has_book and hand_proximity != "holding_book":
            category = "prop_without_engagement"
        elif opinions["gaze"] >= HIGH and has_phone:
            category = "context_conflict"
        elif ci < 0.25:
            category = "coherent"

        return ContradictionResult(contradiction_index=ci, category=category, modality_opinions=opinions)
