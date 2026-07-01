"""
penalty_matrix.py — complete 6x6 ConfusionPenaltyLoss matrix.

Previously train_classifier.py's ConfusionPenaltyLoss only specified 3 of
30 off-diagonal entries (phone<->reading, absent->screen), defaulting
everything else to 1.0 (i.e. plain cross-entropy for those pairs). This
completes the matrix with pedagogically-reasoned weights for every
off-diagonal cell, each with a documented rationale, so the loss function's
behavior is fully specified and reproducible rather than partially
undefined.

Rows = ground truth, columns = predicted. Diagonal = 1.0 (correct
prediction, no penalty). Design principles, highest to lowest cost:

  1. Missing real phone use (GT=using_phone, pred=anything else): the
     model failed at the one thing this system exists to catch. Highest
     cost class in the matrix (2.5x).
  2. False-accusing a reading/attentive student of phone use (GT=reading
     or screen, pred=phone): the single most pedagogically damaging error
     -- suppresses trust in the system, penalizes genuine attention. This
     is the exact motivating case from ARCHITECTURE.md's problem
     statement, and gets the single highest weight (3.0x for reading,
     since reading + phone are the most visually similar "head down"
     states and thus the likeliest real confusion).
  3. Presence/absence errors (GT=absent<->anything, or anything<->absent):
     safety/attendance-relevant -- a missed-absent or falsely-marked-absent
     student both corrupt attendance-adjacent monitoring (2.0x).
  4. Distracted<->absent: still presence-adjacent but a milder error than
     confusing a clearly-present state with absence (1.8x).
  5. Drowsy confusions with anything: a real degree-of-attention error,
     but not a false-accusation or a safety miss (1.5x).
  6. Screen<->reading and screen/reading<->distracted: confusions among
     engagement *degrees* where the student is still clearly present and
     not obviously misconduct-flagged -- the lowest-cost confusions
     (1.2-1.3x).

These weights are a design proposal, not derived from labeled data (no
real dataset exists yet to fit them empirically) -- review/tune with
domain stakeholders once real evaluation data exists. What this module
fixes is the previous state where 27 of 30 off-diagonal cells were silent
1.0x defaults with no stated rationale at all.
"""

from __future__ import annotations

import numpy as np

from training.datasets.base_dataset import BEHAVIOR_CLASSES, CLASS_TO_IDX

_SCREEN     = CLASS_TO_IDX["attentive_screen"]
_READING    = CLASS_TO_IDX["attentive_reading"]
_PHONE      = CLASS_TO_IDX["using_phone"]
_DISTRACTED = CLASS_TO_IDX["distracted"]
_DROWSY     = CLASS_TO_IDX["drowsy"]
_ABSENT     = CLASS_TO_IDX["absent"]

NUM_CLASSES = len(BEHAVIOR_CLASSES)

# (ground_truth, predicted) -> multiplier, with rationale.
# Every one of the 30 off-diagonal (row, col) pairs is listed explicitly
# (verified by test_penalty_matrix.py) -- no silent 1.0x defaults.
_PENALTY_SPEC: dict[tuple[int, int], tuple[float, str]] = {
    # ── False-accusation errors (highest cost) ──────────────────────────
    (_READING,    _PHONE): (3.0, "false-accuses a reading student of phone use -- the core motivating case"),
    (_SCREEN,     _PHONE): (2.0, "false-accuses a screen-attentive student of phone use"),
    (_DISTRACTED, _PHONE): (2.0, "false-accuses a merely-distracted student of phone use"),
    # ── Missed real phone use ────────────────────────────────────────────
    (_PHONE, _READING):    (2.5, "missed real phone use, mistaken for reading"),
    (_PHONE, _SCREEN):     (2.5, "missed real phone use, mistaken for screen attention"),
    (_PHONE, _DISTRACTED): (1.5, "missed real phone use, mistaken for generic distraction"),
    (_PHONE, _DROWSY):     (1.5, "missed real phone use, mistaken for drowsiness"),
    (_PHONE, _ABSENT):     (1.8, "missed real phone use, mistaken for absence"),
    # ── Presence / absence errors ────────────────────────────────────────
    (_ABSENT, _SCREEN):      (2.0, "missed a genuinely absent student (attendance-critical)"),
    (_ABSENT, _READING):     (2.0, "missed a genuinely absent student (attendance-critical)"),
    (_ABSENT, _PHONE):       (2.0, "missed a genuinely absent student (attendance-critical)"),
    (_ABSENT, _DROWSY):      (1.8, "missed a genuinely absent student, mistaken for drowsy"),
    (_ABSENT, _DISTRACTED):  (1.8, "missed a genuinely absent student, mistaken for distracted"),
    (_SCREEN,     _ABSENT):  (2.0, "falsely marks a present, attentive student absent"),
    (_READING,    _ABSENT):  (2.0, "falsely marks a present, attentive student absent"),
    (_DROWSY,     _ABSENT):  (1.8, "falsely marks a present, drowsy student absent"),
    (_DISTRACTED, _ABSENT):  (1.8, "falsely marks a present, distracted student absent"),
    # ── Drowsy confusions (degree-of-attention error, not false-accusation) ──
    (_DROWSY, _SCREEN):      (1.5, "missed drowsiness, mistaken for attentive"),
    (_DROWSY, _READING):     (1.5, "missed drowsiness, mistaken for attentive reading"),
    (_DROWSY, _PHONE):       (1.5, "missed drowsiness, mistaken for phone use"),
    (_DROWSY, _DISTRACTED):  (1.3, "missed drowsiness, mistaken for generic distraction"),
    (_SCREEN,     _DROWSY):  (1.5, "falsely flags an attentive student as drowsy"),
    (_READING,    _DROWSY):  (1.5, "falsely flags a reading student as drowsy"),
    (_DISTRACTED, _DROWSY):  (1.3, "falsely flags a distracted student as drowsy"),
    # ── Engagement-degree confusions (lowest cost -- student still present) ──
    (_SCREEN,     _READING):    (1.2, "confuses two genuinely-attentive states"),
    (_READING,    _SCREEN):     (1.2, "confuses two genuinely-attentive states"),
    (_SCREEN,     _DISTRACTED): (1.3, "mild degree-of-engagement error"),
    (_READING,    _DISTRACTED): (1.3, "mild degree-of-engagement error"),
    (_DISTRACTED, _SCREEN):     (1.3, "mild degree-of-engagement error"),
    (_DISTRACTED, _READING):    (1.3, "mild degree-of-engagement error"),
}


def build_penalty_matrix() -> np.ndarray:
    """Build the complete (NUM_CLASSES, NUM_CLASSES) penalty matrix."""
    P = np.ones((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    for (gt, pred), (weight, _rationale) in _PENALTY_SPEC.items():
        P[gt, pred] = weight
    return P


def penalty_rationale(gt_idx: int, pred_idx: int) -> str:
    """Human-readable rationale for a given (ground_truth, predicted) cell."""
    if gt_idx == pred_idx:
        return "correct prediction, no penalty"
    entry = _PENALTY_SPEC.get((gt_idx, pred_idx))
    return entry[1] if entry else "no specific rationale documented (default 1.0x)"
