"""
test_penalty_matrix.py — regression tests for the completed 6x6
ConfusionPenaltyLoss matrix (training/penalty_matrix.py).

No torch needed -- pure numpy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from training.datasets.base_dataset import CLASS_TO_IDX
from training.penalty_matrix import _PENALTY_SPEC, NUM_CLASSES, build_penalty_matrix, penalty_rationale


def test_matrix_shape():
    P = build_penalty_matrix()
    assert P.shape == (NUM_CLASSES, NUM_CLASSES)


def test_diagonal_is_one():
    P = build_penalty_matrix()
    assert np.all(np.diag(P) == 1.0)


def test_every_off_diagonal_cell_is_documented():
    """The bug this fixes: previously only 3 of 30 off-diagonal pairs were
    specified, the rest silently defaulted to 1.0x. Every off-diagonal
    cell must now have an explicit, documented rationale."""
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i == j:
                continue
            assert (i, j) in _PENALTY_SPEC, f"Undocumented pair: ({i}, {j})"


def test_matrix_is_intentionally_asymmetric():
    """Not a copy-paste symmetric placeholder: some pairs must have
    genuinely different weights in each direction."""
    P = build_penalty_matrix()
    assert not np.allclose(P, P.T)


def test_reading_phone_confusion_is_highest_cost():
    """The core motivating case (ARCHITECTURE.md's problem statement):
    false-accusing a reading student of phone use is the single highest
    penalty in the matrix."""
    P = build_penalty_matrix()
    reading, phone = CLASS_TO_IDX["attentive_reading"], CLASS_TO_IDX["using_phone"]
    assert P[reading, phone] == P.max()


def test_penalty_rationale_for_diagonal():
    assert penalty_rationale(0, 0) == "correct prediction, no penalty"


def test_every_off_diagonal_rationale_is_non_empty():
    for (i, j) in _PENALTY_SPEC:
        rationale = penalty_rationale(i, j)
        assert rationale and rationale != "no specific rationale documented (default 1.0x)"
