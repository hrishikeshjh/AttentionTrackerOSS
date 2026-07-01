"""
test_train_val_split.py — regression test for the train/val leakage bug.

train_classifier.py::_train_val_split previously did a naive random
permutation over pooled windows, which let overlapping sliding windows
from the same clip/subject land on both sides of the split. This asserts
the fix: no group_id appears in both the train and val index sets.

No torch/cv2/mediapipe needed -- pure numpy + scikit-learn.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from training.train_classifier import _train_val_split


def test_no_group_leaks_across_split():
    rng = np.random.default_rng(0)
    n_groups = 20
    windows_per_group = 10

    X, y, groups = [], [], []
    for g in range(n_groups):
        for _ in range(windows_per_group):
            X.append(rng.random((15, 13)).astype(np.float32))
            y.append(rng.integers(0, 6))
            groups.append(f"clip_{g}")

    X = np.stack(X)
    y = np.array(y)
    groups = np.array(groups, dtype=object)

    X_tr, y_tr, X_val, y_val = _train_val_split(X, y, groups, val_ratio=0.2, seed=1)

    # Recover which group each returned sample belongs to by re-deriving
    # membership via a lookup table (groups are known ahead of time here).
    group_of: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        group_of.setdefault(g, []).append(i)

    # Re-run split index recovery: since _train_val_split only returns
    # X/y (not indices), verify via set membership on the raw arrays
    # instead -- reconstruct index sets by matching rows.
    def _row_group(row):
        idx = np.where((X == row).all(axis=(1, 2)))[0]
        return groups[idx[0]]

    train_groups = {_row_group(X_tr[i]) for i in range(len(X_tr))}
    val_groups = {_row_group(X_val[i]) for i in range(len(X_val))}

    assert train_groups.isdisjoint(val_groups), (
        f"Groups leaked across split: {train_groups & val_groups}"
    )
    assert len(val_groups) > 0
    assert len(train_groups) > 0


def test_degenerate_single_group_falls_back_gracefully():
    """A single-group input can't be group-split -- must not crash."""
    rng = np.random.default_rng(0)
    X = rng.random((50, 15, 13)).astype(np.float32)
    y = rng.integers(0, 6, size=50)
    groups = np.array(["only_group"] * 50, dtype=object)

    X_tr, y_tr, X_val, y_val = _train_val_split(X, y, groups, val_ratio=0.2, seed=1)
    assert len(X_tr) + len(X_val) == 50
    assert len(X_val) > 0
