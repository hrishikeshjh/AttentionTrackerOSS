"""
behavior_metrics.py — Behavior classification evaluation metrics.

Implements all per-person behavior metrics from ARCHITECTURE.md §7:
    • Accuracy (overall)
    • F1-score macro (target ≥ 0.80)
    • Precision (phone)    (target ≥ 0.90)
    • Recall (attentive_reading) (target ≥ 0.85)
    • Temporal consistency  (target ≥ 0.90)
    • Per-class breakdown

Usage:
    from evaluation.behavior_metrics import BehaviorEvaluator
    ev = BehaviorEvaluator()
    ev.update(y_pred_list, y_true_list)
    report = ev.compute()
"""

from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.datasets.base_dataset import BEHAVIOR_CLASSES, CLASS_TO_IDX

NUM_CLASSES = len(BEHAVIOR_CLASSES)


# ── Targets from ARCHITECTURE.md §7 ──────────────────────────────────────────

TARGETS = {
    "accuracy":                 0.85,
    "f1_macro":                 0.80,
    "precision_using_phone":    0.90,
    "recall_attentive_reading": 0.85,
    "temporal_consistency":     0.90,
}


class BehaviorEvaluator:
    """
    Accumulates predictions and ground-truth labels, then computes all metrics.

    Designed for both batch evaluation and streaming (per-frame updates).
    """

    def __init__(self) -> None:
        self._y_true: list[int] = []
        self._y_pred: list[int] = []
        # For temporal consistency: consecutive prediction sequences per person
        self._sequences: dict[str, list[int]] = defaultdict(list)

    # ── accumulation ─────────────────────────────────────────────────────────

    def update(
        self,
        y_pred: int | Sequence[int],
        y_true: int | Sequence[int],
        person_id: str | None = None,
    ) -> None:
        """
        Add predictions to the evaluator.

        Args:
            y_pred:    Predicted label index / list of indices.
            y_true:    Ground-truth label index / list of indices.
            person_id: Optional ID for temporal consistency tracking.
        """
        if isinstance(y_pred, int):
            y_pred, y_true = [y_pred], [y_true]
        self._y_true.extend(list(y_true))
        self._y_pred.extend(list(y_pred))

        if person_id is not None:
            self._sequences[person_id].extend(list(y_pred))

    def reset(self) -> None:
        self._y_true.clear()
        self._y_pred.clear()
        self._sequences.clear()

    # ── computation ──────────────────────────────────────────────────────────

    def compute(self) -> dict:
        """
        Compute all behavior metrics.

        Returns:
            Dict with keys: accuracy, f1_macro, per_class, temporal_consistency,
            and a targets_met sub-dict indicating which ARCHITECTURE targets are hit.
        """
        if not self._y_true:
            return {}

        y_true = np.array(self._y_true)
        y_pred = np.array(self._y_pred)

        accuracy = float((y_true == y_pred).mean())
        cm       = _confusion_matrix(y_true, y_pred, NUM_CLASSES)

        per_class: dict[str, dict] = {}
        precisions, recalls, f1s   = [], [], []

        for i, cls_name in enumerate(BEHAVIOR_CLASSES):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class[cls_name] = {
                "precision": round(prec, 4),
                "recall":    round(rec,  4),
                "f1":        round(f1,   4),
                "support":   int(cm[i, :].sum()),
            }
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        f1_macro = float(np.mean(f1s))
        tc       = _temporal_consistency(self._sequences)

        result = {
            "accuracy":             round(accuracy,  4),
            "f1_macro":             round(f1_macro,  4),
            "per_class":            per_class,
            "temporal_consistency": round(tc,        4),
            "n_samples":            len(y_true),
            "confusion_matrix":     cm.tolist(),
        }

        # Evaluate against ARCHITECTURE targets
        result["targets_met"] = {
            "accuracy":                 accuracy                                        >= TARGETS["accuracy"],
            "f1_macro":                 f1_macro                                        >= TARGETS["f1_macro"],
            "precision_using_phone":    per_class["using_phone"]["precision"]           >= TARGETS["precision_using_phone"],
            "recall_attentive_reading": per_class["attentive_reading"]["recall"]        >= TARGETS["recall_attentive_reading"],
            "temporal_consistency":     tc                                              >= TARGETS["temporal_consistency"],
        }

        return result

    def print_report(self) -> None:
        """Print a human-readable evaluation report."""
        metrics = self.compute()
        if not metrics:
            print("No data.")
            return

        print("\n" + "=" * 60)
        print("  CognEn Behavior Classification Report")
        print("=" * 60)
        print(f"  Accuracy     : {metrics['accuracy']:.4f}   (target ≥ {TARGETS['accuracy']})")
        print(f"  F1-macro     : {metrics['f1_macro']:.4f}   (target ≥ {TARGETS['f1_macro']})")
        print(f"  Temp. Consist: {metrics['temporal_consistency']:.4f}   (target ≥ {TARGETS['temporal_consistency']})")
        print(f"  Samples      : {metrics['n_samples']}")
        print()
        print(f"  {'Class':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
        print("  " + "-" * 50)
        for cls, m in metrics["per_class"].items():
            print(f"  {cls:<25} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>6d}")
        print()
        print("  Targets met:")
        for k, v in metrics["targets_met"].items():
            status = "✓" if v else "✗"
            print(f"    {status} {k}")
        print("=" * 60 + "\n")


# ── helpers ───────────────────────────────────────────────────────────────────

def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _temporal_consistency(sequences: dict[str, list[int]], window: int = 50) -> float:
    """
    Fraction of consecutive same-label windows inside each *stable* 10-second
    (≈50-frame) block. Returns 1.0 if no sequences available.
    """
    if not sequences:
        return 1.0

    scores: list[float] = []
    for seq in sequences.values():
        if len(seq) < window:
            continue
        for start in range(0, len(seq) - window + 1, window):
            chunk = seq[start: start + window]
            mode  = max(set(chunk), key=chunk.count)
            scores.append(chunk.count(mode) / window)

    return float(np.mean(scores)) if scores else 1.0
