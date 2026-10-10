"""
behavior_metrics.py — Behavior classification evaluation metrics.

Implements all per-person behavior metrics from ARCHITECTURE.md §7:
    • Accuracy (overall)
    • F1-score macro (target ≥ 0.80)
    • Precision (phone)    (target ≥ 0.90)
    • Recall (attentive_reading) (target ≥ 0.85)
    • Temporal consistency  (target ≥ 0.90)
    • Per-class breakdown

Metrics that cannot be computed from the data given return None / NaN with
a status, never a placeholder pass:
    • temporal_consistency is NaN ("insufficient_data") when no person
      sequence has a ground-truth-stable transition (it used to return a
      vacuous 1.0 and count as a met target);
    • precision_using_phone / recall_attentive_reading are None when the
      class has no support (no predictions / no ground truth);
    • f1_macro averages the classes PRESENT in the ground truth (an absent
      class's F1 is undefined, not 0; f1_macro_all_classes keeps the old
      6-class mean for reference);
    • targets_met values are True / False / None (None = not evaluable).

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
from training.datasets.base_dataset import BEHAVIOR_CLASSES

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
        # Per-person PREDICTION sequences, in order.
        self._sequences: dict[str, list[int]] = defaultdict(list)
        # Per-person GROUND-TRUTH sequences, in the same order -- needed to
        # compute the real Temporal Consistency Coefficient (TCC), which
        # must condition on whether ground truth actually changed between
        # consecutive frames. Previously this evaluator only ever stored
        # y_pred per person, so temporal "consistency" could not
        # distinguish bad flicker (prediction changes when it shouldn't)
        # from correctly following a real transition (prediction changes
        # because the true label did) -- it was structurally incapable of
        # measuring that distinction.
        self._true_sequences: dict[str, list[int]] = defaultdict(list)
        self._n_stable_transitions = 0

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
            self._true_sequences[person_id].extend(list(y_true))

    def reset(self) -> None:
        self._y_true.clear()
        self._y_pred.clear()
        self._sequences.clear()
        self._true_sequences.clear()

    def temporal_consistency(self) -> float:
        """
        Temporal Consistency Coefficient (TCC):

            TCC = (1 / |S|) * sum_{t in S} 1[pred_t == pred_{t-1}]

        where S is the set of consecutive-frame transitions, RESTRICTED to
        those where the ground truth did NOT change (y_true_t ==
        y_true_{t-1}). This is the key design choice: conditioning on
        ground-truth stability is what distinguishes bad flicker (the
        model's prediction changes even though the true behavior didn't)
        from correctly following a real transition (the true behavior
        changed, so the prediction changing too is CORRECT, not
        inconsistency) -- an unconditioned "did consecutive predictions
        match" metric cannot tell these apart and penalizes a model for
        correctly tracking real, rapid behavior changes.

        Pooled over all tracked persons/sequences. Returns NaN when there is
        no ground-truth-stable transition at all (no person ids were passed,
        or every sequence has < 2 frames): the metric is then NOT MEASURED.
        It used to return a vacuous 1.0, which evaluate.py reported as a met
        target although it never passed a person id.
        """
        total_stable_transitions = 0
        total_matches = 0

        for person_id, preds in self._sequences.items():
            trues = self._true_sequences.get(person_id, [])
            if len(preds) != len(trues) or len(preds) < 2:
                continue
            for t in range(1, len(preds)):
                if trues[t] == trues[t - 1]:  # ground truth did not change
                    total_stable_transitions += 1
                    if preds[t] == preds[t - 1]:
                        total_matches += 1

        self._n_stable_transitions = total_stable_transitions
        if total_stable_transitions == 0:
            return float("nan")
        return total_matches / total_stable_transitions

    def n_stable_transitions(self) -> int:
        """Number of GT-stable consecutive transitions TCC is computed over."""
        self.temporal_consistency()
        return self._n_stable_transitions

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
        f1_present, f1_all = [], []

        for i, cls_name in enumerate(BEHAVIOR_CLASSES):
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            support = int(cm[i, :].sum())
            n_pred = tp + fp

            prec = tp / n_pred if n_pred > 0 else None      # undefined without predictions
            rec  = tp / support if support > 0 else None    # undefined without support
            denom = 2 * tp + fp + fn
            f1 = 2 * tp / denom if denom > 0 else None

            per_class[cls_name] = {
                "precision": None if prec is None else round(prec, 4),
                "recall":    None if rec is None else round(rec, 4),
                "f1":        None if f1 is None else round(f1, 4),
                "support":   support,
                "n_predicted": n_pred,
            }
            f1_all.append(f1 or 0.0)
            if support > 0:
                f1_present.append(f1 or 0.0)

        f1_macro = float(np.mean(f1_present)) if f1_present else float("nan")
        tc       = self.temporal_consistency()
        n_stable = self._n_stable_transitions

        result = {
            "accuracy":             round(accuracy,  4),
            "f1_macro":             round(f1_macro,  4) if np.isfinite(f1_macro) else None,
            "f1_macro_classes":     [c for c in BEHAVIOR_CLASSES if per_class[c]["support"] > 0],
            "f1_macro_all_classes": round(float(np.mean(f1_all)), 4),
            "per_class":            per_class,
            # None (JSON null) when not measured -- never NaN in a report
            "temporal_consistency": round(tc, 4) if np.isfinite(tc) else None,
            "temporal_consistency_status": "ok" if n_stable > 0 else
                "insufficient_data: no ground-truth-stable transitions with a person id",
            "temporal_consistency_transitions": n_stable,
            "n_samples":            len(y_true),
            "n_sequences":          len(self._sequences),
            "confusion_matrix":     cm.tolist(),
        }

        def _target(value, key):
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                return None
            return bool(value >= TARGETS[key])

        # Evaluate against ARCHITECTURE targets (None = not evaluable on this data)
        phone_prec = per_class["using_phone"]["precision"] if per_class["using_phone"]["support"] > 0 else None
        result["targets_met"] = {
            "accuracy":                 _target(accuracy, "accuracy"),
            "f1_macro":                 _target(f1_macro, "f1_macro"),
            "precision_using_phone":    _target(phone_prec, "precision_using_phone"),
            "recall_attentive_reading": _target(per_class["attentive_reading"]["recall"],
                                                "recall_attentive_reading"),
            "temporal_consistency":     _target(tc, "temporal_consistency"),
        }

        return result

    def print_report(self) -> None:
        """Print a human-readable evaluation report."""
        self.print_metrics(self.compute())

    @staticmethod
    def print_metrics(metrics: dict, title: str = "") -> None:
        """Print a compute() result."""
        if not metrics:
            print("No data.")
            return

        print("\n" + "=" * 60)
        print(f"  CognEn Behavior Classification Report {title}".rstrip())
        print("=" * 60)
        print(f"  Accuracy     : {metrics['accuracy']:.4f}   (target ≥ {TARGETS['accuracy']})")
        print(f"  F1-macro     : {metrics['f1_macro']:.4f}   (target ≥ {TARGETS['f1_macro']})")
        tc = metrics["temporal_consistency"]
        tc_s = f"{tc:.4f}" if tc is not None else "n/a"
        print(f"  Temp. Consist: {tc_s}   (target ≥ {TARGETS['temporal_consistency']})"
              f"  [{metrics['temporal_consistency_status']}]")
        print(f"  Samples      : {metrics['n_samples']}")
        print()
        print(f"  {'Class':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
        print("  " + "-" * 50)
        def _f(v):
            return f"{v:>6.3f}" if v is not None else f"{'n/a':>6}"
        for cls, m in metrics["per_class"].items():
            print(f"  {cls:<25} {_f(m['precision'])} {_f(m['recall'])} {_f(m['f1'])} {m['support']:>6d}")
        print()
        print("  Targets met:")
        for k, v in metrics["targets_met"].items():
            status = "✓" if v is True else ("✗" if v is False else "–  (not evaluable)")
            print(f"    {status} {k}")
        print("=" * 60 + "\n")


# ── helpers ───────────────────────────────────────────────────────────────────

def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm
