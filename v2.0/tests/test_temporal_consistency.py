"""
test_temporal_consistency.py — regression tests for
BehaviorEvaluator.temporal_consistency() (the real Temporal Consistency
Coefficient, TCC).

Bug this fixes: the evaluator previously only stored per-person
PREDICTION sequences (never ground truth), so its "temporal consistency"
metric was a mode-agreement score over raw prediction blocks -- it could
not distinguish bad flicker (prediction changes when the true behavior
didn't) from correctly following a real transition (prediction changes
because ground truth did too), and would silently return 1.0 whenever no
person-tracked sequences existed, which looks like "perfectly
consistent" when it actually means "not measured at all".

No torch/cv2 needed -- pure numpy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.behavior_metrics import BehaviorEvaluator


def test_perfectly_stable_predictions_score_1():
    ev = BehaviorEvaluator()
    # Ground truth never changes; predictions never change either.
    for _ in range(20):
        ev.update(y_pred=0, y_true=0, person_id="p1")
    assert ev.temporal_consistency() == 1.0


def test_flicker_during_stable_ground_truth_is_penalized():
    """The literal regression test: ground truth is constant, but
    predictions flicker between two labels every frame -- this must score
    low, which the old prediction-only metric could conflate with a
    genuine oscillating ground truth."""
    ev = BehaviorEvaluator()
    for i in range(20):
        pred = i % 2  # flickers 0,1,0,1,...
        ev.update(y_pred=pred, y_true=0, person_id="p1")  # GT constant
    tcc = ev.temporal_consistency()
    assert tcc < 0.2, f"Expected heavy penalty for flicker, got {tcc}"


def test_correctly_following_a_real_transition_is_not_penalized():
    """The key distinction TCC must make: when ground truth genuinely
    changes and the prediction correctly tracks it, that transition must
    NOT count against consistency (it's excluded from the denominator
    entirely, since it's conditioned on GT staying the same)."""
    ev = BehaviorEvaluator()
    # First half: GT=0, pred=0 (stable, correct). Second half: GT=1,
    # pred=1 (stable, correct, but a real transition happened once).
    for _ in range(10):
        ev.update(y_pred=0, y_true=0, person_id="p1")
    for _ in range(10):
        ev.update(y_pred=1, y_true=1, person_id="p1")
    # The single GT-changing transition is excluded; every OTHER
    # transition has matching predictions -> TCC should be 1.0, not
    # penalized for the one real transition.
    assert ev.temporal_consistency() == 1.0


def test_no_tracked_sequences_returns_vacuous_1():
    ev = BehaviorEvaluator()
    ev.update(y_pred=0, y_true=0)  # no person_id -> not tracked for TCC
    assert ev.temporal_consistency() == 1.0


def test_multiple_persons_are_averaged():
    ev = BehaviorEvaluator()
    # p1: perfectly consistent
    for _ in range(10):
        ev.update(y_pred=0, y_true=0, person_id="p1")
    # p2: always flickers under stable ground truth
    for i in range(10):
        ev.update(y_pred=i % 2, y_true=0, person_id="p2")
    tcc = ev.temporal_consistency()
    assert 0.0 < tcc < 1.0
