"""
test_model_tuning_params.py — Unit tests for model tuning parameters and calibration.
Tests:
  - train_classifier hyperparameter argument parsing and defaults
  - train_yolo dry-run and argument validation
  - tune_hyperparameters Optuna objective creation and dry-run
  - calibrate_probabilities temperature scaling and ECE reduction
  - ByteTrackWrapper configurable tracking thresholds
  - BehaviorClassifier temperature scaling mechanics
"""

import sys
from pathlib import Path

import numpy as np

V2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V2_ROOT))

from inference.behavior_classifier import BehaviorClassifier, BehaviorResult
from tracking.byte_tracker import ByteTrackWrapper
from training.calibrate_probabilities import (
    calibrate_and_evaluate,
    compute_ece,
    compute_nll,
    fit_temperature,
    softmax,
)
from training.train_classifier import _parse_args
from training.train_yolo import build_arg_parser as build_yolo_parser
from training.train_yolo import train_yolo
from training.tune_hyperparameters import build_arg_parser as build_tune_parser
from training.tune_hyperparameters import run_tuning


def test_train_classifier_tuning_arguments():
    """Verify that train_classifier exposes the hyperparameter arguments from parameters.md."""
    # Test default values
    parser_args = _parse_args()
    assert parser_args.weight_decay == 1e-4
    assert parser_args.label_smoothing == 0.05
    assert parser_args.eta_min == 1e-5
    assert parser_args.clip_grad == 1.0
    assert parser_args.grad_accum_steps == 1


def test_train_yolo_dry_run():
    """Verify train_yolo dry-run validates parameters without failing."""
    parser = build_yolo_parser()
    args = parser.parse_args(["--data", "classroom.yaml", "--dry-run", "--batch", "8", "--imgsz", "640"])
    res = train_yolo(args)
    assert res["status"] == "dry_run"
    assert res["model"] == "yolov8s.pt"


def test_tune_hyperparameters_dry_run():
    """Verify tune_hyperparameters dry-run passes."""
    parser = build_tune_parser()
    args = parser.parse_args(["--dry-run", "--n-trials", "5"])
    res = run_tuning(args)
    assert res["status"] == "dry_run"
    assert res["n_trials"] == 5


def test_probability_calibration_ece_reduction():
    """Verify that temperature scaling reduces Expected Calibration Error on overconfident logits."""
    rng = np.random.default_rng(42)
    n = 200
    targets = rng.integers(0, 6, size=n)
    logits = rng.normal(0, 1.0, size=(n, 6))
    for i in range(n):
        logits[i, targets[i]] += 3.0  # Overconfident ground truth

    res = calibrate_and_evaluate(logits, targets)
    assert "temperature" in res
    assert res["calibrated_ece"] < res["uncalibrated_ece"]
    assert res["ece_reduction_pct"] > 0.0


def test_softmax_temperature_behavior():
    """Verify temperature scaling softens or sharpens probability distributions."""
    logits = np.array([[2.0, 1.0, 0.0]])
    p_standard = softmax(logits, temperature=1.0)
    p_soft = softmax(logits, temperature=2.0)
    p_sharp = softmax(logits, temperature=0.5)

    # Higher temperature -> higher entropy (flatter distribution)
    assert p_soft[0, 0] < p_standard[0, 0]
    # Lower temperature -> lower entropy (sharper distribution)
    assert p_sharp[0, 0] > p_standard[0, 0]
    # Sums to 1.0
    assert np.isclose(np.sum(p_soft), 1.0)
    assert np.isclose(np.sum(p_sharp), 1.0)


def test_bytetrack_configurable_thresholds():
    """Verify ByteTrackWrapper records custom tracking thresholds."""
    tracker = ByteTrackWrapper(
        frame_rate=20,
        track_thresh=0.55,
        track_buffer=60,
        match_thresh=0.75,
        min_conf=0.12,
    )
    assert tracker.frame_rate == 20
    assert tracker.track_thresh == 0.55
    assert tracker.track_buffer == 60
    assert tracker.match_thresh == 0.75
    assert tracker.min_conf == 0.12


def test_behavior_classifier_temperature_init():
    """Verify BehaviorClassifier initializes with configurable temperature."""
    clf = BehaviorClassifier(temperature=1.35)
    assert clf._temperature == 1.35
    assert isinstance(clf._last, BehaviorResult)


if __name__ == "__main__":
    test_train_classifier_tuning_arguments()
    test_train_yolo_dry_run()
    test_tune_hyperparameters_dry_run()
    test_probability_calibration_ece_reduction()
    test_softmax_temperature_behavior()
    test_bytetrack_configurable_thresholds()
    test_behavior_classifier_temperature_init()
    print("All model tuning tests passed successfully!")
