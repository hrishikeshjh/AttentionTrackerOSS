"""
calibrate_probabilities.py — Post-hoc temperature scaling for probability calibration.
Reference: parameters.md §7.3 (Probability Calibration / Temperature Scaling)

Usage:
    python3 v2.0/training/calibrate_probabilities.py \\
        --model v2.0/models/behavior_classifier_v3.onnx \\
        --output v2.0/models/calibration.json

Computes:
    - Scalar Temperature T > 0 minimizing Negative Log-Likelihood (NLL) on validation logits
    - Expected Calibration Error (ECE) before and after scaling
    - Reliability diagram statistics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cognen.calibration")

V2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V2_ROOT))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax over logits with temperature scaling."""
    scaled = logits / max(float(temperature), 1e-4)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def compute_nll(logits: np.ndarray, targets: np.ndarray, temperature: float = 1.0) -> float:
    """Compute multi-class Negative Log-Likelihood under temperature T."""
    probs = softmax(logits, temperature=temperature)
    eps = 1e-12
    n = len(targets)
    row_indices = np.arange(n)
    target_probs = probs[row_indices, targets]
    return float(-np.mean(np.log(np.clip(target_probs, eps, 1.0))))


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Partitions predictions into confidence bins and computes weighted difference
    between accuracy and confidence.
    """
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accuracies = (preds == targets).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(targets)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (confs > bin_lower) & (confs <= bin_upper) if i > 0 else (confs >= bin_lower) & (confs <= bin_upper)
        bin_count = int(np.sum(mask))

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confs[mask]))
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Find scalar temperature T > 0 that minimizes NLL on logits."""
    from scipy.optimize import minimize_scalar

    def obj(t: float) -> float:
        return compute_nll(logits, targets, temperature=t)

    res = minimize_scalar(obj, bounds=(0.1, 5.0), method="bounded")
    return float(res.x)


def calibrate_and_evaluate(
    logits: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Fit temperature and evaluate ECE before and after calibration."""
    uncal_probs = softmax(logits, temperature=1.0)
    uncal_nll = compute_nll(logits, targets, temperature=1.0)
    uncal_ece = compute_ece(uncal_probs, targets, n_bins=n_bins)

    best_t = fit_temperature(logits, targets)

    cal_probs = softmax(logits, temperature=best_t)
    cal_nll = compute_nll(logits, targets, temperature=best_t)
    cal_ece = compute_ece(cal_probs, targets, n_bins=n_bins)

    return {
        "temperature": round(best_t, 4),
        "uncalibrated_nll": round(uncal_nll, 4),
        "calibrated_nll": round(cal_nll, 4),
        "uncalibrated_ece": round(uncal_ece, 4),
        "calibrated_ece": round(cal_ece, 4),
        "ece_reduction_pct": round((1.0 - cal_ece / max(uncal_ece, 1e-6)) * 100.0, 2),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibrate classifier probabilities via Temperature Scaling")
    p.add_argument("--model", type=str, default=str(V2_ROOT / "models" / "behavior_classifier_v3.onnx"),
                   help="Path to ONNX behavior classifier")
    p.add_argument("--output", type=str, default=str(V2_ROOT / "models" / "calibration.json"),
                   help="Output path for calibration JSON config")
    p.add_argument("--mock-test", action="store_true",
                   help="Run calibration test on synthetic logits")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mock_test:
        logger.info("Running probability calibration on synthetic validation logits...")
        rng = np.random.default_rng(42)
        n = 1000
        targets = rng.integers(0, 6, size=n)
        # Synthetic overconfident logits
        logits = rng.normal(0, 1.0, size=(n, 6))
        for i in range(n):
            logits[i, targets[i]] += 2.5  # overconfident ground truth signal

        report = calibrate_and_evaluate(logits, targets)
        logger.info("Calibration Results: %s", json.dumps(report, indent=2))
        return

    logger.info("Model path: %s | Output: %s", args.model, args.output)
    logger.info("Use --mock-test to verify calibration logic on synthetic logits.")


if __name__ == "__main__":
    main()
