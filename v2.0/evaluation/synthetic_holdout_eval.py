"""
synthetic_holdout_eval.py — evaluate a behavior ONNX model on an
INDEPENDENT-SEED synthetic hold-out set.

This is NOT a real-data evaluation. The hold-out windows come from the same
hand-parameterised generator the model was trained on
(training/datasets/realistic_synthetic_loader.py), only with a different RNG
seed, so the numbers measure whether the model learned the generator
(circular), not classroom behaviour. It is useful only as a regression /
plumbing check (feature version, sign convention, export).

Reports:
  clean      — base windows, seed --seed
  augmented  — the same windows passed through augment_window (time warp,
               feature dropout, gaze jitter, context injection), seed --seed+1
  sign_flip  — reading / phone windows with pitch_norm and gaze_y negated
               (i.e. the OLD +down convention); recall should collapse,
               showing the model depends on the down = negative convention.

Usage:
    python3 v2.0/evaluation/synthetic_holdout_eval.py \\
        --model v2.0/models/behavior_classifier_v3.onnx --seed 2026 --n-per-class 1000 \\
        --out v2.0/evaluation/reports/synthetic_holdout_v3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.datasets.base_dataset import BEHAVIOR_CLASSES, CLASS_TO_IDX  # noqa: E402
from training.datasets.realistic_synthetic_loader import (  # noqa: E402
    RealisticSyntheticDataset,
    augment_window,
)


def _predict(sess, windows: np.ndarray) -> np.ndarray:
    dim = int(sess.get_inputs()[0].shape[1])
    x = np.ascontiguousarray(windows[:, :, :dim].transpose(0, 2, 1)).astype(np.float32)
    out = []
    for i in range(0, len(x), 512):
        out.append(sess.run(["logits"], {"input": x[i:i + 512]})[0].argmax(1))
    return np.concatenate(out)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Accuracy / macro-F1 (all six classes are present in this balanced set) / per class."""
    n = len(BEHAVIOR_CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    per = {}
    f1s = []
    for i, c in enumerate(BEHAVIOR_CLASSES):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4), "f1": round(float(f1), 4)}
        f1s.append(f1)
    return {
        "n": int(len(y)),
        "accuracy": round(float((y == p).mean()), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "per_class": per,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(a.model)

    ds = RealisticSyntheticDataset(split="val", n_per_class=a.n_per_class, augment_factor=0,
                                   seed=a.seed, val_fraction=1.0)
    X, y = ds._windows, ds._labels
    rng = np.random.default_rng(a.seed + 1)
    X_aug = np.stack([augment_window(w, BEHAVIOR_CLASSES[int(c)], rng) for w, c in zip(X, y)])

    down_cls = [CLASS_TO_IDX["attentive_reading"], CLASS_TO_IDX["using_phone"]]
    mask = np.isin(y, down_cls)
    X_flip = X[mask].copy()
    X_flip[:, :, 1] *= -1   # gaze_y
    X_flip[:, :, 2] *= -1   # pitch_norm
    p_flip = _predict(sess, X_flip)

    report = {
        "data_source": "synthetic",
        "claims_allowed": False,
        "WARNING": "synthetic-only, circular: hold-out drawn from the training generator with another seed; "
                   "says nothing about real classroom accuracy",
        "model": Path(a.model).name,
        "input_channels": int(sess.get_inputs()[0].shape[1]),
        "seed": a.seed,
        "n_per_class": a.n_per_class,
        "clean": _metrics(y, _predict(sess, X)),
        "augmented": _metrics(y, _predict(sess, X_aug)),
        "sign_flip_reading_phone_recall": round(float((p_flip == y[mask]).mean()), 4),
    }
    txt = json.dumps(report, indent=2)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
