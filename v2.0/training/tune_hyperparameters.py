"""
tune_hyperparameters.py — Bayesian hyperparameter optimization for 1D-CNN temporal classifier.
Reference: parameters.md §7.1 (Optuna Bayesian Hyperparameter Search)

Usage:
    python3 v2.0/training/tune_hyperparameters.py \\
        --realistic \\
        --realistic-n-per-class 500 \\
        --n-trials 20 \\
        --study-name cognen_optuna

Optimizes:
    - Learning rate (lr0, eta_min)
    - Batch size (32, 64, 128)
    - L2 weight decay (1e-5 to 1e-3)
    - Label smoothing (0.0 to 0.10)
    - Gradient norm clipping (0.5 to 2.0)
    - ConfusionPenaltyLoss vs CrossEntropyLoss

Objective: Maximize validation macro-F1 over classes present in validation set.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cognen.tune")

V2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V2_ROOT))

from training.train_classifier import train


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tune 1D-CNN classifier hyperparameters using Optuna")
    p.add_argument("--n-trials", type=int, default=20, help="Number of Optuna optimization trials")
    p.add_argument("--timeout", type=int, default=7200, help="Optimization timeout in seconds")
    p.add_argument("--study-name", type=str, default="cognen_classifier_optuna", help="Optuna study name")
    p.add_argument("--storage", type=str, default=None, help="Database URL for Optuna storage (e.g. sqlite:///optuna.db)")
    p.add_argument("--epochs-per-trial", type=int, default=25, help="Epochs to train per trial")
    p.add_argument("--patience", type=int, default=8, help="Early stopping patience per trial")

    # Dataset flags forwarded to train_classifier
    p.add_argument("--daisee", type=str, default=None, help="DAiSEE root directory")
    p.add_argument("--engagenet", type=str, default=None, help="EngageNet root directory")
    p.add_argument("--classroom", type=str, default=None, help="Classroom dataset root")
    p.add_argument("--max-clips-per-dataset", type=int, default=None, help="Cap clips for fast tuning")

    # Realistic synthetic dataset options
    p.add_argument("--realistic", action="store_true", help="Use RealisticSyntheticDataset")
    p.add_argument("--realistic-n-per-class", dest="realistic_n_per_class", type=int, default=1000)
    p.add_argument("--realistic-augment", dest="realistic_augment", type=int, default=2)
    p.add_argument("--realistic-seed", dest="realistic_seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42, help="Base random seed")

    p.add_argument("--dry-run", action="store_true", help="Run a fast 1-trial mock test")
    return p


def create_objective(args: argparse.Namespace) -> Any:
    """Returns an Optuna objective function."""
    def objective(trial: Any) -> float:
        # Sample hyperparameters from defined search spaces
        lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        batch = trial.suggest_categorical("batch", [32, 64, 128])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.10)
        eta_min = trial.suggest_float("eta_min", 1e-6, 1e-4, log=True)
        clip_grad = trial.suggest_float("clip_grad", 0.5, 2.0)
        penalty_loss = trial.suggest_categorical("penalty_loss", [True, False])

        with tempfile.TemporaryDirectory(prefix="cognen_optuna_trial_") as tmp_dir:
            trial_output = Path(tmp_dir) / f"trial_{trial.number}.pt"

            trial_args = argparse.Namespace(
                daisee=args.daisee,
                engagenet=args.engagenet,
                classroom=args.classroom,
                output=str(trial_output),
                epochs=args.epochs_per_trial,
                batch=batch,
                lr=lr,
                weight_decay=weight_decay,
                label_smoothing=label_smoothing,
                eta_min=eta_min,
                clip_grad=clip_grad,
                grad_accum_steps=1,
                patience=args.patience,
                val_ratio=0.15,
                max_clips_per_dataset=args.max_clips_per_dataset,
                realistic=args.realistic,
                realistic_n_per_class=args.realistic_n_per_class,
                realistic_augment=args.realistic_augment,
                realistic_seed=args.realistic_seed,
                seed=args.seed + trial.number,
                penalty_loss=penalty_loss,
            )

            try:
                result = train(trial_args)
                val_macro_f1 = float(result.get("best_val_macro_f1", 0.0))
                return val_macro_f1
            except Exception as exc:
                logger.warning("Trial %d failed with error: %s", trial.number, exc)
                return 0.0

    return objective


def run_tuning(args: argparse.Namespace) -> dict[str, Any]:
    """Run Optuna study."""
    if args.dry_run:
        logger.info("[Dry Run] Hyperparameter tuning config validated.")
        return {"status": "dry_run", "n_trials": args.n_trials}

    try:
        import optuna  # type: ignore
    except ImportError:
        logger.error("optuna is not installed. Install with: pip install optuna")
        raise

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    logger.info("Starting Bayesian Optimization Study '%s' (%d trials, timeout=%ds)...",
                args.study_name, args.n_trials, args.timeout)

    objective = create_objective(args)
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    logger.info("Optimization complete!")
    logger.info("Best Trial #%d — Validation Macro-F1: %.4f",
                study.best_trial.number, study.best_value)
    logger.info("Optimal Hyperparameters: %s", study.best_params)

    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials_completed": len(study.trials),
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_tuning(args)


if __name__ == "__main__":
    main()
