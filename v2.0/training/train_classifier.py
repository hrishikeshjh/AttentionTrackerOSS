"""
train_classifier.py — Train the 1D-CNN temporal behavior classifier.

Usage:
    python3 v2.0/training/train_classifier.py \\
        --daisee    /data/daisee \\
        --engagenet /data/engagenet \\
        --classroom /data/classroom \\
        --epochs 50 \\
        --batch  64 \\
        --output v2.0/models/behavior_classifier.pt

The trained checkpoint is saved and then exported to ONNX via export_onnx.py
(which raises if the ONNX logits differ from PyTorch).

Split: each loader's own train / val split is used (never pooled and
re-split); every window carries a group id (subject for real data, the base
window for synthetic data incl. its augmented copies) and the split is
asserted group-disjoint. Model selection: val macro-F1 over the classes
present in val (ties: lower val CE). Seeds: --seed seeds python / numpy /
torch and the batch order.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    # Only needed for type annotations below -- the real import stays
    # lazy (inside train()/_get_penalty()/_compute_class_weights()) so
    # this module stays importable/testable without torch installed.
    import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cognen.train")

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.datasets.base_dataset import BEHAVIOR_CLASSES, FEATURE_DIM, FEATURE_VECTOR_VERSION, WINDOW_SIZE

# NUM_CLASSES/BehaviorClassifierCNN come from training.model, which imports
# torch at module level -- keep that import lazy (only inside train(), which
# actually needs torch) so _load_all_samples/_train_val_split/data-loading
# logic stays testable without torch installed. NUM_CLASSES itself is just
# len(BEHAVIOR_CLASSES), so define it locally instead of importing it.
NUM_CLASSES = len(BEHAVIOR_CLASSES)


# ── Dataset helpers ───────────────────────────────────────────────────────────

class GroupLeakageError(AssertionError):
    """A group (subject / generated sequence) appears in both train and val."""


def assert_no_group_leakage(train_groups, val_groups) -> None:
    """Raise GroupLeakageError if any group id is on both sides of the split."""
    overlap = set(np.asarray(train_groups, dtype=object).tolist()) & \
        set(np.asarray(val_groups, dtype=object).tolist())
    if overlap:
        sample = sorted(map(str, overlap))[:5]
        raise GroupLeakageError(f"{len(overlap)} group(s) in both train and val, e.g. {sample}")


def _load_all_samples(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect (window, label, group_id, split) from the configured datasets.

    split is the loader's OWN split ("train" / "val"): every real dataset and
    the synthetic generator ship a disjoint train/val split, which
    _assemble_split keeps (pooling everything and re-splitting threw it away
    and, with ungrouped synthetic windows, put augmented copies of a training
    window into val). Only the smoke-test fallback has split None.
    """
    all_windows: list[np.ndarray] = []
    all_labels:  list[int]        = []
    all_groups:  list[object]     = []
    all_splits:  list[object]     = []

    def _consume(dataset, split):
        logger.info("Loading %s (%s) …", type(dataset).__name__, split)
        n = 0
        for window, label, group_id in dataset:
            if group_id is None:
                raise ValueError(f"{type(dataset).__name__} yielded a window without a group id")
            all_windows.append(window)
            all_labels.append(label)
            all_groups.append(group_id)
            all_splits.append(split)
            n += 1
        dropped = getattr(dataset, "dropped_absent_windows", 0)
        logger.info("  → %d windows (%d dropped: face missing, class 'absent' not groundable)", n, dropped)

    if args.daisee:
        from training.datasets.daisee_loader import DAiSEEDataset
        for split in ("train", "val"):
            _consume(DAiSEEDataset(args.daisee, split=split,
                                   max_clips=args.max_clips_per_dataset), split)

    if args.engagenet:
        from training.datasets.engagenet_loader import EngageNetDataset
        for split in ("train", "val"):
            _consume(EngageNetDataset(args.engagenet, split=split,
                                      max_images=args.max_clips_per_dataset), split)

    if args.classroom:
        from training.datasets.classroom_loader import ClassroomDataset
        for split in ("train", "val"):
            _consume(ClassroomDataset(args.classroom, split=split,
                                      max_clips=args.max_clips_per_dataset), split)

    if args.realistic:
        logger.info("Using RealisticSyntheticDataset (class-conditional AR(1) windows) …")
        from training.datasets.realistic_synthetic_loader import RealisticSyntheticDataset
        for split in ("train", "val"):
            ds = RealisticSyntheticDataset(
                split=split,
                n_per_class=args.realistic_n_per_class,
                augment_factor=args.realistic_augment if split == "train" else 0,
                seed=getattr(args, "realistic_seed", 42),
            )
            logger.info("  %s: %d windows  dist=%s", split, len(ds), ds.class_distribution())
            _consume(ds, split)

    if not all_windows:
        logger.warning("No dataset paths provided. Using random synthetic data (smoke-test only).")
        rng = np.random.default_rng(getattr(args, "seed", 42))
        all_windows = [rng.random((WINDOW_SIZE, FEATURE_DIM)).astype(np.float32)
                       for _ in range(1000)]
        all_labels  = rng.integers(0, NUM_CLASSES, size=1000).tolist()
        all_groups  = [f"smoke:{i}" for i in range(1000)]
        all_splits  = [None] * 1000

    X = np.stack(all_windows)   # (N, W, F)
    y = np.array(all_labels, dtype=np.int64)
    groups = np.array(all_groups, dtype=object)
    splits = np.array(all_splits, dtype=object)
    return X, y, groups, splits


def _assemble_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: np.ndarray,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Train / val index sets that respect the loaders' own splits.

    * windows tagged "train" / "val" stay on that side;
    * untagged windows (split None) are group-split with GroupShuffleSplit;
    * a group that a dataset placed in BOTH of its official splits (e.g. a
      participant in DAiSEE Train and Validation) is removed from val
      (logged + counted), since keeping it would leak that participant;
    * the result is checked with assert_no_group_leakage.

    Returns (train_idx, val_idx, groups_train, groups_val, info).
    """
    n = len(X)
    tags = np.array([s if s in ("train", "val") else None for s in splits.tolist()], dtype=object)
    train_mask = tags == "train"
    val_mask = tags == "val"
    untagged = np.where(np.array([t is None for t in tags.tolist()]))[0]
    if len(untagged):
        sub_tr, sub_val = _group_split_indices(groups[untagged], val_ratio, seed)
        train_mask[untagged[sub_tr]] = True
        val_mask[untagged[sub_val]] = True

    train_groups = set(groups[train_mask].tolist())
    leaking = np.array([val_mask[i] and groups[i] in train_groups for i in range(n)], dtype=bool)
    n_leak_groups = len(set(groups[leaking].tolist()))
    if leaking.any():
        logger.warning("Removing %d val windows of %d group(s) that are also in train "
                       "(the dataset's official splits share them)", int(leaking.sum()), n_leak_groups)
        val_mask &= ~leaking

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    assert_no_group_leakage(groups[train_idx], groups[val_idx])
    info = {
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_train_groups": len(set(groups[train_idx].tolist())),
        "n_val_groups": len(set(groups[val_idx].tolist())),
        "val_windows_removed_for_leakage": int(leaking.sum()),
        "val_groups_removed_for_leakage": int(n_leak_groups),
    }
    return train_idx, val_idx, groups[train_idx], groups[val_idx], info


def _group_split_indices(groups: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """GroupShuffleSplit indices over *groups* (>= 2 groups required)."""
    from sklearn.model_selection import GroupShuffleSplit

    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        raise ValueError(
            "cannot make a leakage-free split from a single group: provide data from "
            "more than one subject / sequence"
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    idx = np.arange(len(groups))
    train_idx, val_idx = next(splitter.split(idx, idx, groups=groups))
    return train_idx, val_idx


def _train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by GROUP (subject / generated sequence), not by window, for data
    without its own split. Overlapping sliding windows of one subject must
    not land on both sides. A single group cannot be split without leakage
    and raises (it used to fall back to a random per-window split, i.e.
    silent leakage).
    """
    train_idx, val_idx = _group_split_indices(groups, val_ratio, seed)
    assert_no_group_leakage(groups[train_idx], groups[val_idx])
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _iter_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Mini-batches; shuffling uses *rng* (seeded by train()) so runs repeat."""
    n   = len(X)
    if shuffle:
        idx = (rng if rng is not None else np.random.default_rng()).permutation(n)
    else:
        idx = np.arange(n)
    for start in range(0, n, batch_size):
        bi = idx[start: start + batch_size]
        yield X[bi], y[bi]


# ── Training loop ─────────────────────────────────────────────────────────────

class ConfusionPenaltyLoss:
    """
    CrossEntropyLoss with pair-specific penalty multipliers.

    Asymmetrically penalises the most costly class confusions. The full
    6x6 matrix (all 30 off-diagonal pairs, each with a documented
    rationale) lives in training/penalty_matrix.py -- previously only 3 of
    30 pairs were specified here, with the remaining 27 silently
    defaulting to 1.0x (plain cross-entropy).
    """

    PENALTY: np.ndarray | None = None

    @classmethod
    def _get_penalty(cls) -> torch.Tensor:
        import torch

        from training.penalty_matrix import build_penalty_matrix
        if cls.PENALTY is None:
            cls.PENALTY = build_penalty_matrix()
        return torch.from_numpy(cls.PENALTY)

    @classmethod
    def compute(cls, logits: torch.Tensor, targets: torch.Tensor,
                class_weights: torch.Tensor | None = None,
                label_smoothing: float = 0.05) -> torch.Tensor:
        import torch.nn.functional as F

        penalty = cls._get_penalty().to(logits.device)    # (C, C)
        ce = F.cross_entropy(
            logits, targets,
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction="none",
        )                                                  # (B,)
        pred  = logits.argmax(dim=-1)                      # (B,)
        pen   = penalty[targets, pred]                     # (B,)
        return (ce * pen).mean()


def class_weights_np(y: np.ndarray) -> np.ndarray:
    """
    Inverse-frequency class weights, normalised to mean 1 over the classes
    PRESENT in *y*. Absent classes get weight 0 (previously their count was
    set to 1, giving them the largest weight and skewing the normalisation --
    e.g. DAiSEE has no reading / phone / absent windows at all).
    """
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    present = counts > 0
    weights = np.zeros(NUM_CLASSES, dtype=np.float64)
    weights[present] = 1.0 / counts[present]
    if present.any():
        weights[present] *= present.sum() / weights[present].sum()
    return weights.astype(np.float32)


def _compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weighting for class imbalance (see class_weights_np)."""
    import torch
    return torch.from_numpy(class_weights_np(y))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro-F1 over the classes present in y_true (a class with no support
    has an undefined F1 and must not count as 0). Model selection uses this,
    not accuracy: with imbalanced real data, accuracy rewards predicting the
    majority class.
    """
    classes = np.unique(y_true)
    if len(classes) == 0:
        return float("nan")
    f1s = []
    for c in classes:
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def is_better(f1: float, ce: float, best_f1: float, best_ce: float, eps: float = 1e-4) -> bool:
    """Checkpoint rule: higher val macro-F1; ties (within 1e-9) broken by lower val CE."""
    if f1 > best_f1 + 1e-9:
        return True
    return abs(f1 - best_f1) <= 1e-9 and ce < best_ce - eps


def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch (and make cuDNN deterministic)."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args: argparse.Namespace) -> dict:
    import torch
    import torch.nn as nn

    from training.model import BehaviorClassifierCNN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device: %s", device)
    seed = getattr(args, "seed", 42)
    seed_everything(seed)
    batch_rng = np.random.default_rng(seed)

    # ── Data ──
    X, y, groups, splits = _load_all_samples(args)
    logger.info("Total samples: %d  |  classes: %s",
                len(X), np.bincount(y, minlength=NUM_CLASSES))

    tr_idx, val_idx, _g_tr, _g_val, split_info = _assemble_split(
        X, y, groups, splits, val_ratio=args.val_ratio, seed=seed)
    X_tr, y_tr, X_val, y_val = X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]
    logger.info("Split (group-disjoint, asserted): %s", split_info)
    if len(X_val) == 0:
        raise ValueError("empty validation set after the leakage-free split")

    # ── Model ──
    model  = BehaviorClassifierCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", n_params)

    class_weights = _compute_class_weights(y_tr).to(device)
    logger.info("Class weights (inverse frequency, present classes): %s",
                np.round(class_weights.cpu().numpy(), 3).tolist())
    use_penalty   = getattr(args, "penalty_loss", False)
    if use_penalty:
        logger.info("Using ConfusionPenaltyLoss (training/penalty_matrix.py)")
        criterion = None   # loss computed inline
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_f1       = -1.0
    best_val_loss = float("inf")
    best_epoch    = 0
    patience_left = args.patience
    output_path   = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_n = 0.0, 0, 0

        for Xb, yb in _iter_batches(X_tr, y_tr, args.batch, rng=batch_rng):
            # Shape: (B, W, F) → (B, F, W)
            Xt = torch.from_numpy(Xb).float().permute(0, 2, 1).to(device)
            yt = torch.from_numpy(yb).long().to(device)

            optimizer.zero_grad()
            logits = model(Xt)
            if use_penalty:
                loss = ConfusionPenaltyLoss.compute(logits, yt, class_weights)
            else:
                loss = criterion(logits, yt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss    += loss.item() * len(yb)
            train_correct += (logits.argmax(1) == yt).sum().item()
            train_n       += len(yb)

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        preds_all: list[np.ndarray] = []
        with torch.no_grad():
            for Xb, yb in _iter_batches(X_val, y_val, args.batch, shuffle=False):
                Xt = torch.from_numpy(Xb).float().permute(0, 2, 1).to(device)
                yt = torch.from_numpy(yb).long().to(device)
                vlogits = model(Xt)
                val_loss_sum += nn.functional.cross_entropy(vlogits, yt, reduction="sum").item()
                preds_all.append(vlogits.argmax(1).cpu().numpy())

        val_pred  = np.concatenate(preds_all)
        val_acc   = float((val_pred == y_val).mean())
        val_f1    = macro_f1(y_val, val_pred)
        val_loss  = val_loss_sum / len(y_val)
        train_acc = train_correct / train_n
        avg_loss  = train_loss    / train_n
        lr_now    = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %3d/%d  loss=%.4f  train_acc=%.3f  val_acc=%.3f  val_macroF1=%.3f  val_ce=%.4f  lr=%.2e",
            epoch, args.epochs, avg_loss, train_acc, val_acc, val_f1, val_loss, lr_now,
        )

        # ── Checkpoint ── val macro-F1 (present classes), ties -> lower val CE
        if is_better(val_f1, val_loss, best_f1, best_val_loss):
            best_f1, best_val_loss, best_epoch = val_f1, val_loss, epoch
            patience_left = args.patience
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1":     best_f1,
                "val_acc":          val_acc,
                "val_ce":           best_val_loss,
                "selection_metric": "val_macro_f1 (classes present in val), ties by val_ce",
                "classes":          BEHAVIOR_CLASSES,
                "feature_dim":      FEATURE_DIM,
                "feature_vector_version": FEATURE_VECTOR_VERSION,
                "window_size":      WINDOW_SIZE,
                "seed":             seed,
                "split":            split_info,
                "n_params":         n_params,
            }, output_path)
            logger.info("  ✓ Saved best checkpoint (val_macroF1=%.4f)", best_f1)
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch %d", epoch)
                break

    logger.info("Training complete. Best val macro-F1=%.4f (epoch %d)", best_f1, best_epoch)

    # Auto-export to ONNX (raises if the ONNX graph does not reproduce the
    # PyTorch logits, see export_onnx.check_onnx_parity)
    from training.export_onnx import export_onnx
    onnx_path = output_path.with_suffix(".onnx")
    export_onnx(str(output_path), str(onnx_path), device=device)
    return {"best_val_macro_f1": best_f1, "best_epoch": best_epoch, "split": split_info,
            "checkpoint": str(output_path), "onnx": str(onnx_path)}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CognEn 1D-CNN behavior classifier")
    p.add_argument("--daisee",    type=str, default=None, help="DAiSEE root directory")
    p.add_argument("--engagenet", type=str, default=None, help="EngageNet root directory")
    p.add_argument("--classroom", type=str, default=None, help="Classroom dataset root")
    p.add_argument("--output",    type=str,
                   default=str(Path(__file__).parent.parent / "models" / "behavior_classifier.pt"),
                   help="Output checkpoint path (.pt)")
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--batch",     type=int,   default=64)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--patience",  type=int,   default=10, help="Early stopping patience")
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--max_clips_per_dataset", type=int, default=None,
                   help="Cap clips/images per dataset (for quick debug runs)")
    # Realistic synthetic data options
    p.add_argument("--realistic", action="store_true",
                   help="Use RealisticSyntheticDataset (class-conditional AR(1) windows)")
    p.add_argument("--realistic-n-per-class", dest="realistic_n_per_class", type=int, default=2000,
                   help="Base windows per class for realistic synthetic data")
    p.add_argument("--realistic-augment", dest="realistic_augment", type=int, default=3,
                   help="Augmented copies per base window (train only)")
    p.add_argument("--realistic-seed", dest="realistic_seed", type=int, default=42,
                   help="RNG seed of the synthetic generator (use a different seed for a held-out test set)")
    p.add_argument("--seed", type=int, default=42, help="torch / numpy / random seed for training")
    # Loss function options
    p.add_argument("--penalty-loss", dest="penalty_loss", action="store_true",
                   help="Use ConfusionPenaltyLoss instead of standard CrossEntropyLoss")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
