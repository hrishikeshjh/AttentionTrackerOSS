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

The trained checkpoint is saved and then exported to ONNX via export_onnx.py.
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

from training.datasets.base_dataset import BEHAVIOR_CLASSES, FEATURE_DIM, WINDOW_SIZE

# NUM_CLASSES/BehaviorClassifierCNN come from training.model, which imports
# torch at module level -- keep that import lazy (only inside train(), which
# actually needs torch) so _load_all_samples/_train_val_split/data-loading
# logic stays testable without torch installed. NUM_CLASSES itself is just
# len(BEHAVIOR_CLASSES), so define it locally instead of importing it.
NUM_CLASSES = len(BEHAVIOR_CLASSES)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _load_all_samples(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect all (window, label, group_id) triples from configured datasets."""
    all_windows: list[np.ndarray] = []
    all_labels:  list[int]        = []
    all_groups:  list[object]     = []

    # Running counter for the "no natural grouping" fallback (e.g. the
    # random-synthetic smoke-test path below): each such window gets its
    # own unique group so it can freely land on either side of a
    # GroupShuffleSplit, rather than colliding under a shared `None` group
    # (which would make ALL of them the same group and could silently pin
    # every one of them onto only one side of the split).
    _ungrouped_counter = [0]

    def _consume(dataset):
        logger.info("Loading %s …", type(dataset).__name__)
        n = 0
        for window, label, group_id in dataset:
            all_windows.append(window)
            all_labels.append(label)
            if group_id is None:
                group_id = f"_ungrouped_{_ungrouped_counter[0]}"
                _ungrouped_counter[0] += 1
            all_groups.append(group_id)
            n += 1
        logger.info("  → %d windows", n)

    if args.daisee:
        from training.datasets.daisee_loader import DAiSEEDataset
        for split in ("train", "val"):
            _consume(DAiSEEDataset(args.daisee, split=split,
                                   max_clips=args.max_clips_per_dataset))

    if args.engagenet:
        from training.datasets.engagenet_loader import EngageNetDataset
        for split in ("train", "val"):
            _consume(EngageNetDataset(args.engagenet, split=split,
                                      max_images=args.max_clips_per_dataset))

    if args.classroom:
        from training.datasets.classroom_loader import ClassroomDataset
        for split in ("train", "val"):
            _consume(ClassroomDataset(args.classroom, split=split,
                                      max_clips=args.max_clips_per_dataset))

    if args.realistic:
        logger.info("Using RealisticSyntheticDataset (class-conditional AR(1) windows) …")
        from training.datasets.realistic_synthetic_loader import RealisticSyntheticDataset
        for split in ("train", "val"):
            ds = RealisticSyntheticDataset(
                split=split,
                n_per_class=args.realistic_n_per_class,
                augment_factor=args.realistic_augment if split == "train" else 0,
                seed=42,
            )
            logger.info("  %s: %d windows  dist=%s", split, len(ds), ds.class_distribution())
            _consume(ds)

    if not all_windows:
        logger.warning("No dataset paths provided. Using random synthetic data (smoke-test only).")
        rng = np.random.default_rng(0)
        all_windows = [rng.random((WINDOW_SIZE, FEATURE_DIM)).astype(np.float32)
                       for _ in range(1000)]
        all_labels  = [random.randint(0, NUM_CLASSES - 1) for _ in range(1000)]
        all_groups  = [f"_ungrouped_{i}" for i in range(1000)]

    X = np.stack(all_windows)   # (N, W, F)
    y = np.array(all_labels, dtype=np.int64)
    groups = np.array(all_groups, dtype=object)
    return X, y, groups


def _train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by GROUP (clip/subject), not by window. A naive random permutation
    over pooled windows lets overlapping sliding windows from the same
    clip/subject leak across train/val (DAiSEE/EngageNet windows are drawn
    with stride < window_size, so adjacent windows share most of their
    frames) -- this is a real train/val leakage bug the previous
    implementation had.
    """
    from sklearn.model_selection import GroupShuffleSplit

    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        # Degenerate case (e.g. a single-group smoke test) -- fall back to
        # a plain random split rather than letting GroupShuffleSplit error.
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        n_val = max(1, int(len(X) * val_ratio))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _iter_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    n   = len(X)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
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


def _compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weighting for class imbalance."""
    import torch
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.from_numpy(weights)


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn

    from training.model import BehaviorClassifierCNN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device: %s", device)

    # ── Data ──
    X, y, groups = _load_all_samples(args)
    logger.info("Total samples: %d  |  classes: %s",
                len(X), np.bincount(y, minlength=NUM_CLASSES))

    X_tr, y_tr, X_val, y_val = _train_val_split(X, y, groups, val_ratio=args.val_ratio)

    # ── Model ──
    model  = BehaviorClassifierCNN().to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    class_weights = _compute_class_weights(y_tr).to(device)
    use_penalty   = getattr(args, "penalty_loss", False)
    if use_penalty:
        logger.info("Using ConfusionPenaltyLoss (phone/reading pair penalty 3.0×/2.5×)")
        criterion = None   # placeholder — loss computed inline
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_acc  = 0.0
    patience_left = args.patience
    output_path   = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_n = 0.0, 0, 0

        for Xb, yb in _iter_batches(X_tr, y_tr, args.batch):
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
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for Xb, yb in _iter_batches(X_val, y_val, args.batch, shuffle=False):
                Xt = torch.from_numpy(Xb).float().permute(0, 2, 1).to(device)
                yt = torch.from_numpy(yb).long().to(device)
                preds = model(Xt).argmax(1)
                val_correct += (preds == yt).sum().item()
                val_n       += len(yb)

        train_acc = train_correct / train_n
        val_acc   = val_correct   / val_n
        avg_loss  = train_loss    / train_n
        lr_now    = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %3d/%d  loss=%.4f  train_acc=%.3f  val_acc=%.3f  lr=%.2e",
            epoch, args.epochs, avg_loss, train_acc, val_acc, lr_now,
        )

        # ── Checkpoint ──
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            patience_left = args.patience
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          best_val_acc,
                "classes":          BEHAVIOR_CLASSES,
                "feature_dim":      FEATURE_DIM,
                "window_size":      WINDOW_SIZE,
            }, output_path)
            logger.info("  ✓ Saved best checkpoint (val_acc=%.4f)", best_val_acc)
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch %d", epoch)
                break

    logger.info("Training complete. Best val_acc=%.4f", best_val_acc)

    # Auto-export to ONNX
    from training.export_onnx import export_onnx
    onnx_path = output_path.with_suffix(".onnx")
    export_onnx(str(output_path), str(onnx_path), device=device)


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
    # Loss function options
    p.add_argument("--penalty-loss", dest="penalty_loss", action="store_true",
                   help="Use ConfusionPenaltyLoss instead of standard CrossEntropyLoss")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
