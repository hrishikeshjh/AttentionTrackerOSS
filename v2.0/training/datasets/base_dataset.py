"""
base_dataset.py — Abstract base class for all CognEn training datasets.

Every dataset loader must:
  1. Yield (feature_window, label_index) pairs via __iter__
  2. Expose CLASSES and label_map class attributes
  3. Implement _iter_samples() as a generator
"""

from __future__ import annotations
import abc
from typing import Generator, Iterator

import numpy as np


# Canonical 6-class taxonomy (index order is fixed — used by the model)
BEHAVIOR_CLASSES: list[str] = [
    "attentive_screen",   # 0
    "attentive_reading",  # 1
    "using_phone",        # 2
    "distracted",         # 3
    "drowsy",             # 4
    "absent",             # 5
]

CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(BEHAVIOR_CLASSES)}

# Attention scores per class (matches temporal-classifier.ts BEHAVIOR_ATTENTION_SCORES)
CLASS_ATTENTION_SCORES: dict[str, float] = {
    "attentive_screen":  85.0,
    "attentive_reading": 80.0,
    "distracted":        45.0,
    "using_phone":       25.0,
    "drowsy":            20.0,
    "absent":             0.0,
}

# Feature dimension (matches temporal-classifier.ts TemporalFeatureFrame)
FEATURE_DIM = 13
WINDOW_SIZE  = 15


class BaseBehaviorDataset(abc.ABC):
    """
    Abstract dataset. Subclasses yield sliding-window feature arrays.

    Each sample is a tuple (window, label, group_id):
        window   : np.ndarray, shape (WINDOW_SIZE, FEATURE_DIM), dtype float32
        label    : int in [0, 5]
        group_id : hashable identifier (e.g. clip_id, or f"{clip_id}_{track_id}")
            shared by every window drawn from the same underlying clip/subject.
            None for datasets with no natural grouping (e.g. fully independent
            synthetic samples). Used by train_classifier.py's GroupShuffleSplit
            to prevent overlapping sliding windows from the same clip leaking
            across the train/val split.
    """

    CLASSES      = BEHAVIOR_CLASSES
    CLASS_TO_IDX = CLASS_TO_IDX
    FEATURE_DIM  = FEATURE_DIM
    WINDOW_SIZE  = WINDOW_SIZE

    def __init__(self, root: str, split: str = "train") -> None:
        """
        Args:
            root:  Path to dataset root directory.
            split: One of "train", "val", "test".
        """
        self.root  = root
        self.split = split

    @abc.abstractmethod
    def _iter_samples(self) -> Generator[tuple[np.ndarray, int, object], None, None]:
        """Yield (window_array, label_index, group_id) tuples."""

    def __iter__(self) -> Iterator[tuple[np.ndarray, int, object]]:
        return self._iter_samples()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def build_windows(
        frames: list[np.ndarray],
        labels: list[int],
        window_size: int = WINDOW_SIZE,
        stride: int = 1,
        group_id: object = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Slide a window over a per-frame feature sequence.

        Args:
            frames:      List of 1-D feature vectors, each shape (FEATURE_DIM,).
            labels:      Per-frame label indices, length == len(frames).
            window_size: Number of frames per window.
            stride:      Step between consecutive windows.
            group_id:    Identifier shared by every window this call produces
                (e.g. a clip_id) -- see class docstring. None if the caller
                has no natural grouping for these frames.

        Returns:
            windows : np.ndarray (N, window_size, FEATURE_DIM) float32
            y       : np.ndarray (N,) int64 — label of the *last* frame
            groups  : np.ndarray (N,) object — group_id repeated per window
        """
        n = len(frames)
        if n < window_size:
            return (np.empty((0, window_size, FEATURE_DIM), dtype=np.float32),
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=object))

        windows, y, groups = [], [], []
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            win = np.stack(frames[start:end]).astype(np.float32)  # (W, F)
            windows.append(win)
            y.append(labels[end - 1])  # label of last frame in window
            groups.append(group_id)

        return np.stack(windows), np.array(y, dtype=np.int64), np.array(groups, dtype=object)

    @staticmethod
    def encode_feature(
        gaze_x: float = 0.0,
        gaze_y: float = 0.0,
        pitch_norm: float = 0.0,
        yaw_norm: float = 0.0,
        ear: float = 0.3,
        perclos: float = 0.05,
        stability: float = 1.0,
        has_book: float = 0.0,
        has_phone: float = 0.0,
        has_laptop: float = 0.0,
        has_none: float = 1.0,
        fidget: float = 0.0,
        proximity_code: float = 0.0,
    ) -> np.ndarray:
        """Build the 13-dim feature vector used by the 1D-CNN."""
        return np.array(
            [gaze_x, gaze_y, pitch_norm, yaw_norm, ear, perclos,
             stability, has_book, has_phone, has_laptop, has_none,
             fidget, proximity_code],
            dtype=np.float32,
        )
