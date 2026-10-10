"""
base_dataset.py — Abstract base class for all CognEn training datasets.

Every dataset loader must:
  1. Yield (feature_window, label_index, group_id) triples via __iter__
  2. Declare DATASET_NAME (namespaces its group ids) and GROUNDED_CLASSES
     (the behavior classes its labels can actually ground); __iter__ raises
     if a loader ever emits a class outside GROUNDED_CLASSES.
  3. Implement _iter_samples() as a generator, or iter_sequences() (per-frame
     LabelledSequence objects from training/datasets/sequence_encoder.py,
     windowed by the default _iter_samples).

Real webcam datasets (DAiSEE, EngageNet) carry ENGAGEMENT labels only. They
can ground attentive_screen / distracted / drowsy (as proxies, see each
loader), never attentive_reading / using_phone (no objects in view) and never
absent (a recorded participant is always present). Windows in which the face
is missing often enough for the deployed classifier to answer "absent"
(>= ABSENT_MIN_NO_FACE_FRAMES of 15) are DROPPED for such datasets instead of
being labelled with an engagement class the features contradict.
"""

from __future__ import annotations

import abc
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Iterable, Iterator, Protocol

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))   # v2.0 root (features/)

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


class DropCounter(Protocol):
    """Anything with a mutable ``dropped_absent_windows`` tally (a dataset)."""
    dropped_absent_windows: int

# Attention scores per class (matches temporal-classifier.ts BEHAVIOR_ATTENTION_SCORES)
CLASS_ATTENTION_SCORES: dict[str, float] = {
    "attentive_screen":  85.0,
    "attentive_reading": 80.0,
    "distracted":        45.0,
    "using_phone":       25.0,
    "drowsy":            20.0,
    "absent":             0.0,
}

# Feature dimension of the v2 model input: the versioned vector from
# features/feature_vector.py (FEATURE_VECTOR_VERSION 2 = 13 v1.0-compatible
# channels + 2 posture channels). FEATURE_DIM_V1 = 13 is the v1.0 encode13 size.
from features.feature_vector import FEATURE_DIM, FEATURE_DIM_V1, FEATURE_VECTOR_VERSION  # noqa: E402,F401
from inference.behavior_classifier import ABSENT_MIN_NO_FACE_FRAMES  # noqa: E402

WINDOW_SIZE  = 15

# Classes a webcam engagement dataset (DAiSEE, EngageNet) can ground.
ENGAGEMENT_PROXY_CLASSES: frozenset[str] = frozenset({"attentive_screen", "distracted", "drowsy"})

if TYPE_CHECKING:
    from training.datasets.sequence_encoder import LabelledSequence

logger = logging.getLogger("cognen.datasets")


class BaseBehaviorDataset(abc.ABC):
    """
    Abstract dataset. Subclasses yield sliding-window feature arrays.

    Each sample is a tuple (window, label, group_id):
        window   : np.ndarray, shape (WINDOW_SIZE, FEATURE_DIM), dtype float32
        label    : int in [0, 5]
        group_id : hashable identifier (e.g. clip_id, or f"{clip_id}_{track_id}")
            shared by every window drawn from the same underlying clip/subject.
            Namespaced by DATASET_NAME (e.g. "daisee:110001" = one subject).
            Every window of one subject / generated sequence (incl. its
            augmented copies) shares it; train_classifier.py asserts that no
            group appears in both train and val.
    """

    CLASSES      = BEHAVIOR_CLASSES
    CLASS_TO_IDX = CLASS_TO_IDX
    FEATURE_DIM  = FEATURE_DIM
    WINDOW_SIZE  = WINDOW_SIZE
    DATASET_NAME: str = "dataset"
    GROUNDED_CLASSES: frozenset[str] = frozenset(BEHAVIOR_CLASSES)

    def __init__(self, root: str, split: str = "train") -> None:
        """
        Args:
            root:  Path to dataset root directory.
            split: One of "train", "val", "test".
        """
        self.root  = root
        self.split = split
        self.stride = getattr(self, "stride", 5)
        # Windows dropped because the face was missing in >= ABSENT_MIN_NO_FACE_FRAMES
        # frames and this dataset cannot ground "absent" (see module docstring).
        self.dropped_absent_windows = 0

    def iter_sequences(self) -> Iterable[LabelledSequence]:
        """Per-frame labelled sequences (override in sequence-based loaders)."""
        raise NotImplementedError

    def _iter_samples(self) -> Generator[tuple[np.ndarray, int, object], None, None]:
        """Default: slide WINDOW_SIZE windows over iter_sequences()."""
        for seq in self.iter_sequences():
            if len(seq) < self.WINDOW_SIZE:
                continue
            windows, y, groups = self.build_windows(
                list(seq.vectors), list(seq.labels),
                window_size=self.WINDOW_SIZE, stride=self.stride, group_id=seq.group_id,
                no_face=list(seq.no_face),
                max_no_face=None if "absent" in self.GROUNDED_CLASSES else ABSENT_MIN_NO_FACE_FRAMES - 1,
                drop_counter=self,
            )
            for i in range(len(y)):
                yield windows[i], int(y[i]), groups[i]

    def __iter__(self) -> Iterator[tuple[np.ndarray, int, object]]:
        grounded = {CLASS_TO_IDX[c] for c in self.GROUNDED_CLASSES}
        for window, label, group in self._iter_samples():
            if int(label) not in grounded:
                raise ValueError(
                    f"{type(self).__name__} emitted class {BEHAVIOR_CLASSES[int(label)]!r}, which its "
                    f"labels cannot ground (GROUNDED_CLASSES={sorted(self.GROUNDED_CLASSES)})"
                )
            yield window, label, group

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def build_windows(
        frames: list[np.ndarray],
        labels: list[int],
        window_size: int = WINDOW_SIZE,
        stride: int = 1,
        group_id: object = None,
        no_face: list[bool] | None = None,
        max_no_face: int | None = None,
        drop_counter: DropCounter | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Slide a window over a per-frame feature sequence.

        Args:
            frames:      List of 1-D feature vectors, each shape (FEATURE_DIM,).
            labels:      Per-frame label indices, length == len(frames).
            window_size: Number of frames per window.
            stride:      Step between consecutive windows.
            group_id:    Identifier shared by every window this call produces
                (e.g. a subject id) -- see class docstring.
            no_face:     Optional per-frame "landmarker found no face" flags.
            max_no_face: With no_face: drop windows with MORE no-face frames
                than this (None = keep all).
            drop_counter: object whose ``dropped_absent_windows`` counts drops.

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
            if max_no_face is not None and no_face is not None \
                    and sum(bool(v) for v in no_face[start:end]) > max_no_face:
                if drop_counter is not None:
                    drop_counter.dropped_absent_windows += 1
                continue
            win = np.stack(frames[start:end]).astype(np.float32)  # (W, F)
            windows.append(win)
            y.append(labels[end - 1])  # label of last frame in window
            groups.append(group_id)

        if not windows:
            return (np.empty((0, window_size, FEATURE_DIM), dtype=np.float32),
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=object))
        return np.stack(windows), np.array(y, dtype=np.int64), np.array(groups, dtype=object)

    @staticmethod
    def encode_feature(
        gaze_x: float = 0.0,
        gaze_y: float = 0.0,
        pitch_norm: float = 0.0,
        yaw_norm: float = 0.0,
        ear: float = 0.3,
        perclos: float = 0.05,
        stability: float = 0.9,
        has_book: float = 0.0,
        has_phone: float = 0.0,
        has_laptop: float = 0.0,
        has_none: float = 1.0,
        fidget: float = 0.05,
        proximity_code: float = 0.0,
        posture_upright: float = 1.0,
        posture_lean_norm: float = 0.0,
    ) -> np.ndarray:
        """Build the FEATURE_DIM (15) vector used by the v2 1D-CNN, already
        normalised (pitch_norm/gaze_y: down = negative; see feature_vector.py).
        Defaults equal features.feature_vector.encode_feature_vector_v2's
        defaults (missing metrics), i.e. what inference encodes."""
        return np.array(
            [gaze_x, gaze_y, pitch_norm, yaw_norm, ear, perclos,
             stability, has_book, has_phone, has_laptop, has_none,
             fidget, proximity_code, posture_upright, posture_lean_norm],
            dtype=np.float32,
        )
