"""
engagenet_loader.py — EngageNet dataset loader.

EngageNet
  - 127,000 annotated frames from MOOCs / online lectures
  - Per-frame engagement score: 0.0 – 1.0 (continuous)
  - Two annotation formats depending on release version:
      v1: CSV with columns [image_path, engagement_score]
      v2: CSV with columns [image, score, split]

Download: https://github.com/engagenet/engagenet

Expected directory layout:
    <root>/
        images/
            <image_filename>.jpg   (or .png)
        annotations.csv            (v1) or labels.csv (v2)

Engagement score → CognEn behavior mapping:
    ≥ 0.75 → attentive_screen
    ≥ 0.50 → attentive_screen  (above median)
    ≥ 0.25 → distracted
    < 0.25  → drowsy
"""

from __future__ import annotations
import csv
import logging
import sys
from pathlib import Path
from typing import Generator

import numpy as np

from .base_dataset import BaseBehaviorDataset, CLASS_TO_IDX, FEATURE_DIM, WINDOW_SIZE

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from features.feature_vector import extract_feature_vector_from_landmarks

logger = logging.getLogger("cognen.datasets.engagenet")

# EngageNet has no temporal structure per-se; we synthesise pseudo-windows
# by treating consecutive rows in the CSV as sequential frames.
_PSEUDO_SEQUENCE_LEN = 30  # group every 30 frames into a sequence

_SCORE_TO_CLASS = [
    (0.75, "attentive_screen"),
    (0.50, "attentive_screen"),
    (0.25, "distracted"),
    (0.00, "drowsy"),
]


def _score_to_label(score: float) -> int:
    for threshold, cls in _SCORE_TO_CLASS:
        if score >= threshold:
            return CLASS_TO_IDX[cls]
    return CLASS_TO_IDX["drowsy"]


class EngageNetDataset(BaseBehaviorDataset):
    """
    Loads EngageNet image-level annotations, extracts face features per image,
    groups consecutive frames into pseudo-sequences, and yields windows.

    Args:
        root:           Path to EngageNet root.
        split:          "train" | "val" | "test" (filtered from CSV if column present).
        window_size:    Frames per window (default 15).
        stride:         Window stride within each pseudo-sequence.
        max_images:     Cap on images processed (None = all).
        csv_filename:   Name of the annotation CSV file.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        window_size: int = WINDOW_SIZE,
        stride: int = 5,
        max_images: int | None = None,
        csv_filename: str = "annotations.csv",
    ) -> None:
        super().__init__(root, split)
        self.window_size  = window_size
        self.stride       = stride
        self.max_images   = max_images
        self._csv_path    = Path(root) / csv_filename
        self._image_dir   = Path(root) / "images"
        self._face_det    = None

    # ── internal ─────────────────────────────────────────────────────────────

    def _load_rows(self) -> list[tuple[Path, float]]:
        rows: list[tuple[Path, float]] = []
        if not self._csv_path.exists():
            logger.warning("EngageNet CSV not found: %s", self._csv_path)
            return rows

        with open(self._csv_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            has_split  = "split" in fieldnames

            for row in reader:
                if has_split and row.get("split", "").strip() != self.split:
                    continue
                # Accept both 'image' and 'image_path' column names
                img_name = row.get("image") or row.get("image_path", "")
                img_name = img_name.strip()
                score    = float(row.get("score") or row.get("engagement_score", 0))
                img_path = self._image_dir / img_name
                rows.append((img_path, score))

        return rows

    def _get_face_detector(self):
        if self._face_det is None:
            import sys
            from pathlib import Path as _P
            sys.path.insert(0, str(_P(__file__).parent.parent.parent))
            from perception.face_landmarks import FaceLandmarksDetector
            self._face_det = FaceLandmarksDetector(max_faces=1)
        return self._face_det

    def _frame_features(self, img_path: Path) -> np.ndarray:
        """
        Uses the canonical extract_feature_vector_from_landmarks (calls
        calculate_ear/calculate_head_pose/calculate_gaze from
        features/face_metrics.py) instead of a hand-rolled pitch/EAR
        formula -- this also fills in gaze_x/gaze_y, which the previous
        version of this method left at 0.0 (encode_feature's default,
        i.e. no gaze signal at all was ever extracted from EngageNet
        images).
        """
        try:
            import cv2
            frame = cv2.imread(str(img_path))
            if frame is None:
                return np.zeros(FEATURE_DIM, dtype=np.float32)
        except ImportError:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        det = self._get_face_detector()
        lms = det.detect(frame)
        if not lms:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        return extract_feature_vector_from_landmarks(lms[0])

    def _iter_samples(self) -> Generator[tuple[np.ndarray, int, object], None, None]:
        rows = self._load_rows()
        if self.max_images:
            rows = rows[: self.max_images]

        # Group into pseudo-sequences of _PSEUDO_SEQUENCE_LEN frames
        for seq_start in range(0, len(rows), _PSEUDO_SEQUENCE_LEN):
            seq = rows[seq_start: seq_start + _PSEUDO_SEQUENCE_LEN]
            if len(seq) < self.window_size:
                continue

            frame_features, frame_labels = [], []
            for img_path, score in seq:
                feats = self._frame_features(img_path)
                frame_features.append(feats)
                frame_labels.append(_score_to_label(score))

            windows, y, groups = self.build_windows(
                frame_features, frame_labels,
                window_size=self.window_size,
                stride=self.stride,
                group_id=seq_start,
            )
            for i in range(len(y)):
                yield windows[i], int(y[i]), groups[i]
