"""
daisee_loader.py — DAiSEE engagement video dataset loader.

DAiSEE (Dataset for Affective States in E-Environments)
  - 9,068 video clips, ~10 sec each, ~5 FPS after decoding
  - 4 engagement labels: 0=very low, 1=low, 2=high, 3=very high
  - Also has boredom, confusion, frustration labels (ignored here)

Download: https://iith.ac.in/~daisee-dataset/

Expected directory layout:
    <root>/
        DataSet/
            Train/
                <student_id>/
                    <video_id>/
                        <clip>.avi
            Test/
                ...
            Validation/
                ...
        Labels/
            TrainLabels.csv
            TestLabels.csv
            ValidationLabels.csv

CSV format: ClipID, Boredom, Engagement, Confusion, Frustration
  (all 0-3)

Engagement label mapping → CognEn behavior classes:
    3 (very high) → attentive_screen   (full attention)
    2 (high)      → attentive_screen   (attending)
    1 (low)       → distracted         (partially distracted)
    0 (very low)  → drowsy             (disengaged / sleeping)

Note: DAiSEE has no object context, so object/hand features are zeroed out.
The loader extracts FACE features only via MediaPipe FaceMesh on each frame.
"""

from __future__ import annotations
import csv
import logging
import os
from pathlib import Path
from typing import Generator

import numpy as np

from .base_dataset import BaseBehaviorDataset, CLASS_TO_IDX, FEATURE_DIM, WINDOW_SIZE

logger = logging.getLogger("cognen.datasets.daisee")

# DAiSEE engagement → behavior class
_ENGAGEMENT_TO_CLASS: dict[int, str] = {
    3: "attentive_screen",
    2: "attentive_screen",
    1: "distracted",
    0: "drowsy",
}

_SPLIT_DIR: dict[str, str] = {
    "train": "Train",
    "val":   "Validation",
    "test":  "Test",
}

_LABEL_FILE: dict[str, str] = {
    "train": "TrainLabels.csv",
    "val":   "ValidationLabels.csv",
    "test":  "TestLabels.csv",
}


class DAiSEEDataset(BaseBehaviorDataset):
    """
    Loads DAiSEE clips, extracts per-frame face features using MediaPipe FaceMesh,
    and yields sliding-window samples.

    Args:
        root:          Path to DAiSEE dataset root.
        split:         "train" | "val" | "test".
        window_size:   Frames per temporal window (default 15).
        stride:        Window stride (default 5 — 1-sec step at 5 FPS).
        max_clips:     Cap on number of clips loaded (None = all; use for quick runs).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        window_size: int = WINDOW_SIZE,
        stride: int = 5,
        max_clips: int | None = None,
    ) -> None:
        super().__init__(root, split)
        self.window_size = window_size
        self.stride      = stride
        self.max_clips   = max_clips

        self._label_path = Path(root) / "Labels" / _LABEL_FILE[split]
        self._video_dir  = Path(root) / "DataSet" / _SPLIT_DIR[split]
        self._clip_labels = self._load_labels()

        # Lazy-import heavy dependencies
        self._face_det = None

    # ── internal ─────────────────────────────────────────────────────────────

    def _load_labels(self) -> dict[str, int]:
        """Return {clip_id: engagement_label} from CSV."""
        labels: dict[str, int] = {}
        if not self._label_path.exists():
            logger.warning("DAiSEE label file not found: %s", self._label_path)
            return labels
        with open(self._label_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip_id    = row["ClipID"].strip()
                engagement = int(row["Engagement"])
                labels[clip_id] = engagement
        return labels

    def _get_face_detector(self):
        if self._face_det is None:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from perception.face_landmarks import FaceLandmarksDetector
            self._face_det = FaceLandmarksDetector(max_faces=1)
        return self._face_det

    def _extract_frame_features(self, frame) -> np.ndarray:
        """
        Extract 13-dim feature vector from a single BGR frame.
        Falls back to zeros if no face is detected.
        """
        det = self._get_face_detector()
        lms = det.detect(frame)
        if not lms:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        pts = lms[0]  # list[Point2D], 68 points

        # EAR — right eye 36-41, left eye 42-47
        def _ear(indices: list[int]) -> float:
            p = [pts[i] for i in indices]
            v1 = np.hypot(p[1].x - p[5].x, p[1].y - p[5].y)
            v2 = np.hypot(p[2].x - p[4].x, p[2].y - p[4].y)
            h  = np.hypot(p[0].x - p[3].x, p[0].y - p[3].y)
            return (v1 + v2) / (2.0 * h + 1e-6)

        ear = (_ear(list(range(36, 42))) + _ear(list(range(42, 48)))) / 2.0

        # Rough gaze from eye centre vs face centre
        eye_cx = (pts[36].x + pts[45].x) / 2.0
        eye_cy = (pts[36].y + pts[45].y) / 2.0
        nose   = pts[30]
        gaze_x = float(np.clip((eye_cx - nose.x) / 50.0, -1, 1)) * 0.5 + 0.5
        gaze_y = float(np.clip((eye_cy - nose.y) / 50.0, -1, 1)) * 0.5 + 0.5

        # Rough head pitch from nose-to-chin angle
        chin  = pts[8]
        dy    = chin.y - nose.y
        pitch_norm = float(np.clip(dy / 100.0, -1, 1))

        return self.encode_feature(
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            pitch_norm=pitch_norm,
            ear=float(np.clip(ear, 0, 1)),
        )

    def _iter_samples(self) -> Generator[tuple[np.ndarray, int], None, None]:
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not installed; cannot decode DAiSEE videos.")
            return

        clips_seen = 0
        for clip_id, engagement in self._clip_labels.items():
            if self.max_clips and clips_seen >= self.max_clips:
                break

            behavior   = _ENGAGEMENT_TO_CLASS[engagement]
            label_idx  = CLASS_TO_IDX[behavior]

            # Find video file (search recursively in split dir)
            matches = list(self._video_dir.rglob(f"{clip_id}*"))
            if not matches:
                continue
            video_path = matches[0]

            cap = cv2.VideoCapture(str(video_path))
            frame_features, frame_labels = [], []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                feats = self._extract_frame_features(frame)
                frame_features.append(feats)
                frame_labels.append(label_idx)

            cap.release()
            clips_seen += 1

            windows, y = self.build_windows(
                frame_features, frame_labels,
                window_size=self.window_size,
                stride=self.stride,
            )
            for i in range(len(y)):
                yield windows[i], int(y[i])
