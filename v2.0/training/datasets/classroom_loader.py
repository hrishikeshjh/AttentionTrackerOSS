"""
classroom_loader.py — Custom classroom CCTV dataset loader.

Expected dataset layout:
    <root>/
        videos/
            <clip_id>.mp4          (or .avi)
        annotations/
            <clip_id>.json         per-clip annotation file
        metadata.csv               index of all clips

Annotation JSON schema per clip:
    {
      "clip_id": "clip_0001",
      "fps": 5,
      "persons": [
        {
          "track_id": 1,
          "seat_id": "seat_3",
          "frames": [
            {
              "frame_idx": 0,
              "bbox_xyxy": [x1, y1, x2, y2],
              "behavior": "attentive_screen"
            },
            ...
          ]
        }
      ]
    }

This loader extracts per-person face crops → face features → temporal windows
using the same FaceLandmarksDetector pipeline as inference.

metadata.csv format: clip_id, split (train|val|test), duration_sec, num_persons
"""

from __future__ import annotations
import csv
import json
import logging
from pathlib import Path
from typing import Generator

import numpy as np

from .base_dataset import (
    BaseBehaviorDataset, CLASS_TO_IDX, BEHAVIOR_CLASSES, FEATURE_DIM, WINDOW_SIZE
)

logger = logging.getLogger("cognen.datasets.classroom")

_PROXIMITY_CODE: dict[str, float] = {
    "hands_not_visible": 0.0,
    "hands_on_desk":     0.25,
    "holding_pen":       0.50,
    "holding_book":      0.75,
    "holding_phone":     1.00,
}


class ClassroomDataset(BaseBehaviorDataset):
    """
    Loads annotated classroom CCTV clips, extracts per-person temporal feature
    windows, and yields (window, label) samples.

    Args:
        root:          Dataset root directory.
        split:         "train" | "val" | "test".
        window_size:   Frames per window.
        stride:        Window stride.
        max_clips:     Cap on number of clips (None = all).
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
        self._meta_path  = Path(root) / "metadata.csv"
        self._face_det   = None

    # ── internal ─────────────────────────────────────────────────────────────

    def _load_metadata(self) -> list[str]:
        clip_ids: list[str] = []
        if not self._meta_path.exists():
            logger.warning("Classroom metadata.csv not found: %s", self._meta_path)
            return clip_ids
        with open(self._meta_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", "").strip() == self.split:
                    clip_ids.append(row["clip_id"].strip())
        return clip_ids

    def _get_face_detector(self):
        if self._face_det is None:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from perception.face_landmarks import FaceLandmarksDetector
            self._face_det = FaceLandmarksDetector(max_faces=1)
        return self._face_det

    def _extract_features_from_crop(
        self,
        face_crop,
        hand_proximity: str,
        detected_objects: list[str],
    ) -> np.ndarray:
        det = self._get_face_detector()
        lms = det.detect(face_crop)

        if not lms:
            return self.encode_feature(
                has_none=1.0,
                proximity_code=_PROXIMITY_CODE.get(hand_proximity, 0.0),
            )

        pts   = lms[0]
        nose  = pts[30]
        chin  = pts[8]
        pitch_norm = float(np.clip((chin.y - nose.y) / 100.0, -1, 1))

        def _ear(indices):
            p  = [pts[i] for i in indices]
            v1 = np.hypot(p[1].x - p[5].x, p[1].y - p[5].y)
            v2 = np.hypot(p[2].x - p[4].x, p[2].y - p[4].y)
            h  = np.hypot(p[0].x - p[3].x, p[0].y - p[3].y)
            return (v1 + v2) / (2.0 * h + 1e-6)

        ear = (_ear(range(36, 42)) + _ear(range(42, 48))) / 2.0

        has_book   = float("book"   in detected_objects or "notebook" in detected_objects)
        has_phone  = float("phone"  in detected_objects)
        has_laptop = float("laptop" in detected_objects)
        has_none   = float(not (has_book or has_phone or has_laptop))

        return self.encode_feature(
            pitch_norm=pitch_norm,
            ear=float(np.clip(ear, 0, 1)),
            has_book=has_book,
            has_phone=has_phone,
            has_laptop=has_laptop,
            has_none=has_none,
            proximity_code=_PROXIMITY_CODE.get(hand_proximity, 0.0),
        )

    def _iter_samples(self) -> Generator[tuple[np.ndarray, int], None, None]:
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not installed; cannot decode classroom videos.")
            return

        root    = Path(self.root)
        clip_ids = self._load_metadata()
        if self.max_clips:
            clip_ids = clip_ids[: self.max_clips]

        for clip_id in clip_ids:
            ann_path   = root / "annotations" / f"{clip_id}.json"
            video_path = next(
                (p for ext in (".mp4", ".avi", ".mkv")
                 if (p := root / "videos" / f"{clip_id}{ext}").exists()),
                None,
            )
            if not ann_path.exists() or video_path is None:
                logger.debug("Skipping clip %s — missing video or annotation", clip_id)
                continue

            with open(ann_path) as f:
                ann = json.load(f)

            # Decode all frames once
            cap    = cv2.VideoCapture(str(video_path))
            frames = {}
            fi     = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames[fi] = frame
                fi += 1
            cap.release()

            # Per-person windows
            for person in ann.get("persons", []):
                frame_features, frame_labels = [], []

                for fdata in person.get("frames", []):
                    fidx     = fdata["frame_idx"]
                    behavior = fdata.get("behavior", "distracted")
                    if behavior not in CLASS_TO_IDX:
                        continue
                    if fidx not in frames:
                        continue

                    frame = frames[fidx]
                    x1, y1, x2, y2 = (int(v) for v in fdata.get("bbox_xyxy", [0, 0, 0, 0]))
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    feats = self._extract_features_from_crop(
                        face_crop,
                        fdata.get("hand_proximity", "hands_not_visible"),
                        fdata.get("detected_objects", []),
                    )
                    frame_features.append(feats)
                    frame_labels.append(CLASS_TO_IDX[behavior])

                windows, y = self.build_windows(
                    frame_features, frame_labels,
                    window_size=self.window_size,
                    stride=self.stride,
                )
                for i in range(len(y)):
                    yield windows[i], int(y[i])
