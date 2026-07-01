"""
face_landmarks.py — MediaPipe FaceLandmarker wrapper for v2.0.

Detects 478 facial landmarks per face (Tasks API) and re-maps a 68-point
subset that is compatible with the dlib/ibug convention used by
features/face_metrics.py.

Reference mapping (MediaPipe → 68-point ibug landmark indices):
  Jaw:         0-16   → MP indices  10, 338, 297, 332, 284, 251, 389, 356,
                                    454, 323, 361, 288, 397, 365, 379, 378, 400
                        (17 points, slightly asymmetric in MP)
  Right brow:  17-21  → MP 70, 63, 105, 66, 107
  Left brow:   22-26  → MP 336, 296, 334, 293, 300
  Nose bridge: 27-30  → MP 168, 6, 197, 195
  Nose tip:    31-35  → MP 4, 240, 97, 2, 326
  Right eye:   36-41  → MP 33, 160, 158, 133, 153, 144
  Left eye:    42-47  → MP 362, 385, 387, 263, 373, 380
  Outer mouth: 48-59  → MP 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375
  Inner mouth: 60-67  → MP 78, 82, 13, 312, 308, 317, 14, 87

The mapping is approximate — face_metrics.py accesses specific point ranges
(e.g. landmarks[36:42] for right eye) and only needs plausible geometry, not
pixel-perfect dlib equivalence.

Model
-----
Requires face_landmarker.task in the same directory as download_models.py
(v2.0/models/).  Run download_models.py to fetch it automatically, or place
the file there manually.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("cognen.face_landmarks")

# MediaPipe → ibug 68-point mapping (0-indexed MediaPipe mesh indices)
_MP_TO_68: list[int] = [
    # Jaw line (0-16)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
    # Right eyebrow (17-21)
    70, 63, 105, 66, 107,
    # Left eyebrow (22-26)
    336, 296, 334, 293, 300,
    # Nose bridge (27-30)
    168, 6, 197, 195,
    # Nose tip (31-35)
    4, 240, 97, 2, 326,
    # Right eye (36-41)
    33, 160, 158, 133, 153, 144,
    # Left eye (42-47)
    362, 385, 387, 263, 373, 380,
    # Outer mouth (48-59)
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375,
    # Inner mouth (60-67)
    78, 82, 13, 312, 308, 317, 14, 87,
]

assert len(_MP_TO_68) == 68, f"Landmark mapping has {len(_MP_TO_68)} entries, expected 68"

# face_landmarker.task lives next to download_models.py
_MODELS_DIR = Path(__file__).parent.parent / "models"
_MODEL_FILE = _MODELS_DIR / "face_landmarker.task"
# Official MediaPipe model bundle (float16, ~1 MB)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model() -> Optional[Path]:
    """Return path to face_landmarker.task, downloading if necessary."""
    if _MODEL_FILE.exists():
        return _MODEL_FILE
    logger.info(
        "face_landmarker.task not found — downloading from MediaPipe assets (~1 MB)..."
    )
    try:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_FILE)
        logger.info("Downloaded face_landmarker.task to %s", _MODEL_FILE)
        return _MODEL_FILE
    except Exception as exc:
        logger.error("Could not download face_landmarker.task: %s", exc)
        logger.error(
            "Download manually:\n  curl -o %s '%s'", _MODEL_FILE, _MODEL_URL
        )
        return None


@dataclass
class Point2D:
    x: float
    y: float


class FaceLandmarksDetector:
    """
    Wraps MediaPipe FaceLandmarker (Tasks API) to produce 68 (x, y) pixel
    landmarks per face.

    Usage
    -----
    detector = FaceLandmarksDetector()
    landmarks = detector.detect(bgr_frame)   # list[list[Point2D]] — one per face
    """

    def __init__(self, max_faces: int = 1) -> None:
        self._max_faces = max_faces
        self._landmarker = None
        self._available = False
        # Monotonically increasing timestamp (ms) required by VIDEO running mode
        self._timestamp_ms: int = 0
        self._init()

    def _init(self) -> None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _ensure_model()
            if model_path is None:
                logger.warning(
                    "FaceLandmarksDetector: model file unavailable — "
                    "face metrics will be disabled."
                )
                return

            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(model_path)
                ),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=self._max_faces,
                min_face_detection_confidence=0.5,
                min_face_presence_score=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._available = True
            logger.info("FaceLandmarksDetector: MediaPipe FaceLandmarker ready")
        except ImportError:
            logger.warning(
                "FaceLandmarksDetector: mediapipe not installed — "
                "face metrics will be unavailable. "
                "Install with: pip install mediapipe"
            )
        except Exception as exc:
            logger.error("FaceLandmarksDetector: init failed — %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def detect(
        self, bgr_frame: np.ndarray
    ) -> list[list[Point2D]]:
        """
        Run FaceLandmarker on *bgr_frame* and return up to *max_faces* sets of
        68 landmarks in pixel coordinates.

        Returns an empty list if MediaPipe is unavailable or no face is found.
        """
        if not self._available or self._landmarker is None:
            return []

        import cv2
        import mediapipe as mp

        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Tasks API requires an mp.Image and a monotonically increasing timestamp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33  # ~30 FPS tick; any positive increment works

        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.face_landmarks:
            return []

        output: list[list[Point2D]] = []
        for face_lms in result.face_landmarks:
            # face_lms is a list of NormalizedLandmark with .x, .y (0-1)
            landmarks_68: list[Point2D] = [
                Point2D(x=face_lms[idx].x * w, y=face_lms[idx].y * h)
                for idx in _MP_TO_68
            ]
            output.append(landmarks_68)

        return output

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
