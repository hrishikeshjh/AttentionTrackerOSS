"""
face_landmarks.py — MediaPipe FaceMesh wrapper for v2.0.

Detects 468 facial landmarks per face and re-maps a 68-point subset that is
compatible with the dlib/ibug convention used by features/face_metrics.py.

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
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Sequence

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


@dataclass
class Point2D:
    x: float
    y: float


class FaceLandmarksDetector:
    """
    Wraps MediaPipe FaceMesh to produce 68 (x, y) pixel landmarks per face.

    Usage
    -----
    detector = FaceLandmarksDetector()
    landmarks = detector.detect(bgr_frame)   # list[list[Point2D]] — one per face
    """

    def __init__(self, max_faces: int = 1) -> None:
        self._max_faces = max_faces
        self._mesh = None
        self._available = False
        self._init()

    def _init(self) -> None:
        try:
            import mediapipe as mp  # type: ignore
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self._max_faces,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._available = True
            logger.info("FaceLandmarksDetector: MediaPipe FaceMesh ready")
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
        Run FaceMesh on *bgr_frame* and return up to *max_faces* sets of
        68 landmarks in pixel coordinates.

        Returns an empty list if MediaPipe is unavailable or no face is found.
        """
        if not self._available or self._mesh is None:
            return []

        import cv2  # type: ignore

        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)

        if not result.multi_face_landmarks:
            return []

        output: list[list[Point2D]] = []
        for face_lms in result.multi_face_landmarks:
            pts = face_lms.landmark  # 468 NormalizedLandmark objects
            landmarks_68: list[Point2D] = [
                Point2D(x=pts[idx].x * w, y=pts[idx].y * h)
                for idx in _MP_TO_68
            ]
            output.append(landmarks_68)

        return output

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None
