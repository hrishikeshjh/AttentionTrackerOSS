"""
hand_landmarks.py — MediaPipe HandLandmarker wrapper for v2.0 (DACE Annex A.2 port).

Used by inference/room_analyzer.py when COGNEN_HAND_PROXIMITY=hands: runs
the Tasks-API HandLandmarker (IMAGE mode, up to 2 hands) on each tracked
person's clipped box and returns the 21 landmarks per hand in FRAME pixel
coordinates. features/hand_object.py turns them into holding / near / pinch
relations with the person's object boxes; if no hand is found the analyzer
falls back to the YOLOv8-pose wrist method (features/hand_proximity.py).

IMAGE mode for the same reason as perception/face_landmarks.py: crops of
different people go through one detector.

Model: hand_landmarker.task (float16, Apache-2.0) in v2.0/models/,
downloaded on first use like face_landmarker.task. Initialisation never
raises; failures are logged once and exposed via ``status`` / ``init_error``.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from perception.face_landmarks import (
    STATUS_INIT_FAILED,
    STATUS_MEDIAPIPE_MISSING,
    STATUS_MODEL_MISSING,
    STATUS_READY,
    Point2D,
)

logger = logging.getLogger("cognen.hand_landmarks")

_MODELS_DIR = Path(__file__).parent.parent / "models"
_MODEL_FILE = _MODELS_DIR / "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass
class HandObservation:
    landmarks: list[Point2D] = field(default_factory=list)   # 21 points, frame pixels
    handedness: str | None = None                             # "Left" / "Right" (MediaPipe label)
    score: float = 0.0


def _ensure_model() -> Optional[Path]:
    if _MODEL_FILE.exists():
        return _MODEL_FILE
    logger.info("hand_landmarker.task not found — downloading from MediaPipe assets...")
    try:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_FILE)
        return _MODEL_FILE
    except Exception as exc:
        logger.error("Could not download hand_landmarker.task: %s (curl -o %s '%s')", exc, _MODEL_FILE, _MODEL_URL)
        return None


class HandLandmarksDetector:
    def __init__(self, max_hands: int = 2, model_path: Optional[Path] = None,
                 min_confidence: float = 0.35) -> None:
        self._max_hands = max_hands
        self._model_path = model_path
        self._min_conf = min_confidence      # DACE c_min
        self._landmarker = None
        self._available = False
        self.status: str = STATUS_INIT_FAILED
        self.init_error: Optional[str] = None
        self._detect_error_logged = False
        self._init()

    def _fail(self, status: str, reason: str) -> None:
        self.status, self.init_error, self._available = status, reason, False
        logger.error("HandLandmarksDetector DISABLED (%s): %s -- hand proximity falls back to pose wrists.",
                     status, reason)

    def _init(self) -> None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            self._fail(STATUS_MEDIAPIPE_MISSING, f"mediapipe not importable ({exc})")
            return
        model_path = self._model_path or _ensure_model()
        if model_path is None or not Path(model_path).exists():
            self._fail(STATUS_MODEL_MISSING, f"model file unavailable: {model_path or _MODEL_FILE}")
            return
        try:
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=self._max_hands,
                min_hand_detection_confidence=self._min_conf,
                min_hand_presence_confidence=self._min_conf,
                min_tracking_confidence=self._min_conf,
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            self._fail(STATUS_INIT_FAILED, f"{type(exc).__name__}: {exc}")
            return
        self._available, self.status, self.init_error = True, STATUS_READY, None
        logger.info("HandLandmarksDetector: MediaPipe HandLandmarker ready (IMAGE mode)")

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, bgr_image: np.ndarray, offset: tuple[float, float] = (0.0, 0.0)) -> list[HandObservation]:
        if not self._available or self._landmarker is None:
            return []
        if bgr_image is None or bgr_image.size == 0 or bgr_image.ndim != 3:
            return []
        import cv2
        import mediapipe as mp

        h, w = bgr_image.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        try:
            result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        except Exception as exc:
            if not self._detect_error_logged:
                logger.error("HandLandmarker.detect failed (logged once): %s", exc)
                self._detect_error_logged = True
            return []
        return hands_from_result(result, w, h, offset)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            self._available = False


def hands_from_result(result, crop_w: int, crop_h: int, offset: tuple[float, float] = (0.0, 0.0)) -> list[HandObservation]:
    ox, oy = float(offset[0]), float(offset[1])
    hands = list(getattr(result, "hand_landmarks", None) or [])
    handed = list(getattr(result, "handedness", None) or [])
    out = []
    for i, lms in enumerate(hands):
        if len(lms) < 21:
            continue
        cat = handed[i][0] if i < len(handed) and handed[i] else None
        out.append(HandObservation(
            landmarks=[Point2D(x=p.x * crop_w + ox, y=p.y * crop_h + oy) for p in lms],
            handedness=getattr(cat, "category_name", None),
            score=float(getattr(cat, "score", 0.0) or 0.0),
        ))
    return out
