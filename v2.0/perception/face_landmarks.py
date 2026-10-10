"""
face_landmarks.py — MediaPipe FaceLandmarker wrapper for v2.0.

Detects 478 facial landmarks per face (Tasks API) and re-maps a 68-point
subset onto the dlib/ibug 300-W layout used by features/face_metrics.py.

68-point mapping (_MP_TO_68)
----------------------------
Derived from the canonical MediaPipe face mesh (canonical_face_model.obj in
the mediapipe repo, 468 vertices; x < 0 is the subject's right = image left
for a non-mirrored camera) and the connection tables in
mediapipe.tasks.python.vision.FaceLandmarksConnections (FACE_OVAL, LIPS,
LEFT/RIGHT_EYE, LEFT/RIGHT_EYEBROW) of mediapipe 0.10.31:

  Jaw         0-16  127 234 93 132 58 136 150 176 [152] 400 379 365 288 361 323 454 356
                    FACE_OVAL from ear level (127/356) through the chin (152,
                    the lowest vertex), sampled at equal arc length; mirror
                    pairs are symmetric in the canonical model.
  Right brow 17-21  70 63 105 66 107        (outer -> inner, upper contour)
  Left brow  22-26  336 296 334 293 300     (inner -> outer)
  Nose line  27-30  168 197 195 4           (nasion ... pronasale; 4 is the
                                             most anterior vertex = nose tip)
  Nostrils   31-35  98 97 2 326 327         (2 = subnasale)
  Right eye  36-41  33 160 158 133 153 144  (outer, upper x2, inner, lower x2)
  Left eye   42-47  362 385 387 263 373 380 (inner, upper x2, outer, lower x2)
  Outer lip  48-59  61 39 37 0 267 269 291 321 314 17 84 91
                    (48/54 corners, 51 upper-lip top centre, 57 lower-lip
                    bottom centre)
  Inner lip  60-67  78 81 13 311 308 402 14 178

The previous table was wrong in ways that broke features/face_metrics.py:
the "jaw" was the first 17 FACE_OVAL vertices starting at the forehead
(10), index 30 (the nose tip used by calculate_head_pose) was 195 (mid
bridge), and the outer-lip run was shifted so that 54 was 267 (upper lip,
not the corner) and 57 was 409 (upper lip, not the bottom centre), which
made MAR (48/54/51/57) meaningless.  tests/test_face_landmarks.py checks the
table against the canonical-mesh geometry.

Running mode
------------
IMAGE mode.  RoomAnalyzer feeds crops of *different* tracked people through
one detector; VIDEO mode would carry tracking state (and monotonically
increasing timestamps) from one person's crop to the next.

Head pose / gaze outputs (DACE port)
------------------------------------
The landmarker is created with ``output_facial_transformation_matrixes=True``
and ``output_face_blendshapes=True``. ``detect_full`` returns, per face, a
features.dace_measurement.FaceObservation with the 68 points, the full
478-point mesh (incl. iris 468-477) in frame pixels, the 4x4 transformation
matrix (numpy, row/column order, translation in column 3) and the
blendshape scores. features/dace_measurement.py turns these into pitch in
degrees (+down, relative to a per-track reference), blendshape gaze with an
iris cross-check and gaze-corrected eye openness. ``detect`` keeps the old
68-point-only return type.

Coordinates
-----------
``detect(img, offset=(ox, oy))`` returns pixel coordinates in the frame the
crop was cut from: landmark * crop size + offset.

Failure reporting
-----------------
Initialisation never raises (the server keeps running without face
metrics), but a failure is logged ONCE at ERROR level with the real reason
and exposed through ``status`` / ``init_error`` / ``available``.  Before this
fix the options object was built with a non-existent keyword
(``min_face_presence_score``; the real option is
``min_face_presence_confidence``), so FaceLandmarker never initialised and
face metrics were silently always off.

Model
-----
Requires face_landmarker.task in v2.0/models/ (downloaded on first use,
or run models/download_models.py).
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("cognen.face_landmarks")

# MediaPipe → ibug 68-point mapping (0-indexed MediaPipe mesh indices).
_MP_TO_68: list[int] = [
    # Jaw line (0-16), subject's right ear -> chin (8) -> left ear
    127, 234, 93, 132, 58, 136, 150, 176, 152, 400, 379, 365, 288, 361, 323, 454, 356,
    # Right eyebrow (17-21), outer -> inner
    70, 63, 105, 66, 107,
    # Left eyebrow (22-26), inner -> outer
    336, 296, 334, 293, 300,
    # Nose bridge (27-30); 30 = nose tip
    168, 197, 195, 4,
    # Lower nose / nostrils (31-35); 33 = subnasale
    98, 97, 2, 326, 327,
    # Right eye (36-41): outer corner, upper lid x2, inner corner, lower lid x2
    33, 160, 158, 133, 153, 144,
    # Left eye (42-47): inner corner, upper lid x2, outer corner, lower lid x2
    362, 385, 387, 263, 373, 380,
    # Outer lip (48-59): 48 right corner, 51 top centre, 54 left corner, 57 bottom centre
    61, 39, 37, 0, 267, 269, 291, 321, 314, 17, 84, 91,
    # Inner lip (60-67): 60 right corner, 62 top centre, 64 left corner, 66 bottom centre
    78, 81, 13, 311, 308, 402, 14, 178,
]

assert len(_MP_TO_68) == 68, f"Landmark mapping has {len(_MP_TO_68)} entries, expected 68"
assert len(set(_MP_TO_68)) == 68, "Landmark mapping has duplicate MediaPipe indices"

# face_landmarker.task lives next to download_models.py
_MODELS_DIR = Path(__file__).parent.parent / "models"
_MODEL_FILE = _MODELS_DIR / "face_landmarker.task"
# Official MediaPipe model bundle (float16)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Values for FaceLandmarksDetector.status
STATUS_READY = "ready"
STATUS_MEDIAPIPE_MISSING = "mediapipe_missing"
STATUS_MODEL_MISSING = "model_missing"
STATUS_INIT_FAILED = "init_failed"


def _ensure_model() -> Optional[Path]:
    """Return path to face_landmarker.task, downloading if necessary."""
    if _MODEL_FILE.exists():
        return _MODEL_FILE
    logger.info(
        "face_landmarker.task not found — downloading from MediaPipe assets..."
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
    Wraps MediaPipe FaceLandmarker (Tasks API, IMAGE mode) to produce 68
    (x, y) pixel landmarks per face.

    Usage
    -----
    detector = FaceLandmarksDetector()
    if not detector.available:
        print(detector.status, detector.init_error)
    faces = detector.detect(bgr_crop, offset=(x1, y1))   # list[list[Point2D]]
    """

    def __init__(self, max_faces: int = 1, model_path: Optional[Path] = None) -> None:
        self._max_faces = max_faces
        self._model_path = model_path
        self._landmarker = None
        self._available = False
        self.status: str = STATUS_INIT_FAILED
        self.init_error: Optional[str] = None
        self._detect_error_logged = False
        self._init()

    def _fail(self, status: str, reason: str) -> None:
        self.status = status
        self.init_error = reason
        self._available = False
        logger.error(
            "FaceLandmarksDetector DISABLED (%s): %s -- face metrics (EAR, MAR, "
            "head pose, gaze, PERCLOS, blink rate) will be unavailable for every person.",
            status, reason,
        )

    def _init(self) -> None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            self._fail(
                STATUS_MEDIAPIPE_MISSING,
                f"mediapipe not importable ({exc}); install the pinned version "
                "from server/requirements.txt",
            )
            return

        model_path = self._model_path or _ensure_model()
        if model_path is None or not Path(model_path).exists():
            self._fail(STATUS_MODEL_MISSING, f"model file unavailable: {model_path or _MODEL_FILE}")
            return

        try:
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=self._max_faces,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            self._fail(STATUS_INIT_FAILED, f"{type(exc).__name__}: {exc}")
            return

        self._available = True
        self.status = STATUS_READY
        self.init_error = None
        logger.info("FaceLandmarksDetector: MediaPipe FaceLandmarker ready (IMAGE mode)")

    @property
    def available(self) -> bool:
        return self._available

    def detect(
        self,
        bgr_image: np.ndarray,
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> list[list[Point2D]]:
        """
        Run FaceLandmarker on *bgr_image* (a full frame or a crop of one) and
        return up to *max_faces* sets of 68 landmarks in pixel coordinates of
        the ORIGINAL frame: ``x = lm.x * crop_w + offset[0]`` (same for y).

        Returns an empty list if MediaPipe is unavailable, the image is
        empty, or no face is found.
        """
        return [obs.landmarks68 for obs in self.detect_full(bgr_image, offset)]

    def detect_full(
        self,
        bgr_image: np.ndarray,
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> list:
        """
        Like ``detect`` but returns features.dace_measurement.FaceObservation
        objects (68 points, 478-point mesh, transformation matrix,
        blendshapes; matrix/blendshapes None if MediaPipe omitted them).
        """
        if not self._available or self._landmarker is None:
            return []
        if bgr_image is None or bgr_image.size == 0 or bgr_image.ndim != 3:
            return []

        import cv2
        import mediapipe as mp

        h, w = bgr_image.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self._landmarker.detect(mp_image)
        except Exception as exc:
            if not self._detect_error_logged:
                logger.error("FaceLandmarker.detect failed (logged once): %s", exc)
                self._detect_error_logged = True
            return []

        return observations_from_result(result, w, h, offset)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            self._available = False


def observations_from_result(result, crop_w: int, crop_h: int, offset: tuple[float, float] = (0.0, 0.0)) -> list:
    """FaceLandmarkerResult -> list[FaceObservation] in frame pixel coordinates."""
    from features.dace_measurement import FaceObservation

    ox, oy = float(offset[0]), float(offset[1])
    faces = list(getattr(result, "face_landmarks", None) or [])
    mats = list(getattr(result, "facial_transformation_matrixes", None) or [])
    blends = list(getattr(result, "face_blendshapes", None) or [])
    out = []
    for i, face in enumerate(faces):
        if len(face) < 468:
            continue
        mesh = [Point2D(x=p.x * crop_w + ox, y=p.y * crop_h + oy) for p in face]
        lm68 = [mesh[idx] for idx in _MP_TO_68]
        matrix = np.asarray(mats[i], dtype=np.float64) if i < len(mats) and mats[i] is not None else None
        bs = None
        if i < len(blends) and blends[i]:
            bs = {c.category_name: float(c.score) for c in blends[i]}
        out.append(FaceObservation(landmarks68=lm68, mesh=mesh, matrix=matrix, blendshapes=bs))
    return out


def landmarks_to_68(
    faces,
    crop_w: int,
    crop_h: int,
    offset: tuple[float, float] = (0.0, 0.0),
) -> list[list[Point2D]]:
    """
    Convert MediaPipe normalised landmark lists (one per face, each with
    .x/.y in [0, 1] relative to the crop) to 68-point ibug landmarks in
    frame pixel coordinates.  Faces with fewer than 468 points are skipped.
    """
    ox, oy = float(offset[0]), float(offset[1])
    out: list[list[Point2D]] = []
    for face in faces or []:
        if len(face) < 468:
            continue
        out.append([
            Point2D(x=face[idx].x * crop_w + ox, y=face[idx].y * crop_h + oy)
            for idx in _MP_TO_68
        ])
    return out
