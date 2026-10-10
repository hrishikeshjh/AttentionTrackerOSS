"""
ObjectDetector — YOLOv8-s object detection (ARCHITECTURE.md §3 Perception Layer)

Detects study-context objects: book, phone, laptop, pen, keyboard, mouse.
Model must be fine-tuned on COCO (at minimum book/phone/laptop classes).
~4 ms/frame on GPU (ARCHITECTURE.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from ultralytics import YOLO

from perception.study_objects import COCO_ID_TO_STUDY
from server.config import OBJECT_CONF_THRESHOLD, OBJECT_DETECTOR_MODEL, require_model_file

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

# COCO class IDs relevant to study context (0-indexed). Single source:
# perception/study_objects.py (also used by the Open Images fine-tuning loader).
STUDY_CLASSES = COCO_ID_TO_STUDY


@dataclass
class ObjectDetection:
    cls: str                                          # e.g. "book"
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]     # pixel coords
    bbox_norm: tuple[float, float, float, float]     # cx,cy,w,h normalised [0-1]


class ObjectDetector:
    def __init__(self, frame_w: int, frame_h: int):
        self.model = YOLO(str(require_model_file(OBJECT_DETECTOR_MODEL)))
        # ultralytics infers Model.overrides as dict[str, str] from its own
        # string-valued assignments; the predictor accepts any cfg value.
        overrides: dict[str, Any] = self.model.overrides
        overrides["verbose"] = False
        self.fw = frame_w
        self.fh = frame_h

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        # Model.__call__ is annotated for stream=True (Iterator) and embed
        # (Tensor) too; a plain call returns list[Results].
        results = cast("list[Results]", self.model(
            frame,
            classes=list(STUDY_CLASSES.keys()),
            conf=OBJECT_CONF_THRESHOLD,
            verbose=False,
        ))
        out: list[ObjectDetection] = []
        boxes = results[0].boxes
        if boxes is None:   # only unset for non-detection tasks
            return out
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(box.cls[0])
            if cls_id not in STUDY_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = ((x1 + x2) / 2) / self.fw
            cy = ((y1 + y2) / 2) / self.fh
            w  = (x2 - x1) / self.fw
            h  = (y2 - y1) / self.fh
            out.append(ObjectDetection(
                cls=STUDY_CLASSES[cls_id],
                confidence=float(box.conf[0]),
                bbox_xyxy=(x1, y1, x2, y2),
                bbox_norm=(cx, cy, w, h),
            ))
        return out
