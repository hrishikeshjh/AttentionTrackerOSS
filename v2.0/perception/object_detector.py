"""
ObjectDetector — YOLOv8-s object detection (ARCHITECTURE.md §3 Perception Layer)

Detects study-context objects: book, phone, laptop, pen, keyboard, mouse.
Model must be fine-tuned on COCO (at minimum book/phone/laptop classes).
~4 ms/frame on GPU (ARCHITECTURE.md §4).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO
from server.config import OBJECT_DETECTOR_MODEL, OBJECT_CONF_THRESHOLD

# COCO class IDs relevant to study context (0-indexed)
STUDY_CLASSES = {
    63: "laptop",
    64: "mouse",
    66: "keyboard",
    67: "phone",
    73: "book",
}


@dataclass
class ObjectDetection:
    cls: str                                          # e.g. "book"
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]     # pixel coords
    bbox_norm: tuple[float, float, float, float]     # cx,cy,w,h normalised [0-1]


class ObjectDetector:
    def __init__(self, frame_w: int, frame_h: int):
        self.model = YOLO(str(OBJECT_DETECTOR_MODEL))
        self.model.overrides["verbose"] = False
        self.fw = frame_w
        self.fh = frame_h

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        results = self.model(
            frame,
            classes=list(STUDY_CLASSES.keys()),
            conf=OBJECT_CONF_THRESHOLD,
            verbose=False,
        )
        out: list[ObjectDetection] = []
        for box in results[0].boxes:
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
