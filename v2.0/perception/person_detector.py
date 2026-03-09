"""
PersonDetector — YOLOv8-m person detection (ARCHITECTURE.md §3 Perception Layer)

Detects all persons in a 1280×720 frame.
Output: list of normalized [cx, cy, w, h] bounding boxes + confidence.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO
from server.config import PERSON_DETECTOR_MODEL, PERSON_CONF_THRESHOLD


@dataclass
class PersonDetection:
    bbox_xyxy: tuple[float, float, float, float]  # pixel coords
    confidence: float


class PersonDetector:
    """
    Wraps YOLOv8-m, filters to COCO class 0 (person).
    ~8 ms/frame on GPU (ARCHITECTURE.md §4).
    """

    def __init__(self):
        self.model = YOLO(str(PERSON_DETECTOR_MODEL))
        self.model.overrides["verbose"] = False

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """
        @param frame  BGR frame (H×W×3)
        @returns      PersonDetection list, one per detected person
        """
        results = self.model(
            frame,
            classes=[0],              # person only
            conf=PERSON_CONF_THRESHOLD,
            verbose=False,
        )
        detections: list[PersonDetection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            detections.append(PersonDetection(bbox_xyxy=(x1, y1, x2, y2), confidence=conf))
        return detections
