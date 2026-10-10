"""
PersonDetector — YOLOv8-m person detection (ARCHITECTURE.md §3 Perception Layer)

Detects all persons in a 1280×720 frame.
Output: list of normalized [cx, cy, w, h] bounding boxes + confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from ultralytics import YOLO

from server.config import PERSON_CONF_THRESHOLD, PERSON_DETECTOR_MODEL, require_model_file

if TYPE_CHECKING:
    from ultralytics.engine.results import Results


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
        self.model = YOLO(str(require_model_file(PERSON_DETECTOR_MODEL)))
        # ultralytics infers Model.overrides as dict[str, str] from its own
        # string-valued assignments; the predictor accepts any cfg value.
        overrides: dict[str, Any] = self.model.overrides
        overrides["verbose"] = False

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """
        @param frame  BGR frame (H×W×3)
        @returns      PersonDetection list, one per detected person
        """
        # Model.__call__ is annotated for stream=True (Iterator) and embed
        # (Tensor) too; a plain call returns list[Results].
        results = cast("list[Results]", self.model(
            frame,
            classes=[0],              # person only
            conf=PERSON_CONF_THRESHOLD,
            verbose=False,
        ))
        detections: list[PersonDetection] = []
        boxes = results[0].boxes
        if boxes is None:   # only unset for non-detection tasks
            return detections
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            detections.append(PersonDetection(bbox_xyxy=(x1, y1, x2, y2), confidence=conf))
        return detections
