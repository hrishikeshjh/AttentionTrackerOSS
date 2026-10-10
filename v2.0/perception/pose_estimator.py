"""
PoseEstimator — YOLOv8-pose, 17 COCO keypoints per person.
Used for posture analysis (ARCHITECTURE.md §5.4).
~6 ms/frame on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from ultralytics import YOLO

from server.config import POSE_CONF_THRESHOLD, POSE_ESTIMATOR_MODEL, require_model_file

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

# COCO keypoint indices
KP = dict(
    nose=0,
    l_shoulder=5, r_shoulder=6,
    l_elbow=7,    r_elbow=8,
    l_wrist=9,    r_wrist=10,
    l_hip=11,     r_hip=12,
)


@dataclass
class Keypoint:
    x: float   # pixel
    y: float   # pixel
    conf: float


@dataclass
class PoseDetection:
    """17 COCO keypoints + derived posture features."""
    keypoints: list[Keypoint] = field(default_factory=list)
    bbox_xyxy: tuple[float, float, float, float] = (0, 0, 0, 0)


@dataclass
class PostureResult:
    is_upright: bool
    lean_angle: float   # degrees from vertical
    is_head_dropped: bool


class PoseEstimator:
    def __init__(self):
        self.model = YOLO(str(require_model_file(POSE_ESTIMATOR_MODEL)))
        # ultralytics infers Model.overrides as dict[str, str] from its own
        # string-valued assignments; the predictor accepts any cfg value.
        overrides: dict[str, Any] = self.model.overrides
        overrides["verbose"] = False

    def detect(self, frame: np.ndarray) -> list[PoseDetection]:
        # Model.__call__ is annotated for stream=True (Iterator) and embed
        # (Tensor) too; a plain call returns list[Results].
        results = cast("list[Results]", self.model(frame, conf=POSE_CONF_THRESHOLD, verbose=False))
        out: list[PoseDetection] = []
        for r in results:
            if r.keypoints is None:
                continue
            for i, kps_tensor in enumerate(r.keypoints.data):
                kps = [
                    Keypoint(x=float(kp[0]), y=float(kp[1]), conf=float(kp[2]))
                    for kp in kps_tensor
                ]
                bbox = tuple(r.boxes.xyxy[i].cpu().numpy()) if r.boxes else (0, 0, 0, 0)
                out.append(PoseDetection(keypoints=kps, bbox_xyxy=bbox))
        return out

    @staticmethod
    def analyze_posture(pose: PoseDetection) -> PostureResult:
        """
        Derive posture features from COCO keypoints.
        Implements ARCHITECTURE.md §5.4 pseudo-code.
        """
        kps = pose.keypoints
        if len(kps) < 13:
            return PostureResult(is_upright=True, lean_angle=0.0, is_head_dropped=False)

        nose      = kps[KP["nose"]]
        l_sh      = kps[KP["l_shoulder"]]
        r_sh      = kps[KP["r_shoulder"]]
        l_hip     = kps[KP["l_hip"]]
        r_hip     = kps[KP["r_hip"]]

        sh_mid  = ((l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2)
        hip_mid = ((l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2)

        # Torso vector (shoulder → hip)
        tv = (sh_mid[0] - hip_mid[0], sh_mid[1] - hip_mid[1])
        length = (tv[0] ** 2 + tv[1] ** 2) ** 0.5 or 1.0
        # Angle from vertical (0, -1)
        import math
        lean_angle = abs(math.degrees(math.acos(
            max(-1.0, min(1.0, -tv[1] / length))
        )))

        # Head drop: nose y-coord below shoulder midpoint = slumped
        is_head_dropped = nose.y > sh_mid[1] and nose.conf > 0.3

        return PostureResult(
            is_upright=lean_angle < 15,
            lean_angle=lean_angle,
            is_head_dropped=is_head_dropped,
        )
