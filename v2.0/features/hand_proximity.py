"""
hand_proximity.py — hand/object proximity label from pose wrists (opt-in).

Why: RoomAnalyzer used to pass a hardcoded "hands_not_visible" to the
behavior classifier and the CMCV, so the Context-Gate Override
(behavior_classifier._context_gate, which needs "holding_phone" /
"holding_book") could never fire on the server. v2.0 has no hand-landmark
model, but it already runs YOLOv8-pose, whose COCO keypoints include both
wrists. This module derives the proximity label from those wrists and the
object boxes already associated with the person -- no new model.

It is a port of v1.0's legacy HandObjectProximityDetector
(v1.0/src/lib/scoring/aggregator.ts), with the hand centre replaced by the
pose wrist:
  - no visible wrist                  -> "hands_not_visible"
  - priority phone, then book/notebook: any visible wrist within
    `max_dist` of the object's box centre -> "holding_phone" / "holding_book"
  - otherwise (wrists visible)        -> "hands_on_desk"
  - "holding_pen" is never produced (COCO has no pen class).

Distance normalisation: v1.0 measures distance in frame-normalised
coordinates of a single-person webcam frame. The v2.0 analogue of that
frame is the person's bounding box, so offsets are divided by the person
box width (x) and height (y) before comparing with `max_dist` (default
0.20, server/config.py HAND_PROXIMITY_NORM, the v1.0 threshold).

This is NOT DACE's hand-object relation (holding = IoU >= 0.1 between a
hand-landmark box and the object box, ARCHITECTURE.md §11.5); that one is
features/hand_object.py (COGNEN_HAND_PROXIMITY=hands, MediaPipe
HandLandmarker, falling back to this module). A wrist point is a weaker cue
and the threshold is unvalidated on CCTV footage, but it is ON by default
(COGNEN_HAND_PROXIMITY=pose) because with "off" the context gate could never
fire; "off" remains an explicit opt-out.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

logger = logging.getLogger("cognen.hand_proximity")

# COCO-17 keypoint indices (YOLOv8-pose)
COCO_L_WRIST = 9
COCO_R_WRIST = 10

DEFAULT_MAX_DIST = 0.20      # v1.0 HandObjectProximityDetector.PROXIMITY_THRESHOLD
DEFAULT_MIN_KP_CONF = 0.30   # same visibility cut as PoseEstimator.analyze_posture (nose)

# (object class, label) in v1.0 priority order. "pen" is omitted: COCO has no pen.
_PRIORITY: tuple[tuple[str, str], ...] = (
    ("phone", "holding_phone"),
    ("book", "holding_book"),
    ("notebook", "holding_book"),
)

HAND_PROXIMITY_SOURCES = ("off", "pose", "hands")
DEFAULT_HAND_PROXIMITY_SOURCE = "pose"


def parse_hand_proximity_source(value: str | None = None) -> str:
    """Validate COGNEN_HAND_PROXIMITY. Unset/empty or unknown -> "pose" (default)."""
    v = (value or DEFAULT_HAND_PROXIMITY_SOURCE).strip().lower()
    if v not in HAND_PROXIMITY_SOURCES:
        logger.warning("COGNEN_HAND_PROXIMITY=%r is not one of %s; using %r",
                       value, HAND_PROXIMITY_SOURCES, DEFAULT_HAND_PROXIMITY_SOURCE)
        return DEFAULT_HAND_PROXIMITY_SOURCE
    return v


def _kp(kp: Any) -> tuple[float, float, float]:
    """Accept a Keypoint-like object (x, y, conf) or an (x, y, conf) sequence."""
    if hasattr(kp, "x"):
        return float(kp.x), float(kp.y), float(getattr(kp, "conf", 1.0))
    x, y, c = kp[0], kp[1], (kp[2] if len(kp) > 2 else 1.0)
    return float(x), float(y), float(c)


def visible_wrists(keypoints: Sequence[Any] | None, min_conf: float = DEFAULT_MIN_KP_CONF) -> list[tuple[float, float]]:
    if not keypoints or len(keypoints) <= COCO_R_WRIST:
        return []
    out = []
    for idx in (COCO_L_WRIST, COCO_R_WRIST):
        x, y, c = _kp(keypoints[idx])
        if c >= min_conf:
            out.append((x, y))
    return out


def estimate_hand_proximity(
    keypoints: Sequence[Any] | None,
    person_bbox_xyxy: Sequence[float],
    objects: Sequence[Any],
    *,
    max_dist: float = DEFAULT_MAX_DIST,
    min_conf: float = DEFAULT_MIN_KP_CONF,
) -> str:
    """
    Args:
        keypoints:        17 COCO keypoints (pixel x, y, conf) of the person's pose.
        person_bbox_xyxy: the person's tracked box (pixels).
        objects:          detections with `.cls` and `.bbox_xyxy` (pixels), already
                          associated with this person.
    Returns one of the HandObjectProximity labels used by behavior_classifier.
    """
    wrists = visible_wrists(keypoints, min_conf)
    if not wrists:
        return "hands_not_visible"

    px1, py1, px2, py2 = (float(v) for v in person_bbox_xyxy)
    w, h = px2 - px1, py2 - py1
    if w <= 0 or h <= 0:
        return "hands_not_visible"

    for cls, label in _PRIORITY:
        for o in objects:
            if getattr(o, "cls", None) != cls:
                continue
            ox1, oy1, ox2, oy2 = (float(v) for v in o.bbox_xyxy)
            ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            for wx, wy in wrists:
                if math.hypot((wx - ocx) / w, (wy - ocy) / h) < max_dist:
                    return label

    return "hands_on_desk"
