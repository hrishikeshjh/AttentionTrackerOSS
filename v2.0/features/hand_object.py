"""
hand_object.py -- hand/object relations from hand landmarks (DACE Annex A.2).

Port of v1.0/src/lib/scoring/dace/hand-object.ts for the v2.0 server:

    holding(h, b) = 1[IoU(box(h), b) >= 0.1]
    near(h, b)    = 1[||centre(h) - centre(b)|| < 0.5 * diag(b)]
    pinch_t       = 1[||thumb_tip - index_tip|| / ||wrist - middle_MCP|| < 0.35]

All coordinates are FRAME PIXELS (isotropic), so v1.0's frame-aspect
correction is not needed. Boxes are (x1, y1, x2, y2). MediaPipe hand
landmark indices: 0 wrist, 4 thumb tip, 8 index tip, 9 middle-finger MCP.

``proximity_from_hands`` turns per-person hands + object boxes into the
HandObjectProximity label consumed by the behavior classifier (v2.0 codes in
features/feature_vector.py):
    any hand holding a phone          -> "holding_phone"   (DACE P_t uses holding only)
    any hand holding or near a book   -> "holding_book"    (DACE book evidence: held or near)
    any hand in a pen grip (pinch)    -> "holding_pen"
    hands present otherwise           -> "hands_on_desk"
    no hands                          -> None (caller falls back to pose wrists)
Single-frame pinch is a weak pen cue (DACE writing also needs wrist motion
over the window); it only sets the 0.5 proximity code, never a label.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

HOLD_IOU = 0.1           # prose constant: holding = IoU >= 0.1
NEAR_DIAG_FACTOR = 0.5   # prose constant: near = centre distance < 0.5 x object diagonal
PI_MAX = 0.35            # Table A.6 pinch ratio (strict <)

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9

Box = tuple[float, float, float, float]


def _xy(p: Any) -> tuple[float, float]:
    if hasattr(p, "x"):
        return float(p.x), float(p.y)
    return float(p[0]), float(p[1])


def hand_box(landmarks: Sequence[Any]) -> Box | None:
    """Axis-aligned box around all hand landmarks."""
    if not landmarks:
        return None
    xs, ys = zip(*(_xy(p) for p in landmarks))
    return min(xs), min(ys), max(xs), max(ys)


def iou(a: Box, b: Box) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _centre(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def holding(hand_landmarks: Sequence[Any], obj_box: Box, hold_iou: float = HOLD_IOU) -> bool:
    hb = hand_box(hand_landmarks)
    return hb is not None and iou(hb, obj_box) >= hold_iou


def near(hand_landmarks: Sequence[Any], obj_box: Box, factor: float = NEAR_DIAG_FACTOR) -> bool:
    hb = hand_box(hand_landmarks)
    if hb is None:
        return False
    (hx, hy), (ox, oy) = _centre(hb), _centre(obj_box)
    diag = math.hypot(obj_box[2] - obj_box[0], obj_box[3] - obj_box[1])
    return math.hypot(hx - ox, hy - oy) < factor * diag


def hand_length(hand_landmarks: Sequence[Any]) -> float | None:
    if len(hand_landmarks) <= MIDDLE_MCP:
        return None
    (wx, wy), (mx, my) = _xy(hand_landmarks[WRIST]), _xy(hand_landmarks[MIDDLE_MCP])
    d = math.hypot(wx - mx, wy - my)
    return d if d > 0 else None


def pinch_ratio(hand_landmarks: Sequence[Any]) -> float | None:
    length = hand_length(hand_landmarks)
    if length is None or len(hand_landmarks) <= INDEX_TIP:
        return None
    (tx, ty), (ix, iy) = _xy(hand_landmarks[THUMB_TIP]), _xy(hand_landmarks[INDEX_TIP])
    return math.hypot(tx - ix, ty - iy) / length


def is_pinch(hand_landmarks: Sequence[Any], pi_max: float = PI_MAX) -> bool:
    r = pinch_ratio(hand_landmarks)
    return r is not None and r < pi_max


def _obj_box(o: Any) -> Box:
    x1, y1, x2, y2 = (float(v) for v in o.bbox_xyxy)
    return x1, y1, x2, y2


def proximity_from_hands(hands: Sequence[Sequence[Any]], objects: Sequence[Any]) -> str | None:
    """
    hands:   per hand, its 21 landmarks in frame pixels.
    objects: detections with ``.cls`` and ``.bbox_xyxy`` (pixels), already
             associated with this person.
    Returns a HandObjectProximity label, or None when no hand is present.
    """
    hands = [h for h in hands if h]
    if not hands:
        return None
    phones = [_obj_box(o) for o in objects if getattr(o, "cls", None) == "phone"]
    books = [_obj_box(o) for o in objects if getattr(o, "cls", None) in ("book", "notebook")]
    if any(holding(h, b) for h in hands for b in phones):
        return "holding_phone"
    if any(holding(h, b) or near(h, b) for h in hands for b in books):
        return "holding_book"
    if any(is_pinch(h) for h in hands):
        return "holding_pen"
    return "hands_on_desk"
