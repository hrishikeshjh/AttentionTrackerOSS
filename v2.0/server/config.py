"""
CognEn v2.0 — Server Configuration
Loads settings from environment / .env file.
"""

from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

# ── Model paths ──────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / "models"

PERSON_DETECTOR_MODEL  = MODELS_DIR / "yolov8m.pt"
OBJECT_DETECTOR_MODEL  = MODELS_DIR / "yolov8s-objects.pt"
POSE_ESTIMATOR_MODEL   = MODELS_DIR / "yolov8m-pose.pt"
BEHAVIOR_CLASSIFIER_MODEL = MODELS_DIR / "behavior_classifier.onnx"

# ── Inference settings ───────────────────────────────────────────────────────

INPUT_WIDTH    = 1280
INPUT_HEIGHT   = 720
TARGET_FPS     = 15

PERSON_CONF_THRESHOLD  = 0.40   # YOLOv8 person detection confidence
OBJECT_CONF_THRESHOLD  = 0.25   # Object detection confidence
POSE_CONF_THRESHOLD    = 0.40   # Keypoint confidence
SEAT_IOU_THRESHOLD     = 0.30   # IoU to assign a person to a seat
HAND_PROXIMITY_NORM    = 0.20   # Normalised distance threshold (same as v1.0)

# ── Attention thresholds (mirrors v1.0 ml-engine.ts) ────────────────────────

ATTENTIVE_MIN  = 70
DISTRACTED_MIN = 40

# ── Temporal classifier ──────────────────────────────────────────────────────

TEMPORAL_WINDOW_SIZE = 15       # frames (3 s at 5 FPS)
EMA_ALPHA            = 0.15

# ── Seat maps ────────────────────────────────────────────────────────────────

SEAT_MAPS_DIR = Path(__file__).parent.parent / "data" / "seat_maps"

def load_seat_map(room_id: str) -> dict[str, list[tuple[float, float]]]:
    """
    Load seat polygons for a room.
    Format: { "seat_1": [[x1,y1], [x2,y2], ...], ... }
    Coordinates are normalised [0, 1].
    """
    path = SEAT_MAPS_DIR / f"{room_id}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

# ── WebSocket / API ──────────────────────────────────────────────────────────

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
WS_PATH  = "/ws/room/{room_id}"

# ── Privacy settings ─────────────────────────────────────────────────────────

ANONYMISE_BY_DEFAULT = True     # Use track IDs instead of names unless mapped
DATA_RETENTION_HOURS = 24       # Auto-purge analytics after N hours
