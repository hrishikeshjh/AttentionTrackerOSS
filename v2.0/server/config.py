"""
CognEn v2.0 — Server Configuration
Loads settings from environment / .env file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# room_id flows directly into filesystem paths (load_seat_map below) and
# is used as a dict key elsewhere -- restrict to a safe charset so a
# room_id like "../../etc/passwd" can't escape SEAT_MAPS_DIR.
ROOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

MAX_CONCURRENT_ROOMS = int(os.getenv("COGNEN_MAX_CONCURRENT_ROOMS", "20"))

# ── Model paths ──────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / "models"

PERSON_DETECTOR_MODEL  = MODELS_DIR / "yolov8m.pt"
OBJECT_DETECTOR_MODEL  = MODELS_DIR / "yolov8s-objects.pt"
POSE_ESTIMATOR_MODEL   = MODELS_DIR / "yolov8m-pose.pt"
BEHAVIOR_CLASSIFIER_MODEL = MODELS_DIR / "behavior_classifier.onnx"


def require_model_file(path: Path) -> Path:
    """
    Validate a model file exists before handing it to a heavy loader
    (YOLO/onnxruntime). Without this, a missing .pt/.onnx file surfaces as
    an opaque exception deep inside the third-party library's own loading
    code -- this gives a clear, actionable error pointing at
    download_models.py instead.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Run `python3 v2.0/models/download_models.py` to fetch required model weights."
        )
    return path

# ── Inference settings ───────────────────────────────────────────────────────

INPUT_WIDTH    = 1280
INPUT_HEIGHT   = 720
TARGET_FPS     = 15

# ── Frame source settings ────────────────────────────────────────────────────

# Analyse 1 out of every N frames (higher = less CPU, more latency)
# At 15 FPS source: N=1 → 15 analysis FPS, N=3 → 5 analysis FPS
PROCESS_EVERY_N_FRAMES  = int(os.getenv("COGNEN_PROCESS_EVERY_N", "1"))

# RTSP transport protocol: "tcp" (reliable) or "udp" (lower latency, packet loss risk)
RTSP_TRANSPORT          = os.getenv("COGNEN_RTSP_TRANSPORT", "tcp")

# OpenCV internal buffer size for live sources (1 = always read the newest frame)
CAMERA_BUFFER_SIZE      = int(os.getenv("COGNEN_CAMERA_BUFFER", "1"))

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

    Defense-in-depth: rejects a room_id outside ROOM_ID_PATTERN even though
    the FastAPI layer (server/main.py) should already have validated it --
    this function builds a filesystem path directly from room_id, so it
    must not trust an unvalidated caller.
    """
    if not ROOM_ID_PATTERN.match(room_id):
        raise ValueError(f"invalid room_id: {room_id!r}")
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
