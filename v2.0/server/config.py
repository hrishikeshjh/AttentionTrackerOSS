"""
CognEn v2.0 — Server Configuration
Loads settings from environment / .env file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from features.hand_proximity import parse_hand_proximity_source


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is in server/requirements.txt; absent only
        # in the light CI test tier, where no .env file is needed.
        return
    load_dotenv()


_load_env_file()

# room_id flows directly into filesystem paths (load_seat_map below) and
# is used as a dict key elsewhere -- restrict to a safe charset so a
# room_id like "../../etc/passwd" can't escape SEAT_MAPS_DIR.
# Length-capped (64) so it can't be used to bloat logs / dict keys either.
ROOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

MAX_CONCURRENT_ROOMS = int(os.getenv("COGNEN_MAX_CONCURRENT_ROOMS", "20"))

# ── Model paths ──────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / "models"

PERSON_DETECTOR_MODEL  = MODELS_DIR / "yolov8m.pt"
OBJECT_DETECTOR_MODEL  = MODELS_DIR / "yolov8s-objects.pt"
POSE_ESTIMATOR_MODEL   = MODELS_DIR / "yolov8m-pose.pt"
BEHAVIOR_CLASSIFIER_MODEL = MODELS_DIR / "behavior_classifier_v3.onnx"


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

# OpenCV internal buffer size for live sources (1 = always read the newest frame).
# NOT WIRED: nothing reads this; frame_source.py sources take their own
# `buffer_size` argument (default 1), so COGNEN_CAMERA_BUFFER has no effect.
CAMERA_BUFFER_SIZE      = int(os.getenv("COGNEN_CAMERA_BUFFER", "1"))

PERSON_CONF_THRESHOLD  = 0.40   # YOLOv8 person detection confidence
OBJECT_CONF_THRESHOLD  = 0.25   # Object detection confidence
POSE_CONF_THRESHOLD    = 0.40   # Keypoint confidence
SEAT_IOU_THRESHOLD     = 0.30   # IoU to assign a person to a seat
HAND_PROXIMITY_NORM    = 0.20   # wrist-to-object distance threshold, as a fraction of the
                                # person box (v1.0 legacy threshold); used by the pose-wrist
                                # method (features/hand_proximity.py)

# Source of the per-person hand/object proximity label:
#   "pose"  (default) -- YOLOv8-pose wrists + the person's object boxes
#           (features/hand_proximity.py). Thresholds unvalidated on CCTV.
#   "hands" -- MediaPipe HandLandmarker hand boxes, IoU holding / near
#           (features/hand_object.py, DACE Annex A.2), falling back to wrists.
#   "off"   -- explicit opt-out: "hands_not_visible" for everyone (the
#           Context-Gate Override then never sees a holding_* label).
HAND_PROXIMITY_SOURCE  = parse_hand_proximity_source(os.getenv("COGNEN_HAND_PROXIMITY"))

# ── Reference values from v1.0 (NOT USED by v2.0 code) ───────────────────────
# v2.0 has no per-person MLEngine/EMA and no attention-state thresholds: the
# per-person attention_score is a fixed lookup of the behavior label
# (inference/behavior_classifier.py ATTENTION_SCORES) and room_attention is
# their plain mean. The window size actually used is
# behavior_classifier.WINDOW_SIZE. These constants record the v1.0 legacy
# values (ml-engine.ts thresholds, EMA alpha) for a future port; changing
# them has no effect today. v1.0's opt-in DACE mode adds "Unsure"/score-hold
# semantics (ARCHITECTURE.md §11.7) that v2.0 does not implement.

ATTENTIVE_MIN  = 70
DISTRACTED_MIN = 40

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

# Default bind is loopback (secure default). `python -m server.main` refuses
# a non-loopback API_HOST unless COGNEN_API_KEY is set (or
# COGNEN_ALLOW_INSECURE=1); see server/security.py::check_bind_allowed.
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
WS_PATH  = "/ws/room/{room_id}"

# ── Security settings (parsed in server/security.py / input/frame_source.py) ─
# They are read from the environment there (not as constants here) so tests
# and reloads see changes; this block is the reference list.
#
# Authentication
#   COGNEN_API_KEY            comma-separated API keys (>= 16 chars each; several
#                             = rotation). Required on every route except
#                             /healthz and /static: header X-API-Key, or
#                             ?api_key= on the WebSocket. Unset => protected
#                             routes return 503 and non-loopback binds refused.
#   COGNEN_ALLOW_INSECURE     "1" = run WITHOUT auth when no key is set (local
#                             dev only; logs a loud warning). Default off.
#   COGNEN_AUTH_FAIL_RATE_PER_MIN / COGNEN_AUTH_FAIL_BURST
#                             failed-auth bucket per client IP (10/min, burst 10)
#                             -> 429 once exhausted.
# HTTP hardening
#   COGNEN_START_RATE_PER_MIN / COGNEN_START_BURST
#                             /rooms/{id}/start token bucket, applied per key AND
#                             per client IP (default 6/min, burst 3) -> 429.
#   COGNEN_MAX_BODY_BYTES     request body cap (default 16384) -> 413.
#   COGNEN_CORS_ORIGINS       comma-separated exact origins (https://host[:port]);
#                             default none; "*" refused; never credentials.
#   COGNEN_ENABLE_DOCS        "1" = expose /docs, /redoc, /openapi.json (default off).
# WebSocket
#   COGNEN_WS_MAX_PER_KEY (8) / COGNEN_WS_MAX_PER_ROOM (32) / COGNEN_WS_MAX_TOTAL (256)
#   COGNEN_WS_SEND_TIMEOUT_S (2.0)   slower sends drop the client (1008)
#   COGNEN_WS_PUSH_INTERVAL_S (0.2)  latest-snapshot push period
#   COGNEN_WS_HEARTBEAT_S (15)       resend an unchanged snapshot this often
#   COGNEN_WS_MAX_MESSAGE_BYTES (4096), COGNEN_WS_MAX_CLIENT_MSGS_PER_S (5)
#   Origin must be in COGNEN_CORS_ORIGINS when the client sends one.
# Video sources requested via the API (input/frame_source.py)
#   COGNEN_VIDEO_DIRS         os.pathsep-separated roots for file sources
#                             (default v2.0/data/videos); realpath containment.
#   COGNEN_VIDEO_EXTENSIONS   allowed file extensions (default common video types)
#   COGNEN_STREAM_SCHEMES     enabled stream schemes (default rtsp,rtsps; may add
#                             http,https)
#   COGNEN_STREAM_HOSTS       allowlist of hostnames and/or CIDRs. When set, the
#                             stream host must match a name or every resolved IP
#                             must fall in a CIDR. Names may resolve to private
#                             IPs; only a CIDR can admit loopback/link-local/etc.
#   COGNEN_ALLOW_PRIVATE_STREAMS "1" = allow RFC1918/CGNAT/ULA targets without an
#                             allowlist (loopback, link-local/metadata, multicast,
#                             reserved stay denied).
#   COGNEN_STREAM_PORTS       optional comma-separated port allowlist
#   COGNEN_ALLOW_URL_CREDENTIALS "1" = permit rtsp://user:pass@... (redacted in logs)
#   COGNEN_ALLOW_CAMERA_SOURCE   "1" = permit source_type="camera" via the API

# ── Privacy settings ─────────────────────────────────────────────────────────
# NOT ENFORCED: nothing reads these two constants. Output is anonymous only
# because v2.0 has no identity mapping at all (results carry track IDs), and
# there is no persistence layer to purge; no retention job exists.

ANONYMISE_BY_DEFAULT = True     # intended: use track IDs instead of names unless mapped
DATA_RETENTION_HOURS = 24       # intended: auto-purge analytics after N hours
