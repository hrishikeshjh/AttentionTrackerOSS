# CognEn v2.0 — Multi-Person CCTV Room Analyzer

> **Server-based, multi-person cognitive engagement analysis for classrooms**
> For the single-user browser version, see **[v1.0 README](../v1.0/README.md)**.

> **Status (honest).** v2.0 is implemented and unit-tested (light tier: 86 pass / 6 skip) but has never been run on real classroom/CCTV footage or trained on real data; every accuracy/latency figure here is a target until measured on dataset D1 / pilot set P1. **DACE (Downward-Attention Context Engine, [ARCHITECTURE.md §11](../ARCHITECTURE.md)) is v1.0-only**: v2.0 deliberately keeps the legacy head-pitch formula and has no "Unsure"/score-hold. v2.0 changes made alongside DACE: opt-in `COGNEN_HAND_PROXIMITY=pose` (hand/object proximity from YOLOv8-pose wrists, `features/hand_proximity.py`); cross-language contract tests (`tests/test_cross_language_contract.py`, generated fixture `shared/fixtures/encode13_cross_language.json`, TS to Python one-way); pitch characterisation test (`tests/test_pitch_characterisation.py`); unused `config.py` constants labelled as such.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Perception Layer](#3-perception-layer)
4. [Tracking Layer](#4-tracking-layer)
5. [Feature Extraction](#5-feature-extraction)
6. [Behavior Classifier (1D-CNN)](#6-behavior-classifier-1d-cnn)
7. [Training Pipeline](#7-training-pipeline)
8. [Context-Gate Override](#8-context-gate-override)
9. [Room Analytics](#9-room-analytics)
10. [Video Input Sources](#10-video-input-sources)
11. [FastAPI Server & WebSocket](#11-fastapi-server--websocket)
12. [Tech Stack](#12-tech-stack)
13. [Repository Structure](#13-repository-structure)
14. [Setup & Installation](#14-setup--installation)
15. [CLI Usage](#15-cli-usage)
16. [REST API Reference](#16-rest-api-reference)
17. [Testing & Evaluation](#17-testing--evaluation)
18. [Performance Metrics](#18-performance-metrics)
19. [Datasets](#19-datasets)
20. [Privacy](#20-privacy)
21. [Cross-Reference: v1.0](#21-cross-reference-v10)
22. [Cross-Modal Contradiction Vector (Novel)](#22-cross-modal-contradiction-vector-novel)
23. [Training Pipeline Correctness Fixes](#23-training-pipeline-correctness-fixes)
24. [Server Hardening](#24-server-hardening)
25. [Code Quality, Testing & CI](#25-code-quality-testing--ci)

---

## 1. Overview

CognEn v2.0 scales the [v1.0 single-user pipeline](../v1.0/README.md) to **multi-person CCTV room analysis**. It processes RTSP streams, video files, or USB cameras at 1280×720 @ 15 FPS using a Python/FastAPI backend with GPU acceleration.

**Core equation**: `v2.0 = multi_person_detection + tracking + [v1.0_per_person_pipeline × N] + room_analytics`

Each tracked person gets their own instance of Python ports of the v1.0 **legacy** detectors (BlinkDetector, HeadPoseTracker, FatigueCalculator) plus a BehaviorClassifier. This is the legacy path only: v2.0 has no DACE, no per-person MLEngine and no EMA; the per-person `attention_score` is a fixed lookup of the behavior label and `room_attention` is their mean.

---

## 2. Architecture

### 2.1 Full Pipeline

```mermaid
flowchart TD
    SRC["Video Source<br/>RTSP · File · Camera"] --> GRAB["Frame Grabber<br/>process every N frames"]
    GRAB --> P1["YOLOv8-m<br/>Person Detection"]
    GRAB --> P2["YOLOv8-s<br/>Object Detection"]
    GRAB --> P3["YOLOv8-pose<br/>17 Keypoints"]

    P1 & P2 & P3 --> FUSE["Detection Fusion<br/>associate objects+pose<br/>with person bboxes"]
    FUSE --> BT["ByteTrack<br/>persistent track IDs"]
    BT --> SM["SeatMapper<br/>Shapely polygon IoU"]

    SM --> LOOP["Per-Person Loop × N"]
    LOOP --> FL["Face Crop → MediaPipe<br/>FaceMesh → 68-pt ibug"]
    LOOP --> FM["FaceMetrics<br/>EAR · PERCLOS · Gaze<br/>Head Pose · Fatigue"]
    LOOP --> PA["PostureAnalysis<br/>torso angle · head drop"]
    LOOP --> OI["Object Interaction<br/>book · phone proximity"]

    FL & FM & PA & OI --> FF["Feature Fusion<br/>13-dim vector / frame"]
    FF --> TC["TemporalClassifier<br/>1D-CNN · 15-frame window<br/>+ context-gate"]
    TC --> BL["Behavior Label<br/>+ Attention Score"]

    BL --> RA["Room Analytics<br/>per-seat occupancy<br/>room attention mean"]
    RA --> WS["WebSocket 5 Hz<br/>→ Dashboard"]
```

### 2.2 Sequence Diagram

```mermaid
sequenceDiagram
    participant S as FrameSource
    participant RA as RoomAnalyzer
    participant PD as PersonDetector
    participant OD as ObjectDetector
    participant PE as PoseEstimator
    participant BT as ByteTrack
    participant SM as SeatMapper
    participant PS as PersonState
    participant BC as BehaviorClassifier
    participant WS as WebSocket

    S->>RA: frame (every Nth)
    RA->>PD: detect persons (YOLOv8-m)
    RA->>OD: detect objects (YOLOv8-s)
    RA->>PE: estimate poses (YOLOv8-pose)
    RA->>RA: fuse detections
    RA->>BT: update tracks
    BT-->>RA: tracks with persistent IDs
    RA->>SM: assign seats (polygon IoU)

    loop Per tracked person
        RA->>PS: get/create PersonState
        RA->>PS: face crop → landmarks → metrics
        RA->>PS: associate objects + pose
        PS->>BC: update(metrics, objects, proximity)
        BC-->>PS: BehaviorResult
    end

    RA->>RA: compute room_attention mean
    RA-->>WS: JSON snapshot @ 5 Hz
```

---

## 3. Perception Layer

### 3.1 Person Detection — YOLOv8-m

| Property | Value |
|----------|-------|
| Model | `yolov8m.pt` (25.9M params) |
| Why chosen | Best mAP/speed for medium scenes; handles occlusion well |
| Confidence | ≥ 0.40 |
| Inference | ~8ms/frame GPU |
| Alternatives | RT-DETR (heavier), Faster R-CNN (slower) |

### 3.2 Object Detection — YOLOv8-s

| Property | Value |
|----------|-------|
| Model | `yolov8s.pt` (11.2M params) |
| Target classes | book, phone, laptop, pen, notebook |
| Confidence | ≥ 0.25 |
| Inference | ~4ms/frame GPU |

### 3.3 Pose Estimation — YOLOv8-pose

| Property | Value |
|----------|-------|
| Model | `yolov8m-pose.pt` |
| Keypoints | 17 COCO (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) |
| Confidence | ≥ 0.40 |
| Inference | ~6ms/frame GPU |

**Posture Analysis** from pose keypoints:
```
torso_vector = shoulder_midpoint − hip_midpoint
lean_angle = angle_between(torso_vector, vertical)
is_upright = lean_angle < 15°
is_head_dropped = nose.y < shoulder_mid.y
```

### 3.4 Face Landmarks — MediaPipe FaceMesh

- 478 facial landmarks detected via MediaPipe Tasks API (VIDEO mode)
- Re-mapped to **68-point ibug convention** for compatibility with v1.0 face_metrics
- Mapping covers: jaw (17), eyebrows (10), nose (9), eyes (12), mouth (20)

---

## 4. Tracking Layer

### 4.1 ByteTrack

| Property | Value |
|----------|-------|
| Library | `boxmot` (includes ByteTrack) |
| Why chosen | SOTA MOT; handles occlusion via low-confidence detections; no ReID needed |
| Tracking | ~1ms/frame CPU |
| Target MOTA | ≥ 75% |
| Target IDF1 | ≥ 80% |
| Target ID switches | < 5/min |

### 4.2 Seat Mapper

Pre-configured seat zones as **Shapely polygons** in normalized [0,1] coordinates.

**Assignment algorithm**:
1. Primary: centroid containment (person bbox center inside seat polygon → score 1.0)
2. Fallback: IoU overlap (intersection/union ≥ 0.30)
3. If two persons overlap same seat, highest score wins

Example seat map (`data/seat_maps/room_default.json`):
```json
{
  "seat_1": [[0.05, 0.10], [0.30, 0.10], [0.30, 0.90], [0.05, 0.90]],
  "seat_2": [[0.35, 0.10], [0.60, 0.10], [0.60, 0.90], [0.35, 0.90]],
  "seat_3": [[0.65, 0.10], [0.95, 0.10], [0.95, 0.90], [0.65, 0.90]]
}
```

---

## 5. Feature Extraction

Python port of v1.0 `face-geometry.ts`. Per-person stateful detectors:

| Class | State | Function |
|-------|-------|----------|
| `BlinkDetector` | EAR history (120s), blink events | Blink detection, blink rate, PERCLOS |
| `HeadPoseTracker` | Pose history (30s) | Stability score, fidget score, pose variance |
| `FatigueCalculator` | PERCLOS/blink/EAR history (5min) | Composite fatigue index (0–100) |

**13-dim feature vector** (same layout as v1.0's `encode13`, indices 0-11 numerically equal per the generated fixture; index 12 uses v2.0's proximity codes, which equal only v1.0's ONNX input path, NOT v1.0's rule layer, which uses book 1.0 / pen 0.8 / desk 0.5 / phone 0.2; checked by `tests/test_cross_language_contract.py`. Known divergence: `behavior_classifier._extract` maps EAR 0.0 to 0.3):

| # | Feature | Normalization |
|---|---------|---------------|
| 0 | gaze_x | [-1, 1] |
| 1 | gaze_y | [-1, 1] |
| 2 | pitch_norm | pitch/90 → [-1, 1] |
| 3 | yaw_norm | yaw/90 → [-1, 1] |
| 4 | ear | [0, 1] |
| 5 | perclos | /100 → [0, 1] |
| 6 | stability | /100 → [0, 1] |
| 7–10 | object one-hots | book, phone, laptop, none |
| 11 | fidget | /100 → [0, 1] |
| 12 | proximity_code | `hands_not_visible` 0.00, `hands_on_desk` 0.25, `holding_pen` 0.50, `holding_book` 0.75, `holding_phone` 1.00 |

---

## 6. Behavior Classifier (1D-CNN)

### 6.1 Model Architecture

```mermaid
flowchart LR
    IN["Input<br/>(B, 13, 15)"] --> C1["Conv1D 13→32<br/>k=3, pad=1<br/>+ BN + ReLU"]
    C1 --> C2["Conv1D 32→64<br/>k=3, pad=1<br/>+ BN + ReLU<br/>+ Dropout 0.2"]
    C2 --> C3["Conv1D 64→64<br/>k=3, pad=1<br/>+ BN + ReLU"]
    C3 --> GAP["AdaptiveAvgPool1d(1)<br/>(B, 64)"]
    GAP --> FC["Linear 64→6"]
    FC --> OUT["Logits (6 classes)"]
```

| Property | Value |
|----------|-------|
| Parameters | 20,550 |
| Input shape | `(B, 13, 15)` channel-first |
| Output | 6-class logits |
| Receptive field | 7 frames = 1.4 seconds |
| Inference latency | 0.02 ms CPU |
| ONNX opset | 17 |
| Weight init | Kaiming (conv), Xavier (linear) |

### 6.2 Six Behavior Classes

| Class | Score | Description |
|-------|-------|-------------|
| `attentive_screen` | 85 | Watching lecture/screen |
| `attentive_reading` | 80 | Reading book/notes |
| `distracted` | 45 | Looking around, talking |
| `using_phone` | 25 | Phone distraction |
| `drowsy` | 20 | Head down + low EAR |
| `absent` | 0 | Seat empty |

---

## 7. Training Pipeline

### 7.1 Training CLI

```bash
# Realistic synthetic data
python3 v2.0/training/train_classifier.py \
    --realistic \
    --realistic-n-per-class 2000 \
    --realistic-augment 3 \
    --penalty-loss \
    --epochs 100 --batch 256

# With real datasets
python3 v2.0/training/train_classifier.py \
    --daisee /data/daisee \
    --engagenet /data/engagenet \
    --penalty-loss --epochs 100
```

### 7.2 ConfusionPenaltyLoss

Asymmetric loss penalizing costly confusions. **Updated**: previously only 3 of the 30 off-diagonal class pairs were specified (the rest silently defaulted to 1.0×); the complete matrix now lives in `training/penalty_matrix.py`, with every pair documented and a test (`tests/test_penalty_matrix.py`) asserting full coverage and that the matrix is intentionally asymmetric (not a copy-paste placeholder):

| True → Predicted | Penalty | Rationale |
|------------------|---------|-----------|
| `attentive_reading` → `using_phone` | 3.0× | Highest cost — false-accuses a reading student of phone use (the core motivating case) |
| `using_phone` → `attentive_reading` / `attentive_screen` | 2.5× | Missed real phone use |
| `attentive_screen` → `using_phone`, `distracted` → `using_phone` | 2.0× | False-accusation of phone use |
| `absent` ↔ any present state | 1.8–2.0× | Attendance-critical — missed absence or falsely-marked absence |
| `drowsy` confusions with any other class | 1.3–1.5× | Degree-of-attention error, not a false accusation |
| `attentive_screen` ↔ `attentive_reading`, and either ↔ `distracted` | 1.2–1.3× | Lowest cost — student still clearly present |

These weights are a design proposal derived from pedagogical-cost reasoning, not fit to labeled data (no real dataset exists yet to fit them empirically) — see `training/penalty_matrix.py`'s per-pair rationale comments for the full reasoning, and review/tune with domain stakeholders once real evaluation data exists.

### 7.3 Training Details

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (η_min = lr×0.01) |
| Label smoothing | 0.05 |
| Class weighting | Inverse-frequency |
| Early stopping | Patience 10 epochs |
| Validation split | 15% |
| Gradient clipping | max_norm = 1.0 |

### 7.4 ONNX Export

```bash
python3 v2.0/training/export_onnx.py \
    --checkpoint v2.0/models/behavior_classifier.pt \
    --output v2.0/models/behavior_classifier_v2.onnx \
    --verify
```

---

## 8. Context-Gate Override

Post-processes CNN softmax probabilities with hard object-context rules.

> **Reachability (honest).** By default the Context-Gate never fires on the server: `hand_proximity` is always `hands_not_visible`, and the `holding_*` conditions need `COGNEN_HAND_PROXIMITY=pose` (wrist-based, unvalidated on CCTV, `holding_pen` never produced). The "Downward gaze" row is effectively unreachable: v2.0 keeps the legacy pitch formula, where looking down makes pitch rise, so the `pitch < -15` check cannot trigger (documented in `tests/test_pitch_characterisation.py`; DACE fixes this in v1.0 only).

| Condition | Action |
|-----------|--------|
| Phone detected + holding_phone | Boost `using_phone` to ≥0.85, suppress `attentive_reading` to ≤0.05 |
| Book detected + holding_book, no phone | Boost `attentive_reading` to ≥0.80, suppress `using_phone` to ≤0.08 |
| Downward gaze + no objects | Compress confidence: `p = p×0.7 + uniform×0.3` |

---

## 9. Room Analytics

Per-frame output structure:
```json
{
  "ts": 1741519810.4,
  "persons": [
    {
      "track_id": 1,
      "seat_id": "seat_2",
      "behavior": "attentive_screen",
      "attention_score": 82.5,
      "confidence": 0.94,
      "contradiction_index": 0.12,
      "contradiction_category": "coherent"
    }
  ],
  "occupancy": {
    "occupied": ["seat_2", "seat_4"],
    "absent": ["seat_1", "seat_3", "seat_5", "seat_6"],
    "total": 6
  },
  "room_attention": 82.5
}
```

---

## 10. Video Input Sources

Unified `FrameSource` abstraction supporting three source types:

| Source | Class | Key Config |
|--------|-------|------------|
| RTSP | `RTSPSource` | URL, transport (tcp/udp), auto-reconnect |
| File | `VideoFileSource` | Path, loop option, encoded FPS pacing |
| Camera | `CameraSource` | Device index, resolution, FPS |

Frame-skip: `--process-every N` analyses 1 in N frames. At 15 FPS with N=3 → ~5 analysis FPS.

---

## 11. FastAPI Server & WebSocket

```bash
export COGNEN_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m server.main                       # binds API_HOST (default 127.0.0.1)
# or: uvicorn server.main:app --host 0.0.0.0 --port 8000 --ws-max-size 4096
```

WebSocket pushes the latest room snapshot (up to **5 Hz**, unchanged snapshots
only as a 15 s heartbeat) to connected dashboard clients. Every endpoint except
`/healthz` and `/static` requires an API key — see [§24 Security](#security).

---

## 12. Tech Stack

| Layer | Technology |
|-------|------------|
| API server | FastAPI + uvicorn |
| Person detection | YOLOv8-m (Ultralytics) |
| Object detection | YOLOv8-s |
| Pose estimation | YOLOv8-pose (17 COCO keypoints) |
| Face landmarks | MediaPipe FaceMesh → 68-pt ibug |
| Tracking | ByteTrack (boxmot) |
| Seat mapping | Shapely polygon IoU |
| Behavior classifier | 1D-CNN ONNX (20,550 params) |
| Video input | OpenCV VideoCapture |
| Dashboard | FastAPI WebSocket @ 5 Hz |
| Training | PyTorch + ONNX export |

---

## 13. Repository Structure

```
v2.0/
├── server/
│   ├── main.py                  # FastAPI app, REST + WebSocket
│   ├── config.py                # Settings, model paths, thresholds
│   └── requirements.txt
├── input/
│   └── frame_source.py          # FrameSource ABC: RTSP, File, Camera
├── perception/
│   ├── person_detector.py       # YOLOv8-m
│   ├── object_detector.py       # YOLOv8-s
│   ├── pose_estimator.py        # YOLOv8-pose + posture analysis
│   └── face_landmarks.py        # MediaPipe FaceMesh → 68-pt ibug
├── tracking/
│   ├── byte_tracker.py          # ByteTrack wrapper (boxmot)
│   └── seat_mapper.py           # Seat zone assignment (Shapely)
├── features/
│   ├── face_metrics.py          # Python port of v1.0 face-geometry.ts / scoring/
│   └── feature_vector.py        # Canonical 13-dim feature encoder (see §23)
├── inference/
│   ├── behavior_classifier.py   # Temporal 1D-CNN + context-gate
│   ├── contradiction_detector.py # Cross-Modal Contradiction Vector (see §22)
│   └── room_analyzer.py         # Main orchestrator
├── training/
│   ├── model.py                 # BehaviorClassifierCNN (20,550 params)
│   ├── train_classifier.py      # Training CLI (GroupShuffleSplit, see §23)
│   ├── penalty_matrix.py        # Complete 6×6 ConfusionPenaltyLoss matrix (see §23)
│   ├── export_onnx.py           # ONNX export + verification
│   ├── feature_extractor.py     # Thin shim over features/feature_vector.py
│   └── datasets/
│       ├── base_dataset.py      # build_windows() now tracks group_id (see §23)
│       ├── realistic_synthetic_loader.py  # _ClassSpec is now a typed dataclass
│       ├── daisee_loader.py
│       ├── engagenet_loader.py
│       ├── classroom_loader.py
│       ├── annotation_tools/
│       │   └── cvat_to_classroom_json.py  # CVAT XML → ClassroomDataset format converter
│       └── CALIBRATION.md       # Template for validating synthetic vs. real distributions
├── evaluation/
│   ├── evaluate.py              # Full evaluation runner (no longer fabricates results — see §23)
│   ├── behavior_metrics.py      # Accuracy, F1, real Temporal Consistency Coefficient (see §23)
│   ├── tracking_metrics.py      # MOTA, IDF1, ID switches
│   ├── occupancy_metrics.py     # Occupancy accuracy, absence latency
│   ├── latency_benchmark.py     # P95 latency
│   └── reports/
├── models/
│   ├── behavior_classifier_v2.onnx
│   └── download_models.py
├── dashboard/static/            # Required by server/main.py's StaticFiles mount
├── data/seat_maps/
│   └── room_default.json
├── tests/                       # pytest — see §25 for the lightweight-tier test list
├── pyproject.toml               # ruff + mypy config (see §25)
├── run_analyzer.py              # CLI entry point
├── yolov8m-pose.pt
└── yolov8s.pt
```

---

## 14. Setup & Installation

```bash
cd v2.0
python3 -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt

# GPU (NVIDIA Linux/Windows only):
# pip install onnxruntime-gpu==1.18.0

# Download YOLO weights
python3 models/download_models.py
```

> `server/requirements.txt` is pinned with `==` (not `>=`) for reproducible installs — each version is the one this code was actually written/tested against; unpinned ranges previously let a fresh install silently pull in an untested newer release. Every pin is verified to exist on PyPI. To upgrade a dependency: bump its pin, run the full test suite, then commit the bump on its own.

For a lightweight dev setup that only needs `pytest tests/` (excludes YOLO/torch/opencv-dependent tests), see §25.

---

## 15. CLI Usage

```bash
# RTSP
python3 run_analyzer.py --source rtsp \
    --url "rtsp://admin:pass@192.168.1.100:554/ch1" \
    --room room_default --process-every 3 --show

# Video file
python3 run_analyzer.py --source file \
    --url /data/classroom.mp4 --room room_default --loop

# Webcam
python3 run_analyzer.py --source camera --device 0 --show

# JSON output
python3 run_analyzer.py --source camera --print-json | jq '.room_attention'
```

---

## 16. REST API Reference

| Method | Path | Description | Error responses |
|--------|------|-------------|------------------|
| `POST` | `/rooms/{room_id}/start` | Start analysis (SourceConfig body) | `422` invalid `room_id` or source config; `429` if `MAX_CONCURRENT_ROOMS` reached; `500` if model weights are missing |
| `POST` | `/rooms/{room_id}/stop` | Stop analysis | `404` room not found (previously returned `200` with an `{"error": ...}` body) |
| `GET` | `/rooms/{room_id}/snapshot` | Latest analytics JSON | `404` room not found |
| `GET` | `/rooms` | List all rooms | — |
| `WS` | `/ws/room/{room_id}` | Real-time push (latest snapshot, ≤ 5 Hz) | Pre-accept close (HTTP 403) `1008` for bad `room_id` / Origin / API key, `1013` when a connection cap is hit; after accept `1008` room not found or slow client, `1009` oversized client message, `1011` internal error |
| `GET` | `/healthz` | Liveness, unauthenticated: `{"status":"ok"}` | — |

All routes except `/healthz` and `/static` additionally return `401` (no key), `403` (wrong key), `429` (rate limit / too many failed auths, with `Retry-After`), `503` (no `COGNEN_API_KEY` configured), `413` (body > `COGNEN_MAX_BODY_BYTES`).

`room_id` must match `^[a-zA-Z0-9_-]{1,64}$` — see [§24](#24-server-hardening) for why.

Environment variables: `COGNEN_PROCESS_EVERY_N`, `COGNEN_RTSP_TRANSPORT`, `COGNEN_CAMERA_BUFFER`, `COGNEN_MAX_CONCURRENT_ROOMS` (default 20), `API_HOST`, `API_PORT`

API source policy (`input/frame_source.py::validate_api_source`) and all security env vars: see [§24 Security](#security). `rtsp_transport` must be `tcp` or `udp`. The CLI (`run_analyzer.py`) is not restricted.

---

## 17. Testing & Evaluation

```bash
# Full test suite (needs the full requirements.txt: torch, opencv, ultralytics, shapely, etc.)
python3 -m pytest tests/ -v

# Lightweight tier only (numpy + pytest + scikit-learn — no GPU/ML deps needed;
# this is what CI runs, see §25): feature encoding, penalty matrix, temporal
# consistency, train/val split, CVAT conversion, v1-pipeline logic, behavior rules
python3 -m pytest tests/ --ignore=tests/test_tracking.py --ignore=tests/test_v2_pipeline.py -v

# Current light tier: 86 pass / 6 skip. Includes the cross-language contract tests
# (test_cross_language_contract.py, reads shared/fixtures/encode13_cross_language.json),
# test_hand_proximity.py and test_pitch_characterisation.py.

# Evaluation against ARCHITECTURE.md targets
python3 v2.0/evaluation/evaluate.py \
    --model v2.0/models/behavior_classifier_v2.onnx \
    --classroom /path/to/classroom/dataset \
    --seat_map v2.0/data/seat_maps/room_default.json
```

> **Updated**: `evaluate.py` previously fabricated numeric results (`pred = label if rng.random() > 0.2 else random_class`) when no `--classroom` dataset path was supplied — this has been fixed. Running `evaluate.py` without a real dataset now reports `"status": "SKIPPED"` for each section instead of a synthesized score. See §18's banner for the current, honest state of the numbers below.

---

## 18. Performance Metrics

> ⚠ **These are targets, not measured results** — see [README.md §7](../README.md#7-performance-metrics) for the full, current honesty banner. The table below was trained/tested entirely on the same synthetic AR(1) generator (train seed=42, test seed=43) — a circular evaluation, not a generalization test — which is why every score reads a suspiciously perfect **1.000**. No real, subject-disjoint test set has been evaluated yet. `evaluate.py`'s fabrication bug (see §17) has been fixed, but that only means it now correctly reports `SKIPPED` instead of a fake number — it does not mean these numbers are real.

Trained on 42,600 synthetic windows (AR(1) temporal autocorrelation). Tested on 3,000 independent windows drawn from the *same* synthetic distribution.

| Metric | Score | Target | ✓ |
|--------|-------|--------|---|
| Behavior accuracy | **1.000** | ≥ 0.85 | ✓ |
| F1-macro | **1.000** | ≥ 0.80 | ✓ |
| Precision (using_phone) | **1.000** | ≥ 0.90 | ✓ |
| Recall (attentive_reading) | **1.000** | ≥ 0.85 | ✓ |
| Temporal consistency | **1.000** | ≥ 0.90 | ✓ |
| Tracking MOTA | **1.000** | ≥ 0.75 | ✓ |
| Tracking IDF1 | **1.000** | ≥ 0.80 | ✓ |
| Seat occupancy accuracy | **0.963** | ≥ 0.95 | ✓ |
| Inference latency P95 | **< 0.1 ms** | < 100 ms | ✓ |

> Real-classroom accuracy estimated at 0.82–0.88 pending DAiSEE/EngageNet fine-tuning.

### System Performance Targets

| Metric | Target |
|--------|--------|
| Latency | < 100ms/frame (GPU) |
| FPS | 15 FPS (server) |
| GPU memory | < 4 GB |

---

## 19. Datasets

| Dataset | Purpose | Size |
|---------|---------|------|
| DAiSEE | Engagement detection | 9K videos, 4 levels |
| EngageNet | Student engagement | 127K frames |
| COCO | Person + pose pretraining | 330K images |
| Open Images v7 | Object detection | 9M images |
| MPII Human Pose | Pose estimation | 25K images |
| MOT20 | Multi-object tracking | 8 sequences |
| Custom classroom | Fine-tuning | Collect 50–100 hours |

---

## 20. Privacy

1. **No raw video stored** — only derived feature vectors and scores
2. **On-premise** — all inference local; no cloud API calls (v2.0 has no Gemini/report call)
3. **Anonymization** — track IDs only (no name mapping is implemented; `ANONYMISE_BY_DEFAULT` in `config.py` is unused)
4. **Retention** — auto-purge is **not implemented**: `DATA_RETENTION_HOURS` exists in `config.py` but nothing reads it. Snapshots are served live and held in memory only.

---

## 21. Cross-Reference: v1.0

v2.0 reuses hand-ported copies of v1.0's *legacy* per-person math (not the DACE path, not shared code; kept in step by hand, with only the generated `encode13` fixture and the pitch characterisation test as automated checks). See [v1.0 README](../v1.0/README.md) for browser deployment details.

| Component | v1.0 (Browser) | v2.0 (Server) |
|-----------|----------------|---------------|
| Face detection | face-api.js | YOLOv8-m + MediaPipe |
| Landmarks | face-api.js 68-pt (MediaPipe Face Landmarker in opt-in DACE mode) | MediaPipe → ibug 68-pt (`_MP_TO_68` has known wrong indices, see CLAUDE.md findings) |
| EAR/MAR/Gaze | TypeScript | Python port |
| Object detection | YOLOv8-nano ONNX.js | YOLOv8-s Ultralytics |
| Temporal classifier | Browser ONNX | Server ONNX |
| Tracking | Single person | ByteTrack multi-person |
| Seat mapping | N/A | Shapely polygons |
| Posture analysis | N/A | YOLOv8-pose keypoints |
| Deployment | Browser + Chrome Ext | FastAPI + GPU |

---

## 22. Cross-Modal Contradiction Vector (Novel)

Additive to the already-documented Context-Gate Override (§8): that mechanism is a rule that fires on *one* known confusion (phone vs. reading) and discards the disagreement magnitude once applied. The **Cross-Modal Contradiction Vector (CMCV)**, ported from v1.0 (v1.0 legacy path only; DACE mode is not ported) (see [v1.0 README §21](../v1.0/README.md#21-cross-modal-contradiction-vector-novel) for the full spec), is a general, symmetric, per-frame disagreement statistic over all modality pairs (gaze, posture-stability, ocular/PERCLOS, object-context, hand-action).

**Implementation**: `inference/contradiction_detector.py`, called from `RoomAnalyzer._analyze_person()` alongside the behavior classifier — computed per tracked person, using the same `metrics`, `person_objs`, and `hand_proximity` already assembled there, and exposed as `contradiction_index`/`contradiction_category` sibling fields on the per-person result dict (never folded into `attention_score`). It uses the same taxonomy (`fatigue_masking`, `prop_without_engagement`, `context_conflict`, `coherent`) as the TypeScript implementation, but there is **no automated cross-language check** (no shared fixture for CMCV); the two ports are kept in step by hand, and the v1.0 `hands_on_desk` / `coherent` quirks listed in CLAUDE.md findings may differ. With the default `COGNEN_HAND_PROXIMITY=off` the hand-action modality always reads `hands_not_visible`.

**Integration point**: feeds the currently-empty `v2.0/dashboard/` and `v2.0/analytics/` stub packages — a natural, currently-unused home for a "flag high-contradiction students for teacher review" view.

**Validation status**: same caveat as v1.0 — computationally free and unit-tested against constructed cases, but full empirical validation needs human-annotated modality-conflict labels that don't exist in DAiSEE/EngageNet yet.

---

## 23. Training Pipeline Correctness Fixes

A pass through the training/eval pipeline surfaced and fixed several real bugs, independent of the "make the benchmark numbers real" effort (§18):

- **Feature-encoding mismatch (live train/inference bug)**: `training/feature_extractor.py` normalized gaze/pitch/yaw to `[0,1]`, while `realistic_synthetic_loader.py` (the synthetic generator actually used for training) and `inference/behavior_classifier.py::_extract` (the actual inference-time consumer) both use `[-1,1]`. A model trained partly on `[0,1]`-encoded windows and partly on `[-1,1]`-encoded windows would learn a broken, inconsistent decision boundary. Fixed by consolidating all feature extraction into one canonical `features/feature_vector.py`, standardized on `[-1,1]`, imported by `feature_extractor.py` and all three real-data loaders (`daisee_loader.py`, `engagenet_loader.py`, `classroom_loader.py`) — which previously each hand-reimplemented their own crude EAR/pitch/gaze math instead of calling `features/face_metrics.py`'s corrected functions. This also filled in `gaze_x`/`gaze_y` for `engagenet_loader.py` and `classroom_loader.py`, which previously never computed gaze at all (defaulted to 0.0, i.e. no gaze signal).
- **Train/val leakage**: `train_classifier.py::_train_val_split` previously split pooled sliding-window samples via a naive random permutation — for DAiSEE/EngageNet, overlapping windows from the same clip/subject (stride < window_size) could land on both sides of the split. Fixed by threading a `group_id` (clip_id, or `f"{clip_id}_{track_id}"` for classroom data) through `base_dataset.py::build_windows()` and every loader's `_iter_samples()`, and switching to `sklearn.model_selection.GroupShuffleSplit`. Verified with a regression test (`tests/test_train_val_split.py`) asserting no group_id appears on both sides of a split.
- **Complete ConfusionPenaltyLoss matrix**: see §7.2.
- **Real Temporal Consistency Coefficient**: `BehaviorEvaluator` previously only stored per-person *prediction* sequences (never ground truth), so its "temporal consistency" metric was a mode-agreement score over raw prediction blocks — structurally incapable of distinguishing bad flicker (prediction changes when the true behavior didn't) from correctly following a real transition (prediction changes because ground truth did too). Fixed: `BehaviorEvaluator.temporal_consistency()` now tracks both sequences and computes `TCC = (matching consecutive predictions) / (transitions where ground truth did NOT change)` — conditioning on ground-truth stability is the key fix. See `evaluation/behavior_metrics.py` and `tests/test_temporal_consistency.py`.
- **`_ClassSpec` dataclass conversion**: `realistic_synthetic_loader.py`'s per-class distribution spec was a `__slots__` + `**kwargs`/`setattr` class that mypy couldn't verify attribute existence/types through — converted to a `@dataclass(slots=True)` with explicit typed fields (same call sites, same memory layout, now statically verifiable). This alone resolved 20 of the first mypy pass's 55 findings.
- **Dataset scaffolding**: `training/datasets/annotation_tools/cvat_to_classroom_json.py` converts a CVAT-for-video-1.1 XML export into `ClassroomDataset`'s expected annotation format — nothing previously produced that format, so the classroom loader was unusable without hand-authoring JSON. Validated end-to-end against a hand-built fixture XML and the real `ClassroomDataset` loader (`tests/test_cvat_to_classroom_json.py`). `training/datasets/CALIBRATION.md` is a template for validating the synthetic generator's class-conditional constants against real extracted distributions once real footage exists — currently unfilled, since no real footage has been run through the pipeline yet.

---

## 24. Server Hardening

- **Path-traversal / injection fix**: `server/config.py::load_seat_map()` built a filesystem path directly from an unvalidated `room_id` path parameter (`SEAT_MAPS_DIR / f"{room_id}.json"`). Fixed by validating `room_id` against `^[a-zA-Z0-9_-]+$` (`ROOM_ID_PATTERN`) both at the FastAPI layer (`server/main.py::_validate_room_id`, `422` response) and defense-in-depth inside `load_seat_map()` itself (`ValueError`) — verified with a live `TestClient` that command-injection-style room_ids (`room;rm -rf`, `room$(whoami)`) are correctly rejected.
- **Model-loading error handling**: `PersonDetector`/`ObjectDetector`/`PoseEstimator` previously called `YOLO(str(MODEL_PATH))` with no existence check — a missing `.pt` file surfaced as an opaque stack trace deep inside `ultralytics`. `server/config.py::require_model_file()` now validates existence first, raising a clear `FileNotFoundError` pointing at `models/download_models.py`; `server/main.py::start_room` catches this and returns `HTTPException(500, ...)` with the actionable message instead of crashing.
- **HTTP status codes**: `stop_room`/`snapshot` previously returned `{"error": "Room not found"}` with an implicit `200 OK` — now `404`. See §16.
- **WebSocket robustness**: `room_websocket` previously had no exception handling around snapshot serialization — an exception there crashed the streaming task silently with no client-visible error. Now wraps serialization per-tick (logs and skips a bad tick rather than dying) and closes with an explicit code (`1008` policy violation for an invalid/missing room, `1011` internal error on an unexpected failure) instead of hanging.
- **Concurrent-room rate limit**: `/rooms/{room_id}/start` could previously spawn unbounded `asyncio.create_task()` calls. Now capped at `COGNEN_MAX_CONCURRENT_ROOMS` (default 20, env-configurable) — the `(N+1)`th concurrent room-start request gets `429`. No new dependency (a simple in-process counter, not `slowapi`).
- **`dashboard/static` directory**: `server/main.py` mounts `StaticFiles(directory=.../dashboard/static)` at import time — this directory didn't exist anywhere in the repo, meaning the server would have crashed on startup before ever handling a request. Created (with a `.gitkeep`) so the mount succeeds; found and fixed incidentally while writing a `TestClient`-based verification for the fixes above.

<a id="security"></a>
### Security (API auth, SSRF, WebSocket)

Implemented in `server/security.py`, `server/main.py`, `input/frame_source.py`; tests in `tests/test_server_security.py`. Env vars are also listed in `server/config.py`.

**Secure defaults**: binds `127.0.0.1`; with no `COGNEN_API_KEY` protected routes answer `503` and `python -m server.main` refuses a non-loopback `API_HOST`; `/docs` & `/openapi.json` off; no CORS origins; private/loopback/metadata stream targets, http(s) streams, URL credentials and camera sources all denied.

| Variable | Default | Purpose |
|---|---|---|
| `COGNEN_API_KEY` | unset | Comma-separated keys (≥ 16 chars; several = rotation). Send `X-API-Key: <key>`; WebSocket may use `?api_key=` (browsers can't set WS headers — query strings can land in proxy logs, prefer the header for non-browser clients). Compared with `secrets.compare_digest`. |
| `COGNEN_ALLOW_INSECURE` | off | `1` = no auth when no key is set (local dev; loud warning once). |
| `COGNEN_AUTH_FAIL_RATE_PER_MIN` / `_BURST` | 10 / 10 | Failed-auth bucket per client IP → `429`. |
| `COGNEN_START_RATE_PER_MIN` / `COGNEN_START_BURST` | 6 / 3 | `/start` token bucket, per key **and** per IP → `429` + `Retry-After`. |
| `COGNEN_MAX_BODY_BYTES` | 16384 | Request body cap → `413` (Content-Length and streamed). |
| `COGNEN_CORS_ORIGINS` | none | Exact origins (`https://host[:port]`); `*` refused; credentials never allowed. Also the WebSocket Origin allowlist (requests without Origin = non-browser, still need a key). |
| `COGNEN_ENABLE_DOCS` | off | Expose `/docs`, `/redoc`, `/openapi.json`. |
| `COGNEN_WS_MAX_PER_KEY` / `_PER_ROOM` / `_TOTAL` | 8 / 32 / 256 | WebSocket connection caps (`1013`). |
| `COGNEN_WS_SEND_TIMEOUT_S` | 2.0 | A send slower than this drops the client (backpressure; only the latest snapshot is ever sent, nothing is queued). |
| `COGNEN_WS_PUSH_INTERVAL_S` / `COGNEN_WS_HEARTBEAT_S` | 0.2 / 15 | Push period / resend period for an unchanged snapshot. |
| `COGNEN_WS_MAX_MESSAGE_BYTES` / `COGNEN_WS_MAX_CLIENT_MSGS_PER_S` | 4096 / 5 | Client→server message limits (`1009` / `1008`); text `ping` → `{"type":"pong"}`. `python -m server.main` also sets uvicorn `ws_max_size` and protocol pings (20 s). |
| `COGNEN_VIDEO_DIRS` | `v2.0/data/videos` | Roots for `file` sources; the **realpath** (all symlinks resolved) must be inside, re-checked at open. |
| `COGNEN_VIDEO_EXTENSIONS` | common video types | File extension allowlist (blocks `.m3u8`/`.ffconcat` playlists). |
| `COGNEN_STREAM_SCHEMES` | `rtsp,rtsps` | Add `http,https` to permit HTTP streams (redirects disabled). |
| `COGNEN_STREAM_HOSTS` | unset | Allowlist of hostnames **and/or CIDRs** (e.g. `192.168.10.0/24,cam1.lan`). When set, other hosts are refused. A listed name may resolve to private IPs; only a CIDR can admit loopback/link-local. |
| `COGNEN_ALLOW_PRIVATE_STREAMS` | off | Allow RFC1918 / CGNAT / IPv6 ULA targets without an allowlist. |
| `COGNEN_STREAM_PORTS` | any | Optional port allowlist. |
| `COGNEN_ALLOW_URL_CREDENTIALS` | off | Allow `rtsp://user:pass@…` (always redacted in logs). |
| `COGNEN_ALLOW_CAMERA_SOURCE` | off | Allow `source_type: "camera"` via the API. |

**SSRF policy**: every A/AAAA record of the stream host is checked; any disallowed record rejects the URL. Always denied unless a `COGNEN_STREAM_HOSTS` CIDR covers it: loopback, link-local (incl. `169.254.169.254` metadata, `fe80::/10`), unspecified, multicast, reserved. IPv4-mapped, NAT64, 6to4 and Teredo IPv6 forms are judged by the embedded IPv4. FFmpeg runs with protocol/demuxer whitelists (no HLS/DASH/concat, no `file:` from a stream).

**DNS rebinding**: `rtsp://` hostnames are pinned — FFmpeg connects to the validated IP literal. `rtsps`/`http(s)` can't be pinned without breaking TLS/Host, so they are re-resolved and re-validated before every (re)connect; a small window between that check and FFmpeg's own lookup remains. FFmpeg cannot turn off RTSP 3xx redirects (remaining risk, limited by the rtsp/rtp-only protocol whitelist). Prefer `COGNEN_STREAM_HOSTS` with CIDRs for CCTV networks.

**Errors and logs**: unhandled errors return `{"detail":"internal server error","request_id":…}` (traceback only in the server log); 422 bodies don't repeat submitted values; a missing-model 500 carries no filesystem path. Log records from the `uvicorn*`/`cognen*` loggers are passed through a redaction filter (`api_key=`, `X-API-Key`, URL userinfo); stream URLs are logged without credentials or query strings; keys are logged only as a SHA-256 prefix. Responses carry `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, and on API paths `Cache-Control: no-store` + `CSP default-src 'none'`.

**Proxies**: rate limits use the TCP peer address. Behind a reverse proxy, run uvicorn with `--proxy-headers --forwarded-allow-ips=<proxy ip>`. `X-Forwarded-For` is never trusted directly.

---

## 25. Code Quality, Testing & CI

- **Linting**: `ruff check .` via `pyproject.toml` (`select = ["E", "F", "I"]` — pyflakes + pycodestyle errors + import sorting; a lenient first pass, not a full rewrite). `ruff format --check` is deliberately **not** enforced — this codebase uses intentional visual alignment in tabular data structures (e.g. `training/penalty_matrix.py`, `realistic_synthetic_loader.py`'s `_SPECS` table) that Black-style auto-formatting would destroy.
- **Type checking**: `mypy .` via `pyproject.toml` (`disallow_untyped_defs = false` — non-strict, since much of the codebase predates any type-checker; catches real errors in code that does have annotations without demanding annotations everywhere). First pass found 55 issues; genuine bugs and one real code-quality fix (see §23's `_ClassSpec` conversion) brought it to 26 remaining, individually verified to be benign pre-existing patterns (mostly "declare as `None`, lazily assign a real object" across 3+ loader files, and a couple of dict-value-type inference limits) rather than actual defects. CI runs `mypy .` as **informational, not a merge gate** (`continue-on-error: true`) given this backlog — new code should still aim to be mypy-clean.
- **Test suite**: see §17 for the lightweight-vs-full test tier split. Mypy's first pass also caught a real regression: `evaluation/evaluate.py` still unpacked `ClassroomDataset` iteration as a 2-tuple after the train/val-leakage fix (§23) changed every loader to yield 3-tuples `(window, label, group_id)` — would have crashed the moment real classroom evaluation ran. Fixed.
- **Dependency pinning**: see §14.
- **`pip-audit`**: runs against `server/requirements.txt` in CI.
- **CI**: `.github/workflows/ci.yml` (shared file with v1.0, path-filtered) runs `ruff check`, `mypy` (informational), the lightweight pytest tier, and `pip-audit` on every push/PR to `main`. See [README.md](../README.md) for the full pipeline description.
