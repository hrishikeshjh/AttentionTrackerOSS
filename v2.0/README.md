# AttentionTrackerOSS v2.0 — Multi-Person CCTV Room Analyzer

> **Lead Developer & System Architect:** **Hrishikesh Jha** ([@hrishikeshjh](https://github.com/hrishikeshjh) · `hrishikeshjhapersonal@gmail.com`)  
> **Repository:** [AttentionTrackerOSS](https://github.com/hrishikeshjh/AttentionTrackerOSS)  
> **Subsystem:** Server-based, multi-person cognitive engagement analysis for classrooms & auditoriums  
> For the single-user browser version, see **[v1.0 README](../v1.0/README.md)**. For the ESP32 hardware hub, see **[ESP32 README](../esp32_relay_app/README.md)**.

> **Status (honest).** v2.0 is implemented and unit-tested (light tier 268 pass / 43 skip; heavy tier 338 pass; `ruff` clean) but it has **never been run on real classroom/CCTV footage, a real RTSP stream or a GPU, and no model has been trained or evaluated on real data**. Every accuracy, latency and FPS figure here is a **target** until measured on dataset D1 / pilot set P1. The served model, `behavior_classifier_v3`, was trained and tested on the project's own hand-written synthetic generator, so its 1.000 scores are **circular** (see [`evaluation/reports/behavior_classifier_v3_synthetic_note.md`](evaluation/reports/behavior_classifier_v3_synthetic_note.md)).
>
> **What v2.0 does now (all unit-tested only):** DACE's *measurement* is ported (head pitch from the MediaPipe facial transformation matrix against a per-track screen-facing reference, blendshape gaze, gaze-corrected eye openness/PERCLOS) with one sign convention (down = negative); the Context-Gate Override is live by default (`COGNEN_HAND_PROXIMITY=pose`); a 15-dim feature vector adds posture; the API is authenticated with SSRF/WebSocket hardening; ByteTrack is real (boxmot 16). DACE's abstention ("Unsure"/score-hold) and rule layer are **v1.0-only** and are not in v2.0 ([ARCHITECTURE.md §11](../ARCHITECTURE.md)).

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

Each tracked person gets their own instance of Python ports of the v1.0 **legacy** detectors (BlinkDetector, HeadPoseTracker, FatigueCalculator), a port of the DACE *measurement* (§5) and a BehaviorClassifier. v2.0 has no DACE abstention/rule layer, no per-person MLEngine and no EMA; the per-person `attention_score` is a fixed lookup of the behavior label and `room_attention` is their mean.

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
    LOOP --> FL["Face Crop → MediaPipe<br/>FaceLandmarker → 68-pt ibug"]
    LOOP --> FM["FaceMetrics<br/>EAR · PERCLOS · Gaze<br/>Head Pose · Fatigue"]
    LOOP --> PA["PostureAnalysis<br/>torso angle · head drop"]
    LOOP --> OI["Object Interaction<br/>book · phone proximity"]

    FL & FM & PA & OI --> FF["Feature Fusion<br/>15-dim vector / frame"]
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

### 3.4 Face Landmarks: MediaPipe FaceLandmarker

- 478 landmarks per face via the MediaPipe Tasks API in **IMAGE mode** (RoomAnalyzer feeds crops of different people through one detector, so VIDEO mode's tracking state would leak between people), run on the upper square of each tracked person's clipped box
- Re-mapped to the **68-point ibug convention** (`_MP_TO_68`, rebuilt from the canonical face mesh and checked by `tests/test_face_landmarks.py`; the earlier table had a wrong jaw, nose tip and outer-lip corners, which made MAR meaningless)
- Also returns the 4x4 facial transformation matrix, blendshapes and iris landmarks, which feed `features/dace_measurement.py` (§5)
- Failure is loud: a missing model or init error logs an error and sets `available=False`; metrics then carry `face_detected=None` instead of silently defaulting

---

## 4. Tracking Layer

### 4.1 ByteTrack

| Property | Value |
|----------|-------|
| Library | `boxmot==16.0.11` (`ByteTrack` exported at the package root). Older boxmot (10.x) did not export it and the stub tracker was silently used; the wrapper still falls back to a simple stub if boxmot is missing or incompatible, so check the startup log |
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

Python port of v1.0's *legacy* `scoring/` math (`features/face_metrics.py`) plus a port of v1.0's DACE *measurement* (`features/dace_measurement.py`). Per-track state lives in `features/track_features.py` (`TrackMetricsState`), which the training loaders reuse so train and inference measure identically.

| Class | State | Function |
|-------|-------|----------|
| `BlinkDetector` | EAR history (120s), blink events | Blink detection, blink rate, PERCLOS |
| `HeadPoseTracker` | Pose history (30s) | Stability score, fidget score, pose variance |
| `FatigueCalculator` | PERCLOS/blink/EAR history (5min) | Composite fatigue index (0–100) |

**Measurement (ported from v1.0 DACE, unit-tested on synthetic matrices only).** Pitch comes from the MediaPipe transformation matrix, relative to a per-track median "screen-facing" reference; gaze comes from blendshapes with an iris cross-check; eye openness and PERCLOS are gaze-corrected. The legacy 68-point pitch formula (`calculate_head_pose`) is not used for the model input: it has a neutral offset near +12.6 and *rises* when the head tilts down (`tests/test_pitch_characterisation.py`). **Sign convention (single negation): pitch and `gaze_y` are DOWN = NEGATIVE in the feature vector**; the measurement layer is +down and is flipped exactly once (`to_feature_pitch`, `to_feature_gaze_y`). Thresholds (15 deg, 0.5) are initial values awaiting pilot calibration.

**Feature vectors** (`features/feature_vector.py`, `FEATURE_VECTOR_VERSION = 2`):

- `encode_feature_vector`: **13 dims**, v1.0-compatible. Indices 0-11 equal v1.0's `encode13` per the generated fixture `shared/fixtures/encode13_cross_language.json`; index 12 uses v2.0's proximity codes, which equal v1.0's ONNX input path but **not** v1.0's rule layer (book 1.0 / pen 0.8 / desk 0.5 / phone 0.2). `tests/test_cross_language_contract.py` checks this and now shows the two encoders agree, including EAR 0.0 (inference no longer replaces 0.0 with 0.3; only `None` defaults).
- `encode_feature_vector_v2`: **15 dims** = the 13 above + 2 posture channels from YOLOv8-pose. This is what the model consumes (fixture `shared/fixtures/encode15_v2.json`, generated from Python).

| # | Feature | Normalization |
|---|---------|---------------|
| 0 | gaze_x | [-1, 1] |
| 1 | gaze_y | [-1, 1], eyes down = negative |
| 2 | pitch_norm | relative pitch/90 → [-1, 1], head down = negative |
| 3 | yaw_norm | yaw/90 → [-1, 1] |
| 4 | ear | [0, 1] (gaze-corrected on the server) |
| 5 | perclos | /100 → [0, 1] |
| 6 | stability | /100 → [0, 1] |
| 7–10 | object one-hots | book, phone, laptop, none |
| 11 | fidget | /100 → [0, 1] |
| 12 | proximity_code | `hands_not_visible` 0.00, `hands_on_desk` 0.25, `holding_pen` 0.50, `holding_book` 0.75, `holding_phone` 1.00 |
| 13 | posture_upright | {0, 1} (lean angle < 15 deg; unknown = 1) |
| 14 | posture_lean_norm | lean angle/90 → [0, 1] (unknown = 0) |

Older 13-channel models (`behavior_classifier.onnx`, `_v2.onnx`) still load: the classifier reads the channel count from the ONNX input shape and feeds the first 13 channels.

---

## 6. Behavior Classifier (1D-CNN)

### 6.1 Model Architecture

```mermaid
flowchart LR
    IN["Input<br/>(B, 15, 15)"] --> C1["Conv1D 15→32<br/>k=3, pad=1<br/>+ BN + ReLU"]
    C1 --> C2["Conv1D 32→64<br/>k=3, pad=1<br/>+ BN + ReLU<br/>+ Dropout 0.2"]
    C2 --> C3["Conv1D 64→64<br/>k=3, pad=1<br/>+ BN + ReLU"]
    C3 --> GAP["AdaptiveAvgPool1d(1)<br/>(B, 64)"]
    GAP --> FC["Linear 64→6"]
    FC --> OUT["Logits (6 classes)"]
```

| Property | Value |
|----------|-------|
| Parameters | 20,742 (15 input channels; 20,550 for the older 13-channel models) |
| Input shape | `(B, 15, 15)` channel-first (`FEATURE_DIM` 15 × `WINDOW_SIZE` 15) |
| Output | 6-class logits |
| Receptive field | 7 frames = 1.4 s at the 5 Hz training cadence |
| Inference latency | not measured end to end (the CNN alone is ~0.02 ms; the per-stage pipeline benchmark in `evaluation/latency_benchmark.py` has never been run with real models/video) |
| ONNX opset | 17 |
| Weight init | Kaiming (conv), Xavier (linear) |

**Cadence.** The model is trained on windows sampled at **5 Hz** (`TrackSequenceEncoder`, 200 ms period, so 15 frames = 3 s). The server analyses `1 / COGNEN_PROCESS_EVERY_N` of the source frames (default **1**, i.e. every frame, which would make a 15 FPS stream a 1 s window), so **set `COGNEN_PROCESS_EVERY_N=3` (or `--process-every 3`, or `process_every_n: 3` in the start request) for a 15 FPS source** to match training. This mismatch is not enforced anywhere in code.

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
# Synthetic only (how the served v3 model was trained: circular, see §18)
python3 v2.0/training/train_classifier.py \
    --realistic --realistic-n-per-class 2000 --realistic-augment 3 \
    --penalty-loss --epochs 40 --patience 8 --batch 256 \
    --seed 42 --realistic-seed 42 --output v2.0/models/behavior_classifier_v3.pt

# With real datasets (NOT DONE: no real dataset has been downloaded or trained on)
python3 v2.0/training/train_classifier.py \
    --daisee /data/daisee --engagenet /data/engagenet \
    --penalty-loss --epochs 100
```

Real-data loaders sample frames at the **5 Hz** server cadence by timestamp (`training/datasets/sequence_encoder.py::TrackSequenceEncoder`, the same encoder path as inference), group windows by subject, and only produce the classes the data can ground: DAiSEE and EngageNet give `attentive_screen` / `distracted` / `drowsy` proxies, never phone, reading or absent (`GROUNDED_CLASSES`). Open Images MIDs for the object fine-tune are resolved from the official class-descriptions CSV, and `perception/study_objects.py` is the single class mapping shared with `ObjectDetector`.

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
| Validation split | each loader's own train/val split (never pooled and re-split); every window carries a group id (subject for real data, the base window incl. its augmented copies for synthetic) and the split is asserted group-disjoint (`assert_no_group_leakage`) |
| Model selection | best val macro-F1 (ties: lower val CE); `--seed` seeds python/numpy/torch and batch order |
| Gradient clipping | max_norm = 1.0 |

### 7.4 ONNX Export

```bash
python3 v2.0/training/export_onnx.py \
    --checkpoint v2.0/models/behavior_classifier_v3.pt \
    --output v2.0/models/behavior_classifier_v3.onnx \
    --verify
```

`export_onnx.py` checks ONNX/PyTorch logit parity and raises on a mismatch. Models in `v2.0/models/`: `behavior_classifier.onnx` (13 channels, random-synthetic Exp 1, val acc 0.213 = noise), `behavior_classifier_v2.onnx` (13 channels, synthetic), `behavior_classifier_v3.onnx` (15 channels, synthetic; **the server default**, `BEHAVIOR_CLASSIFIER_MODEL`). The `.onnx`/`.pt` files are tracked in git despite `.gitignore`.

---

## 8. Context-Gate Override

Post-processes CNN softmax probabilities with hard object-context rules.

> **Reachability.** The gate is live by default: `COGNEN_HAND_PROXIMITY` defaults to `pose` (YOLOv8-pose wrists vs the person's object boxes, `features/hand_proximity.py`), so `holding_phone` / `holding_book` can be produced; `hands` uses MediaPipe HandLandmarker (`features/hand_object.py`, falling back to wrists); `off` pins everyone to `hands_not_visible`, which keeps the gate from ever firing. The wrist thresholds are unvalidated on CCTV, `holding_pen` is never produced (COCO has no pen class), and the gate's effect has never been measured on real data. The "downward gaze" row is now reachable because pitch/gaze use the corrected down = negative measurement (§5), but the thresholds are uncalibrated. The gate also uses window-level evidence (phone in use in ≥ 60% of the window; book evidence ≥ 50% with no phone), and `absent` is returned when enough of the window had no detected face.

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

Frame-skip: `--process-every N` (CLI), `COGNEN_PROCESS_EVERY_N` or `process_every_n` in the start request analyses 1 in N frames; the default is **1**. At 15 FPS with N=3 → ~5 analysis FPS, which is the cadence the model was trained on (§6.1). Inference runs in a worker thread (`asyncio.to_thread`) so one room does not stall the event loop; frames of one room are still processed strictly in order.

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
| API server | FastAPI 0.141 + Starlette 1.6 + uvicorn (API-key auth, see §24) |
| Numerics | numpy 2.4, torch 2.13, onnxruntime 1.24, Python 3.10-3.12 |
| Person detection | YOLOv8-m (Ultralytics 8.4) |
| Object detection | YOLOv8-s |
| Pose estimation | YOLOv8-pose (17 COCO keypoints) |
| Face landmarks | MediaPipe FaceLandmarker (Tasks API) → 68-pt ibug + transformation matrix + blendshapes |
| Tracking | ByteTrack (boxmot 16) |
| Seat mapping | Shapely polygon IoU |
| Behavior classifier | 1D-CNN ONNX (20,742 params, 15 input channels) |
| Video input | OpenCV VideoCapture |
| Dashboard | FastAPI WebSocket @ 5 Hz |
| Training | PyTorch + ONNX export |

---

## 13. Repository Structure

```
v2.0/
├── server/
│   ├── main.py                  # FastAPI app, REST + WebSocket
│   ├── security.py              # API-key auth, rate limits, WS caps, headers, log redaction
│   ├── config.py                # Settings, model paths, thresholds
│   └── requirements.txt
├── input/
│   └── frame_source.py          # FrameSource ABC: RTSP, File, Camera
├── perception/
│   ├── person_detector.py       # YOLOv8-m
│   ├── object_detector.py       # YOLOv8-s
│   ├── pose_estimator.py        # YOLOv8-pose + posture analysis
│   ├── face_landmarks.py        # MediaPipe FaceLandmarker → 68-pt ibug + matrix + blendshapes
│   ├── hand_landmarks.py        # MediaPipe HandLandmarker (COGNEN_HAND_PROXIMITY=hands)
│   └── study_objects.py         # the one COCO id → study-object mapping
├── tracking/
│   ├── byte_tracker.py          # ByteTrack wrapper (boxmot)
│   └── seat_mapper.py           # Seat zone assignment (Shapely)
├── features/
│   ├── face_metrics.py          # Python port of v1.0 legacy scoring/ (EAR/MAR, trackers)
│   ├── dace_measurement.py      # DACE measurement port: matrix pitch, blendshape gaze, gaze-corrected openness
│   ├── track_features.py        # Per-track metrics state shared by server and training loaders
│   ├── hand_proximity.py        # Pose-wrist hand/object proximity (default)
│   ├── hand_object.py           # HandLandmarker IoU holding/near
│   └── feature_vector.py        # Canonical 13-dim (v1) and 15-dim (v2) encoders (see §23)
├── inference/
│   ├── behavior_classifier.py   # 15-frame rules/ONNX 1D-CNN + context-gate (shared ONNX session)
│   ├── contradiction_detector.py # Cross-Modal Contradiction Vector (see §22)
│   └── room_analyzer.py         # Main orchestrator
├── training/
│   ├── model.py                 # BehaviorClassifierCNN (20,742 params at 15 channels)
│   ├── train_classifier.py      # Training CLI (GroupShuffleSplit, see §23)
│   ├── penalty_matrix.py        # Complete 6×6 ConfusionPenaltyLoss matrix (see §23)
│   ├── export_onnx.py           # ONNX export + verification
│   ├── feature_extractor.py     # Thin shim over features/feature_vector.py
│   └── datasets/
│       ├── base_dataset.py      # build_windows() now tracks group_id (see §23)
│       ├── realistic_synthetic_loader.py  # synthetic generator (typed _ClassSpec, group ids)
│       ├── sequence_encoder.py   # TrackSequenceEncoder: 5 Hz timestamp sampling, same encoder as inference
│       ├── open_images_loader.py # Open Images boxes → YOLO labels (MIDs from the official CSV)
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
│   ├── latency_benchmark.py     # per-stage pipeline latency (SKIPPED without real models/video)
│   ├── synthetic_holdout_eval.py # independent-seed SYNTHETIC hold-out (circular)
│   └── reports/                 # behavior_classifier_v3_synthetic_note.md; archive/ = retracted 2026-03 outputs
├── models/
│   ├── behavior_classifier_v3.onnx   # served default (15 channels, synthetic-only); older .onnx/.pt kept
│   └── download_models.py
├── dashboard/static/            # Required by server/main.py's StaticFiles mount
├── data/seat_maps/
│   └── room_default.json
├── tests/                       # pytest — see §25 for the lightweight-tier test list
├── pyproject.toml               # ruff + mypy config (see §25)
└── run_analyzer.py              # CLI entry point
```

---

## 14. Setup & Installation

**Requirements:** Python **3.10-3.12** (boxmot 16 does not support 3.13+). The pins moved to a numpy 2.x stack: torch 2.13.0, torchvision 0.28.0, numpy 2.4.6, boxmot 16.0.11, ultralytics 8.4.146, opencv-python 4.13.0.92, onnxruntime 1.24.4, shapely 2.1.2, mediapipe 0.10.31, onnx 1.22.0, Pillow 12.3.0, scikit-learn 1.7.2, pytest 9.0.3, fastapi 0.141.1, starlette 1.6.0, pydantic 2.13.5. Resolution was verified for CPython 3.11 on Linux x86_64 (CI) and macOS arm64; **the heavy stack (YOLO inference, FaceLandmarker, ByteTrack, ONNX classifier) has not been smoke-tested end to end on real video.** The default Linux `torch` wheel is a CUDA 13 build; on a CPU-only server install it from the CPU index (`pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu`) before the rest.

```bash
cd v2.0
python3 -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt

# GPU (NVIDIA Linux/Windows only):
# pip install onnxruntime-gpu==1.24.4

# Download YOLO weights and the MediaPipe .task models
python3 models/download_models.py
```

> `server/requirements.txt` is pinned with `==` (not `>=`) for reproducible installs. To upgrade a dependency: bump its pin, check resolution (`pip install --dry-run -r server/requirements.txt`), run `pip-audit` and the full test suite, then commit the bump on its own. CI's `v2-audit` resolves the whole file and blocks on any advisory (§25).

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

Environment variables (non-security): `COGNEN_PROCESS_EVERY_N` (default 1; use 3 at 15 FPS, see §6.1), `COGNEN_RTSP_TRANSPORT`, `COGNEN_HAND_PROXIMITY` (`pose` default / `hands` / `off`), `COGNEN_MAX_CONCURRENT_ROOMS` (default 20), `API_HOST` (default `127.0.0.1`), `API_PORT`. `COGNEN_CAMERA_BUFFER` is read but **not wired** (it has no effect).

API source policy (`input/frame_source.py::validate_api_source`) and all security env vars: see [§24 Security](#security). `rtsp_transport` must be `tcp` or `udp`. The CLI (`run_analyzer.py`) is not restricted.

---

## 17. Testing & Evaluation

```bash
# Full test suite (needs the full requirements.txt: torch, opencv, ultralytics, shapely, fastapi...)
python3 -m pytest tests/ -v          # heavy tier: 338 passed (local, last verified)

# Lightweight tier (numpy + pytest + scikit-learn + onnxruntime; what CI runs, see §25)
python3 -m pytest tests/ --ignore=tests/test_tracking.py --ignore=tests/test_v2_pipeline.py -v
# light tier: 268 passed / 43 skipped (the skips need torch/fastapi/opencv/...)

# Lint
ruff check .

# Evaluation against ARCHITECTURE.md targets (needs real data; see below)
python3 v2.0/evaluation/evaluate.py \
    --model v2.0/models/behavior_classifier_v3.onnx \
    --classroom /path/to/classroom/dataset \
    --seat_map v2.0/data/seat_maps/room_default.json
```

All of these are unit/integration tests with synthetic inputs and mocks. They include the cross-language contract tests (`test_cross_language_contract.py`, `test_feature_vector_v2_contract.py`, the CMCV fixture check in `test_contradiction_detector.py`), the pitch characterisation, hand-proximity, server-security and room-analyzer suites. Untested against real inputs: YOLO inference, FaceLandmarker on real faces, boxmot tracking quality, RTSP sources and any GPU path. `tests/test_v1_pipeline.py` tests the Python classifier's v1-style rules, not the v1.0 TypeScript.

`evaluate.py` evaluates the **deployed** path (rules + context gate), not just the raw ONNX model, and every report carries `data_source` (`real` / `synthetic` / `none`) and `claims_allowed` (true only for real held-out data). Tracking metrics are real MOTA (unclipped) and IDF1 (identity-based, not per-frame F1); the temporal-consistency coefficient needs person ids and is no longer vacuous; latency is a per-stage benchmark of the full pipeline; occupancy/absence latency is computed from events, not re-recorded every frame. Without `--classroom` data each section reports `"status": "SKIPPED"` and `data_source: "none"`. `evaluation/synthetic_holdout_eval.py` runs an independent-seed **synthetic** hold-out, which is circular by construction.

> **History**: the earlier `evaluate.py` fabricated results (synthetic "80% correct" predictions, noise-injected ground truth, a CNN-only latency). Those outputs are retracted and archived in [`evaluation/reports/archive/`](evaluation/reports/archive/README.md); nothing in that folder is a valid measurement.

---

## 18. Performance Metrics

> **Nothing here has been measured on real data.** There is no real-data evaluation of any CognEn model, no real MOTA/IDF1, no real occupancy accuracy and no end-to-end latency measurement. The only behaviour numbers that exist are on **synthetic** data drawn from the same hand-written generator the model was trained on, so they are circular and say nothing about classroom accuracy. See [README.md §7](../README.md#7-performance-metrics) for the project-wide banner.

What exists for `behavior_classifier_v3` (full detail and caveats in [`evaluation/reports/behavior_classifier_v3_synthetic_note.md`](evaluation/reports/behavior_classifier_v3_synthetic_note.md)):

| Check | Result | What it means |
|-------|--------|---------------|
| Independent-seed synthetic hold-out (clean and augmented, 6,000 windows each) | accuracy 1.000, macro-F1 1.000 | The generator's classes are trivially separable; **not** a generalisation test. An ablation shows the model does not even need the phone/book channels on synthetic data, which will not hold on real video |
| Reading/phone windows with pitch and `gaze_y` negated | recall 0.000 | The model depends on the down = negative convention, so it must be fed the corrected measurement, never the legacy +down pitch |
| ONNX/PyTorch parity | < 1e-4 | Export correctness only |

The published v3 checkpoint was selected with a validation split that leaked augmented copies of training windows (since fixed; a re-run with the fixed split gave identical hold-out results, so the shipped files were not replaced).

### Targets (not results)

| Metric | Target |
|--------|--------|
| Behaviour accuracy / macro-F1 (real held-out, subject-disjoint) | ≥ 0.85 / ≥ 0.80 |
| Precision (using_phone) / recall (attentive_reading) | ≥ 0.90 / ≥ 0.85 |
| Temporal consistency | ≥ 0.90 |
| Tracking MOTA / IDF1 | ≥ 0.75 / ≥ 0.80 |
| Seat occupancy accuracy | ≥ 0.95 |
| End-to-end latency | < 100 ms/frame (GPU) |
| Frame rate | 15 FPS source, ~5 analysis FPS (`process_every_n = 3`) |
| GPU memory | < 4 GB |

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

v2.0 reuses hand-ported copies of v1.0's *legacy* per-person math (not the DACE path, not shared code; kept in step by hand, with generated-fixture contract tests as the automated checks: `encode13` and the CMCV fixture from the TypeScript, `encode15_v2` from Python, plus the pitch characterisation test; all drift-guarded in CI). See [v1.0 README](../v1.0/README.md) for browser deployment details.

| Component | v1.0 (Browser) | v2.0 (Server) |
|-----------|----------------|---------------|
| Face detection | face-api.js | YOLOv8-m + MediaPipe |
| Landmarks | face-api.js 68-pt (MediaPipe Face Landmarker in opt-in DACE mode) | MediaPipe FaceLandmarker → ibug 68-pt (`_MP_TO_68` rebuilt and geometry-tested) |
| EAR/MAR/Gaze/pitch | TypeScript (legacy formulas; DACE measurement opt-in) | Python port of legacy EAR/MAR plus the DACE measurement (always on) |
| Object detection | YOLOv8-nano ONNX.js | YOLOv8-s Ultralytics |
| Temporal classifier | Browser ONNX | Server ONNX |
| Tracking | Single person | ByteTrack multi-person |
| Seat mapping | N/A | Shapely polygons |
| Posture analysis | N/A | YOLOv8-pose keypoints |
| Deployment | Browser + Chrome Ext | FastAPI + GPU |

---

## 22. Cross-Modal Contradiction Vector (Novel)

Additive to the already-documented Context-Gate Override (§8): that mechanism is a rule that fires on *one* known confusion (phone vs. reading) and discards the disagreement magnitude once applied. The **Cross-Modal Contradiction Vector (CMCV)**, ported from v1.0 (v1.0 CMCV, not DACE mode) (see [v1.0 README §21](../v1.0/README.md#21-cross-modal-contradiction-vector-novel) for the full spec), is a general, symmetric, per-frame disagreement statistic over all modality pairs (gaze, posture-stability, ocular/PERCLOS, object-context, hand-action).

**Implementation**: `inference/contradiction_detector.py`, called from `RoomAnalyzer._analyze_person()` alongside the behavior classifier — computed per tracked person, using the same `metrics`, `person_objs`, and `hand_proximity` already assembled there, and exposed as `contradiction_index`/`contradiction_category` sibling fields on the per-person result dict (never folded into `attention_score`). It uses the corrected taxonomy (`fatigue_masking`, `prop_without_engagement`, `context_conflict`, `coherent`, `unclassified`) and receives the behaviour label, so an `attentive_reading` window counts as engagement with a visible book. The match with the TypeScript `v2` taxonomy is checked by the generated fixture `shared/fixtures/cmcv_cross_language.json` (`tests/test_contradiction_detector.py`; the fixture is regenerated and drift-guarded in the `v1-test` CI job). With `COGNEN_HAND_PROXIMITY=off` the hand-action modality always reads `hands_not_visible`; the default is `pose`.

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

- **Later round (all unit-tested only)**: the training loaders now use the same per-track measurement as the server (`features/track_features.py`) through `TrackSequenceEncoder`, sampled at 5 Hz by timestamp (DAiSEE at its native 30 fps previously made 15 frames about 0.5 s); no-face frames are encoded like inference instead of as `np.zeros`; PERCLOS, stability and fidget are no longer constant in the loaders; group ids are subject-level (DAiSEE by subject, EngageNet no longer by row offset) and synthetic windows share a group id with their augmented copies, with `assert_no_group_leakage` enforcing a disjoint split; real-data loaders only emit classes they can ground. The evaluation flaws listed in §17 (vacuous TCC, per-frame "IDF1", clipped MOTA, raw-ONNX-only scoring, CNN-only latency, 0-latency occupancy events) are fixed. Runtime: one shared ONNX session instead of one per tracked person, per-person state pruning, track id 0 handled, one-to-one seat matching, inference offloaded to a worker thread and background tasks tracked.

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
- **Type checking**: `mypy .` via `pyproject.toml` (`disallow_untyped_defs = false` — non-strict, since much of the codebase predates any type-checker; catches real errors in code that does have annotations without demanding annotations everywhere). First pass found 55 issues; genuine bugs and one real code-quality fix (see §23's `_ClassSpec` conversion) brought it to 26. That backlog (72 findings by the time it was fixed with only mypy installed, as CI runs it, and 109 with the full server stack installed, most of them in tests) has since been fixed properly, with no `# type: ignore` comments and no behaviour change: Optional values narrowed with explicit checks, Optional-typed locals for values set on only some branches, `Protocol`s for duck-typed parameters (`DropCounter` in `base_dataset.py`, `WebSocketSender` for the WebSocket send helpers in `server/main.py`), correct parameter and return types, `TYPE_CHECKING` imports for optional dependencies, and a documented `cast` where ultralytics' `Model.__call__` is annotated more widely than a plain call returns. `mypy .` is clean both with only mypy installed and with the heavy stack, and CI runs it as a **blocking** merge gate.
- **Test suite**: see §17 for the lightweight-vs-full test tier split. Mypy's first pass also caught a real regression: `evaluation/evaluate.py` still unpacked `ClassroomDataset` iteration as a 2-tuple after the train/val-leakage fix (§23) changed every loader to yield 3-tuples `(window, label, group_id)` — would have crashed the moment real classroom evaluation ran. Fixed.
- **Dependency pinning**: see §14.
- **`pip-audit`**: `v2-audit` resolves `server/requirements.txt` (full ML stack in a fresh venv) and **blocks** on any advisory (`V2_AUDIT_RESOLVE="true"`; set it to `"false"` only if the pins ever stop resolving together, which audits the pins as written without resolution). The old pin set was unresolvable (`ResolutionImpossible`), which is why the stack moved to numpy 2.x.
- **CI**: `.github/workflows/ci.yml` (shared file with v1.0, path-filtered; `workflow_dispatch` runs every job) runs `v2-lint` (`ruff check`), `v2-typecheck` (`mypy`, blocking), `v2-test` (the light pytest tier, installed with the production pins as constraints) and `v2-audit` on push/PR to `main` and `dev`. The heavy tier (torch, opencv, ultralytics, fastapi) is run locally, not in CI. A manual run on `dev` had all 12 jobs green. See [README.md](../README.md).
