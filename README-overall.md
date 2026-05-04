# CognEn — Cognitive Engagement Analyzer

> **Consolidated Master Documentation**
> CognEn is a real-time cognitive engagement monitoring system. It ships as two versioned sub-systems that share the same underlying Machine Learning core:
>
> - **[v1.0 (Browser Version)](v1.0/README.md)**: Single-student, webcam-based Next.js web application.
> - **[v2.0 (Server Version)](v2.0/README.md)**: Multi-person, CCTV-based Python FastAPI server.

---

## Table of Contents

- [Novel Algorithmic Contributions](#novel-algorithmic-contributions)
1. [System Comparison](#1-system-comparison)
2. [Key Features](#2-key-features)
3. [Architecture Diagrams](#3-architecture-diagrams)
4. [Machine Learning Core](#4-machine-learning-core)
5. [Technical Stack](#5-technical-stack)
6. [Repository Structure](#6-repository-structure)
7. [Performance Metrics](#7-performance-metrics)
8. [Build and Run — v1.0 (Browser)](#8-build-and-run--v10-browser)
9. [Build and Run — v2.0 (Server)](#9-build-and-run--v20-server)
10. [Testing & Evaluation](#10-testing--evaluation)
11. [Privacy & Security](#11-privacy--security)

---

## Novel Algorithmic Contributions

While this system utilizes existing state-of-the-art perception models (e.g., YOLOv8, MediaPipe) for foundational feature extraction, the core scientific contributions lie in the custom algorithmic pipelines and mathematical formulations designed specifically for cognitive engagement monitoring.

### 1. The Context-Gate Override Algorithm
A novel post-processing algorithm that intercepts the output of a neural network (the 1D-CNN temporal classifier) and modifies its probabilities based on deterministic spatial heuristics. It resolves behavioral ambiguities by mathematically compressing confidence scores when context is unclear, or artificially boosting/suppressing specific classes (e.g., prioritizing `using_phone` over `attentive_reading` when a phone object and hand interaction are detected in close proximity).

### 2. Asymmetric "ConfusionPenaltyLoss" Function
An asymmetric loss function algorithm designed to optimize the temporal classifier by minimizing high-cost pedagogical misclassifications. It applies custom mathematical penalties based on the real-world cost of a mistake—for instance, applying a `3.0×` penalty multiplier if the model predicts "reading" when the student is actually hiding a "phone", while applying standard `1.0×` multipliers for less critical errors.

### 3. Context-Aware Attention Scoring & Fusion Formula
A multimodal feature fusion algorithm that normalizes and combines 13–16 disparate signals (including Eye Aspect Ratio, Head Pose, Object One-Hots, and Proximity Codes). The algorithm applies specific domain-tuned weights to each feature (e.g., `gazeDeviation: 0.12`, `temporalPattern: 0.03`) and smooths the final 0–100 engagement score over time using an Exponential Moving Average (EMA) with a decay factor of `α=0.15`.

### 4. Prioritized Seat Mapping Assignment Algorithm (v2.0)
A specialized spatial algorithm that maps tracked bounding boxes to physical classroom seats in multi-person environments. It utilizes a three-step prioritization logic based on Shapely polygons: (1) Primary validation via centroid containment, (2) Fallback validation using Intersection over Union (IoU ≥ 0.30), and (3) A tie-breaker heuristic for overlapping detections.

### 5. Fatigue Onset Detection Algorithm (v1.0)
A rolling-window statistical algorithm for the early detection of cognitive fatigue. It employs Ordinary Least Squares (OLS) linear regression to calculate the degradation slope of the attention score. Anomalies are triggered based on a composite condition requiring a negative slope (`< -0.5`), high confidence (`R² > 0.4`), and an average score drop below a defined threshold (`< 60`).

---

## 1. System Comparison

| Feature                   | v1.0 (Browser / Extension)          | v2.0 (Server / CCTV)                      |
| ------------------------- | ----------------------------------- | ----------------------------------------- |
| **Primary Target**        | Single student learning at a desk   | Classroom, multi-student environment      |
| **Input Source**          | Webcam (640×480 @ 5 FPS)            | RTSP / video file / USB camera (1280×720) |
| **Tracking**              | Single person (first detected face) | Multi-person (ByteTrack MOT)              |
| **Person/Face Detection** | face-api.js (TinyFaceDetector)      | YOLOv8-m (Person) + MediaPipe (FaceMesh)  |
| **Object Detection**      | YOLOv8-nano ONNX (WebGPU/WASM)      | YOLOv8-s (PyTorch/GPU)                    |
| **Behavior Classifier**   | 1D-CNN ONNX (Browser runtime)       | 1D-CNN ONNX (Server runtime)              |
| **Output**                | Real-time Dashboard + PDF Report    | Room analytics WebSocket stream           |

---

## 2. Key Features

- **Context-Aware Attention Scoring:** Distinguishes phone use from book reading via YOLOv8 object detection and MediaPipe hand tracking. The uniform downward-gaze penalty is replaced with context-conditional scoring.
- **Temporal Behavior Classification:** 1D-CNN sliding-window classifier (15 frames / 3 s) determining 6 classes: `attentive_screen`, `attentive_reading`, `using_phone`, `distracted`, `drowsy`, `absent`.
- **Context-Gate Override:** Hard-rule post-processor that resolves phone/book ambiguity using object context when the model confidence is below a defined threshold.
- **Multi-Person Room Analytics (v2.0):** Per-seat occupancy tracking with Shapely polygon IoU, ByteTrack persistent IDs, and a room-level attention heatmap.
- **Flexible Video Input (v2.0):** RTSP CCTV streams, local video files, USB/built-in cameras via a unified `FrameSource` abstraction with configurable frame-skipping.
- **Chrome Extension (v1.0):** Manifest V3 extension that works inside Google Meet / Zoom / Teams via an offscreen document.

---

## 3. Architecture Diagrams

### 3.1 v1.0 — Browser Pipeline

```mermaid
flowchart LR
    A[Webcam\n640×480 @ 5 FPS] --> B[face-api.js\nTinyFaceDetector]
    B --> C[68-pt Landmarks]
    C --> D[FaceMetricsAggregator\nEAR · MAR · Gaze\nHead Pose · Blink\nFatigue · PERCLOS]

    A --> E[YOLOv8-nano ONNX\nevery 3rd frame]
    E --> F[Detected Objects\nbook · phone · laptop]

    A --> G[MediaPipe Hands\nevery 3rd frame]
    G --> H[Hand–Object\nProximity]

    D & F & H --> I[Feature Fusion\n13-dim vector]
    I --> J[TemporalBehaviorClassifier\n1D-CNN · 15-frame window]
    J --> K[Context-Gate\nOverride]
    K --> L[MLEngine\nContext-Aware Scoring\nEMA α=0.15]
    L --> M[Attention Score 0–100\nBehavior Label]
    M --> N[Dashboard\n+ Firebase + PDF]
```

### 3.2 v2.0 — Server Pipeline

```mermaid
flowchart TD
    SRC["Video Source\nRTSP · File · Camera\nFrameSource abstraction"] --> GRAB["Frame Grabber\nprocess every N frames"]
    GRAB --> P1[YOLOv8-m\nPerson Detection]
    GRAB --> P2[YOLOv8-s\nObject Detection]
    GRAB --> P3[YOLOv8-pose\n17 Keypoints]

    P1 & P2 & P3 --> FUSE[Detection Fusion\nassociate objects+pose\nwith person bboxes]
    FUSE --> BT[ByteTrack\npersistent track IDs]
    BT --> SM[SeatMapper\nShapely polygon IoU]

    SM --> LOOP["Per-Person Loop × N"]
    LOOP --> FL[Face Crop\n→ MediaPipe FaceMesh\n→ 68-pt Landmarks]
    LOOP --> FM[FaceMetrics\nEAR · PERCLOS · Gaze\nHead Pose · Fatigue]
    LOOP --> PA[PostureAnalysis\ntorso angle · head drop]
    LOOP --> OI[Object Interaction\nbook · phone proximity]

    FL & FM & PA & OI --> FF[Feature Fusion\n13-dim vector / frame]
    FF --> TC[TemporalClassifier\n1D-CNN · 15-frame window\n+ context-gate override]
    TC --> BL[Behavior Label\n+ Attention Score]

    BL --> RA[Room Analytics\nper-seat occupancy\nroom attention mean]
    RA --> WS[WebSocket\n5 Hz push\nto Dashboard]
```

---

## 4. Machine Learning Core

Both versions of the app share an identical mathematical structure for determining engagement.

### 4.1 Perception Models

- **YOLOv8 (Ultralytics):** Used across both versions for object detection, person detection, and pose estimation.
- **MediaPipe:** Used for Hand tracking (v1.0) and FaceMesh 478-point extraction (v2.0).
- **face-api.js:** Used exclusively in v1.0 for optimized browser-based face detection.

### 4.2 1D-CNN Temporal Classifier

A 3-layer Convolutional Neural Network (1D) evaluating a 15-frame sliding window to determine behavior.

- **Input**: `(Batch, 13-dim feature vector, 15 frames)`
- **Output**: Softmax probabilities for 6 behavior classes (`attentive_screen`, `attentive_reading`, `distracted`, `using_phone`, `drowsy`, `absent`).
- **Parameters**: ~20,550

### 4.3 Statistical Analytics

- **Exponential Moving Average (EMA):** Used in the `MLEngine` to smoothly transition the user's attention score over time (α=0.15).
- **Linear Regression (OLS):** Identifies macro-trends (e.g., calculating the slope of the attention score over a 5-minute period to determine fatigue onset).

---

## 5. Technical Stack

### v1.0 (Browser)

| Layer               | Technology                                             |
| ------------------- | ------------------------------------------------------ |
| Framework           | Next.js 15 (App Router, Turbopack)                     |
| Face detection      | face-api.js TinyFaceDetector                           |
| Landmarks           | face-api.js 68-point model                             |
| Object detection    | YOLOv8-nano ONNX via `onnxruntime-web` (WebGPU → WASM) |
| Hand tracking       | MediaPipe Hands Lite (`@mediapipe/hands`)              |
| Temporal classifier | Rule-based + ONNX 1D-CNN                               |
| Auth / Storage      | Firebase Auth + Firestore                              |
| AI report           | Google Genkit + Gemini (PDF generation)                |
| Extension           | Chrome MV3 + offscreen document                        |

### v2.0 (Server)

| Layer                 | Technology                                     |
| --------------------- | ---------------------------------------------- |
| API server            | FastAPI + uvicorn                              |
| Person detection      | YOLOv8-m                                       |
| Object detection      | YOLOv8-s                                       |
| Pose estimation       | YOLOv8-pose (17 COCO keypoints)                |
| Face landmarks        | MediaPipe FaceMesh → 68-point ibug mapping     |
| Multi-person tracking | ByteTrack (via `boxmot`)                       |
| Seat mapping          | Shapely polygon IoU                            |
| Behavior classifier   | 1D-CNN ONNX (20,550 params, 0.02 ms/inference) |
| Video input           | OpenCV `VideoCapture` — RTSP / file / camera   |
| Dashboard             | FastAPI WebSocket push at 5 Hz                 |

---

## 6. Repository Structure

```
CognEn-VCSir/
├── ARCHITECTURE.md              # System design source of truth
├── README.md                    # This consolidated master file
│
├── v1.0/                        # Next.js web app + Chrome Extension
│   ├── src/
│   │   ├── lib/
│   │   │   ├── types.ts             # Core types: ExtendedFaceMetrics, BehaviorResult…
│   │   │   ├── face-geometry.ts     # EAR/MAR, BlinkDetector, HeadPoseTracker,
│   │   │   │                        #   FatigueCalculator, HandObjectProximityDetector
│   │   │   ├── ml-engine.ts         # MLEngine: context-aware scoring, FEATURE_WEIGHTS
│   │   │   ├── object-detector.ts   # YOLOv8-nano ONNX wrapper (browser)
│   │   │   ├── hand-tracker.ts      # MediaPipe Hands wrapper (browser)
│   │   │   ├── temporal-classifier.ts # 1D-CNN sliding-window classifier (browser)
│   │   │   └── analytics.ts         # SessionAnalyticsEngine
│   │   ├── components/dashboard/    # UI components
│   │   └── ai/flows/                # Genkit flows (session summary, PDF report)
│   ├── extension/                   # Chrome Extension Source
│   └── public/                      # Static assets, models, and ONNX runtime
│
└── v2.0/                        # Python FastAPI server
    ├── server/
    │   ├── main.py                  # FastAPI app, REST + WebSocket endpoints
    │   ├── config.py                # Settings (env vars, model paths, thresholds)
    │   └── requirements.txt
    ├── input/
    │   └── frame_source.py          # FrameSource ABC: VideoFileSource, RTSPSource, CameraSource
    ├── perception/
    │   ├── person_detector.py       # YOLOv8-m
    │   ├── object_detector.py       # YOLOv8-s
    │   ├── pose_estimator.py        # YOLOv8-pose
    │   └── face_landmarks.py        # MediaPipe FaceMesh → 68-pt ibug mapping
    ├── tracking/
    │   ├── byte_tracker.py          # ByteTrack wrapper (boxmot)
    │   └── seat_mapper.py           # Seat zone assignment (Shapely IoU)
    ├── features/
    │   └── face_metrics.py          # Python port of v1.0 face-geometry.ts
    ├── inference/
    │   ├── behavior_classifier.py   # Temporal 1D-CNN + context-gate override
    │   └── room_analyzer.py         # Main orchestrator: run_source() pipeline loop
    ├── training/                    # 1D-CNN model architecture, CLI, datasets
    ├── evaluation/                  # Full evaluation suite (accuracy, F1, latency)
    ├── models/                      # YOLOv8 weights and behavior_classifier_v2.onnx
    ├── tests/                       # 61 tests (pytest)
    └── run_analyzer.py              # CLI entry point
```

---

## 7. Performance Metrics

Trained on 42,600 class-conditional synthetic windows. Evaluated on 3,000 independent test windows.

| Metric                     | Score        | Target   | Status |
| -------------------------- | ------------ | -------- | ------ |
| Behavior accuracy          | **1.000**    | ≥ 0.85   | ✓      |
| F1-macro                   | **1.000**    | ≥ 0.80   | ✓      |
| Precision (using_phone)    | **1.000**    | ≥ 0.90   | ✓      |
| Recall (attentive_reading) | **1.000**    | ≥ 0.85   | ✓      |
| Temporal consistency       | **1.000**    | ≥ 0.90   | ✓      |
| Tracking MOTA              | **1.000**    | ≥ 0.75   | ✓      |
| Tracking IDF1              | **1.000**    | ≥ 0.80   | ✓      |
| Seat occupancy accuracy    | **0.963**    | ≥ 0.95   | ✓      |
| v2.0 inference latency P95 | **< 0.1 ms** | < 100 ms | ✓      |

> **Note**: Behavior scores are on the realistic synthetic test set (seed=43). Real-classroom accuracy is estimated at 0.82–0.88 pending DAiSEE/EngageNet fine-tuning.

---

## 8. Build and Run — v1.0 (Browser)

1. **Install Dependencies:**

   ```bash
   cd v1.0
   npm install
   ```

2. **Configure Environment:** Create `.env.local` with your Firebase and Genkit keys.

3. **Set up Models:**

   ```bash
   bash scripts/setup-models.sh
   ```

4. **Start Dev Server:**
   ```bash
   npm run dev
   ```

For detailed Chrome Extension setup, view the [v1.0 README](v1.0/README.md).

---

## 9. Build and Run — v2.0 (Server)

1. **Setup Environment:**

   ```bash
   cd v2.0
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r server/requirements.txt
   ```

2. **Download YOLO Models:**

   ```bash
   python3 models/download_models.py
   ```

3. **Run Locally (CLI):**

   ```bash
   # Webcam
   python3 run_analyzer.py --source camera --device 0 --width 1280 --height 720 --show

   # RTSP Stream
   python3 run_analyzer.py --source rtsp --url "rtsp://admin:pass@192.168.1.100:554/ch1" --room room_default --show
   ```

4. **Run FastAPI Server:**
   ```bash
   uvicorn server.main:app --host 0.0.0.0 --port 8000
   ```

For full REST API and WebSocket documentation, view the [v2.0 README](v2.0/README.md).

---

## 10. Testing & Evaluation

### v2.0 — Python Tests

```bash
cd v2.0
python3 -m pytest tests/ -v
```

### v2.0 — Evaluation Suite

```bash
python3 v2.0/evaluation/evaluate.py --model v2.0/models/behavior_classifier_v2.onnx --seat_map v2.0/data/seat_maps/room_default.json
```

### v1.0 — Type Checks

```bash
cd v1.0
npm run typecheck && npm run lint
```

---

## 11. Privacy & Security

1. **No Raw Video Stored:** Only derived feature vectors and metrics are persisted.
2. **On-Premise:** All inference runs locally (in-browser for v1.0, on local GPU for v2.0).
3. **Anonymization:** Track IDs are used by default in v2.0.
4. **Data Retention:** Analytics auto-purge after 24 hours.
