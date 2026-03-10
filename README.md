# CognEn — Cognitive Engagement Analyzer

CognEn is a real-time cognitive engagement monitoring system. It ships as two versioned sub-systems that share the same ML core:

| | v1.0 | v2.0 |
|---|---|---|
| **Target** | Single student, webcam | Classroom, CCTV |
| **Deployment** | Browser / Chrome Extension | Python server (GPU) |
| **Input** | Webcam (640×480 @ 5 FPS) | RTSP / video file / USB camera |
| **Tracking** | Single person | Multi-person (ByteTrack) |
| **Output** | Attention score + PDF report | Room analytics dashboard (WebSocket) |

---

## Key Features

- **Context-aware attention scoring** — distinguishes phone use from book reading via YOLOv8-nano object detection and MediaPipe hand tracking. Uniform downward-gaze penalty replaced with context-conditional scoring.
- **Temporal behavior classification** — 1D-CNN sliding-window classifier (15 frames / 3 s) with 6 classes: `attentive_screen`, `attentive_reading`, `using_phone`, `distracted`, `drowsy`, `absent`.
- **Context-gate override** — hard rule post-processor that resolves phone/book ambiguity using object context when confidence is below threshold.
- **Multi-person room analytics** (v2.0) — per-seat occupancy tracking with Shapely polygon IoU, ByteTrack persistent IDs, room-level attention heatmap.
- **Flexible video input** (v2.0) — RTSP CCTV streams, local video files, USB/built-in cameras via a unified `FrameSource` abstraction. Configurable frame-skip for reduced CPU/GPU load.
- **Chrome Extension** (v1.0) — Manifest V3, works inside Google Meet / Zoom / Teams via offscreen document.

---

## Technical Stack

### v1.0 (Browser)
| Layer | Technology |
|---|---|
| Framework | Next.js 15 (App Router, Turbopack) |
| Face detection | face-api.js TinyFaceDetector |
| Landmarks | face-api.js 68-point model |
| Object detection | YOLOv8-nano ONNX via `onnxruntime-web` (WebGPU → WASM) |
| Hand tracking | MediaPipe Hands Lite (`@mediapipe/hands`) |
| Temporal classifier | Rule-based + ONNX 1D-CNN |
| Auth / Storage | Firebase Auth + Firestore |
| AI report | Google Genkit + Gemini (PDF generation) |
| Extension | Chrome MV3 + offscreen document |

### v2.0 (Server)
| Layer | Technology |
|---|---|
| API server | FastAPI + uvicorn |
| Person detection | YOLOv8-m |
| Object detection | YOLOv8-s |
| Pose estimation | YOLOv8-pose (17 COCO keypoints) |
| Face landmarks | MediaPipe FaceMesh → 68-point ibug mapping |
| Multi-person tracking | ByteTrack (via `boxmot`) |
| Seat mapping | Shapely polygon IoU |
| Behavior classifier | 1D-CNN ONNX (20,550 params, 0.02 ms/inference) |
| Video input | OpenCV `VideoCapture` — RTSP / file / camera |
| Dashboard | FastAPI WebSocket push at 5 Hz |

---

## Architecture Diagrams

### v1.0 — Browser Pipeline

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

### v2.0 — Server Pipeline

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

## Repository Structure

```
CognEn-VCSir/
├── ARCHITECTURE.md              # System design source of truth
├── README.md                    # This file
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
│   │   ├── components/dashboard/
│   │   │   └── webcam-panel.tsx     # Main capture loop + context module integration
│   │   └── ai/flows/                # Genkit flows (session summary, PDF report)
│   ├── extension/
│   │   ├── lib/ml-core.js           # JS port: MLEngineV2, TemporalBehaviorClassifier
│   │   ├── offscreen.js             # Offscreen document: webcam + face-api.js
│   │   ├── background.js            # Service worker
│   │   └── content.js               # Injects attention widget into Meet/Zoom/Teams
│   ├── public/
│   │   ├── models/                  # face-api.js model weights (.gitkeep → see setup)
│   │   └── onnx/                    # onnxruntime-web WASM binaries (.gitkeep → see setup)
│   └── scripts/
│       └── setup-models.sh          # Copies WASM from node_modules; prints YOLO instructions
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
    ├── training/
    │   ├── model.py                 # BehaviorClassifierCNN (20,550 params)
    │   ├── train_classifier.py      # Training CLI (--realistic, --penalty-loss)
    │   ├── export_onnx.py           # ONNX export + ORT verification
    │   ├── feature_extractor.py     # 13-dim feature vector (mirrors temporal-classifier.ts)
    │   └── datasets/
    │       ├── base_dataset.py          # Abstract base, build_windows()
    │       ├── realistic_synthetic_loader.py  # Class-conditional AR(1) synthetic data
    │       ├── daisee_loader.py         # DAiSEE engagement video loader
    │       ├── engagenet_loader.py      # EngageNet frame loader
    │       └── classroom_loader.py      # Custom CCTV annotation loader
    ├── evaluation/
    │   ├── evaluate.py              # Full evaluation runner → JSON report
    │   ├── behavior_metrics.py      # BehaviorEvaluator: accuracy, F1, temporal consistency
    │   ├── tracking_metrics.py      # TrackingEvaluator: MOTA, IDF1, ID switches
    │   ├── occupancy_metrics.py     # OccupancyEvaluator: accuracy, absence latency P95
    │   ├── latency_benchmark.py     # LatencyBenchmark context manager
    │   └── reports/                 # Experiment JSON reports + analysis markdown
    ├── models/
    │   ├── behavior_classifier_v2.onnx  # Production model (Exp 2, 100% on test set)
    │   └── download_models.py           # Download YOLOv8 weights from Ultralytics
    ├── data/seat_maps/
    │   └── room_default.json        # Example 6-seat room layout (normalised polygons)
    ├── tests/                       # 61 tests (pytest)
    │   ├── test_v1_pipeline.py
    │   ├── test_v2_pipeline.py
    │   ├── test_integration.py
    │   ├── test_behavior.py
    │   └── test_tracking.py
    └── run_analyzer.py              # CLI entry point — no FastAPI needed for local testing
```

---

## Performance Metrics (Experiment 2)

Trained on 42,600 class-conditional synthetic windows (biometric-grounded distributions, AR(1) temporal autocorrelation). Evaluated on 3,000 independent test windows.

| Metric | Score | Target | Status |
|---|---|---|---|
| Behavior accuracy | **1.000** | ≥ 0.85 | ✓ |
| F1-macro | **1.000** | ≥ 0.80 | ✓ |
| Precision (using_phone) | **1.000** | ≥ 0.90 | ✓ |
| Recall (attentive_reading) | **1.000** | ≥ 0.85 | ✓ |
| Temporal consistency | **1.000** | ≥ 0.90 | ✓ |
| Tracking MOTA | **1.000** | ≥ 0.75 | ✓ |
| Tracking IDF1 | **1.000** | ≥ 0.80 | ✓ |
| Seat occupancy accuracy | **0.963** | ≥ 0.95 | ✓ |
| v2.0 inference latency P95 | **< 0.1 ms** | < 100 ms | ✓ |

> **Note**: Behavior scores are on the realistic synthetic test set (seed=43). Real-classroom accuracy is estimated at 0.82–0.88 pending DAiSEE/EngageNet fine-tuning. See `v2.0/evaluation/reports/experiment_report_v2.md` for full analysis.

**Model**: `v2.0/models/behavior_classifier_v2.onnx`
**Architecture**: 3× Conv1D(BN+ReLU) → GlobalAvgPool → FC(6)
**Parameters**: 20,550 | **Input**: `(B, 13, 15)` channel-first | **Latency**: 0.02 ms CPU

---

## Build and Run — v1.0

### Prerequisites
- Node.js ≥ 20
- A Firebase project (for Auth + Firestore)
- A Google AI API key (for Genkit / Gemini PDF reports)

### 1. Install dependencies

```bash
cd v1.0
npm install
```

### 2. Set up environment variables

```bash
cp .env.example .env.local   # or create manually
```

Required variables:
```
NEXT_PUBLIC_FIREBASE_API_KEY=...
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=...
NEXT_PUBLIC_FIREBASE_PROJECT_ID=...
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=...
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=...
NEXT_PUBLIC_FIREBASE_APP_ID=...
GOOGLE_GENAI_API_KEY=...
```

### 3. Set up ONNX Runtime WASM binaries

```bash
bash scripts/setup-models.sh
```

This copies the `onnxruntime-web` WASM files from `node_modules` into `public/onnx/`.

### 4. Place the YOLOv8-nano ONNX model

Export a YOLOv8-nano model with the study-context classes:

```bash
# On a machine with Ultralytics installed:
python3 -c "
from ultralytics import YOLO
m = YOLO('yolov8n.pt')
m.export(format='onnx', imgsz=640, opset=17, simplify=True)
"
# Copy the output to:
cp yolov8n.onnx v1.0/public/models/yolov8n-objects.onnx
```

### 5. Start the dev server

```bash
cd v1.0
npm run dev           # Next.js on http://localhost:9002 (Turbopack)
```

### 6. Load the Chrome Extension (optional)

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** and select `v1.0/extension/`

### Available commands

```bash
npm run dev           # Dev server (Turbopack, port 9002)
npm run build         # Production build
npm run lint          # ESLint
npm run typecheck     # TypeScript type check (no emit)
npm run genkit:dev    # Genkit AI dev server
```

---

## Build and Run — v2.0

### Prerequisites
- Python ≥ 3.10
- CUDA-capable GPU recommended (CPU works but is slower)
- For RTSP streams: an IP camera or NVR with RTSP output, or VLC to stream a test video

### 1. Create virtual environment and install dependencies

```bash
cd v2.0
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r server/requirements.txt
```

> **GPU acceleration (NVIDIA, Linux/Windows only)**
> `requirements.txt` installs the CPU build of ONNX Runtime, which works on all platforms including Apple Silicon (CoreML is selected automatically). For NVIDIA GPU inference on Linux/Windows, install the GPU build after the base install:
> ```bash
> pip install onnxruntime-gpu>=1.18.0
> ```

### 2. Download YOLO model weights

```bash
python3 models/download_models.py
```

This downloads `yolov8m.pt`, `yolov8s.pt`, and `yolov8m-pose.pt` from Ultralytics into `v2.0/models/`.

The behavior classifier is already included at `models/behavior_classifier_v2.onnx`. Update `server/config.py` to point `BEHAVIOR_CLASSIFIER_MODEL` at it:

```python
# v2.0/server/config.py
BEHAVIOR_CLASSIFIER_MODEL = MODELS_DIR / "behavior_classifier_v2.onnx"
```

### 3. Configure a seat map (optional)

Edit `data/seat_maps/room_default.json`. Coordinates are normalised `[0, 1]` relative to frame width/height:

```json
{
  "seat_1": [[0.05, 0.10], [0.30, 0.10], [0.30, 0.90], [0.05, 0.90]],
  "seat_2": [[0.35, 0.10], [0.60, 0.10], [0.60, 0.90], [0.35, 0.90]],
  "seat_3": [[0.65, 0.10], [0.95, 0.10], [0.95, 0.90], [0.65, 0.90]]
}
```

### 4. Run locally — CLI (no server needed)

#### RTSP CCTV stream

```bash
python3 run_analyzer.py \
    --source rtsp \
    --url "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101" \
    --room room_default \
    --process-every 3 \
    --show
```

`--process-every 3` analyses 1 in 3 frames — on a 15 FPS stream this gives ~5 analysis FPS, reducing GPU load by 3×.
`--show` opens an annotation overlay window (requires a display).

#### RTSP with UDP transport (lower latency, requires stable LAN)

```bash
python3 run_analyzer.py \
    --source rtsp \
    --url "rtsp://192.168.1.100:554/live" \
    --rtsp-transport udp \
    --process-every 2
```

#### Local video file

```bash
python3 run_analyzer.py \
    --source file \
    --url /data/classroom_recording.mp4 \
    --room room_default \
    --loop                   # restart when file ends
```

#### Webcam / USB camera

```bash
python3 run_analyzer.py \
    --source camera \
    --device 0 \
    --width 1280 --height 720 \
    --process-every 1 \
    --show
```

#### Print JSON snapshots to stdout

```bash
python3 run_analyzer.py --source camera --print-json | jq '.room_attention'
```

### 5. Run as FastAPI server

```bash
cd v2.0
python3 -m server.main
# or:
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

#### Start a room via REST

```bash
# RTSP camera
curl -X POST "http://localhost:8000/rooms/room_a/start" \
     -H "Content-Type: application/json" \
     -d '{
           "source_type":    "rtsp",
           "url":            "rtsp://192.168.1.100:554/ch1",
           "process_every_n": 3,
           "rtsp_transport": "tcp"
         }'

# Video file
curl -X POST "http://localhost:8000/rooms/lab/start" \
     -H "Content-Type: application/json" \
     -d '{"source_type":"file","url":"/data/lecture.mp4"}'

# Webcam
curl -X POST "http://localhost:8000/rooms/demo/start" \
     -H "Content-Type: application/json" \
     -d '{"source_type":"camera","device":0}'
```

#### REST endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/rooms/{room_id}/start` | Start analysis with a `SourceConfig` body |
| `POST` | `/rooms/{room_id}/stop`  | Stop analysis |
| `GET`  | `/rooms/{room_id}/snapshot` | Latest analytics JSON |
| `GET`  | `/rooms` | List all rooms and running status |
| `WS`   | `/ws/room/{room_id}` | Real-time push at 5 Hz |

#### WebSocket snapshot format

```json
{
  "ts": 1741519810.4,
  "persons": [
    {
      "track_id": 1,
      "seat_id": "seat_2",
      "behavior": "attentive_screen",
      "attention_score": 82.5,
      "confidence": 0.94
    }
  ],
  "occupancy": {
    "occupied": ["seat_2", "seat_4"],
    "absent":   ["seat_1", "seat_3", "seat_5", "seat_6"],
    "total": 6
  },
  "room_attention": 82.5
}
```

### 6. Environment variables

All settings can be overridden without editing `config.py`:

| Variable | Default | Description |
|---|---|---|
| `COGNEN_PROCESS_EVERY_N` | `1` | Process 1 in N frames |
| `COGNEN_RTSP_TRANSPORT` | `tcp` | RTSP transport (`tcp`/`udp`) |
| `COGNEN_CAMERA_BUFFER` | `1` | OpenCV capture buffer size |
| `API_HOST` | `0.0.0.0` | FastAPI bind address |
| `API_PORT` | `8000` | FastAPI port |

Example:
```bash
COGNEN_PROCESS_EVERY_N=3 COGNEN_RTSP_TRANSPORT=tcp python3 -m server.main
```

---

## Testing

### v2.0 — Python tests (61 tests)

```bash
cd v2.0
python3 -m pytest tests/ -v
```

Run a specific suite:

```bash
python3 -m pytest tests/test_v1_pipeline.py -v    # v1.0 ML logic (Python port)
python3 -m pytest tests/test_v2_pipeline.py -v    # multi-person pipeline
python3 -m pytest tests/test_integration.py -v    # feature extractor + datasets
python3 -m pytest tests/test_behavior.py -v       # behavior classifier rules
python3 -m pytest tests/test_tracking.py -v       # SeatMapper
```

Skip tests that require PyTorch (e.g., in CI without GPU):

```bash
python3 -m pytest tests/ -v -k "not torch and not onnx_export"
```

### v2.0 — Evaluation suite

Run the full evaluation against ARCHITECTURE.md targets:

```bash
python3 v2.0/evaluation/evaluate.py \
    --model v2.0/models/behavior_classifier_v2.onnx \
    --seat_map v2.0/data/seat_maps/room_default.json
```

With a classroom dataset:

```bash
python3 v2.0/evaluation/evaluate.py \
    --model   v2.0/models/behavior_classifier_v2.onnx \
    --classroom /data/classroom \
    --seat_map  v2.0/data/seat_maps/room_default.json
```

Reports are written to `v2.0/evaluation/reports/eval_<timestamp>.json`.

### v2.0 — Retrain the behavior classifier

```bash
# Realistic synthetic data (class-conditional AR(1) windows)
python3 v2.0/training/train_classifier.py \
    --realistic \
    --realistic-n-per-class 2000 \
    --realistic-augment 3 \
    --penalty-loss \
    --epochs 100 \
    --batch  256 \
    --output v2.0/models/behavior_classifier_v2.pt

# With real datasets (when available)
python3 v2.0/training/train_classifier.py \
    --daisee    /data/daisee \
    --engagenet /data/engagenet \
    --penalty-loss \
    --epochs 100
```

### v1.0 — Web app type checks

```bash
cd v1.0
npm run typecheck
npm run lint
```

---

## RTSP Setup Tips

### Finding your camera's RTSP URL

Most IP cameras follow one of these patterns:

| Brand | URL pattern |
|---|---|
| Hikvision | `rtsp://user:pass@ip:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@ip:554/cam/realmonitor?channel=1&subtype=0` |
| Axis | `rtsp://user:pass@ip/axis-media/media.amp` |
| Generic ONVIF | `rtsp://user:pass@ip:554/onvif1` |

### Testing an RTSP stream with VLC

```bash
vlc "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101"
```

### Streaming a local video as RTSP for testing

```bash
# Requires ffmpeg
ffmpeg -re -i /path/to/video.mp4 -c copy -f rtsp rtsp://localhost:8554/test
```

Then connect CognEn to `rtsp://localhost:8554/test`.

### TCP vs UDP transport

- **TCP** (default): reliable, slightly higher latency (~50–200 ms). Use on WiFi or WAN.
- **UDP**: lower latency (~10–50 ms), packet loss causes artefacts. Use on wired LAN only.

---

## Privacy

1. **No raw video stored** — only derived feature vectors and scores are persisted.
2. **On-premise** — all inference runs locally; no cloud API calls for video data.
3. **Anonymisation** — track IDs are used by default; name mapping is opt-in.
4. **Retention** — analytics auto-purge after `DATA_RETENTION_HOURS` (default 24 h).
