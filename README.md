# CognEn — Cognitive Engagement Analyzer

> **Consolidated Master Documentation**
> CognEn is a real-time cognitive engagement monitoring system. It ships as two versioned sub-systems that share the same underlying Machine Learning core:
>
> - **[v1.0 (Browser Version)](v1.0/README.md)**: Single-student, webcam-based Next.js web application.
> - **[v2.0 (Server Version)](v2.0/README.md)**: Multi-person, CCTV-based Python FastAPI server.

---

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
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
12. [Complete Reproduction Specification](#12-complete-reproduction-specification)
13. [Peer Review Q&A — Academic Reference](#13-peer-review-qa--academic-reference)
14. [Recommended Research Directions & Paper Angles](#14-recommended-research-directions--paper-angles)
15. [CI/CD & Repository Hygiene](#15-cicd--repository-hygiene)

---

## Hardware Requirements

Full breakdown of what's needed to actually run each sub-system, from a bare-minimum dev setup to a production-realistic deployment. See [§12.1](#121-hardware-requirements--throughput) for measured/estimated throughput numbers at each tier.

### v1.0 — Browser (Web App + Chrome Extension)

Runs entirely client-side (webcam capture, face-api.js, ONNX Runtime Web, MediaPipe Hands all execute in the browser) — no GPU server needed. The Next.js app itself (dashboard, auth, Genkit AI report generation) needs a lightweight Node.js host, but the ML inference load is on the end user's device, not the server.

| Component | Minimum | Recommended |
|---|---|---|
| **CPU** | Dual-core, 2015+ (any modern laptop/desktop CPU) | Quad-core+, 2019+ |
| **GPU** | None required — WASM fallback for ONNX Runtime Web | Any GPU with WebGPU support (Chrome 113+) — object detection drops from ~50-80ms/frame (WASM) to ~15ms/frame (WebGPU) |
| **RAM** | 4 GB system RAM (browser tab uses ~150-250 MB: face-api.js models ~50 MB, ONNX runtime + YOLOv8-nano weights, MediaPipe Hands Lite) | 8 GB system RAM |
| **Webcam** | Any UVC-compliant webcam, 640×480 @ 5 FPS minimum | 720p+ for better landmark accuracy in low light |
| **Browser** | Chrome/Edge 90+ (WASM support) | Chrome/Edge 113+ (WebGPU support), for the extension specifically: Manifest V3 support (Chrome 88+) |
| **Network** | Broadband for initial model download (~10-15 MB: face-api.js weights + ONNX models via CDN/jsdelivr) — fully offline after first load except Genkit AI report generation, which needs connectivity to Google's Gemini API | Same |
| **Dev/build host (Next.js server, not the ML inference)** | 1 vCPU, 512 MB RAM (e.g. Firebase App Hosting's default `maxInstances: 1` tier) — this only serves pages and proxies Genkit calls, it does not run any ML inference | 2 vCPU, 1 GB RAM for concurrent multi-user report generation |

### v2.0 — Server (Multi-Person CCTV Analysis)

This is the resource-intensive side of the system: YOLOv8-m (person detection), YOLOv8-s (object detection), YOLOv8-pose, and MediaPipe FaceMesh all run per-frame on the server, fanned out per tracked person.

| Component | Minimum (CPU-only, degraded FPS) | Recommended (single room, GPU) | Multi-Room / Production |
|---|---|---|---|
| **CPU** | 4-core modern x86_64 (e.g. Intel i5-10th-gen / AMD Ryzen 5) — YOLOv8 on CPU is viable only at reduced resolution/frame-skip | 6-core+ (e.g. Intel i7 / AMD Ryzen 7) to keep the detection→tracking→per-person pipeline from CPU-bottlenecking the GPU | 8-core+ per concurrent room being processed (ByteTrack + per-person feature extraction is CPU-bound even with GPU-accelerated detection) |
| **GPU** | None (CPU-only via ONNX Runtime CPU provider — expect ~5-8 FPS effective throughput at 720p, well under the 15 FPS design target) | NVIDIA GPU, CUDA ≥ 8.6, ≥ 6 GB VRAM (e.g. RTX 3060) — meets the ARCHITECTURE.md target of ~35ms/frame (~28 FPS) | NVIDIA GPU with ≥ 8-12 GB VRAM per 1-2 concurrently-processed rooms (e.g. RTX 3080/4070 or better; A-series/L-series datacenter GPUs for multi-room server deployments) |
| **RAM** | 8 GB system RAM | 16 GB system RAM | 32 GB+ system RAM for multiple concurrent `RoomAnalyzer` instances (each tracked person maintains independent `BlinkDetector`/`HeadPoseTracker`/`FatigueCalculator` state — memory scales with room occupancy × number of concurrent rooms) |
| **VRAM** | N/A (CPU-only) | ≥ 6 GB (YOLOv8-m + YOLOv8-s + YOLOv8-pose + MediaPipe FaceMesh loaded concurrently) | ≥ 8-12 GB per room being processed simultaneously on the same GPU |
| **Storage** | ~5 GB (YOLOv8 weights: yolov8m.pt ~50 MB, yolov8s.pt ~22 MB, yolov8m-pose.pt ~50 MB, plus Python/CUDA/PyTorch/ultralytics dependency tree) | ~10 GB (add space for recorded footage if archiving CCTV input, training data, model checkpoints) | 100 GB+ if archiving classroom footage for later fine-tuning/audit; NVMe SSD recommended for RTSP stream buffering under multi-room load |
| **Network** | 100 Mbps+ for a single RTSP CCTV stream at 1280×720 | 1 Gbps+ if pulling multiple RTSP streams from networked cameras | 1 Gbps+ dedicated, plus sufficient upstream bandwidth for the WebSocket dashboard push (5 Hz, lightweight JSON payloads — not bandwidth-intensive itself) |
| **OS** | Linux (Ubuntu 20.04+) or macOS (Apple Silicon: CPU-only via ONNX Runtime, no CUDA) — Windows via WSL2 for CUDA support | Linux (Ubuntu 22.04+) with NVIDIA drivers + CUDA toolkit matching the `torch`/`onnxruntime-gpu` pinned versions (see `server/requirements.txt`) | Same, containerized (no official Dockerfile exists yet — see the "what else needs improvement" follow-up list) |

### Platform-Specific Guidance: Apple (macOS) & Windows

Both v1.0 and v2.0 run on Apple and Windows hardware today; the practical differences are in *which acceleration backend* v2.0 uses (CUDA is NVIDIA/Windows-or-Linux-only) and a few v1.0 browser/webcam-driver nuances. v1.0 has no OS-specific code path at all (it's a browser app) — the notes below for v1.0 are about the *host OS's* browser/webcam behavior, not the app itself.

#### Apple (macOS — Apple Silicon M1/M2/M3/M4, and Intel Macs)

| | v1.0 (Browser) | v2.0 (Server) |
|---|---|---|
| **Supported today** | Yes, fully — Chrome/Edge on macOS run face-api.js/ONNX Runtime Web/MediaPipe Hands identically to any other platform | Yes — CPU-only by default (no CUDA on any Mac), with an optional GPU-accelerated path on Apple Silicon (see below) |
| **GPU acceleration path** | WebGPU is supported in Chrome/Edge on macOS (Metal-backed) — same ~15ms/frame object-detection benefit as elsewhere | **Apple Silicon (M1 and later)**: PyTorch's Metal Performance Shaders (MPS) backend (`torch.device("mps")`) and `ultralytics`'s `device="mps"` option can accelerate YOLOv8 inference on the on-chip GPU/Neural Engine without CUDA. `onnxruntime`'s CoreML execution provider is a second option for the ONNX-based behavior classifier. **Neither is the pinned/tested default path** in `server/requirements.txt` (which assumes `onnxruntime`/CUDA-capable `torch`) — using MPS requires manually passing `device="mps"` where the code currently defaults to `"cuda" if available else "cpu"`, and is generally slower than an equivalent NVIDIA GPU for these workloads, though meaningfully faster than CPU-only. **Intel Macs**: no viable GPU acceleration path (AMD GPUs found in Intel Macs have no PyTorch/CUDA support) — CPU-only. |
| **Recommended Apple Silicon tier** | Any M1 or later (8 GB unified memory minimum) | M2 Pro/Max or M3/M4 Pro or better, 16 GB+ unified memory, for single-room dev/testing via MPS. Unified memory means system RAM and "VRAM" are the same pool — a 16 GB M-series Mac has no dedicated VRAM the way a discrete NVIDIA GPU does, so the "RAM" and "VRAM" figures in the v2.0 table above should be read as one shared budget on Apple Silicon, not additive. |
| **Webcam / camera access** | macOS's camera permission prompt applies per-browser (System Settings → Privacy & Security → Camera) — grant it to Chrome/Edge, not just the OS level | N/A for CCTV/RTSP sources; a Mac's built-in FaceTime camera works fine for the `--source camera` CLI mode via OpenCV's AVFoundation backend |
| **Realistic expectation** | Full v1.0 functionality, no caveats | Development/single-room testing is realistic on M-series Macs via MPS/CoreML; production multi-room CUDA-target throughput (§ table above) is **not achievable** on any Mac — plan to deploy v2.0's production tier on Linux/Windows with an NVIDIA GPU, using a Mac only for local development |

#### Windows

| | v1.0 (Browser) | v2.0 (Server) |
|---|---|---|
| **Supported today** | Yes, fully — Chrome/Edge on Windows are the most-tested combination (this is also the primary target for the Chrome extension's Meet/Zoom/Teams use case) | Yes — this is the primary non-Linux target for the CUDA-accelerated path, since NVIDIA GPUs are far more common on Windows desktops/laptops than on Macs |
| **GPU acceleration path** | WebGPU support in Chrome/Edge on Windows depends on the installed GPU driver (NVIDIA/AMD/Intel) being reasonably current — same fallback-to-WASM behavior as elsewhere if unsupported | **NVIDIA GPU (native Windows)**: `torch`/`onnxruntime-gpu` both ship native Windows wheels — install NVIDIA's Windows driver + matching CUDA toolkit version (see `server/requirements.txt`'s pinned `torch==2.2.0`/`onnxruntime==1.18.0` for the CUDA version they were built against) and run natively, no WSL2 required. **AMD GPU (native Windows)**: no CUDA equivalent — falls back to CPU-only (DirectML is a possible alternative backend for ONNX Runtime but is not the tested/pinned path in this repo). **WSL2**: an alternative for developers who want a Linux-like shell environment; modern WSL2 supports NVIDIA GPU passthrough for CUDA workloads, but adds a layer of driver/setup complexity most Windows-native NVIDIA users don't need — native Windows is the simpler recommended path when the GPU is NVIDIA. |
| **Recommended tier** | Same CPU/RAM tiers as the general v1.0 table above — no Windows-specific adjustment | Same as the "Recommended (single room, GPU)" / "Multi-Room / Production" tiers in the v2.0 table above, using an NVIDIA GPU natively on Windows |
| **Webcam / camera access** | Windows camera privacy settings (Settings → Privacy & Security → Camera) must allow desktop apps/browsers access | `--source camera` uses OpenCV's native Windows camera backend (DirectShow/Media Foundation) — works with any UVC webcam, same as other platforms |
| **Realistic expectation** | Full v1.0 functionality, no caveats | Full v2.0 functionality including the production GPU-accelerated throughput target, natively — Windows + NVIDIA GPU is hardware-equivalent to the Linux + NVIDIA GPU "Recommended"/"Production" path, not a degraded alternative to it |

### Overall System Notes

- **v1.0 and v2.0 have fundamentally different hardware profiles**: v1.0 pushes inference cost to the client (many cheap devices, no server GPU needed); v2.0 centralizes inference cost on one server per set of monitored rooms (one expensive GPU box, cheap client dashboards).
- **Platform summary**: v1.0 is platform-agnostic (any OS with a modern Chrome/Edge). v2.0's production-throughput path is NVIDIA-GPU-bound, which in practice means **Linux or Windows** for full production performance; **macOS (any Mac) can run v2.0 for development and single-room testing** (CPU-only on Intel Macs, MPS/CoreML-accelerated but still sub-CUDA-tier on Apple Silicon) but is not a production deployment target for the CUDA-tier throughput numbers in this document.
- **CI/build infrastructure** (`.github/workflows/ci.yml`, see [§15](#15-cicd--repository-hygiene)): GitHub Actions' standard `ubuntu-latest` runners (2 vCPU, 7 GB RAM) are sufficient for all CI jobs — the lightweight v2.0 test tier deliberately avoids installing torch/ultralytics/opencv in CI specifically to stay within standard runner resource/time limits.
- **The CPU-only path for v2.0 is real but degraded**: `onnxruntime` (CPU) and PyTorch CPU builds work without a GPU, but expect roughly 4-6× higher per-frame latency than the GPU-accelerated path, which will fall short of the 15 FPS / <100ms-per-frame design targets in `ARCHITECTURE.md` §7 — acceptable for development/testing, not for the production throughput target. This applies identically on Intel Mac, Apple Silicon (unless using MPS/CoreML), and Windows-without-a-supported-GPU.

---

## Novel Algorithmic Contributions

While this system utilizes existing state-of-the-art perception models (e.g., YOLOv8, MediaPipe) for foundational feature extraction, the core scientific contributions lie in the custom algorithmic pipelines and mathematical formulations designed specifically for cognitive engagement monitoring. These contributions resolve critical failure modes documented in state-of-the-art engagement monitoring literature.

> **Honesty banner for this section.** All items below are implemented in code and unit-tested; **none has been measured on real data**. Every "Empirical Proof" figure quoted for items 1-4 (+12.4% precision, +9.1% recall, +18.5% macro-F1, 0.742 to 0.931 temporal consistency, 96.3% occupancy) is a design **target / placeholder**, not a result (see §7). The DACE items (7-10) are likewise targets until measured on pilot set P1 / dataset D1.

### 1. The Context-Gate Override Algorithm
A novel post-processing algorithm that intercepts the output of a neural network (the 1D-CNN temporal classifier in v2.0; v1.0 has only a rule form, and on v2.0 it never fires by default because hand/object proximity is `hands_not_visible` unless `COGNEN_HAND_PROXIMITY=pose` is set, a wrist-based opt-in that is unvalidated on CCTV) and modifies its probabilities based on deterministic spatial heuristics. It resolves behavioral ambiguities by mathematically compressing confidence scores when context is unclear, or artificially boosting/suppressing specific classes (e.g., prioritizing `using_phone` over `attentive_reading` when a phone object and hand interaction are detected in close proximity).
**Empirical Proof:** Resolves the biometric ambiguity between visually homologous states, yielding a **+12.4% precision** improvement for `using_phone` and **+9.1% recall** for `attentive_reading`.

### 2. Asymmetric "ConfusionPenaltyLoss" Function
An asymmetric loss function algorithm designed to optimize the temporal classifier by minimizing high-cost pedagogical misclassifications. It applies custom mathematical penalties based on the real-world cost of a mistake—for instance, applying a `3.0×` penalty multiplier if the model predicts "reading" when the student is actually hiding a "phone", while applying standard `1.0×` multipliers for less critical errors.
**Empirical Proof:** Aligns the optimization objective with real-world pedagogical utility, achieving a **+18.5% improvement in macro-F1** over standard cross-entropy.

### 3. Context-Aware Attention Scoring + AR(1) Temporal Fusion
A multimodal feature fusion algorithm that normalizes and combines 13 disparate signals (including Eye Aspect Ratio, Head Pose, Object One-Hots, and Proximity Codes) under AR(1) temporal dynamics. The algorithm applies specific domain-tuned weights to each feature (e.g., `gazeDeviation: 0.12`, `temporalPattern: 0.03`) and smooths the final 0–100 engagement score over time using an Exponential Moving Average (EMA) with a decay factor of `α=0.15`.
**Target (unmeasured):** eliminate high-frequency state oscillation (temporal consistency 0.742 to 0.931 is a placeholder target). On the v1.0 legacy path this item was degraded by the pitch/gaze/detector defects; DACE (item 7) is what makes it functional, behind a flag.

### 4. Prioritized Seat Mapping Assignment Algorithm (v2.0)
A specialized spatial algorithm that maps tracked bounding boxes to physical classroom seats in multi-person environments. It utilizes a three-step prioritization logic based on Shapely polygons: (1) Primary validation via centroid containment, (2) Fallback validation using Intersection over Union (IoU ≥ 0.30), and (3) A tie-breaker heuristic for overlapping detections.
**Empirical Proof:** Decouples persistent physical identities from transient bounding-box IDs, resulting in **96.3% occupancy tracking accuracy** in dense environments.

### 5. Fatigue Onset Detection Algorithm (v1.0)
A rolling-window statistical algorithm for the early detection of cognitive fatigue. It employs Ordinary Least Squares (OLS) linear regression to calculate the degradation slope of the attention score. Anomalies are triggered based on a composite condition requiring a negative slope (`< -0.5`), high confidence (`R² > 0.4`), and an average score drop below a defined threshold (`< 60`).

### 6. Cross-Modal Contradiction Vector (CMCV) — New
A general, symmetric, per-frame disagreement statistic computed over **all** modality pairs (gaze, head-pose stability, ocular/PERCLOS state, object-context, hand-action), distinct from the Context-Gate Override above. Where the Context-Gate Override is a rule that fires on *one* known confusion (phone vs. reading) and discards the disagreement magnitude once applied, CMCV retains disagreement as a first-class signal in its own right, and classifies the *sign pattern* of disagreement into an actionable taxonomy: `fatigue_masking` (gaze looks attentive, posture/eyelids signal fatigue — the classic "eyes forced open" case), `prop_without_engagement` (a study object is visible but hands aren't on it — a gaming-the-system signal), `context_conflict` (surfaces the Context-Gate's territory as data rather than just clamping it), and `coherent`.
```
o_i(t) ∈ [0,1] per modality (reuses already-computed sub-scores — zero new sensing, zero new model inference)
d_ij(t) = |o_i(t) − o_j(t)|
CI(t) = weighted mean of d_ij, EMA-smoothed at α=0.35 (faster than the α=0.15 attention EMA)
```
Implemented in `v1.0/src/lib/contradiction-detector.ts` and ported to `v2.0/inference/contradiction_detector.py` (legacy path only). There is **no automated cross-language check** for CMCV; the two are kept in step by hand. **Empirical status**: computationally free and unit-tested against constructed synthetic conflict cases; full validation (precision/recall of the taxonomy against human-annotated ambiguity) requires labeled modality-conflict data that doesn't exist in DAiSEE/EngageNet — see [§14 Recommended Research Directions](#14-recommended-research-directions--paper-angles) below for how this could become a standalone publishable contribution.


### 7-10. DACE contributions (v1.0 only; implemented and unit-tested, opt-in, not browser-tested; see [ARCHITECTURE.md §11](ARCHITECTURE.md#11-dace-downward-attention-context-engine-v10-only) and [v1.0 README §24](v1.0/README.md#24-dace-downward-attention-context-engine--details))
DACE (Downward-Attention Context Engine) adds four contributions on top of items 1-6, bringing the total to **ten**. None is empirically validated; nothing has been run in a browser or against a camera (v1.0: 257 Vitest tests; v2.0 light tier 86 pass / 6 skip, neither involves DACE on real media). All accuracy, latency and FDR-DA figures are targets until measured on the pilot set (P1) and dataset (D1). DACE is off by default and does not change the legacy path. Items 1-6 are retained unchanged; in Stage 2 the Context-Gate may additionally be used as a model input rather than only a hard override, but it remains a named contribution.
7. **Context-conditioned looking-down penalty**: a downward head/eye posture is scored from gaze-corrected eye openness and desk-zone evidence (phone in use, writing, book, desk visibility) instead of a uniform pitch penalty.
8. **Calibrated abstention with reason codes**: an explicit "Unsure" output (U1-U5; rules in Stage 1, conformal prediction in Stage 2) that holds the engagement score and is reported separately, never as a penalty.
9. **Webcam eye-kinematics for reading while the head is pitched down** (Stage 2 feasibility study with a pre-registered go/no-go).
10. **FDR-DA metric and benchmark**: false-distraction rate on attentive reading/note-taking windows (share of reading/writing windows labelled distracted, drowsy or phone; computed by `v1.0/scripts/dace/metrics.ts`). Stage 1 targets, unmeasured: FDR-DA reduced by at least 30% relative to baseline with phone recall at least 0.80.

Supporting (not counted) additions: gaze-corrected eye openness/PERCLOS, hand-landmark writing detection, desk-visibility estimate. Item 9 (eye kinematics) is a Stage 2 go/no-go feasibility study and exists only as a stub. Unsure semantics: the score is held, the time is reported separately, never a penalty; reason codes U1-U5. Presets: A, A', A'', B, C, F, H (I = F on desk-out-of-frame segments). Config: `dace.enabled`, `measurement` (`legacy|pnp68|mediapipe`), `evidence`, `rules`, `abstain`, `detector`, `log`, set via `NEXT_PUBLIC_DACE_*` (`NEXT_PUBLIC_DACE_RESEARCH=1` for `/replay` and `?dace=` overrides). YOLOv8 is AGPL-3.0 (research use only, decision R2) versus the Apache-2.0 MediaPipe detector. The pitch defect DACE addresses can be reproduced with `python docs/proposals/evidence/pitch_sim.py` (needs numpy; simulated, not measured).

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
- **Context-Gate Override:** Hard-rule post-processor that resolves phone/book ambiguity using object context when the model confidence is below a defined threshold. (On v2.0 it is inactive by default: it needs `holding_*` proximity, which exists only with `COGNEN_HAND_PROXIMITY=pose`.)
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
    M --> N[Dashboard\n+ optional PDF report]
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

### 4.4 Dataset Specification & Preprocessing (Target Composition)

> ⚠ This section describes the **target** dataset composition, not one that
> has been assembled and trained on. See
> [`v2.0/training/datasets/`](v2.0/training/datasets/) for the current,
> real state: synthetic AR(1) generation is implemented and working; the
> DAiSEE and EngageNet loaders exist but have never been run against
> downloaded real corpora (no such corpora exist in this repo); the
> classroom loader exists but nothing currently produces its expected
> input format. Current training runs are synthetic-data-only.

The target hybrid dataset comprises **55,100 annotated temporal sequences** (15 frames per window; 3-second behavioral segments sampled at 5 Hz).

*   **Source & Characteristics:** A two-phase composition. Phase I: 42,600 class-conditional synthetic sequences using AR(1) autoregressive dynamics (ρ = 0.60–0.90 per feature) to simulate realistic sustained behavioral continuity based on biometric literature — this phase is implemented (`v2.0/training/datasets/realistic_synthetic_loader.py`). Phase II: 12,500 real-world annotated clips from the DAiSEE and EngageNet public corpora, intended to introduce variance in lighting, partial occlusion, and detection jitter — loaders exist but have not yet been run against real data, and neither corpus natively provides `using_phone`/`attentive_reading`/`absent` labels (no object-interaction annotations in either dataset), so a proxy-labeling or supplementary classroom-data strategy is required to cover those classes.
*   **Data Split (target):** Split by subject/clip (never by window, to prevent leakage from overlapping sliding windows of the same clip) — target ratio 70% train / 15% val / 15% test on real data, with synthetic data usable only for training.
*   **Preprocessing:** Raw frames convert into a 13-dimensional spatial feature vector. Gaze and yaw/pitch angles normalize to `[-1, 1]`. Biometrics (EAR, PERCLOS) clamp to `[0, 1]`. Object semantics map to one-hot vectors, and hand-object interactions encode as continuous proximity coefficients.
*   **Augmentation:** Stochastic augmentations during training include temporal time-warping (±20%), localized feature dropout (8%), Gaussian gaze jitter (σ = 0.05), and targeted object-context injection.

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
| Auth / Storage      | Firebase Auth (Firestore initialised but unused)       |
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
├── ARCHITECTURE.md              # System design source of truth (see §10 for post-upgrade status)
├── README.md                    # This consolidated master file
├── .github/workflows/ci.yml     # Shared CI pipeline for v1.0 + v2.0 (see §15)
│
├── v1.0/                        # Next.js web app + Chrome Extension
│   ├── src/
│   │   ├── lib/
│   │   │   ├── scoring/              # Single-source-of-truth scoring core (see v1.0 README §20)
│   │   │   │   ├── pose/head-pose.ts #   Corrected head-pose formula (shared normalizer)
│   │   │   │   ├── pose/gaze.ts      #   Gaze direction estimation
│   │   │   │   ├── trackers.ts       #   BlinkDetector, HeadPoseTracker, FatigueCalculator
│   │   │   │   ├── face-selection.ts #   PrimaryFaceSelector (multi-face handling)
│   │   │   │   └── __tests__/        #   Vitest unit + regression tests
│   │   │   ├── contradiction-detector.ts # Cross-Modal Contradiction Vector (new, see §6 above)
│   │   │   ├── types.ts             # Core types: ExtendedFaceMetrics, BehaviorResult…
│   │   │   ├── face-geometry.ts     # Re-export shim over scoring/
│   │   │   ├── ml-engine.ts         # MLEngine: context-aware scoring, FEATURE_WEIGHTS
│   │   │   ├── object-detector.ts   # YOLOv8-nano ONNX wrapper (browser)
│   │   │   ├── hand-tracker.ts      # MediaPipe Hands wrapper (browser)
│   │   │   ├── temporal-classifier.ts # 1D-CNN sliding-window classifier (browser)
│   │   │   └── analytics.ts         # SessionAnalyticsEngine
│   │   ├── components/dashboard/    # UI components
│   │   ├── app/error.tsx, global-error.tsx # React error boundaries
│   │   └── ai/flows/                # Genkit flows (session summary, PDF report)
│   ├── extension/                   # Chrome Extension Source (host_permissions narrowed)
│   ├── eslint.config.mjs            # ESLint flat config
│   ├── vitest.config.ts             # Vitest test runner config
│   ├── .env.example                 # Documents required env vars
│   └── public/                      # Static assets, models, and ONNX runtime
│
└── v2.0/                        # Python FastAPI server
    ├── server/
    │   ├── main.py                  # FastAPI app, REST + WebSocket endpoints (hardened, see §10.4 in ARCHITECTURE.md)
    │   ├── config.py                # Settings (env vars, model paths, thresholds, ROOM_ID_PATTERN)
    │   └── requirements.txt         # Pinned with == (see §15)
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
    │   ├── face_metrics.py          # Python port of v1.0 scoring/ (corrected head-pose, real gaze)
    │   └── feature_vector.py        # Canonical 13-dim feature encoder (new, see v2.0 README §23)
    ├── inference/
    │   ├── behavior_classifier.py   # Temporal 1D-CNN + context-gate override
    │   ├── contradiction_detector.py # Cross-Modal Contradiction Vector (new)
    │   └── room_analyzer.py         # Main orchestrator: run_source() pipeline loop
    ├── training/                    # 1D-CNN model architecture, CLI, datasets, penalty_matrix.py
    ├── evaluation/                  # Full evaluation suite (accuracy, F1, latency — no longer fabricates results)
    ├── models/                      # YOLOv8 weights and behavior_classifier_v2.onnx
    ├── pyproject.toml               # ruff + mypy config
    ├── tests/                       # pytest — see v2.0 README §17/§25 for the test tier split
    └── run_analyzer.py              # CLI entry point
```

---

## 7. Performance Metrics

> ⚠ **TARGET METRICS — not yet measured on real, held-out data.** The table
> below states design targets, not empirical results. No training run
> against a real, subject-disjoint test set has produced these numbers yet:
> there is no 55,100-sequence dataset, no DAiSEE/EngageNet integration run,
> and no committed evaluation log backing them. The only real artifacts
> produced so far are the synthetic-only smoke tests in
> [`v2.0/evaluation/reports/experiment_report.md`](v2.0/evaluation/reports/experiment_report.md)
> and [`experiment_report_v2.md`](v2.0/evaluation/reports/experiment_report_v2.md),
> which are trained and tested on the same synthetic AR(1) generator
> (train seed=42, test seed=43) — a circular evaluation, not a
> generalization test — and say so explicitly. `v2.0/evaluation/evaluate.py`
> previously fabricated numeric scores (`pred = label if rng.random() > 0.2
> else random_class`) when no real dataset was supplied; this has been
> fixed to report `"status": "SKIPPED"` instead of a synthesized score. See
> the training/evaluation pipeline plan for the work required to produce
> real numbers here.

### 7.1 Quantitative Evaluation (Targets)
The target dataset composition is 55,100 annotated temporal windows spanning both single-subject (proximal webcam, v1.0) and multi-subject (classroom CCTV, v2.0) environments, sharing the identical underlying 1D-CNN temporal classification core. These are the targets the architecture is designed to meet, not measured results:

| Metric | Target | Measured (synthetic smoke-test) |
|---|---|---|
| Behavior Accuracy | ≥ 0.85 | see experiment reports — circular eval, not comparable |
| F1-macro | ≥ 0.80 | see experiment reports — circular eval, not comparable |
| Precision (`using_phone`) | ≥ 0.90 | not yet measured |
| Recall (`attentive_reading`) | ≥ 0.85 | not yet measured |
| Temporal Consistency Coefficient | ≥ 0.90 | not yet measured |
| Tracking MOTA (v2.0) | ≥ 0.75 | not yet measured on real video |
| Tracking IDF1 (v2.0) | ≥ 0.80 | not yet measured on real video |
| Seat Occupancy Accuracy (v2.0) | ≥ 0.95 | no real-data evaluation path implemented yet |
| v2.0 inference latency P95 | < 100 ms | measured on synthetic input, see latency_benchmark.py |

### 7.2 Baseline Comparisons (Targets)
Two baseline paradigms the architecture is intended to beat, pending real evaluation:
1. **Naive Temporal Baseline:** a 1D-CNN trained without class-discriminating autoregressive signals, expected to perform near chance (~16.7% for 6 classes) — demonstrating that network topology alone is insufficient without structured temporal data.
2. **Deterministic Spatial Heuristic:** a non-parametric, rule-based frame-wise evaluator, expected to plateau well below the learned model, especially under occlusion/object-detector failures.

### 7.3 Ablation Study (Targets, not yet run)
Three ablations are planned for the v2.0 classifier (removing Context-Gate Override, the Asymmetric ConfusionPenaltyLoss, and AR(1) temporal autocorrelation) once a real, subject-disjoint test set exists; separate DACE presets A/A'/A''/B/C/F/H/I cover the v1.0 ablations. All percentages quoted for them are unmeasured targets. None of these have been run against real data; any percentage deltas quoted for them prior to that point are design targets, not results.

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
python3 -m pytest tests/ -v   # light tier: 86 pass / 6 skip; includes cross-language contract, hand-proximity and pitch-characterisation tests
```

### v2.0 — Evaluation Suite

```bash
python3 v2.0/evaluation/evaluate.py --model v2.0/models/behavior_classifier_v2.onnx --seat_map v2.0/data/seat_maps/room_default.json
```

### v1.0 — Type Checks

```bash
cd v1.0
npm run typecheck && npm run lint
npm test        # Vitest, 257 tests (legacy-equivalence golden fixtures + DACE suites; browser libs mocked)
```

---

## 11. Privacy & Security

1. **No Raw Video Stored:** Only derived feature vectors and metrics are persisted.
2. **On-Premise:** All inference runs locally (in-browser for v1.0, on local GPU for v2.0).
3. **Anonymization:** v2.0 uses track IDs only; no name mapping is implemented.
4. **Data Retention:** Auto-purge is **not implemented**: `DATA_RETENTION_HOURS` (24 h) exists in `v2.0/server/config.py` but nothing reads it. v1.0 DACE evaluation logs are manual-download only and have no retention limit yet.
5. **Cloud calls:** v1.0 report export sends derived session statistics (no frames) to Gemini via Genkit; nothing is written to Firestore.

---

## 12. Complete Reproduction Specification

To ensure rigorous reproducibility, the hardware, software, and hyperparameter configurations are detailed below.

### 12.1 Hardware Requirements & Throughput
| Component | v1.0 (Browser Edge) | v2.0 (Server / Centralized) |
|---|---|---|
| **GPU Acceleration** | Integrated (WebGPU / WASM) | Nvidia RTX 3060 (CUDA ≥ 8.6) |
| **System RAM** | 8 GB minimum | 16 GB minimum |
| **Video Input** | 640×480 @ 5 FPS | 1280×720 @ 30 FPS |
| **Pipeline Throughput** | ~125 ms/frame | ~35 ms/frame (~28 FPS) |
| **Isolated Classifier Latency** | ~0.02 ms per subject | ~0.02 ms per subject |

### 12.2 Training Hyperparameters (1D-CNN Temporal Classifier)
The 1D-CNN architecture (3 Conv1D layers + BatchNorm + Global Average Pooling + Linear, totaling ~20,550 parameters) was optimized using the following configuration:
*   **Optimizer:** `AdamW` with Weight Decay of `1 × 10⁻⁴`
*   **Learning Rate:** Base `1 × 10⁻³`, decayed via `CosineAnnealingLR` (Min `1 × 10⁻⁵`)
*   **Batch Size:** `64`
*   **Max Epochs:** 50 (Early Stopping Patience of 15 epochs)
*   **Loss Function:** Custom `ConfusionPenaltyLoss` (label smoothing `0.05`). 

### 12.3 Inference Pipeline Thresholds
*   **Context-Gate Neural Confidence Threshold (τ):** `0.80`
*   **Phone Override Clamp:** `p_phone ← max(p_phone, 0.85)`
*   **Reading Override Clamp:** `p_read ← max(p_read, 0.80)`
*   **Ambiguity Confidence Compression:** `confidence × 0.70`
*   **EMA Decay Factor (α):** `0.15`
*   **ByteTrack Seat IoU Assignment Threshold:** `≥ 0.30`


---

## 13. Peer Review Q&A — Academic Reference

> ⚠ **Read this before the answers below.** This section was originally written as if the numbers it quotes (86.5% accuracy, F1 0.852, the ablation deltas, etc.) were measured results. They are not — see [§7's honesty banner](#7-performance-metrics) for the full account: no real, subject-disjoint test set has been evaluated; `v2.0/evaluation/evaluate.py` was found to actively fabricate results when no real dataset was supplied (now fixed to report `SKIPPED` instead); and the one real training run that exists was evaluated circularly (trained and tested on the same synthetic generator). The Q&A below is kept **as a template for the answers a real evaluation would need to produce** — the questions themselves are exactly what a reviewer would ask, and the *structure* of each answer (what to report, at what precision, with what caveats) is sound. Treat every specific number below as a placeholder pending real evaluation, not as a citable result. Q12 (added below) covers what's needed to convert this from a template into real answers.

**Q1. What are the accuracy, precision, recall, and F1-score values for both single-student and multi-person setups?**
> *(Template answer — numbers below are targets, not measured results; see banner above.)*
Empirical evaluations across both single-subject (proximal webcam) and multi-subject (classroom CCTV) environments—which share the identical 1D-CNN temporal classification core—yielded the following metrics on an independent real-world hybrid test set (55,100 annotated temporal windows):
*   **Behavior Accuracy:** 0.865
*   **F1-macro:** 0.852
*   **Precision (`using_phone`):** 0.914
*   **Recall (`attentive_reading`):** 0.873
*   **Temporal Consistency Coefficient:** 0.931

**Q2. Which baseline methods were compared against the proposed system, and what are their performance metrics?**
> *(Template answer — targets, not measured; see banner above Q1.)*
The framework was benchmarked against two established baseline paradigms:
1.  **Naive Temporal Baseline:** A 1D-CNN trained exclusively on uniform temporal sequences devoid of class-discriminating autoregressive signals. This yielded a peak validation accuracy of just **21.3%** (marginally above 16.7% random chance), proving network topology alone is insufficient without structured temporal data.
2.  **Deterministic Spatial Heuristic:** A non-parametric, rule-based inference engine executing frame-wise spatial feature evaluation. This achieved an accuracy of **80.8%**, but suffered from severe classification degradation in complex spatial scenarios (e.g., partial occlusions, object-detector failures).

**Q3. What is the impact on overall performance when individual components (e.g., Context-Gate Override Algorithm, Asymmetric ConfusionPenaltyLoss Function) are removed or modified?**
> *(Template answer — targets, not measured. The three ablations described here are now fully specified and ready to run for real: `v2.0/evaluation/evaluate.py` no longer fabricates results, and `v2.0/training/penalty_matrix.py` provides the complete 30-pair matrix needed to run the "−ConfusionPenaltyLoss" condition meaningfully. See Q11 and §14.)*
A planned ablation study is intended to isolate the quantitative penalties of removing key algorithmic modules. **The percentages below are unmeasured targets, not results; the studies have not been run:**
*   **Context-Gate Override (target, unmeasured; the gate is inactive by default on v2.0):** Removing this module reintroduced severe biometric ambiguities between visually homologous states (e.g., reading a book vs. texting in lap), equating to a **12.4% penalty in `using_phone` precision** and a **9.1% penalty in `attentive_reading` recall**.
*   **Asymmetric ConfusionPenaltyLoss:** Reverting to standard cross-entropy removed the optimization of pedagogical edge-cases, leading to an **18.5% drop in macro-F1**.
*   **AR(1) Temporal Autocorrelation:** Removing realistic temporal variance forced the model to flicker between transient states, dropping temporal consistency from **0.931 to 0.742**.

**Q4. What is the size and detailed description of the dataset used for evaluation, including the number of samples and distribution across different classes?**
> *(Template answer — target composition, not an assembled dataset; see [§4.4](#44-dataset-specification--preprocessing-target-composition) for current real state.)*
The empirical validation utilized a hybrid dataset comprising **55,100 annotated temporal sequences** (15 frames per window; 3-second behavioral segments sampled at 5 Hz). The dataset is strictly balanced across 6 behavioral classes (`attentive_screen`, `attentive_reading`, `using_phone`, `distracted`, `drowsy`, `absent`).
*   **Training Split:** 42,600 heavily augmented sequences, sourced from a combination of class-conditional AR(1) autoregressive dynamics and real-world annotated clips from the DAiSEE and EngageNet corpora.
*   **Validation Split:** 1,800 unaugmented sequences formed a deterministic 15% holdout.
*   **Testing Split:** 3,000 unaugmented windows (500 per class) generated via an independent seed (`seed=43`) to strictly prevent data leakage.

**Q5. How do each of the proposed algorithms specifically improve the system's performance, and what are the specific challenges they address?**
*   **Context-Gate Override Algorithm (inactive by default on v2.0):** Addresses the challenge of biometric ambiguity (e.g., distinguishing reading from hidden device usage) by mathematically compressing or boosting probabilistic outputs based on deterministic spatial heuristics. 
*   **Asymmetric ConfusionPenaltyLoss:** Addresses the limitation of standard loss functions treating all errors equally. It improves pedagogical utility by applying severe algorithmic penalties to high-cost misclassifications (e.g., a `3.0×` penalty multiplier if the model predicts "reading" when the student is actually using a "phone").
*   **AR(1) Temporal Autocorrelation & EMA Smoothing:** Addresses the challenge of high-frequency state oscillation (flickering) by forcing the classification engine to learn sustained behavioral continuity rather than reacting to transient frame-level noise.
*   **Prioritized Seat Mapping (v2.0):** Addresses tracking identity fragmentation in dense environments by decoupling transient bounding-box IDs from persistent physical identities via Shapely polygon IoU validation.

**Q6. What are the hyperparameters used in the training process, including learning rate, batch size, and number of epochs?**
> *(These specific hyperparameters ARE real and implemented — verified in `v2.0/training/train_classifier.py`: AdamW, CosineAnnealingLR, batch 64, label smoothing 0.05, gradient clip 1.0, early-stopping patience are all live CLI defaults, not aspirational. What's still pending is a real training *run* against real, subject-disjoint data to produce results with these settings — the settings themselves are already correct.)*
The 1D-CNN temporal classifier was optimized using the following rigorous configuration:
*   **Optimizer:** `AdamW` with a Weight Decay of `1 × 10⁻⁴`
*   **Learning Rate:** Base rate of `1 × 10⁻³`, decayed via a `CosineAnnealingLR` scheduler to a minimum of `1 × 10⁻⁵`
*   **Batch Size:** `64`
*   **Max Epochs:** 50, strictly constrained by an Early Stopping Patience of 15 epochs to prevent overfitting.
*   **Loss Function:** Custom `ConfusionPenaltyLoss` with a label smoothing factor of `0.05`.

**Q7. What baseline methods did you compare your proposed method against?**
> *(Template answer — targets, not measured; see banner above Q1.)*
The proposed CognEn framework was benchmarked against two established baseline paradigms:
1.  **Naive Temporal 1D-CNN (Ablated Architecture):** A 1D-CNN with an identical network topology trained exclusively on uniformly-distributed temporal sequences, stripped of class-discriminating AR(1) autoregressive signals. This baseline achieved a peak validation accuracy of just **21.3%** — marginally above the 16.7% random-chance floor for a 6-class problem — demonstrating that network capacity alone is insufficient without structured temporal representation.
2.  **Deterministic Spatial Heuristic Engine:** A non-parametric, rule-based inference engine performing frame-wise evaluation of spatial features (head pose, gaze deviation, EAR thresholds) without any temporal modeling, object context, or learned classification. This achieved an overall accuracy of **80.8%**, but exhibited severe classification degradation under complex spatial conditions (partial occlusions, object-detector failures), particularly for visually homologous states such as `attentive_reading` vs. `using_phone`.

The proposed system surpasses both baselines with a test accuracy of **86.5%** and a macro-F1 of **0.852**, closing the gap beyond the rule-based ceiling (80.8%) through the novel algorithmic contributions listed above (the planned v2.0 ablations isolate three of the ten: Context-Gate Override, ConfusionPenaltyLoss and AR(1) temporal fusion; the DACE items 7-10 are covered by the presets in §7-10 above).

**Q8. Please provide a clear and concise problem statement.**
Existing automated student engagement monitoring systems fail to disambiguate visually homologous behavioral states — specifically, distinguishing a student reading a physical textbook from one covertly using a mobile device — because they rely on uniform spatial heuristics that apply identical penalties to any downward head pitch, regardless of contextual evidence. This misclassification is pedagogically costly: falsely penalizing attentive reading suppresses educator trust in automated analytics, while failing to detect phone usage undermines the system's utility as a classroom management tool. Furthermore, single-frame inference methods are inherently susceptible to transient noise and behavioral flickering; current systems lack the temporal depth to distinguish a momentary gaze shift from a sustained distraction episode, and existing loss functions treat all misclassification errors as equal-cost, misaligning model optimization with real-world pedagogical consequences.

This paper presents **CognEn**, a real-time cognitive engagement monitoring framework that resolves these limitations through novel algorithmic contributions (ten in total, see above), three of which are central to the v2.0 classifier: (1) a **Context-Gate Override Algorithm** for biometric ambiguity resolution, (2) an **Asymmetric ConfusionPenaltyLoss** function for pedagogically-weighted optimization, and (3) a **Multimodal Autoregressive Temporal Fusion** architecture for temporally consistent behavioral inference — deployed across both a single-student webcam environment (v1.0) and a multi-student CCTV classroom environment (v2.0).

**Q9. What is the name and size of the dataset used in your experiments? Please provide details on preprocessing, splits, and any other relevant information.**
> *(Template answer — the "CognEn Hybrid Behavioral Temporal Dataset" as described has not been assembled; see [§4.4](#44-dataset-specification--preprocessing-target-composition). The preprocessing/splitting **methodology** described below is sound and implemented — e.g. the split-by-subject-not-by-window fix in `train_classifier.py`'s `GroupShuffleSplit` — it just hasn't been run against 55,100 real sequences.)*
The empirical validation utilized the **CognEn Hybrid Behavioral Temporal Dataset**, comprising **55,100 annotated temporal sequences** (15 frames per window; 3-second behavioral segments sampled at 5 Hz), balanced across 6 behavioral classes (`attentive_screen`, `attentive_reading`, `using_phone`, `distracted`, `drowsy`, `absent`).

*   **Construction — Phase I (Synthetic, 42,600 sequences):** Class-conditional sequences generated using AR(1) autoregressive dynamics (autocorrelation coefficient ρ = 0.60–0.90 per feature channel) to simulate realistic behavioral continuity grounded in biometric literature.
*   **Construction — Phase II (Real-world, 12,500 sequences):** Annotated clips from the publicly available **DAiSEE** and **EngageNet** corpora, introducing variance in lighting, partial occlusion, and detection jitter to bridge the sim-to-real gap.
*   **Preprocessing:** Gaze deviation and head yaw/pitch angles normalized to `[−1, 1]`; biometric signals (EAR, PERCLOS) clamped to `[0, 1]`; object semantics mapped to one-hot vectors; hand-object interactions encoded as continuous proximity coefficients.
*   **Augmentation (Training only):** Temporal time-warping (±20%), localized feature dropout (8%), Gaussian gaze jitter (σ = 0.05), and targeted object-context injection.
*   **Splits (strict leakage prevention):**
    *   **Training:** 42,600 heavily augmented sequences (Phase I + II combined).
    *   **Validation:** 1,800 unaugmented sequences — a deterministic 15% holdout used exclusively for early stopping.
    *   **Test:** 3,000 unaugmented windows (500 per class) generated under an independent seed (`seed=43`), strictly disjoint from training.

**Q10. What evaluation protocol did you use (e.g., k-fold cross-validation, train/test split ratios, random seeds)?**
> *(Template answer describing the intended protocol — genuinely implemented pieces: the `GroupShuffleSplit`-based leakage-safe split, and a precisely-defined Temporal Consistency Coefficient in `evaluation/behavior_metrics.py::temporal_consistency()` that conditions on ground-truth stability, fixing a version that couldn't actually distinguish flicker from real transitions. Not yet run against real held-out data.)*
A fixed train/validation/test split protocol with strict seed isolation was employed to prevent data leakage. The training set (42,600 sequences) was optimized over up to 50 epochs with online stochastic augmentation using AdamW (lr = 1×10⁻³, weight decay = 1×10⁻⁴) and CosineAnnealingLR decay (min lr = 1×10⁻⁵), with early stopping at patience = 15 epochs. The validation holdout (1,800 sequences, deterministic) was used solely for early stopping — not for architecture selection. Final performance was reported once on the held-out test set (3,000 sequences, `seed=43`), evaluated using macro-averaged F1, per-class precision and recall (with emphasis on the pedagogically critical `using_phone` precision and `attentive_reading` recall pair), overall accuracy, and Temporal Consistency Coefficient over stable 10-second behavioral windows. For the ablation study, three conditions (−Context-Gate Override, −ConfusionPenaltyLoss, −AR(1) Autocorrelation) were each retrained from scratch under the identical split and seed with all other components held constant, isolating individual contribution magnitudes.

**Q11. What is the specific methodology detail missing from the paper?**
> ✅ **Update: item 1 below is RESOLVED.** The complete 6×6 penalty matrix is now implemented in `v2.0/training/penalty_matrix.py`, with every one of the 30 off-diagonal pairs assigned a documented, pedagogically-reasoned weight (verified by `tests/test_penalty_matrix.py` to have full coverage and intentional asymmetry). It has NOT yet been empirically validated — the weights are a design proposal derived from cost reasoning, not fit to labeled data — but the reproducibility gap this question originally identified (an incomplete, unreproducible matrix) is closed. Items 2 and 3 below remain open. See §14 for how to turn "the matrix exists" into "the matrix is validated."

The methodology detail that was missing (and is now completed as noted above) was the **complete formal specification of the `ConfusionPenaltyLoss` penalty weight matrix** across all class pairs. The paper originally declared only a single off-diagonal entry: a `3.0×` multiplier for the highest-cost misclassification (`attentive_reading` predicted when ground truth is `using_phone`). A 6-class classifier has 30 ordered off-diagonal pairs; the penalty weights for the remaining 29 pairs were not stated, making the loss function unreproducible and the claimed **+18.5% macro-F1 improvement** over standard cross-entropy unverifiable — that improvement figure is itself still an unmeasured target (see banner above Q1), independent of the matrix now being complete.

The three specific sub-details originally identified as missing:
1.  ✅ **RESOLVED** — **The full 6×6 asymmetric penalty weight matrix** `W[i][j]` is now fully specified in `training/penalty_matrix.py` for all 30 off-diagonal entries, each with a documented rationale (see [v2.0 README §7.2](v2.0/README.md#72-confusionpenaltyloss) for the summary table).
2.  **Still open** — **The closed-form loss equation** specifying how the penalty matrix integrates with cross-entropy and label smoothing (`ε = 0.05`) is implemented in code (`ConfusionPenaltyLoss.compute()` in `train_classifier.py`) but not written out formally here: `L = −Σᵢ W[yᵢ, ŷᵢ] · Σ_c y_smooth(c) · log p(c)`, where label smoothing is applied to the target distribution prior to penalty weighting.
3.  **Still open** — **A sensitivity analysis for the Context-Gate confidence threshold `τ = 0.80`** — the value is used as a fixed inference threshold but no ablation over candidate values (e.g., τ ∈ {0.65, 0.70, 0.75, 0.80, 0.85}) has been run to justify this specific operating point. This requires real evaluation data (§14).

**Q12. What changed since the original methodology write-up, and what would it take to make the claimed numbers real? (Added)**
Three categories of change:
1. **Fabrication removed, not just documented.** `v2.0/evaluation/evaluate.py` was found to actively synthesize fake predictions when no real dataset was supplied (`pred = label if rng.random() > 0.2 else random_class`), rather than merely lacking real data — a materially different problem (an evaluation harness that lies vs. one that's simply incomplete). Fixed: it now reports `"status": "SKIPPED"`.
2. **Real, verified pipeline bugs fixed independent of the "make numbers real" effort**: a live train/inference feature-encoding mismatch (gaze/pitch/yaw encoded as `[0,1]` in training-time feature extraction vs. `[-1,1]` at inference and in the synthetic generator — these would have silently taught a model the wrong decision boundary), a train/val leakage bug (naive random split let overlapping sliding windows from the same subject land on both sides), and a Temporal Consistency Coefficient that structurally could not distinguish flicker from real transitions (it never tracked ground truth). Full list: [v2.0 README §23](v2.0/README.md#23-training-pipeline-correctness-fixes).
3. **What's still required for real numbers**: acquiring DAiSEE (academic access request) and/or EngageNet, recording or licensing classroom footage (neither DAiSEE nor EngageNet has object-interaction annotations, so `using_phone`/`attentive_reading`/`absent` labels need a supplementary source), running the now-correct training pipeline against that real data with a subject-disjoint split, and running the three ablations for real. See §14 for a prioritized breakdown of this work as distinct research contributions rather than one monolithic "collect data" task.

---

## 14. Recommended Research Directions & Paper Angles

This section answers a different question than §13: not "what does the current documentation claim," but "if this codebase were the basis of an actual research paper, what are the distinct, defensible contributions worth writing up, and what does each one need before it's publishable." Ranked by how close each is to publishable today.

### 14.1 Closest to publishable: methodology/reproducibility paper on the evaluation-harness fabrication

**Angle**: a short methods/reproducibility-focused paper (or a workshop paper on ML evaluation pitfalls) documenting the discovery that `evaluate.py` silently fabricated benchmark numbers when real data was absent, and the broader pattern of circular evaluation (train and test on the same synthetic generator) that produced suspiciously perfect 1.000 scores across every metric. This is a genuinely interesting case study in how easy it is for an evaluation harness's synthetic-fallback path to become indistinguishable from a real result once the code around it stops being scrutinized — a real, verifiable incident (git history + the fix), not a hypothetical. **What it needs**: nothing further — this could be written up today from the diff and the honesty-banner work already done in this document.

### 14.2 Nearly there: real benchmark numbers for the existing 1D-CNN + Context-Gate + ConfusionPenaltyLoss system

**Angle**: the paper the existing README.md aspires to be — a systems paper reporting real accuracy/F1/precision/recall on a genuine held-out, subject-disjoint test set, with real ablations for Context-Gate Override, ConfusionPenaltyLoss (now fully specified, §7.2), and AR(1) temporal structure. **What it needs**: real data (DAiSEE access + EngageNet + supplementary classroom footage for phone/reading/absent coverage, since neither public dataset has object-interaction labels), then running the training pipeline (already correct — feature-encoding and leakage bugs are fixed) and the evaluation protocol (already correct — TCC is properly defined, ablation methodology is specified in §7.3) for real. This is a data-acquisition and compute problem, not an algorithm-design problem — the hard engineering work is done.

### 14.3 Genuinely novel, needs validation data: Cross-Modal Contradiction Vector (CMCV)

**Angle**: CMCV (item 6 under "Novel Algorithmic Contributions" above) is the strongest candidate for a *new* algorithmic contribution beyond what's already claimed — it's structurally different from the Context-Gate Override (general vs. single-pair-specific, retained as data vs. discarded after use) and has a concrete, testable taxonomy (`fatigue_masking`, `prop_without_engagement`, `context_conflict`). **What it needs**: this is the one contribution that needs *new* data collection specifically for it, since DAiSEE/EngageNet don't have modality-conflict annotations. Two paths: (a) a dedicated annotation pass where human raters flag ambiguous/conflicting segments in existing footage, or (b) a controlled synthetic-injection study — actors deliberately performing conflicting behaviors (e.g., holding a phone with eyes fixed on a screen vs. reading with a phone visible but untouched) — which is cheaper and faster to stand up than (a). Success metric should be precision/recall of the taxonomy categories against the human-flagged/scripted ground truth, not overall attention-score accuracy. If the taxonomy doesn't discriminate well, the fallback claim (scalar contradiction index correlating with classifier error rate) is still a legitimate, easier-to-validate result.

### 14.4 Not yet started, but well-scoped: decomposed uncertainty for teacher-facing confidence

**Angle**: a companion contribution to CMCV that was scoped in an earlier design pass but not implemented — decomposing total prediction uncertainty into named, attributable components (occlusion, tracking instability, semantic/context ambiguity via CMCV) using a noisy-OR combination, rather than a single opaque confidence score. The novelty argument: unlike generic calibration techniques (temperature scaling, MC-Dropout, deep ensembles), this needs zero extra forward passes and gives an interpretable "why is confidence low" breakdown, at the cost of not yet being empirically shown to be *better calibrated* (Expected Calibration Error) than the baseline. **What it needs**: implementation (straightforward — reuses `faceArea`/`poseVariance`, already computed, plus CMCV's output), a new `calibration_metrics.py` (doesn't exist yet), and — like CMCV — real labeled data to compute ECE against. Higher risk than CMCV: if ECE doesn't improve, the fallback claim ("attributable uncertainty decomposition for interpretability" rather than "calibrated confidence") is weaker and needs to be pre-registered as the acceptable outcome, not discovered post-hoc.

### 14.5 Room-level, v2.0-specific, unexplored: attention contagion across seats

**Angle**: v2.0's ByteTrack-tracked seat graph is currently treated as N independent per-person pipelines. A genuinely novel v2.0-specific contribution would model spatial/social correlation in attention state across neighboring seats (a distracted student affecting neighbors) via a graph-based or spatial-correlation model over the seat graph — explicitly *not* just averaging neighbor states, which wouldn't be novel. **What it needs**: this is the least-developed idea here — no design work has been done beyond noting it's plausible. Before investing in it, do a quick literature check for adjacent work in classroom social-network/peer-effect modeling to confirm the delta over existing approaches, since "graph neural network over spatial proximity" alone is not novel by itself.

### Priority for someone picking this up

If the goal is one real, defensible published contribution as fast as possible: **§14.1** (the fabrication case study) is publishable with zero further engineering — it's a documentation/write-up task. If the goal is to make the *system* paper (the one this README already reads like) real: **§14.2** is the critical path, gated entirely on data acquisition, not further code changes. **§14.3 (CMCV)** is the best next *novel-algorithm* candidate specifically because its data-collection need (a small, targeted human-annotation or synthetic-injection study) is far smaller in scope than acquiring a full training dataset.

---

## 15. CI/CD & Repository Hygiene

`.github/workflows/ci.yml` is a single workflow file, shared by both `v1.0` and `v2.0`, that runs on every push and pull request targeting `main`. Jobs are path-filtered (via a `changes` detection job using `dorny/paths-filter`) so a v1.0-only change doesn't waste time running v2.0 jobs and vice versa.

| Scope | Checks |
|---|---|
| v1.0 | `npm run typecheck`, `npm run lint` (ESLint, `eslint.config.mjs` — previously no config existed), `npx vitest run`, `npm run build` (previously would succeed even with type/lint errors due to now-removed `ignoreBuildErrors`/`ignoreDuringBuilds` flags), an **informational** `npm audit --audit-level=high` (`continue-on-error`; advisories appear daily), plus DACE guards in the test job: regenerate `shared/fixtures/encode13_cross_language.json`, `git diff --exit-code` on `shared/` and `docs/proposals/evidence/`, and `python docs/proposals/evidence/pitch_sim.py` (numpy only) |
| v2.0 | `ruff check .` (previously no lint config existed), `mypy .` (informational only — `continue-on-error: true`, given a pre-existing, individually-verified-benign backlog of ~26 findings from the first pass), the lightweight pytest tier (`numpy`/`pytest`/`scikit-learn` only — excludes torch/opencv/ultralytics-dependent tests), an **informational** `pip-audit --no-deps --disable-pip` on `requirements.txt` (the file is not currently resolvable, so it audits pins as written) |
| Repo-wide | Secret scanning via `gitleaks` |

The workflow sets top-level `permissions: contents: read, pull-requests: read` (required by paths-filter and gitleaks on PRs); the path filter treats `shared/**` and `docs/proposals/evidence/**` as both v1 and v2 changes. The workflow has not been re-run on GitHub since these changes. A final `all-checks-passed` job aggregates every job's result (treating a path-filtered-out/skipped job as passing) into one required status check, so branch protection only needs to reference a single job rather than every individual check.

**Dependency hygiene**: `v1.0`'s `npm audit` findings went from 107 (6 critical, 22 high) to 35 (0 critical, 0 high) via `npm audit fix`, targeted version bumps, and `package.json` `overrides` for transitive dependencies pulled in by `genkit`. The redundant `yarn.lock` — checked in alongside `package-lock.json` for the same `package.json`, causing GitHub's Dependabot to scan both and roughly double-count the same underlying issues — was removed. `v2.0`'s `server/requirements.txt` is now pinned with `==` rather than `>=` (every pin verified to exist on PyPI) for reproducible installs.
