# CognEn System Architecture — v1.0 Upgrade & v2.0 Design

> **Author**: Generated from deep codebase analysis of CognEn v1.0  
> **Scope**: Single-user webcam (v1.0) → Multi-person CCTV (v2.0)

---

## Table of Contents

1. [v1.0 Analysis — What Exists](#1-v10-analysis)
2. [v1.0 Upgrade — Study Behavior Understanding](#2-v10-upgrade--study-behavior-understanding)
3. [v2.0 Design — Multi-Person Room Analysis](#3-v20-design--multi-person-cctv-room-analysis)
4. [Model Choices & Justification](#4-model-choices--justification)
5. [Pseudo-Code for Key Algorithms](#5-pseudo-code-for-key-algorithms)
6. [Dataset Suggestions](#6-dataset-suggestions)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Implementation Folder Structure](#8-implementation-folder-structure)
9. [End-to-End Inference Pipeline](#9-end-to-end-inference-pipeline)
10. [Post-Upgrade: Correctness, Security & CI (Status Update)](#10-post-upgrade-correctness-security--ci-status-update)
11. [DACE: Downward-Attention Context Engine (v1.0 only)](#11-dace-downward-attention-context-engine-v10-only)

---

## 1. v1.0 Analysis

### Current Architecture

```
Webcam (640×480; both web-app pipelines analyse on a fixed 200 ms
        FixedTickScheduler tick, the "5 FPS" design rate, and a prediction is
        recorded once per second. Originally the legacy loop was
        requestAnimationFrame-bound; that is no longer true. Rates are design
        values, never measured)
  → face-api.js TinyFaceDetector (320px, ~150ms)
  → 68-point facial landmarks
  → FaceMetricsAggregator (12 features)
  → MLEngine (weighted scoring + EMA α=0.15)
  → Attention classification: Attentive / Distracted / Multitasking
```

### Existing Modules (Reuse Map)

| Module | File | Lines | Reuse in v1.0+ | Reuse in v2.0 |
|--------|------|-------|----------------|---------------|
| Face detection | `face-api.js` (TinyFaceDetector) | external | Kept as the legacy/default provider; flagged for replacement by MediaPipe Face Landmarker under decision R1 (DACE `measurement=mediapipe`, behind a flag, pending pilot results; §11) | ❌ Replace with YOLOv8 |
| Landmark detection | `face-api.js` (68-point) | external | Kept as the legacy/default provider; flagged for replacement by MediaPipe Face Landmarker under decision R1 (DACE `measurement=mediapipe`, behind a flag, pending pilot results; §11) | 🔄 Per-person crop |
| EAR / MAR / Head Pose | `face-geometry.ts` (607 lines) | L54-150 | ✅ Reuse directly | ✅ Reuse per-person |
| Blink detector | `face-geometry.ts` `BlinkDetector` | L155-243 | ✅ Reuse | ✅ Reuse per-person |
| Head pose tracker | `face-geometry.ts` `HeadPoseTracker` | L385-446 | ✅ Reuse | ✅ Reuse per-person |
| Fatigue calculator | `face-geometry.ts` `FatigueCalculator` | L449-498 | ✅ Reuse | ✅ Reuse per-person |
| Metrics aggregator | `face-geometry.ts` `FaceMetricsAggregator` | L502-592 | 🔄 Extend | 🔄 Extend per-person |
| ML Engine | `ml-engine.ts` `MLEngine` | L235-395 | 🔄 Redesign scoring | 🔄 Redesign scoring |
| Feature weights | `ml-engine.ts` FEATURE_WEIGHTS | L19-42 | 🔄 Add object/posture weights | 🔄 Add object/posture weights |
| **Scoring core (post-upgrade)** | `src/lib/scoring/` (new, see §10) | — | ✅ Single source of truth | Hand-ported to `features/face_metrics.py`; kept in sync manually (no automated cross-language check, see §10.1) |
| Session analytics | `analytics.ts` `SessionAnalyticsEngine` | L225-369 | ✅ Reuse | 🔄 Per-person analytics |
| Genkit AI flow | `analyze-facial-features.ts` | 51 lines | ❌ Remove (latency) | ❌ Remove |
| Chrome extension | `ml-core.js`, `offscreen.js`, etc. | ~1500 lines | ✅ Keep | N/A (server-based) |

### Key Limitations of v1.0 (status update: code for 1–4 exists, but 1–3 do not take effect in the shipped default path; 5 fixed separately)

1. **No object detection** — cannot distinguish book vs phone vs screen. **Code exists, model not shipped by default** (when the model is missing the web app now keeps no detector and scores with the LEGACY weights instead of the upgraded weights with empty context): the YOLOv8-nano wrapper is implemented (§2.1), but the model file (`*.onnx`) is gitignored and `scripts/setup-models.sh` does not download it, so a default install runs without object detection. COCO also has no `notebook`, `pen` or `tablet` class, so those §2.1 targets cannot be met with a COCO-trained detector.
2. **Looking down = distracted** — pitch penalty is uniform; reading a book is penalized. **Not yet addressed in the shipped default path.** Context-aware pitch scoring code exists (§2.2), but per the DACE diagnosis (§11) the legacy pitch formula rises rather than falls when the head tilts down, so the `pitch < -15` reading/phone branch does not run for typical downward tilts; (the former defect where an absent detector model still selected the upgraded weights with an empty object list is fixed: the legacy weights now apply). The interocular-normalizer change described in the note below did not fix the sign/scale problem. DACE (§11) is the proposed fix; it is behind a flag and has not been evaluated on real data.
3. **No body/hand tracking** — misses posture and hand-object interactions. **Code exists, partly effective**: MediaPipe Hands + `HandObjectProximityDetector` are implemented (§2.4), but the hand-object proximity signal depends on detected objects, which are absent by default (see item 1).
4. **Single-frame decisions** — EMA smoothing only; no true temporal modeling. **Partially addressed**: a rule-based 15-frame sliding-window classifier exists (§2.3); the trained 1D-CNN version described in README.md is not yet backed by a real trained model (see README.md §7's honesty banner).
5. **Single person only** — `detections[0]` hardcoded everywhere. **Fixed** (independent of the v1.0 upgrade path below): replaced with `PrimaryFaceSelector` in `v1.0/src/lib/scoring/face-selection.ts` — locks onto the largest, most-centered face with identity-continuity hysteresis. See "Shared Scoring Core & Correctness Fixes" note below.

> **Shared Scoring Core & Correctness Fixes (post-upgrade)**: beyond the upgrade path described in this section, a later pass found and fixed real bugs in the *existing* math this document describes as "reuse directly": the head-pose formula (§5.4 in both READMEs) used nose-chin distance to normalize pitch, which foreshortens as the head pitches and biased pitch against itself — now both yaw and pitch share the same normalizer (interocular distance). The web app, Chrome extension, and v2.0's Python port had also independently drifted to different thresholds/weights/formulas for the same signals; `v1.0/src/lib/scoring/` is now the single source of truth for the web app, with the extension and v2.0 kept as manually maintained copies (see §10.1 and [v1.0 README §20](v1.0/README.md#20-shared-scoring-core--cross-language-consistency) for the full account, including a real bug this uncovered in v2.0: it never computed gaze at all).

---

## 2. v1.0 Upgrade — Study Behavior Understanding

### Core Problem

v1.0 penalizes any downward gaze. A student reading a textbook gets a low score. The upgrade must distinguish:

| Behavior | Head Pitch | Gaze | Object Near Hands | Classification |
|----------|-----------|------|-------------------|----------------|
| Reading book/notebook | Down (−20° to −40°) | Downward, stable | Book/notebook detected | **Attentive** |
| Looking at phone | Down (−15° to −35°) | Downward, small saccades | Phone detected | **Distracted** |
| Watching screen/lecture | Neutral (±10°) | Centered | None / laptop | **Attentive** |
| Looking around | Variable | Variable | None | **Distracted** |
| Sleeping/drowsy | Down (−30°+) | Eyes closed (low EAR) | None | **Multitasking** |

### New Modules Added to v1.0

#### 2.1 Object Detection Module (NEW)

**Model**: YOLOv8-nano (via ONNX.js or TensorFlow.js)

**Why YOLOv8-nano**: 3.2M params, ~6ms inference on GPU, runs in browser via ONNX Runtime Web. Alternatives: MobileNet-SSD (less accurate), RT-DETR (heavier). YOLOv8-nano gives the best accuracy/speed tradeoff for browser deployment.

**Target classes**: `book`, `notebook`, `phone`, `laptop`, `pen`, `tablet`

**Integration**: Runs every 3rd frame (to save compute). Object bounding boxes are checked for proximity to hand/face landmarks.

#### 2.2 Context-Aware Pitch Scoring (MODIFIED)

Replace the uniform pitch penalty in `ml-engine.ts` with context-aware scoring:

```
if pitch < -15° AND object_near_hands == "book":
    pitch_score = 85  # Reading = attentive
elif pitch < -15° AND object_near_hands == "phone":
    pitch_score = 20  # Phone = distracted
elif pitch < -15° AND no_object:
    pitch_score = 40  # Unknown downward gaze
else:
    pitch_score = standard_pitch_scoring(pitch)
```

#### 2.3 Temporal Behavior Window (NEW)

Replace simple EMA with a sliding window classifier:

```
behavior_t = g(features[t-N:t])
```

**Implementation**: Lightweight 1D-CNN or GRU over the last 15 frames (3 seconds at 5 FPS). Input: `[gaze_x, gaze_y, pitch, yaw, ear, object_class, hand_proximity]` × 15 frames.

**Why not pure EMA**: EMA cannot distinguish "briefly looked down to check notes" from "sustained phone usage." A temporal window captures the behavioral pattern.

#### 2.4 Hand-Object Proximity (NEW)

Using MediaPipe Hands. The legacy path runs via `@mediapipe/hands` in the browser (that package has had no release since February 2023). DACE mode (§11) uses the maintained `@mediapipe/tasks-vision` `HandLandmarker` instead; the legacy package is kept until decision R1 and the pilot are complete.
- Detect hand keypoints (21 per hand)
- Compute distance between hand center and detected object bounding box
- Signal: `hand_object_proximity ∈ {holding_book, holding_phone, hands_on_desk, hands_not_visible}`
- On v2.0 this signal is produced by `COGNEN_HAND_PROXIMITY` (`pose` = default: YOLOv8-pose wrists vs object boxes in `features/hand_proximity.py`, unvalidated on CCTV, `holding_pen` never produced; `hands` = MediaPipe HandLandmarker in `features/hand_object.py`; `off` = `hands_not_visible` for everyone, which keeps the Context-Gate Override from ever firing). The gate is therefore live by default in v2.0; its effect has not been measured on real data.

### Modified Feature Weights (v1.0 Upgrade)

```javascript
const FEATURE_WEIGHTS_V1_UPGRADED = {
    // Existing (adjusted)
    gazeDeviation: 0.12,      // Reduced — context matters more now
    yawDeviation: 0.08,
    headPoseStability: 0.10,
    ear: 0.10,
    perclos: 0.08,
    fatigueIndex: 0.07,
    blinkRate: 0.05,
    fidgetScore: 0.04,
    faceDistance: 0.02,
    facePresence: 0.02,
    mouthOpenness: 0.02,
    // NEW features
    contextAwarePitch: 0.12,  // Pitch + object context
    objectContext: 0.10,      // What objects are present
    handObjectProximity: 0.05,// Hand-object interaction
    temporalPattern: 0.03,   // Temporal consistency bonus
};
```

### What Changes in v1.0 Code

| File | Change |
|------|--------|
| `face-geometry.ts` | Add `HandObjectProximityDetector` class |
| `ml-engine.ts` | Replace `pitchDeviation` scoring with context-aware version; add `objectContext`, `handObjectProximity`, `temporalPattern` features |
| `webcam-panel.tsx` | Add object detection inference alongside face detection |
| `types.ts` | Extend `ExtendedFaceMetrics` with `detectedObjects`, `handObjectProximity`, `studyContext` |
| NEW: `object-detector.ts` | YOLOv8-nano wrapper for browser |
| NEW: `hand-tracker.ts` | MediaPipe Hands wrapper |
| NEW: `temporal-classifier.ts` | Sliding window behavior classifier |
| `ml-core.js` (extension) | Mirror the same changes for Chrome extension |

---

## 3. v2.0 Design — Multi-Person CCTV Room Analysis

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                                      │
│  CCTV Feed (1080p / 720p, 15-30 FPS)                                │
│  → Frame Grabber (RTSP/HTTP stream)                                  │
│  → Resolution Normalization (1280×720)                               │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   PERCEPTION LAYER                                   │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Person Det.  │  │ Object Det.  │  │ Pose Est.    │               │
│  │ YOLOv8-m     │  │ YOLOv8-s     │  │ YOLOv8-pose  │               │
│  │ (people)     │  │ (book/phone) │  │ (17 keypts)  │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         ↓                 ↓                  ↓                       │
│  ┌──────────────────────────────────────────────────────┐           │
│  │              Unified Detection Fusion                 │           │
│  │  Associates objects + poses with person bounding boxes│           │
│  └──────────────────────────┬───────────────────────────┘           │
└─────────────────────────────┼───────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    TRACKING LAYER                                    │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ ByteTrack    │  │ Seat Map     │  │ Identity     │               │
│  │ (motion +    │  │ (fixed zones │  │ Registry     │               │
│  │  appearance) │  │  per desk)   │  │ (track→name) │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         └──────────────────┼──────────────────┘                      │
│                            ↓                                         │
│              Per-Person Track State                                   │
│              {track_id, seat_id, bbox, pose, objects}                 │
└─────────────────────────────┼───────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  FEATURE EXTRACTION LAYER                            │
│                                                                      │
│  For each tracked person:                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Face Crop    │  │ Posture      │  │ Object       │               │
│  │ → Landmarks  │  │ Analysis     │  │ Interaction  │               │
│  │ → EAR/Gaze   │  │ → Upright?   │  │ → Book/Phone │               │
│  │ → Head Pose  │  │ → Leaning?   │  │ → Hand prox. │               │
│  │ (REUSED v1)  │  │ → Slumped?   │  │              │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
│  Output: feature_vector[t] per person                                │
└─────────────────────────────┼───────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  BEHAVIOR INFERENCE LAYER                            │
│                                                                      │
│  ┌──────────────────────────────────────────────────────┐           │
│  │         Temporal Behavior Classifier                  │           │
│  │  Input: feature_vector[t-N:t] (sliding window)        │           │
│  │  Model: 1D-CNN + FC layers (lightweight)              │           │
│  │  Output: behavior_class + confidence                  │           │
│  │                                                       │           │
│  │  Classes:                                             │           │
│  │    • attentive_screen  (watching lecture/screen)       │           │
│  │    • attentive_reading (reading book/notes)           │           │
│  │    • using_phone       (phone distraction)            │           │
│  │    • distracted        (looking around, talking)      │           │
│  │    • drowsy            (sleeping, head down + low EAR)│           │
│  │    • absent            (seat empty)                   │           │
│  └──────────────────────────┬───────────────────────────┘           │
└─────────────────────────────┼───────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    ANALYTICS LAYER                                   │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Per-Person   │  │ Room-Level   │  │ Seat Occup.  │               │
│  │ Analytics    │  │ Dashboard    │  │ Tracker      │               │
│  │ (REUSED v1   │  │ • Heat map   │  │ • Present    │               │
│  │  analytics)  │  │ • Aggregate  │  │ • Absent     │               │
│  │              │  │ • Alerts     │  │ • Duration   │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### Seat Occupancy Detection

**Approach**: Pre-configured seat zones (polygon regions in frame). Occupancy = person bounding box IoU > 0.3 with seat zone.

```
seat_map = {
    "seat_1": polygon([(x1,y1), (x2,y2), ...]),
    "seat_2": polygon([...]),
    ...
}

for each tracked_person:
    for each seat in seat_map:
        if IoU(person.bbox, seat.polygon) > 0.3:
            seat.occupant = person.track_id
            seat.status = "occupied"

unmatched_seats → status = "absent"
```

### How v1.0 Scales to v2.0

The key insight is that **v1.0's per-person analysis pipeline is reused as-is inside v2.0**. v2.0 adds a multi-person wrapper:

```
v2.0 = multi_person_detection + tracking + [v1.0_per_person_pipeline × N] + room_analytics
```

Specifically:
- `FaceMetricsAggregator` — instantiated per tracked person
- `MLEngine` — instantiated per tracked person (separate EMA state)
- `SessionAnalyticsEngine` — instantiated per tracked person
- `BlinkDetector`, `HeadPoseTracker`, `FatigueCalculator` — per person

New in v2.0:
- Person detection + tracking (ByteTrack)
- Seat mapping + occupancy
- Room-level aggregation dashboard
- Server-side Python backend (not browser-based)

---

## 4. Model Choices & Justification

| Component | Model | Why Chosen | Alternatives | Compute |
|-----------|-------|-----------|-------------|---------|
| **Person detection** | YOLOv8-m | Best mAP/speed for medium scenes; handles occlusion well | RT-DETR (heavier, better for dense), Faster R-CNN (slower) | ~8ms/frame GPU |
| **Object detection** | YOLOv8-s | Lightweight; detects book/phone/laptop accurately after fine-tuning | MobileNet-SSD (less accurate), EfficientDet (slower) | ~4ms/frame GPU |
| **Pose estimation** | YOLOv8-pose | Unified with detection; 17 COCO keypoints; single-pass | MediaPipe (per-crop only), OpenPose (heavier, multi-person native) | ~6ms/frame GPU |
| **Tracking** | ByteTrack | SOTA in MOT; handles occlusion via low-confidence detections; no ReID needed | DeepSORT (needs ReID model, heavier), OC-SORT (similar perf) | ~1ms/frame CPU |
| **Face landmarks** | face-api.js 68-point (v1 reuse) **(design choice; superseded: v2.0 as built uses MediaPipe FaceLandmarker mapped to 68 points, see §10.7)** | Already proven in v1.0; runs on cropped face | MediaPipe FaceMesh (468 pts, overkill), dlib (C++ only) | ~15ms/crop GPU |
| **v1.0 face landmarks (DACE mode)** | MediaPipe Face Landmarker (`@mediapipe/tasks-vision`) | DACE (§11) adopts it for v1.0 behind a flag (decision R1): 478 points including iris, eye blendshapes and a facial transformation matrix. The 68-point model has no iris points and its head pose is not in degrees, which is the root of the pitch/gaze defects in §11. The "overkill" note above applies to the v2.0 row only | face-api.js 68-point (legacy default), solvePnP on 68 points (baseline A'') | Not measured (p95 ≤ 200 ms per tick is a target) |
| **Temporal classifier** | 1D-CNN (3 conv layers + FC) | Lightweight; captures local temporal patterns; trainable on small data | GRU/LSTM (heavier), Transformer (overkill for 15-frame window) | ~1ms/batch CPU |
| **v1.0 browser object det.** | YOLOv8-nano (ONNX.js) | 3.2M params; runs in browser; sufficient for single-user close-up | TF.js MobileNet (less accurate) | ~15ms/frame WebGPU |

### v2.0 Backend Choice: Python

v2.0 runs on a **server/edge GPU**, not in a browser. Reasons:
- CCTV feeds require RTSP stream processing (not available in browsers)
- Multi-person detection at 1080p needs GPU acceleration (PyTorch/ONNX Runtime GPU)
- Ultralytics YOLOv8 has a mature Python API
- ByteTrack reference implementation is Python

**Framework**: FastAPI server + Ultralytics + OpenCV + PyTorch

---

## 5. Pseudo-Code for Key Algorithms

### 5.1 Context-Aware Attention Scoring (v1.0 Upgrade)

```python
def score_attention(metrics, detected_objects, hand_proximity):
    # Standard v1.0 features
    base_score = weighted_sum(
        gaze_score(metrics.gaze_deviation),
        yaw_score(metrics.yaw),
        ear_score(metrics.ear),
        stability_score(metrics.head_pose_stability),
        fatigue_score(metrics.fatigue_index),
        ...
    )

    # NEW: Context-aware pitch scoring
    pitch = metrics.pitch
    nearest_object = get_nearest_object(detected_objects, hand_proximity)

    if pitch < -15:  # Looking downward
        if nearest_object in ["book", "notebook", "textbook"]:
            pitch_modifier = +15  # Reading = attentive
        elif nearest_object in ["phone", "smartphone"]:
            pitch_modifier = -25  # Phone = distracted
        elif metrics.ear < 0.22 and metrics.perclos > 20:
            pitch_modifier = -30  # Drowsy
        else:
            pitch_modifier = -10  # Unknown downward
    else:
        pitch_modifier = standard_pitch_score(pitch)

    final_score = base_score + pitch_modifier * PITCH_WEIGHT
    return clamp(ema.update(final_score), 0, 100)
```

### 5.2 Multi-Person Pipeline (v2.0)

```python
class RoomAnalyzer:
    def __init__(self, seat_map, num_seats):
        self.seat_map = seat_map  # Pre-configured seat polygons
        self.tracker = ByteTrack()
        self.person_states = {}   # track_id → PersonState
        self.detector = YOLOv8("yolov8m.pt")
        self.object_detector = YOLOv8("yolov8s_objects.pt")
        self.pose_estimator = YOLOv8("yolov8m-pose.pt")

    def process_frame(self, frame, timestamp):
        # 1. PERCEPTION
        person_dets = self.detector.predict(frame, classes=[0])  # person class
        object_dets = self.object_detector.predict(frame,
                            classes=[book, phone, laptop, pen])
        pose_results = self.pose_estimator.predict(frame)

        # 2. TRACKING
        tracks = self.tracker.update(person_dets, frame)

        # 3. PER-PERSON ANALYSIS
        results = []
        for track in tracks:
            person_id = track.track_id

            # Initialize state if new person
            if person_id not in self.person_states:
                self.person_states[person_id] = PersonState()
                # PersonState contains: FaceMetricsAggregator,
                # MLEngine, BlinkDetector, HeadPoseTracker (reused from v1.0)

            state = self.person_states[person_id]

            # Crop face from person bbox
            face_crop = extract_face_crop(frame, track.bbox)
            landmarks = detect_landmarks_68(face_crop)

            # v1.0 feature extraction (REUSED)
            metrics = state.aggregator.calculate_extended_metrics(
                landmarks, face_crop.width, face_crop.height, timestamp
            )

            # Associate objects with this person
            person_objects = associate_objects(track.bbox, object_dets)
            person_pose = associate_pose(track.bbox, pose_results)

            # Posture analysis from pose keypoints
            posture = analyze_posture(person_pose)

            # Context-aware scoring (v1.0 UPGRADED logic)
            behavior = state.temporal_classifier.classify(
                metrics, person_objects, posture, timestamp
            )

            # Seat assignment
            seat_id = assign_seat(track.bbox, self.seat_map)

            results.append({
                "track_id": person_id,
                "seat_id": seat_id,
                "behavior": behavior,  # attentive_screen, reading, phone, etc.
                "attention_score": behavior.score,
                "confidence": behavior.confidence,
            })

        # 4. SEAT OCCUPANCY
        occupied_seats = {r["seat_id"] for r in results if r["seat_id"]}
        absent_seats = set(self.seat_map.keys()) - occupied_seats

        return {
            "persons": results,
            "occupancy": {
                "occupied": list(occupied_seats),
                "absent": list(absent_seats),
                "total": len(self.seat_map),
            },
            "room_attention": mean([r["attention_score"] for r in results]),
        }
```

### 5.3 Temporal Behavior Classifier

> **Design sketch.** The implemented classifier is `v2.0/inference/behavior_classifier.py` (15-dim vector incl. posture, a shared ONNX session, context gate, `absent` handling, default model `behavior_classifier_v3.onnx`); see §10.7.

```python
class TemporalBehaviorClassifier:
    def __init__(self, window_size=15):
        self.window_size = window_size  # 15 frames = 3 sec at 5 FPS
        self.feature_buffer = deque(maxlen=window_size)
        self.model = load_1d_cnn("behavior_classifier.onnx")

    def classify(self, metrics, objects, posture, timestamp):
        # Build feature vector for this frame
        feature = [
            metrics.gaze_x, metrics.gaze_y,
            metrics.pitch, metrics.yaw,
            metrics.ear, metrics.perclos,
            metrics.head_pose_stability,
            encode_object(objects),   # one-hot: book, phone, laptop, none
            posture.is_upright,       # 0 or 1
            posture.lean_angle,
            metrics.fidget_score,
        ]

        self.feature_buffer.append(feature)

        if len(self.feature_buffer) < self.window_size:
            return fallback_single_frame_classification(metrics, objects)

        # Run temporal model
        window = np.array(self.feature_buffer)  # shape: (15, 11)
        prediction = self.model.predict(window)

        return BehaviorResult(
            label=CLASSES[prediction.argmax()],
            score=compute_attention_score(prediction),
            confidence=prediction.max(),
        )

# Classes
CLASSES = [
    "attentive_screen",   # Watching lecture/screen
    "attentive_reading",  # Reading book/notes
    "using_phone",        # Phone distraction
    "distracted",         # Looking around, talking
    "drowsy",             # Sleeping, head down
    "absent",             # Not at seat
]
```

**As implemented in v2.0** (`features/feature_vector.py`, FEATURE_VECTOR_VERSION 2): 15 channels per frame = `[gaze_x, gaze_y, pitch_norm, yaw_norm, ear, perclos, head_pose_stability, obj_book, obj_phone, obj_laptop, obj_none, fidget, hand_proximity_code, posture_upright, posture_lean_norm]`; the first 13 are identical to v1.0's `encode13` model input, posture comes from §5.4 (`lean_angle / 90`; unknown posture = (1, 0)). `pitch_norm` is relative pitch / 90 and `gaze_y` vertical gaze, both **down = negative**.

### 5.4 Posture Analysis from Pose Keypoints

```python
def analyze_posture(pose_keypoints):
    """
    pose_keypoints: 17 COCO keypoints (x, y, confidence)
    Indices: 0=nose, 5=l_shoulder, 6=r_shoulder, 11=l_hip, 12=r_hip
    """
    nose = pose_keypoints[0]
    l_shoulder = pose_keypoints[5]
    r_shoulder = pose_keypoints[6]
    l_hip = pose_keypoints[11]
    r_hip = pose_keypoints[12]

    # Shoulder midpoint
    shoulder_mid = midpoint(l_shoulder, r_shoulder)
    hip_mid = midpoint(l_hip, r_hip)

    # Torso angle from vertical
    torso_vector = (shoulder_mid.x - hip_mid.x, shoulder_mid.y - hip_mid.y)
    vertical = (0, -1)
    lean_angle = angle_between(torso_vector, vertical)

    is_upright = lean_angle < 15  # degrees

    # Head-to-shoulder ratio (head drop detection)
    head_drop = nose.y - shoulder_mid.y
    is_head_dropped = head_drop < 0  # Nose below shoulders = slumped

    return PostureResult(
        is_upright=is_upright,
        lean_angle=lean_angle,
        is_head_dropped=is_head_dropped
    )
```

---

## 6. Dataset Suggestions

| Dataset | Purpose | Size | Use |
|---------|---------|------|-----|
| **DAiSEE** | Engagement detection from video | 9K videos, 4 levels | Train temporal behavior classifier |
| **EngageNet** | Student engagement in online learning | 127K frames | Fine-tune attention scoring |
| **COCO** | Person detection + pose | 330K images | Pre-trained YOLOv8 base |
| **Open Images v7** | Object detection (book, phone, laptop) | 9M images | Fine-tune object detector |
| **EmoReact** | Facial expression in learning contexts | 1.1K videos | Validate micro-expression features |
| **Custom classroom data** | Real classroom CCTV footage | Collect 50-100 hours | Fine-tune all models for deployment |
| **MPII Human Pose** | Pose estimation | 25K images | Augment pose training |
| **MOT20** | Multi-object tracking benchmark | 8 sequences | Evaluate ByteTrack tuning |

### Data Collection Protocol for Custom Dataset

1. Record 720p CCTV footage in actual classrooms (with consent)
2. Annotate per-person: bounding box, behavior label, objects held
3. Minimum: 20 hours across 5 different rooms
4. Balance classes: ~equal representation of each behavior
5. Include edge cases: partial occlusion, poor lighting, crowded desks

---

## 7. Evaluation Metrics

### Per-Person Behavior Classification

| Metric | Target | Description |
|--------|--------|-------------|
| **Accuracy** | ≥ 85% | Overall correct classification |
| **F1-score (macro)** | ≥ 0.80 | Balanced across all 6 behavior classes |
| **Precision (phone)** | ≥ 0.90 | Must not classify reading as phone use |
| **Recall (attentive_reading)** | ≥ 0.85 | Must correctly identify book reading |
| **Temporal consistency** | ≥ 0.90 | Same behavior ≥ 90% of time in stable 10s windows |

### Tracking Quality

| Metric | Target | Description |
|--------|--------|-------------|
| **MOTA** | ≥ 75% | Multi-object tracking accuracy |
| **IDF1** | ≥ 80% | Identity preservation across frames |
| **ID switches** | < 5/min | Track identity stability |

### System Performance

| Metric | Target (v1.0) | Target (v2.0) |
|--------|--------------|--------------|
| **Latency** | < 200ms/frame | < 100ms/frame (GPU) |
| **FPS** | 5 FPS (browser) | 15 FPS (server) |
| **GPU memory** | N/A (CPU/WebGPU) | < 4 GB |
| **CPU usage** | < 30% | N/A (dedicated) |

### Seat Occupancy

| Metric | Target |
|--------|--------|
| **Occupancy accuracy** | ≥ 95% |
| **Absence detection latency** | < 30 seconds |

---

## 8. Implementation Folder Structure

```
CognEn-VCSir/
├── v1.0/                          # EXISTING — maintained with upgrades
│   ├── extension/                 # Chrome extension (KEPT)
│   │   ├── lib/
│   │   │   ├── face-api.min.js
│   │   │   ├── ml-core.js        # 🔄 MODIFIED: add context-aware scoring
│   │   │   └── object-detector.js # ✨ NEW: YOLOv8-nano ONNX wrapper
│   │   ├── background.js
│   │   ├── content.js
│   │   └── offscreen.js
│   ├── src/
│   │   ├── lib/
│   │   │   ├── face-geometry.ts   # ✅ REUSED
│   │   │   ├── ml-engine.ts      # 🔄 MODIFIED: context-aware scoring
│   │   │   ├── analytics.ts      # ✅ REUSED
│   │   │   ├── types.ts          # 🔄 MODIFIED: new fields
│   │   │   ├── object-detector.ts # ✨ NEW: YOLOv8-nano browser wrapper
│   │   │   ├── hand-tracker.ts    # ✨ NEW: MediaPipe Hands wrapper
│   │   │   ├── temporal-classifier.ts # ✨ NEW: sliding window classifier
│   │   │   ├── scoring/dace/      # DACE pure core (§11): no DOM, node-testable
│   │   │   └── dace/              # DACE browser/stateful code (§11)
│   │   │       ├── config.ts, presets.ts
│   │   │       ├── engine.ts, engine-factory.ts, rule-runner.ts, scheduler.ts, latency.ts
│   │   │       ├── eval-log.ts, self-test.ts, object-shuffle.ts, mediapipe-runtime.ts, eval/{metrics,cvat}.ts
│   │   │       ├── measurement/{provider,legacy-faceapi,pnp68,face-landmarker}.ts
│   │   │       ├── hands/hand-landmarker.ts
│   │   │       ├── detectors/{types,yolo-detector,mediapipe-object-detector}.ts
│   │   │       └── replay/frame-source.ts
│   │   ├── app/replay/            # DACE research-only replay page
│   │   └── components/dashboard/  # ✅ REUSED with minor updates
│   ├── scripts/
│   │   ├── setup-models.sh        # model/WASM setup (downloads + checksum verification)
│   │   ├── models.sha256          # pinned model + WASM checksums
│   │   └── dace/                  # replay-features.ts, metrics.ts, cvat-windows.ts
│   ├── public/mediapipe/wasm/     # gitignored copy of @mediapipe/tasks-vision WASM
│   └── models/                    # ✨ NEW: ONNX model files
│       ├── yolov8n-objects.onnx
│       └── behavior-classifier.onnx
│
├── docs/proposals/evidence/       # DACE evidence scripts: pitch_sim.py, aruco_pose.py
│
├── v2.0/                          # NEW — server-based multi-person system
│   ├── server/
│   │   ├── main.py               # FastAPI entry point
│   │   ├── config.py             # Room config, seat maps, thresholds
│   │   └── requirements.txt
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── person_detector.py    # YOLOv8-m person detection
│   │   ├── object_detector.py    # YOLOv8-s object detection
│   │   ├── pose_estimator.py     # YOLOv8-pose estimation
│   │   └── face_analyzer.py      # 68-point landmarks (REUSED v1.0 logic)
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── byte_tracker.py       # ByteTrack multi-person tracker
│   │   ├── seat_mapper.py        # Seat zone ↔ person association
│   │   └── identity_registry.py  # Track ID ↔ student name mapping
│   ├── features/
│   │   ├── __init__.py
│   │   ├── face_metrics.py       # PORTED from v1.0 face-geometry.ts
│   │   ├── posture_analyzer.py   # Pose keypoint → posture features
│   │   ├── object_interaction.py # Object-person association
│   │   └── feature_aggregator.py # Combine all features per person
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── behavior_classifier.py # Temporal 1D-CNN classifier
│   │   ├── attention_scorer.py    # Context-aware scoring (PORTED v1.0)
│   │   └── room_analyzer.py       # Main orchestrator
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── per_person.py         # PORTED from v1.0 analytics.ts
│   │   ├── room_dashboard.py     # Room-level aggregation
│   │   └── alerts.py             # Threshold-based alerts
│   ├── dashboard/                # Web dashboard (FastAPI + WebSocket)
│   │   ├── static/
│   │   ├── templates/
│   │   └── websocket_handler.py
│   ├── models/                   # Pre-trained model weights
│   │   ├── yolov8m.pt
│   │   ├── yolov8s-objects.pt
│   │   ├── yolov8m-pose.pt
│   │   └── behavior_classifier.onnx
│   ├── data/
│   │   ├── seat_maps/            # JSON seat zone configs per room
│   │   └── training/             # Training data for behavior classifier
│   └── tests/
│       ├── test_perception.py
│       ├── test_tracking.py
│       ├── test_behavior.py
│       └── test_integration.py
```

---

## 9. End-to-End Inference Pipeline

### v1.0 Upgraded Pipeline (Browser)

```
Webcam (640×480 @ 5 FPS)
  │
  ├─→ face-api.js TinyFaceDetector → 68 landmarks
  │     → FaceMetricsAggregator (REUSED)
  │       → EAR, MAR, gaze, head pose, blink, fatigue (REUSED)
  │
  ├─→ YOLOv8-nano (every 3rd frame) → object bboxes (NEW)
  │     → {book, phone, laptop, pen}
  │
  ├─→ MediaPipe Hands (every 3rd frame) → hand keypoints (NEW)
  │     → hand-object proximity
  │
  └─→ Feature Fusion
        → context_aware_pitch_score (NEW)
        → temporal_window[t-15:t] (NEW)
        → MLEngine.predict(extended_features)
        → EMA smoothing
        → Behavior: attentive_screen | attentive_reading | using_phone | distracted | drowsy
```

### v2.0 Pipeline (Server)

```
CCTV Stream (1280×720 @ 15 FPS)
  │
  ├─→ YOLOv8-m → person bounding boxes ──┐
  ├─→ YOLOv8-s → object bounding boxes ──┤
  ├─→ YOLOv8-pose → 17 keypoints ────────┤
  │                                        ↓
  │                               Detection Fusion
  │                          (associate objects + pose with persons)
  │                                        ↓
  │                                   ByteTrack
  │                          (assign persistent track IDs)
  │                                        ↓
  │                               Seat Assignment
  │                          (track bbox ↔ seat zone IoU)
  │                                        ↓
  │                         ┌──── Per-Person Loop ────┐
  │                         │                          │
  │                         │  Face crop → landmarks   │
  │                         │  → FaceMetrics (v1.0)    │
  │                         │                          │
  │                         │  Pose → PostureAnalysis  │
  │                         │                          │
  │                         │  Objects → Interaction   │
  │                         │                          │
  │                         │  Feature Fusion          │
  │                         │  → TemporalClassifier    │
  │                         │  → Behavior label        │
  │                         │  → Attention score       │
  │                         └──────────────────────────┘
  │                                        ↓
  │                               Room Analytics
  │                         • Per-person attention scores
  │                         • Seat occupancy map
  │                         • Room average attention
  │                         • Alerts (phone usage, absence)
  │                                        ↓
  │                            WebSocket → Dashboard
  │                         • Real-time room heat map
  │                         • Per-student behavior timeline
  │                         • Aggregate statistics
```

### Privacy Considerations

1. **No raw video storage** — only derived feature vectors and scores are persisted
2. **On-premise processing** — no cloud API calls for inference
3. **Configurable anonymization** — track IDs can be used instead of names (v2.0 uses track IDs only; name mapping is not implemented)
4. **Data retention policy** — analytics auto-purge after configurable period (**design intent only; not implemented in v2.0**: `DATA_RETENTION_HOURS` in `server/config.py` is not read by any code)
5. **Consent framework** — system logs consent status per student (design intent; not implemented)

The list above is the v2.0 design intent. **Current v1.0 status**:

- v1.0 currently writes **no** session data to Firestore. Only Firebase Auth is used (`firebase.ts` initialises Firestore, but nothing reads or writes it).
- When the user exports a report, the session summary (derived scores and statistics, no frames) is sent to the Genkit/Gemini report generator.
- No webcam frames are stored or sent anywhere.
- DACE evaluation logs (§11) contain no images and no user id, use a random session id, and are only downloaded manually in research builds.
- A data-retention limit is still to be implemented (DACE proposal section 7).

---

## Summary: Reuse vs. Redesign

| Component | v1.0 → v1.0+ | v1.0 → v2.0 |
|-----------|-------------|-------------|
| Face detection (face-api.js) | ✅ Reuse as legacy default (replacement by MediaPipe Face Landmarker pending decision R1, §11) | ❌ Replace (YOLOv8) |
| 68-point landmarks | ✅ Reuse as legacy default (see R1, §11) | ✅ Reuse (per-person crop) |
| EAR/MAR/Gaze/HeadPose | ✅ Reuse | ✅ Port to Python |
| BlinkDetector | ✅ Reuse | ✅ Port to Python |
| HeadPoseTracker | ✅ Reuse | ✅ Port to Python |
| FatigueCalculator | ✅ Reuse | ✅ Port to Python |
| FaceMetricsAggregator | 🔄 Extend | ✅ Port + extend |
| MLEngine scoring | 🔄 Redesign | 🔄 Redesign + port |
| Feature weights | 🔄 Add new features | 🔄 Add new features |
| EMA smoothing | ✅ Reuse | ✅ Reuse |
| SessionAnalytics | ✅ Reuse | ✅ Port per-person |
| Object detection | ✨ New (YOLOv8-nano) | ✨ New (YOLOv8-s) |
| Hand tracking | ✨ New (MediaPipe) | Pose covers this |
| Temporal classifier | ✨ New (1D-CNN) | ✨ New (1D-CNN) |
| Multi-person tracking | N/A | ✨ New (ByteTrack) |
| Seat mapping | N/A | ✨ New |
| Posture analysis | N/A | ✨ New (from pose) |
| Room dashboard | N/A | ✨ New |
| Chrome extension | ✅ Keep + update | N/A |
| Genkit AI flow | ❌ Remove (as built: **kept** for v1.0 PDF report generation, `generate-session-summary.ts`; the unused `analyze-facial-features` flow was removed) | ❌ Remove |

---

## 10. Post-Upgrade: Correctness, Security & CI (Status Update)

Everything above this section describes the original design analysis and upgrade plan. This section records what was actually built and fixed in subsequent passes, without removing or superseding the design rationale above — treat this as the current-state addendum.

### 10.1 Shared Scoring Core (v1.0 + v2.0)

The design above (§1's "Reuse Map") assumed `face-geometry.ts`/`ml-engine.ts` could simply be "reused directly" or "ported" as-is. In practice, three independent copies of this math (web app, Chrome extension, v2.0 Python port) drifted to different thresholds, weights, and formulas for the same signals — e.g. attention thresholds 70/40 (web) vs 80/50 (extension), gaze multiplier 2× vs 3×, and a head-pose pitch formula that normalized by nose-chin distance (which foreshortens as the head pitches, biasing pitch against itself) instead of the more stable interocular distance.

**Fix**: `v1.0/src/lib/scoring/` is now the single source of truth for all face-geometry math. `face-geometry.ts`/`ml-engine.ts` are re-export shims over it (existing imports unchanged). The Chrome extension's plain-JS `ml-core.js` was later rewritten to match it (EMA 0.15, thresholds 70/40, same trackers and aggregator outputs, legacy weights, IIFE) and is guarded by `v1.0/src/lib/__tests__/extension-equivalence.test.ts`, which loads it in node and compares it with the TypeScript on synthetic landmark sequences; it is still a hand-written copy, not generated. `v2.0/features/face_metrics.py` is a hand-maintained port. **There is no `EQUIVALENCE.md` and no automatic sync.** What exists are generated-fixture contract tests, all regenerated and drift-guarded by the `v1-test` CI job: `shared/fixtures/encode13_cross_language.json` (TypeScript, 13-dim encoder; read by `v2.0/tests/test_cross_language_contract.py`, which also checks the ONNX `loadModel`/export contract and now shows the encoders agree, including EAR 0.0, which v2.0 inference no longer replaces with 0.3; only the documented proximity-code divergence remains), `shared/fixtures/cmcv_cross_language.json` (TypeScript, CMCV; `v2.0/tests/test_contradiction_detector.py`), `shared/fixtures/encode15_v2.json` (Python, the 15-dim encoder; `v2.0/tests/test_feature_vector_v2_contract.py`), and `docs/proposals/evidence/legacy_pitch_constants.json` (emitted by the v1.0 pitch-characterisation test and checked by `v2.0/tests/test_pitch_characterisation.py` and `pitch_sim.py`). The scoring formulas themselves (EAR/MAR, blink, fatigue, gaze) have no shared numeric fixtures between TypeScript and Python. A formula change still bumps `scoring/VERSION.ts` and must be ported by hand in the same change. The golden-fixture test that guards the legacy path (`src/lib/__tests__/legacy-equivalence.test.ts`) snapshots the bare `new MLEngine()` behaviour, i.e. the as-shipped baseline.

Also added: `v1.0/src/lib/scoring/face-selection.ts`'s `PrimaryFaceSelector`, replacing the `detections[0]` hardcoding flagged as Limitation 5 above — locks onto the largest, most-centered face with identity-continuity hysteresis (v2.0 doesn't need this; ByteTrack already provides persistent per-person track IDs).

Full account: [v1.0 README §20](v1.0/README.md#20-shared-scoring-core--cross-language-consistency).

### 10.2 Cross-Modal Contradiction Vector (New Novel Contribution)

Beyond the novel algorithms already claimed in [README.md](README.md#novel-algorithmic-contributions) (Context-Gate Override, Asymmetric ConfusionPenaltyLoss, AR(1) Temporal Fusion, Prioritized Seat Mapping, Fatigue Onset Detection), a new additive contribution was designed and implemented: the **Cross-Modal Contradiction Vector (CMCV)**. Where the Context-Gate Override is a rule that fires on one known confusion (phone vs. reading) and discards the disagreement magnitude, CMCV is a general, symmetric, per-frame disagreement statistic over all modality pairs, retained as a first-class signal (not folded back into the attention score) and classified into an actionable taxonomy (`fatigue_masking`, `prop_without_engagement`, `context_conflict`, `coherent`, and `unclassified` in the corrected `v2` taxonomy).

Implemented in both `v1.0/src/lib/contradiction-detector.ts` and `v2.0/inference/contradiction_detector.py` with the same corrected taxonomy; the match is checked by `shared/fixtures/cmcv_cross_language.json`, generated from the TypeScript and drift-guarded in CI. The v1.0 engine keeps the original `v1` rules by default (the legacy golden fixture freezes them) and the web app uses `v2` (`APP_ENGINE_OPTIONS`): in `v2`, `prop_without_engagement` no longer fires for `hands_on_desk` or an `attentive_reading` label, and an unmatched high disagreement is `unclassified` instead of a silent `coherent`. v2.0 passes the behaviour label to it. Only the v1.0 legacy path is ported (not DACE). Computationally free (reuses already-computed sub-scores, zero new model inference) and unit-tested against constructed cases; full empirical validation is gated on labeled modality-conflict data that doesn't yet exist (see [v1.0 README §21](v1.0/README.md#21-cross-modal-contradiction-vector-novel) for the complete spec, integration points, and honest validation-status assessment).

### 10.3 Research-Integrity Fix: Real Training/Evaluation Pipeline

README.md's headline benchmark numbers (86.5% accuracy, F1 0.852, a 55,100-sequence dataset) were not backed by any real training run. Worse, `v2.0/evaluation/evaluate.py` was found to **actively fabricate results** — synthesizing predictions (`pred = label if rng.random() > 0.2 else random_class`) whenever no real dataset path was supplied — and the one real training run that did exist was evaluated circularly (trained and tested on the same synthetic AR(1) generator). This has been addressed in three parts:

1. **Fabrication removed, outputs retracted**: `evaluate.py` was rewritten. It evaluates the deployed classifier (rules + context gate) as well as the raw ONNX model, reports real MOTA (unclipped) and identity-based IDF1, a non-vacuous temporal-consistency coefficient (person ids are passed), a per-stage pipeline latency benchmark and event-based occupancy latency; without real data each section is `"status": "SKIPPED"`, and every report carries `data_source` (`real`/`synthetic`/`none`) and `claims_allowed`. The earlier `eval_20260309_*.json` and `experiment_report*.md` are retracted and moved to `v2.0/evaluation/reports/archive/` (see its README). The current report is `v2.0/evaluation/reports/behavior_classifier_v3_synthetic_note.md`, explicitly synthetic and circular.
2. **Pure-code training-pipeline bugs fixed** (no external dataset access needed): the train/inference feature-encoding mismatch, a complete ConfusionPenaltyLoss matrix, train/val leakage (loaders keep their own subject-level splits, synthetic windows share a group id with their augmented copies, and `assert_no_group_leakage` enforces a disjoint split; the earlier `GroupShuffleSplit` fix had left synthetic windows without group ids, so augmented copies still leaked, which is now fixed), selection by val macro-F1, ONNX/PyTorch parity checks, and loaders that reuse the server's per-track measurement through `TrackSequenceEncoder` sampled at 5 Hz by timestamp. Full details: [v2.0 README §23](v2.0/README.md#23-training-pipeline-correctness-fixes).
3. **Data tooling corrected**: Open Images MIDs are resolved from the official class-descriptions CSV (the hard-coded MIDs were wrong), and `perception/study_objects.py` is the single COCO-id mapping shared by the detector and the fine-tuning labels.

**Still blocked** on external access (unchanged from before this pass): acquiring real DAiSEE/EngageNet data, recording/licensing classroom footage, and running real ablations against a genuine held-out test set. README.md's numbers remain explicitly labeled as targets, not measured results, until that work happens.

### 10.4 Security Hardening (v2.0 Server, v1.0 report action, extension)

**v2.0 server** (`v2.0/server/security.py`, `main.py`, `input/frame_source.py`; tests in `tests/test_server_security.py`): `X-API-Key` authentication on every route except `/healthz` (keys from `COGNEN_API_KEY`, comma-separated for rotation, constant-time comparison; WebSocket may also use `?api_key=`), secure defaults (binds `127.0.0.1`; protected routes answer `503` with no key configured; `python -m server.main` refuses a non-loopback bind without a key unless `COGNEN_ALLOW_INSECURE=1`; `/docs` off unless `COGNEN_ENABLE_DOCS=1`; no CORS origins unless `COGNEN_CORS_ORIGINS`), request-body cap (`COGNEN_MAX_BODY_BYTES`), token-bucket rate limits on `/start` and failed auth, WebSocket hardening (origin allowlist, per-key/per-room/total caps `COGNEN_WS_MAX_PER_KEY`/`_PER_ROOM`/`_TOTAL`, send timeout, message-size and message-rate limits, heartbeat), an SSRF policy for stream sources (`COGNEN_STREAM_SCHEMES`, `COGNEN_STREAM_HOSTS` allowlist of names/CIDRs, `COGNEN_ALLOW_PRIVATE_STREAMS`, `COGNEN_STREAM_PORTS`, `COGNEN_ALLOW_URL_CREDENTIALS`; loopback/link-local/metadata always denied; DNS pinning for `rtsp://`), file sources confined to `COGNEN_VIDEO_DIRS` (realpath) with an extension allowlist, camera sources off unless `COGNEN_ALLOW_CAMERA_SOURCE=1`, log redaction and security response headers. Earlier fixes remain: `room_id` validation (path traversal), clear model-missing errors, correct `404`s, WebSocket exception handling and the `COGNEN_MAX_CONCURRENT_ROOMS` cap. Remaining risks are listed in the v2.0 README §24 (DNS-rebinding window for TLS streams, FFmpeg RTSP redirects). None of this has been exercised against a real network deployment. Full details: [v2.0 README §24](v2.0/README.md#24-server-hardening).

**v1.0 report action**: `generateReportAction` was an unauthenticated server action (Gemini cost exposure). It now requires a Firebase ID token verified server-side against Google's signing keys (`src/lib/server/firebase-id-token.ts`, `node:crypto`), a per-user rate limit and a zod-validated input. The dashboard's sign-in guard is still client-side UI gating only.

**v1.0 extension**: rewritten to web parity with a signed message channel (per-session HMAC key delivered through `executeScript` args, verified and replay-checked in an isolated-world `bridge.js`), restricted `host_permissions` and `web_accessible_resources`, `storage.session` persistence and an idempotent `window.__cognen` namespace; see `v1.0/extension/README.md`. Page scripts in the MAIN world can still read the posted messages and tamper with the injected globals. Not run in Chrome.

### 10.5 Dependency & Code-Quality Hygiene

- **v1.0**: `npm audit` is now **0 vulnerabilities** (from 107, 6 critical, via bumps to next 15.5.27, genkit 1.42, firebase 11.10, the Tailwind 3 to 4 migration that removed `braces`, a local stub `scripts/stubs/extract-zip`, OpenTelemetry forced to 2.x with the `scripts/patch-genkit-otel2.mjs` postinstall, a longer `package.json` `overrides` list, and explicit `@emnapi` devDependencies so the lockfile is complete on Linux). The redundant `yarn.lock` was removed, an ESLint config added (and the `ignoreBuildErrors`/`ignoreDuringBuilds` flags removed), React error boundaries added, Firebase config moved to env vars, `.env.example` added. Not verified: Genkit's runtime behaviour after the OTel jump with a real Gemini call. Full details: [v1.0 README §22–23](v1.0/README.md#22-testing-linting--ci).
- **v2.0**: `ruff`/`mypy` config added (none existed before). First mypy pass found 55 issues; fixing them surfaced one genuine regression (`evaluate.py` still unpacking a 2-tuple after a loader interface change elsewhere had moved to 3-tuples) and one worthwhile structural fix (`_ClassSpec` converted from an unverifiable `**kwargs`/`setattr` pattern to a typed dataclass, resolving 20 findings at once). The remaining backlog was later fixed properly (Optional narrowing, typed optional locals, `Protocol`s, correct signatures; no `# type: ignore`), so `mypy .` is clean with and without the heavy dependencies installed and `v2-typecheck` is a blocking CI job. `requirements.txt` is pinned with `==` on a numpy 2.x stack that resolves (the old pin set was `ResolutionImpossible` and held vulnerable torch/Pillow/onnx/starlette): torch 2.13.0, torchvision 0.28.0, numpy 2.4.6, boxmot 16.0.11 (a real ByteTrack; 10.x did not export it and a stub tracker was silently used), ultralytics 8.4.146, opencv-python 4.13.0.92, onnxruntime 1.24.4, shapely 2.1.2, mediapipe 0.10.31, onnx 1.22.0, Pillow 12.3.0, scikit-learn 1.7.2, pytest 9.0.3, fastapi 0.141.1 + starlette 1.6.0, pydantic 2.13.5; Python 3.10-3.12 only; the default Linux torch wheel is a CUDA 13 build (use the CPU index on CPU-only servers). Full details: [v2.0 README §25](v2.0/README.md#25-code-quality-testing--ci).

### 10.6 Continuous Integration

`.github/workflows/ci.yml` (one file, shared by both sub-projects, plus `.github/dependabot.yml`, `.github/gitleaks.toml` and `.github/ci/requirements-*.txt`) was rebuilt. Triggers: push and pull request on `main` and `dev`, and `workflow_dispatch`, which runs **every** job regardless of paths (`gh workflow run ci.yml --ref dev`). Hygiene: Node 24, `ubuntu-24.04`, every action pinned by commit SHA, top-level `contents: read` with `pull-requests: read` only on `changes` and `secret-scan`, job timeouts, superseded runs cancelled except on `main`. Jobs (12): `changes`; v1 typecheck, lint, test, build, audit; v2 lint, typecheck, test, audit; `secret-scan`; `all-checks-passed` (includes `changes`; a path-filtered skip counts as passing). `v1-test` runs Vitest and regenerates the `encode13`, CMCV and `encode15_v2` fixtures and `pitch_sim.py`, failing on any diff or untracked generated file (fixture drift guards). `v1-audit` runs `npm audit --audit-level=low` and **blocks**. `v2-audit` resolves `server/requirements.txt` and **blocks** (`V2_AUDIT_RESOLVE` switches to a pins-only informational fallback). `v2-test` runs the light pytest tier installed with the production pins as constraints; `v2-typecheck` (mypy) is clean and blocking, so no job is informational. gitleaks allowlists the public Firebase web key, model checksums and fake test keys. A manual run on `dev` had all 12 jobs green. Dependabot opens weekly grouped PRs against `dev` with a 14-day cooldown.

### 10.7 v2.0 Measurement, Context Gate, Posture and Model (status)

All of the following is implemented and unit-tested; none has been run on real footage, a real camera stream or a GPU.

- **Measurement**: head pitch from the MediaPipe facial transformation matrix against a per-track screen-facing reference, blendshape gaze with an iris cross-check and gaze-corrected eye openness/PERCLOS (`features/dace_measurement.py`, ported from v1.0 DACE); one sign convention for the feature vector and rules, **down = negative**, with a single negation at the boundary. The legacy 68-point pitch (rising when the head tilts down, neutral offset about +12.6) is only a labelled fallback. This fixes the earlier state where every `pitch < -15` rule was unreachable on the server.
- **Landmarks**: FaceLandmarker runs in IMAGE mode, fails loudly (`available=False`, error logged) and the 68-point map `_MP_TO_68` was rebuilt from the canonical mesh (the old table had a wrong jaw, nose tip and outer-lip corners, making MAR meaningless).
- **Context gate**: live by default (`COGNEN_HAND_PROXIMITY=pose`; `hands` or `off` selectable); window-level phone/book evidence; `absent` is reachable (no-face share of the window).
- **Features and model**: `FEATURE_VECTOR_VERSION` 2 = 13 v1.0-compatible channels plus `posture_upright` and `posture_lean_norm` (15 dims; `encode_feature_vector` stays 13-dim for v1.0); served model `behavior_classifier_v3.onnx` (20,742 parameters), trained on the hand-written synthetic generator only, so its 1.000 hold-out scores are **circular**. Older 13-channel models still load.
- **Runtime**: one shared ONNX session, per-person state pruning, track id 0 handled, one-to-one seat matching, inference offloaded to a worker thread, background tasks tracked, real ByteTrack via boxmot 16 (a stub remains only as a fallback when boxmot is missing or incompatible). CMCV is wired to the behaviour label.
- **Cadence caveat**: training uses 5 Hz windows (`TrackSequenceEncoder`); the server default `COGNEN_PROCESS_EVERY_N=1` would feed windows at the source rate, so a 15 FPS source needs `COGNEN_PROCESS_EVERY_N=3`. Not enforced in code.

---

## 11. DACE: Downward-Attention Context Engine (v1.0 only)

> **Status (read first).** Every DACE number in this section is a **target** or an **initial parameter**. Nothing has been measured on real data: no pilot recordings (P1) exist yet. DACE is **off by default**, and with DACE off the legacy path is unchanged (guarded by the golden-fixture test `src/lib/__tests__/legacy-equivalence.test.ts`). Sources: the DACE proposal and Technical Annex (v1.0, 10 Oct 2026) and the implementation plan; where they differ, the deviations (D1-D15) are listed in the plan and need supervisor sign-off.

### 11.1 Purpose

Looking down is not looking away: a student reading or writing at a desk looks down, and CognEn should not score that as distraction. CognEn's design (§2) already aims at this, but the DACE proposal (section 1) found three defects that stop the design from taking effect in v1.0: (1) the legacy head-pitch formula has the wrong sign and scale, rising rather than falling as the head tilts down, so the `pitch < -15` reading/phone branch rarely runs; (2) "gaze" is computed from eyelid points, because the 68-point model has no iris points; (3) the object-detector model is not shipped (the original defect that the web app then passed an empty object list, selecting the upgraded weights with no context, has since been fixed on the legacy path). DACE fixes measurement, adds desk-zone evidence, applies ordered temporal rules, and outputs an explicit "Unsure" when a single webcam cannot resolve the case.

### 11.2 Pipeline and stages

```
Webcam frame (fixed 200 ms tick in DACE mode)
  │
  ├─ M  Measurement: MediaPipe Face Landmarker → pitch/yaw/roll (deg, +down),
  │      blendshape gaze with iris cross-check, gaze-corrected eye openness/PERCLOS
  ├─ E  Desk-zone evidence: object detector (every 3rd tick) + HandLandmarker
  │      → phone in use? book? writing? desk visible?
  ├─ K  Eye kinematics ............................ Stage 2 stub only (forced off)
  ▼
  T  Temporal fusion: ordered rules over a 15-frame (3 s) window
  │     (learned temporal layer via loadModel() ............ Stage 2 only)
  ▼
  U  "Unsure" with reason code U1-U4 (rules)
  │     (U5 conformal abstention ........................... Stage 2 only)
  ▼
  Existing MLEngine (FEATURE_WEIGHTS, EMA, CMCV, contribution UI) via feature substitution
  ("Unsure" holds the score; reported separately)
```

**Stage 1** (committed): M, E, rule-based T and rule-based U. **Stage 2** (conditional on ethics approval and the Stage 1 go/no-go): eye kinematics K (`scoring/dace/kinematics.ts`), a learned temporal model (`temporalModel=onnx`; the `loadModel()` contract now matches the v2.0 export: input `input`, channels-first (1, 13, 15), output `logits` + softmax, v2.0 proximity codes, plan D10; no trained real-data model exists yet), and conformal abstention U5 (`scoring/dace/conformal.ts`). In Stage 1 these exist only as stubs with no runtime caller.

**Implementation status.** Implemented: the pure core in `src/lib/scoring/dace/` (params, head pose from matrix, pitch reference, blendshape gaze, eye openness, corrected PERCLOS, looking-down test, solvePnP, mesh adapter, hand-object relations, evidence, window, rules, abstention, score hold, feature mapping, Stage 2 stubs); the browser layer in `src/lib/dace/` (config, presets, measurement providers, MediaPipe hand landmarker and object detector, YOLO adapter with self-test/checksum, engine, rule runner, scheduler, latency probes, eval log, self-test gate, replay frame source); the opt-in MLEngine path and dashboard/report integration; the `/replay` page; `scripts/dace/*`; the model download/checksum step in `setup-models.sh`; `aruco_pose.py`. All of it is covered by node unit tests with mocked browser libraries. **Not verified**: nothing has been run in a browser or against a camera or recording yet (MediaPipe WASM loading under Next.js, GPU/CPU delegate behaviour, real latency, real accuracy); those checks are part of the Stage 1 pilot.

### 11.3 Configuration flags (`src/lib/dace/config.ts`)

Zod-validated; an invalid layer is skipped with a warning so DACE can never stop CognEn from producing a score.

| Flag | Values (default first) | Meaning |
|------|------------------------|---------|
| `enabled` | `false` / `true` | Master switch. `false` = legacy path, unchanged |
| `measurement` | `legacy` / `pnp68` / `mediapipe` | Measurement provider: face-api.js formulas, solvePnP on the 68 points (baseline A''), or MediaPipe Face Landmarker |
| `evidence` | `false` / `true` | Desk-zone evidence E (objects + hands) |
| `rules` | `false` / `true` | Ordered rule layer T |
| `abstain` | `off` / `rules` / `conformal` | "Unsure" output. `conformal` is Stage 2 and degrades to `rules` with a warning in Stage 1 |
| `detector` | `yolo` / `mediapipe` | Object detector: YOLOv8n (AGPL-3.0) or MediaPipe EfficientDet-Lite0 (Apache-2.0), decision R2 |
| `kinematics` | `off` / `on` | Eye kinematics K. Stage 2 feasibility study; forced `off` in Stage 1 |
| `temporalModel` | `rules` / `onnx` | `onnx` is Stage 2 (warns; the rule layer is the fallback) |
| `baseline` | `internal` / `peb` | Pitch reference / open-eye level source. PEB is interface-only (`BaselineProvider`) |
| `log` | `off` / `windows` / `frames` | FR9 evaluation log granularity |
| `replay.shuffleObjects` | `false` / `true` | Negative control: shuffle object labels across windows (preset H) |

**Resolution order** (later layers override earlier ones): (1) code defaults (DACE off) → (2) `NEXT_PUBLIC_DACE_PRESET`, then the individual `NEXT_PUBLIC_DACE_*` flags → (3) research-only override from `?dace=` or `localStorage['cognen.dace']` (URL wins), honoured **only** when `NEXT_PUBLIC_DACE_RESEARCH=1` → (4) an explicit object from the replay runner. Override strings accept a preset name, `key=value` pairs (optionally after a preset, e.g. `F,log=windows`), or a JSON object. The resolved config is deep-frozen for the session; its FNV-1a hash and `DACE_VERSION` (`scoring/dace/VERSION.ts`, currently `0.1.0`) are stamped on every evaluation-log record.

### 11.4 Ablation presets (`src/lib/dace/presets.ts`)

| Preset | Configuration | Question | Detector model |
|--------|---------------|----------|----------------|
| A | DACE off | CognEn as shipped (reference) | absent |
| A' (alias `A1`) | DACE off | Fair baseline | present |
| A'' (alias `A2`) | `measurement=pnp68`, no E/T/U | Does a minimal pitch fix suffice? | present |
| B | `measurement=mediapipe` | RQ1: measurement error vs missing context | present |
| C | B + `evidence` | RQ2: does desk evidence separate reading/writing from phone? | present |
| F | C + `rules` + `abstain=rules` | RQ3 (pilot): does abstention reduce harmful errors? | present |
| H | F + `replay.shuffleObjects` | Negative control: is the gain really from context? | present |
| I | F, evaluated on desk-out-of-frame segments only | Stress test: does it abstain instead of guessing? | present |

Stage 1 F = C + rule T + rule U (plan D12); the annex's F also depends on Stage 2 parts. "Detector model" is an environment requirement of the run, not a DACE flag.

### 11.5 Parameters (`src/lib/scoring/dace/params.ts`)

**Table A.6 initial values** (exact; snapshot-tested; change only with a documented tuning run on P1):

| Parameter | Value | Use |
|-----------|-------|-----|
| `thetaDownDeg` (θ_down) | 15 | relative pitch counted as looking down (inclusive) |
| `gammaDown` (γ_down) | 0.5 | eye-down blendshape level counted as looking down (inclusive) |
| `kOpen` (k) | 0.7 | residual openness below which the eye is closed (strict) |
| `pDrowsy` | 0.15 | gaze-corrected PERCLOS (fraction, 60 s) treated as drowsy |
| `cMin` | 0.35 | minimum detector confidence |
| `rhoUse` (ρ_use) | 0.6 | share of W with phone-in-use evidence |
| `piMax` (π_max) | 0.35 | pinch ratio for a pen grip (strict) |
| `mMin` / `mMax` | 0.02 / 0.15 | writing wrist motion band (hand lengths per frame) |
| `tauGlanceMs` (τ_g) | 3000 | longest downward glance treated as brief (≤) |
| `alphaConformal` (α) | 0.1 | conformal error level (Stage 2) |

**Prose constants** (annex values): 5 FPS base rate (200 ms frame interval); window W = 15 frames (3 s); pitch reference 5 s; absent if no face in ≥ 80% of W; U3 if face missing in ≥ 50% of W; holding = IoU ≥ 0.1; near = centre distance < 0.5 × box diagonal; phone present ≥ 30% of W; book ≥ 50% of W; writing pinch and motion each ≥ 60% of W; "most" = more than 50%; desk visibility over 25 frames at ≥ 30%; phone label hold 10 s; continuous closure ≥ 2 s = drowsy; PERCLOS window 60 s; detector every 3rd frame.

**TBD / provisional** (defaults exist only so the code runs; each is to be fitted or confirmed on P1 and is not a claim): `cameraMovePitchJumpDeg` (25), `cameraMoveFaceScaleJump` (0.3), `earPrior` {b0 0.3, b1 -0.12, b2 0}, `earScale` (0.3), `gazeScale` (1), `pitchSign` (+1), `opennessMinSamples` (50), `opennessMinGazeRange` (0.3), `huberDelta` (1.345), `perclosMinSamples` (10), `episodeGapFrames` (1), `irisMaxOffset` (0.5), `irisSignTol` (0.05), `screenGazeMax` (0.3), `screenYawMaxDeg` (15), `screenPitchUpMaxDeg` (15), `frameAspect` (640/480), `handFaceMaxDistFaceWidths` (3). Stage 2 constants (kinematics 30 FPS, 3-6 s windows, go/no-go at 2 macro-F1 points) are unused in Stage 1.

### 11.6 Rule order (`src/lib/scoring/dace/rules.ts`)

Each 3 s window is labelled by the **first** rule that applies (plan D11):

1. **absent**: no face in ≥ 80% of W
2. **U3** (poor tracking): face missing in ≥ 50% of W → Unsure
3. **drowsy**: gaze-corrected PERCLOS ≥ p_drowsy, or one continuous closure ≥ 2 s
4. **U4** (conflicting evidence): phone in use in one hand and pen-grip motion in the other → Unsure
5. **using_phone**: phone in use; the label is held for 10 s after the last firing so repeated short glances accumulate
6. **attentive_reading**: looking down in most of W **and** writing or book evidence
7. **brief glance**: in a downward episode of ≤ τ_g → previous output kept
8. **long downward episode without evidence** (> τ_g) → Unsure U2 if the desk is not visible, else U1
9. **screen**: not looking down → `attentive_screen` or `distracted` (existing screen-attention criteria)

With `abstain=off`, U3 and U4 are skipped and step 8 outputs `distracted`, so ablations B/C can run without "Unsure". **Pending supervisor sign-off:** the D11 ordering (where U3/U4 sit) and the U4 definition (the annex wording could never fire; the implemented definition is "phone held in one hand + pen-grip motion in the other"). This ordering affects the phone-recall guard (target ≥ 0.80).

### 11.7 "Unsure" reason codes and semantics

Reason codes (`src/lib/scoring/dace/abstain.ts`, `UNSURE_REASON_TEXT`):

| Code | Text | Notes |
|------|------|-------|
| U1 | no phone, book or writing evidence | |
| U2 | desk not visible | takes precedence over U1 |
| U3 | poor tracking | |
| U4 | conflicting evidence | definition pending sign-off (D11) |
| U5 | model uncertain | Stage 2 only (conformal) |

Semantics (FR7, `score-hold.ts`):

- "Unsure" is **not** an `AttentionState` and **not** a `BehaviorLabel`; it is carried as optional flags (`unsure`, `unsureReason`, `scoreHeld`) on `MLPrediction` / `SessionDataPoint`, so existing charts, stats and extension types are unaffected.
- The score is **held**: the EMA is not updated, the displayed score is the last smoothed value, the attention state is kept; the contradiction detector is still updated. On leaving Unsure the EMA resumes from the held value (no jump).
- Unsure on the very first tick (no smoothed value yet): the UI shows "-" and no data point is recorded.
- Session averages, trend, periods and the PDF report exclude Unsure points. The report shows the Unsure share and dominant reason (e.g. "could not determine the activity for 12% of the session (mostly: desk not visible)").
- "Unsure" is never converted into a penalty.

### 11.8 Measurement conventions (M)

- **Pitch** in degrees, **positive = head down** (FR1), from the MediaPipe facial transformation matrix (`head-pose-matrix.ts`): column-major 4x4, y-up camera frame, decomposed as R = Ry(yaw) · Rx(pitch) · Rz(roll), pitch = asin(-r12), yaw = atan2(r02, r22), roll = atan2(r10, r11). Used **relative** to a 5 s screen-facing median reference (`pitch-reference.ts`), re-estimated on return from absent and after a camera move; frames with eyes looking down do not contribute to the reference. The sign is to be confirmed on the scripted up/down pilot segments (`pitchSign` exists only to correct a measured inversion).
- **Legacy field substitution (plan D7)**: existing code and models assume down = negative, so in DACE mode the legacy fields `pitch` and `gazeY` are filled with the **negated** DACE values. Every existing `pitch < -15` consumer then sees correct values.
- **Gaze** from eye blendshapes (vertical g = mean eyeLookDown - mean eyeLookUp, +down; horizontal from eyeLookIn/Out), kept separate from head pose, with an **iris cross-check**: if the iris is missing, implausible or contradicts the blendshape direction, gaze is marked invalid and looking-down falls back to head pitch only.
- **Looking down**: D_t = [Δθ ≥ θ_down or g ≥ γ_down] (gaze clause only when gaze is valid).
- **Eye openness corrected for vertical gaze** (`eye-openness.ts`): expected openness E(g) = b0 + b1·g + b2·g² (Huber fit on attentive frames, provisional prior until identifiable); residual r = EAR / E(g); the eye is closed if r < k. **Gaze-corrected PERCLOS** (`perclos.ts`) is the share of closed frames over 60 s, plus the longest continuous closure.
- **Mesh-to-68 adapter** (`mesh-adapter.ts`): an independently audited 478 → 68 index table (not v2.0's `_MP_TO_68`) so the legacy aggregator (MAR, face area, blink/fatigue trackers) can run on MediaPipe landmarks; the left/right mirror table was verified against the canonical MediaPipe face model (`canonical_face_model.obj`, exactly symmetric) and against the connection tables of the installed `@mediapipe/tasks-vision` package (`scoring/dace/__tests__/mesh-topology.test.ts`; no errors found), and the eye/lip points are the standard mesh EAR/MAR points.

### 11.9 Feature mapping and the FR4 gate

DACE adds **no new scoring features**; it substitutes values of the existing `FEATURE_WEIGHTS` keys so weights, EMA, CMCV and the contribution UI keep working (`feature-mapping.ts`, plan section 6, **pending sign-off**):

| Feature | DACE value |
|---------|------------|
| `ear` | residual openness × `earScale` |
| `perclos` | gaze-corrected PERCLOS × 100 |
| `gazeDeviation` | (\|g_h\| + \|g_v\|) × `gazeScale`, only when not looking down and gaze valid |
| `yawDeviation` | yaw in degrees |
| `contextAwarePitch` | reading 85, phone 20, drowsy 15, glance = previous value, otherwise the legacy standard curve on relative pitch |
| `objectContext` | phone in use 25, reading/writing evidence 80, otherwise 65 (a phone merely present never lowers it) |
| `handObjectProximity` | phone in use 20, book 80, writing 75, else 65 (hands visible) / 60 |
| `temporalPattern` | per-label behaviour score (same table as the temporal classifier); 60 while Unsure |

Implementation choices to review: preset C (evidence on, rules off) labels only from evidence (phone in use → `using_phone`; looking down in most of W with book/writing → `attentive_reading`; otherwise the standard curve); preset A'' substitutes pose only, in absolute solvePnP degrees; with `measurement=legacy` the legacy pitch formula is passed through uncorrected.

Absent keeps current semantics (no data point); absent windows are recorded in the eval log but an absent share is not yet shown on the dashboard. Whether absence should lower the score is a supervisor decision.

**FR4 gate**: in DACE mode the legacy "empty object list selects upgraded weights" path is never used. Object evidence is used only when the detector has loaded and passed its self-test; otherwise `detectorAvailable=false` and evidence comes from hands only (writing, desk visibility), so a long downward episode becomes Unsure rather than distracted.

### 11.10 Fallbacks and runtime

- **Fallbacks** (plan section 8): MediaPipe initialisation failure → automatic fallback to the legacy provider with a toast; any component throw (or a frame over 2x budget) → legacy scoring for that frame, logged; detector missing or self-test failure → `detectorAvailable=false`; hand landmarker failure → no holding/writing evidence.
- **Scheduler**: fixed 200 ms tick with skip-if-busy and a dropped-tick count; no-face frames are fed to the window (needed for absent/U3). Object detector on ticks i % 3 == 0, hands on i % 3 == 1, so they do not coincide.
- **Latency**: per-component p50/p95/max probes. **p95 ≤ 200 ms per tick is a target, not a measurement.**

### 11.11 Evaluation tooling

- `/replay` page (`src/app/replay/`): research-only, returns 404 unless `NEXT_PUBLIC_DACE_RESEARCH=1`; reads local files only, deterministic frame stepping.
- FR9 JSONL evaluation log (`eval-log.ts`): per-window (or per-frame) features, labels and reason codes; no images, random session id, no user id, manual download only; every record carries the config hash and `DACE_VERSION`.
- `scripts/dace/replay-features.ts` (`npm run dace:replay`): offline E/T/U re-run from frame logs.
- `scripts/dace/metrics.ts`: FDR-DA (share of reading/writing windows labelled distracted, drowsy or phone), phone recall, Unsure rate on desk-out-of-frame segments, answer rate, latency.
- `scripts/dace/cvat-windows.ts`: CVAT annotation → 3 s window labels.
- `docs/proposals/evidence/pitch_sim.py` (present) reproduces the legacy-pitch characterisation table; `aruco_pose.py` estimates head pose from a headband ArUco marker (pitch in DACE's +down convention, relative to a 5 s reference) for the sign/error check; syntax-checked only, not yet run on a recording.

Stage 1 targets (proposal objectives, unmeasured): correct pitch sign in 100% of scripted segments and pitch error ≤ 5° vs marker ground truth; FDR-DA reduced by ≥ 30% relative to baseline with phone recall ≥ 0.80; Unsure on ≥ 90% of desk-out-of-view segments; p95 latency ≤ 200 ms.

### 11.12 Models and licences

`scripts/setup-models.sh` downloads pinned `face_landmarker.task`, `hand_landmarker.task` and `efficientdet_lite0.tflite` (Apache-2.0) from `storage.googleapis.com/mediapipe-models`, copies the `@mediapipe/tasks-vision` WASM to `public/mediapipe/wasm` (gitignored), and verifies everything against `scripts/models.sha256` (hashes recorded 2026-10-10). It still only prints instructions for the YOLO model; its hash can be recorded in `models.sha256` and passed to the browser via `NEXT_PUBLIC_DACE_YOLO_SHA256`. **YOLOv8 is AGPL-3.0 (decision R2)**: research use only, pending approval; the MediaPipe EfficientDet-Lite0 detector is the Apache-2.0 comparison. COCO has no pen/paper class, which is why writing is detected from hand landmarks.

### 11.13 Scope

DACE as a whole (E/T/U, scheduler, UI) is v1.0 web app only; the Chrome extension keeps the **old** formulas (`extension/lib/ml-core.js` is a hand copy, not shared code). **v2.0 has ported the measurement layer M** (`v2.0/features/dace_measurement.py`: MediaPipe transformation-matrix pitch, +down, relative to a per-track 5 s screen-facing reference with absent/camera-move resets; blendshape gaze with iris cross-check; gaze-corrected eye openness and PERCLOS) and the hand-object relations (`features/hand_object.py`, MediaPipe HandLandmarker, `COGNEN_HAND_PROXIMITY=hands`; default `pose` = YOLOv8-pose wrists, `off` = opt-out). v2.0 flips the measurement sign once, so its feature vector and rules use **down = negative** like v1.0's D7 legacy-field substitution; the 68-point formula is only a labelled fallback. v2.0's model input is versioned: FEATURE_VECTOR_VERSION 2 = the 13 v1.0-compatible channels + 2 posture channels (15; `behavior_classifier_v3.onnx`, synthetic-only training), and v1.0 `loadModel(url, { featureDim: 15 })` pads posture with (1, 0). Contract tests: `tests/test_cross_language_contract.py` (generated `encode13` fixture; first 13 channels), `tests/test_feature_vector_v2_contract.py` (`shared/fixtures/encode15_v2.json`), `tests/test_pitch_characterisation.py` (legacy pitch defect, fallback only), `tests/test_dace_measurement.py`, `tests/test_context_gate.py` and `tests/test_hand_proximity.py`. `scoring/dace/` is not shared with the extension and is not re-exported from `scoring/index.ts`; the annex sentence saying `scoring/` is shared with the extension is inaccurate (plan D3).

### 11.14 Decisions pending supervisors

| ID | Decision | Default assumed in the code |
|----|----------|-----------------------------|
| R1 | Replace face-api.js with MediaPipe Face Landmarker in v1.0 | Yes, but only behind a flag; legacy kept until pilot results |
| R2 | YOLOv8 (AGPL-3.0) for research, compared with Apache-2.0 MediaPipe detector | Research use only |
| R4 | Ethics: whether the team-only pilot needs committee approval | No recordings made yet |
| D11 / U4 | Rule order and U4 definition | As in §11.6 |
| Plan §6 | Feature mapping into `FEATURE_WEIGHTS` | As in §11.9 |
| Absence | Whether absence should lower the score | DACE mode: no (current semantics kept). Legacy path: yes, `Idle` scored 0 through the EMA after a 3 s grace |
