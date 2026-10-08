# CognEn System Architecture — v1.0 Upgrade & v2.0 Design

> **Author**: Generated from deep codebase analysis of CognEn v1.0  
> **Scope**: Single-user webcam (v1.0) → Multi-person CCTV (v2.0)

---

## Table of Contents

1. [v1.0 Analysis — What Exists](#1-v10-analysis)
2. [v1.0 Upgrade — Study Behavior Understanding](#2-v10-upgrade)
3. [v2.0 Design — Multi-Person Room Analysis](#3-v20-design)
4. [Model Choices & Justification](#4-model-choices)
5. [Pseudo-Code for Key Algorithms](#5-pseudo-code)
6. [Dataset Suggestions](#6-datasets)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Implementation Folder Structure](#8-folder-structure)
9. [End-to-End Inference Pipeline](#9-inference-pipeline)
10. [Post-Upgrade: Correctness, Security & CI (Status Update)](#10-post-upgrade-status)

---

## 1. v1.0 Analysis

### Current Architecture

```
Webcam (640×480 @ 5 FPS)
  → face-api.js TinyFaceDetector (320px, ~150ms)
  → 68-point facial landmarks
  → FaceMetricsAggregator (12 features)
  → MLEngine (weighted scoring + EMA α=0.15)
  → Attention classification: Attentive / Distracted / Multitasking
```

### Existing Modules (Reuse Map)

| Module | File | Lines | Reuse in v1.0+ | Reuse in v2.0 |
|--------|------|-------|----------------|---------------|
| Face detection | `face-api.js` (TinyFaceDetector) | external | ✅ Keep | ❌ Replace with YOLOv8 |
| Landmark detection | `face-api.js` (68-point) | external | ✅ Keep | 🔄 Per-person crop |
| EAR / MAR / Head Pose | `face-geometry.ts` (607 lines) | L54-150 | ✅ Reuse directly | ✅ Reuse per-person |
| Blink detector | `face-geometry.ts` `BlinkDetector` | L155-243 | ✅ Reuse | ✅ Reuse per-person |
| Head pose tracker | `face-geometry.ts` `HeadPoseTracker` | L385-446 | ✅ Reuse | ✅ Reuse per-person |
| Fatigue calculator | `face-geometry.ts` `FatigueCalculator` | L449-498 | ✅ Reuse | ✅ Reuse per-person |
| Metrics aggregator | `face-geometry.ts` `FaceMetricsAggregator` | L502-592 | 🔄 Extend | 🔄 Extend per-person |
| ML Engine | `ml-engine.ts` `MLEngine` | L235-395 | 🔄 Redesign scoring | 🔄 Redesign scoring |
| Feature weights | `ml-engine.ts` FEATURE_WEIGHTS | L19-42 | 🔄 Add object/posture weights | 🔄 Add object/posture weights |
| **Scoring core (post-upgrade)** | `src/lib/scoring/` (new, see §10) | — | ✅ Single source of truth | ✅ Ported to `features/face_metrics.py`, kept in sync via golden-fixture contract |
| Session analytics | `analytics.ts` `SessionAnalyticsEngine` | L225-369 | ✅ Reuse | 🔄 Per-person analytics |
| Genkit AI flow | `analyze-facial-features.ts` | 51 lines | ❌ Remove (latency) | ❌ Remove |
| Chrome extension | `ml-core.js`, `offscreen.js`, etc. | ~1500 lines | ✅ Keep | N/A (server-based) |

### Key Limitations of v1.0 (status update: 1–4 addressed by the v1.0 upgrade below; 5 fixed separately)

1. **No object detection** — cannot distinguish book vs phone vs screen. **Addressed**: YOLOv8-nano object detection is implemented (§2.1).
2. **Looking down = distracted** — pitch penalty is uniform; reading a book is penalized. **Addressed**: context-aware pitch scoring is implemented (§2.2), though the underlying head-pose formula itself had a separate correctness bug (see "Shared Scoring Core" note below) that has also since been fixed.
3. **No body/hand tracking** — misses posture and hand-object interactions. **Addressed**: MediaPipe Hands + `HandObjectProximityDetector` implemented (§2.4).
4. **Single-frame decisions** — EMA smoothing only; no true temporal modeling. **Partially addressed**: a rule-based 15-frame sliding-window classifier exists (§2.3); the trained 1D-CNN version described in README-overall.md is not yet backed by a real trained model (see README-overall.md §7's honesty banner).
5. **Single person only** — `detections[0]` hardcoded everywhere. **Fixed** (independent of the v1.0 upgrade path below): replaced with `PrimaryFaceSelector` in `v1.0/src/lib/scoring/face-selection.ts` — locks onto the largest, most-centered face with identity-continuity hysteresis. See "Shared Scoring Core & Correctness Fixes" note below.

> **Shared Scoring Core & Correctness Fixes (post-upgrade)**: beyond the upgrade path described in this section, a later pass found and fixed real bugs in the *existing* math this document describes as "reuse directly": the head-pose formula (§5.4 in both READMEs) used nose-chin distance to normalize pitch, which foreshortens as the head pitches and biased pitch against itself — now both yaw and pitch share the same normalizer (interocular distance). The web app, Chrome extension, and v2.0's Python port had also independently drifted to different thresholds/weights/formulas for the same signals; `v1.0/src/lib/scoring/` is now the single source of truth (see [v1.0 README §20](v1.0/README.md#20-shared-scoring-core--cross-language-consistency) for the full account, including a real bug this uncovered in v2.0: it never computed gaze at all).

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

Using MediaPipe Hands (runs via `@mediapipe/hands` in browser):
- Detect hand keypoints (21 per hand)
- Compute distance between hand center and detected object bounding box
- Signal: `hand_object_proximity ∈ {holding_book, holding_phone, hands_on_desk, hands_not_visible}`

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
| **Face landmarks** | face-api.js 68-point (v1 reuse) | Already proven in v1.0; runs on cropped face | MediaPipe FaceMesh (468 pts, overkill), dlib (C++ only) | ~15ms/crop GPU |
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
│   │   │   └── temporal-classifier.ts # ✨ NEW: sliding window classifier
│   │   └── components/dashboard/  # ✅ REUSED with minor updates
│   └── models/                    # ✨ NEW: ONNX model files
│       ├── yolov8n-objects.onnx
│       └── behavior-classifier.onnx
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
3. **Configurable anonymization** — track IDs can be used instead of names
4. **Data retention policy** — analytics auto-purge after configurable period
5. **Consent framework** — system logs consent status per student

---

## Summary: Reuse vs. Redesign

| Component | v1.0 → v1.0+ | v1.0 → v2.0 |
|-----------|-------------|-------------|
| Face detection (face-api.js) | ✅ Reuse | ❌ Replace (YOLOv8) |
| 68-point landmarks | ✅ Reuse | ✅ Reuse (per-person crop) |
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
| Genkit AI flow | ❌ Remove | ❌ Remove |

---

## 10. Post-Upgrade: Correctness, Security & CI (Status Update)

Everything above this section describes the original design analysis and upgrade plan. This section records what was actually built and fixed in subsequent passes, without removing or superseding the design rationale above — treat this as the current-state addendum.

### 10.1 Shared Scoring Core (v1.0 + v2.0)

The design above (§1's "Reuse Map") assumed `face-geometry.ts`/`ml-engine.ts` could simply be "reused directly" or "ported" as-is. In practice, three independent copies of this math (web app, Chrome extension, v2.0 Python port) drifted to different thresholds, weights, and formulas for the same signals — e.g. attention thresholds 70/40 (web) vs 80/50 (extension), gaze multiplier 2× vs 3×, and a head-pose pitch formula that normalized by nose-chin distance (which foreshortens as the head pitches, biasing pitch against itself) instead of the more stable interocular distance.

**Fix**: `v1.0/src/lib/scoring/` is now the single source of truth for all face-geometry math. `face-geometry.ts`/`ml-engine.ts` are re-export shims over it (existing imports unchanged). The Chrome extension's plain-JS `ml-core.js` has its formulas hand-synced to match (documented in its header comment — an automated build step is a scoped future improvement). `v2.0/features/face_metrics.py` is kept in sync via a documented golden-fixture equivalence contract rather than code-generation. Porting this fix surfaced a real, previously-undocumented bug: **v2.0 never computed gaze at all** — `room_analyzer.py` hardcoded `gaze_x=None, gaze_y=None`, silently defaulting to "looking dead center" downstream.

Also added: `v1.0/src/lib/scoring/face-selection.ts`'s `PrimaryFaceSelector`, replacing the `detections[0]` hardcoding flagged as Limitation 5 above — locks onto the largest, most-centered face with identity-continuity hysteresis (v2.0 doesn't need this; ByteTrack already provides persistent per-person track IDs).

Full account: [v1.0 README §20](v1.0/README.md#20-shared-scoring-core--cross-language-consistency).

### 10.2 Cross-Modal Contradiction Vector (New Novel Contribution)

Beyond the novel algorithms already claimed in [README-overall.md](README-overall.md#novel-algorithmic-contributions) (Context-Gate Override, Asymmetric ConfusionPenaltyLoss, AR(1) Temporal Fusion, Prioritized Seat Mapping, Fatigue Onset Detection), a new additive contribution was designed and implemented: the **Cross-Modal Contradiction Vector (CMCV)**. Where the Context-Gate Override is a rule that fires on one known confusion (phone vs. reading) and discards the disagreement magnitude, CMCV is a general, symmetric, per-frame disagreement statistic over all modality pairs, retained as a first-class signal (not folded back into the attention score) and classified into an actionable taxonomy (`fatigue_masking`, `prop_without_engagement`, `context_conflict`, `coherent`).

Implemented identically in both `v1.0/src/lib/contradiction-detector.ts` and `v2.0/inference/contradiction_detector.py`, cross-verified to produce matching categorizations on identical inputs. Computationally free (reuses already-computed sub-scores, zero new model inference) and unit-tested against constructed cases; full empirical validation is gated on labeled modality-conflict data that doesn't yet exist (see [v1.0 README §21](v1.0/README.md#21-cross-modal-contradiction-vector-novel) for the complete spec, integration points, and honest validation-status assessment).

### 10.3 Research-Integrity Fix: Real Training/Evaluation Pipeline

README-overall.md's headline benchmark numbers (86.5% accuracy, F1 0.852, a 55,100-sequence dataset) were not backed by any real training run. Worse, `v2.0/evaluation/evaluate.py` was found to **actively fabricate results** — synthesizing predictions (`pred = label if rng.random() > 0.2 else random_class`) whenever no real dataset path was supplied — and the one real training run that did exist was evaluated circularly (trained and tested on the same synthetic AR(1) generator). This has been addressed in two parts:

1. **Fabrication removed**: `evaluate.py` now reports `"status": "SKIPPED"` instead of a synthesized score when no real dataset is available.
2. **Pure-code training-pipeline bugs fixed** (no external dataset access needed): a live train/inference feature-encoding mismatch (`[0,1]` vs `[-1,1]` for gaze/pitch/yaw across different modules), a train/val leakage bug (naive random split instead of grouping by clip/subject), an incomplete ConfusionPenaltyLoss matrix (3 of 30 pairs specified), and a Temporal Consistency Coefficient that could not actually distinguish flicker from real transitions (it never tracked ground truth, only predictions). All fixed with regression tests. Full details: [v2.0 README §23](v2.0/README.md#23-training-pipeline-correctness-fixes).

**Still blocked** on external access (unchanged from before this pass): acquiring real DAiSEE/EngageNet data, recording/licensing classroom footage, and running real ablations against a genuine held-out test set. README-overall.md's numbers remain explicitly labeled as targets, not measured results, until that work happens.

### 10.4 Security Hardening (v2.0 Server)

A hardening pass on `v2.0/server/` fixed: a path-traversal/injection vector in `room_id` handling (unvalidated path parameter flowing into a filesystem path — now validated against `^[a-zA-Z0-9_-]+$` at both the API layer and defense-in-depth inside `config.py`), unhandled model-loading crashes (missing `.pt`/`.onnx` files previously surfaced as opaque stack traces — now a clear `FileNotFoundError` pointing at `download_models.py`), incorrect HTTP status codes (`404` cases were returning `200` with an error body), missing WebSocket exception handling (an unhandled exception could crash the streaming task silently), and unbounded concurrent-room creation (now capped via `COGNEN_MAX_CONCURRENT_ROOMS`, default 20). Full details: [v2.0 README §24](v2.0/README.md#24-server-hardening).

### 10.5 Dependency & Code-Quality Hygiene

- **v1.0**: `npm audit` findings reduced from 107 (6 critical, 22 high) to 35 (0 critical, 0 high) via `npm audit fix`, targeted version bumps, and `package.json` overrides. Redundant `yarn.lock` (causing GitHub to double-scan the same dependency tree) removed. ESLint config added (none existed before — `next.config.ts` was also silently swallowing type/lint errors at build time via `ignoreBuildErrors`/`ignoreDuringBuilds`, both now removed). React error boundaries (`error.tsx`, `global-error.tsx`) added. Chrome extension `host_permissions` narrowed (removed an unnecessary `<all_urls>` grant). Firebase config moved from hardcoded values to env vars (not a secret fix — Firebase Web SDK keys are meant to be public — but closes a real inconsistency where `next.config.ts` declared env vars nothing populated). Full details: [v1.0 README §22–23](v1.0/README.md#22-testing-linting--ci).
- **v2.0**: `ruff`/`mypy` config added (none existed before). First mypy pass found 55 issues; fixing them surfaced one genuine regression (`evaluate.py` still unpacking a 2-tuple after a loader interface change elsewhere had moved to 3-tuples) and one worthwhile structural fix (`_ClassSpec` converted from an unverifiable `**kwargs`/`setattr` pattern to a typed dataclass, resolving 20 findings at once); 26 pre-existing, individually-verified-benign findings remain and are surfaced as an informational (non-blocking) CI check rather than forced through a larger speculative rewrite. `requirements.txt` pinned with `==` (every pin verified to exist on PyPI). Full details: [v2.0 README §25](v2.0/README.md#25-code-quality-testing--ci).

### 10.6 Continuous Integration

`.github/workflows/ci.yml` (one file, shared by both sub-projects, path-filtered so a v1.0-only change doesn't trigger v2.0 jobs and vice versa) runs on every push/PR to `main`: for v1.0 — typecheck, lint, test, build, `npm audit --audit-level=high`; for v2.0 — `ruff check`, `mypy` (informational), the lightweight pytest tier, `pip-audit`; repo-wide — secret scanning (`gitleaks`). A single `all-checks-passed` job aggregates every job's result (treating skipped jobs as passing) so branch protection only needs to reference one required check.
