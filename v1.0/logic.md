# CognEn - Technical Architecture & ML Logic

> **Status note.** This file describes the legacy scoring path (DACE off) of the v1.0 web app and has been corrected to match `src/lib/scoring/constants.ts`. Authoritative sources: `src/lib/scoring/` (code), `ARCHITECTURE.md` (design; DACE in §11) and `v1.0/README.md`. Corrected here: EMA alpha 0.15 (web app and Chrome extension; the extension was rewritten to match), attention thresholds 70/40, head-pose gains/normalizer, legacy feature weights, and gaze (eyelid-based, not iris-based). Known defect: the legacy pitch value has no neutral offset and rises (not falls) when the head tilts down, so pitch-based "looking down" logic is unreliable; the opt-in DACE mode (README §24) addresses this. Nothing in DACE has been browser-tested.

## Overview

CognEn (Cognitive Engagement) uses computer vision and machine learning to analyze facial features in real-time and compute an **attention score** from 0-100. This document details the mathematical and ML foundations.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│  Webcam Feed (640×480; fixed 200 ms ticks, prediction recorded at 1 Hz)  │
│         ↓                                                        │
│  face-api.js TinyFaceDetector                                   │
│         ↓                                                        │
│  68-Point Facial Landmark Detection                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Extraction                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Eye Metrics  │  │ Head Pose    │  │ Temporal     │          │
│  │ • EAR        │  │ • Yaw        │  │ • PERCLOS    │          │
│  │ • MAR        │  │ • Pitch      │  │ • Blink Rate │          │
│  │ • Gaze (X,Y) │  │ • Roll       │  │ • EMA Score  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      ML Engine                                   │
├─────────────────────────────────────────────────────────────────┤
│  1. Normalize each feature to 0-100 score                       │
│  2. Apply weighted combination (13 legacy features; 16 in the upgraded weight table)                    │
│  3. Temporal smoothing via EMA (α = 0.15 web app)                       │
│  4. Classify: Attentive (≥70) | Distracted (40-69) | Multitasking (<40) │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Extraction

### 1. Eye Aspect Ratio (EAR)

Measures eye openness using the 6 landmarks per eye.

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```

Where `p1-p6` are the 6 eye landmarks (lateral corners and upper/lower lids).

| EAR Value | Interpretation |
|-----------|---------------|
| 0.30-0.40 | Fully open |
| 0.25-0.30 | Normal |
| 0.21-0.25 | Partially closed |
| < 0.21    | Blink/closed |

**Blink Detection**: A blink is registered when:
- EAR drops below 0.21 (eye closed)
- Then rises back above 0.21 (eye opens)
- Duration: 50-500ms (filters involuntary closures)

### 2. PERCLOS (Percentage Eye Closure)

Standard drowsiness metric from the automotive industry.

```
PERCLOS = (frames with EAR < 0.21) / (total frames in last 60s) × 100%
```

| PERCLOS | Interpretation |
|---------|---------------|
| < 15%   | Alert |
| 15-30%  | Mildly fatigued |
| > 30%   | Drowsy |

### 3. Mouth Aspect Ratio (MAR)

Detects yawning and speaking states.

```
MAR = ||top_lip - bottom_lip|| / ||left_corner - right_corner||
```

| MAR Value | Interpretation |
|-----------|---------------|
| < 0.10    | Mouth closed |
| 0.10-0.25 | Speaking |
| > 0.25    | Yawning |

### 4. Head Pose Estimation

Computed from facial landmark geometry without 3D model.

```
Yaw   = ((nose.x - eye_midpoint.x) / eye_distance) × 50      (YAW_GAIN)
Pitch = ((nose.y - eye_midpoint.y) / eye_distance) × 33.3    (PITCH_GAIN; same interocular normalizer; not true degrees, no neutral offset)
Roll  = atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
```

**Stability Score**: Variance of pose over 30 seconds.

```
Stability = 100 - (σ²_yaw + σ²_pitch + 0.5×σ²_roll) / 2.5 / 10
```

### 5. Gaze Direction

Approximates where the user is looking from eyelid landmarks (the 68-point model has no iris points, so this is not true iris gaze; DACE mode uses MediaPipe blendshape gaze instead).

```
Gaze_X = (iris_center.x - eye_center.x) / eye_width × 2
Gaze_Y = (iris_center.y - eye_center.y) / eye_height × 2
Gaze_Deviation = √(Gaze_X² + Gaze_Y²)
```

| Gaze Deviation | Interpretation |
|----------------|---------------|
| < 0.15         | Looking at screen |
| 0.15-0.30      | Slight glance away |
| 0.30-0.50      | Looking away |
| > 0.50         | Not focused on screen |

### 6. Blink Rate

Computed as blinks per minute, scaled from available data.

```
Blink_Rate = (blinks_in_window / window_duration_ms) × 60000
```

| Rate (bpm) | Interpretation |
|------------|---------------|
| 12-20      | Normal |
| < 12       | Staring (eye strain) |
| > 20       | Fatigue indicator |

### 7. Fidget Score

Measures head movement intensity over time.

```
Movement = Σ|Δyaw| + |Δpitch| + |Δroll| over recent frames
Fidget_Score = clamp(Movement × 8, 0, 100)
```

### 8. Fatigue Index

Composite metric combining multiple fatigue indicators.

```
Fatigue_Index = PERCLOS_score + Blink_rate_score + EAR_trend_score
```

Components:
- **PERCLOS Score**: `clamp(PERCLOS × 2.5, 0, 50)`
- **Blink Rate Score**: Penalty if outside 15-20 bpm range
- **EAR Trend Score**: Penalty if average EAR declining

---

## ML Engine: Weighted Feature Scoring

Each feature is normalized to a 0-100 score, then combined with weights.

### Feature Weights

| Feature | Weight | Rationale |
|---------|--------|-----------|
| EAR | 0.15 | Eye openness |
| PERCLOS | 0.12 | Drowsiness |
| Head Pose Stability | 0.12 | Overall engagement |
| Gaze Deviation | 0.10 | Primary attention indicator |
| Fatigue Index | 0.10 | Accumulated fatigue |
| Blink Rate | 0.08 | Eye strain indicator |
| Yaw Deviation | 0.08 | Looking left/right |
| Fidget Score | 0.07 | Restlessness |
| Pitch Deviation | 0.06 | Looking up/down |
| Eyebrow Raise | 0.04 | Surprise/confusion |
| Mouth Openness | 0.03 | Yawning detection |
| Face Distance | 0.03 | Proximity to camera |
| Face Presence | 0.02 | Face in frame |

These are `FEATURE_WEIGHTS_LEGACY`. When object detection data is present the engine uses `FEATURE_WEIGHTS_V1_UPGRADED` (adds contextAwarePitch, objectContext, handObjectProximity, temporalPattern; see v1.0/README.md §4.2). The score is the weighted mean times a quality factor.

### Score Calculation

```javascript
// For each feature
score_i = scoreFeature(feature_name, value)  // Returns 0-100
quality_i = getQuality(feature_name, value)  // 0-1 confidence

// Weighted combination
raw_score = Σ(score_i × weight_i × quality_i) / Σ(weight_i × quality_i)
```

### Temporal Smoothing (EMA)

To reduce jitter, scores pass through an Exponential Moving Average:

```
smoothed_score = α × raw_score + (1 - α) × prev_smoothed_score
```

With **α = 0.15** (web app and extension), the system responds to changes within 1-2 seconds while filtering noise.

---

## Attention State Classification

| Score Range | State | Description |
|-------------|-------|-------------|
| 70-100 | **Attentive** | Focused, looking at screen, stable head position |
| 40-69 | **Distracted** | Occasional glances away, some movement |
| 0-39 | **Multitasking** | Looking away, excessive movement, or no face |

---

## Extension Architecture

Details: `extension/README.md`. Unit-tested only (`extension-equivalence.test.ts`); never loaded in Chrome.

### Component Flow

```
Popup.js (UI) ──▶ Background.js (service worker: session record, badge, alarms, notifications)
                      │
         ┌────────────┴─────────────────────────────┐
         ▼ standalone                               ▼ Meet / Zoom / Teams
   Offscreen.js (own camera,                  Content.js (MAIN world, analyser + widget)
   200 ms setTimeout loop)                          │ HMAC-signed postMessage
         │                                          ▼
         └──── both use lib/ml-core.js        Bridge.js (ISOLATED world: verify MAC,
               (legacy-path port)                    replay check) ──▶ chrome.runtime ──▶ Background.js
```

### Data Flow

1. **Detection**: face-api.js extracts 68 facial landmarks (200 ms tick)
2. **Metrics**: the `ml-core.js` aggregator computes all features
3. **Prediction**: the `ml-core.js` engine (legacy weights, EMA 0.15, thresholds 70/40) produces score and state once per second
4. **Storage**: the background owns the session record (`chrome.storage.session`, falling back to `local`); the page-`localStorage` channel no longer exists
5. **Display**: the in-page widget (call mode) and the popup read from the background

### Dual Mode Operation

| Mode | When | How |
|------|------|-----|
| **Meet / Zoom / Teams mode** | On a supported call tab | Injects the analyser into the page and analyses the self-view `<video>` (live local-capture track, else mirrored element, else largest) |
| **Standalone Mode** | Any other tab | Uses an offscreen document with its own camera access |

---

## Performance Considerations

- **Frame Rate**: web app (legacy and DACE) and extension all tick every 200 ms (5 FPS design rate, not measured) and record a prediction once per second
- **Detection Model**: TinyFaceDetector (320px input, ~150ms inference)
- **Memory**: ~50MB for face-api models
- **CPU Usage**: ~15-25% on modern hardware

---

## Accuracy Limitations

1. **Lighting**: Poor lighting reduces landmark accuracy
2. **Angle**: Extreme head angles (>40°) lose tracking
3. **Glasses**: Reflective glasses may affect gaze estimation
4. **Multiple Faces**: the web app tracks one primary face chosen by `PrimaryFaceSelector` (largest/most centered, with hysteresis); the extension uses the same `PrimaryFaceSelector` hysteresis (ported)
5. **Camera Quality**: Low-res cameras reduce accuracy

---

## References

- **EAR Formula**: Soukupová, T., & Čech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks
- **PERCLOS**: Dinges, D. F., & Grace, R. (1998). PERCLOS: A Valid Psychophysiological Measure of Alertness
- **face-api.js**: https://github.com/vladmandic/face-api
- **68-Point Landmarks**: iBUG 300-W dataset standard
