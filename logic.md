# CognEn - Technical Architecture & ML Logic

## Overview

CognEn (Cognitive Engagement) uses computer vision and machine learning to analyze facial features in real-time and compute an **attention score** from 0-100. This document details the mathematical and ML foundations.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│  Webcam Feed (640×480 @ 5 FPS)                                  │
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
│  2. Apply weighted combination (12 features)                    │
│  3. Temporal smoothing via EMA (α = 0.25)                       │
│  4. Classify: Attentive (≥75) | Distracted (45-74) | Low (<45) │
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
Yaw   = ((nose.x - eye_midpoint.x) / eye_distance) × 60°
Pitch = ((nose.y - eye_midpoint.y) / nose_chin_distance) × 60°
Roll  = atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
```

**Stability Score**: Variance of pose over 30 seconds.

```
Stability = 100 - (σ²_yaw + σ²_pitch + 0.5×σ²_roll) / 5
```

### 5. Gaze Direction

Estimates where the user is looking based on iris position within eye.

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
| Gaze Deviation | 0.18 | Primary attention indicator |
| Yaw Deviation | 0.12 | Looking left/right |
| Head Pose Stability | 0.12 | Overall engagement |
| EAR | 0.12 | Eye openness |
| PERCLOS | 0.10 | Drowsiness |
| Pitch Deviation | 0.08 | Looking up/down |
| Fatigue Index | 0.08 | Accumulated fatigue |
| Blink Rate | 0.06 | Eye strain indicator |
| Fidget Score | 0.05 | Restlessness |
| Face Presence | 0.03 | Face in frame |
| Face Distance | 0.02 | Proximity to camera |
| Mouth Openness | 0.02 | Yawning detection |
| Eyebrow Raise | 0.02 | Surprise/confusion |

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

With **α = 0.25**, the system responds to changes within 1-2 seconds while filtering noise.

---

## Attention State Classification

| Score Range | State | Description |
|-------------|-------|-------------|
| 75-100 | **Attentive** | Focused, looking at screen, stable head position |
| 45-74 | **Distracted** | Occasional glances away, some movement |
| 0-44 | **Multitasking** | Looking away, excessive movement, or no face |

---

## Extension Architecture

### Component Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Popup.js      │────▶│  Background.js  │────▶│  Content.js     │
│   (UI)          │     │  (Coordinator)  │     │  (Meet Inject)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                        ┌─────────────────┐              │
                        │  Offscreen.js   │◀─────────────┘
                        │  (Standalone)   │   (or inject into Meet)
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │  ml-core.js     │
                        │  (ML Engine)    │
                        └─────────────────┘
```

### Data Flow

1. **Detection**: face-api.js extracts 68 facial landmarks
2. **Metrics**: `FaceMetricsAggregator` computes all features
3. **Prediction**: `MLEngine` produces attention score and state
4. **Storage**: Results written to localStorage (Meet mode) or chrome.storage
5. **Display**: Widget on page + popup UI read from storage

### Dual Mode Operation

| Mode | When | How |
|------|------|-----|
| **Meet Mode** | On Google Meet tab | Injects into page, reads existing video element |
| **Standalone Mode** | Not on Meet | Uses offscreen document with own camera access |

---

## Performance Considerations

- **Frame Rate**: 5 FPS target (~200ms intervals)
- **Detection Model**: TinyFaceDetector (320px input, ~150ms inference)
- **Memory**: ~50MB for face-api models
- **CPU Usage**: ~15-25% on modern hardware

---

## Accuracy Limitations

1. **Lighting**: Poor lighting reduces landmark accuracy
2. **Angle**: Extreme head angles (>40°) lose tracking
3. **Glasses**: Reflective glasses may affect gaze estimation
4. **Multiple Faces**: Only first detected face is tracked
5. **Camera Quality**: Low-res cameras reduce accuracy

---

## References

- **EAR Formula**: Soukupová, T., & Čech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks
- **PERCLOS**: Dinges, D. F., & Grace, R. (1998). PERCLOS: A Valid Psychophysiological Measure of Alertness
- **face-api.js**: https://github.com/vladmandic/face-api
- **68-Point Landmarks**: iBUG 300-W dataset standard
