# CognEn / AttentionTrackerOSS: Open-Weights Model Tuning & Parameter Calibration Guide

This document is the definitive reference for fine-tuning, calibrating, and optimizing all open-weights models and algorithmic modules within the CognEn / AttentionTrackerOSS ecosystem. Follow these parameter specifications, search ranges, and recipes to achieve state-of-the-art accuracy, eliminate domain shift, prevent identity leakage, and optimize runtime performance across both edge and server environments.

---

## Table of Contents

1. [Architectural Overview & Model Hierarchy](#1-architectural-overview--model-hierarchy)
2. [YOLOv8 Open-Weights Vision Backbone Tuning](#2-yolov8-open-weights-vision-backbone-tuning)
   - [2.1 Models & Detection Tasks](#21-models--detection-tasks)
   - [2.2 Complete Hyperparameter Dictionary](#22-complete-hyperparameter-dictionary)
   - [2.3 Augmentation Parameters for Classroom/Desk Occlusion](#23-augmentation-parameters-for-classroomdesk-occlusion)
   - [2.4 Layer Freezing & Transfer Learning Strategy](#24-layer-freezing--transfer-learning-strategy)
   - [2.5 Concrete Training Recipe (Python & CLI)](#25-concrete-training-recipe-python--cli)
3. [1D-CNN Temporal Behavior Classifier Tuning](#3-1d-cnn-temporal-behavior-classifier-tuning)
   - [3.1 Model Architecture & Input Structure](#31-model-architecture--input-structure)
   - [3.2 Training & Optimization Hyperparameters](#32-training--optimization-hyperparameters)
   - [3.3 Asymmetric ConfusionPenaltyLoss Matrix](#33-asymmetric-confusionpenaltyloss-matrix)
   - [3.4 Class Imbalance & Label Smoothing](#34-class-imbalance--label-smoothing)
   - [3.5 Preventing Train/Val Group Leakage](#35-preventing-trainval-group-leakage)
   - [3.6 Synthetic Data Calibration Parameters](#36-synthetic-data-calibration-parameters)
4. [Multi-Object Tracking (ByteTrack) Parameter Calibration](#4-multi-object-tracking-bytetrack-parameter-calibration)
   - [4.1 Association & Track Retention Parameters](#41-association--track-retention-parameters)
   - [4.2 Kalman Filter & Noise Covariance Tuning](#42-kalman-filter--noise-covariance-tuning)
   - [4.3 Seat-Mapping Spatial Assignment Thresholds](#43-seat-mapping-spatial-assignment-thresholds)
5. [Facial Landmark, Bio-Signal & DACE Tuning Parameters](#5-facial-landmark-bio-signal--dace-tuning-parameters)
   - [5.1 Head Pose & Pitch Calibration](#51-head-pose--pitch-calibration)
   - [5.2 Eye Openness & Huber Regression Parameters](#52-eye-openness--huber-regression-parameters)
   - [5.3 Gaze-Corrected PERCLOS & Fatigue Metrics](#53-gaze-corrected-perclos--fatigue-metrics)
   - [5.4 Hand-Object Interaction & Pinch Mechanics](#54-hand-object-interaction--pinch-mechanics)
6. [Open-Weights Multimodal LLM / VLM Fine-Tuning (Gemma / Llama / Qwen)](#6-open-weights-multimodal-llm--vlm-fine-tuning-gemma--llama--qwen)
   - [6.1 Role in CognEn (Session Reports & Analytical Explanations)](#61-role-in-cognen-session-reports--analytical-explanations)
   - [6.2 LoRA / QLoRA Fine-Tuning Hyperparameters](#62-lora--qlora-fine-tuning-hyperparameters)
   - [6.3 Generation & Structured Output Decoding Parameters](#63-generation--structured-output-decoding-parameters)
7. [Systematic Tuning Strategy for Maximum Accuracy](#7-systematic-tuning-strategy-for-maximum-accuracy)
   - [7.1 Optuna Bayesian Hyperparameter Search](#71-optuna-bayesian-hyperparameter-search)
   - [7.2 Stratified Group K-Fold Cross-Validation](#72-stratified-group-k-fold-cross-validation)
   - [7.3 Probability Calibration (Temperature Scaling)](#73-probability-calibration-temperature-scaling)
   - [7.4 Hard-Negative Mining Pipeline](#74-hard-negative-mining-pipeline)
8. [Export, Quantization & Edge Inference Calibration](#8-export-quantization--edge-inference-calibration)
   - [8.1 ONNX Export Parameters](#81-onnx-export-parameters)
   - [8.2 INT8 Post-Training Quantization (PTQ) Calibration](#82-int8-post-training-quantization-ptq-calibration)
9. [Master Parameter Reference Matrix](#9-master-parameter-reference-matrix)

---

## 1. Architectural Overview & Model Hierarchy

CognEn / AttentionTrackerOSS processes attention dynamics through a multi-tier neural-symbolic pipeline:

```
[Raw Video Stream: RTSP CCTV / Browser 5-15 FPS]
     │
     ├─ Tier 1: Perception Backbones (Open-Weights Ultralytics YOLOv8)
     │    ├── YOLOv8-m       → Classroom Person Detection & Bounding Boxes
     │    ├── YOLOv8-s       → Study Context Objects (Book, Notebook, Phone, Laptop, Pen)
     │    └── YOLOv8-m-pose  → 17 Keypoints (Body Posture, Slump, Wrist Coordinates)
     │
     ├─ Tier 2: Tracking & Spatial Association
     │    └── ByteTrack (boxmot) → Identity Preservation, Occlusion Recovery, Seat Mapping
     │
     ├─ Tier 3: Bio-Signal & Facial Geometry (MediaPipe / DACE)
     │    ├── MediaPipe Face Landmarker → 478 Landmarks, Blendshapes, Transformation Matrix
     │    ├── Relative Pitch Decomposition → Screen-Facing Baseline Normalization
     │    └── Huber-Fitted Openness / PERCLOS → Drowsiness & Gaze Correction
     │
     ├─ Tier 4: Temporal Behavior Classification (CognEn 1D-CNN)
     │    └── 15-Frame Sliding Window (5 Hz) → 15-Dim Features → 6 Behavior Classes
     │
     └─ Tier 5: Session Intelligence & Reporting (Open-Weights LLM / VLM)
          └── Gemma 2/3 / Llama 3.2 Vision (LoRA) → Structured JSON & Executive Summary
```

Fine-tuning open weights directly adapts these models from general-purpose benchmarks (COCO, ImageNet) to the specific challenges of academic environments: desk occlusion, extreme camera angles, hand-held object confusion, and micro-movements.

---

## 2. YOLOv8 Open-Weights Vision Backbone Tuning

### 2.1 Models & Detection Tasks

| Model Variant | Base Open Weights | Target Task | Input Size | Target Classes |
|---|---|---|---|---|
| **YOLOv8-m** | `yolov8m.pt` | Multi-person detection in CCTV | 1280×1280 (or 640×640) | `person` (Class 0) |
| **YOLOv8-s** | `yolov8s.pt` | Study-context object detection | 640×640 | `book`, `notebook`, `phone`, `laptop`, `pen`, `tablet` |
| **YOLOv8-m-pose** | `yolov8m-pose.pt` | Multi-person pose & wrist tracking | 640×640 | 17 Keypoints (Wrists, Shoulders, Nose, Eyes, Ears) |
| **YOLOv8-nano** | `yolov8n.pt` / `yolov8n.onnx` | Browser-side client detection (v1.0) | 640×640 | `book`, `phone`, `laptop` |

### 2.2 Complete Hyperparameter Dictionary

When fine-tuning YOLOv8 weights on custom classroom footage (e.g., Open Images v7 subset, CVAT-annotated classroom videos), use the following configuration parameters:

| Hyperparameter | Default | Recommended Fine-Tuning | Search Range | Description & Impact on Accuracy |
|---|---|---|---|---|
| `lr0` | `0.01` | `0.001` (SGD) / `0.0005` (AdamW) | `1e-4` – `5e-3` | Initial learning rate. Lower values preserve pre-trained backbone representations. |
| `lrf` | `0.01` | `0.01` | `0.001` – `0.1` | Final learning rate ratio (`lr_min = lr0 * lrf`). Uses cosine decay schedule. |
| `optimizer` | `auto` | `AdamW` | `AdamW`, `SGD` | `AdamW` converges significantly faster and more reliably on small domain-specific object sets. |
| `momentum` | `0.937` | `0.95` (SGD) / `0.90` (AdamW) | `0.85` – `0.98` | Gradient momentum / running average beta1. |
| `weight_decay` | `0.0005` | `0.001` | `1e-4` – `5e-3` | L2 penalty preventing overfitting on repetitive classroom backgrounds. |
| `warmup_epochs` | `3.0` | `5.0` | `2.0` – `10.0` | Gradual LR increase at training start. Essential to prevent head gradients from distorting features. |
| `warmup_momentum`| `0.8` | `0.8` | `0.5` – `0.95` | Momentum during warmup phase. |
| `warmup_bias_lr` | `0.1` | `0.05` | `0.01` – `0.1` | Bias learning rate during warmup. |
| `box` | `7.5` | `8.5` | `5.0` – `10.0` | Bounding box regression loss gain (GIoU/CIoU). Higher weight yields tighter boxes for small pens/phones. |
| `cls` | `0.5` | `1.2` | `0.5` – `2.0` | Classification loss gain (BCE). Boosted to prevent confusing books with open laptops or phones with calculators. |
| `dfl` | `1.5` | `1.5` | `1.0` – `2.5` | Distribution Focal Loss gain for fine-grained boundary localization. |
| `pose` | `12.0` | `14.0` | `10.0` – `18.0` | Keypoint position loss weight (pose models only). Ensures high precision on wrists and nose. |
| `kobj` | `1.0` | `1.5` | `0.8` – `2.0` | Keypoint objectness loss gain. Helps detect partially occluded wrists under desks. |
| `imgsz` | `640` | `1280` (CCTV) / `640` (Desk) | `640` – `1280` | Image resolution. 1280 is mandatory for CCTV ceiling feeds to resolve small pens and student smartphones. |
| `batch` | `16` | `16` or `32` | `8` – `64` | Batch size. If using batch size < 16, synchronize BatchNorm across GPUs or freeze BN layers. |
| `epochs` | `100` | `150` | `80` – `300` | Training epochs. Combined with early stopping patience. |
| `patience` | `100` | `25` | `15` – `40` | Early stopping epochs without validation mAP improvement. |

### 2.3 Augmentation Parameters for Classroom/Desk Occlusion

Classroom CCTV frames present high perspective distortion, lighting variations (projectors, windows), and frequent hand/desk occlusions. Standard augmentations must be tuned specifically:

| Augmentation Parameter | Value | Rationale for Classroom / Attention Tracking |
|---|---|---|
| `mosaic` | `0.5` (or `0.0` for CCTV) | Full 1.0 mosaic harms multi-scale context in static room cameras. Reduce to `0.5` or `0.0` if using fixed CCTV. |
| `close_mosaic` | `15` | Disables mosaic augmentation during the final 15 epochs. **Critical** for fine-tuning anchor boundaries and improving validation mAP. |
| `mixup` | `0.1` | Mixes two images with alpha blending. Low value prevents ghosting on small objects. |
| `copy_paste` | `0.2` | Copies segment masks into other images. Effective for multiplying rare `pen` and `smartphone` instances. |
| `degrees` | `10.0` | Small rotation range (students do not sit upside down; limits synthetic distortions). |
| `translate` | `0.1` | Simulates slight camera shifts and student positional variance. |
| `scale` | `0.4` | Scale jitter simulating different seating distances from ceiling cameras. |
| `shear` | `2.0` | Accommodates perspective tilt of wall/ceiling-mounted cameras. |
| `perspective` | `0.0005` | Mild perspective transformation. |
| `hsv_h` | `0.015` | Hue adjustment for varying fluorescent and ambient classroom lighting. |
| `hsv_s` | `0.4` | Saturation adjustment. |
| `hsv_v` | `0.3` | Value/brightness adjustment (compensates for projector dimming). |
| `flipud` | `0.0` | Strictly `0.0` (students are never upside down). |
| `fliplr` | `0.5` | Horizontal flip is valid (desks and handedness are bilateral). |
| `erasing` | `0.3` | Random erasing. Forces network to identify pens/phones even when partially occluded by hands. |

### 2.4 Layer Freezing & Transfer Learning Strategy

When fine-tuning on a small custom dataset (< 5,000 annotated classroom frames):
1. **Stage 1 (Head Only - Epochs 1 to 20):** Freeze the backbone and neck (`freeze=15` in Ultralytics). Optimize only the newly initialized detection heads with `lr0=0.001`.
2. **Stage 2 (Neck + Head - Epochs 21 to 80):** Unfreeze down to layer 10 (`freeze=10`). Reduce learning rate to `lr0=0.0003`.
3. **Stage 3 (Full End-to-End Fine-Tuning - Epochs 81 to 120):** Unfreeze all layers (`freeze=0`). Use `lr0=0.00005` with Cosine Annealing to gently adapt backbone convolution kernels to lens distortion.

### 2.5 Concrete Training Recipe (Python & CLI)

```python
# train_yolo_objects.py
from ultralytics import YOLO

# Load pre-trained open weights
model = YOLO("yolov8s.pt")

# Fine-tune on classroom objects dataset
results = model.train(
    data="classroom_objects.yaml",
    epochs=120,
    patience=25,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=0.0008,
    lrf=0.01,
    weight_decay=0.001,
    warmup_epochs=5,
    box=8.5,
    cls=1.2,
    dfl=1.5,
    mosaic=0.5,
    close_mosaic=15,
    copy_paste=0.2,
    erasing=0.3,
    degrees=10.0,
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.3,
    fliplr=0.5,
    flipud=0.0,
    freeze=10,            # Freeze backbone for initial epochs
    save=True,
    project="runs/cognen_objects",
    name="yolov8s_study_v1",
)
```

---

## 3. 1D-CNN Temporal Behavior Classifier Tuning

### 3.1 Model Architecture & Input Structure

The temporal behavior classifier (`v2.0/training/model.py`) converts sequential feature vectors into one of 6 behavioral states:
- **Input Tensor Dimensions**: `(batch_size, FEATURE_DIM, WINDOW_SIZE)` = `(B, 15, 15)`
  - Window Size: 15 consecutive frames (3.0 seconds at 5 Hz)
  - Feature Dimension: 15 channels (FEATURE_VECTOR_VERSION 2)
- **Feature Channels**:
  1. `gaze_x` (Horizontal gaze offset from camera center)
  2. `gaze_y` (Vertical gaze: down = negative)
  3. `pitch_norm` (Relative head pitch / 90: down = negative)
  4. `yaw_norm` (Head yaw / 90)
  5. `ear` (Eye Aspect Ratio, gaze-corrected)
  6. `perclos` (Percentage of Eye Closure over window)
  7. `head_pose_stability` (Exponential moving variance of head pose)
  8. `obj_book` (Presence/IoU of book or notebook)
  9. `obj_phone` (Presence/IoU of smartphone)
  10. `obj_laptop` (Presence/IoU of laptop screen)
  11. `obj_none` (No study object in proximity)
  12. `fidget` (High-frequency micro-movement coefficient)
  13. `hand_proximity_code` (0: none, 1: near desk, 2: holding pen/writing, 3: holding phone)
  14. `posture_upright` (Binary indicator: torso inclination < 15°)
  15. `posture_lean_norm` (Continuous lean angle / 90)
- **Output Classes** (6):
  `0: attentive_screen`, `1: attentive_reading`, `2: using_phone`, `3: distracted`, `4: drowsy`, `5: absent`.

### 3.2 Training & Optimization Hyperparameters

| Hyperparameter | Flag / Parameter | Default | Recommended Optimal | Search Space | Effect on Convergence |
|---|---|---|---|---|---|
| Epochs | `--epochs` | `50` | `60` | `40` – `100` | Full convergence with early stopping |
| Batch Size | `--batch` | `64` | `64` | `32` – `128` | Balance between gradient variance and convergence speed |
| Initial Learning Rate | `--lr` | `1e-3` | `1e-3` | `3e-4` – `3e-3` | Initial learning rate for `AdamW` |
| Scheduler | `CosineAnnealingLR` | `T_max=epochs` | `T_max=epochs, eta_min=1e-5` | `eta_min: 1e-6 - 1e-4` | Smooth annealing prevents late oscillations |
| Weight Decay | `weight_decay` | `1e-4` | `1e-4` | `1e-5` – `1e-3` | L2 regularization prevents overfitting on synthetic patterns |
| Gradient Clipping | `clip_grad_norm_` | `1.0` | `1.0` | `0.5` – `5.0` | Prevents explosive gradients during batch transitions |
| Label Smoothing | `label_smoothing` | `0.05` | `0.05` | `0.0` – `0.10` | Mitigates overconfidence on ambiguous boundary frames |
| Early Stopping Patience | `--patience` | `10` | `15` | `8` – `20` | Stops training when validation macro-F1 ceases improving |
| Validation Ratio | `--val_ratio` | `0.15` | `0.15` | `0.10` – `0.20` | Fraction reserved for evaluation (subject-disjoint) |

### 3.3 Asymmetric ConfusionPenaltyLoss Matrix

Standard cross-entropy treats all classification errors equally. In pedagogical attention tracking, falsely accusing a student reading a book of using a phone destroys trust, whereas confusing screen attention with slight distraction is benign.

The loss function multiplies the Cross-Entropy loss by an asymmetric penalty matrix $P \in \mathbb{R}^{6 \times 6}$ defined in `v2.0/training/penalty_matrix.py`:

$$\mathcal{L}_{\text{ConfusionPenalty}} = \frac{1}{B} \sum_{i=1}^B P[y_i, \hat{y}_i] \cdot \ell_{\text{CE}}(z_i, y_i)$$

Where rows are Ground Truth ($y_i$) and columns are Predictions ($\hat{y}_i$):

| Ground Truth \ Predicted | `attentive_screen` | `attentive_reading` | `using_phone` | `distracted` | `drowsy` | `absent` |
|---|---|---|---|---|---|---|
| **`attentive_screen`** | **1.0x** | 1.2x | **2.5x** | 1.2x | 1.5x | 2.0x |
| **`attentive_reading`**| 1.2x | **1.0x** | **3.0x** *(Max Penalty)* | 1.3x | 1.5x | 2.0x |
| **`using_phone`**      | **2.5x** | **2.5x** | **1.0x** | **2.5x** | 2.0x | 2.0x |
| **`distracted`**       | 1.2x | 1.3x | 2.0x | **1.0x** | 1.5x | 1.8x |
| **`drowsy`**           | 1.5x | 1.5x | 1.8x | 1.5x | **1.0x** | 2.0x |
| **`absent`**           | **2.0x** | **2.0x** | **2.0x** | 1.8x | 2.0x | **1.0x** |

**Penalty Hierarchy:**
- `3.0x`: Ground Truth = `reading`, Prediction = `using_phone`. Prevents penalizing reading due to downward head tilt.
- `2.5x`: Ground Truth = `using_phone`, Prediction = `attentive_reading` or `attentive_screen`. Prevents cheating detection failure.
- `2.0x`: Presence/Absence confusion in either direction (impacts attendance integrity).

### 3.4 Class Imbalance & Label Smoothing

Real datasets (DAiSEE, EngageNet) are heavily dominated by `attentive_screen` and `distracted`, with rare occurrences of `using_phone` or `absent`.

1. **Normalized Inverse Frequency Weighting:**
   Compute weights for class $c \in \{0 \dots 5\}$ present in the training batch:
   $$w_c = \frac{1}{N_c}, \quad \tilde{w}_c = w_c \cdot \frac{\sum_{k \in \text{Present}} 1}{\sum_{k \in \text{Present}} w_k}$$
   Absent classes receive weight $0.0$ to prevent skewing the normalizer.

2. **Label Smoothing ($\alpha = 0.05$):**
   Replaces one-hot vector $y_k$ with:
   $$y'_k = (1 - \alpha) y_k + \frac{\alpha}{K}$$
   Prevents the 1D-CNN from producing extreme uncalibrated logit outputs on ambiguous transition frames.

### 3.5 Preventing Train/Val Group Leakage

A common flaw in time-series evaluation is splitting windows randomly across train and validation sets. Consecutive 15-frame sliding windows from the same student are highly autocorrelated, causing artificial 99%+ accuracy that collapses in production.

**Strict Split Policy (`train_classifier.py`):**
- Split exclusively by **Participant / Sequence ID**, using `GroupShuffleSplit`.
- Augmented synthetic copies must inherit the exact parent `group_id`.
- The dataset loader executes `assert_no_group_leakage(train_groups, val_groups)`:
  ```python
  def assert_no_group_leakage(train_groups, val_groups):
      overlap = set(train_groups) & set(val_groups)
      if overlap:
          raise GroupLeakageError(f"Leaked {len(overlap)} subjects between train and val!")
  ```

### 3.6 Synthetic Data Calibration Parameters

When training with the class-conditional AR(1) synthetic generator (`RealisticSyntheticDataset`), calibrate the generative specifications (`_SPECS`) to match real biometric distributions:

$$x_t = \mu + \phi (x_{t-1} - \mu) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}\left(0, \sigma^2 (1 - \phi^2)\right)$$

| Class | Relative Pitch $\mu$ (/90) | Gaze $y$ $\mu$ | EAR $\mu$ | PERCLOS $\mu$ | Stability $\mu$ | Fidget $\mu$ | AR(1) $\phi$ |
|---|---|---|---|---|---|---|---|
| `attentive_screen` | `0.00` | `0.00` | `0.32` | `0.03` | `0.87` | `0.10` | `0.85` |
| `attentive_reading`| `-0.30` | `-0.55` | `0.30` | `0.04` | `0.82` | `0.08` | `0.80` |
| `using_phone`      | `-0.36` | `-0.60` | `0.30` | `0.04` | `0.76` | `0.28` | `0.70` |
| `distracted`       | `+0.03` | `0.00` | `0.31` | `0.05` | `0.48` | `0.38` | `0.65` |
| `drowsy`           | `-0.08` | `+0.02` | `0.18` | `0.32` | `0.58` | `0.04` | `0.90` |
| `absent`           | `0.00` | `0.00` | `0.30` | `0.00` | `0.00` | `0.00` | `0.00` |

---

## 4. Multi-Object Tracking (ByteTrack) Parameter Calibration

Multi-person tracking associates bounding boxes across frames without Re-ID models, using ByteTrack (via `boxmot 16.0+`).

### 4.1 Association & Track Retention Parameters

| Parameter | Location | Default | Optimal Range | Tuning Guidance & Impact |
|---|---|---|---|---|
| `track_thresh` | `byte_tracker.py` | `0.25` | `0.40` – `0.60` | Detection score threshold for first-stage association. Raise to `0.45` in busy rooms to eliminate false person detections. |
| `min_conf` / `low_thresh` | `byte_tracker.py` | `0.10` | `0.08` – `0.15` | Second-stage detection threshold. Recovers tracks through severe partial occlusions (e.g. student leaning behind desk). |
| `match_thresh` | `byte_tracker.py` | `0.80` | `0.70` – `0.85` | Hungarian matching IoU cost threshold for stage 1. Lower values prevent false track switching when students sit closely. |
| `track_buffer` | `byte_tracker.py` | `30` | `30` – `60` | Number of frames to retain lost tracks before deletion. At 15 FPS, 45 frames gives a 3-second grace period for occlusions. |
| `frame_rate` | `byte_tracker.py` | `15` | Source FPS | Frame rate used by Kalman filter for velocity integration. Must match incoming stream FPS exactly. |

### 4.2 Kalman Filter & Noise Covariance Tuning

ByteTrack relies on a constant-velocity Kalman filter tracking state vector:
$$x = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}, \dot{r}]^T$$
where $(u, v)$ is center coordinate, $s$ is bounding box area, and $r$ is aspect ratio.

- **Process Noise Covariance ($Q$):**
  - Increase positional variance weights if students change postures quickly (reduces tracking lag).
  - Decrease scale variance $\dot{s}$ to avoid bounding box jitter when students remain seated.
- **Measurement Noise Covariance ($R$):**
  - Increase measurement noise if YOLO bounding box boundaries exhibit frame-to-frame flutter.

### 4.3 Seat-Mapping Spatial Assignment Thresholds

| Parameter | Variable | Default | Optimal | Impact on Spatial Accuracy |
|---|---|---|---|---|
| Seat IoU Threshold | `SEAT_IOU_THRESHOLD` | `0.30` | `0.25` – `0.35` | Minimum Intersection-over-Union between person bounding box and predefined desk polygon to assign seat ID. |
| Distance Normalizer | `HAND_PROXIMITY_NORM` | `0.20` | `0.18` – `0.25` | Wrist-to-object proximity threshold as a fraction of person bounding box height. |

---

## 5. Facial Landmark, Bio-Signal & DACE Tuning Parameters

### 5.1 Head Pose & Pitch Calibration

In the Downward-Attention Context Engine (DACE) and v2.0 `features/dace_measurement.py`:
- **Convention:** Head pitch is measured in degrees, where **positive = head down** (+down).
- **Transformation Matrix:** Extracted from MediaPipe Face Landmarker 4x4 matrix:
  $$R = R_y(\text{yaw}) \cdot R_x(\text{pitch}) \cdot R_z(\text{roll}), \quad \text{pitch} = \arcsin(-r_{12})$$
- **Relative Pitch Normalization:** Measured relative to a **5.0-second screen-facing median reference baseline** ($\tau_{\text{ref}} = 5.0\text{s}$).

| Parameter | Symbol | Default | Range | Calibration Method |
|---|---|---|---|---|
| Looking Down Pitch Threshold | $\theta_{\text{down}}$ | `15.0°` | `12.0°` – `18.0°` | Head tilt beyond which a student is classified as looking down towards the desk/book/phone. |
| Looking Down Blendshape Gaze | $\gamma_{\text{down}}$ | `0.50` | `0.40` – `0.60` | Vertical eye gaze blendshape (`eyeLookDown`) threshold indicating downward gaze without head tilt. |
| Pitch Camera Reset Threshold | $\Delta \theta_{\text{jump}}$ | `25.0°` | `20.0°` – `30.0°` | Sudden pitch jump between 1-second medians indicating webcam readjustment. Triggers baseline reset. |
| Scale Camera Reset Threshold | $\Delta \text{scale}$ | `0.30` | `0.20` – `0.40` | Relative jump in face bounding box area indicating student movement or webcam repositioning. |

### 5.2 Eye Openness & Huber Regression Parameters

Physical EAR varies significantly between individuals and changes naturally when looking downward. DACE replaces fixed EAR thresholds with a personalized, gaze-corrected Huber regression model:

$$\hat{E}(g) = b_0 + b_1 g + b_2 g^2, \quad \text{Residual } r = \frac{\text{EAR}}{\hat{E}(g)}$$

```
               ┌────────────────────────────────────────────────────────┐
               │ Huber Loss:                                            │
               │   L_δ(a) = 0.5 * a²                 for |a| ≤ δ        │
               │   L_δ(a) = δ * (|a| - 0.5 * δ)      for |a| > δ        │
               └────────────────────────────────────────────────────────┘
```

| Parameter | Symbol | Default | Range | Description |
|---|---|---|---|---|
| Huber Loss Delta | $\delta$ | `1.345` | `1.0` – `1.5` | Threshold transitioning quadratic loss to linear loss (robust against blink outliers). |
| Quadratic Prior Intercept | $b_0$ | `0.30` | `0.26` – `0.34` | Baseline nominal eye openness when looking straight ahead. |
| Quadratic Prior Linear Coeff | $b_1$ | `-0.12` | `-0.18` – `-0.08` | Openness reduction per unit of downward gaze. |
| Quadratic Prior Quad Coeff | $b_2$ | `0.00` | `-0.05` – `0.05` | Second-order curvature term. |
| Minimum Calibration Samples | $N_{\text{min}}$ | `50` | `30` – `100` | Number of screen-facing attentive frames required before personalizing prior coefficients. |
| Gaze Range Requirement | $\Delta g_{\text{min}}$ | `0.30` | `0.20` – `0.40` | Required vertical gaze variance before fitting higher-order terms $b_1, b_2$. |
| Eye Closed Threshold | $k_{\text{open}}$ | `0.70` | `0.65` – `0.75` | Residual ratio $r < k_{\text{open}}$ below which the eye is counted as closed. |

### 5.3 Gaze-Corrected PERCLOS & Fatigue Metrics

$$\text{PERCLOS} = \frac{\sum_{t \in W_{60}} \mathbb{I}(r_t < k_{\text{open}})}{|W_{60}|}$$

| Parameter | Default | Range | Condition for Activation |
|---|---|---|---|
| Drowsiness PERCLOS Threshold ($p_{\text{drowsy}}$) | `0.15` (15%) | `0.12` – `0.20` | PERCLOS over 60 seconds $\ge 0.15$ triggers `drowsy` classification. |
| Continuous Eye Closure ($\tau_{\text{closure}}$) | `2000 ms` | `1500` – `3000 ms` | Single continuous eye closure $\ge 2.0$ seconds overrides PERCLOS and immediately triggers `drowsy`. |
| Blink EAR Threshold | `0.21` | `0.18` – `0.23` | Instantaneous EAR threshold for involuntary blink detection. |
| Minimum Blink Duration | `2 frames` | `2` – `3` | Filters single-frame landmark tracking jitter. |
| Maximum Blink Duration | `400 ms` | `300` – `500 ms` | Distinguishes regular blinks from microsleep / drowsiness. |

### 5.4 Hand-Object Interaction & Pinch Mechanics

| Parameter | Symbol | Default | Range | Behavioral Meaning |
|---|---|---|---|---|
| Pen Pinch Ratio | $\pi_{\text{max}}$ | `0.35` | `0.30` – `0.40` | $\frac{\|\text{Thumb Tip} - \text{Index Tip}\|}{\text{Palm Length}} < \pi_{\text{max}}$ identifies a writing pen grip. |
| Writing Wrist Motion Min | $m_{\text{min}}$ | `0.02` | `0.01` – `0.03` | Lower bound of wrist speed (hand lengths/frame) indicating active writing. |
| Writing Wrist Motion Max | $m_{\text{max}}$ | `0.15` | `0.10` – `0.20` | Upper bound of wrist speed (distinguishes writing from wild gesticulation/fidgeting). |
| Holding Box IoU | $\text{IoU}_{\text{hold}}$ | `0.10` | `0.05` – `0.20` | IoU between hand bounding box and object box indicating holding. |
| Near Object Diagonal Factor | $d_{\text{near}}$ | `0.50` | `0.40` – `0.60` | Centre distance $< d_{\text{near}} \times \text{diagonal}(\text{box})$ treats object as near hand. |
| Phone Hold Persistence | $\tau_{\text{phone}}$ | `10.0 s` | `5.0` – `15.0 s` | Retains `using_phone` label across brief glances to detect repeated hidden phone checks. |

---

## 6. Open-Weights Multimodal LLM / VLM Fine-Tuning (Gemma / Llama / Qwen)

### 6.1 Role in CognEn (Session Reports & Analytical Explanations)

Open-weights Vision-Language Models (VLMs) and Large Language Models (LLMs) (e.g., **Gemma-2-9B-It**, **Llama-3.2-11B-Vision**, or **Qwen2.5-VL-7B**) generate executive session reports, explain complex multi-modal contradictions (CMCV), and synthesize classroom engagement patterns into actionable pedagogy feedback.

### 6.2 LoRA / QLoRA Fine-Tuning Hyperparameters

When fine-tuning open-weights LLMs/VLMs with Hugging Face `peft` and `transformers`:

| Hyperparameter | Parameter Name | Value | Rationale |
|---|---|---|---|
| Quantization | `load_in_4bit` | `True` (NF4) | 4-bit NormalFloat quantization fits 9B-11B models in 16GB–24GB VRAM. |
| Compute Dtype | `bnb_4bit_compute_dtype` | `torch.bfloat16` | Prevents underflow in attention heads during LoRA updates. |
| Double Quantization | `bnb_4bit_use_double_quant` | `True` | Compresses quantization constants, saving ~0.4 bits/parameter. |
| LoRA Rank | `r` | `32` (or `64`) | Higher rank captures nuanced educational terminology and statistical reasoning. |
| LoRA Alpha | `lora_alpha` | `64` (2× $r$) | Scaling factor for LoRA adapter updates ($\Delta W = \frac{\alpha}{r} B A$). |
| LoRA Dropout | `lora_dropout` | `0.05` | Prevents adapter overfitting on repetitive session templates. |
| Target Modules | `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Fine-tuning all linear projection layers achieves superior instruction-following accuracy. |
| Learning Rate | `learning_rate` | `1.5e-4` | Optimal learning rate for QLoRA adapters with `AdamW_8bit`. |
| LR Scheduler | `lr_scheduler_type` | `cosine` | Cosine decay with 5% warmup steps. |
| Warmup Ratio | `warmup_ratio` | `0.05` | Prevents early gradient spikes from corrupting pre-trained attention. |
| Batch Size / Accumulation | `per_device_train_batch_size` / `gradient_accumulation_steps` | `4` / `8` (Effective: `32`) | Large effective batch size stabilizes multi-student summary generation. |
| Max Sequence Length | `max_seq_length` | `4096` | Accommodates 60-minute session event timelines and JSON metric dumps. |
| Weight Decay | `weight_decay` | `0.01` | Regularization on adapter weights. |

### 6.3 Generation & Structured Output Decoding Parameters

Deterministic, hallucination-free generation is essential for session reports:

```python
generation_config = {
    "temperature": 0.15,          # Low temperature enforces adherence to session metrics
    "top_p": 0.90,                # Nucleus sampling bound
    "top_k": 40,                  # Top-K restriction
    "repetition_penalty": 1.12,   # Prevents loops in metric summaries
    "max_new_tokens": 1500,       # Upper bound for report markdown
    "do_sample": True,
}
```

---

## 7. Systematic Tuning Strategy for Maximum Accuracy

### 7.1 Optuna Bayesian Hyperparameter Search

Use Bayesian optimization via `Optuna` with the Tree-structured Parzen Estimator (`TPESampler`) to tune the 1D-CNN and loss matrix simultaneously.

```python
# tune_hyperparameters.py
import optuna
from v2.0.training.train_classifier import train

def objective(trial: optuna.Trial) -> float:
    args = argparse.Namespace(
        lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        batch=trial.suggest_categorical("batch", [32, 64, 128]),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        epochs=50,
        patience=12,
        val_ratio=0.15,
        penalty_loss=True,
        # Optimize loss penalty multipliers
        phone_miss_penalty=trial.suggest_float("phone_miss", 1.8, 3.5),
        reading_accuse_penalty=trial.suggest_float("reading_accuse", 2.0, 4.0),
    )
    results = train(args)
    # Target objective: maximize validation macro-F1
    return results["best_val_macro_f1"]

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, timeout=14400)
print("Best parameters:", study.best_params)
```

### 7.2 Stratified Group K-Fold Cross-Validation

To verify performance across disparate room layouts and lighting conditions without data leakage:
1. Group by `classroom_id` or `student_subject_id`.
2. Apply `StratifiedGroupKFold(n_splits=5)`.
3. Ensure every fold contains balanced representations of all 6 behavior classes while maintaining zero identity overlap between train and validation folds.

### 7.3 Probability Calibration (Temperature Scaling)

Raw softmax outputs $\sigma(z_i)$ often yield overconfident probabilities. Use post-hoc **Temperature Scaling** on validation logits:

$$\hat{p}_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

- Optimize scalar $T > 0$ by minimizing negative log-likelihood (NLL) on the held-out validation set.
- Typical optimal temperature: $T \in [1.15, 1.45]$.
- Reduces Expected Calibration Error (ECE) below 2.5%, allowing confident thresholding on the `Unsure` state (U5 conformal abstention).

### 7.4 Hard-Negative Mining Pipeline

1. Run the trained model over unlabeled classroom validation sessions.
2. Extract sequences where:
   - Prediction probability is ambiguous ($0.35 < \max_c p_c < 0.60$).
   - Cross-Modal Contradiction Vector (CMCV) triggers `prop_without_engagement` or `context_conflict`.
   - The detector predicted `using_phone` when hand proximity was absent.
3. Add mined sequences into the training set with 2× sample weighting in the next training epoch.

---

## 8. Export, Quantization & Edge Inference Calibration

### 8.1 ONNX Export Parameters

To deploy the PyTorch 1D-CNN temporal classifier to ONNX Runtime (browser WebGPU / server ONNX):

```python
# export_onnx.py
torch.onnx.export(
    model,
    dummy_input,            # Shape: (1, 15, 15)
    "behavior_classifier_v3.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input":  {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=17,
    do_constant_folding=True,
)
```

Verify output parity:
$$\max |z_{\text{PyTorch}} - z_{\text{ONNX}}| < 1\times 10^{-5}$$

### 8.2 INT8 Post-Training Quantization (PTQ) Calibration

For deployment on edge devices (NVIDIA Jetson, Raspberry Pi 5, or local servers):
- Use **Static Quantization** with ONNX Runtime Quantization tool.
- Calibration dataset: Run a representative calibration set of 500 real classroom feature windows through `CalibrationDataReader`.
- Method: `QuantType.QInt8` with `CalibrationMethod.Entropy` (minimizes KL-divergence between FP32 and INT8 activations).
- Result: 4× reduction in memory footprint (~22 KB) and 2.5× throughput speedup with $< 0.3\%$ macro-F1 degradation.

---

## 9. Master Parameter Reference Matrix

| Subsystem | Parameter Name | Canonical File Reference | Default Value | Recommended Tuned Value | Primary Impact |
|---|---|---|---|---|---|
| **YOLOv8** | `lr0` | Training YAML / script | `0.01` | `0.0008` (AdamW) | Backbone feature preservation |
| **YOLOv8** | `imgsz` | Training YAML / CLI | `640` | `1280` (CCTV) / `640` (Desk) | Small object & pen resolution |
| **YOLOv8** | `box` | Training YAML | `7.5` | `8.5` | Bounding box spatial tightness |
| **YOLOv8** | `cls` | Training YAML | `0.5` | `1.2` | Object classification precision |
| **YOLOv8** | `close_mosaic` | Training YAML | `10` | `15` | Fine boundary convergence |
| **1D-CNN** | `lr` | `v2.0/training/train_classifier.py` | `1e-3` | `1e-3` (Cosine to `1e-5`) | Sequence model convergence |
| **1D-CNN** | `batch_size` | `v2.0/training/train_classifier.py` | `64` | `64` | Gradient stability |
| **1D-CNN** | `label_smoothing` | `v2.0/training/train_classifier.py` | `0.05` | `0.05` | Softens overconfident outputs |
| **1D-CNN** | Reading-to-Phone Penalty | `v2.0/training/penalty_matrix.py` | `1.0x` (Legacy) | `3.0x` | Eliminates false phone accusations |
| **1D-CNN** | Phone-Miss Penalty | `v2.0/training/penalty_matrix.py` | `1.0x` (Legacy) | `2.5x` | Prevents missed phone infractions |
| **ByteTrack** | `track_thresh` | `v2.0/tracking/byte_tracker.py` | `0.25` | `0.45` | Eliminates ghost person tracks |
| **ByteTrack** | `low_thresh` | `v2.0/tracking/byte_tracker.py` | `0.10` | `0.10` | Tracks students through occlusion |
| **ByteTrack** | `track_buffer` | `v2.0/tracking/byte_tracker.py` | `30` | `45` (at 15 FPS) | Occlusion recovery window |
| **DACE** | $\theta_{\text{down}}$ | `v1.0/src/lib/scoring/dace/params.ts` | `15.0°` | `15.0°` | Downward gaze classification |
| **DACE** | $\tau_{\text{ref}}$ | `v1.0/src/lib/scoring/dace/params.ts` | `5000 ms` | `5000 ms` | Screen-facing baseline reference |
| **DACE** | Huber $\delta$ | `v1.0/src/lib/scoring/dace/params.ts` | `1.345` | `1.345` | Robust eye openness fitting |
| **DACE** | $k_{\text{open}}$ | `v1.0/src/lib/scoring/dace/params.ts` | `0.70` | `0.70` | Gaze-corrected closure boundary |
| **DACE** | $p_{\text{drowsy}}$ | `v1.0/src/lib/scoring/dace/params.ts` | `0.15` | `0.15` | 60-second PERCLOS drowsiness flag |
| **DACE** | $\pi_{\text{max}}$ | `v1.0/src/lib/scoring/dace/params.ts` | `0.35` | `0.35` | Pen-pinch grip ratio |
| **DACE** | $\tau_{\text{phone}}$ | `v1.0/src/lib/scoring/dace/params.ts` | `10.0 s` | `10.0 s` | Phone glance accumulation window |
| **LLM/VLM** | `lora_r` | PEFT Config | `16` | `32` – `64` | Adapter expressive capacity |
| **LLM/VLM** | `temperature` | Generation Config | `0.70` | `0.15` | Factual, deterministic reports |
| **LLM/VLM** | `learning_rate` | Training Config | `2e-4` | `1.5e-4` | QLoRA adapter convergence |
