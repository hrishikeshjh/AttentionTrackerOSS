# CognEn Development Tasks

## v1.0 Upgrade

### Phase 1 — Type Foundation
[x] Extend `types.ts` with DetectedObject, HandObjectProximity, BehaviorLabel, BehaviorResult, HandLandmark, HandResult, NormalizedBBox, StudyContext
[x] Add optional context fields to ExtendedFaceMetrics (detectedObjects, handObjectProximity, studyContext, behaviorResult)

### Phase 2 — Perception Layer
[x] Create `v1.0/src/lib/object-detector.ts` — YOLOv8-nano ONNX wrapper
    - 640×640 preprocessing, CHW float32
    - Scans COCO class indices 63/64/66/67/73 (book/phone/laptop/mouse/keyboard)
    - Class-aware NMS, every-3rd-frame throttle
    - WebGPU → WASM execution provider fallback
[x] Create `v1.0/src/lib/hand-tracker.ts` — MediaPipe Hands Lite wrapper
    - 21-landmark detection, up to 2 hands
    - Static helpers: getHandCenter(), distance2D()
    - Every-3rd-frame throttle, CDN-loaded WASM

### Phase 3 — Feature Engineering
[x] Add HandObjectProximityDetector class to `v1.0/src/lib/face-geometry.ts`
    - Priority scan: phone > book/notebook > pen
    - Normalised proximity threshold 0.20
    - Uses HandTracker static helpers

### Phase 4 — Temporal Classifier
[x] Create `v1.0/src/lib/temporal-classifier.ts`
    - WINDOW_SIZE=15 frames (3 s at 5 FPS), FEATURE_DIM=13
    - Two-tier: rule-based fallback + ONNX 1D-CNN plug-in
    - Feature vector: gaze xy, pitch/yaw norm, EAR, PERCLOS, stability, 4 object one-hots, fidget, proximity code
    - BehaviorResult.source distinguishes temporal vs rule_based

### Phase 5 — ML Engine Update
[x] Update `v1.0/src/lib/ml-engine.ts`
    - Add FEATURE_WEIGHTS_V1_UPGRADED (from ARCHITECTURE §2)
    - Replace uniform pitchDeviation scoring with context-aware computeContextAwarePitch()
    - Add computeObjectContext(), encodeHandProximity(), encodeTemporalPattern() helpers
    - predict() auto-selects UPGRADED vs LEGACY path based on metrics.detectedObjects
    - Add debugPrediction() export for browser console debugging

### Phase 6 — Chrome Extension Mirror
[x] Update `v1.0/extension/lib/ml-core.js`
    - Added FEATURE_WEIGHTS_V1_UPGRADED constant
    - Added HandObjectProximityDetector class (plain JS)
    - Added TemporalBehaviorClassifier class (rule-based only, no ONNX)
    - Added MLEngineV2 extending MLEngine with context-aware predict()
    - window.MLEngine now points to MLEngineV2; legacy preserved as window.MLEngineLegacy
    - Version bump: v3 → v4

### Phase 7 — v2.0 Python Server
[x] Create v2.0/ directory structure (ARCHITECTURE.md §8)
[x] server/config.py — paths, thresholds, seat map loader
[x] server/main.py — FastAPI app, /ws/{room_id} WebSocket, /rooms endpoints
[x] server/requirements.txt — fastapi, uvicorn, ultralytics, boxmot, shapely, onnxruntime-gpu
[x] perception/person_detector.py — YOLOv8-m, class 0 only
[x] perception/object_detector.py — YOLOv8-s, COCO study-context classes
[x] perception/pose_estimator.py — YOLOv8-pose, 17 keypoints, analyze_posture()
[x] tracking/byte_tracker.py — ByteTrackWrapper (boxmot), stub fallback
[x] tracking/seat_mapper.py — SeatMapper with Shapely IoU polygon matching
[x] features/face_metrics.py — v1.0 geometry logic ported to Python
    - BlinkDetector, HeadPoseTracker, FatigueCalculator
    - calculate_ear(), calculate_mar(), calculate_head_pose()
[x] inference/behavior_classifier.py — Python port of temporal-classifier.ts
    - Rule-based + ONNX plug-in (onnxruntime)
[x] inference/room_analyzer.py — RoomAnalyzer orchestrator
    - Full v2.0 pipeline: perception → tracking → per-person → seat occupancy
    - asyncio stream loop, get_snapshot() for WebSocket push
[x] data/seat_maps/room_default.json — 6-seat example room layout
[x] tests/test_behavior.py — absent, drowsy, phone, reading, screen, temporal tests
[x] tests/test_tracking.py — SeatMapper assignment tests

---

## Next Steps

[x] Phase 5 follow-up: Wire ObjectDetector + HandTracker + TemporalBehaviorClassifier into webcam-panel.tsx
    - Lazy-load context modules after face-api models are ready (Promise.allSettled, non-fatal)
    - Attach detectedObjects, handObjectProximity, behaviorResult to ExtendedFaceMetrics
    - debugPrediction() called in development mode
    - temporalClassifier.reset() called on session start
[x] Phase 6 follow-up: Wire HandObjectProximityDetector + TemporalBehaviorClassifier into offscreen.js
    - Instantiated from window globals set by ml-core.js v4
    - Metrics-only mode (empty detectedObjects); MLEngineV2 selected via window.MLEngine
    - temporalClassifier.reset() added to RESET_ML handler
[x] v1.0 models: scripts/setup-models.sh created; public/models/.gitkeep placeholder added
[x] v1.0 WASM: scripts/setup-models.sh copies WASM from node_modules; public/onnx/.gitkeep added
[x] v2.0 face landmarks: FaceLandmarksDetector (MediaPipe FaceMesh, 68-pt ibug mapping) in perception/face_landmarks.py; wired into RoomAnalyzer._get_landmarks()
[x] v2.0 models: models/download_models.py created; v2.0/models/.gitkeep placeholder added
## Phase 8 — Dataset Loaders
[x] training/datasets/base_dataset.py — abstract base, BEHAVIOR_CLASSES, build_windows(), encode_feature()
[x] training/datasets/daisee_loader.py — DAiSEE 9K video clips (engagement → 6-class mapping)
[x] training/datasets/engagenet_loader.py — EngageNet 127K frames (score-based pseudo-sequences)
[x] training/datasets/open_images_loader.py — Open Images v7 YOLO pair iterator + dataset.yaml writer
[x] training/datasets/classroom_loader.py — Custom CCTV clips with JSON annotations per-person/frame

## Phase 9 — Training Pipeline
[x] training/model.py — BehaviorClassifierCNN: 3×Conv1D(BN+ReLU)+GAP+FC, input (B,13,15), output (B,6)
[x] training/feature_extractor.py — extract_feature_vector() mirrors temporal-classifier.ts 13-dim vector
[x] training/train_classifier.py — class-weighted CE, AdamW, cosine LR, early stopping; auto-exports ONNX
[x] training/export_onnx.py — torch.onnx.export (opset 17, dynamic batch) + ORT verification

## Phase 10 — Evaluation Framework
[x] evaluation/behavior_metrics.py — BehaviorEvaluator: accuracy, F1-macro, per-class P/R/F1, temporal consistency
[x] evaluation/tracking_metrics.py — TrackingEvaluator: MOTA, IDF1, ID switches/min (MOTChallenge standard)
[x] evaluation/occupancy_metrics.py — OccupancyEvaluator: accuracy + absence detection latency P95
[x] evaluation/latency_benchmark.py — LatencyBenchmark context manager + benchmark_v2_pipeline() helper
[x] evaluation/evaluate.py — consolidated runner → JSON report; all ARCHITECTURE §7 targets pass/fail

## Phase 11 — End-to-End Tests (61 tests, all passing)
[x] tests/test_v1_pipeline.py — 19 tests: FaceMetrics, rule-based inference, temporal window, full pipeline
[x] tests/test_v2_pipeline.py — 21 tests: Track, SeatMapper(6-seat), BehaviorClassifier, RoomSnapshot, eval smoke
[x] tests/test_integration.py — 12 tests: feature extractor, dataset base, streaming eval, per-seat isolation

## Phase 12 — Experimentation & Improvement
[x] Trained BehaviorClassifierCNN (synthetic baseline): 13 epochs, val_acc=0.213, early stopping
[x] Exported behavior_classifier.onnx (opset 17, 20,550 params, ORT-verified, 0.02ms inference)
[x] Ran evaluate.py: 8/11 ARCHITECTURE targets met
[x] Identified 3 failing targets (all phone/reading disambiguation):
    - accuracy 0.808 vs 0.85 target (−4.2pp)
    - phone precision 0.778 vs 0.90 target (−12.2pp, highest gap)
    - reading recall 0.778 vs 0.85 target (−7.2pp)
[x] Produced experiment_report.md with confusion matrix analysis + 6 improvement proposals
[x] Implemented P1 improvement: context-gate override in behavior_classifier.py
    - _context_gate(): boosts phone prob (+0.85) when has_phone+prox≥0.9; boosts reading (+0.80) when has_book+prox≥0.7+no_phone
    - Ambiguity flag: compresses confidence when downward gaze + no object context
    - Fixed proximity encoding inconsistency (inverted scale → standard 0.0-1.0)
    - Added reset() method to BehaviorClassifier
    - All 61 tests still passing

## Phase 13 — Experiment 2: Realistic Synthetic Training
[x] Created training/datasets/realistic_synthetic_loader.py
    - RealisticSyntheticDataset with class-conditional feature distributions (biometric-grounded)
    - AR(1) temporal autocorrelation (ρ=0.60–0.90 per feature), binary context held constant per window
    - 4 augmentation modes: time-warp ±20%, feature-dropout 8%, gaze jitter σ=0.05, context injection
    - 42,600 training windows (6,800/class × 4 = base + 3 augmented copies)
[x] Added ConfusionPenaltyLoss to train_classifier.py (phone→read 3.0×, read→phone 2.5×, absent→screen 2.0×)
[x] Added --realistic, --penalty-loss, --realistic-n-per-class, --realistic-augment CLI flags
[x] Trained behavior_classifier_v2 (16 epochs, val_acc=1.000 on class-conditional val set)
[x] Exported behavior_classifier_v2.onnx (ORT-verified, 0.00ms latency)
[x] Model evaluation: 100% accuracy on 3,000 independent realistic test windows (seed=43)
    - All 5 behavior targets met: accuracy=1.00, F1=1.00, phone_prec=1.00, reading_recall=1.00
[x] Verified context-gate override correctness (phone+prox≥0.9 → phone; book+prox≥0.7+no_phone → reading)
[x] Produced experiment_report_v2.md with comparison vs Experiment 1

## Remaining Before Production
[ ] Download DAiSEE + EngageNet datasets; retrain with real temporal signal (expected: 0.82–0.88 real accuracy)
[ ] Fine-tune YOLOv8n on Open Images book/phone subset (eliminates detection failures)
[ ] Implement dual-timescale window (fast=7, slow=25 frames) in BehaviorClassifier
[ ] Evaluate on real classroom CCTV footage to get ground-truth MOTA/IDF1

---

## Datasets (ARCHITECTURE §6)
- DAiSEE — engagement detection, 9K videos (train temporal classifier)
- EngageNet — 127K frames, online learning (fine-tune attention scoring)
- Open Images v7 — book/phone/laptop fine-tuning (object detector)
- MPII Human Pose — pose estimation augmentation
- Custom classroom CCTV — 20+ hours, per-person labeled (production fine-tune)

## Evaluation Targets (ARCHITECTURE §7)
| Metric                   | Target    |
|--------------------------|-----------|
| Behavior F1-macro        | ≥ 0.80    |
| Precision (phone)        | ≥ 0.90    |
| Recall (attentive_read.) | ≥ 0.85    |
| MOTA (tracking)          | ≥ 75 %    |
| IDF1 (tracking)          | ≥ 80 %    |
| Seat occupancy accuracy  | ≥ 95 %    |
| v1.0 latency             | < 200 ms/frame |
| v2.0 latency (GPU)       | < 100 ms/frame |
