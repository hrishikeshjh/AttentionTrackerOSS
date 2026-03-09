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

[ ] Phase 5 follow-up: Wire ObjectDetector + HandTracker + TemporalBehaviorClassifier into webcam-panel.tsx
    - Initialise modules on session start
    - Attach context fields to ExtendedFaceMetrics before calling mlEngine.predict()
[ ] Phase 6 follow-up: Wire HandObjectProximityDetector + TemporalBehaviorClassifier into offscreen.js
[ ] v1.0 models: Download / fine-tune yolov8n-objects.onnx; place in public/models/
[ ] v1.0 WASM: Copy onnxruntime-web WASM binaries to public/onnx/
[ ] v2.0 face landmarks: Integrate face_alignment or dlib into RoomAnalyzer._get_landmarks()
[ ] v2.0 models: Download yolov8m.pt, yolov8s-objects.pt, yolov8m-pose.pt into v2.0/models/
[ ] Training: Collect classroom dataset (ARCHITECTURE §6); train behavior_classifier.onnx
[ ] Evaluation: Run metrics from ARCHITECTURE §7 on held-out test set

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
