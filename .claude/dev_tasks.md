# CognEn Development Tasks

> Status note: `[x]` below means "code written and unit-tested", not "validated". Nothing here has been evaluated on real data, and nothing (DACE, the legacy path, the extension, v2.0) has been run in a browser, against a camera, as a loaded Chrome extension, on real CCTV footage or on a GPU. Items marked NOT DONE are real-data or runtime items that remain open. Test counts in older phases (61) are historical: the current v1.0 Vitest count is 326 (33 files), v2.0 light tier 268 pass / 43 skip, v2.0 heavy tier 338 pass. Phase 6 (extension mirror) described the original, divergent copy; the extension was rewritten in Phase 22. Phase 12-13 results (val_acc, "8/11 targets", 1.000) are synthetic/circular and their reports are RETRACTED (archived in `v2.0/evaluation/reports/archive/`); see Phases 24-25.

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

### Phase 6 — Chrome Extension Mirror (original copy; superseded by the Phase 22 rewrite)
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

## Phase 11 — End-to-End Tests (61 tests at the time; superseded, see status note)
[x] tests/test_v1_pipeline.py — 19 tests: FaceMetrics, rule-based inference, temporal window, full pipeline
[x] tests/test_v2_pipeline.py — 21 tests: Track, SeatMapper(6-seat), BehaviorClassifier, RoomSnapshot, eval smoke
[x] tests/test_integration.py — 12 tests: feature extractor, dataset base, streaming eval, per-seat isolation

## Phase 12 — Experimentation & Improvement
[x] Trained BehaviorClassifierCNN (synthetic baseline): 13 epochs, val_acc=0.213, early stopping
[x] Exported behavior_classifier.onnx (opset 17, 20,550 params, ORT-verified, 0.02ms inference)
[x] Ran evaluate.py: 8/11 ARCHITECTURE targets met (RETRACTED: the runner then fabricated its behavior/tracking/occupancy sections; see Phase 24)
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

## Phase 13 — Experiment 2: Realistic Synthetic Training (results circular; report retracted)
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

## DACE (Downward-Attention Context Engine) — v1.0 web app only; Stage 1 implemented, Stage 2 hooks only

Plan: `DACE-IMPLEMENTATION-PLAN.md`; spec: `ARCHITECTURE.md` §11. The extension is out of scope; v2.0 initially kept the legacy pitch formula and later ported the DACE measurement layer only (Phase 24).

### Phase 14 — W1 (M0): foundations
[x] vitest alias/node env, `tsx`, `dace:replay` script, DACE config (zod) + presets A, A', A'', B, C, F, H, I, `DaceParams` (Table A.6)
[x] Golden-fixture legacy-equivalence test (`src/lib/__tests__/legacy-equivalence.test.ts`) guarding the DACE-off path
[x] Pitch characterisation test + `docs/proposals/evidence/pitch_sim.py` (reads constants from generated JSON)
[x] `scripts/setup-models.sh` downloads + verifies pinned MediaPipe models (`scripts/models.sha256`), `.gitignore` for WASM/tflite
[x] Detector self-test, eval log (FR9), latency probes, replay skeleton

### Phase 15 — W2-3 (M1): measurement
[x] Provider abstraction (legacy-faceapi / pnp68 / mediapipe), matrix pitch (+down), pitch reference, blendshape gaze, eye openness, corrected PERCLOS, looking-down test, solvePnP, audited mesh adapter (+topology test)
[ ] Pilot recordings P1, ArUco validation (sign 100%, error <= 5 deg), fit earScale/prior -- NOT DONE (no recordings)

### Phase 16 — W4-5 (M2): evidence
[x] Hand landmarker, hand-object relations, evidence (phone-in-use, book, writing, desk visibility), MediaPipe object detector (Apache-2.0 comparison), staggered scheduling, feature mapping for E, offline replay

### Phase 17 — W6 (M3): rules, abstention, integration
[x] Window, ordered rules, abstention U1-U4 (U5 Stage 2), score-hold, opt-in MLEngine DACE path, FixedTickScheduler + no-face frames, dashboard/report/analytics integration ("Undetermined" card, chart spans, pie slice, PDF line)
[ ] Supervisor sign-off on D11 rule order/U4 definition and the feature mapping -- PENDING

### Phase 18 — W7: evaluation tooling
[x] `/replay` page, `scripts/dace/{replay-features,metrics,cvat-windows}.ts` (FDR-DA, phone recall, Unsure rate, answer rate, latency)
[ ] Run all Stage 1 presets over P1; latency profiling (target p95 <= 200 ms); Unsure >= 90% on desk-out-of-frame; FDR-DA and phone recall -- NOT DONE (targets, unmeasured)

### Phase 19 — W8: docs and Stage 2 hooks
[x] Docs (ARCHITECTURE §11, CLAUDE.md, READMEs, novelty list of ten), `loadModel()` contract fix + mocked-ort test, conformal/kinematics stubs (no runtime caller)
[x] v2.0 side: `COGNEN_HAND_PROXIMITY=pose` (opt-in at the time; the default since Phase 24), cross-language contract tests + `shared/fixtures`, pitch characterisation test, unused config constants labelled
[x] CI: permissions block, DACE evidence/fixture guards (the informational audits of this phase were later made blocking; CI was re-run green, see Phase 21)
[ ] Browser/camera verification of MediaPipe WASM loading under Next.js, delegates, latency -- NOT DONE
[ ] Pending decisions: R1 (face-api to MediaPipe), R2 (YOLOv8 AGPL), R4 (ethics for pilot), absence handling, PEB hooks

## Hardening, correctness and CI rounds (Phases 20-25)

All `[x]` items are code written, unit-tested and (for CI) green on a manual `dev` run. Items marked NOT DONE need real data, a browser, Chrome, a camera or a GPU.

### Phase 20 — Dependency security round
[x] v1.0: `npm audit` 107 (6 critical) to 0 vulnerabilities: next 15.5.27, genkit 1.42, firebase 11.10, Tailwind 3 to 4 migration (removes `braces`), `extract-zip` replaced by `scripts/stubs/extract-zip`, OpenTelemetry forced to 2.x via `overrides` + `scripts/patch-genkit-otel2.mjs` postinstall, longer `overrides` list, explicit `@emnapi` devDependencies for Linux lockfile completeness
[x] v2.0: unresolvable numpy 1.26 pin set replaced by a numpy 2.x stack that resolves: torch 2.13.0, torchvision 0.28.0, numpy 2.4.6, boxmot 16.0.11, ultralytics 8.4.146, opencv-python 4.13.0.92, onnxruntime 1.24.4, shapely 2.1.2, mediapipe 0.10.31, onnx 1.22.0, Pillow 12.3.0, scikit-learn 1.7.2, pytest 9.0.3, fastapi 0.141.1, starlette 1.6.0, pydantic 2.13.5; Python 3.10-3.12 only
[ ] Smoke-test the heavy stack (YOLO, FaceLandmarker, boxmot ByteTrack, ONNX) on real video; CPU-only servers need the torch CPU index (default Linux wheel is CUDA 13) -- NOT DONE
[ ] Real Gemini call after the OpenTelemetry change -- NOT DONE

### Phase 21 — CI rebuild
[x] Node 24 actions pinned by SHA, `ubuntu-24.04`, push/PR on `main` and `dev`, `workflow_dispatch` runs every job, least-privilege permissions, timeouts, concurrency
[x] `changes`/`secret-scan` permissions fix; `v1-audit` blocks on any severity (`--audit-level=low`); `v2-audit` resolves and blocks (`V2_AUDIT_RESOLVE`); `all-checks-passed` includes `changes`; `v2-test` installs with production pins as constraints; `mypy` informational
[x] Fixture drift guards (`encode13`, CMCV, `encode15_v2`, `pitch_sim`), gitleaks allowlists, `.github/dependabot.yml`, `.github/ci/requirements-*.txt`
[x] Manual run on `dev`: all 12 jobs green
[ ] Required-status branch protection on `main` referencing `all-checks-passed` -- NOT VERIFIED

### Phase 22 — v1.0 legacy path and extension fixes
[x] Legacy loop on a 200 ms `FixedTickScheduler` (was rAF-bound); duplicate-loop guard; legacy weights reachable when the object model is missing; absence scored as `Idle` after a 3 s grace; `ABSENT_SCORE`
[x] `MLEngineOptions` / `APP_ENGINE_OPTIONS` (default `new MLEngine()` stays the frozen as-shipped baseline for the golden fixture and replay presets A/A'); contributions use the weight table actually used
[x] CMCV `unclassified` category and `prop_without_engagement` fix (taxonomy `v2`); letterboxed object detector; `studyContext` derived; analytics `getAnomalies()`; Gemini text wrapped in the PDF
[x] Authenticated `generateReportAction` (Firebase ID token via `node:crypto` against Google JWKS, per-user rate limit, zod schema); `.env.example`; dead `analyze-facial-features` flow removed
[x] Extension rewritten to web parity: IIFE `ml-core.js`, thresholds 70/40, EMA 0.15, `setTimeout` loop, HMAC-signed `postMessage` -> `bridge.js` -> `chrome.runtime`, `storage.session` persistence, alarms/notifications, restricted `web_accessible_resources`, self-view heuristic; `extension-equivalence.test.ts`
[ ] Load the extension in Chrome and run it on Meet/Zoom/Teams and standalone -- NOT DONE
[ ] Run the web app in a browser with a camera (legacy and DACE) -- NOT DONE
[ ] Align `summary-cards.tsx` colour bands (75/50) with 70/40 -- OPEN

### Phase 23 — v2.0 server security
[x] `X-API-Key` auth (`COGNEN_API_KEY`, rotation), secure defaults (loopback bind, `COGNEN_ALLOW_INSECURE`, docs off, CORS allowlist), body cap, `/start` and failed-auth rate limits, WebSocket hardening (`COGNEN_WS_MAX_PER_KEY` etc.), SSRF policy (`COGNEN_STREAM_HOSTS`, `COGNEN_ALLOW_PRIVATE_STREAMS`, `COGNEN_STREAM_SCHEMES`, `COGNEN_STREAM_PORTS`, `COGNEN_ALLOW_URL_CREDENTIALS`), file roots (`COGNEN_VIDEO_DIRS`), camera opt-in (`COGNEN_ALLOW_CAMERA_SOURCE`), log redaction (`server/security.py`, `tests/test_server_security.py`)
[ ] Exercise against a real network/CCTV deployment (DNS-rebinding window for TLS streams, FFmpeg RTSP redirects remain) -- NOT DONE

### Phase 24 — v2.0 measurement, context gate, posture, retrained model
[x] DACE measurement port (`features/dace_measurement.py`: matrix pitch vs per-track reference, blendshape gaze, gaze-corrected openness/PERCLOS), one down = negative convention, `features/track_features.py`
[x] FaceLandmarker fixed (IMAGE mode, loud on failure, clipped ROI), `_MP_TO_68` rebuilt and geometry-tested; real ByteTrack (boxmot 16)
[x] Context gate on by default (`COGNEN_HAND_PROXIMITY=pose|hands|off`), `absent` reachable, CMCV wired to the behaviour label
[x] 15-dim posture vector (`encode_feature_vector_v2`; `encode_feature_vector` stays 13-dim v1-compatible); `behavior_classifier_v3` trained on the synthetic generator and set as the server default (circular 1.000; see `behavior_classifier_v3_synthetic_note.md`)
[x] Runtime fixes: shared ONNX session, state pruning, track id 0, one-to-one seats, thread offload, task tracking
[x] Contract tests: `encode13` (TS to Python), CMCV (TS to Python), `encode15_v2` (Python), `legacy_pitch_constants.json`
[ ] Calibrate theta_down / gamma_down / k on pilot recordings -- NOT DONE
[ ] Enforce the 5 Hz training cadence on the server (default `COGNEN_PROCESS_EVERY_N=1`; use 3 at 15 FPS) -- OPEN

### Phase 25 — Training and evaluation fixes
[x] Group-leakage guards (`assert_no_group_leakage`, synthetic group ids, subject-level DAiSEE/EngageNet groups, loaders' own splits), selection by val macro-F1, ONNX/PyTorch parity check, `--seed`
[x] `TrackSequenceEncoder` parity with inference, 5 Hz timestamp sampling, no-face frames encoded like inference, `GROUNDED_CLASSES`
[x] Evaluation: non-vacuous TCC, deployed-classifier evaluation, real IDF1, unclipped MOTA, test-split-only tracking, per-stage latency benchmark, event-based occupancy latency, `data_source` / `claims_allowed`, `synthetic_holdout_eval.py`
[x] Retracted reports archived (`v2.0/evaluation/reports/archive/` with README); Open Images MIDs from the official CSV; `perception/study_objects.py`
[ ] Any real-data training or evaluation (DAiSEE/EngageNet/classroom) -- NOT DONE
[ ] Real MOTA/IDF1, occupancy accuracy and end-to-end latency on real video -- NOT DONE

## Remaining Before Production
[ ] Download DAiSEE + EngageNet datasets; retrain with real temporal signal -- NOT DONE (any accuracy figure is a target)
[ ] Fine-tune YOLOv8n on Open Images book/phone subset -- NOT DONE (the detector is stock COCO)
[ ] Implement dual-timescale window (fast=7, slow=25 frames) in BehaviorClassifier -- NOT DONE
[ ] Evaluate on real classroom CCTV footage to get ground-truth MOTA/IDF1 -- NOT DONE

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
