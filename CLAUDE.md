# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent Role

You are the lead ML engineer for CognEn. Before starting any implementation task, read `ARCHITECTURE.md`. If a proposed change conflicts with `ARCHITECTURE.md`, explain the conflict, propose a resolution, and wait for approval.

For each task, produce: implementation plan, code modules, dataset suggestions, evaluation metrics, and test scripts. Prioritize real-time feasibility, interpretable models, and modular architecture.

## Commands

All commands run from the `v1.0/` directory:

```bash
npm run dev          # Start Next.js dev server on port 9002 (Turbopack)
npm run build        # Production build
npm run lint         # ESLint
npm run typecheck    # TypeScript type check (no emit)
npm run genkit:dev   # Start Genkit AI dev server
npm run genkit:watch # Start Genkit AI dev server with file watching
```

The Chrome extension is loaded manually via Chrome's "Load unpacked" from `v1.0/extension/`.

## Architecture Overview

CognEn has two versioned sub-projects:

### v1.0 — Browser-Based Single-User Tracking (active)

```
Webcam (640×480 @ 5 FPS)
  → face-api.js TinyFaceDetector → 68-point landmarks
  → FaceMetricsAggregator (EAR, MAR, gaze, head pose, blink, fatigue)
  → MLEngine (weighted scoring + EMA α=0.15)
  → AttentionState: Attentive / Distracted / Multitasking
  → Firebase Firestore (session data)
  → Genkit AI (Google Gemini, PDF report generation)
```

Two surfaces share the same ML logic:
- **Next.js web app** (`v1.0/src/`) — dashboard, auth, session management
- **Chrome extension** (`v1.0/extension/`) — runs in-browser on Meet/Zoom/Teams via Manifest V3 offscreen page

### v2.0 — Multi-Person CCTV Room Analysis (planned, server-based)

Python/FastAPI server. Adds ByteTrack person tracking, YOLOv8 object/pose detection, per-seat occupancy, and room-level dashboard. Reuses v1.0 ML logic per tracked person. See `ARCHITECTURE.md` §3 for full design.

## Key Files

### v1.0 Web App
| File | Role |
|------|------|
| `v1.0/src/lib/types.ts` | Core type definitions: `ExtendedFaceMetrics`, `MLPrediction`, `SessionAnalytics` |
| `v1.0/src/lib/face-geometry.ts` | Face math: EAR/MAR, `BlinkDetector`, `HeadPoseTracker`, `FatigueCalculator`, `FaceMetricsAggregator` |
| `v1.0/src/lib/ml-engine.ts` | `MLEngine` class: weighted scoring, EMA smoothing, attention classification |
| `v1.0/src/lib/analytics.ts` | `SessionAnalyticsEngine`: trend detection, peak/low periods, fatigue onset |
| `v1.0/src/lib/firebase.ts` | Firebase init (Auth + Firestore) |
| `v1.0/src/app/actions.ts` | Server action: calls Genkit flow to generate PDF report |
| `v1.0/src/ai/flows/` | Genkit AI flows: `analyze-facial-features.ts`, `generate-session-summary.ts` |

### v1.0 Chrome Extension
| File | Role |
|------|------|
| `v1.0/extension/lib/ml-core.js` | Core ML processing (mirrors web app logic) |
| `v1.0/extension/offscreen.js` | Offscreen document: webcam capture + face-api.js inference |
| `v1.0/extension/background.js` | Service worker: alarm scheduling, state coordination |
| `v1.0/extension/content.js` | Content script: injects attention widget into web pages |

## v1.0 Upgrade Path (context for new features)

The planned v1.0 upgrade adds context-aware scoring to fix the core limitation: uniform downward-gaze penalty that incorrectly marks book-reading as distracted. New modules to add:
- `object-detector.ts` — YOLOv8-nano ONNX wrapper (book/phone/laptop detection)
- `hand-tracker.ts` — MediaPipe Hands wrapper
- `temporal-classifier.ts` — 15-frame sliding window 1D-CNN behavior classifier

Modified files: `ml-engine.ts` (context-aware pitch scoring), `face-geometry.ts` (add `HandObjectProximityDetector`), `types.ts` (extend `ExtendedFaceMetrics`).

See `ARCHITECTURE.md` §2 for full spec including modified feature weights.

## Attention Scoring

Thresholds in `ml-engine.ts`:
- Score ≥ 70 → `Attentive`
- Score 40–69 → `Distracted`
- Score < 40 → `Multitasking`

EMA smoothing factor α = 0.15. Feature weights are defined in `FEATURE_WEIGHTS` constant (eye-based features carry highest weight: `ear: 0.15`, `perclos: 0.12`).
