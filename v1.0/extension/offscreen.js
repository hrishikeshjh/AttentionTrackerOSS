/**
 * CognEn Extension - Offscreen Document Script
 * Handles camera access and face detection
 * Only responds to messages with 'type' field (from background)
 */

const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');

let isRunning = false;
let animationFrameId = null;
let metricsAggregator = null;
let mlEngine = null;
let proximityDetector = null;
let temporalClassifier = null;
let modelsLoaded = false;

// Initialize
async function init() {
    console.log('[Offscreen] Initializing...');

    try {
        if (typeof faceapi === 'undefined') {
            console.error('[Offscreen] faceapi not loaded, retrying...');
            setTimeout(init, 500);
            return;
        }

        const modelBaseUrl = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/';

        console.log('[Offscreen] Loading ML models...');
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(modelBaseUrl),
            faceapi.nets.faceLandmark68Net.loadFromUri(modelBaseUrl),
        ]);

        modelsLoaded = true;
        console.log('[Offscreen] Models loaded');

        if (typeof FaceMetricsAggregator !== 'undefined') {
            metricsAggregator = new FaceMetricsAggregator();
        }
        // MLEngineV2 (context-aware) is preferred; MLEngine is the legacy fallback
        if (typeof window.MLEngine !== 'undefined') {
            mlEngine = new window.MLEngine();
        }
        // Behaviour context modules (rule-based only, no ONNX in extension)
        if (typeof window.HandObjectProximityDetector !== 'undefined') {
            proximityDetector = new window.HandObjectProximityDetector();
        }
        if (typeof window.TemporalBehaviorClassifier !== 'undefined') {
            temporalClassifier = new window.TemporalBehaviorClassifier();
        }

        await chrome.storage.local.set({ modelsLoaded: true });
        console.log('[Offscreen] Ready for camera');

    } catch (error) {
        console.error('[Offscreen] Init error:', error);
        await chrome.storage.local.set({ offscreenError: error.message });
    }
}

async function startCamera() {
    console.log('[Offscreen] Starting camera...');

    if (!modelsLoaded) {
        console.log('[Offscreen] Models not ready, waiting...');
        await new Promise(r => setTimeout(r, 2000));
        if (!modelsLoaded) {
            await chrome.storage.local.set({ cameraError: 'Models not loaded' });
            return;
        }
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });

        webcam.srcObject = stream;
        await webcam.play();

        isRunning = true;
        await chrome.storage.local.set({ cameraRunning: true, cameraError: null });

        console.log('[Offscreen] Camera started, running detection');
        runDetection();

    } catch (error) {
        console.error('[Offscreen] Camera error:', error);
        await chrome.storage.local.set({
            cameraRunning: false,
            cameraError: error.message
        });
    }
}

function stopCamera() {
    console.log('[Offscreen] Stopping camera');
    isRunning = false;

    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    if (webcam.srcObject) {
        webcam.srcObject.getTracks().forEach(track => track.stop());
        webcam.srcObject = null;
    }

    chrome.storage.local.set({ cameraRunning: false });
}

let frameCount = 0;
let lastFrameTime = 0;
let lastWrittenScore = null;
let lastWriteTime = 0;

// Only 5fps of data is ever consumed downstream (content.js analyzes at
// 200ms intervals), so running face-api detection unthrottled on every
// requestAnimationFrame (~60fps) wastes the majority of CPU cycles during
// video calls, when CPU is already contended by the call's own codec.
const FRAME_BUDGET_MS = 200; // 5fps target, matches content.js's cadence
const STORAGE_WRITE_MIN_INTERVAL_MS = 500; // badge/UI doesn't need faster than 2fps
const STORAGE_WRITE_SCORE_DELTA = 2; // write early if score moved meaningfully

/**
 * Picks the largest-area detected face instead of trusting detections[0]
 * (face-api.js's detection order is not a stable, explicit selection
 * policy — with 2+ faces in frame it can flip between candidates
 * frame-to-frame, corrupting the fidget/stability trackers, which assume
 * continuity of a single identity's landmark stream). This is a lightweight
 * stopgap; the web app uses the fuller PrimaryFaceSelector with
 * identity-continuity hysteresis from src/lib/scoring/face-selection.ts —
 * porting that here is pending the extension bundle build (see
 * scoring/EQUIVALENCE.md).
 */
function selectLargestFace(detections) {
    if (!detections || detections.length === 0) return null;
    let best = detections[0];
    let bestArea = best.detection.box.width * best.detection.box.height;
    for (let i = 1; i < detections.length; i++) {
        const area = detections[i].detection.box.width * detections[i].detection.box.height;
        if (area > bestArea) {
            best = detections[i];
            bestArea = area;
        }
    }
    return best;
}

function writeDetectionResult(payload) {
    // Fire-and-forget: don't block the detection loop on the storage
    // round-trip, and skip redundant writes when nothing meaningful changed.
    const now = payload.timestamp;
    const score = payload.prediction ? payload.prediction.rawScore : null;
    const scoreChanged = lastWrittenScore === null || score === null ||
        Math.abs(score - lastWrittenScore) >= STORAGE_WRITE_SCORE_DELTA;
    const intervalElapsed = now - lastWriteTime >= STORAGE_WRITE_MIN_INTERVAL_MS;

    if (!scoreChanged && !intervalElapsed) return;

    lastWrittenScore = score;
    lastWriteTime = now;
    chrome.storage.local.set({ lastDetection: payload }).catch((err) => {
        console.error('[Offscreen] Storage write failed:', err);
    });
}

async function runDetection() {
    if (!isRunning || !modelsLoaded) return;

    if (webcam.readyState < 3) {
        animationFrameId = requestAnimationFrame(runDetection);
        return;
    }

    const frameStart = performance.now();
    if (frameStart - lastFrameTime < FRAME_BUDGET_MS) {
        // Not our turn yet — reschedule without doing detection work.
        animationFrameId = requestAnimationFrame(runDetection);
        return;
    }
    lastFrameTime = frameStart;

    try {
        const detections = await faceapi
            .detectAllFaces(webcam, new faceapi.TinyFaceDetectorOptions({ inputSize: 320 }))
            .withFaceLandmarks();

        frameCount++;
        const now = Date.now();
        const primary = selectLargestFace(detections);

        if (primary && metricsAggregator && mlEngine) {
            const landmarks = primary.landmarks.positions;

            const baseMetrics = metricsAggregator.calculateExtendedMetrics(
                landmarks,
                webcam.videoWidth,
                webcam.videoHeight,
                now
            );

            // Extension runs in metrics-only mode (no ONNX object detection)
            const detectedObjects = [];
            const handObjectProximity = proximityDetector
                ? proximityDetector.compute([], detectedObjects)
                : 'hands_not_visible';

            const metrics = Object.assign({}, baseMetrics, {
                detectedObjects,
                handObjectProximity,
            });

            // Update temporal classifier with face metrics only
            if (temporalClassifier) {
                metrics.behaviorResult = temporalClassifier.update({
                    metrics,
                    detectedObjects,
                    handProximity: handObjectProximity,
                });
            }

            const prediction = mlEngine.predict(metrics);

            writeDetectionResult({
                timestamp: now,
                hasFace: true,
                frameCount,
                prediction: {
                    rawScore: prediction.rawScore,
                    attentionState: prediction.attentionState,
                    confidence: prediction.confidence
                },
                metrics: {
                    blinkRate: metrics.blinkRate,
                    fatigueIndex: metrics.fatigueIndex,
                    perclos: metrics.perclos,
                    ear: metrics.ear
                }
            });

            if (frameCount % 60 === 0) {
                console.log(`[Offscreen] Frame ${frameCount}: Score ${prediction.rawScore}`);
            }
        } else {
            writeDetectionResult({
                timestamp: now,
                hasFace: false,
                frameCount
            });
        }
    } catch (error) {
        console.error('[Offscreen] Detection error:', error);
    }

    if (isRunning) {
        animationFrameId = requestAnimationFrame(runDetection);
    }
}

// Message handler - only responds to 'type' messages from background
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    // Only handle messages with 'type' field
    if (!message.type) return false;

    console.log('[Offscreen] Received:', message.type);

    switch (message.type) {
        case 'START_CAMERA':
            startCamera();
            sendResponse({ success: true });
            break;
        case 'STOP_CAMERA':
            stopCamera();
            sendResponse({ success: true });
            break;
        case 'RESET_ML':
            if (metricsAggregator) metricsAggregator.reset();
            if (mlEngine) mlEngine.reset();
            if (temporalClassifier) temporalClassifier.reset();
            frameCount = 0;
            sendResponse({ success: true });
            break;
        default:
            // Ignore unknown types
            break;
    }

    return true;
});

// Initialize
console.log('[Offscreen] Script loaded');
init();
