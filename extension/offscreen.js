/**
 * CognEn Extension - Offscreen Document Script
 * Handles camera access and face detection in the offscreen context
 * Runs continuously in the background
 */

const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');

let isRunning = false;
let animationFrameId = null;
let metricsAggregator = null;
let mlEngine = null;
let modelsLoaded = false;
let initAttempts = 0;

// Initialize
async function init() {
    console.log('[Offscreen] Initializing... attempt:', ++initAttempts);

    try {
        // Check if faceapi is loaded
        if (typeof faceapi === 'undefined') {
            console.error('[Offscreen] faceapi not defined! Retrying in 1s...');
            setTimeout(init, 1000);
            return;
        }

        const modelBaseUrl = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/';

        console.log('[Offscreen] Loading face-api models...');
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(modelBaseUrl),
            faceapi.nets.faceLandmark68Net.loadFromUri(modelBaseUrl),
        ]);

        modelsLoaded = true;
        console.log('[Offscreen] Models loaded successfully');

        // Initialize ML components
        if (typeof FaceMetricsAggregator !== 'undefined') {
            metricsAggregator = new FaceMetricsAggregator();
            console.log('[Offscreen] FaceMetricsAggregator initialized');
        } else {
            console.error('[Offscreen] FaceMetricsAggregator not found! Check ml-core.js');
        }

        if (typeof MLEngine !== 'undefined') {
            mlEngine = new MLEngine();
            console.log('[Offscreen] MLEngine initialized');
        } else {
            console.error('[Offscreen] MLEngine not found! Check ml-core.js');
        }

        // Store ready state
        await chrome.storage.local.set({
            offscreenReady: true,
            offscreenError: null,
            modelsLoaded: true
        });

        console.log('[Offscreen] Ready and waiting for START_CAMERA command');

    } catch (error) {
        console.error('[Offscreen] Failed to initialize:', error);
        await chrome.storage.local.set({
            offscreenReady: false,
            offscreenError: error.message
        });
    }
}

async function startCamera() {
    console.log('[Offscreen] Starting camera...');

    if (!modelsLoaded) {
        console.error('[Offscreen] Models not loaded yet!');
        await chrome.storage.local.set({ cameraError: 'Models not loaded' });
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        webcam.srcObject = stream;

        await new Promise((resolve) => {
            webcam.onloadedmetadata = () => {
                webcam.play().then(resolve);
            };
        });

        isRunning = true;
        console.log('[Offscreen] Camera started successfully, beginning detection');

        await chrome.storage.local.set({
            cameraRunning: true,
            cameraError: null
        });

        // Start detection loop
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
    console.log('[Offscreen] Stopping camera...');
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
let lastLogTime = Date.now();

async function runDetection() {
    if (!isRunning) {
        console.log('[Offscreen] Detection stopped (isRunning=false)');
        return;
    }

    if (!modelsLoaded || !metricsAggregator || !mlEngine) {
        console.error('[Offscreen] ML components not ready');
        animationFrameId = requestAnimationFrame(runDetection);
        return;
    }

    if (webcam.readyState < 3) {
        animationFrameId = requestAnimationFrame(runDetection);
        return;
    }

    try {
        const detections = await faceapi
            .detectAllFaces(webcam, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 }))
            .withFaceLandmarks();

        frameCount++;
        const now = Date.now();

        if (detections && detections.length > 0) {
            const landmarks = detections[0].landmarks.positions;

            // Calculate extended metrics
            const metrics = metricsAggregator.calculateExtendedMetrics(
                landmarks,
                webcam.videoWidth,
                webcam.videoHeight,
                now
            );

            // Get ML prediction
            const prediction = mlEngine.predict(metrics);

            // Store results for popup to read
            await chrome.storage.local.set({
                lastDetection: {
                    timestamp: now,
                    hasFace: true,
                    frameCount,
                    prediction: {
                        rawScore: prediction.rawScore,
                        attentionState: prediction.attentionState,
                        confidence: prediction.confidence
                    },
                    metrics: {
                        ear: metrics.ear,
                        mar: metrics.mar,
                        blinkRate: metrics.blinkRate,
                        blinkDetected: metrics.blinkDetected,
                        fatigueIndex: metrics.fatigueIndex,
                        perclos: metrics.perclos,
                        headPoseStability: metrics.headPoseStability,
                        fidgetScore: metrics.fidgetScore,
                        gazeX: metrics.gazeX,
                        gazeY: metrics.gazeY
                    }
                }
            });

            // Log every 2 seconds
            if (now - lastLogTime > 2000) {
                console.log(`[Offscreen] Frame #${frameCount} | Score: ${prediction.rawScore} | State: ${prediction.attentionState} | Blink Rate: ${metrics.blinkRate?.toFixed(1)}`);
                lastLogTime = now;
            }

        } else {
            // No face detected
            await chrome.storage.local.set({
                lastDetection: {
                    timestamp: now,
                    hasFace: false,
                    frameCount
                }
            });

            if (now - lastLogTime > 2000) {
                console.log(`[Offscreen] Frame #${frameCount} | No face detected`);
                lastLogTime = now;
            }
        }
    } catch (error) {
        console.error('[Offscreen] Detection error:', error);
    }

    if (isRunning) {
        animationFrameId = requestAnimationFrame(runDetection);
    }
}

// Listen for messages
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('[Offscreen] Received message:', message.type);

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
            frameCount = 0;
            sendResponse({ success: true });
            break;

        case 'GET_STATUS':
            sendResponse({
                modelsLoaded,
                isRunning,
                frameCount,
                hasAggregator: !!metricsAggregator,
                hasEngine: !!mlEngine
            });
            break;

        default:
            sendResponse({ success: false, error: 'Unknown message type' });
    }

    return true;
});

// Initialize immediately
console.log('[Offscreen] Script loaded');
init();
