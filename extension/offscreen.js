/**
 * CognEn Extension - Offscreen Document Script
 * Handles camera access and face detection in the offscreen context
 */

const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');

let isRunning = false;
let animationFrameId = null;
let metricsAggregator = null;
let mlEngine = null;
let modelsLoaded = false;

// Initialize
async function init() {
    console.log('Offscreen: Initializing...');

    try {
        const modelBaseUrl = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/';

        console.log('Offscreen: Loading face-api models...');
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(modelBaseUrl),
            faceapi.nets.faceLandmark68Net.loadFromUri(modelBaseUrl),
        ]);

        modelsLoaded = true;
        console.log('Offscreen: Models loaded successfully');

        if (typeof FaceMetricsAggregator !== 'undefined') {
            metricsAggregator = new FaceMetricsAggregator();
            console.log('Offscreen: FaceMetricsAggregator initialized');
        } else {
            console.error('Offscreen: FaceMetricsAggregator not found!');
        }

        if (typeof MLEngine !== 'undefined') {
            mlEngine = new MLEngine();
            console.log('Offscreen: MLEngine initialized');
        } else {
            console.error('Offscreen: MLEngine not found!');
        }

        // Store ready state
        chrome.storage.local.set({ offscreenReady: true });

    } catch (error) {
        console.error('Offscreen: Failed to load models:', error);
        chrome.storage.local.set({ offscreenError: error.message });
    }
}

async function startCamera() {
    console.log('Offscreen: Starting camera...');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        webcam.srcObject = stream;
        await webcam.play();

        isRunning = true;
        console.log('Offscreen: Camera started, beginning detection loop');

        // Store camera state
        chrome.storage.local.set({ cameraRunning: true });

        runDetection();
    } catch (error) {
        console.error('Offscreen: Camera error:', error);
        chrome.storage.local.set({ cameraError: error.message });
    }
}

function stopCamera() {
    console.log('Offscreen: Stopping camera...');
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

let detectionCount = 0;

async function runDetection() {
    if (!isRunning || !modelsLoaded) {
        if (isRunning) {
            animationFrameId = requestAnimationFrame(runDetection);
        }
        return;
    }

    if (webcam.readyState < 3) {
        animationFrameId = requestAnimationFrame(runDetection);
        return;
    }

    try {
        const detections = await faceapi
            .detectAllFaces(webcam, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks();

        detectionCount++;

        if (detections && detections.length > 0) {
            const landmarks = detections[0].landmarks.positions;
            const timestamp = Date.now();

            let prediction = null;
            let metrics = null;

            if (metricsAggregator && mlEngine) {
                metrics = metricsAggregator.calculateExtendedMetrics(
                    landmarks,
                    webcam.videoWidth,
                    webcam.videoHeight,
                    timestamp
                );

                prediction = mlEngine.predict(metrics);

                // Store results in chrome.storage for popup to read
                chrome.storage.local.set({
                    lastDetection: {
                        timestamp,
                        hasFace: true,
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
                    }
                });

                if (detectionCount % 30 === 0) {
                    console.log('Offscreen: Detection #' + detectionCount, 'Score:', prediction.rawScore);
                }
            }
        } else {
            chrome.storage.local.set({
                lastDetection: {
                    timestamp: Date.now(),
                    hasFace: false
                }
            });
        }
    } catch (error) {
        console.error('Offscreen: Detection error:', error);
    }

    if (isRunning) {
        animationFrameId = requestAnimationFrame(runDetection);
    }
}

// Listen for messages from popup/background
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('Offscreen: Received message:', message.type);

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
            sendResponse({ success: true });
            break;
        case 'GET_STATUS':
            sendResponse({
                modelsLoaded,
                isRunning,
                detectionCount
            });
            break;
    }
    return true;
});

// Initialize on load
console.log('Offscreen: Script loaded, calling init...');
init();
