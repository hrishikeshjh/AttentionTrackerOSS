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
    try {
        const modelBaseUrl = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/';

        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(modelBaseUrl),
            faceapi.nets.faceLandmark68Net.loadFromUri(modelBaseUrl),
        ]);

        modelsLoaded = true;

        if (typeof FaceMetricsAggregator !== 'undefined') {
            metricsAggregator = new FaceMetricsAggregator();
        }
        if (typeof MLEngine !== 'undefined') {
            mlEngine = new MLEngine();
        }

        chrome.runtime.sendMessage({ type: 'OFFSCREEN_READY' });
    } catch (error) {
        console.error('Failed to load models:', error);
        chrome.runtime.sendMessage({ type: 'OFFSCREEN_ERROR', error: error.message });
    }
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        webcam.srcObject = stream;
        await webcam.play();

        isRunning = true;
        chrome.runtime.sendMessage({ type: 'CAMERA_STARTED' });

        runDetection();
    } catch (error) {
        console.error('Camera error:', error);
        chrome.runtime.sendMessage({ type: 'CAMERA_ERROR', error: error.message });
    }
}

function stopCamera() {
    isRunning = false;

    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    if (webcam.srcObject) {
        webcam.srcObject.getTracks().forEach(track => track.stop());
        webcam.srcObject = null;
    }

    chrome.runtime.sendMessage({ type: 'CAMERA_STOPPED' });
}

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

                // Send data to background/popup
                chrome.runtime.sendMessage({
                    type: 'DETECTION_RESULT',
                    data: {
                        prediction,
                        metrics: {
                            blinkRate: metrics.blinkRate,
                            fatigueIndex: metrics.fatigueIndex,
                            perclos: metrics.perclos,
                        }
                    }
                });
            }
        } else {
            chrome.runtime.sendMessage({
                type: 'DETECTION_RESULT',
                data: { noFace: true }
            });
        }
    } catch (error) {
        console.error('Detection error:', error);
    }

    if (isRunning) {
        animationFrameId = requestAnimationFrame(runDetection);
    }
}

// Listen for messages from popup/background
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    switch (message.type) {
        case 'START_CAMERA':
            startCamera();
            break;
        case 'STOP_CAMERA':
            stopCamera();
            break;
        case 'RESET_ML':
            if (metricsAggregator) metricsAggregator.reset();
            if (mlEngine) mlEngine.reset();
            break;
    }
});

// Initialize on load
init();
