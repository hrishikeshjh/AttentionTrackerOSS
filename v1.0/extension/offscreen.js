/**
 * CognEn Extension - Offscreen Document Script (standalone mode)
 *
 * Owns the webcam and the analysis loop. Only responds to messages that
 * carry a `type` field (sent by background.js).
 *
 * Loop: a cancellable setTimeout chain at a fixed 200 ms tick (5 FPS).
 * requestAnimationFrame is not used because an offscreen document is never
 * visible, and rAF callbacks are throttled or paused for hidden documents.
 *
 * Scoring: identical to the Meet-mode analyser and the web app -
 * PrimaryFaceSelector -> FaceMetricsAggregator every tick -> legacy-weight
 * MLEngine.predict once per second (EMA alpha 0.15, thresholds 70/40).
 * Each prediction (or no-face status) is sent once to the background as
 * {action: 'DETECTION', detection} with a monotonically increasing `seq`.
 */

(function () {
    'use strict';

    const webcam = document.getElementById('webcam');
    const core = window.__cognen && window.__cognen.mlCore;

    const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/';
    const FRAME_INTERVAL_MS = core ? core.FRAME_INTERVAL_MS : 200;
    const PREDICT_INTERVAL_MS = core ? core.PREDICT_INTERVAL_MS : 1000;

    let isRunning = false;
    let loopTimer = null;
    let loopGeneration = 0;
    let modelsLoaded = false;
    let initPromise = null;

    let faceSelector = null;
    let metricsAggregator = null;
    let mlEngine = null;
    let latestMetrics = null;
    let lastPredictTime = 0;
    let seq = 0;
    let frameCount = 0;

    async function init() {
        if (typeof faceapi === 'undefined' || !core) {
            await chrome.storage.local.set({ offscreenError: 'face-api or ml-core failed to load' });
            return;
        }
        try {
            await Promise.all([
                faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
                faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
            ]);
            faceSelector = new core.PrimaryFaceSelector();
            metricsAggregator = new core.FaceMetricsAggregator();
            mlEngine = new core.MLEngine();
            modelsLoaded = true;
            await chrome.storage.local.set({ modelsLoaded: true, offscreenError: null });
        } catch (error) {
            console.error('[Offscreen] Init error:', error);
            await chrome.storage.local.set({ offscreenError: error.message });
        }
    }

    function resetScoring() {
        if (faceSelector) faceSelector.reset();
        if (metricsAggregator) metricsAggregator.reset();
        if (mlEngine) mlEngine.reset();
        latestMetrics = null;
        lastPredictTime = 0;
        frameCount = 0;
    }

    async function startCamera() {
        await initPromise;
        if (!modelsLoaded) {
            await chrome.storage.local.set({ cameraRunning: false, cameraError: 'Models not loaded' });
            return;
        }
        if (isRunning) return;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
            webcam.srcObject = stream;
            await webcam.play();

            resetScoring();
            isRunning = true;
            loopGeneration += 1;
            await chrome.storage.local.set({ cameraRunning: true, cameraError: null });
            scheduleTick(loopGeneration, 0);
        } catch (error) {
            console.error('[Offscreen] Camera error:', error);
            await chrome.storage.local.set({ cameraRunning: false, cameraError: error.message });
        }
    }

    function stopCamera() {
        isRunning = false;
        loopGeneration += 1; // invalidates any in-flight tick
        if (loopTimer) {
            clearTimeout(loopTimer);
            loopTimer = null;
        }
        if (webcam.srcObject) {
            webcam.srcObject.getTracks().forEach((track) => track.stop());
            webcam.srcObject = null;
        }
        chrome.storage.local.set({ cameraRunning: false });
    }

    function scheduleTick(generation, delay) {
        loopTimer = setTimeout(() => tick(generation), delay);
    }

    function sendDetection(detection) {
        seq += 1;
        detection.seq = seq;
        chrome.runtime.sendMessage({ action: 'DETECTION', detection }).catch(() => {
            // Service worker restarting; the next tick's message wakes it.
        });
    }

    async function analyzeFrame() {
        const detections = await faceapi
            .detectAllFaces(webcam, new faceapi.TinyFaceDetectorOptions({ inputSize: 320 }))
            .withFaceLandmarks();
        frameCount++;

        const now = Date.now();
        const width = webcam.videoWidth;
        const height = webcam.videoHeight;
        const primary = faceSelector.select(
            (detections || []).map((d) => ({ box: d.detection.box, detection: d })),
            width,
            height
        );

        latestMetrics = primary
            ? metricsAggregator.calculateExtendedMetrics(primary.detection.landmarks.positions, width, height, now)
            : null;

        if (now - lastPredictTime < PREDICT_INTERVAL_MS) return;
        lastPredictTime = now;

        if (!latestMetrics) {
            sendDetection({ timestamp: now, hasFace: false, frameCount, prediction: null, metrics: null });
            return;
        }

        // Base metrics only (no detectedObjects) -> LEGACY weights, as in the web app.
        const prediction = mlEngine.predict(latestMetrics);
        sendDetection({
            timestamp: now,
            hasFace: true,
            frameCount,
            prediction: {
                rawScore: prediction.rawScore,
                attentionState: prediction.attentionState,
                confidence: prediction.confidence,
            },
            metrics: {
                blinkRate: latestMetrics.blinkRate,
                fatigueIndex: latestMetrics.fatigueIndex,
                perclos: latestMetrics.perclos,
                ear: latestMetrics.ear,
            },
        });
    }

    async function tick(generation) {
        if (!isRunning || generation !== loopGeneration) return;
        const started = Date.now();
        try {
            if (webcam.readyState >= 3) await analyzeFrame();
        } catch (error) {
            console.error('[Offscreen] Detection error:', error);
        }
        if (isRunning && generation === loopGeneration) {
            scheduleTick(generation, Math.max(0, FRAME_INTERVAL_MS - (Date.now() - started)));
        }
    }

    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (!message || !message.type) return false;
        if (sender.id !== chrome.runtime.id) return false;

        switch (message.type) {
            case 'START_CAMERA':
                startCamera();
                sendResponse({ success: true });
                return false;
            case 'STOP_CAMERA':
                stopCamera();
                sendResponse({ success: true });
                return false;
            case 'RESET_ML':
                resetScoring();
                sendResponse({ success: true });
                return false;
            default:
                return false;
        }
    });

    initPromise = init();
})();
