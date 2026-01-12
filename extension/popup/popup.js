/**
 * CognEn Extension Popup Script
 * Handles webcam, face detection, and attention tracking in the popup
 */

// DOM Elements
const webcam = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const webcamPlaceholder = document.getElementById('webcamPlaceholder');
const loadingOverlay = document.getElementById('loadingOverlay');
const errorOverlay = document.getElementById('errorOverlay');
const toggleCameraBtn = document.getElementById('toggleCameraBtn');
const toggleSessionBtn = document.getElementById('toggleSessionBtn');
const cameraButtonText = document.getElementById('cameraButtonText');
const sessionButtonText = document.getElementById('sessionButtonText');
const playIcon = document.getElementById('playIcon');
const stopIcon = document.getElementById('stopIcon');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const scoreNumber = document.getElementById('scoreNumber');
const scoreProgress = document.getElementById('scoreProgress');
const scoreState = document.getElementById('scoreState');
const sessionTimeEl = document.getElementById('sessionTime');
const avgScoreEl = document.getElementById('avgScore');
const blinkRateEl = document.getElementById('blinkRate');
const fatigueLevelEl = document.getElementById('fatigueLevel');
const retryBtn = document.getElementById('retryBtn');
const settingsBtn = document.getElementById('settingsBtn');

// State
let isCameraOn = false;
let isSessionActive = false;
let modelsLoaded = false;
let animationFrameId = null;
let sessionStartTime = null;
let sessionTimerInterval = null;
let sessionScores = [];
let metricsAggregator = null;
let mlEngine = null;

// Model URL - using local bundled file
const MODEL_URL = chrome.runtime.getURL('models/');

// Initialize
async function init() {
    try {
        // Check if faceapi is available
        if (typeof faceapi === 'undefined') {
            throw new Error('face-api.js not loaded');
        }

        // Load face-api models from jsdelivr (models only, not scripts)
        const modelBaseUrl = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/';

        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(modelBaseUrl),
            faceapi.nets.faceLandmark68Net.loadFromUri(modelBaseUrl),
        ]);

        modelsLoaded = true;
        loadingOverlay.classList.add('hidden');

        // Initialize ML components from ml-core.js
        if (typeof FaceMetricsAggregator !== 'undefined') {
            metricsAggregator = new FaceMetricsAggregator();
        }
        if (typeof MLEngine !== 'undefined') {
            mlEngine = new MLEngine();
        }

        // Check if we have a saved session
        chrome.storage.local.get(['sessionActive', 'sessionData'], (result) => {
            if (result.sessionActive) {
                console.log('Resuming previous session');
            }
        });
    } catch (error) {
        console.error('Failed to load models:', error);
        loadingOverlay.classList.add('hidden');
        errorOverlay.classList.remove('hidden');
    }
}

// Camera Controls
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        webcam.srcObject = stream;
        await webcam.play();

        isCameraOn = true;
        webcamPlaceholder.classList.add('hidden');
        cameraButtonText.textContent = 'Stop Camera';
        toggleCameraBtn.classList.add('active');
        toggleSessionBtn.disabled = false;

        statusDot.classList.add('active');
        statusText.textContent = 'Camera On';

        // Start detection loop
        runDetection();
    } catch (error) {
        console.error('Camera error:', error);
        alert('Could not access camera. Please check permissions.');
    }
}

function stopCamera() {
    if (webcam.srcObject) {
        webcam.srcObject.getTracks().forEach(track => track.stop());
        webcam.srcObject = null;
    }

    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    isCameraOn = false;
    webcamPlaceholder.classList.remove('hidden');
    cameraButtonText.textContent = 'Start Camera';
    toggleCameraBtn.classList.remove('active');
    toggleSessionBtn.disabled = true;

    statusDot.classList.remove('active', 'recording');
    statusText.textContent = 'Inactive';

    // Clear overlay
    const ctx = overlay.getContext('2d');
    if (ctx) {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    // Stop session if active
    if (isSessionActive) {
        stopSession();
    }
}

function toggleCamera() {
    if (isCameraOn) {
        stopCamera();
    } else {
        startCamera();
    }
}

// Session Controls
function startSession() {
    isSessionActive = true;
    sessionStartTime = Date.now();
    sessionScores = [];

    // Reset ML components
    if (metricsAggregator) {
        metricsAggregator.reset();
    }
    if (mlEngine) {
        mlEngine.reset();
    }

    // Update UI
    playIcon.style.display = 'none';
    stopIcon.style.display = 'block';
    sessionButtonText.textContent = 'Stop Session';
    toggleSessionBtn.classList.add('recording');
    statusDot.classList.add('recording');
    statusText.textContent = 'Recording';

    // Start session timer
    sessionTimerInterval = setInterval(updateSessionTimer, 1000);

    // Save state
    chrome.storage.local.set({ sessionActive: true, sessionStartTime });

    // Notify background
    chrome.runtime.sendMessage({ type: 'SESSION_START' });
}

function stopSession() {
    isSessionActive = false;

    // Update UI
    playIcon.style.display = 'block';
    stopIcon.style.display = 'none';
    sessionButtonText.textContent = 'Start Session';
    toggleSessionBtn.classList.remove('recording');
    statusDot.classList.remove('recording');
    statusText.textContent = isCameraOn ? 'Camera On' : 'Inactive';

    // Stop timer
    if (sessionTimerInterval) {
        clearInterval(sessionTimerInterval);
        sessionTimerInterval = null;
    }

    // Calculate final stats
    const avgScore = sessionScores.length > 0
        ? Math.round(sessionScores.reduce((a, b) => a + b, 0) / sessionScores.length)
        : 0;

    // Save session data
    chrome.storage.local.set({
        sessionActive: false,
        lastSession: {
            startTime: sessionStartTime,
            endTime: Date.now(),
            scores: sessionScores,
            avgScore
        }
    });

    // Notify background
    chrome.runtime.sendMessage({ type: 'SESSION_END', data: { avgScore } });
}

function toggleSession() {
    if (isSessionActive) {
        stopSession();
    } else {
        startSession();
    }
}

function updateSessionTimer() {
    if (!sessionStartTime) return;

    const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
    const minutes = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const seconds = String(elapsed % 60).padStart(2, '0');
    sessionTimeEl.textContent = `${minutes}:${seconds}`;
}

// Detection Loop
async function runDetection() {
    if (!isCameraOn || !modelsLoaded) {
        animationFrameId = requestAnimationFrame(runDetection);
        return;
    }

    if (webcam.readyState < 3) {
        animationFrameId = requestAnimationFrame(runDetection);
        return;
    }

    // Set canvas size
    overlay.width = webcam.videoWidth;
    overlay.height = webcam.videoHeight;

    try {
        const detections = await faceapi
            .detectAllFaces(webcam, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks();

        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        if (detections && detections.length > 0) {
            // Draw landmarks
            faceapi.draw.drawFaceLandmarks(overlay, detections);

            const landmarks = detections[0].landmarks.positions;
            const timestamp = Date.now();

            // Calculate metrics
            let prediction = null;
            let extendedMetrics = null;

            if (metricsAggregator && mlEngine) {
                extendedMetrics = metricsAggregator.calculateExtendedMetrics(
                    landmarks,
                    webcam.videoWidth,
                    webcam.videoHeight,
                    timestamp
                );

                prediction = mlEngine.predict(extendedMetrics);

                // Update UI with metrics
                updateUI(prediction, extendedMetrics);

                // Record data if session is active
                if (isSessionActive) {
                    sessionScores.push(prediction.rawScore);

                    // Update badge
                    chrome.runtime.sendMessage({
                        type: 'UPDATE_SCORE',
                        data: { score: prediction.rawScore }
                    });
                }
            } else {
                // Fallback: use basic calculation
                const score = calculateBasicScore(landmarks);
                updateUIBasic(score);

                if (isSessionActive) {
                    sessionScores.push(score);
                }
            }
        } else {
            // No face detected
            updateUINoFace();
        }
    } catch (error) {
        console.error('Detection error:', error);
    }

    animationFrameId = requestAnimationFrame(runDetection);
}

// UI Updates
function updateUI(prediction, metrics) {
    // Update score
    const score = prediction.rawScore;
    scoreNumber.textContent = score;

    // Update ring progress (circumference is ~283)
    const offset = 283 - (283 * score / 100);
    scoreProgress.style.strokeDashoffset = offset;

    // Update ring color
    scoreProgress.classList.remove('good', 'warning', 'danger');
    if (score >= 70) {
        scoreProgress.classList.add('good');
    } else if (score >= 40) {
        scoreProgress.classList.add('warning');
    } else {
        scoreProgress.classList.add('danger');
    }

    // Update state
    scoreState.textContent = prediction.attentionState;
    scoreState.className = 'score-state ' + prediction.attentionState.toLowerCase();

    // Update stats
    if (sessionScores.length > 0) {
        const avg = Math.round(sessionScores.reduce((a, b) => a + b, 0) / sessionScores.length);
        avgScoreEl.textContent = avg + '%';
    }

    if (metrics) {
        blinkRateEl.textContent = metrics.blinkRate.toFixed(1);

        // Fatigue level
        if (metrics.fatigueIndex < 30) {
            fatigueLevelEl.textContent = 'Low';
            fatigueLevelEl.style.color = '#3FB950';
        } else if (metrics.fatigueIndex < 60) {
            fatigueLevelEl.textContent = 'Med';
            fatigueLevelEl.style.color = '#FFA657';
        } else {
            fatigueLevelEl.textContent = 'High';
            fatigueLevelEl.style.color = '#F85149';
        }
    }
}

function updateUIBasic(score) {
    scoreNumber.textContent = score;

    const offset = 283 - (283 * score / 100);
    scoreProgress.style.strokeDashoffset = offset;

    scoreProgress.classList.remove('good', 'warning', 'danger');
    if (score >= 70) {
        scoreProgress.classList.add('good');
        scoreState.textContent = 'Attentive';
        scoreState.className = 'score-state attentive';
    } else if (score >= 40) {
        scoreProgress.classList.add('warning');
        scoreState.textContent = 'Distracted';
        scoreState.className = 'score-state distracted';
    } else {
        scoreProgress.classList.add('danger');
        scoreState.textContent = 'Multitasking';
        scoreState.className = 'score-state multitasking';
    }

    if (sessionScores.length > 0) {
        const avg = Math.round(sessionScores.reduce((a, b) => a + b, 0) / sessionScores.length);
        avgScoreEl.textContent = avg + '%';
    }
}

function updateUINoFace() {
    scoreNumber.textContent = '-';
    scoreProgress.style.strokeDashoffset = 283;
    scoreProgress.classList.remove('good', 'warning', 'danger');
    scoreState.textContent = 'No Face';
    scoreState.className = 'score-state';
}

// Basic score calculation fallback
function calculateBasicScore(landmarks) {
    // Simple EAR-based attention score
    const leftEye = [36, 37, 38, 39, 40, 41].map(i => landmarks[i]);
    const rightEye = [42, 43, 44, 45, 46, 47].map(i => landmarks[i]);

    const calcEAR = (eye) => {
        const vertDist = (
            Math.sqrt((eye[1].x - eye[5].x) ** 2 + (eye[1].y - eye[5].y) ** 2) +
            Math.sqrt((eye[2].x - eye[4].x) ** 2 + (eye[2].y - eye[4].y) ** 2)
        );
        const horzDist = Math.sqrt((eye[0].x - eye[3].x) ** 2 + (eye[0].y - eye[3].y) ** 2);
        return vertDist / (2 * horzDist);
    };

    const leftEAR = calcEAR(leftEye);
    const rightEAR = calcEAR(rightEye);
    const avgEAR = (leftEAR + rightEAR) / 2;

    // Map EAR to score (0.2-0.4 is typical range)
    const score = Math.min(100, Math.max(0, (avgEAR - 0.15) / 0.25 * 100));
    return Math.round(score);
}

// Event Listeners
toggleCameraBtn.addEventListener('click', toggleCamera);
toggleSessionBtn.addEventListener('click', toggleSession);

settingsBtn.addEventListener('click', () => {
    // Open options page or show settings modal
    alert('Settings coming soon!');
});

retryBtn.addEventListener('click', () => {
    errorOverlay.classList.add('hidden');
    loadingOverlay.classList.remove('hidden');
    init();
});

// Handle popup close - keep session running in background
window.addEventListener('beforeunload', () => {
    if (isSessionActive) {
        // Session continues in background
        chrome.runtime.sendMessage({ type: 'POPUP_CLOSED', sessionActive: true });
    }
});

// Initialize on load
document.addEventListener('DOMContentLoaded', init);
