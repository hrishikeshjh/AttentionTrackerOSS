/**
 * CognEn Extension Popup Script
 * Uses offscreen document for camera access
 */

// DOM Elements
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
let sessionStartTime = null;
let sessionTimerInterval = null;
let sessionScores = [];

// Hide loading overlay immediately (models load in offscreen document)
loadingOverlay.classList.add('hidden');

// Initialize
async function init() {
    // Check current state from background
    chrome.runtime.sendMessage({ type: 'GET_STATE' }, (response) => {
        if (response) {
            if (response.sessionActive) {
                // Restore session state
                isSessionActive = true;
                updateSessionUI(true);
            }
            if (response.offscreenDocumentCreated) {
                isCameraOn = true;
                updateCameraUI(true);
            }
        }
    });
}

// Camera Controls
async function startCamera() {
    cameraButtonText.textContent = 'Starting...';
    toggleCameraBtn.disabled = true;

    chrome.runtime.sendMessage({ type: 'START_CAMERA_REQUEST' }, (response) => {
        if (response && response.success) {
            isCameraOn = true;
            updateCameraUI(true);
        } else {
            alert('Failed to start camera: ' + (response?.error || 'Unknown error'));
            cameraButtonText.textContent = 'Start Camera';
            toggleCameraBtn.disabled = false;
        }
    });
}

function stopCamera() {
    chrome.runtime.sendMessage({ type: 'STOP_CAMERA_REQUEST' }, (response) => {
        isCameraOn = false;
        updateCameraUI(false);

        if (isSessionActive) {
            stopSession();
        }
    });
}

function updateCameraUI(isOn) {
    if (isOn) {
        webcamPlaceholder.innerHTML = `
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color: #3FB950;">
        <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
        <path d="M8 12l3 3 5-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <p style="color: #3FB950;">Camera Active</p>
      <p style="font-size: 10px; color: #8B949E;">Processing in background</p>
    `;
        cameraButtonText.textContent = 'Stop Camera';
        toggleCameraBtn.classList.add('active');
        toggleCameraBtn.disabled = false;
        toggleSessionBtn.disabled = false;

        statusDot.classList.add('active');
        statusText.textContent = 'Camera On';
    } else {
        webcamPlaceholder.innerHTML = `
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="2" y="5" width="20" height="14" rx="2" stroke="currentColor" stroke-width="2"/>
        <circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="2"/>
      </svg>
      <p>Camera Off</p>
    `;
        cameraButtonText.textContent = 'Start Camera';
        toggleCameraBtn.classList.remove('active');
        toggleCameraBtn.disabled = false;
        toggleSessionBtn.disabled = true;

        statusDot.classList.remove('active', 'recording');
        statusText.textContent = 'Inactive';
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

    // Reset ML in offscreen document
    chrome.runtime.sendMessage({ type: 'RESET_ML' });

    updateSessionUI(true);

    // Start session timer
    sessionTimerInterval = setInterval(updateSessionTimer, 1000);

    // Notify background
    chrome.runtime.sendMessage({ type: 'SESSION_START' });
    chrome.storage.local.set({ sessionActive: true, sessionStartTime });
}

function stopSession() {
    isSessionActive = false;

    updateSessionUI(false);

    if (sessionTimerInterval) {
        clearInterval(sessionTimerInterval);
        sessionTimerInterval = null;
    }

    const avgScore = sessionScores.length > 0
        ? Math.round(sessionScores.reduce((a, b) => a + b, 0) / sessionScores.length)
        : 0;

    chrome.runtime.sendMessage({ type: 'SESSION_END', data: { avgScore } });
    chrome.storage.local.set({
        sessionActive: false,
        lastSession: {
            startTime: sessionStartTime,
            endTime: Date.now(),
            scores: sessionScores,
            avgScore
        }
    });
}

function updateSessionUI(isActive) {
    if (isActive) {
        playIcon.style.display = 'none';
        stopIcon.style.display = 'block';
        sessionButtonText.textContent = 'Stop Session';
        toggleSessionBtn.classList.add('recording');
        statusDot.classList.add('recording');
        statusText.textContent = 'Recording';
    } else {
        playIcon.style.display = 'block';
        stopIcon.style.display = 'none';
        sessionButtonText.textContent = 'Start Session';
        toggleSessionBtn.classList.remove('recording');
        statusDot.classList.remove('recording');
        statusText.textContent = isCameraOn ? 'Camera On' : 'Inactive';
    }
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

// Listen for detection results from offscreen document
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'DETECTION_RESULT') {
        if (message.data && !message.data.noFace) {
            updateUI(message.data.prediction, message.data.metrics);

            if (isSessionActive && message.data.prediction) {
                sessionScores.push(message.data.prediction.rawScore);
            }
        } else {
            updateUINoFace();
        }
    }
});

// UI Updates
function updateUI(prediction, metrics) {
    if (!prediction) return;

    const score = prediction.rawScore;
    scoreNumber.textContent = score;

    const offset = 283 - (283 * score / 100);
    scoreProgress.style.strokeDashoffset = offset;

    scoreProgress.classList.remove('good', 'warning', 'danger');
    if (score >= 70) {
        scoreProgress.classList.add('good');
    } else if (score >= 40) {
        scoreProgress.classList.add('warning');
    } else {
        scoreProgress.classList.add('danger');
    }

    scoreState.textContent = prediction.attentionState;
    scoreState.className = 'score-state ' + prediction.attentionState.toLowerCase();

    if (sessionScores.length > 0) {
        const avg = Math.round(sessionScores.reduce((a, b) => a + b, 0) / sessionScores.length);
        avgScoreEl.textContent = avg + '%';
    }

    if (metrics) {
        blinkRateEl.textContent = (metrics.blinkRate || 0).toFixed(1);

        const fatigue = metrics.fatigueIndex || 0;
        if (fatigue < 30) {
            fatigueLevelEl.textContent = 'Low';
            fatigueLevelEl.style.color = '#3FB950';
        } else if (fatigue < 60) {
            fatigueLevelEl.textContent = 'Med';
            fatigueLevelEl.style.color = '#FFA657';
        } else {
            fatigueLevelEl.textContent = 'High';
            fatigueLevelEl.style.color = '#F85149';
        }
    }
}

function updateUINoFace() {
    scoreNumber.textContent = '-';
    scoreProgress.style.strokeDashoffset = 283;
    scoreProgress.classList.remove('good', 'warning', 'danger');
    scoreState.textContent = 'No Face';
    scoreState.className = 'score-state';
}

// Event Listeners
toggleCameraBtn.addEventListener('click', toggleCamera);
toggleSessionBtn.addEventListener('click', toggleSession);

settingsBtn.addEventListener('click', () => {
    alert('Settings coming soon!');
});

retryBtn.addEventListener('click', () => {
    errorOverlay.classList.add('hidden');
    init();
});

// Handle popup close
window.addEventListener('beforeunload', () => {
    // Session continues in background
});

// Initialize on load
document.addEventListener('DOMContentLoaded', init);
