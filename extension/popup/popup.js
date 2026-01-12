/**
 * CognEn Extension Popup Script
 * Uses chrome.storage to read detection results from offscreen document
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
let pollInterval = null;

// Hide loading overlay
loadingOverlay.classList.add('hidden');

// Initialize
async function init() {
    // Check current state from storage
    const state = await chrome.storage.local.get(['cameraRunning', 'sessionActive', 'sessionStartTime']);

    if (state.cameraRunning) {
        isCameraOn = true;
        updateCameraUI(true);
        startPolling();
    }

    if (state.sessionActive) {
        isSessionActive = true;
        sessionStartTime = state.sessionStartTime || Date.now();
        updateSessionUI(true);
        sessionTimerInterval = setInterval(updateSessionTimer, 1000);
    }
}

// Poll for detection results
function startPolling() {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        const data = await chrome.storage.local.get(['lastDetection']);

        if (data.lastDetection) {
            const detection = data.lastDetection;
            const age = Date.now() - detection.timestamp;

            // Only use recent detections (less than 2 seconds old)
            if (age < 2000) {
                if (detection.hasFace && detection.prediction) {
                    updateUI(detection.prediction, detection.metrics);

                    if (isSessionActive) {
                        sessionScores.push(detection.prediction.rawScore);

                        // Update badge
                        chrome.runtime.sendMessage({
                            type: 'UPDATE_SCORE',
                            data: { score: detection.prediction.rawScore }
                        });
                    }
                } else {
                    updateUINoFace();
                }
            }
        }
    }, 200); // Poll every 200ms
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// Camera Controls
async function startCamera() {
    cameraButtonText.textContent = 'Starting...';
    toggleCameraBtn.disabled = true;

    try {
        // Create offscreen document and start camera
        const response = await chrome.runtime.sendMessage({ type: 'START_CAMERA_REQUEST' });

        if (response && response.success) {
            // Wait for camera to start
            await new Promise(resolve => setTimeout(resolve, 2000));

            isCameraOn = true;
            updateCameraUI(true);
            startPolling();
        } else {
            throw new Error(response?.error || 'Failed to start camera');
        }
    } catch (error) {
        console.error('Camera start error:', error);
        alert('Failed to start camera: ' + error.message);
        cameraButtonText.textContent = 'Start Camera';
        toggleCameraBtn.disabled = false;
    }
}

function stopCamera() {
    chrome.runtime.sendMessage({ type: 'STOP_CAMERA_REQUEST' });

    isCameraOn = false;
    updateCameraUI(false);
    stopPolling();

    if (isSessionActive) {
        stopSession();
    }
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

        // Reset display
        scoreNumber.textContent = '0';
        scoreProgress.style.strokeDashoffset = 283;
        scoreState.textContent = '-';
        scoreState.className = 'score-state';
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

    // Reset ML
    chrome.runtime.sendMessage({ type: 'RESET_ML' });

    updateSessionUI(true);

    sessionTimerInterval = setInterval(updateSessionTimer, 1000);

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

// Settings modal
function showSettings() {
    chrome.storage.local.get(['settings'], (result) => {
        const settings = result.settings || {
            notificationsEnabled: true,
            lowAttentionThreshold: 40,
            breakReminderMinutes: 45,
            soundEnabled: false
        };

        const modal = document.createElement('div');
        modal.className = 'settings-modal';
        modal.innerHTML = `
      <div class="settings-content">
        <div class="settings-header">
          <h3>Settings</h3>
          <button class="settings-close" id="closeSettings">&times;</button>
        </div>
        <div class="settings-body">
          <label class="setting-item">
            <span>Notifications</span>
            <input type="checkbox" id="notificationsEnabled" ${settings.notificationsEnabled ? 'checked' : ''}>
          </label>
          <label class="setting-item">
            <span>Sound Alerts</span>
            <input type="checkbox" id="soundEnabled" ${settings.soundEnabled ? 'checked' : ''}>
          </label>
          <label class="setting-item">
            <span>Low Attention Threshold</span>
            <input type="number" id="lowAttentionThreshold" value="${settings.lowAttentionThreshold}" min="20" max="60" style="width: 60px;">
          </label>
          <label class="setting-item">
            <span>Break Reminder (min)</span>
            <input type="number" id="breakReminderMinutes" value="${settings.breakReminderMinutes}" min="15" max="120" style="width: 60px;">
          </label>
        </div>
        <div class="settings-footer">
          <button class="btn btn-primary" id="saveSettings">Save</button>
        </div>
      </div>
    `;

        document.body.appendChild(modal);

        document.getElementById('closeSettings').addEventListener('click', () => modal.remove());
        document.getElementById('saveSettings').addEventListener('click', () => {
            const newSettings = {
                notificationsEnabled: document.getElementById('notificationsEnabled').checked,
                soundEnabled: document.getElementById('soundEnabled').checked,
                lowAttentionThreshold: parseInt(document.getElementById('lowAttentionThreshold').value),
                breakReminderMinutes: parseInt(document.getElementById('breakReminderMinutes').value)
            };
            chrome.storage.local.set({ settings: newSettings });
            modal.remove();
        });
    });
}

// Event Listeners
toggleCameraBtn.addEventListener('click', toggleCamera);
toggleSessionBtn.addEventListener('click', toggleSession);
settingsBtn.addEventListener('click', showSettings);

if (retryBtn) {
    retryBtn.addEventListener('click', () => {
        errorOverlay.classList.add('hidden');
        init();
    });
}

// Cleanup on popup close
window.addEventListener('beforeunload', () => {
    stopPolling();
});

// Initialize on load
document.addEventListener('DOMContentLoaded', init);
