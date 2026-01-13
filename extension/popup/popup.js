/**
 * CognEn Extension Popup Script
 * Reads detection results from chrome.storage (written by offscreen document)
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
let lastFrameCount = 0;

// Hide loading overlay
loadingOverlay.classList.add('hidden');

// Initialize
async function init() {
    console.log('[Popup] Initializing...');

    // Check current state from storage
    const state = await chrome.storage.local.get([
        'cameraRunning',
        'sessionActive',
        'sessionStartTime',
        'lastDetection'
    ]);

    console.log('[Popup] Current state:', state);

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

    // Show last detection if available
    if (state.lastDetection && state.lastDetection.hasFace) {
        updateUI(state.lastDetection.prediction, state.lastDetection.metrics);
    }
}

// Poll for detection results from chrome.storage
function startPolling() {
    console.log('[Popup] Starting polling for detection results...');

    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        const data = await chrome.storage.local.get(['lastDetection', 'cameraRunning']);

        // Check if camera is still running
        if (!data.cameraRunning) {
            console.log('[Popup] Camera stopped, stopping polling');
            isCameraOn = false;
            updateCameraUI(false);
            stopPolling();
            return;
        }

        if (data.lastDetection) {
            const detection = data.lastDetection;
            const age = Date.now() - detection.timestamp;

            // Show frame count to verify detection is running
            if (detection.frameCount !== lastFrameCount) {
                lastFrameCount = detection.frameCount;

                if (age < 3000) { // Accept detections up to 3 seconds old
                    if (detection.hasFace && detection.prediction) {
                        updateUI(detection.prediction, detection.metrics);

                        if (isSessionActive) {
                            sessionScores.push(detection.prediction.rawScore);

                            // Update badge
                            chrome.runtime.sendMessage({
                                type: 'UPDATE_SCORE',
                                data: { score: detection.prediction.rawScore }
                            }).catch(() => { });
                        }
                    } else {
                        updateUINoFace();
                    }
                }
            }
        }
    }, 250); // Poll every 250ms
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// Camera Controls
async function startCamera() {
    console.log('[Popup] Starting camera...');

    cameraButtonText.textContent = 'Starting...';
    toggleCameraBtn.disabled = true;

    try {
        const response = await chrome.runtime.sendMessage({ type: 'START_CAMERA_REQUEST' });
        console.log('[Popup] START_CAMERA_REQUEST response:', response);

        if (response && response.success) {
            // Wait for camera to actually start (models need to load)
            cameraButtonText.textContent = 'Loading models...';

            // Poll for camera running state
            let attempts = 0;
            const checkCamera = setInterval(async () => {
                attempts++;
                const state = await chrome.storage.local.get(['cameraRunning', 'cameraError']);
                console.log('[Popup] Checking camera state, attempt', attempts, state);

                if (state.cameraRunning) {
                    clearInterval(checkCamera);
                    isCameraOn = true;
                    updateCameraUI(true);
                    startPolling();
                } else if (state.cameraError) {
                    clearInterval(checkCamera);
                    alert('Camera error: ' + state.cameraError);
                    cameraButtonText.textContent = 'Start Camera';
                    toggleCameraBtn.disabled = false;
                } else if (attempts > 30) { // 15 seconds timeout
                    clearInterval(checkCamera);
                    alert('Timeout waiting for camera to start');
                    cameraButtonText.textContent = 'Start Camera';
                    toggleCameraBtn.disabled = false;
                }
            }, 500);

        } else {
            throw new Error(response?.error || 'Failed to start camera');
        }
    } catch (error) {
        console.error('[Popup] Camera start error:', error);
        alert('Failed to start camera: ' + error.message);
        cameraButtonText.textContent = 'Start Camera';
        toggleCameraBtn.disabled = false;
    }
}

async function stopCamera() {
    console.log('[Popup] Stopping camera...');

    await chrome.runtime.sendMessage({ type: 'STOP_CAMERA_REQUEST' });

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
      <div class="camera-active-indicator">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color: #3FB950;">
          <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
          <path d="M8 12l3 3 5-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <p style="color: #3FB950; font-weight: 600;">Camera Active</p>
        <p style="font-size: 10px; color: #8B949E;">Analyzing in background</p>
        <p id="frameCounter" style="font-size: 10px; color: #6E7681; margin-top: 4px;">Frames: 0</p>
      </div>
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
        avgScoreEl.textContent = '-';
        blinkRateEl.textContent = '-';
        fatigueLevelEl.textContent = 'Low';
        fatigueLevelEl.style.color = '#3FB950';
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
async function startSession() {
    console.log('[Popup] Starting session...');

    isSessionActive = true;
    sessionStartTime = Date.now();
    sessionScores = [];

    // Reset ML in offscreen
    chrome.runtime.sendMessage({ type: 'RESET_ML' }).catch(() => { });

    updateSessionUI(true);
    sessionTimerInterval = setInterval(updateSessionTimer, 1000);

    chrome.runtime.sendMessage({ type: 'SESSION_START' }).catch(() => { });
    await chrome.storage.local.set({ sessionActive: true, sessionStartTime });
}

async function stopSession() {
    console.log('[Popup] Stopping session...');

    isSessionActive = false;
    updateSessionUI(false);

    if (sessionTimerInterval) {
        clearInterval(sessionTimerInterval);
        sessionTimerInterval = null;
    }

    const avgScore = sessionScores.length > 0
        ? Math.round(sessionScores.reduce((a, b) => a + b, 0) / sessionScores.length)
        : 0;

    chrome.runtime.sendMessage({ type: 'SESSION_END', data: { avgScore } }).catch(() => { });
    await chrome.storage.local.set({
        sessionActive: false,
        lastSession: {
            startTime: sessionStartTime,
            endTime: Date.now(),
            scoreCount: sessionScores.length,
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

    // Update ring progress
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

    // Update state label
    scoreState.textContent = prediction.attentionState || 'Unknown';
    scoreState.className = 'score-state ' + (prediction.attentionState || '').toLowerCase();

    // Update average score
    if (sessionScores.length > 0) {
        const avg = Math.round(sessionScores.reduce((a, b) => a + b, 0) / sessionScores.length);
        avgScoreEl.textContent = avg + '%';
    }

    // Update metrics display
    if (metrics) {
        // Blink rate
        const blinkRate = metrics.blinkRate;
        if (blinkRate !== undefined && blinkRate !== null) {
            blinkRateEl.textContent = blinkRate.toFixed(1);
        }

        // Fatigue level
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

        // Update frame counter if present
        const frameCounter = document.getElementById('frameCounter');
        if (frameCounter) {
            frameCounter.textContent = `Frames: ${lastFrameCount}`;
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
          <h3>⚙️ Settings</h3>
          <button class="settings-close" id="closeSettings">&times;</button>
        </div>
        <div class="settings-body">
          <label class="setting-item">
            <span>🔔 Notifications</span>
            <input type="checkbox" id="notificationsEnabled" ${settings.notificationsEnabled ? 'checked' : ''}>
          </label>
          <label class="setting-item">
            <span>🔊 Sound Alerts</span>
            <input type="checkbox" id="soundEnabled" ${settings.soundEnabled ? 'checked' : ''}>
          </label>
          <label class="setting-item">
            <span>⚠️ Low Attention Alert</span>
            <input type="number" id="lowAttentionThreshold" value="${settings.lowAttentionThreshold}" min="20" max="60">
          </label>
          <label class="setting-item">
            <span>☕ Break Reminder (min)</span>
            <input type="number" id="breakReminderMinutes" value="${settings.breakReminderMinutes}" min="15" max="120">
          </label>
        </div>
        <div class="settings-footer">
          <button class="btn btn-primary" id="saveSettings">Save Settings</button>
        </div>
      </div>
    `;

        document.body.appendChild(modal);

        document.getElementById('closeSettings').addEventListener('click', () => modal.remove());
        modal.addEventListener('click', (e) => { if (e.target === modal) modal.remove(); });

        document.getElementById('saveSettings').addEventListener('click', () => {
            const newSettings = {
                notificationsEnabled: document.getElementById('notificationsEnabled').checked,
                soundEnabled: document.getElementById('soundEnabled').checked,
                lowAttentionThreshold: parseInt(document.getElementById('lowAttentionThreshold').value) || 40,
                breakReminderMinutes: parseInt(document.getElementById('breakReminderMinutes').value) || 45
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
    if (sessionTimerInterval) {
        clearInterval(sessionTimerInterval);
    }
});

// Initialize on load
document.addEventListener('DOMContentLoaded', init);
