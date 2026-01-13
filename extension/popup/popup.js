/**
 * CognEn Extension Popup Script
 * Detects if on Meet tab and uses appropriate mode
 */

// DOM Elements
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const statusPlaceholder = document.getElementById('statusPlaceholder');
const toggleSessionBtn = document.getElementById('toggleSessionBtn');
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
const settingsBtn = document.getElementById('settingsBtn');

// State
let isSessionActive = false;
let sessionStartTime = null;
let sessionTimerInterval = null;
let sessionScores = [];
let pollInterval = null;
let currentMode = null; // 'meet' or 'standalone'

// Initialize
async function init() {
    console.log('[Popup] Init');

    // Check if we're on a Meet tab
    const modeCheck = await sendMessage({ action: 'CHECK_MODE' });
    console.log('[Popup] Mode check:', modeCheck);

    if (modeCheck && modeCheck.isMeet) {
        currentMode = 'meet';
        statusPlaceholder.innerHTML = `
      <div style="text-align: center;">
        <div style="font-size: 32px; margin-bottom: 8px;">🎥</div>
        <p style="color: #00FFFF; font-weight: 600;">Google Meet Detected</p>
        <p style="font-size: 11px; color: #8B949E;">Click Start to analyze your video feed</p>
        <p style="font-size: 10px; color: #6E7681; margin-top: 4px;">No extra camera access needed!</p>
      </div>
    `;
    } else {
        currentMode = 'standalone';
        statusPlaceholder.innerHTML = `
      <div style="text-align: center;">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
          <rect x="2" y="5" width="20" height="14" rx="2" stroke="currentColor" stroke-width="2"/>
          <circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="2"/>
        </svg>
        <p>Standalone Mode</p>
        <p style="font-size: 10px; color: #6E7681;">Will use your webcam directly</p>
      </div>
    `;
    }

    // Check existing session
    const state = await chrome.storage.local.get(['sessionActive', 'sessionStartTime', 'lastDetection']);

    if (state.sessionActive) {
        isSessionActive = true;
        sessionStartTime = state.sessionStartTime || Date.now();
        updateSessionUI(true);
        sessionTimerInterval = setInterval(updateSessionTimer, 1000);
        startPolling();

        if (state.lastDetection?.hasFace) {
            updateUI(state.lastDetection.prediction, state.lastDetection.metrics);
        }
    }
}

// Helper to send messages
function sendMessage(message) {
    return new Promise((resolve) => {
        chrome.runtime.sendMessage(message, (response) => {
            if (chrome.runtime.lastError) {
                console.error('[Popup] Message error:', chrome.runtime.lastError);
                resolve(null);
            } else {
                resolve(response);
            }
        });
    });
}

// Poll for detection results
function startPolling() {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        const data = await chrome.storage.local.get(['lastDetection']);

        if (data.lastDetection) {
            const det = data.lastDetection;
            const age = Date.now() - det.timestamp;

            if (age < 3000) {
                if (det.hasFace && det.prediction) {
                    updateUI(det.prediction, det.metrics);
                    sessionScores.push(det.prediction.rawScore);
                } else {
                    updateUINoFace();
                }
            }
        }
    }, 300);
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// Session Controls
async function startSession() {
    console.log('[Popup] Starting session, mode:', currentMode);

    loadingOverlay.classList.remove('hidden');
    loadingText.textContent = currentMode === 'meet' ? 'Connecting to Meet...' : 'Starting camera...';
    toggleSessionBtn.disabled = true;

    try {
        const response = await sendMessage({ action: 'START_SESSION' });
        console.log('[Popup] START_SESSION response:', response);

        if (!response || !response.success) {
            throw new Error(response?.error || 'Failed to start');
        }

        if (response.mode === 'standalone') {
            // Wait for camera in standalone mode
            loadingText.textContent = 'Loading ML models...';

            let attempts = 0;
            while (attempts < 30) {
                await new Promise(r => setTimeout(r, 500));
                attempts++;

                const state = await chrome.storage.local.get(['cameraRunning', 'cameraError']);
                if (state.cameraRunning) break;
                if (state.cameraError) throw new Error(state.cameraError);
            }
        } else {
            // Meet mode - give content script time to initialize
            loadingText.textContent = 'Analyzing Meet video...';
            await new Promise(r => setTimeout(r, 2000));
        }

        // Success
        isSessionActive = true;
        sessionStartTime = Date.now();
        sessionScores = [];

        await chrome.storage.local.set({ sessionActive: true, sessionStartTime });

        loadingOverlay.classList.add('hidden');
        updateSessionUI(true);
        sessionTimerInterval = setInterval(updateSessionTimer, 1000);
        startPolling();

    } catch (error) {
        console.error('[Popup] Start error:', error);
        loadingOverlay.classList.add('hidden');
        toggleSessionBtn.disabled = false;
        alert('Failed to start: ' + error.message);
    }
}

async function stopSession() {
    console.log('[Popup] Stopping session');

    isSessionActive = false;
    stopPolling();

    if (sessionTimerInterval) {
        clearInterval(sessionTimerInterval);
        sessionTimerInterval = null;
    }

    const avgScore = sessionScores.length > 0
        ? Math.round(sessionScores.reduce((a, b) => a + b, 0) / sessionScores.length)
        : 0;

    await sendMessage({ action: 'STOP_SESSION' });

    await chrome.storage.local.set({
        sessionActive: false,
        lastSession: {
            startTime: sessionStartTime,
            endTime: Date.now(),
            avgScore,
            dataPoints: sessionScores.length
        }
    });

    updateSessionUI(false);

    // Show summary
    const duration = Math.floor((Date.now() - sessionStartTime) / 1000);
    const mins = Math.floor(duration / 60);
    const secs = duration % 60;

    statusPlaceholder.innerHTML = `
    <div style="text-align: center;">
      <div style="font-size: 32px; margin-bottom: 8px;">✅</div>
      <p style="color: #3FB950; font-weight: 600;">Session Complete!</p>
      <p style="font-size: 12px; color: #8B949E;">Duration: ${mins}m ${secs}s</p>
      <p style="font-size: 12px; color: #8B949E;">Average: ${avgScore}%</p>
      <p style="font-size: 12px; color: #8B949E;">Data points: ${sessionScores.length}</p>
    </div>
  `;
}

function toggleSession() {
    if (isSessionActive) {
        stopSession();
    } else {
        startSession();
    }
}

function updateSessionUI(isActive) {
    if (isActive) {
        playIcon.style.display = 'none';
        stopIcon.style.display = 'block';
        sessionButtonText.textContent = 'Stop Session';
        toggleSessionBtn.classList.add('recording');
        toggleSessionBtn.disabled = false;
        statusDot.classList.add('active', 'recording');
        statusText.textContent = 'Recording';

        const modeIcon = currentMode === 'meet' ? '🎥' : '📷';
        statusPlaceholder.innerHTML = `
      <div style="text-align: center;">
        <div class="pulse-ring"></div>
        <p style="color: #3FB950; font-weight: 600;">${modeIcon} Tracking Active</p>
        <p style="font-size: 10px; color: #8B949E;">${currentMode === 'meet' ? 'Analyzing Meet video' : 'Using webcam'}</p>
      </div>
    `;
    } else {
        playIcon.style.display = 'block';
        stopIcon.style.display = 'none';
        sessionButtonText.textContent = 'Start Session';
        toggleSessionBtn.classList.remove('recording');
        toggleSessionBtn.disabled = false;
        statusDot.classList.remove('active', 'recording');
        statusText.textContent = 'Ready';
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
    if (score >= 70) scoreProgress.classList.add('good');
    else if (score >= 40) scoreProgress.classList.add('warning');
    else scoreProgress.classList.add('danger');

    scoreState.textContent = prediction.attentionState || 'Unknown';
    scoreState.className = 'score-state ' + (prediction.attentionState || '').toLowerCase();

    if (sessionScores.length > 0) {
        const recent = sessionScores.slice(-20);
        const avg = Math.round(recent.reduce((a, b) => a + b, 0) / recent.length);
        avgScoreEl.textContent = avg + '%';
    }

    if (metrics) {
        if (metrics.blinkRate !== undefined) {
            blinkRateEl.textContent = metrics.blinkRate.toFixed(1);
        }

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

// Settings
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
            <span>⚠️ Alert Threshold</span>
            <input type="number" id="lowAttentionThreshold" value="${settings.lowAttentionThreshold}" min="20" max="60">
          </label>
          <label class="setting-item">
            <span>☕ Break Reminder</span>
            <input type="number" id="breakReminderMinutes" value="${settings.breakReminderMinutes}" min="15" max="120">
          </label>
        </div>
        <div class="settings-footer">
          <button class="btn btn-primary" id="saveSettings">Save</button>
        </div>
      </div>
    `;

        document.body.appendChild(modal);

        document.getElementById('closeSettings').addEventListener('click', () => modal.remove());
        modal.addEventListener('click', (e) => { if (e.target === modal) modal.remove(); });

        document.getElementById('saveSettings').addEventListener('click', () => {
            chrome.storage.local.set({
                settings: {
                    notificationsEnabled: document.getElementById('notificationsEnabled').checked,
                    soundEnabled: document.getElementById('soundEnabled').checked,
                    lowAttentionThreshold: parseInt(document.getElementById('lowAttentionThreshold').value) || 40,
                    breakReminderMinutes: parseInt(document.getElementById('breakReminderMinutes').value) || 45
                }
            });
            modal.remove();
        });
    });
}

// Events
toggleSessionBtn.addEventListener('click', toggleSession);
settingsBtn.addEventListener('click', showSettings);

window.addEventListener('beforeunload', () => {
    stopPolling();
    if (sessionTimerInterval) clearInterval(sessionTimerInterval);
});

document.addEventListener('DOMContentLoaded', init);
