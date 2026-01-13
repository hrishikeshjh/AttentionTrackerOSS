/**
 * CognEn Extension Popup Script v1.1
 * Now with live chart and improved face detection status
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
const chartContainer = document.getElementById('chartContainer');
const liveChart = document.getElementById('liveChart');

// State
let isSessionActive = false;
let sessionStartTime = null;
let sessionTimerInterval = null;
let sessionScores = [];
let pollInterval = null;
let currentMode = null;
let chartData = [];
let noFaceCount = 0;
const MAX_CHART_POINTS = 60;

// Chart rendering
function drawChart() {
    if (!liveChart || chartData.length < 2) return;

    const ctx = liveChart.getContext('2d');
    const width = liveChart.width;
    const height = liveChart.height;

    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = 'rgba(22, 27, 34, 0.8)';
    ctx.fillRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = 'rgba(139, 148, 158, 0.2)';
    ctx.lineWidth = 1;
    for (let y = 0; y <= 100; y += 25) {
        const yPos = height - (y / 100) * height;
        ctx.beginPath();
        ctx.moveTo(0, yPos);
        ctx.lineTo(width, yPos);
        ctx.stroke();
    }

    // Data line
    ctx.beginPath();
    ctx.strokeStyle = '#00FFFF';
    ctx.lineWidth = 2;

    const step = width / (MAX_CHART_POINTS - 1);
    chartData.forEach((score, i) => {
        const x = i * step;
        const y = height - (score / 100) * height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Fill under line
    ctx.lineTo((chartData.length - 1) * step, height);
    ctx.lineTo(0, height);
    ctx.closePath();
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, 'rgba(0, 255, 255, 0.3)');
    gradient.addColorStop(1, 'rgba(0, 255, 255, 0)');
    ctx.fillStyle = gradient;
    ctx.fill();

    // Current score marker
    if (chartData.length > 0) {
        const lastX = (chartData.length - 1) * step;
        const lastY = height - (chartData[chartData.length - 1] / 100) * height;
        ctx.beginPath();
        ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#00FF99';
        ctx.fill();
    }
}

// Initialize
async function init() {
    console.log('[Popup] Init v1.1');

    const modeCheck = await sendMessage({ action: 'CHECK_MODE' });

    if (modeCheck && modeCheck.isMeet) {
        currentMode = 'meet';
        updatePlaceholder('meet-ready');
    } else {
        currentMode = 'standalone';
        updatePlaceholder('standalone-ready');
    }

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

function updatePlaceholder(mode) {
    const el = statusPlaceholder;

    // Clear existing content
    while (el.firstChild) el.removeChild(el.firstChild);

    const wrapper = document.createElement('div');
    wrapper.style.textAlign = 'center';

    if (mode === 'meet-ready') {
        const icon = document.createElement('div');
        icon.textContent = '🎥';
        icon.style.fontSize = '32px';
        icon.style.marginBottom = '8px';

        const title = document.createElement('p');
        title.textContent = 'Google Meet Detected';
        title.style.color = '#00FFFF';
        title.style.fontWeight = '600';

        const sub = document.createElement('p');
        sub.textContent = 'Click Start to analyze your video';
        sub.style.fontSize = '11px';
        sub.style.color = '#8B949E';

        wrapper.appendChild(icon);
        wrapper.appendChild(title);
        wrapper.appendChild(sub);
    } else if (mode === 'standalone-ready') {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '48');
        svg.setAttribute('height', '48');
        svg.setAttribute('viewBox', '0 0 24 24');
        svg.setAttribute('fill', 'none');
        svg.innerHTML = '<rect x="2" y="5" width="20" height="14" rx="2" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="2"/>';

        const title = document.createElement('p');
        title.textContent = 'Standalone Mode';

        const sub = document.createElement('p');
        sub.textContent = 'Will use your webcam directly';
        sub.style.fontSize = '10px';
        sub.style.color = '#6E7681';

        wrapper.appendChild(svg);
        wrapper.appendChild(title);
        wrapper.appendChild(sub);
    } else if (mode === 'tracking') {
        const pulse = document.createElement('div');
        pulse.className = 'pulse-ring';

        const icon = currentMode === 'meet' ? '🎥' : '📷';
        const title = document.createElement('p');
        title.textContent = icon + ' Tracking Active';
        title.style.color = '#3FB950';
        title.style.fontWeight = '600';

        const sub = document.createElement('p');
        sub.textContent = currentMode === 'meet' ? 'Analyzing Meet video' : 'Using webcam';
        sub.style.fontSize = '10px';
        sub.style.color = '#8B949E';

        wrapper.appendChild(pulse);
        wrapper.appendChild(title);
        wrapper.appendChild(sub);
    } else if (mode === 'no-face') {
        const icon = document.createElement('div');
        icon.textContent = '👤';
        icon.style.fontSize = '32px';
        icon.style.marginBottom = '8px';
        icon.style.opacity = '0.5';

        const title = document.createElement('p');
        title.textContent = 'Face Not Detected';
        title.style.color = '#F85149';
        title.style.fontWeight = '600';

        const sub = document.createElement('p');
        sub.textContent = 'Please face the camera';
        sub.style.fontSize = '11px';
        sub.style.color = '#8B949E';

        wrapper.appendChild(icon);
        wrapper.appendChild(title);
        wrapper.appendChild(sub);
    }

    el.appendChild(wrapper);
}

function sendMessage(message) {
    return new Promise((resolve) => {
        chrome.runtime.sendMessage(message, (response) => {
            if (chrome.runtime.lastError) {
                resolve(null);
            } else {
                resolve(response);
            }
        });
    });
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        const data = await chrome.storage.local.get(['lastDetection']);

        if (data.lastDetection) {
            const det = data.lastDetection;
            const age = Date.now() - det.timestamp;

            if (age < 3000) {
                if (det.hasFace && det.prediction) {
                    noFaceCount = 0;
                    updateUI(det.prediction, det.metrics);

                    // Add to chart data
                    chartData.push(det.prediction.rawScore);
                    if (chartData.length > MAX_CHART_POINTS) chartData.shift();
                    drawChart();

                    sessionScores.push(det.prediction.rawScore);
                } else {
                    noFaceCount++;
                    if (noFaceCount >= 5) {
                        updateUINoFace();
                        updatePlaceholder('no-face');
                    }
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

async function startSession() {
    loadingOverlay.classList.remove('hidden');
    loadingText.textContent = currentMode === 'meet' ? 'Connecting to Meet...' : 'Starting camera...';
    toggleSessionBtn.disabled = true;

    try {
        const response = await sendMessage({ action: 'START_SESSION' });

        if (!response || !response.success) {
            throw new Error(response?.error || 'Failed to start');
        }

        if (response.mode === 'standalone') {
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
            loadingText.textContent = 'Analyzing Meet video...';
            await new Promise(r => setTimeout(r, 2000));
        }

        isSessionActive = true;
        sessionStartTime = Date.now();
        sessionScores = [];
        chartData = [];
        noFaceCount = 0;

        await chrome.storage.local.set({ sessionActive: true, sessionStartTime });

        loadingOverlay.classList.add('hidden');
        chartContainer.style.display = 'block';
        updateSessionUI(true);
        sessionTimerInterval = setInterval(updateSessionTimer, 1000);
        startPolling();

    } catch (error) {
        loadingOverlay.classList.add('hidden');
        toggleSessionBtn.disabled = false;
        alert('Failed to start: ' + error.message);
    }
}

async function stopSession() {
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
    chartContainer.style.display = 'none';

    // Show summary
    const duration = Math.floor((Date.now() - sessionStartTime) / 1000);
    const mins = Math.floor(duration / 60);
    const secs = duration % 60;

    const el = statusPlaceholder;
    while (el.firstChild) el.removeChild(el.firstChild);

    const wrapper = document.createElement('div');
    wrapper.style.textAlign = 'center';

    const icon = document.createElement('div');
    icon.textContent = '✅';
    icon.style.fontSize = '32px';
    icon.style.marginBottom = '8px';

    const title = document.createElement('p');
    title.textContent = 'Session Complete!';
    title.style.color = '#3FB950';
    title.style.fontWeight = '600';

    const stats = document.createElement('div');
    stats.style.fontSize = '12px';
    stats.style.color = '#8B949E';
    stats.innerHTML = '';

    const line1 = document.createElement('p');
    line1.textContent = 'Duration: ' + mins + 'm ' + secs + 's';
    const line2 = document.createElement('p');
    line2.textContent = 'Average: ' + avgScore + '%';
    const line3 = document.createElement('p');
    line3.textContent = 'Data points: ' + sessionScores.length;

    wrapper.appendChild(icon);
    wrapper.appendChild(title);
    wrapper.appendChild(line1);
    wrapper.appendChild(line2);
    wrapper.appendChild(line3);

    el.appendChild(wrapper);
}

function toggleSession() {
    if (isSessionActive) stopSession();
    else startSession();
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
        updatePlaceholder('tracking');
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
    sessionTimeEl.textContent = minutes + ':' + seconds;
}

function updateUI(prediction, metrics) {
    if (!prediction) return;

    // Restore tracking placeholder if we had no face before
    if (noFaceCount === 0 && statusPlaceholder.querySelector('p')?.textContent?.includes('Face Not')) {
        updatePlaceholder('tracking');
    }

    const score = prediction.rawScore;
    scoreNumber.textContent = score;

    const offset = 283 - (283 * score / 100);
    scoreProgress.style.strokeDashoffset = offset;

    scoreProgress.classList.remove('good', 'warning', 'danger');
    if (score >= 75) scoreProgress.classList.add('good');
    else if (score >= 45) scoreProgress.classList.add('warning');
    else scoreProgress.classList.add('danger');

    scoreState.textContent = prediction.attentionState || 'Unknown';
    scoreState.className = 'score-state ' + (prediction.attentionState || '').toLowerCase();

    if (sessionScores.length > 0) {
        const recent = sessionScores.slice(-20);
        const avg = Math.round(recent.reduce((a, b) => a + b, 0) / recent.length);
        avgScoreEl.textContent = avg + '%';
    }

    if (metrics) {
        // Blink rate - show even if 0
        const br = metrics.blinkRate;
        if (br !== undefined && br !== null) {
            blinkRateEl.textContent = br.toFixed(1);
        } else {
            blinkRateEl.textContent = '0.0';
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
    scoreState.className = 'score-state danger';
    scoreState.style.color = '#F85149';
}

function showSettings() {
    chrome.storage.local.get(['settings'], (result) => {
        const settings = result.settings || {
            notificationsEnabled: true,
            lowAttentionThreshold: 45,
            breakReminderMinutes: 45,
            soundEnabled: false
        };

        const modal = document.createElement('div');
        modal.className = 'settings-modal';

        const content = document.createElement('div');
        content.className = 'settings-content';

        // Header
        const header = document.createElement('div');
        header.className = 'settings-header';
        const h3 = document.createElement('h3');
        h3.textContent = '⚙️ Settings';
        const closeBtn = document.createElement('button');
        closeBtn.className = 'settings-close';
        closeBtn.textContent = '×';
        closeBtn.onclick = () => modal.remove();
        header.appendChild(h3);
        header.appendChild(closeBtn);

        // Body
        const body = document.createElement('div');
        body.className = 'settings-body';

        // Notification toggle
        const notifItem = createSettingItem('🔔 Notifications', 'checkbox', 'notificationsEnabled', settings.notificationsEnabled);
        body.appendChild(notifItem);

        const soundItem = createSettingItem('🔊 Sound Alerts', 'checkbox', 'soundEnabled', settings.soundEnabled);
        body.appendChild(soundItem);

        const threshItem = createSettingItem('⚠️ Alert Threshold', 'number', 'lowAttentionThreshold', settings.lowAttentionThreshold);
        body.appendChild(threshItem);

        const breakItem = createSettingItem('☕ Break Reminder (min)', 'number', 'breakReminderMinutes', settings.breakReminderMinutes);
        body.appendChild(breakItem);

        // Footer
        const footer = document.createElement('div');
        footer.className = 'settings-footer';
        const saveBtn = document.createElement('button');
        saveBtn.className = 'btn btn-primary';
        saveBtn.textContent = 'Save';
        saveBtn.onclick = () => {
            chrome.storage.local.set({
                settings: {
                    notificationsEnabled: document.getElementById('notificationsEnabled').checked,
                    soundEnabled: document.getElementById('soundEnabled').checked,
                    lowAttentionThreshold: parseInt(document.getElementById('lowAttentionThreshold').value) || 45,
                    breakReminderMinutes: parseInt(document.getElementById('breakReminderMinutes').value) || 45
                }
            });
            modal.remove();
        };
        footer.appendChild(saveBtn);

        content.appendChild(header);
        content.appendChild(body);
        content.appendChild(footer);
        modal.appendChild(content);

        modal.onclick = (e) => { if (e.target === modal) modal.remove(); };

        document.body.appendChild(modal);
    });
}

function createSettingItem(label, type, id, value) {
    const item = document.createElement('label');
    item.className = 'setting-item';

    const span = document.createElement('span');
    span.textContent = label;

    const input = document.createElement('input');
    input.type = type;
    input.id = id;

    if (type === 'checkbox') {
        input.checked = value;
    } else {
        input.value = value;
        input.min = type === 'number' ? '20' : undefined;
        input.max = type === 'number' ? '120' : undefined;
    }

    item.appendChild(span);
    item.appendChild(input);
    return item;
}

// Events
toggleSessionBtn.addEventListener('click', toggleSession);
settingsBtn.addEventListener('click', showSettings);

window.addEventListener('beforeunload', () => {
    stopPolling();
    if (sessionTimerInterval) clearInterval(sessionTimerInterval);
});

document.addEventListener('DOMContentLoaded', init);
