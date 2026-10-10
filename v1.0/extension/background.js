/**
 * CognEn Extension - Background Service Worker (MV3)
 *
 * Responsibilities:
 *  - Start/stop sessions in standalone (offscreen webcam) or Meet/Zoom/Teams
 *    (in-page) mode.
 *  - Receive one detection per second from either analyser:
 *      standalone: offscreen.js  -> {action: 'DETECTION'}
 *      meet:       bridge.js     -> {action: 'MEET_DETECTION'} (HMAC-verified
 *                  relay of the MAIN-world content.js, see bridge.js)
 *    and own the badge, session statistics, and alerts.
 *
 * MV3 service workers are killed when idle, so no state lives only in
 * memory: the session record is persisted in chrome.storage.session (falls
 * back to chrome.storage.local on browsers without it) and re-read on every
 * event. There are no setInterval timers to lose; the break reminder uses
 * chrome.alarms, which survive worker restarts, and incoming detection
 * messages wake the worker.
 */

// Mirrors THRESHOLDS in lib/ml-core.js (canonical: src/lib/scoring/constants.ts).
// Service workers of type "module" cannot importScripts the IIFE, so the
// values are repeated here and checked by extension-equivalence.test.ts.
const SCORE_BANDS = Object.freeze({ ATTENTIVE_MIN: 70, DISTRACTED_MIN: 40 });

const DEFAULT_SETTINGS = Object.freeze({
    notificationsEnabled: true,
    soundEnabled: false,
    lowAttentionThreshold: 40,
    breakReminderMinutes: 45,
});

const BADGE_COLORS = {
    high: '#3FB950',
    medium: '#FFA657',
    low: '#F85149',
    inactive: '#6E7681',
};

const STATE_KEY = 'cognenSession';
const BREAK_ALARM = 'cognen-break-reminder';
const LOW_ATTENTION_SUSTAIN_MS = 30000; // score must stay below threshold this long
const LOW_ATTENTION_COOLDOWN_MS = 5 * 60000; // at most one low-attention alert per 5 min
const MEET_HOSTS = ['meet.google.com', 'zoom.us', 'teams.microsoft.com'];

const stateArea = (chrome.storage && chrome.storage.session) || chrome.storage.local;

// ==================== PERSISTED SESSION STATE ====================

function emptyState() {
    return {
        active: false,
        mode: null,
        tabId: null,
        sessionId: null,
        startTime: null,
        seq: 0,
        stats: { count: 0, sum: 0, noFace: 0, states: { Attentive: 0, Distracted: 0, Multitasking: 0, Idle: 0 } },
        lowSince: null,
        lastLowAlertAt: 0,
    };
}

async function loadState() {
    const data = await stateArea.get(STATE_KEY);
    return Object.assign(emptyState(), data[STATE_KEY] || {});
}

// Serialise read-modify-write cycles so concurrent messages cannot clobber
// each other's updates.
let stateQueue = Promise.resolve();
function updateState(mutator) {
    const run = stateQueue.then(async () => {
        const state = await loadState();
        const result = await mutator(state);
        await stateArea.set({ [STATE_KEY]: state });
        return result;
    });
    stateQueue = run.catch(() => undefined);
    return run;
}

async function getSettings() {
    const { settings } = await chrome.storage.local.get('settings');
    const s = Object.assign({}, DEFAULT_SETTINGS, settings || {});
    s.lowAttentionThreshold = Math.max(0, Math.min(100, Number(s.lowAttentionThreshold) || DEFAULT_SETTINGS.lowAttentionThreshold));
    s.breakReminderMinutes = Math.max(1, Number(s.breakReminderMinutes) || DEFAULT_SETTINGS.breakReminderMinutes);
    return s;
}

function summarize(state) {
    const st = state.stats;
    const dist = {};
    for (const k of Object.keys(st.states)) dist[k] = st.count > 0 ? Math.round((st.states[k] / st.count) * 100) : 0;
    return {
        startTime: state.startTime,
        endTime: Date.now(),
        avgScore: st.count > 0 ? Math.round(st.sum / st.count) : 0,
        dataPoints: st.count,
        noFacePoints: st.noFace,
        stateDistribution: dist,
        mode: state.mode,
    };
}

// ==================== BADGE ====================

function setBadgeScore(score) {
    if (typeof score !== 'number' || !Number.isFinite(score)) {
        chrome.action.setBadgeText({ text: '--' });
        chrome.action.setBadgeBackgroundColor({ color: BADGE_COLORS.inactive });
        return;
    }
    chrome.action.setBadgeText({ text: String(score) });
    let color;
    if (score >= SCORE_BANDS.ATTENTIVE_MIN) color = BADGE_COLORS.high;
    else if (score >= SCORE_BANDS.DISTRACTED_MIN) color = BADGE_COLORS.medium;
    else color = BADGE_COLORS.low;
    chrome.action.setBadgeBackgroundColor({ color });
}

function clearBadge() {
    chrome.action.setBadgeText({ text: '' });
}

// ==================== NOTIFICATIONS ====================

function notify(id, title, message, settings) {
    try {
        chrome.notifications.create(id, {
            type: 'basic',
            iconUrl: chrome.runtime.getURL('icons/icon128.png'),
            title,
            message,
            priority: 1,
            silent: !settings.soundEnabled,
        });
    } catch (e) {
        console.warn('[BG] Notification failed:', e);
    }
}

async function checkLowAttention(state, detection) {
    const settings = await getSettings();
    const score = detection.hasFace && detection.prediction ? detection.prediction.rawScore : null;
    if (typeof score !== 'number' || score >= settings.lowAttentionThreshold) {
        state.lowSince = null;
        return;
    }
    const now = detection.timestamp;
    if (state.lowSince === null) state.lowSince = now;
    if (!settings.notificationsEnabled) return;
    if (now - state.lowSince >= LOW_ATTENTION_SUSTAIN_MS && now - state.lastLowAlertAt >= LOW_ATTENTION_COOLDOWN_MS) {
        state.lastLowAlertAt = now;
        notify('cognen-low-attention', 'CognEn: low attention',
            'Your attention score has been below ' + settings.lowAttentionThreshold + ' for over ' +
            Math.round(LOW_ATTENTION_SUSTAIN_MS / 1000) + ' seconds.', settings);
    }
}

async function scheduleBreakReminder() {
    const settings = await getSettings();
    await chrome.alarms.clear(BREAK_ALARM);
    chrome.alarms.create(BREAK_ALARM, {
        delayInMinutes: settings.breakReminderMinutes,
        periodInMinutes: settings.breakReminderMinutes,
    });
}

chrome.alarms.onAlarm.addListener(async (alarm) => {
    if (alarm.name !== BREAK_ALARM) return;
    const state = await loadState();
    if (!state.active) {
        chrome.alarms.clear(BREAK_ALARM);
        return;
    }
    const settings = await getSettings();
    if (!settings.notificationsEnabled) return;
    notify('cognen-break', 'CognEn: time for a break',
        'You have been in a session for ' + Math.round((Date.now() - state.startTime) / 60000) + ' minutes.', settings);
});

// ==================== OFFSCREEN DOCUMENT ====================

async function hasOffscreenDocument() {
    try {
        const contexts = await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] });
        return contexts.length > 0;
    } catch {
        return false;
    }
}

async function createOffscreenDocument() {
    if (await hasOffscreenDocument()) return true;
    try {
        await chrome.offscreen.createDocument({
            url: 'offscreen.html',
            reasons: ['USER_MEDIA'],
            justification: 'Camera access for face detection',
        });
        return true;
    } catch (error) {
        console.error('[BG] Failed to create offscreen:', error);
        return false;
    }
}

/** Send to the offscreen document, retrying while its script finishes loading. */
async function sendToOffscreen(message, attempts = 5) {
    for (let i = 0; i < attempts; i++) {
        try {
            return await chrome.runtime.sendMessage(message);
        } catch (e) {
            if (i === attempts - 1) throw e;
            await new Promise((r) => setTimeout(r, 300));
        }
    }
    return undefined;
}

async function closeOffscreenDocument() {
    if (await hasOffscreenDocument()) {
        try {
            await chrome.offscreen.closeDocument();
        } catch {
            // already closed
        }
    }
}

// ==================== MEET MODE ====================

function isMeetUrl(url) {
    try {
        const host = new URL(url).hostname;
        return MEET_HOSTS.some((h) => host === h || host.endsWith('.' + h));
    } catch {
        return false;
    }
}

async function checkIfMeetTab() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab) return { isMeet: false };
        return { isMeet: isMeetUrl(tab.url || ''), tabId: tab.id };
    } catch {
        return { isMeet: false };
    }
}

function randomHex(bytes) {
    const buf = new Uint8Array(bytes);
    crypto.getRandomValues(buf);
    return Array.from(buf, (b) => b.toString(16).padStart(2, '0')).join('');
}

async function execResult(details) {
    const results = await chrome.scripting.executeScript(details);
    return results && results[0] ? results[0].result : undefined;
}

/**
 * Inject (idempotently) and start the in-page analyser.
 *  1. bridge.js into the ISOLATED world, armed with a fresh per-session key.
 *  2. face-api / ml-core / content.js into the MAIN world, each only if not
 *     already present (all three are re-injection safe anyway).
 *  3. content.start(key) via executeScript args - the key never touches the
 *     DOM, storage or postMessage.
 */
async function startMeetTracking(tabId, sessionId) {
    const keyHex = randomHex(32);

    await chrome.scripting.executeScript({ target: { tabId }, files: ['bridge.js'] });
    const armed = await execResult({
        target: { tabId },
        func: (k, id) => (window.__cognenBridge ? window.__cognenBridge.arm(k, id) : false),
        args: [keyHex, sessionId],
    });
    if (!armed) throw new Error('Bridge failed to initialise');

    const present = await execResult({
        target: { tabId },
        world: 'MAIN',
        func: () => ({
            faceapi: typeof window.faceapi !== 'undefined',
            core: !!(window.__cognen && window.__cognen.mlCore),
            content: !!(window.__cognen && window.__cognen.content),
        }),
    }) || {};

    if (!present.faceapi) {
        await chrome.scripting.executeScript({ target: { tabId }, files: ['lib/face-api.min.js'], world: 'MAIN' });
    }
    if (!present.core) {
        await chrome.scripting.executeScript({ target: { tabId }, files: ['lib/ml-core.js'], world: 'MAIN' });
    }
    if (!present.content) {
        await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'], world: 'MAIN' });
    }

    const started = await execResult({
        target: { tabId },
        world: 'MAIN',
        func: (k) => (window.__cognen && window.__cognen.content ? window.__cognen.content.start(k) : false),
        args: [keyHex],
    });
    if (!started) throw new Error('In-page analyser failed to start (models or video unavailable)');
}

async function stopMeetTracking(tabId) {
    try {
        await chrome.scripting.executeScript({
            target: { tabId },
            world: 'MAIN',
            func: () => {
                if (window.__cognen && window.__cognen.content) window.__cognen.content.stop();
            },
        });
        await chrome.scripting.executeScript({
            target: { tabId },
            func: () => {
                if (window.__cognenBridge) window.__cognenBridge.disarm();
            },
        });
    } catch {
        // Tab closed or navigated away: nothing left to stop.
    }
}

// ==================== SESSION LIFECYCLE ====================

async function startSession() {
    const current = await loadState();
    if (current.active) await endSession();

    const meetInfo = await checkIfMeetTab();
    const sessionId = crypto.randomUUID();
    const startTime = Date.now();

    if (meetInfo.isMeet && meetInfo.tabId !== undefined) {
        await startMeetTracking(meetInfo.tabId, sessionId);
    } else {
        const created = await createOffscreenDocument();
        if (!created) throw new Error('Could not create offscreen document');
        await sendToOffscreen({ type: 'START_CAMERA' });
    }

    const mode = meetInfo.isMeet ? 'meet' : 'standalone';
    await updateState((s) => {
        Object.assign(s, emptyState(), {
            active: true,
            mode,
            tabId: meetInfo.isMeet ? meetInfo.tabId : null,
            sessionId,
            startTime,
        });
    });
    // UI mirror for the popup (chrome.storage.session is not readable there
    // when the fallback is in use, so the popup reads chrome.storage.local).
    const uiMirror = { sessionActive: true, sessionMode: mode, sessionStartTime: startTime, lastDetection: null };
    if (mode === 'meet') uiMirror.cameraRunning = true; // standalone: offscreen.js sets it
    await chrome.storage.local.set(uiMirror);
    await scheduleBreakReminder();

    chrome.action.setBadgeText({ text: '...' });
    chrome.action.setBadgeBackgroundColor({ color: BADGE_COLORS.medium });
    return { success: true, mode };
}

async function endSession() {
    const summary = await updateState(async (s) => {
        if (!s.active) return null;
        const sum = summarize(s);
        if (s.mode === 'meet' && s.tabId !== null) {
            await stopMeetTracking(s.tabId);
        } else {
            try {
                await chrome.runtime.sendMessage({ type: 'STOP_CAMERA' });
            } catch {
                // offscreen document already gone
            }
            await closeOffscreenDocument();
        }
        Object.assign(s, emptyState());
        return sum;
    });

    await chrome.alarms.clear(BREAK_ALARM);
    const update = { sessionActive: false, sessionMode: null, cameraRunning: false };
    if (summary) update.lastSession = summary;
    await chrome.storage.local.set(update);
    clearBadge();
    return summary;
}

async function handleDetection(detection) {
    await updateState(async (s) => {
        if (!s.active) return;
        s.seq += 1;
        const hasScore = detection.hasFace && detection.prediction && typeof detection.prediction.rawScore === 'number';
        if (hasScore) {
            s.stats.count += 1;
            s.stats.sum += detection.prediction.rawScore;
            const st = detection.prediction.attentionState;
            if (Object.prototype.hasOwnProperty.call(s.stats.states, st)) s.stats.states[st] += 1;
            setBadgeScore(detection.prediction.rawScore);
        } else {
            s.stats.noFace += 1;
            setBadgeScore(null);
        }
        await checkLowAttention(s, detection);
        await chrome.storage.local.set({
            lastDetection: Object.assign({}, detection, { seq: s.seq, sessionId: s.sessionId }),
        });
    });
}

// ==================== MESSAGE HANDLING ====================

function isFromExtensionPage(sender) {
    return sender.id === chrome.runtime.id && !sender.tab;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (!message || !message.action || sender.id !== chrome.runtime.id) return false;

    const respond = (p) => p.then(sendResponse, (e) => sendResponse({ success: false, error: e.message }));

    switch (message.action) {
        case 'CHECK_MODE':
            if (!isFromExtensionPage(sender)) return false;
            respond(checkIfMeetTab());
            return true;

        case 'START_SESSION':
            if (!isFromExtensionPage(sender)) return false;
            respond(startSession());
            return true;

        case 'STOP_SESSION':
            if (!isFromExtensionPage(sender)) return false;
            respond(endSession().then((summary) => ({ success: true, summary })));
            return true;

        case 'GET_SESSION':
            if (!isFromExtensionPage(sender)) return false;
            respond(loadState().then((s) => ({ active: s.active, mode: s.mode, summary: s.active ? summarize(s) : null })));
            return true;

        case 'DETECTION': {
            // Standalone mode: only accepted from our offscreen document.
            const offscreenUrl = chrome.runtime.getURL('offscreen.html');
            if (!isFromExtensionPage(sender) || !(sender.url || '').startsWith(offscreenUrl)) return false;
            loadState().then((s) => {
                if (s.active && s.mode === 'standalone') handleDetection(message.detection);
            });
            return false;
        }

        case 'MEET_DETECTION':
        case 'MEET_STOPPED':
            // Meet mode: only from bridge.js in the session tab, for the current session.
            if (!sender.tab) return false;
            loadState().then((s) => {
                if (!s.active || s.mode !== 'meet' || sender.tab.id !== s.tabId || message.sessionId !== s.sessionId) return;
                if (message.action === 'MEET_DETECTION') handleDetection(message.detection);
                else endSession();
            });
            return false;

        default:
            return false;
    }
});

// End a Meet session if its tab is closed, or reloads / navigates to a new
// document (the injected analyser is gone). Same-document (pushState)
// navigations keep the analyser, so after each load completes we probe for
// it instead of ending on every URL change.
chrome.tabs.onRemoved.addListener(async (tabId) => {
    const s = await loadState();
    if (s.active && s.mode === 'meet' && s.tabId === tabId) endSession();
});

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo) => {
    if (changeInfo.status !== 'complete') return;
    const s = await loadState();
    if (!s.active || s.mode !== 'meet' || s.tabId !== tabId) return;
    let alive = false;
    try {
        alive = await execResult({
            target: { tabId },
            world: 'MAIN',
            func: () => !!(window.__cognen && window.__cognen.content && window.__cognen.content.isTracking()),
        });
    } catch {
        alive = false;
    }
    if (!alive) endSession();
});

// ==================== INITIALIZATION / RESTORE ====================

chrome.runtime.onInstalled.addListener(async () => {
    clearBadge();
    const { settings } = await chrome.storage.local.get('settings');
    await chrome.storage.local.set({
        sessionActive: false,
        cameraRunning: false,
        sessionMode: null,
        settings: Object.assign({}, DEFAULT_SETTINGS, settings || {}),
    });
    await stateArea.set({ [STATE_KEY]: emptyState() });
});

// Browser restart: offscreen documents and injected page scripts are gone.
chrome.runtime.onStartup.addListener(async () => {
    await stateArea.set({ [STATE_KEY]: emptyState() });
    await chrome.storage.local.set({ sessionActive: false, cameraRunning: false, sessionMode: null });
    await chrome.alarms.clear(BREAK_ALARM);
    clearBadge();
});

// Service-worker (re)start while the browser keeps running: reconcile the
// persisted session with reality instead of silently forgetting it.
(async function restoreOnWake() {
    try {
        const s = await loadState();
        if (!s.active) return;
        if (s.mode === 'standalone' && !(await hasOffscreenDocument())) {
            await endSession();
            return;
        }
        if (s.mode === 'meet' && s.tabId !== null) {
            try {
                await chrome.tabs.get(s.tabId);
            } catch {
                await endSession();
                return;
            }
        }
        const alarm = await chrome.alarms.get(BREAK_ALARM);
        if (!alarm) await scheduleBreakReminder();
    } catch (e) {
        console.warn('[BG] Restore failed:', e);
    }
})();
