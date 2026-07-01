/**
 * CognEn Extension - Background Service Worker
 * Injects ALL scripts from extension files to avoid Trusted Types errors
 */

const BADGE_COLORS = {
    high: '#3FB950',
    medium: '#FFA657',
    low: '#F85149',
    inactive: '#6E7681'
};

// ==================== OFFSCREEN DOCUMENT MANAGEMENT ====================

async function hasOffscreenDocument() {
    try {
        const contexts = await chrome.runtime.getContexts({
            contextTypes: ['OFFSCREEN_DOCUMENT']
        });
        return contexts.length > 0;
    } catch (e) {
        return false;
    }
}

async function createOffscreenDocument() {
    if (await hasOffscreenDocument()) return true;

    try {
        await chrome.offscreen.createDocument({
            url: 'offscreen.html',
            reasons: ['USER_MEDIA'],
            justification: 'Camera access for face detection'
        });
        return true;
    } catch (error) {
        console.error('[BG] Failed to create offscreen:', error);
        return false;
    }
}

async function closeOffscreenDocument() {
    if (await hasOffscreenDocument()) {
        try {
            await chrome.offscreen.closeDocument();
        } catch (e) { }
    }
}

// ==================== MEET DETECTION ====================

async function checkIfMeetTab() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab) return { isMeet: false };

        const url = tab.url || '';
        const isMeet = url.includes('meet.google.com') ||
            url.includes('zoom.us') ||
            url.includes('teams.microsoft.com');

        return { isMeet, tabId: tab.id, url };
    } catch (e) {
        return { isMeet: false };
    }
}

async function startMeetTracking(tabId) {
    try {
        console.log('[BG] Injecting scripts into tab', tabId);

        // Inject face-api.js from extension (bundled locally)
        await chrome.scripting.executeScript({
            target: { tabId },
            files: ['lib/face-api.min.js'],
            world: 'MAIN'
        });
        console.log('[BG] Injected face-api.min.js');

        // Wait for face-api to initialize
        await new Promise(r => setTimeout(r, 500));

        // Inject ml-core.js
        await chrome.scripting.executeScript({
            target: { tabId },
            files: ['lib/ml-core.js'],
            world: 'MAIN'
        });
        console.log('[BG] Injected ml-core.js');

        // Inject content script
        await chrome.scripting.executeScript({
            target: { tabId },
            files: ['content.js'],
            world: 'MAIN'
        });
        console.log('[BG] Injected content.js');

        // Set command to start
        await chrome.scripting.executeScript({
            target: { tabId },
            world: 'MAIN',
            func: () => {
                localStorage.setItem('cognen_meetCommand', JSON.stringify('START'));
                localStorage.setItem('cognen_commandTimestamp', JSON.stringify(Date.now()));
                console.log('[CognEn] START command set');
            }
        });

        // Start data sync
        startDataSync(tabId);

        return true;
    } catch (e) {
        console.error('[BG] Script injection failed:', e);
        return false;
    }
}

async function stopMeetTracking(tabId) {
    try {
        await chrome.scripting.executeScript({
            target: { tabId },
            world: 'MAIN',
            func: () => {
                localStorage.setItem('cognen_meetCommand', JSON.stringify('STOP'));
                localStorage.setItem('cognen_commandTimestamp', JSON.stringify(Date.now()));
            }
        });
        stopDataSync();
    } catch (e) {
        console.log('[BG] Could not send stop command');
    }
}

// ==================== DATA SYNC ====================

let dataSyncInterval = null;
let currentSyncTabId = null;

function startDataSync(tabId) {
    currentSyncTabId = tabId;
    if (dataSyncInterval) clearInterval(dataSyncInterval);

    dataSyncInterval = setInterval(async () => {
        if (!currentSyncTabId) return;

        try {
            const results = await chrome.scripting.executeScript({
                target: { tabId: currentSyncTabId },
                world: 'MAIN',
                func: () => {
                    try {
                        const detection = localStorage.getItem('cognen_lastDetection');
                        return detection ? JSON.parse(detection) : null;
                    } catch (e) {
                        return null;
                    }
                }
            });

            if (results && results[0] && results[0].result) {
                const detection = results[0].result;
                await chrome.storage.local.set({ lastDetection: detection });

                if (detection.prediction && detection.prediction.rawScore) {
                    const score = detection.prediction.rawScore;
                    chrome.action.setBadgeText({ text: String(score) });

                    let color;
                    if (score >= 70) color = BADGE_COLORS.high;
                    else if (score >= 40) color = BADGE_COLORS.medium;
                    else color = BADGE_COLORS.low;
                    chrome.action.setBadgeBackgroundColor({ color });
                }
            }
        } catch (e) {
            stopDataSync();
        }
    }, 500);
}

function stopDataSync() {
    if (dataSyncInterval) {
        clearInterval(dataSyncInterval);
        dataSyncInterval = null;
    }
    currentSyncTabId = null;
}

// Standalone mode (offscreen.js) writes lastDetection directly to
// chrome.storage.local, so the badge can update reactively via
// onChanged instead of a separate polling timer — this was previously a
// gap (no polling loop existed for standalone mode, so the badge never
// updated outside of meet-tab mode). Meet-mode badge updates still go
// through the 500ms poll in startDataSync above: background.js cannot
// listen to a third-party page's localStorage directly, so cross-page
// polling there is an intentional, unavoidable decoupling, not a bug.
chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName !== 'local' || !changes.lastDetection) return;
    if (currentSyncTabId) return; // meet-mode poll already owns badge updates

    const detection = changes.lastDetection.newValue;
    if (!detection || !detection.prediction) return;

    const score = detection.prediction.rawScore;
    chrome.action.setBadgeText({ text: String(score) });

    let color;
    if (score >= 70) color = BADGE_COLORS.high;
    else if (score >= 40) color = BADGE_COLORS.medium;
    else color = BADGE_COLORS.low;
    chrome.action.setBadgeBackgroundColor({ color });
});

// ==================== MESSAGE HANDLING ====================

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action) {
        handlePopupMessage(message, sendResponse);
        return true;
    }
    return false;
});

async function handlePopupMessage(message, sendResponse) {
    console.log('[BG] Action:', message.action);

    switch (message.action) {
        case 'CHECK_MODE':
            const meetCheck = await checkIfMeetTab();
            sendResponse(meetCheck);
            break;

        case 'START_SESSION':
            try {
                const meetInfo = await checkIfMeetTab();

                if (meetInfo.isMeet && meetInfo.tabId) {
                    console.log('[BG] Starting Meet mode on tab', meetInfo.tabId);
                    const success = await startMeetTracking(meetInfo.tabId);

                    if (!success) {
                        throw new Error('Failed to inject into Meet page');
                    }

                    await chrome.storage.local.set({
                        sessionMode: 'meet',
                        sessionTabId: meetInfo.tabId,
                        cameraRunning: true,
                        sessionActive: true
                    });

                    chrome.action.setBadgeText({ text: '...' });
                    chrome.action.setBadgeBackgroundColor({ color: BADGE_COLORS.medium });

                    sendResponse({ success: true, mode: 'meet' });
                } else {
                    console.log('[BG] Starting standalone mode');
                    const created = await createOffscreenDocument();

                    if (created) {
                        await new Promise(r => setTimeout(r, 1500));
                        try {
                            await chrome.runtime.sendMessage({ type: 'START_CAMERA' });
                        } catch (e) {
                            await new Promise(r => setTimeout(r, 1000));
                            try {
                                await chrome.runtime.sendMessage({ type: 'START_CAMERA' });
                            } catch (e2) { }
                        }
                    }

                    await chrome.storage.local.set({
                        sessionMode: 'standalone',
                        sessionActive: true
                    });

                    chrome.action.setBadgeText({ text: '...' });
                    chrome.action.setBadgeBackgroundColor({ color: BADGE_COLORS.medium });

                    sendResponse({ success: created, mode: 'standalone' });
                }

            } catch (error) {
                console.error('[BG] START_SESSION error:', error);
                sendResponse({ success: false, error: error.message });
            }
            break;

        case 'STOP_SESSION':
            try {
                const state = await chrome.storage.local.get(['sessionMode', 'sessionTabId']);

                if (state.sessionMode === 'meet' && state.sessionTabId) {
                    await stopMeetTracking(state.sessionTabId);
                } else {
                    try {
                        await chrome.runtime.sendMessage({ type: 'STOP_CAMERA' });
                    } catch (e) { }
                    await closeOffscreenDocument();
                }

                stopDataSync();

                await chrome.storage.local.set({
                    cameraRunning: false,
                    sessionActive: false,
                    sessionMode: null
                });

                chrome.action.setBadgeText({ text: '' });
                sendResponse({ success: true });
            } catch (error) {
                sendResponse({ success: false, error: error.message });
            }
            break;

        default:
            sendResponse({ success: false, error: 'Unknown action' });
    }
}

// ==================== INITIALIZATION ====================

chrome.runtime.onInstalled.addListener(() => {
    chrome.action.setBadgeText({ text: '' });
    chrome.storage.local.set({
        sessionActive: false,
        cameraRunning: false,
        sessionMode: null,
        settings: {
            notificationsEnabled: true,
            soundEnabled: false,
            lowAttentionThreshold: 40,
            breakReminderMinutes: 45
        }
    });
});

console.log('[BG] Background ready');
