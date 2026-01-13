/**
 * CognEn Extension - Background Service Worker
 * Handles both offscreen camera (standalone) and Meet injection modes
 * Uses chrome.scripting to inject content script programmatically
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

async function injectContentScript(tabId) {
    try {
        // First inject the ML core library
        await chrome.scripting.executeScript({
            target: { tabId },
            files: ['lib/ml-core.js']
        });
        console.log('[BG] Injected ml-core.js');

        // Then inject the content script
        await chrome.scripting.executeScript({
            target: { tabId },
            files: ['content.js']
        });
        console.log('[BG] Injected content.js');

        return true;
    } catch (e) {
        console.error('[BG] Script injection failed:', e);
        return false;
    }
}

async function startMeetTracking(tabId) {
    try {
        // First inject the scripts
        const injected = await injectContentScript(tabId);
        if (!injected) {
            throw new Error('Failed to inject scripts');
        }

        // Wait a bit for scripts to initialize
        await new Promise(r => setTimeout(r, 500));

        // Now send the start message
        await chrome.tabs.sendMessage(tabId, { action: 'START_MEET_TRACKING' });
        console.log('[BG] Meet tracking started');
        return true;
    } catch (e) {
        console.error('[BG] Failed to start Meet tracking:', e);
        return false;
    }
}

async function stopMeetTracking(tabId) {
    try {
        await chrome.tabs.sendMessage(tabId, { action: 'STOP_MEET_TRACKING' });
    } catch (e) {
        console.log('[BG] Could not send stop message (tab may be closed)');
    }
}

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
                    // Meet mode: inject into page
                    console.log('[BG] Starting Meet mode on tab', meetInfo.tabId);
                    const success = await startMeetTracking(meetInfo.tabId);

                    if (!success) {
                        throw new Error('Failed to inject into Meet page');
                    }

                    await chrome.storage.local.set({
                        sessionMode: 'meet',
                        sessionTabId: meetInfo.tabId,
                        cameraRunning: true
                    });

                    sendResponse({ success: true, mode: 'meet' });
                } else {
                    // Standalone mode: use offscreen document
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
                            } catch (e2) {
                                console.log('[BG] Could not send START_CAMERA');
                            }
                        }
                    }

                    await chrome.storage.local.set({ sessionMode: 'standalone' });
                    sendResponse({ success: created, mode: 'standalone' });
                }

                chrome.action.setBadgeText({ text: '...' });
                chrome.action.setBadgeBackgroundColor({ color: BADGE_COLORS.medium });

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

        case 'RESET_ML':
            try {
                await chrome.runtime.sendMessage({ type: 'RESET_ML' });
            } catch (e) { }
            sendResponse({ success: true });
            break;

        default:
            sendResponse({ success: false, error: 'Unknown action' });
    }
}

// ==================== BADGE UPDATES ====================

setInterval(async () => {
    const data = await chrome.storage.local.get(['lastDetection', 'sessionActive']);
    if (data.sessionActive && data.lastDetection?.prediction?.rawScore) {
        const score = data.lastDetection.prediction.rawScore;
        chrome.action.setBadgeText({ text: String(score) });

        let color;
        if (score >= 70) color = BADGE_COLORS.high;
        else if (score >= 40) color = BADGE_COLORS.medium;
        else color = BADGE_COLORS.low;
        chrome.action.setBadgeBackgroundColor({ color });
    }
}, 2000);

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
