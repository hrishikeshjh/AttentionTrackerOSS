/**
 * CognEn Extension - Background Service Worker
 * Handles offscreen document management, badge updates, and notifications
 * Keeps offscreen document alive even when popup is closed
 */

// Badge colors
const BADGE_COLORS = {
    high: '#3FB950',
    medium: '#FFA657',
    low: '#F85149',
    inactive: '#6E7681'
};

// State
let currentScore = 0;
let sessionActive = false;

// ==================== OFFSCREEN DOCUMENT MANAGEMENT ====================

async function hasOffscreenDocument() {
    try {
        const contexts = await chrome.runtime.getContexts({
            contextTypes: ['OFFSCREEN_DOCUMENT']
        });
        return contexts.length > 0;
    } catch (e) {
        console.error('Error checking offscreen document:', e);
        return false;
    }
}

async function createOffscreenDocument() {
    if (await hasOffscreenDocument()) {
        console.log('Offscreen document already exists');
        return true;
    }

    try {
        await chrome.offscreen.createDocument({
            url: 'offscreen.html',
            reasons: ['USER_MEDIA'],
            justification: 'Camera access for face detection and attention tracking'
        });
        console.log('Offscreen document created successfully');
        return true;
    } catch (error) {
        console.error('Failed to create offscreen document:', error);
        return false;
    }
}

async function closeOffscreenDocument() {
    if (await hasOffscreenDocument()) {
        try {
            await chrome.offscreen.closeDocument();
            console.log('Offscreen document closed');
        } catch (error) {
            console.error('Failed to close offscreen document:', error);
        }
    }
}

// ==================== MESSAGE HANDLING ====================

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    handleMessage(message, sender, sendResponse);
    return true; // Keep channel open for async response
});

async function handleMessage(message, sender, sendResponse) {
    console.log('Background received message:', message.type);

    switch (message.type) {
        case 'START_CAMERA_REQUEST':
            try {
                const created = await createOffscreenDocument();
                if (created) {
                    // Wait for offscreen to be ready, then start camera
                    setTimeout(async () => {
                        try {
                            await chrome.runtime.sendMessage({ type: 'START_CAMERA' });
                            console.log('START_CAMERA message sent to offscreen');
                        } catch (e) {
                            console.log('Could not send START_CAMERA, offscreen may still be loading');
                        }
                    }, 1500);
                    sendResponse({ success: true });
                } else {
                    sendResponse({ success: false, error: 'Failed to create offscreen document' });
                }
            } catch (error) {
                sendResponse({ success: false, error: error.message });
            }
            break;

        case 'STOP_CAMERA_REQUEST':
            try {
                await chrome.runtime.sendMessage({ type: 'STOP_CAMERA' });
            } catch (e) {
                console.log('Could not send STOP_CAMERA');
            }
            // DON'T close offscreen document - keep it alive
            // Only close when explicitly stopping session
            await closeOffscreenDocument();
            chrome.storage.local.set({ cameraRunning: false });
            sendResponse({ success: true });
            break;

        case 'SESSION_START':
            sessionActive = true;
            chrome.storage.local.set({ sessionActive: true });
            chrome.action.setBadgeText({ text: '...' });
            chrome.action.setBadgeBackgroundColor({ color: BADGE_COLORS.medium });
            break;

        case 'SESSION_END':
            sessionActive = false;
            chrome.storage.local.set({ sessionActive: false });
            chrome.action.setBadgeText({ text: '' });

            if (message.data && message.data.avgScore) {
                showNotification(
                    'Session Complete',
                    `Average attention score: ${message.data.avgScore}%`,
                    'session_end'
                );
            }
            break;

        case 'UPDATE_SCORE':
            if (message.data && typeof message.data.score === 'number') {
                updateBadge(message.data.score);
            }
            break;

        case 'RESET_ML':
            try {
                await chrome.runtime.sendMessage({ type: 'RESET_ML' });
            } catch (e) {
                console.log('Could not send RESET_ML');
            }
            break;

        case 'GET_STATE':
            const hasOffscreen = await hasOffscreenDocument();
            sendResponse({
                sessionActive,
                currentScore,
                hasOffscreen
            });
            break;
    }
}

// ==================== BADGE UPDATES ====================

function updateBadge(score) {
    currentScore = score;

    chrome.action.setBadgeText({ text: String(score) });

    let color;
    if (score >= 70) {
        color = BADGE_COLORS.high;
    } else if (score >= 40) {
        color = BADGE_COLORS.medium;
    } else {
        color = BADGE_COLORS.low;
    }
    chrome.action.setBadgeBackgroundColor({ color });
}

// ==================== NOTIFICATIONS ====================

async function showNotification(title, message, id) {
    const result = await chrome.storage.local.get('settings');
    const settings = result.settings || {};

    if (settings.notificationsEnabled === false) return;

    chrome.notifications.create(id, {
        type: 'basic',
        iconUrl: 'icons/icon128.png',
        title: title,
        message: message,
        priority: 1
    });
}

// ==================== INITIALIZATION ====================

chrome.runtime.onInstalled.addListener(() => {
    console.log('CognEn Extension installed');

    chrome.action.setBadgeBackgroundColor({ color: BADGE_COLORS.inactive });
    chrome.action.setBadgeText({ text: '' });

    chrome.storage.local.set({
        sessionActive: false,
        cameraRunning: false,
        settings: {
            notificationsEnabled: true,
            soundEnabled: false,
            lowAttentionThreshold: 40,
            breakReminderMinutes: 45
        }
    });
});

// Handle notification clicks
chrome.notifications.onClicked.addListener((notificationId) => {
    chrome.notifications.clear(notificationId);
});

// Keep alive - periodically check storage for updates
setInterval(async () => {
    const data = await chrome.storage.local.get(['lastDetection', 'sessionActive']);
    if (data.sessionActive && data.lastDetection && data.lastDetection.prediction) {
        updateBadge(data.lastDetection.prediction.rawScore);
    }
}, 1000);

console.log('CognEn background service worker initialized');
