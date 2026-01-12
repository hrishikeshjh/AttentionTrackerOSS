/**
 * CognEn Extension - Background Service Worker
 * Handles offscreen document management, badge updates, and notifications
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
let offscreenDocumentCreated = false;

// ==================== OFFSCREEN DOCUMENT MANAGEMENT ====================

async function hasOffscreenDocument() {
    const contexts = await chrome.runtime.getContexts({
        contextTypes: ['OFFSCREEN_DOCUMENT']
    });
    return contexts.length > 0;
}

async function createOffscreenDocument() {
    if (await hasOffscreenDocument()) {
        return;
    }

    try {
        await chrome.offscreen.createDocument({
            url: 'offscreen.html',
            reasons: ['USER_MEDIA'],
            justification: 'Camera access for face detection and attention tracking'
        });
        offscreenDocumentCreated = true;
        console.log('Offscreen document created');
    } catch (error) {
        console.error('Failed to create offscreen document:', error);
        throw error;
    }
}

async function closeOffscreenDocument() {
    if (await hasOffscreenDocument()) {
        try {
            await chrome.offscreen.closeDocument();
            offscreenDocumentCreated = false;
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
    switch (message.type) {
        case 'START_CAMERA_REQUEST':
            try {
                await createOffscreenDocument();
                // Wait a bit for the offscreen document to initialize
                setTimeout(() => {
                    chrome.runtime.sendMessage({ type: 'START_CAMERA' });
                }, 1000);
                sendResponse({ success: true });
            } catch (error) {
                sendResponse({ success: false, error: error.message });
            }
            break;

        case 'STOP_CAMERA_REQUEST':
            chrome.runtime.sendMessage({ type: 'STOP_CAMERA' });
            setTimeout(() => {
                closeOffscreenDocument();
            }, 500);
            sendResponse({ success: true });
            break;

        case 'OFFSCREEN_READY':
            console.log('Offscreen document ready');
            break;

        case 'OFFSCREEN_ERROR':
            console.error('Offscreen error:', message.error);
            break;

        case 'CAMERA_STARTED':
            console.log('Camera started in offscreen document');
            break;

        case 'CAMERA_STOPPED':
            console.log('Camera stopped');
            break;

        case 'CAMERA_ERROR':
            console.error('Camera error:', message.error);
            break;

        case 'DETECTION_RESULT':
            // Forward detection results to popup if open
            // Also update badge
            if (message.data && message.data.prediction) {
                updateBadge(message.data.prediction.rawScore);
            }
            break;

        case 'SESSION_START':
            sessionActive = true;
            chrome.action.setBadgeText({ text: '...' });
            chrome.action.setBadgeBackgroundColor({ color: BADGE_COLORS.medium });
            break;

        case 'SESSION_END':
            sessionActive = false;
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

        case 'GET_STATE':
            sendResponse({ sessionActive, currentScore, offscreenDocumentCreated });
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
    const { settings } = await chrome.storage.local.get('settings');

    if (settings && !settings.notificationsEnabled) return;

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
        settings: {
            notificationsEnabled: true,
            multitaskAlerts: true,
            fatigueAlerts: true,
            lowAttentionThreshold: 40,
        }
    });
});

// Handle notification clicks
chrome.notifications.onClicked.addListener((notificationId) => {
    chrome.notifications.clear(notificationId);
});

console.log('CognEn background service worker initialized');
