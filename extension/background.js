/**
 * CognEn Extension - Background Service Worker
 * Handles tab visibility, badge updates, and notifications
 */

// Badge colors
const BADGE_COLORS = {
    high: '#3FB950',    // Green
    medium: '#FFA657',  // Orange
    low: '#F85149',     // Red
    inactive: '#6E7681' // Gray
};

// State
let currentScore = 0;
let sessionActive = false;
let tabHiddenStart = null;
const MULTITASK_THRESHOLD_MS = 3000; // 3 seconds

// Initialize
chrome.runtime.onInstalled.addListener(() => {
    console.log('CognEn Extension installed');

    // Set default badge
    chrome.action.setBadgeBackgroundColor({ color: BADGE_COLORS.inactive });
    chrome.action.setBadgeText({ text: '' });

    // Initialize storage
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

// Handle messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    switch (message.type) {
        case 'SESSION_START':
            sessionActive = true;
            chrome.action.setBadgeText({ text: '...' });
            chrome.action.setBadgeBackgroundColor({ color: BADGE_COLORS.medium });
            break;

        case 'SESSION_END':
            sessionActive = false;
            chrome.action.setBadgeText({ text: '' });

            // Show session summary notification
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

        case 'POPUP_CLOSED':
            // Session continues in background, but we can't do face detection
            // Just maintain the badge state
            break;

        case 'GET_STATE':
            sendResponse({ sessionActive, currentScore });
            break;
    }

    return true; // Keep channel open for async response
});

// Update badge with current score
function updateBadge(score) {
    currentScore = score;

    // Update badge text
    chrome.action.setBadgeText({ text: String(score) });

    // Update badge color based on score
    let color;
    if (score >= 70) {
        color = BADGE_COLORS.high;
    } else if (score >= 40) {
        color = BADGE_COLORS.medium;
    } else {
        color = BADGE_COLORS.low;
    }
    chrome.action.setBadgeBackgroundColor({ color });

    // Check for low attention alerts
    checkLowAttention(score);
}

// Check for low attention and show alert
async function checkLowAttention(score) {
    const { settings } = await chrome.storage.local.get('settings');

    if (settings.notificationsEnabled && score < settings.lowAttentionThreshold) {
        // Debounce - only show once per minute
        const lastAlert = await chrome.storage.local.get('lastLowAttentionAlert');
        const now = Date.now();

        if (!lastAlert.lastLowAttentionAlert || now - lastAlert.lastLowAttentionAlert > 60000) {
            showNotification(
                'Low Attention Detected',
                'Your attention score has dropped. Take a short break or refocus.',
                'low_attention'
            );
            chrome.storage.local.set({ lastLowAttentionAlert: now });
        }
    }
}

// Show notification
async function showNotification(title, message, id) {
    const { settings } = await chrome.storage.local.get('settings');

    if (!settings.notificationsEnabled) return;

    chrome.notifications.create(id, {
        type: 'basic',
        iconUrl: 'icons/icon128.png',
        title: title,
        message: message,
        priority: 1
    });
}

// Tab visibility change detection (for multitasking)
chrome.tabs.onActivated.addListener(async (activeInfo) => {
    if (!sessionActive) return;

    // Tab switched - this is a multitasking event
    const { settings } = await chrome.storage.local.get('settings');

    if (settings.multitaskAlerts) {
        // Store tab switch event
        const { tabSwitches = [] } = await chrome.storage.local.get('tabSwitches');
        tabSwitches.push({
            timestamp: Date.now(),
            tabId: activeInfo.tabId
        });

        // Keep only last 100 switches
        if (tabSwitches.length > 100) {
            tabSwitches.shift();
        }

        chrome.storage.local.set({ tabSwitches });
    }
});

// Alarm for periodic checks (every 5 minutes)
chrome.alarms.create('periodicCheck', { periodInMinutes: 5 });

chrome.alarms.onAlarm.addListener(async (alarm) => {
    if (alarm.name === 'periodicCheck' && sessionActive) {
        // Check fatigue based on session duration
        const { sessionStartTime } = await chrome.storage.local.get('sessionStartTime');

        if (sessionStartTime) {
            const sessionDuration = (Date.now() - sessionStartTime) / 1000 / 60; // minutes

            if (sessionDuration > 45) {
                const { settings } = await chrome.storage.local.get('settings');

                if (settings.fatigueAlerts) {
                    showNotification(
                        'Break Reminder',
                        `You've been focused for ${Math.round(sessionDuration)} minutes. Consider taking a break!`,
                        'break_reminder'
                    );
                }
            }
        }
    }
});

// Handle notification clicks
chrome.notifications.onClicked.addListener((notificationId) => {
    chrome.notifications.clear(notificationId);

    // Open popup when notification is clicked
    chrome.action.openPopup();
});

console.log('CognEn background service worker initialized');
