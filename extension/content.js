/**
 * CognEn Extension - Content Script
 * Detects page activity and provides optional floating widget mode
 */

// Configuration
const CONFIG = {
    activityCheckInterval: 1000,
    idleThreshold: 30000, // 30 seconds of inactivity
};

// State
let lastActivity = Date.now();
let isIdle = false;

// Track user activity
const activityEvents = ['mousedown', 'mousemove', 'keydown', 'scroll', 'touchstart'];

function handleActivity() {
    lastActivity = Date.now();

    if (isIdle) {
        isIdle = false;
        chrome.runtime.sendMessage({ type: 'USER_ACTIVE' });
    }
}

// Add activity listeners
activityEvents.forEach(event => {
    document.addEventListener(event, handleActivity, { passive: true });
});

// Check for idle state
setInterval(() => {
    if (Date.now() - lastActivity > CONFIG.idleThreshold) {
        if (!isIdle) {
            isIdle = true;
            chrome.runtime.sendMessage({ type: 'USER_IDLE' });
        }
    }
}, CONFIG.activityCheckInterval);

// Listen for messages from extension
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    switch (message.type) {
        case 'GET_PAGE_ACTIVITY':
            sendResponse({
                lastActivity,
                isIdle,
                url: window.location.href,
                title: document.title
            });
            break;

        case 'INJECT_WIDGET':
            injectFloatingWidget();
            break;

        case 'REMOVE_WIDGET':
            removeFloatingWidget();
            break;

        case 'UPDATE_WIDGET_SCORE':
            updateWidgetScore(message.data);
            break;
    }

    return true;
});

// Floating Widget (optional feature for future)
let widget = null;

function injectFloatingWidget() {
    if (widget) return;

    widget = document.createElement('div');
    widget.id = 'cognen-widget';
    widget.innerHTML = `
    <div class="cognen-widget-container">
      <div class="cognen-widget-header">
        <span class="cognen-logo">👁 CognEn</span>
        <button class="cognen-close" id="cognen-close">×</button>
      </div>
      <div class="cognen-widget-body">
        <div class="cognen-score" id="cognen-score">--</div>
        <div class="cognen-label">Attention Score</div>
      </div>
    </div>
  `;

    const style = document.createElement('style');
    style.id = 'cognen-widget-styles';
    style.textContent = `
    #cognen-widget {
      position: fixed;
      bottom: 20px;
      right: 20px;
      z-index: 2147483647;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .cognen-widget-container {
      background: rgba(13, 17, 23, 0.95);
      border: 1px solid rgba(0, 255, 255, 0.3);
      border-radius: 12px;
      padding: 12px;
      min-width: 120px;
      box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
      backdrop-filter: blur(10px);
    }
    
    .cognen-widget-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    
    .cognen-logo {
      color: #00FFFF;
      font-size: 12px;
      font-weight: 600;
    }
    
    .cognen-close {
      background: none;
      border: none;
      color: #8B949E;
      cursor: pointer;
      font-size: 18px;
      padding: 0;
      line-height: 1;
    }
    
    .cognen-close:hover {
      color: #F85149;
    }
    
    .cognen-widget-body {
      text-align: center;
    }
    
    .cognen-score {
      font-size: 32px;
      font-weight: 700;
      color: #00FF99;
      line-height: 1;
    }
    
    .cognen-score.warning {
      color: #FFA657;
    }
    
    .cognen-score.danger {
      color: #F85149;
    }
    
    .cognen-label {
      font-size: 10px;
      color: #8B949E;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-top: 4px;
    }
  `;

    document.head.appendChild(style);
    document.body.appendChild(widget);

    // Add close button handler
    document.getElementById('cognen-close').addEventListener('click', () => {
        removeFloatingWidget();
        chrome.runtime.sendMessage({ type: 'WIDGET_CLOSED' });
    });
}

function removeFloatingWidget() {
    if (widget) {
        widget.remove();
        widget = null;
    }

    const style = document.getElementById('cognen-widget-styles');
    if (style) {
        style.remove();
    }
}

function updateWidgetScore(data) {
    if (!widget) return;

    const scoreEl = document.getElementById('cognen-score');
    if (scoreEl && data.score !== undefined) {
        scoreEl.textContent = data.score;
        scoreEl.className = 'cognen-score';

        if (data.score < 40) {
            scoreEl.classList.add('danger');
        } else if (data.score < 70) {
            scoreEl.classList.add('warning');
        }
    }
}

// Notify that content script is ready
chrome.runtime.sendMessage({ type: 'CONTENT_SCRIPT_READY', url: window.location.href });

console.log('CognEn content script loaded');
