/**
 * CognEn Extension - Meet Analyzer Content Script
 * Injects into Google Meet and analyzes the existing self-view video
 * No additional camera access needed - uses Meet's existing stream
 */

(function () {
  'use strict';

  console.log('[CognEn] Content script loaded on:', window.location.href);

  // Only run on supported video conferencing sites
  const SUPPORTED_SITES = [
    'meet.google.com',
    'zoom.us',
    'teams.microsoft.com',
    'webex.com'
  ];

  const hostname = window.location.hostname;
  const isSupported = SUPPORTED_SITES.some(site => hostname.includes(site));

  if (!isSupported) {
    console.log('[CognEn] Not a supported video site, skipping');
    return;
  }

  console.log('[CognEn] Detected video conferencing site:', hostname);

  // State
  let isTracking = false;
  let selfVideoElement = null;
  let analysisInterval = null;
  let canvas = null;
  let ctx = null;
  let faceApiLoaded = false;
  let metricsAggregator = null;
  let mlEngine = null;

  // Create hidden canvas for frame capture
  function createCanvas() {
    canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 240;
    canvas.style.display = 'none';
    document.body.appendChild(canvas);
    ctx = canvas.getContext('2d');
  }

  // Find self-view video element in Google Meet
  function findSelfVideo() {
    // Google Meet self-view selectors (they change, so try multiple)
    const selectors = [
      // Main self-view
      'video[data-self-view="true"]',
      // Pinned self
      '.p2hjlc video',
      // Self preview
      '.OcFwLc video',
      // Any video that's small (likely self-view)
      'video',
    ];

    for (const selector of selectors) {
      const videos = document.querySelectorAll(selector);
      for (const video of videos) {
        // Check if video is playing and has reasonable dimensions
        if (video.videoWidth > 0 && video.videoHeight > 0 && !video.paused) {
          // Self-view is usually smaller or mirrored
          const rect = video.getBoundingClientRect();
          // Skip very large videos (probably the main speaker)
          if (rect.width < 400 || videos.length === 1) {
            console.log('[CognEn] Found self-view video:', video);
            return video;
          }
        }
      }
    }

    // Fallback: just get the first playing video
    const allVideos = document.querySelectorAll('video');
    for (const video of allVideos) {
      if (video.videoWidth > 0 && !video.paused) {
        console.log('[CognEn] Using fallback video element');
        return video;
      }
    }

    return null;
  }

  // Load face-api.js dynamically
  async function loadFaceApi() {
    if (faceApiLoaded) return true;

    return new Promise((resolve) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/dist/face-api.min.js';
      script.onload = async () => {
        console.log('[CognEn] face-api.js loaded');

        try {
          const modelUrl = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/';
          await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(modelUrl),
            faceapi.nets.faceLandmark68Net.loadFromUri(modelUrl),
          ]);
          console.log('[CognEn] Face detection models loaded');
          faceApiLoaded = true;
          resolve(true);
        } catch (e) {
          console.error('[CognEn] Failed to load models:', e);
          resolve(false);
        }
      };
      script.onerror = () => {
        console.error('[CognEn] Failed to load face-api.js');
        resolve(false);
      };
      document.head.appendChild(script);
    });
  }

  // Load ML core
  async function loadMLCore() {
    return new Promise((resolve) => {
      const script = document.createElement('script');
      script.src = chrome.runtime.getURL('lib/ml-core.js');
      script.onload = () => {
        console.log('[CognEn] ml-core.js loaded');
        if (typeof FaceMetricsAggregator !== 'undefined') {
          metricsAggregator = new FaceMetricsAggregator();
        }
        if (typeof MLEngine !== 'undefined') {
          mlEngine = new MLEngine();
        }
        resolve(true);
      };
      script.onerror = () => {
        console.error('[CognEn] Failed to load ml-core.js');
        resolve(false);
      };
      document.head.appendChild(script);
    });
  }

  // Analyze a single frame
  async function analyzeFrame() {
    if (!selfVideoElement || !faceApiLoaded || !ctx) return null;

    try {
      // Draw video frame to canvas
      ctx.drawImage(selfVideoElement, 0, 0, canvas.width, canvas.height);

      // Detect face
      const detections = await faceapi
        .detectAllFaces(canvas, new faceapi.TinyFaceDetectorOptions({ inputSize: 320 }))
        .withFaceLandmarks();

      if (detections && detections.length > 0 && metricsAggregator && mlEngine) {
        const landmarks = detections[0].landmarks.positions;
        const timestamp = Date.now();

        const metrics = metricsAggregator.calculateExtendedMetrics(
          landmarks,
          canvas.width,
          canvas.height,
          timestamp
        );

        const prediction = mlEngine.predict(metrics);

        return { prediction, metrics, hasFace: true };
      }

      return { hasFace: false };
    } catch (e) {
      console.error('[CognEn] Analysis error:', e);
      return null;
    }
  }

  // Start tracking
  async function startTracking() {
    console.log('[CognEn] Starting tracking...');

    // Find video element
    selfVideoElement = findSelfVideo();
    if (!selfVideoElement) {
      console.log('[CognEn] No video element found, will retry...');
      // Retry after a delay
      setTimeout(startTracking, 2000);
      return;
    }

    // Create canvas
    if (!canvas) createCanvas();

    // Load dependencies
    const faceLoaded = await loadFaceApi();
    const mlLoaded = await loadMLCore();

    if (!faceLoaded || !mlLoaded) {
      console.error('[CognEn] Failed to load dependencies');
      chrome.storage.local.set({ cameraError: 'Failed to load ML models' });
      return;
    }

    isTracking = true;
    chrome.storage.local.set({ cameraRunning: true });

    // Start analysis loop
    let frameCount = 0;
    analysisInterval = setInterval(async () => {
      if (!isTracking) return;

      // Re-find video if lost
      if (!selfVideoElement || selfVideoElement.paused || selfVideoElement.videoWidth === 0) {
        selfVideoElement = findSelfVideo();
        if (!selfVideoElement) return;
      }

      const result = await analyzeFrame();
      frameCount++;

      if (result) {
        chrome.storage.local.set({
          lastDetection: {
            timestamp: Date.now(),
            hasFace: result.hasFace,
            frameCount,
            prediction: result.prediction ? {
              rawScore: result.prediction.rawScore,
              attentionState: result.prediction.attentionState,
              confidence: result.prediction.confidence
            } : null,
            metrics: result.metrics ? {
              blinkRate: result.metrics.blinkRate,
              fatigueIndex: result.metrics.fatigueIndex,
              perclos: result.metrics.perclos
            } : null
          }
        });

        if (frameCount % 30 === 0) {
          console.log(`[CognEn] Frame ${frameCount}: Score ${result.prediction?.rawScore || '-'}`);
        }
      }
    }, 200); // Analyze ~5 times per second

    console.log('[CognEn] Tracking started');
  }

  // Stop tracking
  function stopTracking() {
    console.log('[CognEn] Stopping tracking...');
    isTracking = false;

    if (analysisInterval) {
      clearInterval(analysisInterval);
      analysisInterval = null;
    }

    if (metricsAggregator) metricsAggregator.reset();
    if (mlEngine) mlEngine.reset();

    chrome.storage.local.set({ cameraRunning: false });
  }

  // Create floating status widget
  function createWidget() {
    const widget = document.createElement('div');
    widget.id = 'cognen-widget';
    widget.innerHTML = `
      <style>
        #cognen-widget {
          position: fixed;
          bottom: 20px;
          right: 20px;
          z-index: 999999;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          background: rgba(13, 17, 23, 0.95);
          border: 1px solid rgba(0, 255, 255, 0.3);
          border-radius: 12px;
          padding: 12px 16px;
          min-width: 140px;
          box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
          backdrop-filter: blur(10px);
          cursor: move;
          user-select: none;
        }
        #cognen-widget.minimized {
          min-width: auto;
          padding: 8px 12px;
        }
        .cognen-header {
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
          font-size: 16px;
        }
        .cognen-close:hover { color: #F85149; }
        .cognen-score {
          font-size: 36px;
          font-weight: 700;
          text-align: center;
          color: #00FF99;
        }
        .cognen-score.warning { color: #FFA657; }
        .cognen-score.danger { color: #F85149; }
        .cognen-state {
          font-size: 11px;
          text-align: center;
          color: #8B949E;
          text-transform: uppercase;
        }
        .cognen-stats {
          display: flex;
          justify-content: space-around;
          margin-top: 8px;
          padding-top: 8px;
          border-top: 1px solid rgba(255,255,255,0.1);
        }
        .cognen-stat {
          text-align: center;
        }
        .cognen-stat-value {
          font-size: 14px;
          font-weight: 600;
          color: #F0F6FC;
        }
        .cognen-stat-label {
          font-size: 9px;
          color: #6E7681;
        }
      </style>
      <div class="cognen-header">
        <span class="cognen-logo">👁 CognEn</span>
        <button class="cognen-close" id="cognen-minimize">−</button>
      </div>
      <div id="cognen-body">
        <div class="cognen-score" id="cognen-score">--</div>
        <div class="cognen-state" id="cognen-state">Starting...</div>
        <div class="cognen-stats">
          <div class="cognen-stat">
            <div class="cognen-stat-value" id="cognen-blink">-</div>
            <div class="cognen-stat-label">Blinks/m</div>
          </div>
          <div class="cognen-stat">
            <div class="cognen-stat-value" id="cognen-fatigue">-</div>
            <div class="cognen-stat-label">Fatigue</div>
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(widget);

    // Make draggable
    let isDragging = false;
    let startX, startY, initialX, initialY;

    widget.addEventListener('mousedown', (e) => {
      if (e.target.tagName === 'BUTTON') return;
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = widget.getBoundingClientRect();
      initialX = rect.left;
      initialY = rect.top;
    });

    document.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      widget.style.left = (initialX + dx) + 'px';
      widget.style.top = (initialY + dy) + 'px';
      widget.style.right = 'auto';
      widget.style.bottom = 'auto';
    });

    document.addEventListener('mouseup', () => {
      isDragging = false;
    });

    // Minimize button
    document.getElementById('cognen-minimize').addEventListener('click', () => {
      const body = document.getElementById('cognen-body');
      body.style.display = body.style.display === 'none' ? 'block' : 'none';
      widget.classList.toggle('minimized');
    });

    return widget;
  }

  // Update widget with detection results
  function updateWidget() {
    const scoreEl = document.getElementById('cognen-score');
    const stateEl = document.getElementById('cognen-state');
    const blinkEl = document.getElementById('cognen-blink');
    const fatigueEl = document.getElementById('cognen-fatigue');

    if (!scoreEl) return;

    chrome.storage.local.get(['lastDetection'], (data) => {
      if (data.lastDetection) {
        const det = data.lastDetection;
        const age = Date.now() - det.timestamp;

        if (age < 2000 && det.hasFace && det.prediction) {
          const score = det.prediction.rawScore;
          scoreEl.textContent = score;
          scoreEl.className = 'cognen-score';
          if (score < 40) scoreEl.classList.add('danger');
          else if (score < 70) scoreEl.classList.add('warning');

          stateEl.textContent = det.prediction.attentionState;

          if (det.metrics) {
            blinkEl.textContent = det.metrics.blinkRate?.toFixed(1) || '-';
            const fatigue = det.metrics.fatigueIndex || 0;
            fatigueEl.textContent = fatigue < 30 ? 'Low' : fatigue < 60 ? 'Med' : 'High';
          }
        } else if (!det.hasFace) {
          scoreEl.textContent = '--';
          stateEl.textContent = 'No Face';
        }
      }
    });
  }

  // Handle messages from popup/background
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('[CognEn] Content received:', message);

    if (message.action === 'START_MEET_TRACKING') {
      createWidget();
      startTracking();
      // Start widget update interval
      setInterval(updateWidget, 500);
      sendResponse({ success: true });
    }

    if (message.action === 'STOP_MEET_TRACKING') {
      stopTracking();
      const widget = document.getElementById('cognen-widget');
      if (widget) widget.remove();
      sendResponse({ success: true });
    }

    if (message.action === 'CHECK_MEET_STATUS') {
      const video = findSelfVideo();
      sendResponse({
        isVideoSite: true,
        hasVideo: !!video,
        isTracking
      });
    }

    return true;
  });

  // Notify that content script is ready
  chrome.storage.local.set({ contentScriptReady: hostname });
  console.log('[CognEn] Content script ready');

})();
