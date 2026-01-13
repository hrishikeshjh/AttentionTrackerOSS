/**
 * CognEn Extension - Meet Analyzer Content Script
 * Injects into Google Meet and analyzes the existing self-view video
 * No additional camera access needed - uses Meet's existing stream
 */

console.log('[CognEn] Content script injected on:', window.location.href);

// Prevent duplicate injection
if (window.__cognEnInjected) {
  console.log('[CognEn] Already injected, skipping');
} else {
  window.__cognEnInjected = true;

  // State
  let isTracking = false;
  let selfVideoElement = null;
  let analysisInterval = null;
  let canvas = null;
  let ctx = null;
  let faceApiLoaded = false;
  let metricsAggregator = null;
  let mlEngine = null;
  let widget = null;

  // Create hidden canvas for frame capture
  function createCanvas() {
    if (canvas) return;
    canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 240;
    canvas.style.display = 'none';
    document.body.appendChild(canvas);
    ctx = canvas.getContext('2d');
  }

  // Find self-view video element
  function findSelfVideo() {
    // Get all video elements
    const videos = document.querySelectorAll('video');

    for (const video of videos) {
      if (video.videoWidth > 0 && video.videoHeight > 0 && !video.paused) {
        const rect = video.getBoundingClientRect();
        // Self-view is usually smaller (< 300px wide) or there's only one video
        if (rect.width > 0 && (rect.width < 400 || videos.length <= 2)) {
          console.log('[CognEn] Found video:', rect.width, 'x', rect.height);
          return video;
        }
      }
    }

    // Fallback: just get first playing video
    for (const video of videos) {
      if (video.videoWidth > 0 && !video.paused) {
        return video;
      }
    }

    return null;
  }

  // Load face-api.js dynamically
  async function loadFaceApi() {
    if (faceApiLoaded) return true;
    if (typeof faceapi !== 'undefined') {
      faceApiLoaded = true;
      return true;
    }

    return new Promise((resolve) => {
      console.log('[CognEn] Loading face-api.js...');
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
          console.log('[CognEn] Face models loaded');
          faceApiLoaded = true;
          resolve(true);
        } catch (e) {
          console.error('[CognEn] Model load error:', e);
          resolve(false);
        }
      };
      script.onerror = () => resolve(false);
      document.head.appendChild(script);
    });
  }

  // Initialize ML components (from ml-core.js injected earlier)
  function initMLComponents() {
    if (typeof FaceMetricsAggregator !== 'undefined') {
      metricsAggregator = new FaceMetricsAggregator();
      console.log('[CognEn] FaceMetricsAggregator ready');
    }
    if (typeof MLEngine !== 'undefined') {
      mlEngine = new MLEngine();
      console.log('[CognEn] MLEngine ready');
    }
    return !!(metricsAggregator && mlEngine);
  }

  // Analyze a single frame
  async function analyzeFrame() {
    if (!selfVideoElement || !faceApiLoaded || !ctx) return null;

    // Re-check video element
    if (selfVideoElement.paused || selfVideoElement.videoWidth === 0) {
      selfVideoElement = findSelfVideo();
      if (!selfVideoElement) return null;
    }

    try {
      ctx.drawImage(selfVideoElement, 0, 0, canvas.width, canvas.height);

      const detections = await faceapi
        .detectAllFaces(canvas, new faceapi.TinyFaceDetectorOptions({ inputSize: 320 }))
        .withFaceLandmarks();

      if (detections && detections.length > 0 && metricsAggregator && mlEngine) {
        const landmarks = detections[0].landmarks.positions;
        const timestamp = Date.now();

        const metrics = metricsAggregator.calculateExtendedMetrics(
          landmarks, canvas.width, canvas.height, timestamp
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

  // Create floating widget
  function createWidget() {
    if (document.getElementById('cognen-widget')) return;

    widget = document.createElement('div');
    widget.id = 'cognen-widget';
    widget.innerHTML = `
      <style>
        #cognen-widget {
          position: fixed;
          bottom: 20px;
          right: 20px;
          z-index: 999999;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          background: rgba(13, 17, 23, 0.95);
          border: 1px solid rgba(0, 255, 255, 0.3);
          border-radius: 12px;
          padding: 12px 16px;
          min-width: 120px;
          box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(10px);
        }
        .cognen-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
          color: #00FFFF;
          font-size: 11px;
          font-weight: 600;
        }
        .cognen-close {
          background: none;
          border: none;
          color: #8B949E;
          cursor: pointer;
          font-size: 14px;
          padding: 0;
        }
        .cognen-close:hover { color: #F85149; }
        .cognen-score {
          font-size: 42px;
          font-weight: 700;
          text-align: center;
          color: #00FF99;
          line-height: 1;
        }
        .cognen-score.warning { color: #FFA657; }
        .cognen-score.danger { color: #F85149; }
        .cognen-state {
          font-size: 10px;
          text-align: center;
          color: #8B949E;
          text-transform: uppercase;
          margin-top: 4px;
        }
        .cognen-stats {
          display: flex;
          justify-content: space-between;
          margin-top: 8px;
          padding-top: 8px;
          border-top: 1px solid rgba(255,255,255,0.1);
          gap: 12px;
        }
        .cognen-stat { text-align: center; }
        .cognen-stat-value { font-size: 14px; font-weight: 600; color: #F0F6FC; }
        .cognen-stat-label { font-size: 8px; color: #6E7681; text-transform: uppercase; }
      </style>
      <div class="cognen-header">
        <span>👁 CognEn</span>
        <button class="cognen-close" id="cognen-close">✕</button>
      </div>
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
    `;

    document.body.appendChild(widget);

    document.getElementById('cognen-close').addEventListener('click', () => {
      stopTracking();
      chrome.runtime.sendMessage({ action: 'STOP_SESSION' });
    });
  }

  // Update widget display
  function updateWidget(result) {
    const scoreEl = document.getElementById('cognen-score');
    const stateEl = document.getElementById('cognen-state');
    const blinkEl = document.getElementById('cognen-blink');
    const fatigueEl = document.getElementById('cognen-fatigue');

    if (!scoreEl) return;

    if (result && result.hasFace && result.prediction) {
      const score = result.prediction.rawScore;
      scoreEl.textContent = score;
      scoreEl.className = 'cognen-score';
      if (score < 40) scoreEl.classList.add('danger');
      else if (score < 70) scoreEl.classList.add('warning');

      stateEl.textContent = result.prediction.attentionState;

      if (result.metrics) {
        blinkEl.textContent = result.metrics.blinkRate?.toFixed(1) || '-';
        const fatigue = result.metrics.fatigueIndex || 0;
        fatigueEl.textContent = fatigue < 30 ? 'Low' : fatigue < 60 ? 'Med' : 'High';
      }
    } else {
      scoreEl.textContent = '--';
      stateEl.textContent = 'No Face';
    }
  }

  // Start tracking
  async function startTracking() {
    console.log('[CognEn] Starting tracking...');

    // Find video
    selfVideoElement = findSelfVideo();
    if (!selfVideoElement) {
      console.log('[CognEn] No video found, retrying in 2s...');
      setTimeout(startTracking, 2000);
      return;
    }

    // Create canvas
    createCanvas();

    // Init ML components
    const mlReady = initMLComponents();
    if (!mlReady) {
      console.error('[CognEn] ML components not ready');
    }

    // Load face-api
    const faceLoaded = await loadFaceApi();
    if (!faceLoaded) {
      console.error('[CognEn] Failed to load face-api');
      chrome.storage.local.set({ cameraError: 'Failed to load ML models' });
      return;
    }

    // Create widget
    createWidget();

    isTracking = true;
    chrome.storage.local.set({ cameraRunning: true });

    // Start analysis loop
    let frameCount = 0;
    analysisInterval = setInterval(async () => {
      if (!isTracking) return;

      const result = await analyzeFrame();
      frameCount++;

      if (result) {
        // Update widget
        updateWidget(result);

        // Store for popup
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
      }
    }, 200);

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

    if (widget) {
      widget.remove();
      widget = null;
    }

    chrome.storage.local.set({ cameraRunning: false });
  }

  // Message handler
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('[CognEn] Received:', message);

    if (message.action === 'START_MEET_TRACKING') {
      startTracking();
      sendResponse({ success: true });
    }

    if (message.action === 'STOP_MEET_TRACKING') {
      stopTracking();
      sendResponse({ success: true });
    }

    return true;
  });

  console.log('[CognEn] Content script ready');
}
