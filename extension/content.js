/**
 * CognEn Extension - Meet Analyzer Content Script
 * Runs in MAIN world to access ML classes
 * Uses chrome.storage polling for commands (MAIN world can't use message passing)
 */

(function () {
  'use strict';

  console.log('[CognEn] Content script loaded in MAIN world');

  // Prevent duplicate injection
  if (window.__cognEnInjected) {
    console.log('[CognEn] Already injected');
    return;
  }
  window.__cognEnInjected = true;

  // State
  let isTracking = false;
  let selfVideoElement = null;
  let analysisInterval = null;
  let commandPollInterval = null;
  let canvas = null;
  let ctx = null;
  let faceApiLoaded = false;
  let metricsAggregator = null;
  let mlEngine = null;
  let widget = null;
  let lastCommandTimestamp = 0;

  // Simple storage wrapper (works in MAIN world via window.postMessage bridge)
  const storage = {
    async get(keys) {
      return new Promise(resolve => {
        // Use localStorage as a bridge
        try {
          const data = {};
          keys.forEach(key => {
            const val = localStorage.getItem('cognen_' + key);
            if (val) data[key] = JSON.parse(val);
          });
          resolve(data);
        } catch (e) {
          resolve({});
        }
      });
    },
    async set(data) {
      try {
        Object.entries(data).forEach(([key, val]) => {
          localStorage.setItem('cognen_' + key, JSON.stringify(val));
        });
      } catch (e) { }
    }
  };

  // Create canvas
  function createCanvas() {
    if (canvas) return;
    canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 240;
    canvas.style.display = 'none';
    document.body.appendChild(canvas);
    ctx = canvas.getContext('2d');
  }

  // Find self-view video
  function findSelfVideo() {
    const videos = document.querySelectorAll('video');
    console.log('[CognEn] Found', videos.length, 'video elements');

    for (const video of videos) {
      if (video.videoWidth > 0 && !video.paused) {
        const rect = video.getBoundingClientRect();
        console.log('[CognEn] Video:', rect.width, 'x', rect.height, 'playing:', !video.paused);
        if (rect.width > 0) {
          return video;
        }
      }
    }
    return null;
  }

  // Load face-api
  async function loadFaceApi() {
    if (faceApiLoaded || typeof faceapi !== 'undefined') {
      if (typeof faceapi !== 'undefined') {
        faceApiLoaded = true;
        return true;
      }
    }

    return new Promise((resolve) => {
      console.log('[CognEn] Loading face-api.js...');
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/dist/face-api.min.js';
      script.onload = async () => {
        console.log('[CognEn] face-api loaded, loading models...');
        try {
          const modelUrl = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/';
          await faceapi.nets.tinyFaceDetector.loadFromUri(modelUrl);
          await faceapi.nets.faceLandmark68Net.loadFromUri(modelUrl);
          console.log('[CognEn] Models loaded!');
          faceApiLoaded = true;
          resolve(true);
        } catch (e) {
          console.error('[CognEn] Model load error:', e);
          resolve(false);
        }
      };
      script.onerror = (e) => {
        console.error('[CognEn] face-api load error:', e);
        resolve(false);
      };
      document.head.appendChild(script);
    });
  }

  // Init ML
  function initML() {
    console.log('[CognEn] Checking ML components...');
    console.log('[CognEn] FaceMetricsAggregator:', typeof FaceMetricsAggregator);
    console.log('[CognEn] MLEngine:', typeof MLEngine);

    if (typeof FaceMetricsAggregator !== 'undefined') {
      metricsAggregator = new FaceMetricsAggregator();
      console.log('[CognEn] Created FaceMetricsAggregator');
    }
    if (typeof MLEngine !== 'undefined') {
      mlEngine = new MLEngine();
      console.log('[CognEn] Created MLEngine');
    }
    return !!(metricsAggregator && mlEngine);
  }

  // Analyze frame
  async function analyzeFrame() {
    if (!selfVideoElement || !faceApiLoaded || !ctx) return null;

    if (selfVideoElement.paused || selfVideoElement.videoWidth === 0) {
      selfVideoElement = findSelfVideo();
      if (!selfVideoElement) return null;
    }

    try {
      ctx.drawImage(selfVideoElement, 0, 0, canvas.width, canvas.height);

      const detections = await faceapi
        .detectAllFaces(canvas, new faceapi.TinyFaceDetectorOptions({ inputSize: 320 }))
        .withFaceLandmarks();

      if (detections && detections.length > 0) {
        const landmarks = detections[0].landmarks.positions;
        const timestamp = Date.now();

        if (metricsAggregator && mlEngine) {
          const metrics = metricsAggregator.calculateExtendedMetrics(
            landmarks, canvas.width, canvas.height, timestamp
          );
          const prediction = mlEngine.predict(metrics);
          return { prediction, metrics, hasFace: true };
        }
      }
      return { hasFace: false };
    } catch (e) {
      console.error('[CognEn] Analysis error:', e);
      return null;
    }
  }

  // Create widget
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
          z-index: 2147483647;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          background: linear-gradient(135deg, rgba(13, 17, 23, 0.98), rgba(22, 27, 34, 0.98));
          border: 2px solid rgba(0, 255, 255, 0.4);
          border-radius: 16px;
          padding: 16px 20px;
          min-width: 140px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 255, 255, 0.2);
        }
        .cognen-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
          color: #00FFFF;
          font-size: 12px;
          font-weight: 700;
        }
        .cognen-close {
          background: none;
          border: none;
          color: #8B949E;
          cursor: pointer;
          font-size: 16px;
          padding: 0 4px;
        }
        .cognen-close:hover { color: #F85149; }
        .cognen-score {
          font-size: 52px;
          font-weight: 800;
          text-align: center;
          color: #00FF99;
          line-height: 1;
          text-shadow: 0 0 20px rgba(0, 255, 153, 0.5);
        }
        .cognen-score.warning { color: #FFA657; text-shadow: 0 0 20px rgba(255, 166, 87, 0.5); }
        .cognen-score.danger { color: #F85149; text-shadow: 0 0 20px rgba(248, 81, 73, 0.5); }
        .cognen-state {
          font-size: 11px;
          text-align: center;
          color: #8B949E;
          text-transform: uppercase;
          letter-spacing: 1px;
          margin-top: 6px;
        }
        .cognen-stats {
          display: flex;
          justify-content: space-around;
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid rgba(255,255,255,0.1);
        }
        .cognen-stat { text-align: center; }
        .cognen-stat-value { font-size: 16px; font-weight: 700; color: #F0F6FC; }
        .cognen-stat-label { font-size: 9px; color: #6E7681; text-transform: uppercase; margin-top: 2px; }
      </style>
      <div class="cognen-header">
        <span>👁 CognEn</span>
        <button class="cognen-close" id="cognen-close" title="Stop tracking">✕</button>
      </div>
      <div class="cognen-score" id="cognen-score">--</div>
      <div class="cognen-state" id="cognen-state">Initializing...</div>
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

    document.getElementById('cognen-close').onclick = () => {
      stopTracking();
      storage.set({ meetCommand: 'STOPPED' });
    };
  }

  // Update widget
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
      stateEl.textContent = result ? 'No Face Detected' : 'Processing...';
    }
  }

  // Start tracking
  async function startTracking() {
    if (isTracking) return;

    console.log('[CognEn] Starting tracking...');

    // Find video
    selfVideoElement = findSelfVideo();
    if (!selfVideoElement) {
      console.log('[CognEn] No video found, retrying in 2s');
      setTimeout(startTracking, 2000);
      return;
    }
    console.log('[CognEn] Found video element');

    // Create canvas
    createCanvas();

    // Init ML
    const mlOk = initML();
    console.log('[CognEn] ML init:', mlOk);

    // Load face-api
    const faceOk = await loadFaceApi();
    console.log('[CognEn] Face-api ready:', faceOk);

    if (!faceOk) {
      console.error('[CognEn] Failed to load face-api');
      return;
    }

    // Create widget
    createWidget();

    isTracking = true;
    storage.set({ cameraRunning: true });

    // Analysis loop
    let frameCount = 0;
    analysisInterval = setInterval(async () => {
      if (!isTracking) return;

      const result = await analyzeFrame();
      frameCount++;

      updateWidget(result);

      // Also store in localStorage so popup can read
      if (result) {
        storage.set({
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

      if (frameCount % 30 === 0) {
        console.log('[CognEn] Frame', frameCount, '| Score:', result?.prediction?.rawScore || '--');
      }
    }, 200);

    console.log('[CognEn] Tracking started!');
  }

  // Stop tracking
  function stopTracking() {
    console.log('[CognEn] Stopping...');
    isTracking = false;

    if (analysisInterval) {
      clearInterval(analysisInterval);
      analysisInterval = null;
    }

    if (widget) {
      widget.remove();
      widget = null;
    }

    storage.set({ cameraRunning: false });
  }

  // Poll for commands from background
  function startCommandPolling() {
    commandPollInterval = setInterval(async () => {
      const data = await storage.get(['meetCommand', 'commandTimestamp']);

      if (data.commandTimestamp && data.commandTimestamp > lastCommandTimestamp) {
        lastCommandTimestamp = data.commandTimestamp;

        if (data.meetCommand === 'START' && !isTracking) {
          console.log('[CognEn] Received START command');
          startTracking();
        } else if (data.meetCommand === 'STOP' && isTracking) {
          console.log('[CognEn] Received STOP command');
          stopTracking();
        }
      }
    }, 500);
  }

  // Auto-start if we're on a Meet page
  console.log('[CognEn] Checking for START command...');

  // Start polling and also check immediately
  startCommandPolling();

  // Check for immediate start command
  setTimeout(async () => {
    const data = await storage.get(['meetCommand']);
    if (data.meetCommand === 'START') {
      console.log('[CognEn] Found START command, starting...');
      startTracking();
    }
  }, 500);

})();
