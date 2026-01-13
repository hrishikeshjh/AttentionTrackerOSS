/**
 * CognEn Extension - Meet Analyzer Content Script
 * Runs in MAIN world - face-api.js is pre-injected by background
 * No dynamic loading needed
 */

(function () {
  'use strict';

  console.log('[CognEn] Content script loaded');

  // Prevent duplicate
  if (window.__cognEnInjected) {
    console.log('[CognEn] Already injected');
    return;
  }
  window.__cognEnInjected = true;

  // State
  let isTracking = false;
  let selfVideoElement = null;
  let analysisInterval = null;
  let canvas = null;
  let ctx = null;
  let metricsAggregator = null;
  let mlEngine = null;
  let widget = null;
  let modelsLoaded = false;

  // Storage wrapper using localStorage
  const storage = {
    set(data) {
      Object.entries(data).forEach(([key, val]) => {
        localStorage.setItem('cognen_' + key, JSON.stringify(val));
      });
    },
    get(keys) {
      const data = {};
      keys.forEach(key => {
        try {
          const val = localStorage.getItem('cognen_' + key);
          if (val) data[key] = JSON.parse(val);
        } catch (e) { }
      });
      return data;
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
    ctx = canvas.getContext('2d', { willReadFrequently: true });
  }

  // Find self-view video
  function findSelfVideo() {
    const videos = document.querySelectorAll('video');
    console.log('[CognEn] Found', videos.length, 'videos');

    for (const video of videos) {
      if (video.videoWidth > 0 && !video.paused && video.readyState >= 2) {
        const rect = video.getBoundingClientRect();
        if (rect.width > 50 && rect.height > 50) {
          console.log('[CognEn] Using video:', rect.width, 'x', rect.height);
          return video;
        }
      }
    }
    return null;
  }

  // Load face-api models (face-api.js already injected)
  async function loadModels() {
    if (modelsLoaded) return true;

    console.log('[CognEn] Checking faceapi...', typeof faceapi);

    if (typeof faceapi === 'undefined') {
      console.error('[CognEn] faceapi not found!');
      return false;
    }

    try {
      console.log('[CognEn] Loading models from CDN...');
      const modelUrl = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/';

      // These are data fetches, not script loads - should work
      await faceapi.nets.tinyFaceDetector.loadFromUri(modelUrl);
      await faceapi.nets.faceLandmark68Net.loadFromUri(modelUrl);

      console.log('[CognEn] Models loaded!');
      modelsLoaded = true;
      return true;
    } catch (e) {
      console.error('[CognEn] Model load error:', e);
      return false;
    }
  }

  // Init ML components
  function initML() {
    console.log('[CognEn] FaceMetricsAggregator:', typeof FaceMetricsAggregator);
    console.log('[CognEn] MLEngine:', typeof MLEngine);

    if (typeof FaceMetricsAggregator !== 'undefined' && !metricsAggregator) {
      metricsAggregator = new FaceMetricsAggregator();
    }
    if (typeof MLEngine !== 'undefined' && !mlEngine) {
      mlEngine = new MLEngine();
    }
    return !!(metricsAggregator && mlEngine);
  }

  // Analyze frame
  async function analyzeFrame() {
    if (!selfVideoElement || !modelsLoaded || !ctx) return null;

    // Re-check video
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

  // Create widget
  function createWidget() {
    const existing = document.getElementById('cognen-widget');
    if (existing) existing.remove();

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
          border: 2px solid rgba(0, 255, 255, 0.5);
          border-radius: 16px;
          padding: 16px 20px;
          min-width: 140px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), 0 0 30px rgba(0, 255, 255, 0.15);
        }
        .cg-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; color: #00FFFF; font-size: 12px; font-weight: 700; }
        .cg-close { background: none; border: none; color: #8B949E; cursor: pointer; font-size: 18px; padding: 0 4px; }
        .cg-close:hover { color: #F85149; }
        .cg-score { font-size: 52px; font-weight: 800; text-align: center; color: #00FF99; line-height: 1; text-shadow: 0 0 20px rgba(0, 255, 153, 0.5); }
        .cg-score.warning { color: #FFA657; text-shadow: 0 0 20px rgba(255, 166, 87, 0.5); }
        .cg-score.danger { color: #F85149; text-shadow: 0 0 20px rgba(248, 81, 73, 0.5); }
        .cg-state { font-size: 11px; text-align: center; color: #8B949E; text-transform: uppercase; letter-spacing: 1px; margin-top: 6px; }
        .cg-stats { display: flex; justify-content: space-around; margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); }
        .cg-stat { text-align: center; }
        .cg-stat-val { font-size: 16px; font-weight: 700; color: #F0F6FC; }
        .cg-stat-lbl { font-size: 9px; color: #6E7681; text-transform: uppercase; margin-top: 2px; }
      </style>
      <div class="cg-header">
        <span>👁 CognEn</span>
        <button class="cg-close" id="cg-close" title="Stop">✕</button>
      </div>
      <div class="cg-score" id="cg-score">--</div>
      <div class="cg-state" id="cg-state">Loading...</div>
      <div class="cg-stats">
        <div class="cg-stat"><div class="cg-stat-val" id="cg-blink">-</div><div class="cg-stat-lbl">Blinks/m</div></div>
        <div class="cg-stat"><div class="cg-stat-val" id="cg-fatigue">-</div><div class="cg-stat-lbl">Fatigue</div></div>
      </div>
    `;

    document.body.appendChild(widget);

    document.getElementById('cg-close').onclick = () => {
      stopTracking();
      storage.set({ meetCommand: 'STOPPED' });
    };
  }

  // Update widget
  function updateWidget(result) {
    const scoreEl = document.getElementById('cg-score');
    const stateEl = document.getElementById('cg-state');
    const blinkEl = document.getElementById('cg-blink');
    const fatigueEl = document.getElementById('cg-fatigue');

    if (!scoreEl) return;

    if (result && result.hasFace && result.prediction) {
      const score = result.prediction.rawScore;
      scoreEl.textContent = score;
      scoreEl.className = 'cg-score';
      if (score < 40) scoreEl.classList.add('danger');
      else if (score < 70) scoreEl.classList.add('warning');

      stateEl.textContent = result.prediction.attentionState;

      if (result.metrics) {
        blinkEl.textContent = result.metrics.blinkRate?.toFixed(1) || '-';
        const f = result.metrics.fatigueIndex || 0;
        fatigueEl.textContent = f < 30 ? 'Low' : f < 60 ? 'Med' : 'High';
      }
    } else {
      scoreEl.textContent = '--';
      stateEl.textContent = result ? 'No Face' : 'Processing...';
    }
  }

  // Start tracking
  async function startTracking() {
    if (isTracking) return;
    console.log('[CognEn] Starting...');

    // Find video
    selfVideoElement = findSelfVideo();
    if (!selfVideoElement) {
      console.log('[CognEn] No video, retry in 2s');
      setTimeout(startTracking, 2000);
      return;
    }

    // Create UI
    createCanvas();
    createWidget();

    // Init ML
    initML();

    // Load models
    const doc = document.getElementById('cg-state');
    if (doc) doc.textContent = 'Loading models...';

    const ok = await loadModels();
    if (!ok) {
      if (doc) doc.textContent = 'Model load failed';
      return;
    }

    isTracking = true;
    storage.set({ cameraRunning: true });

    // Analysis loop
    let fc = 0;
    analysisInterval = setInterval(async () => {
      if (!isTracking) return;

      const result = await analyzeFrame();
      fc++;

      updateWidget(result);

      if (result) {
        storage.set({
          lastDetection: {
            timestamp: Date.now(),
            hasFace: result.hasFace,
            frameCount: fc,
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

      if (fc % 50 === 0) {
        console.log('[CognEn] Frame', fc, 'Score:', result?.prediction?.rawScore || '--');
      }
    }, 200);

    console.log('[CognEn] Tracking active!');
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

  // Check for start command
  function checkCommand() {
    const data = storage.get(['meetCommand']);
    if (data.meetCommand === 'START' && !isTracking) {
      console.log('[CognEn] START command found');
      startTracking();
    } else if (data.meetCommand === 'STOP' && isTracking) {
      stopTracking();
    }
  }

  // Poll for commands
  setInterval(checkCommand, 1000);

  // Check immediately
  setTimeout(checkCommand, 500);

  console.log('[CognEn] Ready and waiting for command');
})();
