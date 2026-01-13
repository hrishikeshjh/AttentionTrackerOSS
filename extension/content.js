/**
 * CognEn Extension - Meet Analyzer Content Script
 * Uses pure DOM APIs to avoid Trusted Types restrictions
 */

(function () {
  'use strict';

  console.log('[CognEn] Content script loaded');

  if (window.__cognEnInjected) return;
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
  let scoreEl, stateEl, blinkEl, fatigueEl;

  // Storage
  const storage = {
    set(data) {
      Object.entries(data).forEach(([k, v]) => {
        localStorage.setItem('cognen_' + k, JSON.stringify(v));
      });
    },
    get(keys) {
      const data = {};
      keys.forEach(k => {
        try {
          const v = localStorage.getItem('cognen_' + k);
          if (v) data[k] = JSON.parse(v);
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
    canvas.style.cssText = 'display:none;position:absolute;';
    document.body.appendChild(canvas);
    ctx = canvas.getContext('2d', { willReadFrequently: true });
  }

  // Find video
  function findSelfVideo() {
    const videos = document.querySelectorAll('video');
    for (const video of videos) {
      if (video.videoWidth > 0 && !video.paused && video.readyState >= 2) {
        const rect = video.getBoundingClientRect();
        if (rect.width > 50) return video;
      }
    }
    return null;
  }

  // Load models
  async function loadModels() {
    if (modelsLoaded) return true;
    if (typeof faceapi === 'undefined') {
      console.error('[CognEn] faceapi not found');
      return false;
    }

    try {
      const url = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/';
      await faceapi.nets.tinyFaceDetector.loadFromUri(url);
      await faceapi.nets.faceLandmark68Net.loadFromUri(url);
      modelsLoaded = true;
      console.log('[CognEn] Models loaded');
      return true;
    } catch (e) {
      console.error('[CognEn] Model error:', e);
      return false;
    }
  }

  // Init ML
  function initML() {
    if (typeof FaceMetricsAggregator !== 'undefined' && !metricsAggregator) {
      metricsAggregator = new FaceMetricsAggregator();
    }
    if (typeof MLEngine !== 'undefined' && !mlEngine) {
      mlEngine = new MLEngine();
    }
    return !!(metricsAggregator && mlEngine);
  }

  // Analyze
  async function analyzeFrame() {
    if (!selfVideoElement || !modelsLoaded || !ctx) return null;

    if (selfVideoElement.paused || selfVideoElement.videoWidth === 0) {
      selfVideoElement = findSelfVideo();
      if (!selfVideoElement) return null;
    }

    try {
      ctx.drawImage(selfVideoElement, 0, 0, canvas.width, canvas.height);
      const det = await faceapi.detectAllFaces(canvas, new faceapi.TinyFaceDetectorOptions({ inputSize: 320 })).withFaceLandmarks();

      if (det && det.length > 0 && metricsAggregator && mlEngine) {
        const landmarks = det[0].landmarks.positions;
        const metrics = metricsAggregator.calculateExtendedMetrics(landmarks, canvas.width, canvas.height, Date.now());
        const prediction = mlEngine.predict(metrics);
        return { prediction, metrics, hasFace: true };
      }
      return { hasFace: false };
    } catch (e) {
      return null;
    }
  }

  // Create element helper
  function el(tag, styles, text) {
    const e = document.createElement(tag);
    if (styles) e.style.cssText = styles;
    if (text) e.textContent = text;
    return e;
  }

  // Create widget using DOM APIs only (no innerHTML)
  function createWidget() {
    if (document.getElementById('cognen-widget')) return;

    // Main container
    widget = el('div', `
      position:fixed; bottom:20px; right:20px; z-index:2147483647;
      font-family:-apple-system,BlinkMacSystemFont,sans-serif;
      background:linear-gradient(135deg,rgba(13,17,23,0.98),rgba(22,27,34,0.98));
      border:2px solid rgba(0,255,255,0.5); border-radius:16px;
      padding:16px 20px; min-width:140px;
      box-shadow:0 8px 32px rgba(0,0,0,0.6),0 0 30px rgba(0,255,255,0.15);
    `);
    widget.id = 'cognen-widget';

    // Header
    const header = el('div', 'display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;');
    const logo = el('span', 'color:#00FFFF;font-size:12px;font-weight:700;', '👁 CognEn');
    const closeBtn = el('button', 'background:none;border:none;color:#8B949E;cursor:pointer;font-size:18px;padding:0 4px;', '✕');
    closeBtn.onmouseover = () => closeBtn.style.color = '#F85149';
    closeBtn.onmouseout = () => closeBtn.style.color = '#8B949E';
    closeBtn.onclick = () => { stopTracking(); storage.set({ meetCommand: 'STOPPED' }); };
    header.appendChild(logo);
    header.appendChild(closeBtn);

    // Score
    scoreEl = el('div', `
      font-size:52px; font-weight:800; text-align:center;
      color:#00FF99; line-height:1;
      text-shadow:0 0 20px rgba(0,255,153,0.5);
    `, '--');

    // State
    stateEl = el('div', `
      font-size:11px; text-align:center; color:#8B949E;
      text-transform:uppercase; letter-spacing:1px; margin-top:6px;
    `, 'Loading...');

    // Stats container
    const stats = el('div', `
      display:flex; justify-content:space-around;
      margin-top:12px; padding-top:12px;
      border-top:1px solid rgba(255,255,255,0.1);
    `);

    // Blink stat
    const blinkStat = el('div', 'text-align:center;');
    blinkEl = el('div', 'font-size:16px;font-weight:700;color:#F0F6FC;', '-');
    const blinkLbl = el('div', 'font-size:9px;color:#6E7681;text-transform:uppercase;margin-top:2px;', 'Blinks/m');
    blinkStat.appendChild(blinkEl);
    blinkStat.appendChild(blinkLbl);

    // Fatigue stat
    const fatigueStat = el('div', 'text-align:center;');
    fatigueEl = el('div', 'font-size:16px;font-weight:700;color:#F0F6FC;', '-');
    const fatigueLbl = el('div', 'font-size:9px;color:#6E7681;text-transform:uppercase;margin-top:2px;', 'Fatigue');
    fatigueStat.appendChild(fatigueEl);
    fatigueStat.appendChild(fatigueLbl);

    stats.appendChild(blinkStat);
    stats.appendChild(fatigueStat);

    // Assemble
    widget.appendChild(header);
    widget.appendChild(scoreEl);
    widget.appendChild(stateEl);
    widget.appendChild(stats);

    document.body.appendChild(widget);
  }

  // Update widget
  function updateWidget(result) {
    if (!scoreEl) return;

    if (result && result.hasFace && result.prediction) {
      const score = result.prediction.rawScore;
      scoreEl.textContent = score;

      if (score < 40) {
        scoreEl.style.color = '#F85149';
        scoreEl.style.textShadow = '0 0 20px rgba(248,81,73,0.5)';
      } else if (score < 70) {
        scoreEl.style.color = '#FFA657';
        scoreEl.style.textShadow = '0 0 20px rgba(255,166,87,0.5)';
      } else {
        scoreEl.style.color = '#00FF99';
        scoreEl.style.textShadow = '0 0 20px rgba(0,255,153,0.5)';
      }

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

  // Start
  async function startTracking() {
    if (isTracking) return;
    console.log('[CognEn] Starting...');

    selfVideoElement = findSelfVideo();
    if (!selfVideoElement) {
      console.log('[CognEn] No video, retry in 2s');
      setTimeout(startTracking, 2000);
      return;
    }

    createCanvas();
    createWidget();
    initML();

    stateEl.textContent = 'Loading models...';
    const ok = await loadModels();
    if (!ok) {
      stateEl.textContent = 'Model load failed';
      return;
    }

    isTracking = true;
    storage.set({ cameraRunning: true });

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
    }, 200);

    console.log('[CognEn] Active!');
  }

  // Stop
  function stopTracking() {
    isTracking = false;
    if (analysisInterval) { clearInterval(analysisInterval); analysisInterval = null; }
    if (widget) { widget.remove(); widget = null; }
    storage.set({ cameraRunning: false });
    console.log('[CognEn] Stopped');
  }

  // Command polling
  function checkCommand() {
    const data = storage.get(['meetCommand']);
    if (data.meetCommand === 'START' && !isTracking) startTracking();
    else if (data.meetCommand === 'STOP' && isTracking) stopTracking();
  }

  setInterval(checkCommand, 1000);
  setTimeout(checkCommand, 500);

  console.log('[CognEn] Ready');
})();
