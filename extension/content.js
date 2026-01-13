/**
 * CognEn Extension - Meet Analyzer v2
 * Pure DOM APIs - NO innerHTML anywhere
 */

(function () {
  'use strict';

  // VERSION CHECK - if you see this in console, correct version is loaded
  console.log('[CognEn] Content script v2 loaded - NO innerHTML version');

  if (window.__cognEnV2) {
    console.log('[CognEn] Already running v2');
    return;
  }
  window.__cognEnV2 = true;

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

  function createCanvas() {
    if (canvas) return;
    canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 240;
    canvas.style.display = 'none';
    canvas.style.position = 'absolute';
    document.body.appendChild(canvas);
    ctx = canvas.getContext('2d', { willReadFrequently: true });
  }

  function findSelfVideo() {
    const videos = document.querySelectorAll('video');
    for (const video of videos) {
      if (video.videoWidth > 0 && !video.paused && video.readyState >= 2) {
        return video;
      }
    }
    return null;
  }

  async function loadModels() {
    if (modelsLoaded) return true;
    if (typeof faceapi === 'undefined') return false;

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

  function initML() {
    if (typeof FaceMetricsAggregator !== 'undefined' && !metricsAggregator) {
      metricsAggregator = new FaceMetricsAggregator();
    }
    if (typeof MLEngine !== 'undefined' && !mlEngine) {
      mlEngine = new MLEngine();
    }
    return !!(metricsAggregator && mlEngine);
  }

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

  // Create a styled element - NO innerHTML used
  function makeEl(tag, styleStr, textContent) {
    const elem = document.createElement(tag);
    if (styleStr) elem.style.cssText = styleStr;
    if (textContent !== undefined) elem.textContent = textContent;
    return elem;
  }

  function createWidget() {
    console.log('[CognEn] Creating widget with DOM APIs only');

    const existing = document.getElementById('cognen-widget-v2');
    if (existing) existing.remove();

    // Main container
    widget = makeEl('div',
      'position:fixed;bottom:20px;right:20px;z-index:2147483647;' +
      'font-family:-apple-system,BlinkMacSystemFont,sans-serif;' +
      'background:linear-gradient(135deg,rgba(13,17,23,0.98),rgba(22,27,34,0.98));' +
      'border:2px solid rgba(0,255,255,0.5);border-radius:16px;' +
      'padding:16px 20px;min-width:140px;' +
      'box-shadow:0 8px 32px rgba(0,0,0,0.6)'
    );
    widget.id = 'cognen-widget-v2';

    // Header row
    const header = makeEl('div', 'display:flex;justify-content:space-between;align-items:center;margin-bottom:12px');

    const logo = makeEl('span', 'color:#00FFFF;font-size:12px;font-weight:700', '👁 CognEn');
    header.appendChild(logo);

    const closeBtn = makeEl('button',
      'background:none;border:none;color:#8B949E;cursor:pointer;font-size:18px;padding:0',
      '✕'
    );
    closeBtn.addEventListener('click', function () {
      stopTracking();
      storage.set({ meetCommand: 'STOPPED' });
    });
    header.appendChild(closeBtn);

    widget.appendChild(header);

    // Score display
    scoreEl = makeEl('div',
      'font-size:52px;font-weight:800;text-align:center;color:#00FF99;line-height:1',
      '--'
    );
    widget.appendChild(scoreEl);

    // State label
    stateEl = makeEl('div',
      'font-size:11px;text-align:center;color:#8B949E;text-transform:uppercase;letter-spacing:1px;margin-top:6px',
      'Loading...'
    );
    widget.appendChild(stateEl);

    // Stats row
    const statsRow = makeEl('div',
      'display:flex;justify-content:space-around;margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.1)'
    );

    // Blink stat
    const blinkBox = makeEl('div', 'text-align:center');
    blinkEl = makeEl('div', 'font-size:16px;font-weight:700;color:#F0F6FC', '-');
    const blinkLabel = makeEl('div', 'font-size:9px;color:#6E7681;text-transform:uppercase;margin-top:2px', 'Blinks/m');
    blinkBox.appendChild(blinkEl);
    blinkBox.appendChild(blinkLabel);
    statsRow.appendChild(blinkBox);

    // Fatigue stat
    const fatigueBox = makeEl('div', 'text-align:center');
    fatigueEl = makeEl('div', 'font-size:16px;font-weight:700;color:#F0F6FC', '-');
    const fatigueLabel = makeEl('div', 'font-size:9px;color:#6E7681;text-transform:uppercase;margin-top:2px', 'Fatigue');
    fatigueBox.appendChild(fatigueEl);
    fatigueBox.appendChild(fatigueLabel);
    statsRow.appendChild(fatigueBox);

    widget.appendChild(statsRow);

    document.body.appendChild(widget);
    console.log('[CognEn] Widget created successfully');
  }

  function updateWidget(result) {
    if (!scoreEl) return;

    if (result && result.hasFace && result.prediction) {
      const score = result.prediction.rawScore;
      scoreEl.textContent = String(score);

      if (score < 40) {
        scoreEl.style.color = '#F85149';
      } else if (score < 70) {
        scoreEl.style.color = '#FFA657';
      } else {
        scoreEl.style.color = '#00FF99';
      }

      stateEl.textContent = result.prediction.attentionState || 'Unknown';

      if (result.metrics) {
        blinkEl.textContent = result.metrics.blinkRate ? result.metrics.blinkRate.toFixed(1) : '-';
        const f = result.metrics.fatigueIndex || 0;
        fatigueEl.textContent = f < 30 ? 'Low' : f < 60 ? 'Med' : 'High';
      }
    } else {
      scoreEl.textContent = '--';
      stateEl.textContent = result ? 'No Face' : 'Processing...';
    }
  }

  async function startTracking() {
    if (isTracking) return;
    console.log('[CognEn] Starting v2...');

    selfVideoElement = findSelfVideo();
    if (!selfVideoElement) {
      console.log('[CognEn] No video, retrying...');
      setTimeout(startTracking, 2000);
      return;
    }

    createCanvas();
    createWidget();
    initML();

    if (stateEl) stateEl.textContent = 'Loading models...';

    const ok = await loadModels();
    if (!ok) {
      if (stateEl) stateEl.textContent = 'Model load failed';
      return;
    }

    isTracking = true;
    storage.set({ cameraRunning: true });

    let fc = 0;
    analysisInterval = setInterval(async function () {
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
              fatigueIndex: result.metrics.fatigueIndex
            } : null
          }
        });
      }
    }, 200);

    console.log('[CognEn] Tracking active!');
  }

  function stopTracking() {
    isTracking = false;
    if (analysisInterval) { clearInterval(analysisInterval); analysisInterval = null; }
    if (widget) { widget.remove(); widget = null; }
    scoreEl = stateEl = blinkEl = fatigueEl = null;
    storage.set({ cameraRunning: false });
    console.log('[CognEn] Stopped');
  }

  function checkCommand() {
    const data = storage.get(['meetCommand']);
    if (data.meetCommand === 'START' && !isTracking) {
      console.log('[CognEn] Received START command');
      startTracking();
    } else if (data.meetCommand === 'STOP' && isTracking) {
      stopTracking();
    }
  }

  setInterval(checkCommand, 1000);
  setTimeout(checkCommand, 500);

  console.log('[CognEn] v2 Ready');
})();
