/**
 * CognEn Extension - Meet/Zoom/Teams analyser (MAIN world) v3
 *
 * Injected by background.js into the page's MAIN world after
 * lib/face-api.min.js and lib/ml-core.js. Pure DOM APIs (no innerHTML).
 *
 * Scoring: uses exactly the same ml-core objects as the standalone
 * (offscreen) mode - PrimaryFaceSelector -> FaceMetricsAggregator at 5 FPS
 * -> legacy-weight MLEngine.predict at 1 Hz (EMA alpha 0.15, thresholds
 * 70/40) - so both modes, and the web app, score identical input identically.
 *
 * Messaging: results go to the extension only through signed
 * window.postMessage messages verified by bridge.js (ISOLATED world), see
 * bridge.js for the protocol. The HMAC key arrives via executeScript args
 * and lives only in this closure.
 *
 * What remains spoofable / readable in the MAIN world (inherent - the page's
 * own scripts share this JS realm):
 *  - The page can READ every message posted here (scores, blink rate,
 *    fatigue), and read the widget DOM.
 *  - The page can call window.__cognen.content.stop() or replace face-api /
 *    ml-core globals, i.e. it can disable or corrupt local analysis.
 *  - A page that patched crypto.subtle, postMessage or Function.prototype
 *    BEFORE injection could capture the session key and forge messages.
 *  What it cannot do without such pre-injection tampering: forge or replay
 *  messages that bridge.js accepts, or reach chrome.runtime directly.
 */
(function (root) {
    'use strict';

    const ns = (root.__cognen = root.__cognen || {});
    if (ns.content) {
        // Already injected (second START): background reuses ns.content.start().
        return;
    }

    const SOURCE_MAIN = 'cognen-main';
    const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/';
    const WIDGET_ID = 'cognen-widget';

    // Capture the primitives used for signing at load time, before any later
    // page-side monkey-patching (does not help against patching done earlier).
    const subtle = root.crypto && root.crypto.subtle;
    const post = root.postMessage.bind(root);
    const encoder = new TextEncoder();

    function core() {
        return ns.mlCore;
    }

    let isTracking = false;
    let loopTimer = null;
    let loopGeneration = 0;
    let hmacKey = null;
    let seq = 0;

    let selfVideoElement = null;
    let manualVideo = null;
    let canvas = null;
    let ctx = null;
    let faceSelector = null;
    let metricsAggregator = null;
    let mlEngine = null;
    let latestMetrics = null;
    let lastPredictTime = 0;
    let modelsLoaded = false;

    let widget = null;
    let scoreEl = null;
    let stateEl = null;
    let blinkEl = null;
    let fatigueEl = null;

    // ==================== SIGNED MESSAGING ====================

    function hexToBytes(hex) {
        const out = new Uint8Array(hex.length / 2);
        for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.substr(i * 2, 2), 16);
        return out;
    }

    function bytesToHex(buf) {
        return Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, '0')).join('');
    }

    async function sendSigned(kind, payload) {
        if (!hmacKey || !subtle) return;
        seq += 1;
        const body = seq + '|' + kind + '|' + JSON.stringify(payload === undefined ? null : payload);
        const mac = bytesToHex(await subtle.sign('HMAC', hmacKey, encoder.encode(body)));
        post({ source: SOURCE_MAIN, kind, seq, payload: payload === undefined ? null : payload, mac }, root.location.origin);
    }

    // ==================== VIDEO SELECTION ====================

    /**
     * Self-view heuristic. Meet/Zoom/Teams render several <video> tiles; the
     * one to analyse is the local camera preview, ranked as:
     *   3 - srcObject is a MediaStream whose video track came from a local
     *       capture device (getSettings().deviceId is set only for
     *       getUserMedia/camera tracks; remote WebRTC tracks have none);
     *   2 - the element (or a close ancestor) is horizontally mirrored
     *       (computed transform with a negative x scale, e.g. scaleX(-1)),
     *       which all three apps do for the self-view only;
     *   0 - otherwise.
     * Ties (including all-zero) are broken by the largest visible area.
     * The user can override with the widget's "Switch video" button, which
     * cycles through the playing videos in ranked order.
     */
    function isPlaying(v) {
        return v.videoWidth > 0 && !v.paused && !v.ended && v.readyState >= 2;
    }

    function visibleArea(v) {
        const r = v.getBoundingClientRect();
        const w = Math.max(0, Math.min(r.right, root.innerWidth) - Math.max(r.left, 0));
        const h = Math.max(0, Math.min(r.bottom, root.innerHeight) - Math.max(r.top, 0));
        return w * h;
    }

    function hasLocalCameraTrack(v) {
        try {
            const stream = v.srcObject;
            if (!stream || typeof stream.getVideoTracks !== 'function') return false;
            return stream.getVideoTracks().some((t) => {
                if (t.readyState !== 'live') return false;
                const s = typeof t.getSettings === 'function' ? t.getSettings() : {};
                return typeof s.deviceId === 'string' && s.deviceId.length > 0;
            });
        } catch {
            return false;
        }
    }

    function isMirrored(v) {
        let el = v;
        for (let depth = 0; el && depth < 4; depth++, el = el.parentElement) {
            const t = root.getComputedStyle(el).transform;
            if (t && t !== 'none') {
                const m = t.match(/^matrix(3d)?\(([^)]+)\)$/);
                if (m) {
                    const a = parseFloat(m[2].split(',')[0]);
                    if (a < 0) return true;
                }
            }
        }
        return false;
    }

    function rankedVideos() {
        return Array.from(document.querySelectorAll('video'))
            .filter(isPlaying)
            .map((v) => ({
                v,
                rank: hasLocalCameraTrack(v) ? 3 : isMirrored(v) ? 2 : 0,
                area: visibleArea(v),
            }))
            .sort((a, b) => b.rank - a.rank || b.area - a.area)
            .map((c) => c.v);
    }

    function findSelfVideo() {
        if (manualVideo && manualVideo.isConnected && isPlaying(manualVideo)) return manualVideo;
        manualVideo = null;
        const list = rankedVideos();
        return list.length > 0 ? list[0] : null;
    }

    function highlight(v) {
        const prev = v.style.outline;
        v.style.outline = '3px solid #00FFFF';
        setTimeout(() => { v.style.outline = prev; }, 1500);
    }

    function cycleVideo() {
        const list = rankedVideos();
        if (list.length === 0) return;
        const idx = list.indexOf(selfVideoElement);
        manualVideo = list[(idx + 1) % list.length];
        selfVideoElement = manualVideo;
        resetScoring(); // a different person's stream must not share history
        highlight(manualVideo);
    }

    // ==================== ML ====================

    function createCanvas() {
        if (canvas) return;
        canvas = document.createElement('canvas');
        canvas.width = 320;
        canvas.height = 240;
        ctx = canvas.getContext('2d', { willReadFrequently: true });
    }

    async function loadModels() {
        if (modelsLoaded) return true;
        if (typeof root.faceapi === 'undefined') return false;
        try {
            await root.faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
            await root.faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
            modelsLoaded = true;
            return true;
        } catch (e) {
            console.error('[CognEn] Model error:', e);
            return false;
        }
    }

    function initML() {
        const c = core();
        if (!c) return false;
        if (!faceSelector) faceSelector = new c.PrimaryFaceSelector();
        if (!metricsAggregator) metricsAggregator = new c.FaceMetricsAggregator();
        if (!mlEngine) mlEngine = new c.MLEngine();
        return true;
    }

    function resetScoring() {
        if (faceSelector) faceSelector.reset();
        if (metricsAggregator) metricsAggregator.reset();
        if (mlEngine) mlEngine.reset();
        latestMetrics = null;
        lastPredictTime = 0;
    }

    /**
     * One 200 ms tick: detect + aggregate every tick; predict once per
     * PREDICT_INTERVAL_MS (1 s) on the latest metrics, as the web app does.
     * Returns a detection payload when a prediction (or a no-face status)
     * is due, otherwise null.
     */
    async function analyzeFrame() {
        if (!modelsLoaded || !ctx) return null;
        if (!selfVideoElement || !isPlaying(selfVideoElement)) {
            selfVideoElement = findSelfVideo();
            if (!selfVideoElement) return null;
        }

        ctx.drawImage(selfVideoElement, 0, 0, canvas.width, canvas.height);
        const det = await root.faceapi
            .detectAllFaces(canvas, new root.faceapi.TinyFaceDetectorOptions({ inputSize: 320 }))
            .withFaceLandmarks();

        const now = Date.now();
        const primary = faceSelector.select(
            (det || []).map((d) => ({ box: d.detection.box, detection: d })),
            canvas.width,
            canvas.height
        );

        if (primary) {
            latestMetrics = metricsAggregator.calculateExtendedMetrics(
                primary.detection.landmarks.positions,
                canvas.width,
                canvas.height,
                now
            );
        } else {
            latestMetrics = null;
        }

        if (now - lastPredictTime < core().PREDICT_INTERVAL_MS) return null;
        lastPredictTime = now;

        if (!latestMetrics) return { timestamp: now, hasFace: false, prediction: null, metrics: null };

        // Base metrics only - no detectedObjects - so MLEngine uses LEGACY weights.
        const prediction = mlEngine.predict(latestMetrics);
        return {
            timestamp: now,
            hasFace: true,
            prediction: {
                rawScore: prediction.rawScore,
                attentionState: prediction.attentionState,
                confidence: prediction.confidence,
            },
            metrics: {
                blinkRate: latestMetrics.blinkRate,
                fatigueIndex: latestMetrics.fatigueIndex,
                perclos: latestMetrics.perclos,
                ear: latestMetrics.ear,
            },
        };
    }

    // ==================== WIDGET ====================

    function makeEl(tag, styleStr, textContent) {
        const elem = document.createElement(tag);
        if (styleStr) elem.style.cssText = styleStr;
        if (textContent !== undefined) elem.textContent = textContent;
        return elem;
    }

    function statBox(label) {
        const box = makeEl('div', 'text-align:center');
        const value = makeEl('div', 'font-size:16px;font-weight:700;color:#F0F6FC', '-');
        box.appendChild(value);
        box.appendChild(makeEl('div', 'font-size:9px;color:#6E7681;text-transform:uppercase;margin-top:2px', label));
        return { box, value };
    }

    function createWidget() {
        const existing = document.getElementById(WIDGET_ID);
        if (existing) existing.remove();

        widget = makeEl('div',
            'position:fixed;bottom:20px;right:20px;z-index:2147483647;' +
            'font-family:-apple-system,BlinkMacSystemFont,sans-serif;' +
            'background:linear-gradient(135deg,rgba(13,17,23,0.98),rgba(22,27,34,0.98));' +
            'border:2px solid rgba(0,255,255,0.5);border-radius:16px;' +
            'padding:16px 20px;min-width:140px;box-shadow:0 8px 32px rgba(0,0,0,0.6)'
        );
        widget.id = WIDGET_ID;

        const header = makeEl('div', 'display:flex;justify-content:space-between;align-items:center;margin-bottom:12px');
        header.appendChild(makeEl('span', 'color:#00FFFF;font-size:12px;font-weight:700', 'CognEn'));
        const closeBtn = makeEl('button', 'background:none;border:none;color:#8B949E;cursor:pointer;font-size:18px;padding:0', '✕');
        closeBtn.title = 'Stop tracking';
        closeBtn.addEventListener('click', () => {
            sendSigned('STOPPED', null).finally(stop);
        });
        header.appendChild(closeBtn);
        widget.appendChild(header);

        scoreEl = makeEl('div', 'font-size:52px;font-weight:800;text-align:center;color:#00FF99;line-height:1', '--');
        widget.appendChild(scoreEl);
        stateEl = makeEl('div', 'font-size:11px;text-align:center;color:#8B949E;text-transform:uppercase;letter-spacing:1px;margin-top:6px', 'Loading...');
        widget.appendChild(stateEl);

        const statsRow = makeEl('div', 'display:flex;justify-content:space-around;margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.1)');
        const blink = statBox('Blinks/m');
        const fatigue = statBox('Fatigue');
        blinkEl = blink.value;
        fatigueEl = fatigue.value;
        statsRow.appendChild(blink.box);
        statsRow.appendChild(fatigue.box);
        widget.appendChild(statsRow);

        const switchBtn = makeEl('button',
            'display:block;margin:10px auto 0;background:none;border:1px solid rgba(139,148,158,0.4);' +
            'border-radius:6px;color:#8B949E;cursor:pointer;font-size:10px;padding:3px 8px',
            'Switch video');
        switchBtn.title = 'Analyse a different video tile (cycles through playing videos)';
        switchBtn.addEventListener('click', cycleVideo);
        widget.appendChild(switchBtn);

        document.body.appendChild(widget);
    }

    function updateWidget(result) {
        if (!scoreEl) return;
        if (result && result.hasFace && result.prediction) {
            const score = result.prediction.rawScore;
            const t = core().THRESHOLDS;
            scoreEl.textContent = String(score);
            scoreEl.style.color = score >= t.ATTENTIVE_MIN ? '#00FF99' : score >= t.DISTRACTED_MIN ? '#FFA657' : '#F85149';
            stateEl.textContent = result.prediction.attentionState || 'Unknown';
            const br = result.metrics ? result.metrics.blinkRate : null;
            blinkEl.textContent = typeof br === 'number' ? br.toFixed(1) : '0.0';
            const f = result.metrics && typeof result.metrics.fatigueIndex === 'number' ? result.metrics.fatigueIndex : 0;
            fatigueEl.textContent = f < 30 ? 'Low' : f < 60 ? 'Med' : 'High';
        } else {
            scoreEl.textContent = '--';
            scoreEl.style.color = '#F85149';
            stateEl.textContent = result ? 'No Face' : 'Processing...';
        }
    }

    // ==================== LOOP ====================

    function scheduleTick(generation, delay) {
        loopTimer = setTimeout(() => tick(generation), delay);
    }

    async function tick(generation) {
        if (!isTracking || generation !== loopGeneration) return;
        const started = Date.now();
        try {
            const result = await analyzeFrame();
            if (generation !== loopGeneration) return;
            if (result) {
                updateWidget(result);
                await sendSigned('DETECTION', result);
            }
        } catch (e) {
            console.error('[CognEn] Analysis error:', e);
        }
        if (isTracking && generation === loopGeneration) {
            const elapsed = Date.now() - started;
            scheduleTick(generation, Math.max(0, core().FRAME_INTERVAL_MS - elapsed));
        }
    }

    async function start(keyHex) {
        if (typeof keyHex !== 'string' || !/^[0-9a-f]{64}$/.test(keyHex) || !subtle) return false;
        hmacKey = await subtle.importKey('raw', hexToBytes(keyHex), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
        seq = 0;
        if (isTracking) {
            // New session on an already-running analyser: fresh history.
            resetScoring();
            return true;
        }

        if (!initML()) return false;
        createCanvas();
        createWidget();
        stateEl.textContent = 'Loading models...';

        const ok = await loadModels();
        if (!ok) {
            if (stateEl) stateEl.textContent = 'Model load failed';
            return false;
        }

        resetScoring();
        isTracking = true;
        loopGeneration += 1;
        stateEl.textContent = 'Looking for video...';
        scheduleTick(loopGeneration, 0);
        return true;
    }

    function stop() {
        isTracking = false;
        loopGeneration += 1;
        if (loopTimer) {
            clearTimeout(loopTimer);
            loopTimer = null;
        }
        if (widget) {
            widget.remove();
            widget = null;
        }
        scoreEl = stateEl = blinkEl = fatigueEl = null;
        selfVideoElement = null;
        manualVideo = null;
        hmacKey = null;
        resetScoring();
    }

    ns.content = Object.freeze({
        start,
        stop,
        isTracking: () => isTracking,
    });
})(window);
