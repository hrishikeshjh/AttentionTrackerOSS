/**
 * CognEn Extension - Meet/Zoom/Teams message bridge (ISOLATED world).
 *
 * Meet mode runs face-api + ml-core + content.js in the page's MAIN world
 * (it needs the page's <video> elements and the page's WebGL context). The
 * MAIN world is shared with the page's own scripts, so anything it exposes
 * there (globals, localStorage, DOM events, postMessage traffic) is visible
 * to - and writable by - the page. This bridge is the only path from the
 * MAIN-world analyser back to the extension:
 *
 *   content.js (MAIN)  --window.postMessage({..., mac})-->  bridge.js (ISOLATED)
 *   bridge.js          --chrome.runtime.sendMessage-->      background.js
 *
 * Authentication: for every session the background generates a random
 * 256-bit key and hands it to both worlds through chrome.scripting
 * executeScript `args` (never through the DOM, storage, or postMessage, so
 * page listeners never see it). The MAIN world signs each message with
 * HMAC-SHA-256 over `${seq}|${kind}|${JSON payload}`; this bridge verifies
 * the MAC with its own (page-unreachable) crypto.subtle, rejects replays
 * (seq must strictly increase) and sanitises the payload before relaying it.
 * A page script that merely observes or posts window messages therefore
 * cannot inject fake scores. Residual risk is documented in content.js.
 *
 * Re-injection safe: guarded by a sentinel on the isolated-world window.
 */
(function () {
    'use strict';

    if (window.__cognenBridge) return;

    const SOURCE_MAIN = 'cognen-main';
    const STATES = ['Attentive', 'Distracted', 'Multitasking', 'Idle'];

    let hmacKey = null; // CryptoKey for the current session
    let sessionId = null;
    let lastSeq = -1;

    function hexToBytes(hex) {
        const out = new Uint8Array(hex.length / 2);
        for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.substr(i * 2, 2), 16);
        return out;
    }

    function finiteOrNull(v, min, max) {
        if (typeof v !== 'number' || !Number.isFinite(v)) return null;
        return Math.max(min, Math.min(max, v));
    }

    /** Whitelist + range-check a detection payload; returns null if malformed. */
    function sanitizeDetection(p) {
        if (!p || typeof p !== 'object') return null;
        const timestamp = finiteOrNull(p.timestamp, 0, Number.MAX_SAFE_INTEGER);
        if (timestamp === null) return null;
        const out = { timestamp, hasFace: p.hasFace === true, prediction: null, metrics: null };
        if (p.prediction && typeof p.prediction === 'object') {
            const rawScore = finiteOrNull(p.prediction.rawScore, 0, 100);
            const state = STATES.includes(p.prediction.attentionState) ? p.prediction.attentionState : null;
            if (rawScore !== null && state !== null) {
                out.prediction = {
                    rawScore: Math.round(rawScore),
                    attentionState: state,
                    confidence: finiteOrNull(p.prediction.confidence, 0, 1),
                };
            }
        }
        if (p.metrics && typeof p.metrics === 'object') {
            out.metrics = {
                blinkRate: finiteOrNull(p.metrics.blinkRate, 0, 1000),
                fatigueIndex: finiteOrNull(p.metrics.fatigueIndex, 0, 100),
                perclos: finiteOrNull(p.metrics.perclos, 0, 100),
                ear: finiteOrNull(p.metrics.ear, 0, 2),
            };
        }
        return out;
    }

    async function verify(data) {
        if (!hmacKey || typeof data.mac !== 'string' || !/^[0-9a-f]{64}$/.test(data.mac)) return false;
        if (!Number.isInteger(data.seq) || data.seq <= lastSeq) return false;
        const body = data.seq + '|' + data.kind + '|' + JSON.stringify(data.payload === undefined ? null : data.payload);
        try {
            return await crypto.subtle.verify(
                'HMAC',
                hmacKey,
                hexToBytes(data.mac),
                new TextEncoder().encode(body)
            );
        } catch {
            return false;
        }
    }

    function send(message) {
        try {
            chrome.runtime.sendMessage(message, () => {
                // Swallow "receiving end does not exist" while the worker restarts.
                void chrome.runtime.lastError;
            });
        } catch {
            // Extension context invalidated (extension reloaded) - stop relaying.
            hmacKey = null;
        }
    }

    window.addEventListener('message', async (event) => {
        if (event.source !== window || event.origin !== window.location.origin) return;
        const data = event.data;
        if (!data || typeof data !== 'object' || data.source !== SOURCE_MAIN) return;
        if (data.kind !== 'DETECTION' && data.kind !== 'STOPPED') return;

        const seq = data.seq;
        if (!(await verify(data))) return;
        if (seq <= lastSeq) return; // re-check after the async verify (ordering)
        lastSeq = seq;

        if (data.kind === 'DETECTION') {
            const detection = sanitizeDetection(data.payload);
            if (detection) send({ action: 'MEET_DETECTION', sessionId, detection });
        } else {
            send({ action: 'MEET_STOPPED', sessionId });
        }
    });

    window.__cognenBridge = {
        /** Called by background via executeScript (ISOLATED world) with the session key. */
        async arm(keyHex, id) {
            if (typeof keyHex !== 'string' || !/^[0-9a-f]{64}$/.test(keyHex)) return false;
            hmacKey = await crypto.subtle.importKey(
                'raw',
                hexToBytes(keyHex),
                { name: 'HMAC', hash: 'SHA-256' },
                false,
                ['verify']
            );
            sessionId = id;
            lastSeq = -1;
            return true;
        },
        disarm() {
            hmacKey = null;
            sessionId = null;
            lastSeq = -1;
        },
    };
})();
