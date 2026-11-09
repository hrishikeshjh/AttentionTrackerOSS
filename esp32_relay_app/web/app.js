/**
 * CognEn Relay Hub — Main Application Controller
 * Orchestrates:
 * 1. Web MediaDevices Camera stream
 * 2. MediaPipe FaceLandmarker Landmark Extraction (via ESM)
 * 3. CognEn Mathematical Core (EAR, MAR, Head Pose, PERCLOS, OLS, EMA)
 * 4. Context-Gate Rule Evaluation per Operating Mode
 * 5. Hardware Bridge Dispatch (USB Serial / Wi-Fi HTTP)
 */

import { FilesetResolver, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";
import { CognEnMathCore } from "./math-engine.js";
import { SYSTEM_MODES } from "./modes.js";
import { HardwareBridge } from "./hardware-bridge.js";

// DOM Elements
const videoEl = document.getElementById("videoElement");
const canvasEl = document.getElementById("overlayCanvas");
const ctx = canvasEl.getContext("2d");

const hudStateDot = document.getElementById("hudStateDot");
const hudStateLabel = document.getElementById("hudStateLabel");
const hudScoreVal = document.getElementById("hudScoreVal");
const hudFps = document.getElementById("hudFps");
const hudLatency = document.getElementById("hudLatency");
const hudFatigueSlope = document.getElementById("hudFatigueSlope");

const valEar = document.getElementById("valEar");
const valPerclos = document.getElementById("valPerclos");
const valPose = document.getElementById("valPose");
const valFidget = document.getElementById("valFidget");
const subFidget = document.getElementById("subFidget");

const relayCardsContainer = document.getElementById("relayCardsContainer");
const currentModeBadge = document.getElementById("currentModeBadge");
const valTransport = document.getElementById("valTransport");
const valReason = document.getElementById("valReason");
const consoleBox = document.getElementById("consoleBox");

const btnSerial = document.getElementById("btnSerial");
const labelSerial = document.getElementById("labelSerial");
const dotSerial = document.getElementById("dotSerial");
const btnWifi = document.getElementById("btnWifi");
const inputIp = document.getElementById("inputIp");
const btnCycleAll = document.getElementById("btnCycleAll");

// Core Instances
const mathCore = new CognEnMathCore();
let activeMode = "proto";
let faceLandmarker = null;
let lastVideoTime = -1;
let frameCount = 0;
let lastFpsTime = performance.now();

// Simulated Objects for quick test injection
const detectedObjects = { phone: false, book: false };
let manualDrowsyOverride = false;

// Manual Relay Overrides (null = auto, 0 = force OFF, 1 = force ON)
let manualOverrides = [null, null, null, null];
let manualOverrideTimeouts = [null, null, null, null];

function setManualOverride(idx, state) {
  manualOverrides[idx] = state;
  if (manualOverrideTimeouts[idx]) clearTimeout(manualOverrideTimeouts[idx]);
  manualOverrideTimeouts[idx] = setTimeout(() => {
    manualOverrides[idx] = null;
  }, 15000);
}

// Console logger
function logToConsole(message, type = "normal") {
  const span = document.createElement("span");
  span.className = `console-entry ${type}`;
  span.textContent = message;
  consoleBox.appendChild(span);
  consoleBox.scrollTop = consoleBox.scrollHeight;
}

const hardwareBridge = new HardwareBridge((msg, type) => {
  logToConsole(msg, type);
  if (hardwareBridge.transportMode === "bridge" || hardwareBridge.isBridgeActive) {
    valTransport.textContent = `Native USB Bridge (${hardwareBridge.bridgePort})`;
    labelSerial.textContent = "Hardware Connected";
    btnSerial.classList.add("connected");
    dotSerial.className = "dot green";
  } else if (hardwareBridge.transportMode === "serial") {
    valTransport.textContent = "Wired (Web Serial)";
  } else if (hardwareBridge.transportMode === "wifi") {
    valTransport.textContent = `Wireless (Wi-Fi: ${hardwareBridge.esp32Ip})`;
  } else {
    valTransport.textContent = "Simulation (Loopback)";
  }
});

// ================= RENDER RELAY CARDS =================
function renderRelayCards() {
  const modeSpec = SYSTEM_MODES[activeMode];
  currentModeBadge.textContent = modeSpec.badge;
  relayCardsContainer.innerHTML = "";

  modeSpec.channels.forEach((ch, idx) => {
    const card = document.createElement("div");
    card.className = "relay-card";
    card.id = `relayCard_${idx}`;

    card.innerHTML = `
      <div class="relay-meta">
        <span class="relay-ch-label">CH ${ch.id} · ${ch.pin}</span>
        <span class="relay-name">${ch.name}</span>
        <span class="relay-desc">${ch.desc}</span>
      </div>
      <div class="relay-action-area">
        <label class="apple-switch">
          <input type="checkbox" id="switchCh_${idx}">
          <span class="switch-slider"></span>
        </label>
      </div>
    `;

    const switchInput = card.querySelector(`#switchCh_${idx}`);
    switchInput.addEventListener("change", (e) => {
      const state = e.target.checked ? 1 : 0;
      setManualOverride(idx, state);
      hardwareBridge.setChannelManual(idx, state);
      updateRelayCardVisuals(idx, e.target.checked);
      logToConsole(`[MANUAL] CH ${idx + 1} locked ${state ? "ON" : "OFF"} for 15s.`, "normal");
    });

    relayCardsContainer.appendChild(card);
  });
}

function updateRelayCardVisuals(chIndex, isActive, isEmergency = false) {
  const card = document.getElementById(`relayCard_${chIndex}`);
  const input = document.getElementById(`switchCh_${chIndex}`);
  if (card && input) {
    if (manualOverrides[chIndex] === null) {
      input.checked = isActive;
    }
    if (isActive) {
      card.classList.add("active");
      if (isEmergency) card.classList.add("emergency");
    } else {
      card.classList.remove("active", "emergency");
    }
  }
}

// ================= CAMERA & MEDIAPIPE INITIALIZATION =================
async function initVisionPipeline() {
  hudStateLabel.textContent = "LOADING MEDIAPIPE VISION CORE...";

  try {
    let vision;
    try {
      vision = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
    } catch {
      vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    }

    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "/models/face_landmarker.task",
          delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode: "VIDEO",
        numFaces: 1
      });
    } catch (gpuErr) {
      logToConsole(`[AI] GPU delegate fallback to CPU: ${gpuErr.message}`, "sim");
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "CPU"
        },
        outputFaceBlendshapes: true,
        runningMode: "VIDEO",
        numFaces: 1
      });
    }

    logToConsole("[AI] MediaPipe FaceLandmarker loaded successfully.", "success");
    hudStateLabel.textContent = "VISION CORE ACTIVE";
  } catch (err) {
    logToConsole(`[AI ERROR] Could not initialize FaceLandmarker: ${err.message}`, "error");
  }

  // Camera stream
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, frameRate: { ideal: 15 } },
      audio: false
    });
    videoEl.srcObject = stream;
    await new Promise(res => videoEl.onloadedmetadata = res);
    canvasEl.width = videoEl.videoWidth || 640;
    canvasEl.height = videoEl.videoHeight || 480;
    requestAnimationFrame(processFrame);
  } catch (err) {
    canvasEl.width = 640;
    canvasEl.height = 480;
    logToConsole(`[CAMERA] ${err.message}. Starting Interactive Simulation Mode.`, "sim");
    hudStateLabel.textContent = "SYNTHETIC SIMULATION (NO CAM)";
    setInterval(processSyntheticFrame, 100);
  }
}

// ================= PROCESS RUNTIME FRAME =================
function processFrame() {
  const startTime = performance.now();

  if (videoEl.readyState >= 2) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    let faceData = null;

    if (faceLandmarker && videoEl.currentTime !== lastVideoTime) {
      lastVideoTime = videoEl.currentTime;
      const results = faceLandmarker.detectForVideo(videoEl, startTime);

      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const lm = results.faceLandmarks[0];

        // Draw minimalist bounding brackets and eye landmarks
        drawMinimalLandmarks(lm);

        // Map canonical indices
        faceData = {
          leftEye: [lm[33], lm[160], lm[158], lm[133], lm[153], lm[144]],
          rightEye: [lm[362], lm[385], lm[387], lm[263], lm[373], lm[380]],
          topLip: lm[13],
          bottomLip: lm[14],
          mouthLeft: lm[61],
          mouthRight: lm[291],
          nose: lm[1],
          chin: lm[152]
        };
      }
    }

    // Manual test injection overrides
    if (manualDrowsyOverride && faceData) {
      faceData.leftEye = [{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0}];
      faceData.rightEye = [{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0}];
    }

    // 1. Evaluate Math Core
    const result = mathCore.process(faceData, detectedObjects, startTime);

    // 2. Evaluate Active Mode Relays
    const modeEvaluator = SYSTEM_MODES[activeMode];
    const { channels, reason } = modeEvaluator.evaluate(result);

    // Apply manual overrides
    for (let i = 0; i < 4; i++) {
      if (manualOverrides[i] !== null) {
        channels[i] = manualOverrides[i];
      }
    }

    // 3. Dispatch to Hardware
    hardwareBridge.dispatchRelays(channels);

    // 4. Update Visuals
    updateDashboardUI(result, channels, reason, performance.now() - startTime);
  }

  // FPS Counter
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    hudFps.textContent = `${frameCount} FPS`;
    frameCount = 0;
    lastFpsTime = now;
  }

  requestAnimationFrame(processFrame);
}

// ================= SYNTHETIC DEMO SIMULATOR =================
let syntheticAngle = 0;
let syntheticBlinkTimer = 0;

function processSyntheticFrame() {
  const startTime = performance.now();
  syntheticAngle += 0.05;
  syntheticBlinkTimer++;

  // Subtle biological micro-movements
  const jitterX = Math.sin(syntheticAngle * 0.7) * 0.005;
  const jitterY = Math.cos(syntheticAngle * 0.5) * 0.004;

  const isBlinking = (syntheticBlinkTimer % 35 === 0 || syntheticBlinkTimer % 35 === 1);
  const eyeOpenY = (manualDrowsyOverride || isBlinking) ? 0.002 : 0.02;

  let pitchOffset = 0;
  let yawOffset = Math.sin(syntheticAngle * 0.3) * 0.02;

  if (detectedObjects.phone) {
    pitchOffset = 0.08; // Head down looking at phone
  } else if (detectedObjects.book) {
    pitchOffset = 0.06; // Head down reading
  }

  const leftEyeCenter = { x: 0.38 + jitterX + yawOffset, y: 0.40 + jitterY + pitchOffset };
  const rightEyeCenter = { x: 0.62 + jitterX + yawOffset, y: 0.40 + jitterY + pitchOffset };

  const fakeFace = {
    leftEye: [
      { x: leftEyeCenter.x - 0.04, y: leftEyeCenter.y },
      { x: leftEyeCenter.x - 0.02, y: leftEyeCenter.y - eyeOpenY },
      { x: leftEyeCenter.x + 0.02, y: leftEyeCenter.y - eyeOpenY },
      { x: leftEyeCenter.x + 0.04, y: leftEyeCenter.y },
      { x: leftEyeCenter.x + 0.02, y: leftEyeCenter.y + eyeOpenY },
      { x: leftEyeCenter.x - 0.02, y: leftEyeCenter.y + eyeOpenY }
    ],
    rightEye: [
      { x: rightEyeCenter.x - 0.04, y: rightEyeCenter.y },
      { x: rightEyeCenter.x - 0.02, y: rightEyeCenter.y - eyeOpenY },
      { x: rightEyeCenter.x + 0.02, y: rightEyeCenter.y - eyeOpenY },
      { x: rightEyeCenter.x + 0.04, y: rightEyeCenter.y },
      { x: rightEyeCenter.x + 0.02, y: rightEyeCenter.y + eyeOpenY },
      { x: rightEyeCenter.x - 0.02, y: rightEyeCenter.y + eyeOpenY }
    ],
    topLip: { x: 0.50 + jitterX, y: 0.62 + jitterY + pitchOffset },
    bottomLip: { x: 0.50 + jitterX, y: 0.67 + jitterY + pitchOffset },
    mouthLeft: { x: 0.45 + jitterX, y: 0.64 + jitterY + pitchOffset },
    mouthRight: { x: 0.55 + jitterX, y: 0.64 + jitterY + pitchOffset },
    nose: { x: 0.50 + jitterX + yawOffset, y: 0.52 + jitterY + pitchOffset * 0.8 },
    chin: { x: 0.50 + jitterX, y: 0.76 + jitterY + pitchOffset }
  };

  if (manualDrowsyOverride) {
    fakeFace.leftEye = [{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0}];
    fakeFace.rightEye = [{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0}];
  }

  // 1. Evaluate Math Core
  const fakeResult = mathCore.process(fakeFace, detectedObjects, startTime);

  // 2. Evaluate Active Mode Relays
  const modeEvaluator = SYSTEM_MODES[activeMode];
  const { channels, reason } = modeEvaluator.evaluate(fakeResult);

  // Apply manual overrides
  for (let i = 0; i < 4; i++) {
    if (manualOverrides[i] !== null) {
      channels[i] = manualOverrides[i];
    }
  }

  // 3. Dispatch to Hardware Bridge
  hardwareBridge.dispatchRelays(channels);

  // 4. Render Dynamic Synthetic Wireframe
  drawSyntheticCanvas(fakeFace, fakeResult);

  // 5. Update UI
  updateDashboardUI(fakeResult, channels, reason, performance.now() - startTime);

  // Maintain FPS display in synthetic simulation
  hudFps.textContent = "15 FPS (Sim)";
}

// Render dynamic HUD avatar on canvas when in simulation mode
function drawSyntheticCanvas(face, result) {
  if (!canvasEl.width || canvasEl.width < 100) {
    canvasEl.width = 640;
    canvasEl.height = 480;
  }
  const w = canvasEl.width;
  const h = canvasEl.height;

  ctx.clearRect(0, 0, w, h);

  // Subtle dark engineering grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  ctx.lineWidth = 1;
  const step = 40;
  for (let x = 0; x < w; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  const isDrowsy = result.behavior === "drowsy";
  const isAlert = result.behavior === "using_phone" || result.behavior === "distracted";
  const mainColor = isDrowsy ? "rgba(255, 69, 58, 0.7)" : isAlert ? "rgba(255, 159, 10, 0.7)" : "rgba(48, 209, 88, 0.6)";

  // Facial oval
  ctx.strokeStyle = mainColor;
  ctx.lineWidth = 1.6;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.ellipse(face.nose.x * w, (face.nose.y * h) - 10, 95, 130, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Eye reticles
  const drawEyeReticle = (pts) => {
    if (pts[0].x === 0 && pts[0].y === 0) return;
    const cx = (pts[0].x + pts[3].x) / 2 * w;
    const cy = (pts[1].y + pts[4].y) / 2 * h;
    const r = Math.max(8, (pts[3].x - pts[0].x) * w * 0.7);

    ctx.strokeStyle = isDrowsy ? "rgba(255, 69, 58, 0.9)" : "rgba(48, 209, 88, 0.8)";
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = isDrowsy ? "#ff453a" : "#30d158";
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fill();
  };

  if (!manualDrowsyOverride) {
    drawEyeReticle(face.leftEye);
    drawEyeReticle(face.rightEye);
  } else {
    // Closed eyes indicator lines
    ctx.strokeStyle = "#ff453a";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(face.nose.x * w - 65, face.nose.y * h - 40);
    ctx.lineTo(face.nose.x * w - 25, face.nose.y * h - 40);
    ctx.moveTo(face.nose.x * w + 25, face.nose.y * h - 40);
    ctx.lineTo(face.nose.x * w + 65, face.nose.y * h - 40);
    ctx.stroke();
  }

  // Crosshair on nose
  ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
  ctx.lineWidth = 1.2;
  const nx = face.nose.x * w;
  const ny = face.nose.y * h;
  ctx.beginPath();
  ctx.moveTo(nx - 8, ny);
  ctx.lineTo(nx + 8, ny);
  ctx.moveTo(nx, ny - 8);
  ctx.lineTo(nx, ny + 8);
  ctx.stroke();

  // Synthetic Watermark (counter scaleX(-1) so text is not flipped)
  ctx.save();
  ctx.scale(-1, 1);
  ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
  ctx.font = "11px 'SF Mono', monospace";
  ctx.fillText("SYNTHETIC SIMULATION (NO WEBCAM)", -w + 16, 28);
  if (isDrowsy) {
    ctx.fillStyle = "#ff453a";
    ctx.fillText("ALERT: DROWSINESS INJECTED", -w + 16, 46);
  } else if (detectedObjects.phone) {
    ctx.fillStyle = "#ff9f0a";
    ctx.fillText("CONTEXT: PHONE IN USE", -w + 16, 46);
  } else if (detectedObjects.book) {
    ctx.fillStyle = "#0a84ff";
    ctx.fillText("CONTEXT: BOOK / READING", -w + 16, 46);
  }
  ctx.restore();
}

// ================= MINIMALIST LANDMARK RENDERER =================
function drawMinimalLandmarks(lm) {
  ctx.strokeStyle = "rgba(48, 209, 88, 0.5)";
  ctx.lineWidth = 1.4;

  const leftEyePt = lm[33];
  const rightEyePt = lm[362];
  const w = canvasEl.width;
  const h = canvasEl.height;

  // Eye tracking target circles
  ctx.beginPath();
  ctx.arc(leftEyePt.x * w, leftEyePt.y * h, 14, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(rightEyePt.x * w, rightEyePt.y * h, 14, 0, Math.PI * 2);
  ctx.stroke();

  // Subtle crosshair on nose
  const nose = lm[1];
  ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
  ctx.beginPath();
  ctx.moveTo(nose.x * w - 7, nose.y * h);
  ctx.lineTo(nose.x * w + 7, nose.y * h);
  ctx.moveTo(nose.x * w, nose.y * h - 7);
  ctx.lineTo(nose.x * w, nose.y * h + 7);
  ctx.stroke();
}

// ================= UPDATE DASHBOARD UI =================
function updateDashboardUI(result, channels, reason, latencyMs) {
  const { behavior, attentionScore, metrics, ols } = result;

  hudScoreVal.textContent = attentionScore;
  hudLatency.textContent = `${Math.round(latencyMs)} ms`;
  valReason.textContent = reason;

  // State Label & Dot Styling
  hudStateDot.className = "dot";
  if (behavior === "attentive_screen" || behavior === "attentive_reading") {
    hudStateDot.classList.add("green");
    hudScoreVal.style.color = "var(--apple-green)";
  } else if (behavior === "distracted") {
    hudStateDot.classList.add("orange");
    hudScoreVal.style.color = "var(--apple-orange)";
  } else {
    hudStateDot.classList.add("red");
    hudScoreVal.style.color = "var(--apple-red)";
  }

  const stateLabels = {
    attentive_screen: "ATTENTIVE (SCREEN)",
    attentive_reading: "ATTENTIVE (READING)",
    using_phone: "USING MOBILE PHONE",
    distracted: "DISTRACTED / LOOKING AWAY",
    drowsy: "DROWSY / MICROSLEEP",
    absent: "USER ABSENT"
  };
  hudStateLabel.textContent = stateLabels[behavior] || behavior.toUpperCase();

  // Metrics
  valEar.textContent = metrics.ear.toFixed(2);
  valPerclos.textContent = `${metrics.perclos}%`;
  valPose.textContent = `${metrics.pitchDeg > 0 ? "+" : ""}${metrics.pitchDeg}° / ${metrics.yawDeg}°`;
  valFidget.textContent = metrics.fidget;
  subFidget.textContent = `Stability: ${metrics.stability}%`;

  if (ols.fatigueOnsetAnomaly) {
    hudFatigueSlope.textContent = "OLS Fatigue Anomaly!";
    hudFatigueSlope.style.color = "var(--apple-red)";
  } else {
    hudFatigueSlope.textContent = "OLS Nominal";
    hudFatigueSlope.style.color = "var(--text-secondary)";
  }

  // Update Relay Cards
  const isEmergency = (activeMode === "truck" && behavior === "drowsy") || 
                      (activeMode === "exam" && behavior === "using_phone");
  channels.forEach((state, i) => updateRelayCardVisuals(i, state === 1, isEmergency));
}

// ================= EVENT LISTENERS =================
// Mode Selector
document.getElementById("modeSelector").addEventListener("click", (e) => {
  if (e.target.classList.contains("segment-btn")) {
    document.querySelectorAll(".segment-btn").forEach(b => b.classList.remove("active"));
    e.target.classList.add("active");
    activeMode = e.target.dataset.mode;
    logToConsole(`[MODE] Switched to: ${SYSTEM_MODES[activeMode].name}`, "info");
    manualOverrides = [null, null, null, null];
    manualOverrideTimeouts.forEach(t => t && clearTimeout(t));
    renderRelayCards();
  }
});

// Hardware Connectors
btnSerial.addEventListener("click", async () => {
  logToConsole("[PROBE] Checking Native Hardware Bridge...", "normal");
  const bridgeOk = await hardwareBridge.probeNativeBridge();
  if (bridgeOk) {
    labelSerial.textContent = "Hardware Connected";
    btnSerial.classList.add("connected");
    dotSerial.className = "dot green";
    return;
  }

  // Fallback to WebSerial only if native bridge is unavailable
  if (!hardwareBridge.isSerialConnected) {
    const ok = await hardwareBridge.connectSerial();
    if (ok) {
      labelSerial.textContent = "USB Connected";
      btnSerial.classList.add("connected");
      dotSerial.className = "dot green";
    }
  } else {
    await hardwareBridge.disconnectSerial();
    labelSerial.textContent = "Connect USB";
    btnSerial.classList.remove("connected");
    dotSerial.className = "dot";
  }
});

btnWifi.addEventListener("click", async () => {
  const ip = inputIp.value.trim();
  if (ip) {
    const ok = await hardwareBridge.configureWifi(ip);
    if (ok) {
      btnWifi.classList.add("connected");
    } else {
      btnWifi.classList.remove("connected");
    }
  }
});

btnCycleAll.addEventListener("click", async () => {
  logToConsole("[SELF-TEST] Cycling all 4 relays for 1500ms...", "tx");
  for (let i = 0; i < 4; i++) setManualOverride(i, 1);
  await hardwareBridge.dispatchRelays([1, 1, 1, 1]);
  setTimeout(async () => {
    for (let i = 0; i < 4; i++) manualOverrides[i] = null;
    await hardwareBridge.dispatchRelays([0, 0, 0, 0]);
    logToConsole("[SELF-TEST] Self-test complete. Relays returned to automatic.", "info");
  }, 1500);
});

// Test Injections
document.getElementById("simPhone").addEventListener("click", () => {
  detectedObjects.phone = true;
  detectedObjects.book = false;
  manualDrowsyOverride = false;
  logToConsole("[INJECTION] Phone proximity triggered.", "tx");
});

document.getElementById("simBook").addEventListener("click", () => {
  detectedObjects.phone = false;
  detectedObjects.book = true;
  manualDrowsyOverride = false;
  logToConsole("[INJECTION] Book reading context triggered.", "tx");
});

document.getElementById("simDrowsy").addEventListener("click", () => {
  manualDrowsyOverride = true;
  detectedObjects.phone = false;
  detectedObjects.book = false;
  logToConsole("[INJECTION] Drowsiness / Eye-closure triggered.", "error");
});

document.getElementById("simReset").addEventListener("click", () => {
  detectedObjects.phone = false;
  detectedObjects.book = false;
  manualDrowsyOverride = false;
  manualOverrides = [null, null, null, null];
  manualOverrideTimeouts.forEach(t => t && clearTimeout(t));
  logToConsole("[INJECTION] Reset to live sensor feeds.", "info");
});

// Boot
renderRelayCards();
initVisionPipeline();
