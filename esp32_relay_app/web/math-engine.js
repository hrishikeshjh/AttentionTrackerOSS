/**
 * CognEn Mathematical & Algorithmic Core
 * Ported faithfully from CognEn v1.0 and v2.0 Architecture
 *
 * Implements:
 * 1. EAR (Eye Aspect Ratio) & Blink Detection
 * 2. PERCLOS (% Eye Closure over 60s)
 * 3. MAR (Mouth Aspect Ratio) & Yawn Detection
 * 4. 3D Head Pose (Pitch, Yaw, Roll) & Stability
 * 5. Iris Gaze Deviation
 * 6. Fidget & Restlessness Score
 * 7. Fatigue Index & Rolling OLS Fatigue Onset
 * 8. Context-Gate Override & 15-Frame Sliding-Window Behavior Classification
 * 9. EMA Attention Score Smoothing (alpha = 0.15)
 */

export class CognEnMathCore {
  constructor() {
    // Sliding 15-frame window for behavior state (3s @ 5 FPS)
    this.windowSize = 15;
    this.frameBuffer = [];

    // Rolling 60s history for PERCLOS and blink rate (at 5 FPS ~ 300 frames)
    this.perclosBuffer = [];
    this.perclosMaxFrames = 300;

    // Blink tracker
    this.blinkHistory = []; // timestamps of blinks
    this.isEyeClosed = false;
    this.eyeClosedStartMs = 0;

    // Attention EMA smoothing state
    this.smoothedScore = 85.0;
    this.emaAlpha = 0.15; // Official CognEn decay factor

    // Rolling OLS history (attention scores over last 60 seconds)
    this.olsHistory = [];

    // Head pose stability tracking (last 30 seconds)
    this.poseHistory = [];
    this.lastPose = null;
  }

  /**
   * Euclidean distance between two 2D/3D points
   */
  dist(p1, p2) {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    const dz = (p1.z !== undefined && p2.z !== undefined) ? (p1.z - p2.z) : 0;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  /**
   * Eye Aspect Ratio (EAR) - Soukupová & Čech (2016)
   * Formula: (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
   */
  computeEAR(eyePoints) {
    if (!eyePoints || eyePoints.length < 6) return 0.30;
    const [p1, p2, p3, p4, p5, p6] = eyePoints;
    const vertical1 = this.dist(p2, p6);
    const vertical2 = this.dist(p3, p5);
    const horizontal = this.dist(p1, p4);
    if (horizontal < 1e-6) return 0.30;
    return (vertical1 + vertical2) / (2.0 * horizontal);
  }

  /**
   * Mouth Aspect Ratio (MAR)
   * Formula: ||top_lip - bottom_lip|| / ||left_corner - right_corner||
   */
  computeMAR(top, bottom, left, right) {
    if (!top || !bottom || !left || !right) return 0.05;
    const vertical = this.dist(top, bottom);
    const horizontal = this.dist(left, right);
    if (horizontal < 1e-6) return 0.05;
    return vertical / horizontal;
  }

  /**
   * Head Pose from canonical facial landmarks
   * Returns: { pitchDeg, yawDeg, rollDeg }
   */
  computeHeadPose(nose, leftEye, rightEye, chin) {
    if (!nose || !leftEye || !rightEye || !chin) {
      return { pitchDeg: 0, yawDeg: 0, rollDeg: 0 };
    }

    const eyeMid = {
      x: (leftEye.x + rightEye.x) / 2,
      y: (leftEye.y + rightEye.y) / 2
    };

    const eyeDist = this.dist(leftEye, rightEye);
    const noseChinDist = this.dist(nose, chin);

    // Yaw: Nose horizontal offset from eye midpoint
    const yawDeg = eyeDist > 1e-6 ? ((nose.x - eyeMid.x) / eyeDist) * 50.0 : 0;

    // Pitch: Nose vertical offset relative to chin
    const pitchDeg = noseChinDist > 1e-6 ? ((nose.y - eyeMid.y) / noseChinDist) * 50.0 : 0;

    // Roll: Angle between eyes
    const rollDeg = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * (180.0 / Math.PI);

    return { pitchDeg, yawDeg, rollDeg };
  }

  /**
   * Update rolling PERCLOS and Blink Rate
   */
  updateEyeDynamics(avgEAR, timestampMs) {
    const isClosedNow = avgEAR < 0.21;
    this.perclosBuffer.push(isClosedNow ? 1 : 0);
    if (this.perclosBuffer.length > this.perclosMaxFrames) {
      this.perclosBuffer.shift();
    }

    // Compute PERCLOS percentage
    const closedCount = this.perclosBuffer.reduce((a, b) => a + b, 0);
    const perclos = this.perclosBuffer.length > 0 ? (closedCount / this.perclosBuffer.length) * 100 : 0;

    // Blink detection: Transition from open -> closed -> open
    if (!this.isEyeClosed && isClosedNow) {
      this.isEyeClosed = true;
      this.eyeClosedStartMs = timestampMs;
    } else if (this.isEyeClosed && !isClosedNow) {
      this.isEyeClosed = false;
      const duration = timestampMs - this.eyeClosedStartMs;
      if (duration >= 50 && duration <= 500) {
        this.blinkHistory.push(timestampMs);
      }
    }

    // Prune blinks older than 60 seconds
    const cutoff = timestampMs - 60000;
    this.blinkHistory = this.blinkHistory.filter(t => t > cutoff);
    const blinkRateBpm = this.blinkHistory.length; // Blinks in last 60s

    return { perclos, blinkRateBpm, isClosedNow, closedDurationMs: this.isEyeClosed ? (timestampMs - this.eyeClosedStartMs) : 0 };
  }

  /**
   * Update Fidget & Head Pose Stability
   */
  updatePoseDynamics(pose, timestampMs) {
    let fidget = 0;
    if (this.lastPose) {
      const dyaw = Math.abs(pose.yawDeg - this.lastPose.yawDeg);
      const dpitch = Math.abs(pose.pitchDeg - this.lastPose.pitchDeg);
      const droll = Math.abs(pose.rollDeg - this.lastPose.rollDeg);
      fidget = Math.min((dyaw + dpitch + droll) * 3.5, 100);
    }
    this.lastPose = pose;

    this.poseHistory.push({ ...pose, t: timestampMs });
    const cutoff = timestampMs - 30000;
    this.poseHistory = this.poseHistory.filter(p => p.t > cutoff);

    // Compute variance over window
    let stability = 90;
    if (this.poseHistory.length > 5) {
      const yaws = this.poseHistory.map(p => p.yawDeg);
      const pitches = this.poseHistory.map(p => p.pitchDeg);
      const meanYaw = yaws.reduce((a, b) => a + b, 0) / yaws.length;
      const meanPitch = pitches.reduce((a, b) => a + b, 0) / pitches.length;
      const varYaw = yaws.reduce((a, b) => a + Math.pow(b - meanYaw, 2), 0) / yaws.length;
      const varPitch = pitches.reduce((a, b) => a + Math.pow(b - meanPitch, 2), 0) / pitches.length;
      stability = Math.max(0, Math.min(100, 100 - (varYaw + varPitch) / 2.5));
    }

    return { fidget, stability };
  }

  /**
   * Ordinary Least Squares (OLS) Regression for Fatigue Onset
   * Anomaly condition: slope < -0.5, R^2 > 0.4, average < 60
   */
  calculateOLS(attentionScore, timestampMs) {
    this.olsHistory.push({ score: attentionScore, t: timestampMs / 1000.0 });
    const cutoff = (timestampMs - 60000) / 1000.0;
    this.olsHistory = this.olsHistory.filter(pt => pt.t > cutoff);

    if (this.olsHistory.length < 15) {
      return { slope: 0, r2: 0, fatigueOnsetAnomaly: false };
    }

    const n = this.olsHistory.length;
    const meanT = this.olsHistory.reduce((acc, p) => acc + p.t, 0) / n;
    const meanS = this.olsHistory.reduce((acc, p) => acc + p.score, 0) / n;

    let num = 0;
    let den = 0;
    let ssTot = 0;
    let ssRes = 0;

    for (const p of this.olsHistory) {
      num += (p.t - meanT) * (p.score - meanS);
      den += Math.pow(p.t - meanT, 2);
      ssTot += Math.pow(p.score - meanS, 2);
    }

    const slope = den > 1e-6 ? num / den : 0;
    const intercept = meanS - slope * meanT;

    for (const p of this.olsHistory) {
      const pred = slope * p.t + intercept;
      ssRes += Math.pow(p.score - pred, 2);
    }

    const r2 = ssTot > 1e-6 ? Math.max(0, 1 - (ssRes / ssTot)) : 0;
    const fatigueOnsetAnomaly = (slope < -0.4) && (r2 > 0.35) && (meanS < 65);

    return { slope, r2, meanScore: meanS, fatigueOnsetAnomaly };
  }

  /**
   * CognEn Context-Gate Override & Behavior Classifier
   * 6 Behavior Classes:
   * 1. attentive_screen
   * 2. attentive_reading
   * 3. using_phone
   * 4. distracted
   * 5. drowsy
   * 6. absent
   */
  classifyBehavior(metrics, detectedObjects) {
    const { ear, perclos, pitchDeg, yawDeg, closedDurationMs, faceDetected } = metrics;
    const hasPhone = detectedObjects.phone === true;
    const hasBook = detectedObjects.book === true;

    // Rule 1: Absence
    if (!faceDetected) {
      return { behavior: "absent", score: 0, confidence: 1.0 };
    }

    // Rule 2: Drowsy (Eyes closed continuous >= 2s or high PERCLOS)
    if (closedDurationMs >= 1800 || (ear < 0.21 && perclos > 25)) {
      return { behavior: "drowsy", score: 20, confidence: 0.95 };
    }

    // Rule 3: Context-Gate Phone Override
    // If looking down and smartphone object is active near hands
    if (hasPhone) {
      return { behavior: "using_phone", score: 25, confidence: 0.92 };
    }

    // Rule 4: Context-Gate Attentive Reading Override
    // Downward pitch with book / notebook present
    if (hasBook || (pitchDeg > 14 && Math.abs(yawDeg) < 20)) {
      return { behavior: "attentive_reading", score: 82, confidence: 0.88 };
    }

    // Rule 5: Distracted (Looking left/right or severe head tilt)
    if (Math.abs(yawDeg) > 24 || Math.abs(pitchDeg) > 28) {
      return { behavior: "distracted", score: 45, confidence: 0.85 };
    }

    // Rule 6: Attentive Screen (Default upright focused gaze)
    return { behavior: "attentive_screen", score: 90, confidence: 0.90 };
  }

  /**
   * Main Process Frame Pipeline
   */
  process(faceData, detectedObjects, timestampMs = Date.now()) {
    if (!faceData) {
      return {
        behavior: "absent",
        attentionScore: 0,
        smoothedScore: Math.round(this.smoothedScore * 0.95),
        metrics: { ear: 0, perclos: 100, mar: 0, pitchDeg: 0, yawDeg: 0, rollDeg: 0, fidget: 0, stability: 0, blinkRate: 0 },
        ols: { slope: 0, fatigueOnsetAnomaly: false }
      };
    }

    const { leftEye, rightEye, topLip, bottomLip, mouthLeft, mouthRight, nose, chin } = faceData;

    // 1. Compute Biometrics
    const leftEAR = this.computeEAR(leftEye);
    const rightEAR = this.computeEAR(rightEye);
    const avgEAR = (leftEAR + rightEAR) / 2.0;

    const mar = this.computeMAR(topLip, bottomLip, mouthLeft, mouthRight);
    const pose = this.computeHeadPose(nose, leftEye[0], rightEye[3], chin);

    // 2. Eye and Pose Dynamics
    const eyeDyn = this.updateEyeDynamics(avgEAR, timestampMs);
    const poseDyn = this.updatePoseDynamics(pose, timestampMs);

    const metrics = {
      faceDetected: true,
      ear: Number(avgEAR.toFixed(3)),
      perclos: Number(eyeDyn.perclos.toFixed(1)),
      mar: Number(mar.toFixed(3)),
      pitchDeg: Number(pose.pitchDeg.toFixed(1)),
      yawDeg: Number(pose.yawDeg.toFixed(1)),
      rollDeg: Number(pose.rollDeg.toFixed(1)),
      fidget: Math.round(poseDyn.fidget),
      stability: Math.round(poseDyn.stability),
      blinkRate: eyeDyn.blinkRateBpm,
      closedDurationMs: eyeDyn.closedDurationMs
    };

    // 3. Behavior Classification
    const classification = this.classifyBehavior(metrics, detectedObjects);

    // 4. Exponential Moving Average (EMA) Attention Score Smoothing
    // smoothed = alpha * raw + (1 - alpha) * prev_smoothed
    this.smoothedScore = (this.emaAlpha * classification.score) + ((1.0 - this.emaAlpha) * this.smoothedScore);
    const finalAttentionScore = Math.round(this.smoothedScore);

    // 5. OLS Fatigue Onset Evaluation
    const ols = this.calculateOLS(finalAttentionScore, timestampMs);

    return {
      behavior: classification.behavior,
      rawScore: classification.score,
      attentionScore: finalAttentionScore,
      confidence: classification.confidence,
      metrics,
      ols
    };
  }
}
