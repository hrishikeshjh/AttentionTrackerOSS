/**
 * CognEn Extension - ML Core Library v5 (scoring core 1.1.0, legacy engine)
 *
 * Plain-JS port of the web app's scoring package so that the extension and
 * the web app score identical landmark input identically:
 *
 *   v1.0/src/lib/scoring/constants.ts      -> constants below
 *   v1.0/src/lib/scoring/geometry.ts       -> distance / clamp / calculateVariance
 *   v1.0/src/lib/scoring/ear-mar.ts        -> calculateEAR / calculateMAR
 *   v1.0/src/lib/scoring/basic-metrics.ts  -> face distance / area / eyebrow / mouth
 *   v1.0/src/lib/scoring/pose/head-pose.ts -> calculateHeadPose
 *   v1.0/src/lib/scoring/pose/gaze.ts      -> calculateGazeDirection
 *   v1.0/src/lib/scoring/trackers.ts       -> BlinkDetector / HeadPoseTracker / FatigueCalculator
 *   v1.0/src/lib/scoring/aggregator.ts     -> FaceMetricsAggregator / HandObjectProximityDetector
 *   v1.0/src/lib/scoring/face-selection.ts -> PrimaryFaceSelector
 *   v1.0/src/lib/scoring/ema.ts            -> ExponentialMovingAverage
 *   v1.0/src/lib/ml-engine.ts (legacy predict path) -> MLEngine
 *
 * Equivalence is enforced by src/lib/__tests__/extension-equivalence.test.ts,
 * which loads this file in node and compares it against the TS modules on
 * the same synthetic landmark sequences. Any formula change in the TS
 * scoring package must be ported here in the same change, or that test fails.
 *
 * Weight set: the extension has no object detector or hand tracker, so it
 * always scores with FEATURE_WEIGHTS_LEGACY, which is exactly what the web
 * MLEngine does when `metrics.detectedObjects === undefined`. Callers in the
 * extension (offscreen.js, content.js) must therefore NOT attach
 * detectedObjects / handObjectProximity / behaviorResult to the metrics.
 *
 * Cadence: the web app runs the aggregator on every analysed frame and calls
 * MLEngine.predict once per second (webcam-panel.tsx DATA_POINT_INTERVAL).
 * The extension does the same (5 FPS aggregation, PREDICT_INTERVAL_MS = 1000
 * prediction), so the EMA alpha of 0.15 has the same time constant in both.
 *
 * Packaging: everything lives inside one IIFE and is published on a single
 * namespace object (`__cognen.mlCore`) so that re-injecting this file into a
 * page does not throw on redeclared top-level `const`/`class` bindings, and
 * so that node can `module.exports` it for the equivalence test.
 */
(function (root) {
    'use strict';

    const ns = (root.__cognen = root.__cognen || {});
    if (ns.mlCore) {
        // Already loaded in this realm (re-injection on a second START).
        if (typeof module !== 'undefined' && module.exports) module.exports = ns.mlCore;
        return;
    }

    const VERSION = '5.0.0';
    const SCORING_CORE_VERSION = '1.1.0';

    // ==================== CONSTANTS (scoring/constants.ts) ====================

    const LEFT_EYE_LANDMARKS = [36, 37, 38, 39, 40, 41];
    const RIGHT_EYE_LANDMARKS = [42, 43, 44, 45, 46, 47];
    const MOUTH_LANDMARKS = [48, 54, 51, 57]; // Left, Right, Top, Bottom
    const LEFT_EYEBROW = [17, 18, 19, 20, 21];
    const RIGHT_EYEBROW = [22, 23, 24, 25, 26];
    const HEAD_POSE_LANDMARKS = { nose: 30, leftEyeCorner: 36, rightEyeCorner: 45, chin: 8 };

    const EAR_BLINK_THRESHOLD = 0.21;
    const PERCLOS_WINDOW_MS = 60000;
    const NORMAL_BLINK_RATE_MIN = 15;
    const NORMAL_BLINK_RATE_MAX = 20;
    const FATIGUE_EAR_THRESHOLD = 0.25;

    // Canonical thresholds: Attentive >= 70, Distracted 40-69, Multitasking < 40.
    // Every extension surface (badge, widget, popup ring, alerts) reads these.
    const THRESHOLDS = Object.freeze({ ATTENTIVE_MIN: 70, DISTRACTED_MIN: 40 });

    const OPTIMAL_RANGES = Object.freeze({
        ear: { min: 0.25, max: 0.4 },
        perclos: { max: 15 },
        blinkRate: { min: 12, max: 20 },
        gaze: { maxDeviation: 0.3 },
        yaw: { maxDeviation: 15 },
        pitch: { maxDeviation: 15 },
        fatigueIndex: { max: 30 },
        fidgetScore: { max: 25 },
        faceDistance: { min: 0.3, max: 0.7 },
    });

    const FEATURE_WEIGHTS_LEGACY = Object.freeze({
        ear: 0.15,
        perclos: 0.12,
        blinkRate: 0.08,
        gazeDeviation: 0.1,
        headPoseStability: 0.12,
        yawDeviation: 0.08,
        pitchDeviation: 0.06,
        fatigueIndex: 0.1,
        fidgetScore: 0.07,
        eyebrowRaise: 0.04,
        mouthOpenness: 0.03,
        faceDistance: 0.03,
        facePresence: 0.02,
    });

    const FEATURE_WEIGHTS_V1_UPGRADED = Object.freeze({
        ear: 0.1,
        perclos: 0.08,
        blinkRate: 0.05,
        gazeDeviation: 0.12,
        headPoseStability: 0.1,
        yawDeviation: 0.08,
        fatigueIndex: 0.07,
        fidgetScore: 0.04,
        mouthOpenness: 0.02,
        faceDistance: 0.02,
        facePresence: 0.02,
        contextAwarePitch: 0.12,
        objectContext: 0.1,
        handObjectProximity: 0.05,
        temporalPattern: 0.03,
    });

    const HEAD_POSE_CALIBRATION = Object.freeze({ YAW_GAIN: 50, PITCH_GAIN: 33.3 });

    /** v1.0 rule-layer proximity codes (temporal-classifier.ts PROXIMITY_CODES_V1). */
    const PROXIMITY_CODES_V1 = Object.freeze({
        holding_book: 1.0,
        holding_pen: 0.8,
        hands_on_desk: 0.5,
        holding_phone: 0.2,
        hands_not_visible: 0.0,
    });

    /** Prediction cadence: the web app predicts once per second. */
    const PREDICT_INTERVAL_MS = 1000;
    /** Frame cadence for the analysis loop (5 FPS). */
    const FRAME_INTERVAL_MS = 200;

    // ==================== GEOMETRY (scoring/geometry.ts) ====================

    function distance(p1, p2) {
        return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
    }

    function calculateVariance(values) {
        if (values.length === 0) return 0;
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        return values.reduce((sum, val) => sum + (val - mean) ** 2, 0) / values.length;
    }

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    // ==================== EAR / MAR (scoring/ear-mar.ts) ====================

    function calculateSingleEAR(landmarks, eyeIndices) {
        const eyePoints = eyeIndices.map((i) => landmarks[i]);
        if (eyePoints.some((p) => !p)) return null;
        const [p1, p2, p3, p4, p5, p6] = eyePoints;
        const verticalDist = distance(p2, p6) + distance(p3, p5);
        const horizontalDist = distance(p1, p4);
        if (horizontalDist === 0) return null;
        return verticalDist / (2 * horizontalDist);
    }

    function calculateEAR(landmarks) {
        if (!landmarks || landmarks.length < 68) return null;
        const leftEAR = calculateSingleEAR(landmarks, LEFT_EYE_LANDMARKS);
        const rightEAR = calculateSingleEAR(landmarks, RIGHT_EYE_LANDMARKS);
        if (leftEAR === null || rightEAR === null) return null;
        return (leftEAR + rightEAR) / 2;
    }

    function calculateMAR(landmarks) {
        if (!landmarks || landmarks.length < 68) return null;
        const mouthPoints = MOUTH_LANDMARKS.map((i) => landmarks[i]);
        if (mouthPoints.some((p) => !p)) return null;
        const [pLeft, pRight, pTop, pBottom] = mouthPoints;
        const verticalDist = distance(pTop, pBottom);
        const horizontalDist = distance(pLeft, pRight);
        if (horizontalDist === 0) return null;
        return verticalDist / horizontalDist;
    }

    // ==================== BASIC METRICS (scoring/basic-metrics.ts) ====================

    function estimateFaceDistance(landmarks, frameWidth) {
        if (!landmarks || landmarks.length < 68) return null;
        const leftEye = landmarks[36];
        const rightEye = landmarks[45];
        if (!leftEye || !rightEye) return null;
        const ipd = distance(leftEye, rightEye);
        const normalizedIPD = ipd / frameWidth;
        return clamp((normalizedIPD - 0.1) / 0.3, 0, 1);
    }

    function calculateFaceArea(landmarks, frameWidth, frameHeight) {
        if (!landmarks || landmarks.length < 68) return null;
        const xs = landmarks.map((p) => p.x);
        const ys = landmarks.map((p) => p.y);
        const faceArea = (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
        return faceArea / (frameWidth * frameHeight);
    }

    function calculateEyebrowRaise(landmarks) {
        if (!landmarks || landmarks.length < 68) return null;
        const ysOf = (indices) => indices
            .map((i) => (landmarks[i] ? landmarks[i].y : undefined))
            .filter((y) => y !== undefined);

        const leftEyebrowY = ysOf(LEFT_EYEBROW);
        const rightEyebrowY = ysOf(RIGHT_EYEBROW);
        if (leftEyebrowY.length === 0 || rightEyebrowY.length === 0) return null;
        const avgEyebrowY = [...leftEyebrowY, ...rightEyebrowY].reduce((a, b) => a + b, 0) /
            (leftEyebrowY.length + rightEyebrowY.length);

        const leftEyeY = ysOf(LEFT_EYE_LANDMARKS);
        const rightEyeY = ysOf(RIGHT_EYE_LANDMARKS);
        if (leftEyeY.length === 0 || rightEyeY.length === 0) return null;
        const avgEyeY = [...leftEyeY, ...rightEyeY].reduce((a, b) => a + b, 0) /
            (leftEyeY.length + rightEyeY.length);

        const eyeWidth = Math.abs(landmarks[36].x - landmarks[45].x);
        if (eyeWidth === 0) return null;

        const eyebrowEyeDistance = (avgEyeY - avgEyebrowY) / eyeWidth;
        return clamp((eyebrowEyeDistance - 0.2) / 0.3, 0, 1);
    }

    function calculateMouthOpenness(landmarks) {
        if (!landmarks || landmarks.length < 68) return null;
        const topLip = landmarks[62];
        const bottomLip = landmarks[66];
        const leftCorner = landmarks[60];
        const rightCorner = landmarks[64];
        if (!topLip || !bottomLip || !leftCorner || !rightCorner) return null;
        const mouthHeight = distance(topLip, bottomLip);
        const mouthWidth = distance(leftCorner, rightCorner);
        if (mouthWidth === 0) return null;
        return clamp(mouthHeight / mouthWidth, 0, 1);
    }

    // ==================== HEAD POSE (scoring/pose/head-pose.ts) ====================

    function calculateHeadPose(landmarks) {
        const none = { yaw: null, pitch: null, roll: null };
        if (!landmarks || landmarks.length < 68) return none;

        const nose = landmarks[HEAD_POSE_LANDMARKS.nose];
        const leftEye = landmarks[HEAD_POSE_LANDMARKS.leftEyeCorner];
        const rightEye = landmarks[HEAD_POSE_LANDMARKS.rightEyeCorner];
        const chin = landmarks[HEAD_POSE_LANDMARKS.chin];
        if (!nose || !leftEye || !rightEye || !chin) return none;

        const eyeMidpoint = { x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2 };
        const eyeDist = distance(leftEye, rightEye);
        if (eyeDist === 0) return none;

        const yawRatio = (nose.x - eyeMidpoint.x) / eyeDist;
        const pitchRatio = (nose.y - eyeMidpoint.y) / eyeDist;
        const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

        return {
            yaw: yawRatio * HEAD_POSE_CALIBRATION.YAW_GAIN,
            pitch: pitchRatio * HEAD_POSE_CALIBRATION.PITCH_GAIN,
            roll: roll * (180 / Math.PI),
        };
    }

    // ==================== GAZE (scoring/pose/gaze.ts) ====================

    function calculateGazeDirection(landmarks) {
        if (!landmarks || landmarks.length < 68) return { gazeX: null, gazeY: null };

        const leftEyeLeft = landmarks[36];
        const leftEyeRight = landmarks[39];
        const rightEyeLeft = landmarks[42];
        const rightEyeRight = landmarks[45];
        if (!leftEyeLeft || !leftEyeRight || !rightEyeLeft || !rightEyeRight) {
            return { gazeX: null, gazeY: null };
        }

        const leftIrisCenter = {
            x: (landmarks[37].x + landmarks[38].x + landmarks[40].x + landmarks[41].x) / 4,
            y: (landmarks[37].y + landmarks[38].y + landmarks[40].y + landmarks[41].y) / 4,
        };
        const rightIrisCenter = {
            x: (landmarks[43].x + landmarks[44].x + landmarks[46].x + landmarks[47].x) / 4,
            y: (landmarks[43].y + landmarks[44].y + landmarks[46].y + landmarks[47].y) / 4,
        };

        const leftEyeWidth = distance(leftEyeLeft, leftEyeRight);
        const rightEyeWidth = distance(rightEyeLeft, rightEyeRight);
        if (leftEyeWidth === 0 || rightEyeWidth === 0) return { gazeX: null, gazeY: null };

        const leftGazeX = ((leftIrisCenter.x - leftEyeLeft.x) / leftEyeWidth - 0.5) * 2;
        const rightGazeX = ((rightIrisCenter.x - rightEyeLeft.x) / rightEyeWidth - 0.5) * 2;

        const leftEyeCenterY = (leftEyeLeft.y + leftEyeRight.y + landmarks[37].y + landmarks[41].y) / 4;
        const rightEyeCenterY = (rightEyeLeft.y + rightEyeRight.y + landmarks[43].y + landmarks[47].y) / 4;
        const leftEyeHeight = Math.abs(landmarks[37].y - landmarks[41].y);
        const rightEyeHeight = Math.abs(landmarks[43].y - landmarks[47].y);

        const leftGazeY = leftEyeHeight > 0 ? (leftEyeCenterY - leftIrisCenter.y) / leftEyeHeight : 0;
        const rightGazeY = rightEyeHeight > 0 ? (rightEyeCenterY - rightIrisCenter.y) / rightEyeHeight : 0;

        return {
            gazeX: clamp((leftGazeX + rightGazeX) / 2, -1, 1),
            gazeY: clamp((leftGazeY + rightGazeY) / 2, -1, 1),
        };
    }

    // ==================== TRACKERS (scoring/trackers.ts) ====================

    /**
     * Blink detector. Same rules as the web BlinkDetector: a blink is an
     * eye-closed run (EAR < 0.21) lasting 50-500 ms, with no extra debounce.
     *
     * The web getBlinkRate() reads Date.now(); here the current frame
     * timestamp is used when available (frame timestamps are Date.now() at
     * capture time in both apps, so live behaviour is the same, and this
     * stays deterministic under test).
     */
    class BlinkDetector {
        constructor() {
            this.reset();
        }

        detectBlink(ear, timestamp) {
            if (ear === null) {
                return { blinkDetected: false, blinkRate: this.getBlinkRate(timestamp), perclos: this.getPERCLOS(timestamp) };
            }

            this.earHistory.push({ time: timestamp, ear });
            const cutoff = timestamp - 120000;
            this.earHistory = this.earHistory.filter((e) => e.time > cutoff);
            this.blinkEvents = this.blinkEvents.filter((t) => t > cutoff);

            const isEyeClosed = ear < EAR_BLINK_THRESHOLD;
            let blinkDetected = false;

            if (isEyeClosed && !this.wasEyeClosed) this.eyeClosureStart = timestamp;

            if (this.wasEyeClosed && !isEyeClosed) {
                if (this.eyeClosureStart) {
                    const blinkDuration = timestamp - this.eyeClosureStart;
                    if (blinkDuration >= 50 && blinkDuration <= 500) {
                        blinkDetected = true;
                        this.blinkEvents.push(timestamp);
                    }
                }
                this.eyeClosureStart = null;
            }

            this.wasEyeClosed = isEyeClosed;

            return { blinkDetected, blinkRate: this.getBlinkRate(timestamp), perclos: this.getPERCLOS(timestamp) };
        }

        getBlinkRate(currentTime) {
            if (this.blinkEvents.length < 2) return 0;
            const now = typeof currentTime === 'number' ? currentTime : Date.now();
            const oneMinuteAgo = now - 60000;
            const recentBlinks = this.blinkEvents.filter((t) => t > oneMinuteAgo);
            const timeSpan = Math.min(60000, now - this.blinkEvents[0]);
            if (timeSpan < 5000) return 0;
            return (recentBlinks.length / timeSpan) * 60000;
        }

        getPERCLOS(currentTime) {
            const oneMinuteAgo = currentTime - PERCLOS_WINDOW_MS;
            const recentHistory = this.earHistory.filter((e) => e.time > oneMinuteAgo);
            if (recentHistory.length < 10) return 0;
            const closedCount = recentHistory.filter((e) => e.ear < EAR_BLINK_THRESHOLD).length;
            return (closedCount / recentHistory.length) * 100;
        }

        getEARStats() {
            if (this.earHistory.length === 0) return { mean: 0, variance: 0 };
            const ears = this.earHistory.map((e) => e.ear);
            const mean = ears.reduce((a, b) => a + b, 0) / ears.length;
            return { mean, variance: calculateVariance(ears) };
        }

        reset() {
            this.earHistory = [];
            this.blinkEvents = [];
            this.wasEyeClosed = false;
            this.eyeClosureStart = null;
        }
    }

    const FIDGET_SMOOTHING_ALPHA = 0.4;

    class HeadPoseTracker {
        constructor() {
            this.reset();
        }

        addPose(pose, timestamp) {
            if (pose.yaw !== null && pose.pitch !== null && pose.roll !== null) {
                this.poseHistory.push({ time: timestamp, yaw: pose.yaw, pitch: pose.pitch, roll: pose.roll });
                const cutoff = timestamp - 30000;
                this.poseHistory = this.poseHistory.filter((p) => p.time > cutoff);
            }
        }

        getStability() {
            if (this.poseHistory.length < 5) return 100;
            const yaws = this.poseHistory.map((p) => p.yaw);
            const pitches = this.poseHistory.map((p) => p.pitch);
            const rolls = this.poseHistory.map((p) => p.roll);
            const combinedVar = (calculateVariance(yaws) + calculateVariance(pitches) + calculateVariance(rolls) * 0.5) / 2.5;
            return clamp(100 - combinedVar / 10, 0, 100);
        }

        getFidgetScore() {
            if (this.poseHistory.length < 10) return 0;
            const alpha = FIDGET_SMOOTHING_ALPHA;
            let sy = this.poseHistory[0].yaw;
            let sp = this.poseHistory[0].pitch;
            let sr = this.poseHistory[0].roll;
            const diffs = [];
            for (let i = 1; i < this.poseHistory.length; i++) {
                const prevY = sy, prevP = sp, prevR = sr;
                const cur = this.poseHistory[i];
                sy = alpha * cur.yaw + (1 - alpha) * sy;
                sp = alpha * cur.pitch + (1 - alpha) * sp;
                sr = alpha * cur.roll + (1 - alpha) * sr;
                diffs.push(Math.abs(sy - prevY) + Math.abs(sp - prevP) + Math.abs(sr - prevR));
            }
            const avgDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
            return clamp(avgDiff * 5, 0, 100);
        }

        getPoseVariance() {
            if (this.poseHistory.length < 2) return 0;
            const yaws = this.poseHistory.map((p) => p.yaw);
            const pitches = this.poseHistory.map((p) => p.pitch);
            const rolls = this.poseHistory.map((p) => p.roll);
            return (calculateVariance(yaws) + calculateVariance(pitches) + calculateVariance(rolls)) / 3;
        }

        reset() {
            this.poseHistory = [];
        }
    }

    class FatigueCalculator {
        constructor() {
            this.reset();
        }

        update(perclos, blinkRate, earMean, timestamp) {
            this.perclosHistory.push({ time: timestamp, value: perclos });
            this.blinkRateHistory.push({ time: timestamp, value: blinkRate });
            this.earMeanHistory.push({ time: timestamp, value: earMean });

            const cutoff = timestamp - 300000;
            this.perclosHistory = this.perclosHistory.filter((p) => p.time > cutoff);
            this.blinkRateHistory = this.blinkRateHistory.filter((p) => p.time > cutoff);
            this.earMeanHistory = this.earMeanHistory.filter((p) => p.time > cutoff);

            return this.calculateFatigueIndex();
        }

        calculateFatigueIndex() {
            if (this.perclosHistory.length < 5) return 0;

            const avgPerclos = this.perclosHistory.reduce((a, b) => a + b.value, 0) / this.perclosHistory.length;
            const perclosScore = clamp(avgPerclos * 2, 0, 40);

            const avgBlinkRate = this.blinkRateHistory.reduce((a, b) => a + b.value, 0) / this.blinkRateHistory.length;
            let blinkScore = 0;
            if (avgBlinkRate < NORMAL_BLINK_RATE_MIN) {
                blinkScore = clamp((NORMAL_BLINK_RATE_MIN - avgBlinkRate) * 2, 0, 30);
            } else if (avgBlinkRate > NORMAL_BLINK_RATE_MAX) {
                blinkScore = clamp(avgBlinkRate - NORMAL_BLINK_RATE_MAX, 0, 30);
            }

            let earTrendScore = 0;
            if (this.earMeanHistory.length > 10) {
                const recentEars = this.earMeanHistory.slice(-10).map((e) => e.value);
                const avgEar = recentEars.reduce((a, b) => a + b, 0) / recentEars.length;
                if (avgEar < FATIGUE_EAR_THRESHOLD) {
                    earTrendScore = clamp((FATIGUE_EAR_THRESHOLD - avgEar) * 100, 0, 30);
                }
            }

            return clamp(perclosScore + blinkScore + earTrendScore, 0, 100);
        }

        reset() {
            this.perclosHistory = [];
            this.blinkRateHistory = [];
            this.earMeanHistory = [];
        }
    }

    // ==================== AGGREGATOR (scoring/aggregator.ts) ====================

    class HandObjectProximityDetector {
        compute(handResults, detectedObjects) {
            if (!handResults || handResults.length === 0) return 'hands_not_visible';
            if (!detectedObjects || detectedObjects.length === 0) return 'hands_on_desk';

            const handCenters = handResults.map((h) => {
                const sum = h.landmarks.reduce((a, l) => ({ x: a.x + l.x, y: a.y + l.y }), { x: 0, y: 0 });
                return { x: sum.x / h.landmarks.length, y: sum.y / h.landmarks.length };
            });

            for (const priority of ['phone', 'book', 'notebook', 'pen']) {
                const obj = detectedObjects.find((o) => o.class === priority);
                if (!obj) continue;
                const objCenter = { x: obj.bbox.x, y: obj.bbox.y };
                for (const hand of handCenters) {
                    if (distance(hand, objCenter) < HandObjectProximityDetector.PROXIMITY_THRESHOLD) {
                        if (priority === 'phone') return 'holding_phone';
                        if (priority === 'book' || priority === 'notebook') return 'holding_book';
                        if (priority === 'pen') return 'holding_pen';
                    }
                }
            }
            return 'hands_on_desk';
        }
    }
    HandObjectProximityDetector.PROXIMITY_THRESHOLD = 0.2;

    class FaceMetricsAggregator {
        constructor() {
            this.reset();
        }

        calculateExtendedMetrics(landmarks, frameWidth, frameHeight, timestamp) {
            const ear = calculateEAR(landmarks);
            const mar = calculateMAR(landmarks);
            const headPose = calculateHeadPose(landmarks);

            if (mar !== null) {
                this.marHistory.push(mar);
                if (this.marHistory.length > 100) this.marHistory.shift();
            }

            const { blinkDetected, blinkRate, perclos } = this.blinkDetector.detectBlink(ear, timestamp);
            const earStats = this.blinkDetector.getEARStats();

            this.headPoseTracker.addPose(headPose, timestamp);
            const headPoseStability = this.headPoseTracker.getStability();
            const fidgetScore = this.headPoseTracker.getFidgetScore();
            const poseVariance = this.headPoseTracker.getPoseVariance();

            const gaze = calculateGazeDirection(landmarks);
            const faceDistance = estimateFaceDistance(landmarks, frameWidth);
            const faceArea = calculateFaceArea(landmarks, frameWidth, frameHeight);
            const eyebrowRaise = calculateEyebrowRaise(landmarks);
            const mouthOpenness = calculateMouthOpenness(landmarks);
            const fatigueIndex = this.fatigueCalculator.update(perclos, blinkRate, earStats.mean, timestamp);

            return {
                ear,
                mar,
                yaw: headPose.yaw,
                pitch: headPose.pitch,
                roll: headPose.roll,
                blinkDetected,
                blinkRate,
                perclos,
                gazeX: gaze.gazeX,
                gazeY: gaze.gazeY,
                faceDistance,
                faceArea,
                fatigueIndex,
                fidgetScore,
                headPoseStability,
                eyebrowRaise,
                mouthOpenness,
                earVariance: earStats.variance,
                marVariance: calculateVariance(this.marHistory),
                poseVariance,
            };
        }

        reset() {
            this.blinkDetector = new BlinkDetector();
            this.headPoseTracker = new HeadPoseTracker();
            this.fatigueCalculator = new FatigueCalculator();
            this.marHistory = [];
        }
    }

    // ==================== FACE SELECTION (scoring/face-selection.ts) ====================

    const DISAPPEAR_FRAMES_BEFORE_UNLOCK = 10;
    const SWITCH_SCORE_MARGIN = 1.2;
    const SWITCH_CONSECUTIVE_FRAMES = 5;

    function scoreCandidate(box, frameWidth, frameHeight) {
        const area = box.width * box.height;
        const areaNorm = Math.min(1, area / (frameWidth * frameHeight));
        const boxCenterX = box.x + box.width / 2;
        const boxCenterY = box.y + box.height / 2;
        const frameCenterX = frameWidth / 2;
        const frameCenterY = frameHeight / 2;
        const maxDist = Math.sqrt(frameCenterX ** 2 + frameCenterY ** 2);
        const dist = Math.sqrt((boxCenterX - frameCenterX) ** 2 + (boxCenterY - frameCenterY) ** 2);
        const centeredness = maxDist > 0 ? 1 - Math.min(1, dist / maxDist) : 1;
        return areaNorm * 0.7 + centeredness * 0.3;
    }

    function iou(a, b) {
        const x1 = Math.max(a.x, b.x);
        const y1 = Math.max(a.y, b.y);
        const x2 = Math.min(a.x + a.width, b.x + b.width);
        const y2 = Math.min(a.y + a.height, b.y + b.height);
        const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const union = a.width * a.height + b.width * b.height - interArea;
        return union > 0 ? interArea / union : 0;
    }

    /**
     * Locks onto one primary face (largest + most centred) with hysteresis:
     * the lock is kept unless it disappears for 10 consecutive frames, or a
     * new candidate outscores it by 20% for 5 consecutive frames.
     * candidates: [{ box: {x, y, width, height}, detection }]
     */
    class PrimaryFaceSelector {
        constructor() {
            this.reset();
        }

        select(candidates, frameWidth, frameHeight) {
            if (!candidates || candidates.length === 0) {
                this.framesSinceSeen++;
                if (this.framesSinceSeen >= DISAPPEAR_FRAMES_BEFORE_UNLOCK) this.lockedBox = null;
                return null;
            }

            const scored = candidates.map((c) => ({ c, score: scoreCandidate(c.box, frameWidth, frameHeight) }));
            scored.sort((a, b) => b.score - a.score);
            const best = scored[0];

            if (this.lockedBox === null) {
                this.lockedBox = best.c.box;
                this.framesSinceSeen = 0;
                this.switchCandidateStreak = 0;
                return best.c;
            }

            let lockMatch = null;
            let lockMatchIoU = 0;
            for (const s of scored) {
                const overlap = iou(s.c.box, this.lockedBox);
                if (overlap > lockMatchIoU) {
                    lockMatchIoU = overlap;
                    lockMatch = s;
                }
            }

            if (lockMatch && lockMatchIoU > 0.1) {
                this.framesSinceSeen = 0;
                if (best.c !== lockMatch.c && best.score > lockMatch.score * SWITCH_SCORE_MARGIN) {
                    this.switchCandidateStreak++;
                    if (this.switchCandidateStreak >= SWITCH_CONSECUTIVE_FRAMES) {
                        this.lockedBox = best.c.box;
                        this.switchCandidateStreak = 0;
                        return best.c;
                    }
                } else {
                    this.switchCandidateStreak = 0;
                }
                this.lockedBox = lockMatch.c.box;
                return lockMatch.c;
            }

            this.framesSinceSeen++;
            if (this.framesSinceSeen >= DISAPPEAR_FRAMES_BEFORE_UNLOCK) {
                this.lockedBox = best.c.box;
                this.framesSinceSeen = 0;
                this.switchCandidateStreak = 0;
                return best.c;
            }
            return null;
        }

        reset() {
            this.lockedBox = null;
            this.framesSinceSeen = 0;
            this.switchCandidateStreak = 0;
        }
    }

    // ==================== EMA (scoring/ema.ts) ====================

    class ExponentialMovingAverage {
        constructor(alpha = 0.15) {
            this.alpha = alpha;
            this.value = null;
        }

        update(newValue) {
            this.value = this.value === null ? newValue : this.alpha * newValue + (1 - this.alpha) * this.value;
            return this.value;
        }

        getValue() {
            return this.value;
        }

        reset() {
            this.value = null;
        }
    }

    // ==================== CONTEXT HELPERS (ml-engine.ts) ====================
    // Not weighted in the extension (legacy weights only), but implemented
    // with the web semantics so scoreFeature never returns a silent constant.

    function computeContextAwarePitch(metrics) {
        const pitch = metrics.pitch;
        if (pitch === null || pitch === undefined) return 50;

        if (pitch < -15) {
            const handProx = metrics.handObjectProximity;
            const objects = metrics.detectedObjects || [];
            const hasBook = objects.some((o) => o.class === 'book' || o.class === 'notebook');
            const hasPhone = objects.some((o) => o.class === 'phone');
            if (handProx === 'holding_phone' || hasPhone) return 20;
            if (handProx === 'holding_book' || hasBook) return 85;
            if ((metrics.ear ?? 0.3) < 0.22 && metrics.perclos > 20) return 15;
            return 40;
        }

        const absPitch = Math.abs(pitch);
        if (absPitch <= OPTIMAL_RANGES.pitch.maxDeviation) {
            return 100 - (absPitch / OPTIMAL_RANGES.pitch.maxDeviation) * 20;
        }
        return Math.max(30, 80 - (absPitch - OPTIMAL_RANGES.pitch.maxDeviation) * 2);
    }

    function computeObjectContext(metrics) {
        const objects = metrics.detectedObjects || [];
        if (objects.length === 0) return 65;
        if (objects.some((o) => o.class === 'phone')) return 25;
        const hasBook = objects.some((o) => o.class === 'book' || o.class === 'notebook');
        if (hasBook) return (metrics.pitch ?? 0) < -10 ? 80 : 70;
        const hasLaptop = objects.some((o) => o.class === 'laptop');
        if (hasLaptop) return Math.abs(metrics.pitch ?? 0) < 15 ? 80 : 65;
        return 65;
    }

    function encodeHandProximity(proximity) {
        switch (proximity) {
            case 'holding_book': return 80;
            case 'holding_pen': return 75;
            case 'hands_on_desk': return 65;
            case 'hands_not_visible': return 60;
            case 'holding_phone': return 20;
            default: return 60;
        }
    }

    function encodeTemporalPattern(behaviorResult) {
        return behaviorResult && typeof behaviorResult.attentionScore === 'number'
            ? behaviorResult.attentionScore
            : 60;
    }

    // ==================== FEATURE SCORING (ml-engine.ts scoreFeature) ====================

    /**
     * Score a single feature -> 0-100 (higher = more attentive). Same cases,
     * constants and quality factors as the web scoreFeature. `weights` is the
     * table used for the contribution (the web legacy path uses the upgraded
     * table first; here the table actually used for scoring is passed).
     */
    function scoreFeature(name, value, weights) {
        if (value === null || value === undefined) return { score: 50, contribution: 0, quality: 0 };

        let score = 50;
        let quality = 1;

        switch (name) {
            case 'ear':
                if (value >= OPTIMAL_RANGES.ear.min && value <= OPTIMAL_RANGES.ear.max) {
                    score = 85 + ((value - OPTIMAL_RANGES.ear.min) / (OPTIMAL_RANGES.ear.max - OPTIMAL_RANGES.ear.min)) * 15;
                } else if (value < OPTIMAL_RANGES.ear.min) {
                    score = Math.max(20, 85 * (value / OPTIMAL_RANGES.ear.min));
                } else {
                    score = 90;
                }
                break;

            case 'perclos':
                score = value <= OPTIMAL_RANGES.perclos.max
                    ? 100 - (value / OPTIMAL_RANGES.perclos.max) * 20
                    : Math.max(10, 80 - (value - OPTIMAL_RANGES.perclos.max) * 2);
                break;

            case 'blinkRate':
                if (value >= OPTIMAL_RANGES.blinkRate.min && value <= OPTIMAL_RANGES.blinkRate.max) {
                    score = 90;
                } else if (value < OPTIMAL_RANGES.blinkRate.min) {
                    score = 70 + (value / OPTIMAL_RANGES.blinkRate.min) * 20;
                } else {
                    score = Math.max(40, 90 - (value - OPTIMAL_RANGES.blinkRate.max) * 3);
                }
                quality = value > 0 ? 1 : 0.5;
                break;

            case 'gazeDeviation':
                score = value <= OPTIMAL_RANGES.gaze.maxDeviation
                    ? 100 - (value / OPTIMAL_RANGES.gaze.maxDeviation) * 30
                    : Math.max(30, 70 - (value - OPTIMAL_RANGES.gaze.maxDeviation) * 100);
                break;

            case 'headPoseStability':
                score = value;
                break;

            case 'yawDeviation':
            case 'pitchDeviation': {
                const maxDev = OPTIMAL_RANGES.yaw.maxDeviation;
                score = Math.abs(value) <= maxDev
                    ? 100 - (Math.abs(value) / maxDev) * 20
                    : Math.max(30, 80 - (Math.abs(value) - maxDev) * 2);
                break;
            }

            case 'fatigueIndex':
                score = value <= OPTIMAL_RANGES.fatigueIndex.max
                    ? 100 - (value / OPTIMAL_RANGES.fatigueIndex.max) * 20
                    : Math.max(20, 80 - (value - OPTIMAL_RANGES.fatigueIndex.max));
                break;

            case 'fidgetScore':
                score = value <= OPTIMAL_RANGES.fidgetScore.max
                    ? 100 - (value / OPTIMAL_RANGES.fidgetScore.max) * 20
                    : Math.max(30, 80 - (value - OPTIMAL_RANGES.fidgetScore.max));
                break;

            case 'eyebrowRaise':
                score = value >= 0.2 && value <= 0.6 ? 80 + (0.5 - Math.abs(value - 0.4)) * 40 : 60;
                quality = 0.7;
                break;

            case 'mouthOpenness':
                score = value < 0.1 ? 85 : value < 0.3 ? 70 : 40;
                quality = 0.6;
                break;

            case 'faceDistance':
                score = value >= OPTIMAL_RANGES.faceDistance.min && value <= OPTIMAL_RANGES.faceDistance.max ? 90 : 60;
                quality = 0.5;
                break;

            case 'facePresence':
                score = value >= 0.05 ? 100 : 20;
                break;

            // Context-aware features arrive as pre-computed 0-100 scores from
            // the helpers above; pass them through (clamped), quality 1.
            case 'contextAwarePitch':
            case 'objectContext':
            case 'handObjectProximity':
            case 'temporalPattern':
                score = Math.max(0, Math.min(100, value));
                break;

            default:
                break;
        }

        const table = weights || FEATURE_WEIGHTS_LEGACY;
        const weight = table[name] ?? 0;
        const contribution = (score - 50) * weight * quality;
        return { score, contribution, quality };
    }

    function classifyScore(score) {
        if (score >= THRESHOLDS.ATTENTIVE_MIN) return 'Attentive';
        if (score >= THRESHOLDS.DISTRACTED_MIN) return 'Distracted';
        return 'Multitasking';
    }

    // ==================== ML ENGINE (ml-engine.ts legacy predict path) ====================

    class MLEngine {
        constructor() {
            this.scoreEMA = new ExponentialMovingAverage(0.15);
            this.predictionHistory = [];
            this.MAX_HISTORY = 100;
        }

        /**
         * Same weight-selection rule as the web predictLegacy():
         * LEGACY weights when `metrics.detectedObjects === undefined`
         * (always the case in the extension), UPGRADED otherwise.
         */
        predict(metrics) {
            const useUpgraded = metrics.detectedObjects !== undefined;
            const weights = useUpgraded ? FEATURE_WEIGHTS_V1_UPGRADED : FEATURE_WEIGHTS_LEGACY;

            const gazeDeviation = metrics.gazeX !== null && metrics.gazeY !== null &&
                metrics.gazeX !== undefined && metrics.gazeY !== undefined
                ? Math.sqrt(metrics.gazeX ** 2 + metrics.gazeY ** 2)
                : null;

            const featureValues = {
                ear: metrics.ear,
                perclos: metrics.perclos,
                blinkRate: metrics.blinkRate,
                gazeDeviation,
                headPoseStability: metrics.headPoseStability,
                yawDeviation: metrics.yaw,
                fatigueIndex: metrics.fatigueIndex,
                fidgetScore: metrics.fidgetScore,
                eyebrowRaise: metrics.eyebrowRaise ?? null,
                mouthOpenness: metrics.mouthOpenness,
                faceDistance: metrics.faceDistance,
                facePresence: metrics.faceArea,
            };

            if (useUpgraded) {
                featureValues.contextAwarePitch = computeContextAwarePitch(metrics);
                featureValues.objectContext = computeObjectContext(metrics);
                featureValues.handObjectProximity = encodeHandProximity(metrics.handObjectProximity);
                featureValues.temporalPattern = encodeTemporalPattern(metrics.behaviorResult);
            } else {
                featureValues.pitchDeviation = metrics.pitch;
            }

            const featureScores = {};
            let totalWeight = 0;
            let weightedSum = 0;
            let qualitySum = 0;

            for (const [name, value] of Object.entries(featureValues)) {
                const weight = weights[name] ?? 0;
                if (weight === 0 || value === null || value === undefined) continue;
                const result = scoreFeature(name, value, weights);
                featureScores[name] = result;
                weightedSum += result.score * weight * result.quality;
                totalWeight += weight * result.quality;
                qualitySum += result.quality;
            }

            const rawScore = totalWeight > 0 ? weightedSum / totalWeight : 50;
            const smoothedScore = this.scoreEMA.update(rawScore);

            const featureCount = Object.values(featureValues).filter((v) => v !== null && v !== undefined).length;
            const maxFeatures = Object.keys(weights).length;
            const completeness = featureCount / maxFeatures;
            const avgQuality = featureCount > 0 ? qualitySum / featureCount : 0;
            const confidence = completeness * avgQuality * 0.9 + 0.1;

            const featureImportance = {};
            const featureContributions = {};
            for (const [name, result] of Object.entries(featureScores)) {
                featureImportance[name] = weights[name] ?? 0;
                featureContributions[name] = result.contribution;
            }

            const prediction = {
                attentionState: classifyScore(smoothedScore),
                confidence,
                rawScore: Math.round(smoothedScore),
                unroundedScore: smoothedScore,
                scoringMode: useUpgraded ? 'upgraded' : 'legacy',
                featureImportance,
                featureContributions,
            };

            this.predictionHistory.push(prediction);
            if (this.predictionHistory.length > this.MAX_HISTORY) this.predictionHistory.shift();
            return prediction;
        }

        reset() {
            this.scoreEMA.reset();
            this.predictionHistory = [];
        }
    }

    // ==================== EXPORT ====================

    const api = Object.freeze({
        VERSION,
        SCORING_CORE_VERSION,
        THRESHOLDS,
        OPTIMAL_RANGES,
        FEATURE_WEIGHTS_LEGACY,
        FEATURE_WEIGHTS_V1_UPGRADED,
        HEAD_POSE_CALIBRATION,
        PROXIMITY_CODES_V1,
        EAR_BLINK_THRESHOLD,
        PREDICT_INTERVAL_MS,
        FRAME_INTERVAL_MS,
        distance,
        clamp,
        calculateVariance,
        calculateEAR,
        calculateMAR,
        estimateFaceDistance,
        calculateFaceArea,
        calculateEyebrowRaise,
        calculateMouthOpenness,
        calculateHeadPose,
        calculateGazeDirection,
        computeContextAwarePitch,
        computeObjectContext,
        encodeHandProximity,
        encodeTemporalPattern,
        scoreFeature,
        classifyScore,
        BlinkDetector,
        HeadPoseTracker,
        FatigueCalculator,
        HandObjectProximityDetector,
        FaceMetricsAggregator,
        PrimaryFaceSelector,
        ExponentialMovingAverage,
        MLEngine,
    });

    ns.mlCore = api;

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
    }
})(typeof window !== 'undefined' ? window : typeof self !== 'undefined' ? self : globalThis);
