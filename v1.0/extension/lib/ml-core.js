/**
 * CognEn Extension - ML Core Library v3
 * VERY STRICT thresholds - looking away = immediate penalty
 */

// ==================== CONSTANTS ====================

const EAR_BLINK_THRESHOLD = 0.21;
const PERCLOS_WINDOW_MS = 60000;
const NORMAL_BLINK_RATE_MIN = 15;
const NORMAL_BLINK_RATE_MAX = 20;
const FATIGUE_EAR_THRESHOLD = 0.25;

// VERY STRICT THRESHOLDS
const THRESHOLDS = {
    ATTENTIVE_MIN: 80,  // Must be very engaged
    DISTRACTED_MIN: 50, // Middle ground
};

// Feature weights - HEAVILY emphasize gaze and head pose
const FEATURE_WEIGHTS = {
    gazeDeviation: 0.25,      // PRIMARY - gaze is critical
    yawDeviation: 0.20,       // Looking left/right is BAD
    pitchDeviation: 0.12,     // Looking down at phone etc
    headPoseStability: 0.10,
    ear: 0.08,
    perclos: 0.08,
    fatigueIndex: 0.06,
    blinkRate: 0.04,
    fidgetScore: 0.03,
    mouthOpenness: 0.02,
    faceDistance: 0.01,
    facePresence: 0.01,
};

// ==================== UTILITY FUNCTIONS ====================

function distance(p1, p2) {
    return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function calculateVariance(values) {
    if (values.length === 0) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
}

// ==================== FACE GEOMETRY ====================

function calculateEAR(landmarks) {
    if (!landmarks || landmarks.length < 68) return null;

    const leftEye = [36, 37, 38, 39, 40, 41].map(i => landmarks[i]);
    const rightEye = [42, 43, 44, 45, 46, 47].map(i => landmarks[i]);

    const calcSingleEAR = (eye) => {
        const vertDist = distance(eye[1], eye[5]) + distance(eye[2], eye[4]);
        const horzDist = distance(eye[0], eye[3]);
        if (horzDist === 0) return null;
        return vertDist / (2 * horzDist);
    };

    const leftEAR = calcSingleEAR(leftEye);
    const rightEAR = calcSingleEAR(rightEye);

    if (leftEAR === null || rightEAR === null) return null;
    return (leftEAR + rightEAR) / 2;
}

function calculateMAR(landmarks) {
    if (!landmarks || landmarks.length < 68) return null;

    const mouthPoints = [48, 54, 51, 57].map(i => landmarks[i]);
    const [pLeft, pRight, pTop, pBottom] = mouthPoints;

    const vertDist = distance(pTop, pBottom);
    const horzDist = distance(pLeft, pRight);

    if (horzDist === 0) return null;
    return vertDist / horzDist;
}

function calculateHeadPose(landmarks) {
    if (!landmarks || landmarks.length < 68) return { yaw: null, pitch: null, roll: null };

    const nose = landmarks[30];
    const leftEye = landmarks[36];
    const rightEye = landmarks[45];
    const chin = landmarks[8];
    const leftCheek = landmarks[0];
    const rightCheek = landmarks[16];

    if (!nose || !leftEye || !rightEye || !chin) return { yaw: null, pitch: null, roll: null };

    const eyeMidpoint = {
        x: (leftEye.x + rightEye.x) / 2,
        y: (leftEye.y + rightEye.y) / 2,
    };

    const eyeDist = distance(leftEye, rightEye);
    const noseChinDist = distance(nose, chin);
    const faceWidth = distance(leftCheek, rightCheek);

    if (eyeDist === 0 || noseChinDist === 0) return { yaw: null, pitch: null, roll: null };

    // MORE SENSITIVE yaw calculation using nose-to-cheek asymmetry
    const leftNoseDist = distance(nose, leftCheek);
    const rightNoseDist = distance(nose, rightCheek);
    const asymmetry = (rightNoseDist - leftNoseDist) / faceWidth;
    const yaw = asymmetry * 100;  // Very sensitive to turning

    // Pitch from nose position relative to eye-chin line
    const pitch = ((nose.y - eyeMidpoint.y) / noseChinDist - 0.35) * 100;

    // Roll from eye angle
    const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * (180 / Math.PI);

    return { yaw, pitch, roll };
}

function calculateGazeDirection(landmarks) {
    if (!landmarks || landmarks.length < 68) return { gazeX: null, gazeY: null };

    // Left eye
    const leftEyeLeft = landmarks[36];
    const leftEyeRight = landmarks[39];
    const leftEyeCenter = {
        x: (leftEyeLeft.x + leftEyeRight.x) / 2,
        y: (leftEyeLeft.y + leftEyeRight.y) / 2,
    };
    const leftIrisCenter = {
        x: (landmarks[37].x + landmarks[38].x + landmarks[40].x + landmarks[41].x) / 4,
        y: (landmarks[37].y + landmarks[38].y + landmarks[40].y + landmarks[41].y) / 4,
    };

    // Right eye
    const rightEyeLeft = landmarks[42];
    const rightEyeRight = landmarks[45];
    const rightEyeCenter = {
        x: (rightEyeLeft.x + rightEyeRight.x) / 2,
        y: (rightEyeLeft.y + rightEyeRight.y) / 2,
    };
    const rightIrisCenter = {
        x: (landmarks[43].x + landmarks[44].x + landmarks[46].x + landmarks[47].x) / 4,
        y: (landmarks[43].y + landmarks[44].y + landmarks[46].y + landmarks[47].y) / 4,
    };

    const leftEyeWidth = distance(leftEyeLeft, leftEyeRight);
    const rightEyeWidth = distance(rightEyeLeft, rightEyeRight);

    if (leftEyeWidth === 0 || rightEyeWidth === 0) return { gazeX: null, gazeY: null };

    // Calculate gaze X (left-right) - VERY SENSITIVE
    const leftGazeX = ((leftIrisCenter.x - leftEyeCenter.x) / leftEyeWidth) * 3;
    const rightGazeX = ((rightIrisCenter.x - rightEyeCenter.x) / rightEyeWidth) * 3;

    // Calculate gaze Y (up-down) - VERY SENSITIVE
    const leftEyeHeight = distance(landmarks[37], landmarks[41]) || 1;
    const rightEyeHeight = distance(landmarks[43], landmarks[47]) || 1;
    const leftGazeY = ((leftIrisCenter.y - leftEyeCenter.y) / leftEyeHeight) * 3;
    const rightGazeY = ((rightIrisCenter.y - rightEyeCenter.y) / rightEyeHeight) * 3;

    return {
        gazeX: clamp((leftGazeX + rightGazeX) / 2, -1.5, 1.5),
        gazeY: clamp((leftGazeY + rightGazeY) / 2, -1.5, 1.5),
    };
}

function estimateFaceDistance(landmarks, frameWidth) {
    if (!landmarks || landmarks.length < 68) return null;

    const ipd = distance(landmarks[36], landmarks[45]);
    const normalizedIPD = ipd / frameWidth;

    return clamp((normalizedIPD - 0.1) / 0.3, 0, 1);
}

function calculateFaceArea(landmarks, frameWidth, frameHeight) {
    if (!landmarks || landmarks.length < 68) return null;

    const xs = landmarks.map(p => p.x);
    const ys = landmarks.map(p => p.y);

    const faceArea = (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
    return faceArea / (frameWidth * frameHeight);
}

function calculateMouthOpenness(landmarks) {
    if (!landmarks || landmarks.length < 68) return null;

    const topLip = landmarks[62];
    const bottomLip = landmarks[66];
    const leftCorner = landmarks[60];
    const rightCorner = landmarks[64];

    const mouthHeight = distance(topLip, bottomLip);
    const mouthWidth = distance(leftCorner, rightCorner);

    if (mouthWidth === 0) return null;
    return clamp(mouthHeight / mouthWidth, 0, 1);
}

// ==================== BLINK DETECTOR ====================

class BlinkDetector {
    constructor() {
        this.earHistory = [];
        this.blinkEvents = [];
        this.wasEyeClosed = false;
        this.eyeClosureStart = null;
        this.lastBlinkTime = 0;
    }

    detectBlink(ear, timestamp) {
        if (ear === null) {
            return { blinkDetected: false, blinkRate: this.getBlinkRate(timestamp), perclos: this.getPERCLOS(timestamp) };
        }

        this.earHistory.push({ time: timestamp, ear });

        // Keep 2 minutes of history
        const cutoff = timestamp - 120000;
        this.earHistory = this.earHistory.filter(e => e.time > cutoff);
        this.blinkEvents = this.blinkEvents.filter(t => t > cutoff);

        const isEyeClosed = ear < EAR_BLINK_THRESHOLD;
        let blinkDetected = false;

        // Detect blink start
        if (isEyeClosed && !this.wasEyeClosed) {
            this.eyeClosureStart = timestamp;
        }

        // Detect blink end
        if (this.wasEyeClosed && !isEyeClosed && this.eyeClosureStart) {
            const blinkDuration = timestamp - this.eyeClosureStart;
            // Valid blink: 50-400ms, and at least 200ms since last blink
            if (blinkDuration >= 50 && blinkDuration <= 400 && (timestamp - this.lastBlinkTime) > 200) {
                blinkDetected = true;
                this.blinkEvents.push(timestamp);
                this.lastBlinkTime = timestamp;
                console.log('[BlinkDetector] Blink detected! Total:', this.blinkEvents.length);
            }
            this.eyeClosureStart = null;
        }

        this.wasEyeClosed = isEyeClosed;

        const blinkRate = this.getBlinkRate(timestamp);
        const perclos = this.getPERCLOS(timestamp);

        return { blinkDetected, blinkRate, perclos };
    }

    getBlinkRate(currentTime) {
        if (this.blinkEvents.length === 0) return 0;

        const now = currentTime || Date.now();
        const oneMinuteAgo = now - 60000;
        const recentBlinks = this.blinkEvents.filter(t => t > oneMinuteAgo);

        // If we have less than 60 seconds of data, scale up
        if (this.earHistory.length < 50) return 0;

        const oldestData = this.earHistory[0].time;
        const timeSpan = now - oldestData;

        if (timeSpan < 5000) return 0; // Need at least 5 seconds

        // Calculate actual blinks per minute
        const actualTimeMinutes = Math.min(timeSpan, 60000) / 60000;
        return recentBlinks.length / actualTimeMinutes;
    }

    getPERCLOS(currentTime) {
        const oneMinuteAgo = currentTime - PERCLOS_WINDOW_MS;
        const recentHistory = this.earHistory.filter(e => e.time > oneMinuteAgo);

        if (recentHistory.length < 10) return 0;

        const closedCount = recentHistory.filter(e => e.ear < EAR_BLINK_THRESHOLD).length;
        return (closedCount / recentHistory.length) * 100;
    }

    getEARStats() {
        if (this.earHistory.length === 0) return { mean: 0, variance: 0 };

        const ears = this.earHistory.map(e => e.ear);
        const mean = ears.reduce((a, b) => a + b, 0) / ears.length;

        return { mean, variance: calculateVariance(ears) };
    }

    reset() {
        this.earHistory = [];
        this.blinkEvents = [];
        this.wasEyeClosed = false;
        this.eyeClosureStart = null;
        this.lastBlinkTime = 0;
    }
}

// ==================== HEAD POSE TRACKER ====================

class HeadPoseTracker {
    constructor() {
        this.poseHistory = [];
    }

    addPose(pose, timestamp) {
        if (pose.yaw !== null && pose.pitch !== null && pose.roll !== null) {
            this.poseHistory.push({ time: timestamp, ...pose });

            const cutoff = timestamp - 30000;
            this.poseHistory = this.poseHistory.filter(p => p.time > cutoff);
        }
    }

    getStability() {
        if (this.poseHistory.length < 5) return 100;

        const yaws = this.poseHistory.map(p => p.yaw);
        const pitches = this.poseHistory.map(p => p.pitch);
        const rolls = this.poseHistory.map(p => p.roll);

        const combinedVar = (calculateVariance(yaws) + calculateVariance(pitches) + calculateVariance(rolls) * 0.5) / 2.5;

        return clamp(100 - (combinedVar / 3), 0, 100);  // More sensitive
    }

    getFidgetScore() {
        if (this.poseHistory.length < 10) return 0;

        const diffs = [];
        for (let i = 1; i < this.poseHistory.length; i++) {
            const prev = this.poseHistory[i - 1];
            const curr = this.poseHistory[i];
            const diff = Math.abs(curr.yaw - prev.yaw) + Math.abs(curr.pitch - prev.pitch) + Math.abs(curr.roll - prev.roll);
            diffs.push(diff);
        }

        const avgDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
        return clamp(avgDiff * 10, 0, 100);
    }

    reset() {
        this.poseHistory = [];
    }
}

// ==================== FATIGUE CALCULATOR ====================

class FatigueCalculator {
    constructor() {
        this.perclosHistory = [];
        this.blinkRateHistory = [];
        this.earMeanHistory = [];
    }

    update(perclos, blinkRate, earMean, timestamp) {
        this.perclosHistory.push({ time: timestamp, value: perclos });
        this.blinkRateHistory.push({ time: timestamp, value: blinkRate });
        this.earMeanHistory.push({ time: timestamp, value: earMean });

        const cutoff = timestamp - 300000;
        this.perclosHistory = this.perclosHistory.filter(p => p.time > cutoff);
        this.blinkRateHistory = this.blinkRateHistory.filter(p => p.time > cutoff);
        this.earMeanHistory = this.earMeanHistory.filter(p => p.time > cutoff);

        return this.calculateFatigueIndex();
    }

    calculateFatigueIndex() {
        if (this.perclosHistory.length < 3) return 0;

        const avgPerclos = this.perclosHistory.reduce((a, b) => a + b.value, 0) / this.perclosHistory.length;
        const perclosScore = clamp(avgPerclos * 3, 0, 50);

        const avgBlinkRate = this.blinkRateHistory.reduce((a, b) => a + b.value, 0) / this.blinkRateHistory.length;
        let blinkScore = 0;
        if (avgBlinkRate > 0) {
            if (avgBlinkRate < NORMAL_BLINK_RATE_MIN) {
                blinkScore = clamp((NORMAL_BLINK_RATE_MIN - avgBlinkRate) * 3, 0, 30);
            } else if (avgBlinkRate > NORMAL_BLINK_RATE_MAX) {
                blinkScore = clamp((avgBlinkRate - NORMAL_BLINK_RATE_MAX) * 2, 0, 30);
            }
        }

        let earTrendScore = 0;
        if (this.earMeanHistory.length > 5) {
            const recentEars = this.earMeanHistory.slice(-5).map(e => e.value).filter(v => v > 0);
            if (recentEars.length > 0) {
                const avgEar = recentEars.reduce((a, b) => a + b, 0) / recentEars.length;
                if (avgEar < FATIGUE_EAR_THRESHOLD) {
                    earTrendScore = clamp((FATIGUE_EAR_THRESHOLD - avgEar) * 200, 0, 30);
                }
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

// ==================== FACE METRICS AGGREGATOR ====================

class FaceMetricsAggregator {
    constructor() {
        this.blinkDetector = new BlinkDetector();
        this.headPoseTracker = new HeadPoseTracker();
        this.fatigueCalculator = new FatigueCalculator();
        this.marHistory = [];
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

        const gaze = calculateGazeDirection(landmarks);
        const faceDistance = estimateFaceDistance(landmarks, frameWidth);
        const faceArea = calculateFaceArea(landmarks, frameWidth, frameHeight);
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
            mouthOpenness,
            earVariance: earStats.variance,
        };
    }

    reset() {
        this.blinkDetector.reset();
        this.headPoseTracker.reset();
        this.fatigueCalculator.reset();
        this.marHistory = [];
    }
}

// ==================== ML ENGINE ====================

class ExponentialMovingAverage {
    constructor(alpha = 0.3) {
        this.alpha = alpha;
        this.value = null;
    }

    update(newValue) {
        if (this.value === null) {
            this.value = newValue;
        } else {
            this.value = this.alpha * newValue + (1 - this.alpha) * this.value;
        }
        return this.value;
    }

    reset() {
        this.value = null;
    }
}

// VERY STRICT feature scoring
function scoreFeature(name, value) {
    if (value === null || isNaN(value)) {
        return { score: 30, quality: 0 };  // Unknown = assume bad
    }

    let score = 50;
    let quality = 1;

    switch (name) {
        case 'gazeDeviation':
            // VERY STRICT - any gaze away is BAD
            if (value <= 0.1) {
                score = 100;  // Looking straight
            } else if (value <= 0.25) {
                score = 70 - ((value - 0.1) / 0.15) * 40;  // Drops to 30
            } else if (value <= 0.5) {
                score = 30 - ((value - 0.25) / 0.25) * 20;  // Drops to 10
            } else {
                score = 10;  // Looking away = very bad
            }
            break;

        case 'yawDeviation':
            // VERY STRICT - turning head is BAD
            const absYaw = Math.abs(value);
            if (absYaw <= 5) {
                score = 100;  // Facing forward
            } else if (absYaw <= 12) {
                score = 80 - ((absYaw - 5) / 7) * 40;  // Drops to 40
            } else if (absYaw <= 25) {
                score = 40 - ((absYaw - 12) / 13) * 25;  // Drops to 15
            } else {
                score = 10;  // Turned away
            }
            break;

        case 'pitchDeviation':
            // Looking up/down
            const absPitch = Math.abs(value);
            if (absPitch <= 8) {
                score = 95;
            } else if (absPitch <= 18) {
                score = 75 - ((absPitch - 8) / 10) * 35;
            } else {
                score = Math.max(15, 40 - (absPitch - 18) * 1.5);
            }
            break;

        case 'headPoseStability':
            score = value;  // Direct mapping
            break;

        case 'ear':
            if (value >= 0.28) {
                score = 95;  // Eyes fully open
            } else if (value >= 0.22) {
                score = 75;  // Eyes normal
            } else {
                score = Math.max(20, value / 0.22 * 60);  // Eyes closing
            }
            break;

        case 'perclos':
            if (value <= 8) {
                score = 95;
            } else if (value <= 20) {
                score = 80 - (value - 8);
            } else {
                score = Math.max(10, 68 - (value - 20) * 2);
            }
            break;

        case 'fatigueIndex':
            if (value <= 15) {
                score = 95;
            } else if (value <= 40) {
                score = 85 - (value - 15);
            } else {
                score = Math.max(20, 60 - (value - 40));
            }
            break;

        case 'blinkRate':
            if (value >= 12 && value <= 20) {
                score = 90;
            } else if (value === 0) {
                score = 70;  // No data yet
                quality = 0.5;
            } else if (value < 12) {
                score = 60;  // Staring
            } else {
                score = Math.max(50, 90 - (value - 20) * 3);
            }
            break;

        case 'fidgetScore':
            if (value <= 10) {
                score = 95;
            } else if (value <= 30) {
                score = 85 - (value - 10);
            } else {
                score = Math.max(30, 65 - (value - 30));
            }
            break;

        case 'mouthOpenness':
            if (value < 0.08) score = 90;
            else if (value < 0.2) score = 70;
            else score = 40;  // Yawning
            quality = 0.4;
            break;

        case 'faceDistance':
        case 'facePresence':
            score = 80;  // These are less important
            quality = 0.3;
            break;
    }

    return { score, quality };
}

class MLEngine {
    constructor() {
        this.scoreEMA = new ExponentialMovingAverage(0.35);  // More responsive
        this.predictionHistory = [];
    }

    predict(metrics) {
        // Calculate gaze deviation
        const gazeDeviation = metrics.gazeX !== null && metrics.gazeY !== null
            ? Math.sqrt(metrics.gazeX ** 2 + metrics.gazeY ** 2)
            : 0.5;  // Assume looking away if no data

        const featureValues = {
            gazeDeviation,
            yawDeviation: metrics.yaw,
            pitchDeviation: metrics.pitch,
            headPoseStability: metrics.headPoseStability,
            ear: metrics.ear,
            perclos: metrics.perclos,
            fatigueIndex: metrics.fatigueIndex,
            blinkRate: metrics.blinkRate,
            fidgetScore: metrics.fidgetScore,
            mouthOpenness: metrics.mouthOpenness,
            faceDistance: metrics.faceDistance,
            facePresence: metrics.faceArea,
        };

        let totalWeight = 0;
        let weightedSum = 0;

        for (const [name, value] of Object.entries(featureValues)) {
            const result = scoreFeature(name, value);
            const weight = FEATURE_WEIGHTS[name] || 0;

            if (weight > 0) {
                weightedSum += result.score * weight * result.quality;
                totalWeight += weight * result.quality;
            }
        }

        let rawScore = totalWeight > 0 ? weightedSum / totalWeight : 40;
        const smoothedScore = this.scoreEMA.update(rawScore);

        let attentionState;
        if (smoothedScore >= THRESHOLDS.ATTENTIVE_MIN) {
            attentionState = 'Attentive';
        } else if (smoothedScore >= THRESHOLDS.DISTRACTED_MIN) {
            attentionState = 'Distracted';
        } else {
            attentionState = 'Multitasking';
        }

        const prediction = {
            attentionState,
            confidence: 0.85,
            rawScore: Math.round(smoothedScore),
        };

        this.predictionHistory.push(prediction);
        if (this.predictionHistory.length > 100) {
            this.predictionHistory.shift();
        }

        return prediction;
    }

    reset() {
        this.scoreEMA.reset();
        this.predictionHistory = [];
    }
}

// ==================== FEATURE WEIGHTS (UPGRADED) ====================
// ARCHITECTURE.md §2 — replaces pitchDeviation when context is available.

const FEATURE_WEIGHTS_V1_UPGRADED = {
    // Eye-based
    ear: 0.10,
    perclos: 0.08,
    blinkRate: 0.05,
    gazeDeviation: 0.12,
    // Head pose (pitchDeviation removed — replaced below)
    headPoseStability: 0.10,
    yawDeviation: 0.08,
    // Fatigue
    fatigueIndex: 0.07,
    fidgetScore: 0.04,
    mouthOpenness: 0.02,
    faceDistance: 0.02,
    facePresence: 0.02,
    // NEW context-aware features
    contextAwarePitch: 0.12,
    objectContext: 0.10,
    handObjectProximity: 0.05,
    temporalPattern: 0.03,
};

// ==================== CONTEXT HELPERS ====================

/**
 * Context-aware pitch score (ARCHITECTURE.md §2.2).
 * Phone+downward→20, book+downward→85, drowsy→15, unknown-down→40.
 */
function computeContextAwarePitch(metrics) {
    const pitch = metrics.pitch;
    if (pitch === null || pitch === undefined) return 50;

    if (pitch < -15) {
        const handProx = metrics.handObjectProximity;
        const objects = metrics.detectedObjects || [];
        const hasBook = objects.some(o => o.class === 'book' || o.class === 'notebook');
        const hasPhone = objects.some(o => o.class === 'phone');

        if (handProx === 'holding_phone' || hasPhone) return 20;
        if (handProx === 'holding_book' || hasBook) return 85;
        if ((metrics.ear || 0.3) < 0.22 && metrics.perclos > 20) return 15;
        return 40;
    }

    const absPitch = Math.abs(pitch);
    if (absPitch <= 15) return 100 - (absPitch / 15) * 20;
    return Math.max(30, 80 - (absPitch - 15) * 2);
}

function computeObjectContext(metrics) {
    const objects = metrics.detectedObjects || [];
    if (objects.length === 0) return 65;

    if (objects.some(o => o.class === 'phone')) return 25;

    const hasBook = objects.some(o => o.class === 'book' || o.class === 'notebook');
    if (hasBook) return (metrics.pitch || 0) < -10 ? 80 : 70;

    const hasLaptop = objects.some(o => o.class === 'laptop');
    if (hasLaptop) return Math.abs(metrics.pitch || 0) < 15 ? 80 : 65;

    return 65;
}

function encodeHandProximity(proximity) {
    switch (proximity) {
        case 'holding_book':      return 80;
        case 'holding_pen':       return 75;
        case 'hands_on_desk':     return 65;
        case 'hands_not_visible': return 60;
        case 'holding_phone':     return 20;
        default:                  return 60;
    }
}

function encodeTemporalPattern(behaviorResult) {
    return behaviorResult ? behaviorResult.attentionScore : 60;
}

// ==================== HAND-OBJECT PROXIMITY DETECTOR ====================

/**
 * HandObjectProximityDetector (ARCHITECTURE.md §2.4)
 * Matches hand centroids to detected object bounding boxes.
 *
 * @param {Array} handResults       [{landmarks: [{x,y,z}×21], handedness}]
 * @param {Array} detectedObjects   [{class, confidence, bbox:{x,y,w,h}}]
 * @returns {string} HandObjectProximity value
 */
class HandObjectProximityDetector {
    constructor() {
        this.PROXIMITY_THRESHOLD = 0.20;
    }

    compute(handResults, detectedObjects) {
        if (!handResults || handResults.length === 0) return 'hands_not_visible';
        if (!detectedObjects || detectedObjects.length === 0) return 'hands_on_desk';

        const handCenters = handResults.map(h => this._handCenter(h.landmarks));

        for (const priority of ['phone', 'book', 'notebook', 'pen']) {
            const obj = detectedObjects.find(o => o.class === priority);
            if (!obj) continue;

            const objCenter = { x: obj.bbox.x, y: obj.bbox.y };
            for (const hand of handCenters) {
                if (this._dist(hand, objCenter) < this.PROXIMITY_THRESHOLD) {
                    if (priority === 'phone') return 'holding_phone';
                    if (priority === 'book' || priority === 'notebook') return 'holding_book';
                    if (priority === 'pen') return 'holding_pen';
                }
            }
        }
        return 'hands_on_desk';
    }

    _handCenter(landmarks) {
        const sum = landmarks.reduce((a, l) => ({ x: a.x + l.x, y: a.y + l.y }), { x: 0, y: 0 });
        return { x: sum.x / landmarks.length, y: sum.y / landmarks.length };
    }

    _dist(a, b) {
        return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
    }
}

// ==================== TEMPORAL BEHAVIOR CLASSIFIER ====================

/**
 * TemporalBehaviorClassifier (ARCHITECTURE.md §2.3)
 * 15-frame sliding window with rule-based classification.
 * No ONNX dependency in the extension — rule-based only.
 *
 * Behavior labels → attention scores:
 *   attentive_screen: 85, attentive_reading: 80, distracted: 45,
 *   using_phone: 25, drowsy: 20, absent: 0
 */
class TemporalBehaviorClassifier {
    constructor() {
        this.WINDOW_SIZE = 15;
        this.FEATURE_DIM = 13;
        this.featureBuffer = [];
        this.lastResult = { label: 'distracted', attentionScore: 50, confidence: 0, source: 'rule_based' };

        this.ATTENTION_SCORES = {
            attentive_screen: 85,
            attentive_reading: 80,
            distracted: 45,
            using_phone: 25,
            drowsy: 20,
            absent: 0,
        };
    }

    update(frame) {
        this.featureBuffer.push(this._extractFeatures(frame));
        if (this.featureBuffer.length > this.WINDOW_SIZE) this.featureBuffer.shift();

        if (this.featureBuffer.length < this.WINDOW_SIZE) {
            this.lastResult = this._singleFrameClassify(frame);
        } else {
            this.lastResult = this._temporalClassify();
        }
        return this.lastResult;
    }

    getLastResult() { return this.lastResult; }

    reset() {
        this.featureBuffer = [];
        this.lastResult = { label: 'distracted', attentionScore: 50, confidence: 0, source: 'rule_based' };
    }

    _extractFeatures(frame) {
        const { metrics, detectedObjects = [], handProximity = 'hands_not_visible' } = frame;
        const hasBook  = detectedObjects.some(o => o.class === 'book' || o.class === 'notebook') ? 1 : 0;
        const hasPhone = detectedObjects.some(o => o.class === 'phone') ? 1 : 0;
        const hasLaptop = detectedObjects.some(o => o.class === 'laptop') ? 1 : 0;
        const hasNone = !hasBook && !hasPhone && !hasLaptop ? 1 : 0;
        const proxCode = encodeHandProximity(handProximity) / 100;

        return [
            clamp(metrics.gazeX || 0, -1, 1),
            clamp(metrics.gazeY || 0, -1, 1),
            clamp((metrics.pitch || 0) / 90, -1, 1),
            clamp((metrics.yaw  || 0) / 90, -1, 1),
            clamp(metrics.ear || 0.3, 0, 1),
            clamp(metrics.perclos / 100, 0, 1),
            clamp(metrics.headPoseStability / 100, 0, 1),
            hasBook, hasPhone, hasLaptop, hasNone,
            clamp(metrics.fidgetScore / 100, 0, 1),
            proxCode,
        ];
    }

    _singleFrameClassify(frame) {
        const { metrics, detectedObjects = [], handProximity } = frame;

        if (!metrics.faceArea || metrics.faceArea < 0.005)
            return this._result('absent', 1.0);

        if ((metrics.ear || 0.3) < 0.22 && metrics.perclos > 20)
            return this._result('drowsy', 0.85);

        const pitch = metrics.pitch || 0;
        const isLookingDown = pitch < -15;
        const phoneDetected = detectedObjects.some(o => o.class === 'phone');
        if (phoneDetected && (handProximity === 'holding_phone' || isLookingDown))
            return this._result('using_phone', 0.80);

        const bookDetected = detectedObjects.some(o => o.class === 'book' || o.class === 'notebook');
        if (bookDetected && (handProximity === 'holding_book' || isLookingDown))
            return this._result('attentive_reading', 0.75);

        const gazeDev = Math.sqrt((metrics.gazeX || 0) ** 2 + (metrics.gazeY || 0) ** 2);
        if (gazeDev < 0.3 && Math.abs(pitch) < 15)
            return this._result('attentive_screen', 0.70);

        return this._result('distracted', 0.60);
    }

    _temporalClassify() {
        const avg = new Array(this.FEATURE_DIM).fill(0);
        for (const v of this.featureBuffer) v.forEach((x, i) => { avg[i] += x; });
        avg.forEach((_, i) => { avg[i] /= this.featureBuffer.length; });

        const [gazeX, gazeY, pitchNorm, , ear, perclos, stability,
               hasBook, hasPhone, , , , proxCode] = avg;
        const pitch = pitchNorm * 90;

        if (ear < 0.22 && perclos > 0.2)         return this._result('drowsy', 0.88);
        if (hasPhone > 0.5 && proxCode < 0.3)    return this._result('using_phone', 0.83);
        if (hasBook > 0.5 && pitch < -15)        return this._result('attentive_reading', 0.82);

        const gazeDev = Math.sqrt(gazeX ** 2 + gazeY ** 2);
        if (gazeDev < 0.3 && Math.abs(pitch) < 15 && stability > 0.6)
            return this._result('attentive_screen', 0.78);

        return this._result('distracted', 0.65);
    }

    _result(label, confidence) {
        return {
            label,
            attentionScore: this.ATTENTION_SCORES[label],
            confidence,
            source: this.featureBuffer.length >= this.WINDOW_SIZE ? 'temporal' : 'rule_based',
        };
    }
}

// ==================== UPGRADED ML ENGINE ====================

/**
 * MLEngineV2 — context-aware scoring (ARCHITECTURE.md §2).
 * Extends MLEngine with the upgraded feature weights and context helpers.
 * Falls back to legacy FEATURE_WEIGHTS when detectedObjects is not present.
 */
class MLEngineV2 extends MLEngine {
    predict(metrics) {
        // Legacy path — no context data attached yet
        if (!metrics.detectedObjects) {
            return super.predict(metrics);
        }

        // Upgraded path
        const gazeDeviation = metrics.gazeX !== null && metrics.gazeY !== null
            ? Math.sqrt(metrics.gazeX ** 2 + metrics.gazeY ** 2)
            : null;

        const featureValues = {
            ear:               metrics.ear,
            perclos:           metrics.perclos,
            blinkRate:         metrics.blinkRate,
            gazeDeviation,
            headPoseStability: metrics.headPoseStability,
            yawDeviation:      metrics.yaw,
            fatigueIndex:      metrics.fatigueIndex,
            fidgetScore:       metrics.fidgetScore,
            mouthOpenness:     metrics.mouthOpenness,
            faceDistance:      metrics.faceDistance,
            facePresence:      metrics.faceArea,
            // Context features
            contextAwarePitch:    computeContextAwarePitch(metrics),
            objectContext:        computeObjectContext(metrics),
            handObjectProximity:  encodeHandProximity(metrics.handObjectProximity),
            temporalPattern:      encodeTemporalPattern(metrics.behaviorResult),
        };

        let totalWeight = 0;
        let weightedSum = 0;

        for (const [name, value] of Object.entries(featureValues)) {
            const weight = FEATURE_WEIGHTS_V1_UPGRADED[name] || 0;
            if (!weight || value === null || value === undefined) continue;

            const result = scoreFeature(name, value);
            weightedSum += result.score * weight * result.quality;
            totalWeight += weight * result.quality;
        }

        const rawScore = totalWeight > 0 ? weightedSum / totalWeight : 50;
        const smoothedScore = this.scoreEMA.update(rawScore);

        let attentionState;
        if (smoothedScore >= THRESHOLDS.ATTENTIVE_MIN) {
            attentionState = 'Attentive';
        } else if (smoothedScore >= THRESHOLDS.DISTRACTED_MIN) {
            attentionState = 'Distracted';
        } else {
            attentionState = 'Multitasking';
        }

        const prediction = {
            attentionState,
            confidence: 0.85,
            rawScore: Math.round(smoothedScore),
        };

        this.predictionHistory.push(prediction);
        if (this.predictionHistory.length > 100) this.predictionHistory.shift();

        return prediction;
    }
}

// Export
if (typeof window !== 'undefined') {
    window.FaceMetricsAggregator = FaceMetricsAggregator;
    window.MLEngine = MLEngineV2;          // Upgraded engine replaces original
    window.MLEngineLegacy = MLEngine;      // Original preserved for debug
    window.BlinkDetector = BlinkDetector;
    window.HeadPoseTracker = HeadPoseTracker;
    window.FatigueCalculator = FatigueCalculator;
    window.HandObjectProximityDetector = HandObjectProximityDetector;
    window.TemporalBehaviorClassifier = TemporalBehaviorClassifier;
}

console.log('[ML-Core] v4 loaded — context-aware scoring enabled');
