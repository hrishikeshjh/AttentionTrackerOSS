/**
 * CognEn Extension - ML Core Library
 * Portable, vanilla JavaScript version of the ML engine for the Chrome extension
 * No external dependencies except face-api.js for landmark detection
 */

// ==================== CONSTANTS ====================

const EAR_BLINK_THRESHOLD = 0.21;
const PERCLOS_WINDOW_MS = 60000;
const NORMAL_BLINK_RATE_MIN = 15;
const NORMAL_BLINK_RATE_MAX = 20;
const FATIGUE_EAR_THRESHOLD = 0.25;

const FEATURE_WEIGHTS = {
    ear: 0.15,
    perclos: 0.12,
    blinkRate: 0.08,
    gazeDeviation: 0.10,
    headPoseStability: 0.12,
    yawDeviation: 0.08,
    pitchDeviation: 0.06,
    fatigueIndex: 0.10,
    fidgetScore: 0.07,
    eyebrowRaise: 0.04,
    mouthOpenness: 0.03,
    faceDistance: 0.03,
    facePresence: 0.02,
};

const THRESHOLDS = {
    ATTENTIVE_MIN: 70,
    DISTRACTED_MIN: 40,
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

    if (!nose || !leftEye || !rightEye || !chin) return { yaw: null, pitch: null, roll: null };

    const eyeMidpoint = {
        x: (leftEye.x + rightEye.x) / 2,
        y: (leftEye.y + rightEye.y) / 2,
    };

    const eyeDist = distance(leftEye, rightEye);
    const noseChinDist = distance(nose, chin);

    if (eyeDist === 0 || noseChinDist === 0) return { yaw: null, pitch: null, roll: null };

    const yaw = ((nose.x - eyeMidpoint.x) / eyeDist) * 50;
    const pitch = ((nose.y - eyeMidpoint.y) / noseChinDist) * 50;
    const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * (180 / Math.PI);

    return { yaw, pitch, roll };
}

function calculateGazeDirection(landmarks) {
    if (!landmarks || landmarks.length < 68) return { gazeX: null, gazeY: null };

    const leftIrisCenter = {
        x: (landmarks[37].x + landmarks[38].x + landmarks[40].x + landmarks[41].x) / 4,
        y: (landmarks[37].y + landmarks[38].y + landmarks[40].y + landmarks[41].y) / 4,
    };
    const rightIrisCenter = {
        x: (landmarks[43].x + landmarks[44].x + landmarks[46].x + landmarks[47].x) / 4,
        y: (landmarks[43].y + landmarks[44].y + landmarks[46].y + landmarks[47].y) / 4,
    };

    const leftEyeWidth = distance(landmarks[36], landmarks[39]);
    const rightEyeWidth = distance(landmarks[42], landmarks[45]);

    if (leftEyeWidth === 0 || rightEyeWidth === 0) return { gazeX: null, gazeY: null };

    const leftGazeX = ((leftIrisCenter.x - landmarks[36].x) / leftEyeWidth - 0.5) * 2;
    const rightGazeX = ((rightIrisCenter.x - landmarks[42].x) / rightEyeWidth - 0.5) * 2;

    return {
        gazeX: clamp((leftGazeX + rightGazeX) / 2, -1, 1),
        gazeY: 0, // Simplified for extension
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
    }

    detectBlink(ear, timestamp) {
        if (ear === null) {
            return { blinkDetected: false, blinkRate: this.getBlinkRate(), perclos: this.getPERCLOS(timestamp) };
        }

        this.earHistory.push({ time: timestamp, ear });

        const cutoff = timestamp - 120000;
        this.earHistory = this.earHistory.filter(e => e.time > cutoff);
        this.blinkEvents = this.blinkEvents.filter(t => t > cutoff);

        const isEyeClosed = ear < EAR_BLINK_THRESHOLD;
        let blinkDetected = false;

        if (isEyeClosed && !this.wasEyeClosed) {
            this.eyeClosureStart = timestamp;
        }

        if (this.wasEyeClosed && !isEyeClosed && this.eyeClosureStart) {
            const blinkDuration = timestamp - this.eyeClosureStart;
            if (blinkDuration >= 50 && blinkDuration <= 500) {
                blinkDetected = true;
                this.blinkEvents.push(timestamp);
            }
            this.eyeClosureStart = null;
        }

        this.wasEyeClosed = isEyeClosed;

        return {
            blinkDetected,
            blinkRate: this.getBlinkRate(),
            perclos: this.getPERCLOS(timestamp),
        };
    }

    getBlinkRate() {
        if (this.blinkEvents.length < 2) return 0;

        const now = Date.now();
        const oneMinuteAgo = now - 60000;
        const recentBlinks = this.blinkEvents.filter(t => t > oneMinuteAgo);

        const timeSpan = Math.min(60000, now - this.blinkEvents[0]);
        if (timeSpan < 5000) return 0;

        return (recentBlinks.length / timeSpan) * 60000;
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

        return clamp(100 - (combinedVar / 10), 0, 100);
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
        return clamp(avgDiff * 5, 0, 100);
    }

    getPoseVariance() {
        if (this.poseHistory.length < 2) return 0;

        const yaws = this.poseHistory.map(p => p.yaw);
        const pitches = this.poseHistory.map(p => p.pitch);
        const rolls = this.poseHistory.map(p => p.roll);

        return (calculateVariance(yaws) + calculateVariance(pitches) + calculateVariance(rolls)) / 3;
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
        if (this.perclosHistory.length < 5) return 0;

        const avgPerclos = this.perclosHistory.reduce((a, b) => a + b.value, 0) / this.perclosHistory.length;
        const perclosScore = clamp(avgPerclos * 2, 0, 40);

        const avgBlinkRate = this.blinkRateHistory.reduce((a, b) => a + b.value, 0) / this.blinkRateHistory.length;
        let blinkScore = 0;
        if (avgBlinkRate < NORMAL_BLINK_RATE_MIN) {
            blinkScore = clamp((NORMAL_BLINK_RATE_MIN - avgBlinkRate) * 2, 0, 30);
        } else if (avgBlinkRate > NORMAL_BLINK_RATE_MAX) {
            blinkScore = clamp((avgBlinkRate - NORMAL_BLINK_RATE_MAX), 0, 30);
        }

        let earTrendScore = 0;
        if (this.earMeanHistory.length > 10) {
            const recentEars = this.earMeanHistory.slice(-10).map(e => e.value);
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
        const poseVariance = this.headPoseTracker.getPoseVariance();

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
            eyebrowRaise: null,
            mouthOpenness,
            earVariance: earStats.variance,
            marVariance: calculateVariance(this.marHistory),
            poseVariance,
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
    constructor(alpha = 0.2) {
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

    getValue() {
        return this.value;
    }

    reset() {
        this.value = null;
    }
}

function scoreFeature(name, value, metrics) {
    if (value === null || isNaN(value)) {
        return { score: 50, contribution: 0, quality: 0 };
    }

    let score = 50;
    let quality = 1;

    switch (name) {
        case 'ear':
            if (value >= 0.25 && value <= 0.4) {
                score = 85 + (value - 0.25) / 0.15 * 15;
            } else if (value < 0.25) {
                score = Math.max(20, 85 * (value / 0.25));
            } else {
                score = 90;
            }
            break;

        case 'perclos':
            if (value <= 15) {
                score = 100 - (value / 15) * 20;
            } else {
                score = Math.max(10, 80 - (value - 15) * 2);
            }
            break;

        case 'blinkRate':
            if (value >= 12 && value <= 20) {
                score = 90;
            } else if (value < 12) {
                score = 70 + (value / 12) * 20;
            } else {
                score = Math.max(40, 90 - (value - 20) * 3);
            }
            quality = value > 0 ? 1 : 0.5;
            break;

        case 'gazeDeviation':
            if (value <= 0.3) {
                score = 100 - (value / 0.3) * 30;
            } else {
                score = Math.max(30, 70 - (value - 0.3) * 100);
            }
            break;

        case 'headPoseStability':
            score = value;
            break;

        case 'yawDeviation':
        case 'pitchDeviation':
            if (Math.abs(value) <= 15) {
                score = 100 - (Math.abs(value) / 15) * 20;
            } else {
                score = Math.max(30, 80 - (Math.abs(value) - 15) * 2);
            }
            break;

        case 'fatigueIndex':
            if (value <= 30) {
                score = 100 - (value / 30) * 20;
            } else {
                score = Math.max(20, 80 - (value - 30));
            }
            break;

        case 'fidgetScore':
            if (value <= 25) {
                score = 100 - (value / 25) * 20;
            } else {
                score = Math.max(30, 80 - (value - 25));
            }
            break;

        case 'mouthOpenness':
            if (value < 0.1) {
                score = 85;
            } else if (value < 0.3) {
                score = 70;
            } else {
                score = 40;
            }
            quality = 0.6;
            break;

        case 'faceDistance':
            if (value >= 0.3 && value <= 0.7) {
                score = 90;
            } else {
                score = 60;
            }
            quality = 0.5;
            break;

        case 'facePresence':
            score = value >= 0.05 ? 100 : 20;
            break;
    }

    const weight = FEATURE_WEIGHTS[name] || 0;
    const contribution = (score - 50) * weight * quality;

    return { score, contribution, quality };
}

class MLEngine {
    constructor() {
        this.scoreEMA = new ExponentialMovingAverage(0.15);
        this.predictionHistory = [];
    }

    predict(metrics) {
        const gazeDeviation = metrics.gazeX !== null && metrics.gazeY !== null
            ? Math.sqrt(metrics.gazeX ** 2 + metrics.gazeY ** 2)
            : null;

        const featureValues = {
            ear: metrics.ear,
            perclos: metrics.perclos,
            blinkRate: metrics.blinkRate,
            gazeDeviation,
            headPoseStability: metrics.headPoseStability,
            yawDeviation: metrics.yaw,
            pitchDeviation: metrics.pitch,
            fatigueIndex: metrics.fatigueIndex,
            fidgetScore: metrics.fidgetScore,
            mouthOpenness: metrics.mouthOpenness,
            faceDistance: metrics.faceDistance,
            facePresence: metrics.faceArea,
        };

        let totalWeight = 0;
        let weightedSum = 0;
        let qualitySum = 0;
        const featureContributions = {};

        for (const [name, value] of Object.entries(featureValues)) {
            const result = scoreFeature(name, value, metrics);
            featureContributions[name] = result.contribution;

            const weight = FEATURE_WEIGHTS[name] || 0;
            if (value !== null && weight > 0) {
                weightedSum += result.score * weight * result.quality;
                totalWeight += weight * result.quality;
                qualitySum += result.quality;
            }
        }

        let rawScore = totalWeight > 0 ? weightedSum / totalWeight : 50;
        const smoothedScore = this.scoreEMA.update(rawScore);

        const featureCount = Object.values(featureValues).filter(v => v !== null).length;
        const maxFeatures = Object.keys(FEATURE_WEIGHTS).length;
        const completeness = featureCount / maxFeatures;
        const avgQuality = featureCount > 0 ? qualitySum / featureCount : 0;
        const confidence = completeness * avgQuality * 0.9 + 0.1;

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
            confidence,
            rawScore: Math.round(smoothedScore),
            featureImportance: FEATURE_WEIGHTS,
            featureContributions,
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

// Export for use in popup.js
if (typeof window !== 'undefined') {
    window.FaceMetricsAggregator = FaceMetricsAggregator;
    window.MLEngine = MLEngine;
    window.BlinkDetector = BlinkDetector;
    window.HeadPoseTracker = HeadPoseTracker;
    window.FatigueCalculator = FatigueCalculator;
}
