'use client';

/**
 * ML Engine for Cognitive Engagement Analysis
 * 
 * Implements a weighted feature combination approach with:
 * - Temporal smoothing (exponential moving average)
 * - Confidence calculation based on feature quality
 * - Feature importance weights derived from cognitive science research
 * - Rule-based attention classification with adaptive thresholds
 */

import type { ExtendedFaceMetrics, MLPrediction, AttentionState } from './types';

// ==================== CONFIGURATION ====================

// Feature weights based on importance for attention detection
// Higher weight = more influence on attention score
const FEATURE_WEIGHTS: Record<string, number> = {
    // Eye-based features (most important for attention)
    ear: 0.15,           // Eye openness
    perclos: 0.12,       // Eye closure percentage
    blinkRate: 0.08,     // Abnormal blink rate indicates issues
    gazeDeviation: 0.10, // Looking away from screen

    // Head pose features
    headPoseStability: 0.12, // Stable head = focused
    yawDeviation: 0.08,      // Looking left/right
    pitchDeviation: 0.06,    // Looking up/down

    // Fatigue indicators
    fatigueIndex: 0.10,  // Composite fatigue score
    fidgetScore: 0.07,   // Restlessness

    // Micro-expressions
    eyebrowRaise: 0.04,   // Surprise/interest
    mouthOpenness: 0.03,  // Yawning/speaking

    // Distance and presence
    faceDistance: 0.03,   // Optimal viewing distance
    facePresence: 0.02,   // Face in frame
};

// Thresholds for attention classification
const THRESHOLDS = {
    ATTENTIVE_MIN: 70,
    DISTRACTED_MIN: 40,
    // Below DISTRACTED_MIN = Multitasking
};

// Optimal ranges for features
const OPTIMAL_RANGES = {
    ear: { min: 0.25, max: 0.4 },           // Normal eye aspect ratio
    perclos: { max: 15 },                   // Max 15% eye closure
    blinkRate: { min: 12, max: 20 },        // Normal blink rate per minute
    gaze: { maxDeviation: 0.3 },            // Max gaze deviation from center
    yaw: { maxDeviation: 15 },              // Max head yaw in degrees
    pitch: { maxDeviation: 15 },            // Max head pitch in degrees
    fatigueIndex: { max: 30 },              // Low fatigue
    fidgetScore: { max: 25 },               // Low fidgeting
    faceDistance: { min: 0.3, max: 0.7 },   // Optimal distance range
};

// ==================== TEMPORAL SMOOTHING ====================

export class ExponentialMovingAverage {
    private alpha: number;
    private value: number | null = null;

    constructor(alpha: number = 0.2) {
        this.alpha = alpha;
    }

    update(newValue: number): number {
        if (this.value === null) {
            this.value = newValue;
        } else {
            this.value = this.alpha * newValue + (1 - this.alpha) * this.value;
        }
        return this.value;
    }

    getValue(): number | null {
        return this.value;
    }

    reset(): void {
        this.value = null;
    }
}

// ==================== FEATURE SCORING ====================

/**
 * Convert a raw feature value to a normalized score (0-100)
 * Higher score = better for attention
 */
function scoreFeature(
    name: string,
    value: number | null,
    metrics: ExtendedFaceMetrics
): { score: number; contribution: number; quality: number } {
    if (value === null) {
        return { score: 50, contribution: 0, quality: 0 };
    }

    let score = 50; // Default neutral score
    let quality = 1; // Feature quality (0-1)

    switch (name) {
        case 'ear':
            // Eye Aspect Ratio: higher is better (more open eyes)
            if (value >= OPTIMAL_RANGES.ear.min && value <= OPTIMAL_RANGES.ear.max) {
                score = 85 + (value - OPTIMAL_RANGES.ear.min) / (OPTIMAL_RANGES.ear.max - OPTIMAL_RANGES.ear.min) * 15;
            } else if (value < OPTIMAL_RANGES.ear.min) {
                score = Math.max(20, 85 * (value / OPTIMAL_RANGES.ear.min));
            } else {
                score = 90; // Very open eyes is fine
            }
            break;

        case 'perclos':
            // PERCLOS: lower is better
            if (value <= OPTIMAL_RANGES.perclos.max) {
                score = 100 - (value / OPTIMAL_RANGES.perclos.max) * 20;
            } else {
                score = Math.max(10, 80 - (value - OPTIMAL_RANGES.perclos.max) * 2);
            }
            break;

        case 'blinkRate':
            // Normal blink rate is optimal
            if (value >= OPTIMAL_RANGES.blinkRate.min && value <= OPTIMAL_RANGES.blinkRate.max) {
                score = 90;
            } else if (value < OPTIMAL_RANGES.blinkRate.min) {
                // Too few blinks might indicate staring (focused) or fatigue
                score = 70 + (value / OPTIMAL_RANGES.blinkRate.min) * 20;
            } else {
                // Too many blinks might indicate stress or fatigue
                score = Math.max(40, 90 - (value - OPTIMAL_RANGES.blinkRate.max) * 3);
            }
            quality = value > 0 ? 1 : 0.5; // Need some blink data
            break;

        case 'gazeDeviation':
            // Gaze should be centered (value is distance from center)
            if (value <= OPTIMAL_RANGES.gaze.maxDeviation) {
                score = 100 - (value / OPTIMAL_RANGES.gaze.maxDeviation) * 30;
            } else {
                score = Math.max(30, 70 - (value - OPTIMAL_RANGES.gaze.maxDeviation) * 100);
            }
            break;

        case 'headPoseStability':
            // Higher stability is better
            score = value; // Already 0-100
            break;

        case 'yawDeviation':
        case 'pitchDeviation':
            // Head should face forward
            const maxDev = OPTIMAL_RANGES.yaw.maxDeviation;
            if (Math.abs(value) <= maxDev) {
                score = 100 - (Math.abs(value) / maxDev) * 20;
            } else {
                score = Math.max(30, 80 - (Math.abs(value) - maxDev) * 2);
            }
            break;

        case 'fatigueIndex':
            // Lower fatigue is better
            if (value <= OPTIMAL_RANGES.fatigueIndex.max) {
                score = 100 - (value / OPTIMAL_RANGES.fatigueIndex.max) * 20;
            } else {
                score = Math.max(20, 80 - (value - OPTIMAL_RANGES.fatigueIndex.max));
            }
            break;

        case 'fidgetScore':
            // Lower fidgeting is better
            if (value <= OPTIMAL_RANGES.fidgetScore.max) {
                score = 100 - (value / OPTIMAL_RANGES.fidgetScore.max) * 20;
            } else {
                score = Math.max(30, 80 - (value - OPTIMAL_RANGES.fidgetScore.max));
            }
            break;

        case 'eyebrowRaise':
            // Slight eyebrow raise indicates interest
            if (value >= 0.2 && value <= 0.6) {
                score = 80 + (0.5 - Math.abs(value - 0.4)) * 40;
            } else {
                score = 60;
            }
            quality = 0.7; // Less reliable feature
            break;

        case 'mouthOpenness':
            // Closed mouth is normal; open might be yawning or speaking
            if (value < 0.1) {
                score = 85;
            } else if (value < 0.3) {
                score = 70; // Might be speaking
            } else {
                score = 40; // Likely yawning
            }
            quality = 0.6;
            break;

        case 'faceDistance':
            // Optimal distance range
            if (value >= OPTIMAL_RANGES.faceDistance.min && value <= OPTIMAL_RANGES.faceDistance.max) {
                score = 90;
            } else {
                score = 60;
            }
            quality = 0.5; // Less important
            break;

        case 'facePresence':
            // Face should be present
            score = value >= 0.05 ? 100 : 20;
            quality = 1;
            break;
    }

    const weight = FEATURE_WEIGHTS[name] || 0;
    const contribution = (score - 50) * weight * quality;

    return { score, contribution, quality };
}

// ==================== ML ENGINE ====================

export class MLEngine {
    private scoreEMA = new ExponentialMovingAverage(0.15); // Smooth score transitions
    private predictionHistory: MLPrediction[] = [];
    private readonly MAX_HISTORY = 100;

    /**
     * Calculate ML prediction from extended face metrics
     */
    predict(metrics: ExtendedFaceMetrics): MLPrediction {
        const featureScores: Record<string, { score: number; contribution: number; quality: number }> = {};
        let totalWeight = 0;
        let weightedSum = 0;
        let qualitySum = 0;

        // Calculate gaze deviation from center
        const gazeDeviation = metrics.gazeX !== null && metrics.gazeY !== null
            ? Math.sqrt(metrics.gazeX ** 2 + metrics.gazeY ** 2)
            : null;

        // Prepare feature values
        const featureValues: Record<string, number | null> = {
            ear: metrics.ear,
            perclos: metrics.perclos,
            blinkRate: metrics.blinkRate,
            gazeDeviation,
            headPoseStability: metrics.headPoseStability,
            yawDeviation: metrics.yaw,
            pitchDeviation: metrics.pitch,
            fatigueIndex: metrics.fatigueIndex,
            fidgetScore: metrics.fidgetScore,
            eyebrowRaise: metrics.eyebrowRaise,
            mouthOpenness: metrics.mouthOpenness,
            faceDistance: metrics.faceDistance,
            facePresence: metrics.faceArea,
        };

        // Score each feature
        for (const [name, value] of Object.entries(featureValues)) {
            const result = scoreFeature(name, value, metrics);
            featureScores[name] = result;

            const weight = FEATURE_WEIGHTS[name] || 0;
            if (value !== null && weight > 0) {
                weightedSum += result.score * weight * result.quality;
                totalWeight += weight * result.quality;
                qualitySum += result.quality;
            }
        }

        // Calculate raw score
        let rawScore = totalWeight > 0 ? weightedSum / totalWeight : 50;

        // Apply temporal smoothing
        const smoothedScore = this.scoreEMA.update(rawScore);

        // Calculate confidence based on feature quality
        const featureCount = Object.values(featureValues).filter(v => v !== null).length;
        const maxFeatures = Object.keys(FEATURE_WEIGHTS).length;
        const completeness = featureCount / maxFeatures;
        const avgQuality = featureCount > 0 ? qualitySum / featureCount : 0;
        const confidence = completeness * avgQuality * 0.9 + 0.1; // Min 10% confidence

        // Classify attention state
        let attentionState: AttentionState;
        if (smoothedScore >= THRESHOLDS.ATTENTIVE_MIN) {
            attentionState = 'Attentive';
        } else if (smoothedScore >= THRESHOLDS.DISTRACTED_MIN) {
            attentionState = 'Distracted';
        } else {
            attentionState = 'Multitasking';
        }

        // Build feature importance map (normalized weights)
        const featureImportance: Record<string, number> = {};
        const featureContributions: Record<string, number> = {};

        for (const [name, result] of Object.entries(featureScores)) {
            featureImportance[name] = FEATURE_WEIGHTS[name] || 0;
            featureContributions[name] = result.contribution;
        }

        const prediction: MLPrediction = {
            attentionState,
            confidence,
            rawScore: Math.round(smoothedScore),
            featureImportance,
            featureContributions,
        };

        // Store in history
        this.predictionHistory.push(prediction);
        if (this.predictionHistory.length > this.MAX_HISTORY) {
            this.predictionHistory.shift();
        }

        return prediction;
    }

    /**
     * Get prediction statistics for the session
     */
    getStats(): {
        avgScore: number;
        avgConfidence: number;
        stateDistribution: Record<AttentionState, number>;
    } {
        if (this.predictionHistory.length === 0) {
            return {
                avgScore: 0,
                avgConfidence: 0,
                stateDistribution: { Attentive: 0, Distracted: 0, Multitasking: 0, Idle: 0 },
            };
        }

        const avgScore = this.predictionHistory.reduce((sum, p) => sum + p.rawScore, 0) / this.predictionHistory.length;
        const avgConfidence = this.predictionHistory.reduce((sum, p) => sum + p.confidence, 0) / this.predictionHistory.length;

        const stateDistribution: Record<AttentionState, number> = {
            Attentive: 0,
            Distracted: 0,
            Multitasking: 0,
            Idle: 0,
        };

        for (const p of this.predictionHistory) {
            stateDistribution[p.attentionState]++;
        }

        // Convert to percentages
        const total = this.predictionHistory.length;
        for (const state of Object.keys(stateDistribution) as AttentionState[]) {
            stateDistribution[state] = Math.round((stateDistribution[state] / total) * 100);
        }

        return { avgScore, avgConfidence, stateDistribution };
    }

    /**
     * Get the most impactful features for current prediction
     */
    getTopInfluencers(prediction: MLPrediction, count: number = 3): Array<{ name: string; impact: number; direction: 'positive' | 'negative' }> {
        const contributions = Object.entries(prediction.featureContributions)
            .map(([name, contribution]) => ({
                name,
                impact: Math.abs(contribution),
                direction: contribution >= 0 ? 'positive' as const : 'negative' as const,
            }))
            .sort((a, b) => b.impact - a.impact)
            .slice(0, count);

        return contributions;
    }

    /**
     * Reset the engine state
     */
    reset(): void {
        this.scoreEMA.reset();
        this.predictionHistory = [];
    }
}

// ==================== SINGLETON INSTANCE ====================

let mlEngineInstance: MLEngine | null = null;

export function getMLEngine(): MLEngine {
    if (!mlEngineInstance) {
        mlEngineInstance = new MLEngine();
    }
    return mlEngineInstance;
}

export function resetMLEngine(): void {
    if (mlEngineInstance) {
        mlEngineInstance.reset();
    }
    mlEngineInstance = new MLEngine();
}
