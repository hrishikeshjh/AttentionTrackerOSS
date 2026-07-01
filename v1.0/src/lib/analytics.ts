'use client';

/**
 * Session Analytics Module
 * 
 * Provides comprehensive session-level analytics including:
 * - Trend detection using linear regression
 * - Peak/low period identification
 * - Fatigue onset detection
 * - Anomaly detection in metrics
 */

import type { SessionDataPoint, SessionAnalytics } from './types';
import { calculateStdDev } from './face-geometry';

// ==================== CONSTANTS ====================

const TREND_WINDOW_SIZE = 30; // Number of data points for trend calculation
const PERIOD_THRESHOLD_HIGH = 75; // Score above this is "peak"
const PERIOD_THRESHOLD_LOW = 45; // Score below this is "low"
const MIN_PERIOD_LENGTH = 5; // Minimum data points to constitute a period
const FATIGUE_DETECTION_WINDOW = 60; // Seconds to look for fatigue pattern
const ANOMALY_STD_THRESHOLD = 2.5; // Standard deviations for anomaly

// ==================== LINEAR REGRESSION ====================

interface RegressionResult {
    slope: number;
    intercept: number;
    rSquared: number;
}

/**
 * Calculate linear regression for trend analysis
 */
function linearRegression(points: Array<{ x: number; y: number }>): RegressionResult {
    if (points.length < 2) {
        return { slope: 0, intercept: 0, rSquared: 0 };
    }

    const n = points.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

    for (const point of points) {
        sumX += point.x;
        sumY += point.y;
        sumXY += point.x * point.y;
        sumX2 += point.x * point.x;
    }

    const denominator = (n * sumX2 - sumX * sumX);
    if (denominator === 0) {
        return { slope: 0, intercept: sumY / n, rSquared: 0 };
    }

    const slope = (n * sumXY - sumX * sumY) / denominator;
    const intercept = (sumY - slope * sumX) / n;

    // Calculate R-squared
    const meanY = sumY / n;
    let ssTot = 0, ssRes = 0;
    for (const point of points) {
        const predictedY = slope * point.x + intercept;
        ssTot += (point.y - meanY) ** 2;
        ssRes += (point.y - predictedY) ** 2;
    }
    const rSquared = ssTot === 0 ? 0 : 1 - (ssRes / ssTot);

    return { slope, intercept, rSquared };
}

// ==================== PERIOD DETECTION ====================

interface Period {
    start: number;
    end: number;
    avgScore: number;
}

/**
 * Find contiguous periods where scores are above/below threshold
 */
function findPeriods(
    data: SessionDataPoint[],
    threshold: number,
    isHigh: boolean
): Period[] {
    const periods: Period[] = [];
    let currentPeriod: SessionDataPoint[] = [];

    for (const point of data) {
        const meetsCondition = isHigh
            ? point.attentionScore >= threshold
            : point.attentionScore <= threshold;

        if (meetsCondition) {
            currentPeriod.push(point);
        } else {
            if (currentPeriod.length >= MIN_PERIOD_LENGTH) {
                periods.push({
                    start: currentPeriod[0].timestamp,
                    end: currentPeriod[currentPeriod.length - 1].timestamp,
                    avgScore: currentPeriod.reduce((sum, p) => sum + p.attentionScore, 0) / currentPeriod.length,
                });
            }
            currentPeriod = [];
        }
    }

    // Check last period
    if (currentPeriod.length >= MIN_PERIOD_LENGTH) {
        periods.push({
            start: currentPeriod[0].timestamp,
            end: currentPeriod[currentPeriod.length - 1].timestamp,
            avgScore: currentPeriod.reduce((sum, p) => sum + p.attentionScore, 0) / currentPeriod.length,
        });
    }

    return periods;
}

// ==================== FATIGUE DETECTION ====================

/**
 * Detect fatigue onset based on declining attention pattern
 */
function detectFatigueOnset(data: SessionDataPoint[]): number | null {
    if (data.length < 20) return null;

    // Look for sustained decline
    const windowSize = Math.min(FATIGUE_DETECTION_WINDOW, Math.floor(data.length / 3));

    for (let i = windowSize; i < data.length; i++) {
        const window = data.slice(i - windowSize, i);
        const points = window.map((d, idx) => ({ x: idx, y: d.attentionScore }));
        const { slope, rSquared } = linearRegression(points);

        // Significant negative slope with good fit indicates fatigue onset
        if (slope < -0.5 && rSquared > 0.4) {
            // Verify that scores are also low
            const avgScore = window.reduce((sum, p) => sum + p.attentionScore, 0) / window.length;
            if (avgScore < 60) {
                return window[0].timestamp;
            }
        }
    }

    return null;
}

// ==================== ANOMALY DETECTION ====================

interface AnomalyResult {
    blinkAnomalies: number;
    anomalyTimestamps: number[];
}

/**
 * Detect anomalies in session data
 */
function detectAnomalies(data: SessionDataPoint[]): AnomalyResult {
    if (data.length < 10) {
        return { blinkAnomalies: 0, anomalyTimestamps: [] };
    }

    const scores = data.map(d => d.attentionScore);
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const stdDev = calculateStdDev(scores);

    const anomalyTimestamps: number[] = [];

    for (const point of data) {
        const zScore = Math.abs(point.attentionScore - mean) / (stdDev || 1);
        if (zScore > ANOMALY_STD_THRESHOLD) {
            anomalyTimestamps.push(point.timestamp);
        }
    }

    // Count significant drops as blink anomalies
    let blinkAnomalies = 0;
    for (let i = 1; i < data.length; i++) {
        const drop = data[i - 1].attentionScore - data[i].attentionScore;
        if (drop > 30) {
            blinkAnomalies++;
        }
    }

    return { blinkAnomalies, anomalyTimestamps };
}

// ==================== OPTIMAL WINDOW DETECTION ====================

/**
 * Find the window with highest average attention
 */
function findOptimalWindow(data: SessionDataPoint[], windowSize: number = 10): { start: number; end: number } | null {
    if (data.length < windowSize) return null;

    let maxAvg = 0;
    let bestStart = 0;
    let bestEnd = windowSize;

    for (let i = 0; i <= data.length - windowSize; i++) {
        const window = data.slice(i, i + windowSize);
        const avg = window.reduce((sum, p) => sum + p.attentionScore, 0) / windowSize;

        if (avg > maxAvg) {
            maxAvg = avg;
            bestStart = i;
            bestEnd = i + windowSize;
        }
    }

    if (maxAvg < 60) return null; // No good window found

    return {
        start: data[bestStart].timestamp,
        end: data[bestEnd - 1].timestamp,
    };
}

// ==================== MAIN ANALYTICS CLASS ====================

export class SessionAnalyticsEngine {
    private data: SessionDataPoint[] = [];

    /**
     * Add a data point to the session
     */
    addDataPoint(point: SessionDataPoint): void {
        this.data.push(point);
    }

    /**
     * Set all session data at once
     */
    setData(data: SessionDataPoint[]): void {
        this.data = [...data];
    }

    /**
     * Calculate comprehensive session analytics
     */
    analyze(): SessionAnalytics {
        if (this.data.length < 5) {
            return {
                attentionTrend: 'stable',
                trendSlope: 0,
                peakAttentionPeriods: [],
                lowAttentionPeriods: [],
                blinkAnomalies: 0,
                fatigueOnset: null,
                optimalEngagementWindow: null,
                attentionVariability: 0,
            };
        }

        // Calculate trend
        const recentData = this.data.slice(-TREND_WINDOW_SIZE);
        const trendPoints = recentData.map((d, i) => ({ x: i, y: d.attentionScore }));
        const { slope, rSquared } = linearRegression(trendPoints);

        let attentionTrend: 'improving' | 'declining' | 'stable';
        if (rSquared > 0.3 && Math.abs(slope) > 0.3) {
            attentionTrend = slope > 0 ? 'improving' : 'declining';
        } else {
            attentionTrend = 'stable';
        }

        // Find periods
        const peakAttentionPeriods = findPeriods(this.data, PERIOD_THRESHOLD_HIGH, true);
        const lowAttentionPeriods = findPeriods(this.data, PERIOD_THRESHOLD_LOW, false);

        // Detect fatigue
        const fatigueOnset = detectFatigueOnset(this.data);

        // Detect anomalies
        const { blinkAnomalies } = detectAnomalies(this.data);

        // Find optimal window
        const optimalEngagementWindow = findOptimalWindow(this.data);

        // Calculate variability
        const scores = this.data.map(d => d.attentionScore);
        const attentionVariability = calculateStdDev(scores);

        return {
            attentionTrend,
            trendSlope: slope,
            peakAttentionPeriods,
            lowAttentionPeriods,
            blinkAnomalies,
            fatigueOnset,
            optimalEngagementWindow,
            attentionVariability,
        };
    }

    /**
     * Get real-time trend (last 10 data points)
     */
    getRealtimeTrend(): { trend: 'up' | 'down' | 'stable'; strength: number } {
        if (this.data.length < 5) {
            return { trend: 'stable', strength: 0 };
        }

        const recentData = this.data.slice(-10);
        const points = recentData.map((d, i) => ({ x: i, y: d.attentionScore }));
        const { slope, rSquared } = linearRegression(points);

        if (rSquared < 0.2 || Math.abs(slope) < 0.5) {
            return { trend: 'stable', strength: rSquared };
        }

        return {
            trend: slope > 0 ? 'up' : 'down',
            strength: Math.min(1, Math.abs(slope) / 5),
        };
    }

    /**
     * Get summary statistics
     */
    getSummary(): {
        totalDataPoints: number;
        averageScore: number;
        minScore: number;
        maxScore: number;
        scoreRange: number;
        stateBreakdown: Record<string, number>;
    } {
        if (this.data.length === 0) {
            return {
                totalDataPoints: 0,
                averageScore: 0,
                minScore: 0,
                maxScore: 0,
                scoreRange: 0,
                stateBreakdown: {},
            };
        }

        const scores = this.data.map(d => d.attentionScore);
        const averageScore = scores.reduce((a, b) => a + b, 0) / scores.length;
        const minScore = Math.min(...scores);
        const maxScore = Math.max(...scores);

        const stateBreakdown: Record<string, number> = {};
        for (const point of this.data) {
            stateBreakdown[point.attentionState] = (stateBreakdown[point.attentionState] || 0) + 1;
        }

        return {
            totalDataPoints: this.data.length,
            averageScore: Math.round(averageScore),
            minScore,
            maxScore,
            scoreRange: maxScore - minScore,
            stateBreakdown,
        };
    }

    /**
     * Clear all data
     */
    reset(): void {
        this.data = [];
    }
}

// ==================== SINGLETON INSTANCE ====================

let analyticsInstance: SessionAnalyticsEngine | null = null;

export function getAnalyticsEngine(): SessionAnalyticsEngine {
    if (!analyticsInstance) {
        analyticsInstance = new SessionAnalyticsEngine();
    }
    return analyticsInstance;
}

export function resetAnalyticsEngine(): void {
    if (analyticsInstance) {
        analyticsInstance.reset();
    }
    analyticsInstance = new SessionAnalyticsEngine();
}

// ==================== UTILITY EXPORTS ====================

export { linearRegression, findPeriods, detectFatigueOnset, detectAnomalies };
