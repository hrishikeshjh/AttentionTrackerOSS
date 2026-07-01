'use client';

/**
 * MLEngine — Cognitive Engagement Analysis
 *
 * v1.0 Upgraded: context-aware scoring (ARCHITECTURE.md §2).
 * Backward-compatible: legacy path (no object/hand data) still works.
 *
 * Two scoring modes, selected automatically based on whether
 * context features are present on the ExtendedFaceMetrics object:
 *
 *   LEGACY  — pitchDeviation scored uniformly (old behaviour)
 *   UPGRADED — contextAwarePitch + objectContext + handObjectProximity +
 *              temporalPattern replace pitchDeviation (ARCHITECTURE §2.2–2.4)
 */

import type {
  ExtendedFaceMetrics,
  MLPrediction,
  AttentionState,
  HandObjectProximity,
  BehaviorResult,
} from './types';
import {
  FEATURE_WEIGHTS_LEGACY,
  FEATURE_WEIGHTS_V1_UPGRADED,
  THRESHOLDS,
  OPTIMAL_RANGES,
} from './scoring/constants';
import { ExponentialMovingAverage } from './scoring/ema';
import { ContradictionDetector } from './contradiction-detector';

// Feature weights and thresholds now live in ./scoring/constants.ts — the
// single source of truth shared with the Chrome extension bundle and kept
// in sync with v2.0's Python port (see scoring/EQUIVALENCE.md). Re-exported
// here for any existing callers importing them from this module.
export { FEATURE_WEIGHTS_LEGACY, FEATURE_WEIGHTS_V1_UPGRADED, THRESHOLDS, OPTIMAL_RANGES };

// ==================== CONTEXT-AWARE HELPERS ====================

/**
 * Context-aware pitch score (ARCHITECTURE.md §2.2, §5.1).
 *
 * Uniform pitch penalty → context-gated score:
 *   pitch < -15° + book near hand → 85  (reading = attentive)
 *   pitch < -15° + phone near hand → 20 (phone = distracted)
 *   pitch < -15° + drowsy signals → 15
 *   pitch < -15° + unknown → 40
 *   otherwise → standard pitch scoring
 */
function computeContextAwarePitch(metrics: ExtendedFaceMetrics): number {
  const pitch = metrics.pitch;
  if (pitch === null) return 50;

  if (pitch < -15) {
    const handProx = metrics.handObjectProximity;
    const objects = metrics.detectedObjects ?? [];
    const hasBook = objects.some(
      (o) => o.class === 'book' || o.class === 'notebook'
    );
    const hasPhone = objects.some((o) => o.class === 'phone');

    // Priority: phone > book > drowsy > unknown (ARCHITECTURE §2.2 ordering)
    if (handProx === 'holding_phone' || hasPhone) return 20;
    if (handProx === 'holding_book' || hasBook) return 85;
    if ((metrics.ear ?? 0.3) < 0.22 && metrics.perclos > 20) return 15;
    return 40;
  }

  // Standard forward/upward pitch scoring
  const absPitch = Math.abs(pitch);
  if (absPitch <= OPTIMAL_RANGES.pitch.maxDeviation) {
    return 100 - (absPitch / OPTIMAL_RANGES.pitch.maxDeviation) * 20;
  }
  return Math.max(30, 80 - (absPitch - OPTIMAL_RANGES.pitch.maxDeviation) * 2);
}

/**
 * Object context score — what objects are visible in the scene.
 */
function computeObjectContext(metrics: ExtendedFaceMetrics): number {
  const objects = metrics.detectedObjects ?? [];
  if (objects.length === 0) return 65; // neutral — no objects

  const hasPhone = objects.some((o) => o.class === 'phone');
  if (hasPhone) return 25;

  const hasBook = objects.some(
    (o) => o.class === 'book' || o.class === 'notebook'
  );
  if (hasBook) {
    return (metrics.pitch ?? 0) < -10 ? 80 : 70;
  }

  const hasLaptop = objects.some((o) => o.class === 'laptop');
  if (hasLaptop) {
    return Math.abs(metrics.pitch ?? 0) < 15 ? 80 : 65;
  }

  return 65; // keyboard / mouse / other — neutral
}

/** Encode HandObjectProximity → numeric score (0–100). */
function encodeHandProximity(
  proximity: HandObjectProximity | undefined
): number {
  switch (proximity) {
    case 'holding_book': return 80;
    case 'holding_pen': return 75;  // writing = attentive
    case 'hands_on_desk': return 65;
    case 'hands_not_visible': return 60;
    case 'holding_phone': return 20;
    default: return 60;
  }
}

/** Encode BehaviorResult.attentionScore → numeric score (0–100). */
function encodeTemporalPattern(
  behaviorResult: BehaviorResult | undefined
): number {
  return behaviorResult?.attentionScore ?? 60; // 60 = neutral when unavailable
}

// ExponentialMovingAverage now lives in ./scoring/ema.ts — re-exported for
// existing callers importing it from this module.
export { ExponentialMovingAverage };

// ==================== FEATURE SCORING ====================

/**
 * Score a single feature → normalized 0–100.
 * Higher = more attentive.
 */
function scoreFeature(
  name: string,
  value: number | null,
  metrics: ExtendedFaceMetrics
): { score: number; contribution: number; quality: number } {
  if (value === null) return { score: 50, contribution: 0, quality: 0 };

  let score = 50;
  let quality = 1;

  switch (name) {
    case 'ear':
      if (value >= OPTIMAL_RANGES.ear.min && value <= OPTIMAL_RANGES.ear.max) {
        score = 85 + ((value - OPTIMAL_RANGES.ear.min) /
          (OPTIMAL_RANGES.ear.max - OPTIMAL_RANGES.ear.min)) * 15;
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
      score = value >= 0.2 && value <= 0.6
        ? 80 + (0.5 - Math.abs(value - 0.4)) * 40
        : 60;
      quality = 0.7;
      break;

    case 'mouthOpenness':
      score = value < 0.1 ? 85 : value < 0.3 ? 70 : 40;
      quality = 0.6;
      break;

    case 'faceDistance':
      score = value >= OPTIMAL_RANGES.faceDistance.min &&
        value <= OPTIMAL_RANGES.faceDistance.max
        ? 90
        : 60;
      quality = 0.5;
      break;

    case 'facePresence':
      score = value >= 0.05 ? 100 : 20;
      break;

    // ── NEW context-aware features ──────────────────────────────────
    // These features arrive as pre-computed 0–100 scores from the helpers
    // above.  scoreFeature() just passes them through with quality=1.
    case 'contextAwarePitch':
    case 'objectContext':
    case 'handObjectProximity':
    case 'temporalPattern':
      score = Math.max(0, Math.min(100, value));
      break;
  }

  const weight = FEATURE_WEIGHTS_V1_UPGRADED[name] ??
    FEATURE_WEIGHTS_LEGACY[name] ?? 0;
  const contribution = (score - 50) * weight * quality;

  return { score, contribution, quality };
}

// ==================== ML ENGINE ====================

export class MLEngine {
  private scoreEMA = new ExponentialMovingAverage(0.15);
  private contradictionDetector = new ContradictionDetector();
  private predictionHistory: MLPrediction[] = [];
  private readonly MAX_HISTORY = 100;

  /**
   * Predict attention state from extended face metrics.
   *
   * Automatically uses the UPGRADED scoring path when
   * `metrics.detectedObjects` is defined (set by webcam-panel after
   * ObjectDetector / HandTracker are loaded). Falls back to LEGACY path
   * when those modules are not yet initialised.
   */
  predict(metrics: ExtendedFaceMetrics): MLPrediction {
    const useUpgraded = metrics.detectedObjects !== undefined;
    const weights = useUpgraded
      ? FEATURE_WEIGHTS_V1_UPGRADED
      : FEATURE_WEIGHTS_LEGACY;

    // ── Build feature value map ──────────────────────────────────────
    const gazeDeviation =
      metrics.gazeX !== null && metrics.gazeY !== null
        ? Math.sqrt(metrics.gazeX ** 2 + metrics.gazeY ** 2)
        : null;

    const featureValues: Record<string, number | null> = {
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
      // Replace pitchDeviation with context-aware features
      featureValues['contextAwarePitch'] = computeContextAwarePitch(metrics);
      featureValues['objectContext'] = computeObjectContext(metrics);
      featureValues['handObjectProximity'] = encodeHandProximity(
        metrics.handObjectProximity
      );
      featureValues['temporalPattern'] = encodeTemporalPattern(
        metrics.behaviorResult
      );
    } else {
      // Legacy: uniform pitch penalty
      featureValues['pitchDeviation'] = metrics.pitch;
    }

    // ── Score each feature ───────────────────────────────────────────
    const featureScores: Record<
      string,
      { score: number; contribution: number; quality: number }
    > = {};
    let totalWeight = 0;
    let weightedSum = 0;
    let qualitySum = 0;

    for (const [name, value] of Object.entries(featureValues)) {
      const weight = weights[name] ?? 0;
      if (weight === 0 || value === null) continue;

      const result = scoreFeature(name, value, metrics);
      featureScores[name] = result;

      weightedSum += result.score * weight * result.quality;
      totalWeight += weight * result.quality;
      qualitySum += result.quality;
    }

    // ── Aggregate + smooth ───────────────────────────────────────────
    const rawScore = totalWeight > 0 ? weightedSum / totalWeight : 50;
    const smoothedScore = this.scoreEMA.update(rawScore);

    // Confidence: fraction of features available × average quality
    const featureCount = Object.values(featureValues).filter(
      (v) => v !== null
    ).length;
    const maxFeatures = Object.keys(weights).length;
    const completeness = featureCount / maxFeatures;
    const avgQuality =
      featureCount > 0 ? qualitySum / featureCount : 0;
    const confidence = completeness * avgQuality * 0.9 + 0.1;

    // ── Classify ─────────────────────────────────────────────────────
    let attentionState: AttentionState;
    if (smoothedScore >= THRESHOLDS.ATTENTIVE_MIN) {
      attentionState = 'Attentive';
    } else if (smoothedScore >= THRESHOLDS.DISTRACTED_MIN) {
      attentionState = 'Distracted';
    } else {
      attentionState = 'Multitasking';
    }

    const featureImportance: Record<string, number> = {};
    const featureContributions: Record<string, number> = {};
    for (const [name, result] of Object.entries(featureScores)) {
      featureImportance[name] = weights[name] ?? 0;
      featureContributions[name] = result.contribution;
    }

    // Cross-Modal Contradiction Vector — computed alongside rawScore but
    // never folded into it (see contradiction-detector.ts).
    const contradiction = this.contradictionDetector.compute(metrics);

    const prediction: MLPrediction = {
      attentionState,
      confidence,
      rawScore: Math.round(smoothedScore),
      featureImportance,
      featureContributions,
      contradictionIndex: contradiction.contradictionIndex,
      contradictionCategory: contradiction.category,
    };

    this.predictionHistory.push(prediction);
    if (this.predictionHistory.length > this.MAX_HISTORY) {
      this.predictionHistory.shift();
    }

    return prediction;
  }

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

    const avgScore =
      this.predictionHistory.reduce((s, p) => s + p.rawScore, 0) /
      this.predictionHistory.length;
    const avgConfidence =
      this.predictionHistory.reduce((s, p) => s + p.confidence, 0) /
      this.predictionHistory.length;

    const dist: Record<AttentionState, number> = {
      Attentive: 0, Distracted: 0, Multitasking: 0, Idle: 0,
    };
    for (const p of this.predictionHistory) dist[p.attentionState]++;
    const total = this.predictionHistory.length;
    for (const k of Object.keys(dist) as AttentionState[]) {
      dist[k] = Math.round((dist[k] / total) * 100);
    }

    return { avgScore, avgConfidence, stateDistribution: dist };
  }

  getTopInfluencers(
    prediction: MLPrediction,
    count = 3
  ): Array<{ name: string; impact: number; direction: 'positive' | 'negative' }> {
    return Object.entries(prediction.featureContributions)
      .map(([name, contribution]) => ({
        name,
        impact: Math.abs(contribution),
        direction: (contribution >= 0 ? 'positive' : 'negative') as
          'positive' | 'negative',
      }))
      .sort((a, b) => b.impact - a.impact)
      .slice(0, count);
  }

  reset(): void {
    this.scoreEMA.reset();
    this.contradictionDetector.reset();
    this.predictionHistory = [];
  }
}

// ==================== DEBUG UTILITY ====================

/**
 * debugPrediction — log a full breakdown of a single prediction.
 * Call from the browser console or from webcam-panel for development.
 *
 * @example
 *   import { debugPrediction } from '@/lib/ml-engine';
 *   debugPrediction(metrics, prediction);
 */
export function debugPrediction(
  metrics: ExtendedFaceMetrics,
  prediction: MLPrediction
): void {
  const useUpgraded = metrics.detectedObjects !== undefined;
  console.group(
    `[MLEngine] ${prediction.attentionState} | score=${prediction.rawScore} | ` +
    `mode=${useUpgraded ? 'UPGRADED' : 'LEGACY'} | conf=${(prediction.confidence * 100).toFixed(0)}%`
  );

  if (useUpgraded) {
    console.log('Context features:', {
      contextAwarePitch: computeContextAwarePitch(metrics).toFixed(0),
      objectContext: computeObjectContext(metrics).toFixed(0),
      handObjectProximity: encodeHandProximity(metrics.handObjectProximity).toFixed(0),
      temporalPattern: encodeTemporalPattern(metrics.behaviorResult).toFixed(0),
      detectedObjects: metrics.detectedObjects?.map((o) => o.class),
      handProximity: metrics.handObjectProximity,
      behaviorLabel: metrics.behaviorResult?.label,
    });
  }

  console.log('Feature contributions (sorted):',
    Object.entries(prediction.featureContributions)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .map(([k, v]) => `${k}: ${v > 0 ? '+' : ''}${v.toFixed(2)}`)
  );

  console.groupEnd();
}

// ==================== SINGLETON ====================

let mlEngineInstance: MLEngine | null = null;

export function getMLEngine(): MLEngine {
  if (!mlEngineInstance) mlEngineInstance = new MLEngine();
  return mlEngineInstance;
}

export function resetMLEngine(): void {
  mlEngineInstance = new MLEngine();
}
