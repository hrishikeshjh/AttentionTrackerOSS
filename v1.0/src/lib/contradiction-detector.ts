/**
 * Cross-Modal Contradiction Vector (CMCV)
 *
 * The existing Context-Gate Override (ml-engine.ts's computeContextAwarePitch)
 * is a rule that fires on ONE known confusion (phone vs. reading) and
 * discards the disagreement magnitude once applied. CMCV is a general,
 * symmetric, per-frame disagreement statistic computed over ALL modality
 * pairs already available in ExtendedFaceMetrics, retained as a first-class
 * signal in its own right — not folded back into rawScore — so downstream
 * consumers (analytics, a teacher dashboard) can see WHEN and WHY signals
 * disagree, not just an averaged-away composite score.
 *
 * This is additive and computationally free: every o_i(t) below reuses a
 * scalar the pipeline already computes; the whole thing is O(K^2)=10
 * absolute differences plus one EMA update per frame.
 *
 * Validation status: the taxonomy below (fatigue_masking,
 * prop_without_engagement, context_conflict) is a hypothesis, not yet
 * empirically validated — it requires human-annotated ambiguity/conflict
 * labels that don't exist in DAiSEE/EngageNet. Until that validation lands,
 * treat the categorical label as exploratory and the scalar
 * contradictionIndex as the more defensible signal (it should correlate
 * with classifier error rate — a much lower bar to validate).
 */

import type { ExtendedFaceMetrics, HandObjectProximity } from './types';
import { ExponentialMovingAverage } from './scoring/ema';
import { clamp } from './scoring/geometry';

export type ContradictionCategory =
  | 'coherent'
  | 'fatigue_masking'
  | 'prop_without_engagement'
  | 'context_conflict';

export type ContradictionResult = {
  contradictionIndex: number; // 0-1, EMA-smoothed
  category: ContradictionCategory;
  modalityOpinions: Record<string, number>;
};

const HIGH = 0.65;
const LOW = 0.35;

function encodeHandProximity01(proximity: HandObjectProximity | undefined): number {
  switch (proximity) {
    case 'holding_book': return 0.8;
    case 'holding_pen': return 0.75;
    case 'hands_on_desk': return 0.65;
    case 'hands_not_visible': return 0.6;
    case 'holding_phone': return 0.2;
    default: return 0.6;
  }
}

/** Pedagogically-motivated pair weights — gaze-vs-posture disagreement
 * ("eyes open, body asleep") is weighted higher than e.g. object-vs-manual,
 * since it's the classic fatigue-masking signature. These are hand-picked
 * pending empirical grounding, same caveat as the rest of this module. */
const PAIR_WEIGHTS: Record<string, number> = {
  'gaze,pose': 1.5,
  'gaze,ocular': 1.2,
  'pose,ocular': 1.0,
  'gaze,object': 1.3,
  'gaze,manual': 1.0,
  'pose,object': 0.8,
  'pose,manual': 0.8,
  'ocular,object': 0.8,
  'ocular,manual': 0.8,
  'object,manual': 0.6,
};

export class ContradictionDetector {
  private ci = new ExponentialMovingAverage(0.35); // faster than the 0.15 attention EMA

  compute(metrics: ExtendedFaceMetrics): ContradictionResult {
    const gazeDeviation =
      metrics.gazeX !== null && metrics.gazeY !== null
        ? Math.sqrt(metrics.gazeX ** 2 + metrics.gazeY ** 2)
        : 0;

    const objects = metrics.detectedObjects ?? [];
    const hasBook = objects.some((o) => o.class === 'book' || o.class === 'notebook');
    const hasPhone = objects.some((o) => o.class === 'phone');

    const opinions: Record<string, number> = {
      gaze: 1 - clamp(gazeDeviation / 0.3, 0, 1),
      pose: clamp((metrics.headPoseStability ?? 50) / 100, 0, 1),
      ocular: 1 - clamp(metrics.perclos / 20, 0, 1),
      object: hasPhone ? 0.25 : hasBook ? 0.8 : 0.65,
      manual: encodeHandProximity01(metrics.handObjectProximity),
    };

    const names = Object.keys(opinions);
    let weightedSum = 0;
    let weightTotal = 0;
    for (let i = 0; i < names.length; i++) {
      for (let j = i + 1; j < names.length; j++) {
        const key = `${names[i]},${names[j]}`;
        const weight = PAIR_WEIGHTS[key] ?? 1;
        weightedSum += weight * Math.abs(opinions[names[i]] - opinions[names[j]]);
        weightTotal += weight;
      }
    }
    const rawCI = weightTotal > 0 ? weightedSum / weightTotal : 0;
    const contradictionIndex = this.ci.update(rawCI);

    let category: ContradictionCategory = 'coherent';
    if (opinions.gaze >= HIGH && opinions.pose <= LOW && opinions.ocular <= LOW) {
      category = 'fatigue_masking';
    } else if (hasBook && metrics.handObjectProximity !== 'holding_book') {
      // Book is visible but the specific "holding this book" signal is
      // absent — a generic manual-engagement score isn't specific enough
      // here (e.g. "hands_not_visible" isn't inherently "low engagement",
      // it just isn't evidence of holding THIS object).
      category = 'prop_without_engagement';
    } else if (opinions.gaze >= HIGH && hasPhone) {
      category = 'context_conflict';
    } else if (contradictionIndex < 0.25) {
      category = 'coherent';
    }

    return { contradictionIndex, category, modalityOpinions: opinions };
  }

  reset(): void {
    this.ci.reset();
  }
}
