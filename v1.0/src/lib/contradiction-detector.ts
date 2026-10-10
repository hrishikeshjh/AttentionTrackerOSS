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

import type { BehaviorLabel, ExtendedFaceMetrics, HandObjectProximity } from './types';
import { ExponentialMovingAverage } from './scoring/ema';
import { clamp } from './scoring/geometry';

/**
 * CMCV taxonomy. 'unclassified' = the modalities disagree (smoothed index
 * >= COHERENT_MAX) but no named pattern matches. Before 2026-10 such frames
 * were silently reported as 'coherent' (the old final branch was a no-op).
 */
export type ContradictionCategory =
  | 'coherent'
  | 'fatigue_masking'
  | 'prop_without_engagement'
  | 'context_conflict'
  | 'unclassified';

export type ContradictionResult = {
  contradictionIndex: number; // 0-1, EMA-smoothed
  category: ContradictionCategory;
  modalityOpinions: Record<string, number>;
};

const HIGH = 0.65;
const LOW = 0.35;
/** smoothed index below which an unmatched frame is 'coherent' */
export const COHERENT_MAX = 0.25;

/**
 * Hand states that count as manual engagement with a visible book: holding
 * it, writing next to it, or hands resting on the desk (a student reading a
 * book lying on the desk need not touch it).
 */
const ENGAGED_PROXIMITY: ReadonlySet<HandObjectProximity> = new Set([
  'holding_book',
  'holding_pen',
  'hands_on_desk',
]);

/**
 * Pure category rule, shared semantics with
 * v2.0/inference/contradiction_detector.py::classify_category (checked by
 * the shared fixture shared/fixtures/cmcv_cross_language.json).
 *
 *   1. fatigue_masking          gaze >= HIGH, pose <= LOW, ocular <= LOW
 *   2. prop_without_engagement  book visible, hands NOT holding it / writing /
 *                               on the desk, and the behaviour label is not
 *                               attentive_reading
 *   3. context_conflict         gaze >= HIGH and a phone is visible
 *   4. coherent                 smoothed index < COHERENT_MAX
 *   5. unclassified             otherwise (disagreement, no named pattern)
 */
/**
 * Taxonomy version.
 *   'v2' (default): classifyContradiction above.
 *   'v1': the original rules, kept bit-for-bit because MLEngine's default
 *         construction is frozen by the legacy golden fixture
 *         (src/lib/__tests__/legacy-equivalence.test.ts). In 'v1',
 *         prop_without_engagement fires for every book without holding_book
 *         and an unmatched high index is reported as 'coherent'. The app
 *         engine (getMLEngine(), APP_ENGINE_OPTIONS in ml-engine.ts) uses 'v2'.
 */
export type ContradictionTaxonomy = 'v1' | 'v2';

function classifyContradictionV1(
  opinions: Record<string, number>,
  hasBook: boolean,
  hasPhone: boolean,
  handObjectProximity: HandObjectProximity | undefined
): ContradictionCategory {
  if (opinions.gaze >= HIGH && opinions.pose <= LOW && opinions.ocular <= LOW) return 'fatigue_masking';
  if (hasBook && handObjectProximity !== 'holding_book') return 'prop_without_engagement';
  if (opinions.gaze >= HIGH && hasPhone) return 'context_conflict';
  return 'coherent';
}

export function classifyContradiction(args: {
  opinions: Record<string, number>;
  contradictionIndex: number;
  hasBook: boolean;
  hasPhone: boolean;
  handObjectProximity: HandObjectProximity | undefined;
  behaviorLabel: BehaviorLabel | undefined;
}): ContradictionCategory {
  const { opinions, contradictionIndex, hasBook, hasPhone, handObjectProximity, behaviorLabel } = args;
  if (opinions.gaze >= HIGH && opinions.pose <= LOW && opinions.ocular <= LOW) {
    return 'fatigue_masking';
  }
  const manualEngagement = handObjectProximity !== undefined && ENGAGED_PROXIMITY.has(handObjectProximity);
  const readingEngagement = behaviorLabel === 'attentive_reading';
  if (hasBook && !manualEngagement && !readingEngagement) {
    // Book visible but neither the hands nor the behaviour classifier show
    // engagement with it ("hands_not_visible" is not evidence of reading).
    return 'prop_without_engagement';
  }
  if (opinions.gaze >= HIGH && hasPhone) return 'context_conflict';
  if (contradictionIndex < COHERENT_MAX) return 'coherent';
  return 'unclassified';
}

/**
 * CMCV "manual" opinion (0-1, higher = more engaged).
 *
 * Proximity code sets differ between modules ON PURPOSE; do not "unify" them:
 *   - this opinion (TS and v2.0 Python CMCV, identical):
 *       book .8, pen .75, desk .65, not visible .6, phone .2
 *   - v1.0 rule layer / encode13 (PROXIMITY_CODES_V1, temporal-classifier.ts):
 *       book 1.0, pen .8, desk .5, phone .2, not visible 0
 *   - v2.0 model input (PROXIMITY_CODES_V2 / feature_vector.py _PROXIMITY_CODE):
 *       not visible 0, desk .25, pen .5, book .75, phone 1.0
 *   - MLEngine handObjectProximity feature (0-100): 80/75/65/60/20, i.e. this
 *     table x 100.
 * The CMCV value is an opinion on a 0-1 "attentive" scale, the other two are
 * categorical model-input codes, so the numbers are not interchangeable.
 */
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

  constructor(private readonly taxonomy: ContradictionTaxonomy = 'v2') {}

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

    const category =
      this.taxonomy === 'v1'
        ? classifyContradictionV1(opinions, hasBook, hasPhone, metrics.handObjectProximity)
        : classifyContradiction({
            opinions,
            contradictionIndex,
            hasBook,
            hasPhone,
            handObjectProximity: metrics.handObjectProximity,
            behaviorLabel: metrics.behaviorResult?.label,
          });

    return { contradictionIndex, category, modalityOpinions: opinions };
  }

  reset(): void {
    this.ci.reset();
  }
}
