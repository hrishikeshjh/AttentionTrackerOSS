/**
 * DACE -> existing scoring features (Annex A.1 "Compatibility"; plan section 6).
 *
 * DACE does not add scoring features: it substitutes VALUES of the existing
 * FEATURE_WEIGHTS keys, so weights, EMA, the contradiction detector and the
 * contribution UI keep working.
 *
 * 1) Measurement substitution (all DACE modes; D7 sign rule):
 *      pitch  = −Δθ           (legacy fields keep "down = negative", so every
 *                              existing `pitch < −15` consumer becomes correct)
 *      gazeY  = −g_v · gazeScale,  gazeX = g_h · gazeScale
 *      ear    = r_t · earScale     (residual openness rescaled into the legacy EAR range)
 *      perclos= corrected PERCLOS × 100
 *    Only the fields supplied are replaced (A'' supplies pose only).
 *
 * 2) Context features (when evidence E is enabled):
 *      gazeDeviation       = (|g_h| + |g_v|) · gazeScale, only when NOT looking down (else omitted)
 *      contextAwarePitch   = reading 85, phone 20, drowsy 15, glance = previous value,
 *                            otherwise the legacy standard curve on relative pitch
 *      objectContext       = phone in use 25, reading/writing evidence 80, otherwise 65
 *                            (a phone merely present never lowers it)
 *      handObjectProximity = phone in use 20, book 80, writing 75, else 65 (hands visible) / 60
 *      temporalPattern     = per-label behaviour score (same table as the temporal classifier);
 *                            60 (neutral) while Unsure — the score is held anyway
 */
import type { BehaviorLabel, DaceMeasurementSnapshot, ExtendedFaceMetrics } from '../../types';
import { OPTIMAL_RANGES } from '../constants';
import { clamp } from '../geometry';
import type { DaceParams } from './params';
import type { DaceWindowOutput } from './types';

/** Same values as temporal-classifier.ts BEHAVIOR_ATTENTION_SCORES (asserted in tests). */
export const DACE_LABEL_SCORES: Readonly<Record<BehaviorLabel, number>> = Object.freeze({
  attentive_screen: 85,
  attentive_reading: 80,
  distracted: 45,
  using_phone: 25,
  drowsy: 20,
  absent: 0,
});

export const NEUTRAL_TEMPORAL_PATTERN = 60;

export const CONTEXT_SCORES = Object.freeze({
  pitch: { reading: 85, phone: 20, drowsy: 15 },
  object: { phoneInUse: 25, readingOrWriting: 80, neutral: 65 },
  hand: { phoneInUse: 20, book: 80, writing: 75, handsVisible: 65, handsNotVisible: 60 },
});

/** Legacy standard pitch curve (ml-engine computeContextAwarePitch, non-down branch). */
export function standardPitchScore(pitchDeg: number | null): number {
  if (pitchDeg === null) return 50;
  const a = Math.abs(pitchDeg);
  const max = OPTIMAL_RANGES.pitch.maxDeviation;
  if (a <= max) return 100 - (a / max) * 20;
  return Math.max(30, 80 - (a - max) * 2);
}

export type MeasurementSubstitution = Partial<{
  /** +down degrees (Δθ, or absolute pitch for A'') */
  pitchDownDeg: number | null;
  yawDeg: number | null;
  rollDeg: number | null;
  gazeH: number | null;
  /** +down */
  gazeV: number | null;
  residualOpenness: number | null;
  /** 0-1 */
  correctedPerclos: number | null;
}>;

export function substituteMeasurement(
  metrics: ExtendedFaceMetrics,
  sub: MeasurementSubstitution,
  params: Pick<DaceParams, 'earScale' | 'gazeScale'>,
  snapshot?: DaceMeasurementSnapshot
): ExtendedFaceMetrics {
  const m: ExtendedFaceMetrics = { ...metrics };
  if ('pitchDownDeg' in sub) m.pitch = sub.pitchDownDeg == null ? null : -sub.pitchDownDeg;
  if ('yawDeg' in sub) m.yaw = sub.yawDeg ?? null;
  if ('rollDeg' in sub) m.roll = sub.rollDeg ?? null;
  if ('gazeH' in sub) m.gazeX = sub.gazeH == null ? null : clamp(sub.gazeH * params.gazeScale, -1, 1);
  if ('gazeV' in sub) m.gazeY = sub.gazeV == null ? null : clamp(-sub.gazeV * params.gazeScale, -1, 1);
  if ('residualOpenness' in sub) m.ear = sub.residualOpenness == null ? null : sub.residualOpenness * params.earScale;
  if ('correctedPerclos' in sub) m.perclos = (sub.correctedPerclos ?? 0) * 100;
  if (snapshot) m.dace = snapshot;
  return m;
}

export type ContextMappingInput = {
  output: Pick<DaceWindowOutput, 'label' | 'unsure' | 'rule'>;
  evidence: { phoneInUse: boolean; book: boolean; writing: boolean };
  handsVisible: boolean;
  /** Δθ (+down) */
  relPitchDeg: number | null;
  /** D_t of the current frame */
  lookingDown: boolean;
  gazeH: number | null;
  gazeV: number | null;
  gazeValid: boolean;
  /** values produced on the previous tick (for glances) */
  previous: { contextAwarePitch: number } | null;
};

export type ContextFeatureValues = {
  gazeDeviation: number | null;
  contextAwarePitch: number;
  objectContext: number;
  handObjectProximity: number;
  temporalPattern: number;
};

export function mapContextFeatures(
  input: ContextMappingInput,
  params: Pick<DaceParams, 'gazeScale'>
): ContextFeatureValues {
  const { output, evidence } = input;
  const lbl = output.unsure ? null : output.label;

  let contextAwarePitch: number;
  if (output.rule === 'glance' && input.previous) contextAwarePitch = input.previous.contextAwarePitch;
  else if (lbl === 'attentive_reading') contextAwarePitch = CONTEXT_SCORES.pitch.reading;
  else if (lbl === 'using_phone') contextAwarePitch = CONTEXT_SCORES.pitch.phone;
  else if (lbl === 'drowsy') contextAwarePitch = CONTEXT_SCORES.pitch.drowsy;
  else contextAwarePitch = standardPitchScore(input.relPitchDeg);

  const objectContext = evidence.phoneInUse
    ? CONTEXT_SCORES.object.phoneInUse
    : evidence.book || evidence.writing
      ? CONTEXT_SCORES.object.readingOrWriting
      : CONTEXT_SCORES.object.neutral;

  const handObjectProximity = evidence.phoneInUse
    ? CONTEXT_SCORES.hand.phoneInUse
    : evidence.book
      ? CONTEXT_SCORES.hand.book
      : evidence.writing
        ? CONTEXT_SCORES.hand.writing
        : input.handsVisible
          ? CONTEXT_SCORES.hand.handsVisible
          : CONTEXT_SCORES.hand.handsNotVisible;

  const temporalPattern = lbl ? DACE_LABEL_SCORES[lbl] : NEUTRAL_TEMPORAL_PATTERN;

  const gazeDeviation =
    !input.lookingDown && input.gazeValid && input.gazeH !== null && input.gazeV !== null
      ? (Math.abs(input.gazeH) + Math.abs(input.gazeV)) * params.gazeScale
      : null;

  return { gazeDeviation, contextAwarePitch, objectContext, handObjectProximity, temporalPattern };
}
