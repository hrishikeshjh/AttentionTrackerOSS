'use client';

/**
 * TemporalBehaviorClassifier
 *
 * Classifies student behavior using a 15-frame (≈3 s at 5 FPS) sliding window,
 * as specified in ARCHITECTURE.md §2.3.
 *
 * Two-tier design:
 *   1. Rule-based fallback  — always available, used when < 15 frames buffered
 *      or when no ONNX model is loaded.  Implements the context-aware logic
 *      from ARCHITECTURE.md §5.1.
 *   2. ONNX 1D-CNN          — plug in a trained `behavior-classifier.onnx`
 *      (future, after dataset collection per ARCHITECTURE.md §6) via loadModel().
 *
 * Feature vector per frame (FEATURE_DIM = 13):
 *   [gaze_x, gaze_y, pitch_norm, yaw_norm, ear, perclos,
 *    head_pose_stability, obj_book, obj_phone, obj_laptop,
 *    obj_none, fidget_score, hand_proximity_code]
 */

import type {
  ExtendedFaceMetrics,
  BehaviorLabel,
  BehaviorResult,
  HandObjectProximity,
  DetectedObject,
} from './types';
import { clamp } from './face-geometry';

// ==================== CONSTANTS ====================

/** Number of frames in the sliding window (3 s at 5 FPS). */
const WINDOW_SIZE = 15;

/** Feature vector dimensionality. Must stay in sync with extractFeatures(). */
const FEATURE_DIM = 13;

/**
 * Attention scores mapped from behavior labels.
 * Used to convert a behavior classification → attentionScore field on MLPrediction.
 */
export const BEHAVIOR_ATTENTION_SCORES: Record<BehaviorLabel, number> = {
  attentive_screen: 85,
  attentive_reading: 80,
  distracted: 45,
  using_phone: 25,
  drowsy: 20,
  absent: 0,
};

// ==================== FRAME INPUT TYPE ====================

export type TemporalFeatureFrame = {
  metrics: ExtendedFaceMetrics;
  detectedObjects: DetectedObject[];
  handProximity: HandObjectProximity;
};

// ==================== TEMPORAL CLASSIFIER ====================

export class TemporalBehaviorClassifier {
  /** Ring buffer of encoded feature vectors. */
  private featureBuffer: number[][] = [];

  /** Optional ONNX session (loaded via loadModel). */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private modelSession: any = null;

  private lastResult: BehaviorResult = {
    label: 'distracted',
    attentionScore: 50,
    confidence: 0,
    source: 'rule_based',
  };

  // ==================== PUBLIC API ====================

  /**
   * Optionally load a trained 1D-CNN ONNX model.
   * If loading fails or is never called, rule-based classification is used.
   *
   * @param modelUrl  e.g. '/models/behavior-classifier.onnx'
   */
  async loadModel(modelUrl: string): Promise<void> {
    try {
      const pkg = 'onnxruntime-web';
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ort: any = await import(pkg as string);
      this.modelSession = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['webgpu', 'wasm'],
      });
    } catch {
      console.warn(
        '[TemporalClassifier] ONNX model unavailable — using rule-based fallback.'
      );
    }
  }

  isModelLoaded(): boolean {
    return this.modelSession !== null;
  }

  /**
   * Add one frame and return the current behavior classification.
   * Call this once per inference frame (every ~200 ms at 5 FPS).
   */
  update(frame: TemporalFeatureFrame): BehaviorResult {
    this.featureBuffer.push(this.extractFeatures(frame));
    if (this.featureBuffer.length > WINDOW_SIZE) this.featureBuffer.shift();

    if (this.featureBuffer.length < WINDOW_SIZE) {
      // Not enough history yet — single-frame rule-based
      this.lastResult = this.ruleBasedSingleFrame(frame);
      return this.lastResult;
    }

    // Prefer ONNX when loaded; otherwise use temporal rule-based
    this.lastResult = this.modelSession
      ? this.onnxClassify()
      : this.ruleBasedTemporal();

    return this.lastResult;
  }

  getLastResult(): BehaviorResult {
    return this.lastResult;
  }

  reset(): void {
    this.featureBuffer = [];
    this.lastResult = {
      label: 'distracted',
      attentionScore: 50,
      confidence: 0,
      source: 'rule_based',
    };
  }

  // ==================== FEATURE EXTRACTION ====================

  /**
   * Encode one frame into a FEATURE_DIM-dimensional vector.
   * Values are normalized to roughly [0, 1] or [-1, 1] for ML compatibility.
   */
  private extractFeatures(frame: TemporalFeatureFrame): number[] {
    const { metrics, detectedObjects, handProximity } = frame;

    const hasBook = detectedObjects.some(
      (o) => o.class === 'book' || o.class === 'notebook'
    )
      ? 1
      : 0;
    const hasPhone = detectedObjects.some((o) => o.class === 'phone') ? 1 : 0;
    const hasLaptop = detectedObjects.some((o) => o.class === 'laptop') ? 1 : 0;
    const hasNone = !hasBook && !hasPhone && !hasLaptop ? 1 : 0;

    return [
      clamp(metrics.gazeX ?? 0, -1, 1),
      clamp(metrics.gazeY ?? 0, -1, 1),
      clamp((metrics.pitch ?? 0) / 90, -1, 1),   // normalised to [-1, 1]
      clamp((metrics.yaw ?? 0) / 90, -1, 1),
      clamp(metrics.ear ?? 0.3, 0, 1),
      clamp(metrics.perclos / 100, 0, 1),
      clamp(metrics.headPoseStability / 100, 0, 1),
      hasBook,
      hasPhone,
      hasLaptop,
      hasNone,
      clamp(metrics.fidgetScore / 100, 0, 1),
      this.encodeProximity(handProximity),
    ];
  }

  /** Map HandObjectProximity → scalar code for the feature vector. */
  private encodeProximity(proximity: HandObjectProximity): number {
    const map: Record<HandObjectProximity, number> = {
      holding_book: 1.0,
      holding_pen: 0.8,
      hands_on_desk: 0.5,
      holding_phone: 0.2,
      hands_not_visible: 0.0,
    };
    return map[proximity] ?? 0;
  }

  // ==================== RULE-BASED CLASSIFIERS ====================

  /**
   * Single-frame rule-based classification (< 15 frames buffered).
   * Implements the context-aware scoring logic from ARCHITECTURE.md §5.1.
   */
  private ruleBasedSingleFrame(frame: TemporalFeatureFrame): BehaviorResult {
    const { metrics, detectedObjects, handProximity } = frame;

    // 1. Absent — face too small / not present
    if (!metrics.faceArea || metrics.faceArea < 0.005) {
      return this.makeResult('absent', 1.0);
    }

    // 2. Drowsy — low EAR + elevated PERCLOS
    if ((metrics.ear ?? 0.3) < 0.22 && metrics.perclos > 20) {
      return this.makeResult('drowsy', 0.85);
    }

    const pitch = metrics.pitch ?? 0;
    const isLookingDown = pitch < -15;

    // 3. Phone usage — phone detected + hand holding it or downward gaze
    const phoneDetected = detectedObjects.some((o) => o.class === 'phone');
    if (
      phoneDetected &&
      (handProximity === 'holding_phone' || isLookingDown)
    ) {
      return this.makeResult('using_phone', 0.80);
    }

    // 4. Reading — book near hand + downward gaze
    const bookDetected = detectedObjects.some(
      (o) => o.class === 'book' || o.class === 'notebook'
    );
    if (bookDetected && (handProximity === 'holding_book' || isLookingDown)) {
      return this.makeResult('attentive_reading', 0.75);
    }

    // 5. Screen attention — stable gaze near centre
    const gazeDeviation = Math.sqrt(
      (metrics.gazeX ?? 0) ** 2 + (metrics.gazeY ?? 0) ** 2
    );
    if (gazeDeviation < 0.3 && Math.abs(pitch) < 15) {
      return this.makeResult('attentive_screen', 0.70);
    }

    return this.makeResult('distracted', 0.60);
  }

  /**
   * Temporal rule-based classification over the averaged 15-frame window.
   * Provides more stable results than single-frame by smoothing transient noise.
   */
  private ruleBasedTemporal(): BehaviorResult {
    // Compute per-feature average across the window
    const avg = new Array<number>(FEATURE_DIM).fill(0);
    for (const vec of this.featureBuffer) {
      vec.forEach((v, i) => {
        avg[i] += v;
      });
    }
    avg.forEach((_, i) => {
      avg[i] /= this.featureBuffer.length;
    });

    const [
      gazeX,
      gazeY,
      pitchNorm,
      ,              // yawNorm (unused in rules)
      ear,
      perclos,
      stability,
      hasBook,
      hasPhone,
      ,              // hasLaptop
      ,              // hasNone
      ,              // fidget
      proximityCode,
    ] = avg;

    const pitch = pitchNorm * 90;
    const isLookingDown = pitch < -15;

    // Drowsy: low ear + high perclos sustained over window
    if (ear < 0.22 && perclos > 0.2) {
      return this.makeResult('drowsy', 0.88);
    }

    // Phone: phone present + hand holding it (proximity < 0.3 = holding_phone)
    if (hasPhone > 0.5 && proximityCode < 0.3) {
      return this.makeResult('using_phone', 0.83);
    }

    // Reading: book present + sustained downward gaze
    if (hasBook > 0.5 && isLookingDown) {
      return this.makeResult('attentive_reading', 0.82);
    }

    // Screen attention: stable, centred gaze
    const gazeDeviation = Math.sqrt(gazeX ** 2 + gazeY ** 2);
    if (gazeDeviation < 0.3 && Math.abs(pitch) < 15 && stability > 0.6) {
      return this.makeResult('attentive_screen', 0.78);
    }

    return this.makeResult('distracted', 0.65);
  }

  /**
   * ONNX 1D-CNN inference over the full 15-frame window.
   * Input tensor shape: [1, WINDOW_SIZE, FEATURE_DIM] = [1, 15, 13]
   * Output: softmax over 6 BehaviorLabel classes.
   *
   * Falls back to temporal rule-based if inference throws.
   */
  private onnxClassify(): BehaviorResult {
    const CLASSES: BehaviorLabel[] = [
      'attentive_screen',
      'attentive_reading',
      'using_phone',
      'distracted',
      'drowsy',
      'absent',
    ];

    try {
      // Flatten [WINDOW_SIZE, FEATURE_DIM] → Float32Array
      const flat = new Float32Array(WINDOW_SIZE * FEATURE_DIM);
      this.featureBuffer.forEach((vec, t) => {
        vec.forEach((v, f) => {
          flat[t * FEATURE_DIM + f] = v;
        });
      });

      // Fire-and-forget async call; return cached result until it resolves.
      // In practice, 1D-CNN on 15×13 is < 1 ms even on CPU.
      this.runOnnxAsync(flat, CLASSES).then((result) => {
        if (result) this.lastResult = result;
      });
    } catch {
      return this.ruleBasedTemporal();
    }

    return this.lastResult;
  }

  private async runOnnxAsync(
    flat: Float32Array,
    classes: BehaviorLabel[]
  ): Promise<BehaviorResult | null> {
    try {
      const pkg2 = 'onnxruntime-web';
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ort: any = await import(pkg2 as string);
      const tensor = new ort.Tensor('float32', flat, [1, WINDOW_SIZE, FEATURE_DIM]);
      const output = await this.modelSession.run({ input: tensor });
      const probs: Float32Array = output['output'].data;

      let maxIdx = 0;
      let maxProb = probs[0];
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > maxProb) {
          maxProb = probs[i];
          maxIdx = i;
        }
      }

      return this.makeResult(classes[maxIdx], maxProb, 'temporal');
    } catch {
      return null;
    }
  }

  // ==================== HELPERS ====================

  private makeResult(
    label: BehaviorLabel,
    confidence: number,
    source: BehaviorResult['source'] = this.featureBuffer.length >= WINDOW_SIZE
      ? 'temporal'
      : 'rule_based'
  ): BehaviorResult {
    return {
      label,
      attentionScore: BEHAVIOR_ATTENTION_SCORES[label],
      confidence: clamp(confidence, 0, 1),
      source,
    };
  }
}
