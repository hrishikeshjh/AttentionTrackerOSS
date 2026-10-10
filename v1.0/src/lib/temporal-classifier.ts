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
 * loadModel() contract (DACE plan D10, FR10), matching the v2.0 export
 * (v2.0/training/export_onnx.py): input name 'input', layout channels-first
 * (1, 13, 15), output name 'logits' (raw logits -> softmax here), class order
 * attentive_screen, attentive_reading, using_phone, distracted, drowsy, absent,
 * and the v2.0 proximity code set (0, .25, .5, .75, 1). All of these are
 * options. Any load or inference failure sets lastError, unloads the model
 * and falls back to the rule layer; the rule-based path is unchanged.
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

/** Class order of the v2.0 1D-CNN (index order fixed). */
export const MODEL_CLASSES: readonly BehaviorLabel[] = Object.freeze([
  'attentive_screen',
  'attentive_reading',
  'using_phone',
  'distracted',
  'drowsy',
  'absent',
]);

/** v1.0 rule-layer proximity codes (unchanged; used by the rule-based path). */
export const PROXIMITY_CODES_V1: Readonly<Record<HandObjectProximity, number>> = Object.freeze({
  holding_book: 1.0,
  holding_pen: 0.8,
  hands_on_desk: 0.5,
  holding_phone: 0.2,
  hands_not_visible: 0.0,
});

/** v2.0 training proximity codes (v2.0/features/feature_vector.py _PROXIMITY_CODE). */
export const PROXIMITY_CODES_V2: Readonly<Record<HandObjectProximity, number>> = Object.freeze({
  hands_not_visible: 0.0,
  hands_on_desk: 0.25,
  holding_pen: 0.5,
  holding_book: 0.75,
  holding_phone: 1.0,
});

export type ModelLoadOptions = {
  /** 'channels_first' = [1, 13, 15] (v2.0 export, default); 'channels_last' = [1, 15, 13] */
  layout?: 'channels_first' | 'channels_last';
  inputName?: string;
  outputName?: string;
  /** true (default): the output is logits and softmax is applied; false: already probabilities */
  outputIsLogits?: boolean;
  /** proximity code set the model was trained with (default 'v2') */
  proximityCodes?: 'v1' | 'v2';
  classes?: readonly BehaviorLabel[];
  /** onnxruntime-web WASM location (default '/onnx/') */
  wasmPaths?: string;
};

type ResolvedModelOptions = Required<Omit<ModelLoadOptions, 'classes'>> & { classes: readonly BehaviorLabel[] };

export function softmax(values: ArrayLike<number>): number[] {
  const arr = Array.from(values);
  const m = Math.max(...arr);
  const e = arr.map((v) => Math.exp(v - m));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

/**
 * Pack a [WINDOW_SIZE][FEATURE_DIM] buffer into the model's input layout.
 * channels_first: index f * WINDOW + t; channels_last: index t * FEATURE_DIM + f.
 */
export function packWindow(buffer: readonly number[][], layout: 'channels_first' | 'channels_last'): Float32Array {
  const T = buffer.length;
  const F = buffer[0]?.length ?? 0;
  const flat = new Float32Array(T * F);
  buffer.forEach((vec, t) => {
    vec.forEach((v, f) => {
      flat[layout === 'channels_first' ? f * T + t : t * F + f] = v;
    });
  });
  return flat;
}

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
  /** proximity label per buffered frame (re-encoded with the model's code set) */
  private proximityBuffer: HandObjectProximity[] = [];

  /** Optional ONNX session (loaded via loadModel). */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private modelSession: any = null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private ort: any = null;
  private modelOptions: ResolvedModelOptions | null = null;
  /** last load/inference error (null when none) */
  lastError: string | null = null;

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
  async loadModel(modelUrl: string, options: ModelLoadOptions = {}): Promise<boolean> {
    const opts: ResolvedModelOptions = {
      layout: options.layout ?? 'channels_first',
      inputName: options.inputName ?? 'input',
      outputName: options.outputName ?? 'logits',
      outputIsLogits: options.outputIsLogits ?? true,
      proximityCodes: options.proximityCodes ?? 'v2',
      classes: options.classes ?? MODEL_CLASSES,
      wasmPaths: options.wasmPaths ?? '/onnx/',
    };
    try {
      const pkg = 'onnxruntime-web';
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ort: any = await import(pkg as string);
      if (ort?.env?.wasm) ort.env.wasm.wasmPaths = opts.wasmPaths;
      const session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['webgpu', 'wasm'],
      });
      const inNames: string[] | undefined = session?.inputNames;
      const outNames: string[] | undefined = session?.outputNames;
      if (inNames && !inNames.includes(opts.inputName)) {
        throw new Error(`model input '${opts.inputName}' not found (has: ${inNames.join(', ')})`);
      }
      if (outNames && !outNames.includes(opts.outputName)) {
        throw new Error(`model output '${opts.outputName}' not found (has: ${outNames.join(', ')})`);
      }
      this.ort = ort;
      this.modelSession = session;
      this.modelOptions = opts;
      this.lastError = null;
      return true;
    } catch (e) {
      this.modelSession = null;
      this.modelOptions = null;
      this.lastError = e instanceof Error ? e.message : String(e);
      console.warn(
        '[TemporalClassifier] ONNX model unavailable — using rule-based fallback.',
        this.lastError
      );
      return false;
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
    this.proximityBuffer.push(frame.handProximity);
    if (this.proximityBuffer.length > WINDOW_SIZE) this.proximityBuffer.shift();

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
    this.proximityBuffer = [];
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
   * ONNX 1D-CNN inference over the full 15-frame window, in the layout,
   * input/output names and proximity code set given to loadModel().
   * Asynchronous: returns the previous result until inference resolves
   * (a 1D-CNN on 15x13 is well under a frame). On failure the model is
   * unloaded, lastError is set and the temporal rule layer takes over.
   */
  private onnxClassify(): BehaviorResult {
    const opts = this.modelOptions;
    if (!opts) return this.ruleBasedTemporal();
    try {
      const codes = opts.proximityCodes === 'v2' ? PROXIMITY_CODES_V2 : PROXIMITY_CODES_V1;
      const buf = this.featureBuffer.map((vec, t) => {
        const v = [...vec];
        const prox = this.proximityBuffer[t];
        v[FEATURE_DIM - 1] = prox !== undefined ? (codes[prox] ?? 0) : v[FEATURE_DIM - 1];
        return v;
      });
      const flat = packWindow(buf, opts.layout);
      const dims = opts.layout === 'channels_first' ? [1, FEATURE_DIM, WINDOW_SIZE] : [1, WINDOW_SIZE, FEATURE_DIM];
      this.runOnnxAsync(flat, dims, opts).then((result) => {
        if (result) this.lastResult = result;
      });
    } catch (e) {
      this.onModelFailure(e);
      return this.ruleBasedTemporal();
    }
    return this.lastResult;
  }

  private async runOnnxAsync(
    flat: Float32Array,
    dims: number[],
    opts: ResolvedModelOptions
  ): Promise<BehaviorResult | null> {
    try {
      const ort = this.ort;
      const tensor = new ort.Tensor('float32', flat, dims);
      const output = await this.modelSession.run({ [opts.inputName]: tensor });
      const raw = output?.[opts.outputName]?.data as ArrayLike<number> | undefined;
      if (!raw || raw.length !== opts.classes.length) {
        throw new Error(`unexpected '${opts.outputName}' size ${raw?.length ?? 'missing'} (expected ${opts.classes.length})`);
      }
      const probs = opts.outputIsLogits ? softmax(raw) : Array.from(raw);

      let maxIdx = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[maxIdx]) maxIdx = i;
      return this.makeResult(opts.classes[maxIdx], probs[maxIdx], 'temporal');
    } catch (e) {
      this.onModelFailure(e);
      return null;
    }
  }

  private onModelFailure(e: unknown): void {
    this.lastError = e instanceof Error ? e.message : String(e);
    this.modelSession = null;
    this.modelOptions = null;
    console.warn('[TemporalClassifier] ONNX inference failed — falling back to rules.', this.lastError);
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
