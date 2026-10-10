/**
 * DaceEngine: per-tick orchestrator for DACE mode (plan sections 3, 8).
 *
 *   measure (provider) -> primary face (PrimaryFaceSelector)
 *     -> staggered desk evidence: detector on ticks i % 3 == 0, hands on i % 3 == 1
 *        (outputs carried forward between runs)
 *     -> M: pose in degrees (+down), pitch reference, gaze, gaze-corrected openness
 *     -> DaceFrame -> RuleRunner (window, PERCLOS, rules, abstention)
 *     -> feature mapping onto the existing FEATURE_WEIGHTS keys
 *     -> { metrics, predict } for MLEngine.predict(metrics, { dace: predict })
 *
 * Reliability (annex B.2): every stage runs in try/catch. A failure, or a
 * frame slower than 2x the 200 ms budget, returns { fallback: true } and the
 * caller scores that frame with the legacy path (MLEngine.predict(metrics)
 * with the unsubstituted legacy metrics). The engine never throws.
 *
 * No-face ticks are fed to the window (needed for `absent` and U3).
 *
 * This module is browser-free (providers are injected), so it runs under
 * vitest with mock providers.
 */
import type { DaceConfig } from './config';
import type { DaceParams } from '../scoring/dace/params';
import { DEFAULT_DACE_PARAMS } from '../scoring/dace/params';
import type { DaceFace, DaceFrame } from '../scoring/dace/types';
import type { BaselineProvider, PitchReferenceReset } from '../scoring/dace/pitch-reference';
import { PitchReference } from '../scoring/dace/pitch-reference';
import { estimateGaze } from '../scoring/dace/gaze-blendshape';
import { OpennessModel, meshEAR } from '../scoring/dace/eye-openness';
import { assignHandsToFace } from '../scoring/dace/hand-object';
import { filterByConfidence } from '../scoring/dace/evidence';
import {
  DACE_LABEL_SCORES,
  mapContextFeatures,
  substituteMeasurement,
  type MeasurementSubstitution,
} from '../scoring/dace/feature-mapping';
import type { RuleId } from '../scoring/dace/types';
import { FaceMetricsAggregator } from '../scoring/aggregator';
import { calculateHeadPose } from '../scoring/pose/head-pose';
import { PrimaryFaceSelector } from '../scoring/face-selection';
import type {
  DaceMeasurementSnapshot,
  DetectedObject,
  ExtendedFaceMetrics,
  HandObjectProximity,
  HandResult,
} from '../types';
import type { DacePredictInput } from '../ml-engine';
import type { FaceMeasurement, FrameInput, MeasurementProvider } from './measurement/provider';
import type { DaceObjectDetector } from './detectors/types';
import type { DaceHandSource } from './hands/hand-landmarker';
import { RuleRunner, type RunnerStep } from './rule-runner';
import { LatencyProbe } from './latency';
import type { EvalLogger } from './eval-log';
import { buildSelfTestReport, contextGate, type ContextGate, type SelfTestReport } from './self-test';

export type DaceEngineDeps = {
  config: Readonly<DaceConfig>;
  params?: DaceParams;
  measurement: MeasurementProvider;
  detector?: DaceObjectDetector | null;
  hands?: DaceHandSource | null;
  logger?: EvalLogger | null;
  latency?: LatencyProbe;
  baseline?: BaselineProvider | null;
  /** per-frame budget (ms); a frame over 2x the budget is scored by the legacy path */
  frameBudgetMs?: number;
  now?: () => number;
};

export type DaceTickResult = {
  i: number;
  t: number;
  fallback: boolean;
  fallbackReason: string | null;
  /** primary face (for drawing); null when no face */
  face: FaceMeasurement | null;
  /** metrics for display; null when no face this tick */
  metrics: ExtendedFaceMetrics | null;
  /** metrics to score with (may be the last face metrics during an Unsure no-face tick); null = no score */
  scoreMetrics: ExtendedFaceMetrics | null;
  /** DACE predict options; null on fallback (score scoreMetrics with the legacy path) or when there is no score */
  predict: DacePredictInput | null;
  step: RunnerStep | null;
  calibrating: boolean;
  /** pitch-reference reset on this tick ('camera_move' -> dashboard toast) */
  reset: PitchReferenceReset | null;
  totalMs: number;
};

const CONSECUTIVE_ERRORS_TO_DISABLE = 5;

/** DACE evidence -> the existing HandObjectProximity field (display + contradiction detector). */
export function proximityFromEvidence(
  s: { phoneInUse: boolean; book: boolean; writing: boolean },
  handsVisible: boolean
): HandObjectProximity {
  if (s.phoneInUse) return 'holding_phone';
  if (s.book) return 'holding_book';
  if (s.writing) return 'holding_pen';
  return handsVisible ? 'hands_on_desk' : 'hands_not_visible';
}

export class DaceEngine {
  readonly config: Readonly<DaceConfig>;
  readonly params: DaceParams;
  readonly latency: LatencyProbe;
  private readonly measurement: MeasurementProvider;
  private detector: DaceObjectDetector | null;
  private hands: DaceHandSource | null;
  private readonly logger: EvalLogger | null;
  private readonly baseline: BaselineProvider | null;
  private readonly budget: number;
  private readonly now: () => number;

  private selector = new PrimaryFaceSelector();
  private aggregator = new FaceMetricsAggregator();
  private pitchRef: PitchReference;
  private openness: OpennessModel;
  private runner: RuleRunner;

  private frameIndex = 0;
  private objects: DetectedObject[] = [];
  private handResults: HandResult[] = [];
  private detectorAvailable = false;
  private handsAvailable = false;
  private gate: ContextGate;
  private detectorErrors = 0;
  private handErrors = 0;
  private prevContextPitch: number | null = null;
  private lastScoreMetrics: ExtendedFaceMetrics | null = null;
  private lastLabelAttentive = false;
  private fallbackCount = 0;
  selfTestReport: SelfTestReport | null = null;

  constructor(deps: DaceEngineDeps) {
    this.config = deps.config;
    this.params = deps.params ?? { ...DEFAULT_DACE_PARAMS, earPrior: { ...DEFAULT_DACE_PARAMS.earPrior } };
    this.measurement = deps.measurement;
    this.detector = deps.detector ?? null;
    this.hands = deps.hands ?? null;
    this.logger = deps.logger ?? null;
    this.latency = deps.latency ?? new LatencyProbe();
    this.baseline = deps.config.baseline === 'peb' ? (deps.baseline ?? null) : null;
    this.budget = deps.frameBudgetMs ?? this.params.frameIntervalMs;
    this.now = deps.now ?? (() => (typeof performance !== 'undefined' ? performance.now() : Date.now()));
    this.pitchRef = new PitchReference(this.params, this.baseline);
    this.openness = new OpennessModel(this.params);
    this.runner = new RuleRunner(this.params, {
      evidence: this.config.evidence,
      rules: this.config.rules,
      abstain: this.config.abstain === 'off' ? 'off' : 'rules',
      shuffleObjects: this.config.replay.shuffleObjects,
    });
    this.detectorAvailable = !!this.detector?.isReady();
    this.handsAvailable = !!this.hands?.isReady();
    this.gate = contextGate(this.config.evidence, this.detectorAvailable, this.handsAvailable);
  }

  /** Run component self-tests and recompute the FR4 context gate. Never throws. */
  async selfTest(): Promise<SelfTestReport> {
    const safe = async <T>(f: (() => Promise<T>) | undefined): Promise<T | null> => {
      if (!f) return null;
      try {
        return await f();
      } catch (e) {
        return { ok: false, ms: null, error: e instanceof Error ? e.message : String(e) } as T;
      }
    };
    const measurement = await safe(this.measurement.selfTest?.bind(this.measurement));
    const hands = this.hands ? await safe(this.hands.selfTest?.bind(this.hands)) ?? { ok: this.hands.isReady(), ms: null } : null;
    let detector = null;
    try {
      detector = this.detector ? this.detector.status() : null;
    } catch {
      detector = null;
    }
    const report = buildSelfTestReport({ measurement, detector, hands });
    this.detectorAvailable = report.detectorAvailable;
    this.handsAvailable = report.handsAvailable;
    this.gate = contextGate(this.config.evidence, this.detectorAvailable, this.handsAvailable);
    this.selfTestReport = report;
    this.logger?.log('event', 0, { event: 'self_test', report, gate: this.gate });
    return report;
  }

  getGate(): Readonly<ContextGate> {
    return this.gate;
  }

  isDetectorAvailable(): boolean {
    return this.detectorAvailable;
  }

  getFallbackCount(): number {
    return this.fallbackCount;
  }

  measurementLabel(): string {
    return this.measurement.label;
  }

  drawFace(canvas: HTMLCanvasElement, face: FaceMeasurement): void {
    try {
      this.measurement.draw(canvas, face);
    } catch {
      /* drawing is cosmetic */
    }
  }

  async processFrame(input: FrameInput): Promise<DaceTickResult> {
    const i = this.frameIndex++;
    const t = input.t;
    const t0 = this.now();
    let base: ExtendedFaceMetrics | null = null;
    let primaryFace: FaceMeasurement | null = null;
    let calibrating = false;
    let reset: PitchReferenceReset | null = null;

    const fallback = (stage: string, e: unknown): DaceTickResult => {
      this.fallbackCount++;
      const reason = `${stage}: ${e instanceof Error ? e.message : String(e)}`;
      this.logger?.log('event', t, { event: 'fallback', i, stage, reason });
      if (base) this.lastScoreMetrics = base;
      return {
        i, t, fallback: true, fallbackReason: reason, face: primaryFace,
        metrics: base, scoreMetrics: base, predict: null, step: null,
        calibrating, reset, totalMs: this.now() - t0,
      };
    };

    // 1. measurement
    let cands: Awaited<ReturnType<MeasurementProvider['measure']>>;
    try {
      cands = await this.latency.time('measure', () => this.measurement.measure(input));
    } catch (e) {
      return fallback('measure', e);
    }
    const primary = this.selector.select(cands, input.width, input.height);
    primaryFace = primary ? primary.detection : null;

    // legacy aggregator (MAR, blink/fatigue trackers, face area...) — also the fallback metrics
    if (primaryFace) {
      try {
        base = this.aggregator.calculateExtendedMetrics(primaryFace.landmarks68, input.width, input.height, t);
      } catch (e) {
        return fallback('aggregate', e);
      }
    }

    // 2. staggered desk evidence (carried forward between runs)
    let objectsFresh = false;
    const N = Math.max(1, this.params.detectorEveryNFrames);
    if (this.config.evidence) {
      if (i % N === 0 && this.gate.useObjects && this.detector) {
        try {
          this.objects = await this.latency.time('detector', () =>
            this.detector!.detect(input.image, t, { minConf: this.params.cMin })
          );
          this.detectorErrors = 0;
        } catch (e) {
          this.objects = [];
          this.onComponentError('detector', t, e);
        }
        objectsFresh = true;
      }
      if (i % N === 1 % N && this.gate.useHands && this.hands) {
        try {
          this.handResults = await this.latency.time('hands', () => this.hands!.detect(input.image, t));
          this.handErrors = 0;
        } catch (e) {
          this.handResults = [];
          this.onComponentError('hands', t, e);
        }
      }
    }

    // 3. M layer: per-face measurement
    let face: DaceFace | null = null;
    let sub: MeasurementSubstitution = {};
    try {
      const r = this.measureFace(primaryFace, input, t);
      face = r.face;
      sub = r.sub;
      calibrating = r.calibrating;
      reset = r.reset;
    } catch (e) {
      return fallback('measurement_layer', e);
    }

    // 4. E/T/U
    let step: RunnerStep;
    try {
      const hands = this.gate.useHands ? assignHandsToFace(this.handResults, face?.faceBox ?? null, this.params) : [];
      const objects = this.gate.useObjects ? this.objects : [];
      const frame: DaceFrame = { t, face, hands, objects };
      step = await this.latency.time('rules', () => this.runner.push(frame, objectsFresh));
      if (step.output?.label === 'absent') this.pitchRef.markAbsent();
      if (face && step.effective.label && !step.effective.unsure) {
        this.lastLabelAttentive = step.effective.label === 'attentive_screen' || step.effective.label === 'attentive_reading';
      }
      this.logFrame(i, t, frame, objectsFresh);
    } catch (e) {
      return fallback('rules', e);
    }

    // 5. mapping onto the existing scoring features
    let metrics: ExtendedFaceMetrics | null = null;
    let scoreMetrics: ExtendedFaceMetrics | null = null;
    let predict: DacePredictInput | null = null;
    try {
      const mapStart = this.now();
      if (base && face) {
        metrics = this.buildMetrics(base, face, sub, step);
        scoreMetrics = metrics;
        this.lastScoreMetrics = metrics;
      } else if (step.effective.unsure && this.lastScoreMetrics) {
        // no face this tick but the window is Unsure (e.g. U3): the score is held,
        // so the last face metrics only feed the contradiction detector.
        scoreMetrics = this.lastScoreMetrics;
      }
      if (scoreMetrics) predict = this.buildPredict(step, face, metrics !== null);
      this.latency.record('mapping', this.now() - mapStart);
    } catch (e) {
      return fallback('mapping', e);
    }

    const totalMs = this.now() - t0;
    this.latency.record('total', totalMs);
    this.logWindow(i, t, step, face, totalMs);

    if (totalMs > 2 * this.budget) {
      this.fallbackCount++;
      const reason = `over_budget: ${totalMs.toFixed(1)} ms > ${2 * this.budget} ms`;
      this.logger?.log('event', t, { event: 'fallback', i, stage: 'budget', reason });
      return {
        i, t, fallback: true, fallbackReason: reason, face: primaryFace,
        metrics: metrics ?? base, scoreMetrics: base, predict: null, step,
        calibrating, reset, totalMs,
      };
    }

    return { i, t, fallback: false, fallbackReason: null, face: primaryFace, metrics, scoreMetrics, predict, step, calibrating, reset, totalMs };
  }

  private onComponentError(component: 'detector' | 'hands', t: number, e: unknown): void {
    const n = component === 'detector' ? ++this.detectorErrors : ++this.handErrors;
    this.logger?.log('event', t, { event: 'component_error', component, n, error: e instanceof Error ? e.message : String(e) });
    if (n >= CONSECUTIVE_ERRORS_TO_DISABLE) {
      if (component === 'detector') this.detectorAvailable = false;
      else this.handsAvailable = false;
      this.gate = contextGate(this.config.evidence, this.detectorAvailable, this.handsAvailable);
      this.logger?.log('event', t, { event: 'component_disabled', component, gate: this.gate });
    }
  }

  private measureFace(
    m: FaceMeasurement | null,
    input: FrameInput,
    t: number
  ): { face: DaceFace | null; sub: MeasurementSubstitution; calibrating: boolean; reset: PitchReferenceReset | null } {
    const p = this.params;
    if (!m) {
      const st = this.pitchRef.update({ t, pitchDeg: null, faceScale: null, gazeV: null });
      return { face: null, sub: {}, calibrating: st.calibrating, reset: st.reset };
    }
    const kind = this.measurement.kind;
    let pitchDeg: number | null = null;
    let yawDeg: number | null = null;
    let rollDeg: number | null = null;
    if (kind === 'legacy') {
      // Legacy formula, NOT degrees and NOT corrected (it rises when the head
      // tilts down, which is why it is passed through as "+down" unchanged).
      const hp = calculateHeadPose(m.landmarks68);
      pitchDeg = hp.pitch;
      yawDeg = hp.yaw;
      rollDeg = hp.roll;
    } else if (m.pose) {
      pitchDeg = m.pose.pitch;
      yawDeg = m.pose.yaw;
      rollDeg = m.pose.roll;
    }

    let gazeV: number | null = null;
    let gazeH: number | null = null;
    let gazeValid = false;
    let ear: number | null = null;
    let residual: number | null = null;
    let eyeClosed: boolean | null = null;
    if (kind === 'mediapipe') {
      const g = estimateGaze(m.blendshapes, m.mesh, p);
      gazeV = g.gazeV;
      gazeH = g.gazeH;
      gazeValid = g.gazeValid;
      ear = meshEAR(m.mesh, input.width / input.height);
      const peb = this.baseline?.openEyeLevel?.() ?? null;
      residual = this.openness.residual(ear, gazeValid ? gazeV : null, peb);
      eyeClosed = this.openness.isClosed(residual);
    }

    const ref = this.pitchRef.update({ t, pitchDeg, faceScale: m.faceBox.width, gazeV: gazeValid ? gazeV : null });

    // openness model samples: eyes-open frames while the student is attentive
    if (kind === 'mediapipe' && ear !== null && this.lastLabelAttentive && eyeClosed !== true) {
      this.openness.addSample(gazeValid ? gazeV : null, ear);
    }
    if (kind === 'mediapipe' && this.frameIndex % p.deskWindowFrames === 0) this.openness.refit();

    const face: DaceFace = {
      pitchDeg,
      relPitchDeg: ref.relPitchDeg,
      yawDeg,
      rollDeg,
      gazeV,
      gazeH,
      gazeValid,
      ear,
      residualOpenness: residual,
      eyeClosed,
      faceBox: m.faceBox,
    };

    let sub: MeasurementSubstitution = {};
    if (kind === 'mediapipe') {
      sub = {
        pitchDownDeg: ref.relPitchDeg,
        yawDeg,
        rollDeg,
        gazeH: gazeValid ? gazeH : null,
        gazeV: gazeValid ? gazeV : null,
        residualOpenness: residual,
      };
    } else if (kind === 'pnp68') {
      // A'': pose only, absolute degrees (minimal-change baseline)
      sub = { pitchDownDeg: pitchDeg, yawDeg, rollDeg };
    }
    return { face, sub, calibrating: ref.calibrating, reset: ref.reset };
  }

  private buildMetrics(base: ExtendedFaceMetrics, face: DaceFace, sub: MeasurementSubstitution, step: RunnerStep): ExtendedFaceMetrics {
    const s = step.summary;
    const isMp = this.measurement.kind === 'mediapipe';
    const fullSub: MeasurementSubstitution = isMp ? { ...sub, correctedPerclos: step.perclos } : sub;
    const snapshot: DaceMeasurementSnapshot = {
      pitchDeg: face.pitchDeg,
      relPitchDeg: face.relPitchDeg,
      yawDeg: face.yawDeg,
      rollDeg: face.rollDeg,
      gazeV: face.gazeV,
      gazeH: face.gazeH,
      gazeValid: face.gazeValid,
      residualOpenness: face.residualOpenness,
      correctedPerclos: step.perclos,
      lookingDown: step.evidence.lookingDown,
      calibrating: this.pitchRef.isCalibrating(),
    };
    const m = substituteMeasurement(base, fullSub, this.params, snapshot);
    if (this.config.evidence) {
      m.deskEvidence = {
        phoneInUse: s.phoneInUse,
        phonePresent: s.phonePresent,
        book: s.book,
        writing: s.writing,
        deskVisible: s.deskVisible,
        conflict: s.conflict,
      };
      m.handObjectProximity = proximityFromEvidence(s, step.frame.hands.length > 0);
      if (this.gate.exposeObjects) m.detectedObjects = filterByConfidence(step.frame.objects, this.params.cMin);
    }
    const e = step.effective;
    if (e.rule !== 'none') m.daceOutput = { label: e.label, unsure: e.unsure, reason: e.reason, rule: e.rule };
    if (e.label && !e.unsure) {
      m.behaviorResult = { label: e.label, attentionScore: DACE_LABEL_SCORES[e.label], confidence: 1, source: 'dace_rules' };
    }
    return m;
  }

  private buildPredict(step: RunnerStep, face: DaceFace | null, hasFace: boolean): DacePredictInput {
    const e = step.effective;
    const out: DacePredictInput = {
      weights: this.gate.weights,
      unsure: e.unsure,
      ...(e.unsure && e.reason ? { unsureReason: e.reason } : {}),
      label: e.unsure ? null : e.label,
      rule: e.rule,
    };
    if (this.gate.mapContext && hasFace && face) {
      const ctx = mapContextFeatures(
        {
          output: { label: e.label, unsure: e.unsure, rule: e.rule as RuleId },
          evidence: { phoneInUse: step.summary.phoneInUse, book: step.summary.book, writing: step.summary.writing },
          handsVisible: step.frame.hands.length > 0,
          relPitchDeg: face.relPitchDeg,
          lookingDown: step.evidence.lookingDown,
          gazeH: face.gazeH,
          gazeV: face.gazeV,
          gazeValid: face.gazeValid,
          previous: this.prevContextPitch === null ? null : { contextAwarePitch: this.prevContextPitch },
        },
        this.params
      );
      this.prevContextPitch = ctx.contextAwarePitch;
      out.featureOverrides = {
        contextAwarePitch: ctx.contextAwarePitch,
        objectContext: ctx.objectContext,
        handObjectProximity: ctx.handObjectProximity,
        temporalPattern: ctx.temporalPattern,
        gazeDeviation: ctx.gazeDeviation,
      };
    } else if (this.gate.mapContext) {
      // no face: the score is held anyway; keep the context features neutral
      out.featureOverrides = { contextAwarePitch: this.prevContextPitch ?? 50, objectContext: 65, handObjectProximity: 60, temporalPattern: 60 };
    }
    return out;
  }

  private logFrame(i: number, t: number, frame: DaceFrame, objectsFresh: boolean): void {
    if (!this.logger?.accepts('frame')) return;
    this.logger.log('frame', t, {
      i,
      objectsFresh,
      detectorAvailable: this.detectorAvailable,
      handsAvailable: this.handsAvailable,
      // the M-layer output + desk inputs BEFORE the runner (no evidence strip,
      // no shuffle): enough to re-run E/T/U offline under any preset
      frame,
    });
  }

  private logWindow(i: number, t: number, step: RunnerStep, face: DaceFace | null, totalMs: number): void {
    if (!this.logger?.accepts('window')) return;
    const s = step.summary;
    this.logger.log('window', t, {
      i,
      label: step.effective.label,
      unsure: step.effective.unsure,
      reason: step.effective.reason,
      rule: step.effective.rule,
      keptFrom: step.output?.keptFrom ?? null,
      face: face !== null,
      relPitchDeg: face?.relPitchDeg ?? null,
      gazeV: face?.gazeV ?? null,
      gazeH: face?.gazeH ?? null,
      gazeValid: face?.gazeValid ?? false,
      residualOpenness: face?.residualOpenness ?? null,
      lookingDown: step.evidence.lookingDown,
      perclos: step.perclos,
      longestClosureMs: step.longestClosureMs,
      summary: {
        noFaceCount: s.noFaceCount,
        lookingDownCount: s.lookingDownCount,
        phoneInUseCount: s.phoneInUseCount,
        phoneDetectedCount: s.phoneDetectedCount,
        phoneInUse: s.phoneInUse,
        phonePresent: s.phonePresent,
        book: s.book,
        writing: s.writing,
        conflict: s.conflict,
        deskVisible: s.deskVisible,
        downEpisodeMs: s.downEpisodeMs,
        screenAttentive: s.screenAttentive,
      },
      detectorAvailable: this.detectorAvailable,
      handsAvailable: this.handsAvailable,
      calibrating: this.pitchRef.isCalibrating(),
      latency: { ...this.latency.lastValues(), total: totalMs },
    });
  }

  reset(): void {
    this.frameIndex = 0;
    this.selector = new PrimaryFaceSelector();
    this.aggregator = new FaceMetricsAggregator();
    this.pitchRef.reset();
    this.openness.reset();
    this.runner.reset();
    this.objects = [];
    this.handResults = [];
    this.prevContextPitch = null;
    this.lastScoreMetrics = null;
    this.lastLabelAttentive = false;
    this.fallbackCount = 0;
  }

  dispose(): void {
    try {
      this.measurement.dispose();
      this.detector?.dispose();
      this.hands?.dispose();
    } catch {
      /* ignore */
    }
  }
}
