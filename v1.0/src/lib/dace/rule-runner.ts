/**
 * E/T/U runner: everything after the measurement layer, as one pure,
 * node-runnable state machine. Used by the live DaceEngine and by the offline
 * replay script (scripts/dace/replay-features.ts), so a frame log recorded in
 * the browser re-runs through exactly the same code.
 *
 *   DaceFrame (M output + hands + objects)
 *     -> [evidence off: hands/objects removed]
 *     -> [shuffleObjects: detector outputs replaced, control H]
 *     -> DaceWindow (D_t, per-frame evidence, W and desk windows, episodes)
 *     -> gaze-corrected PERCLOS + longest closure
 *     -> ordered rules + Stage 1 abstention (when rules are on)
 */
import type { DaceParams } from '../scoring/dace/params';
import type { DaceFrame, DaceWindowOutput, FrameEvidence, RuleId, WindowSummary, BehaviorLabel, UnsureReason } from '../scoring/dace/types';
import { DaceWindow } from '../scoring/dace/window';
import { CorrectedPerclos } from '../scoring/dace/perclos';
import { DaceRuleEngine } from '../scoring/dace/rules';
import { ObjectShuffler, DEFAULT_SHUFFLE_SEED } from './object-shuffle';

export type RuleRunnerOptions = {
  evidence: boolean;
  rules: boolean;
  abstain: 'off' | 'rules';
  shuffleObjects: boolean;
  shuffleSeed?: number;
};

/** The output the feature mapping consumes (rule output, or an evidence-only label). */
export type EffectiveOutput = {
  label: BehaviorLabel | null;
  unsure: boolean;
  reason: UnsureReason | null;
  rule: RuleId | 'evidence_only' | 'none';
};

export type RunnerStep = {
  frame: DaceFrame;
  evidence: FrameEvidence;
  summary: WindowSummary;
  perclos: number | null;
  perclosN: number;
  longestClosureMs: number;
  /** rule-layer output (null when dace.rules is off) */
  output: DaceWindowOutput | null;
  effective: EffectiveOutput;
};

/**
 * Preset C (evidence on, rules off): no temporal rule layer, but the
 * evidence still conditions the looking-down penalty (novelty item 7):
 * phone in use -> using_phone; looking down in most of W with writing or
 * book evidence -> attentive_reading; otherwise no label (standard curve).
 */
export function evidenceOnlyOutput(s: WindowSummary): EffectiveOutput {
  if (s.phoneInUse) return { label: 'using_phone', unsure: false, reason: null, rule: 'evidence_only' };
  if (s.lookingDownMost && (s.writing || s.book)) {
    return { label: 'attentive_reading', unsure: false, reason: null, rule: 'evidence_only' };
  }
  return { label: null, unsure: false, reason: null, rule: 'evidence_only' };
}

export class RuleRunner {
  private window: DaceWindow;
  private perclos: CorrectedPerclos;
  private rules: DaceRuleEngine;
  private shuffler: ObjectShuffler;
  private shuffledObjects: DaceFrame['objects'] = [];

  constructor(
    private readonly params: DaceParams,
    private readonly opts: RuleRunnerOptions
  ) {
    this.window = new DaceWindow(params);
    this.perclos = new CorrectedPerclos(params);
    this.rules = new DaceRuleEngine(params, { abstain: opts.abstain });
    this.shuffler = new ObjectShuffler(opts.shuffleSeed ?? DEFAULT_SHUFFLE_SEED);
  }

  /**
   * @param objectsFresh true when `frame.objects` came from a detector run on
   *   this tick (false when carried forward). Only fresh outputs are shuffled.
   */
  push(input: DaceFrame, objectsFresh = true): RunnerStep {
    let frame: DaceFrame = input;
    if (!this.opts.evidence) {
      frame = { ...frame, hands: [], objects: [] };
    } else if (this.opts.shuffleObjects) {
      if (objectsFresh) this.shuffledObjects = this.shuffler.next(frame.objects);
      frame = { ...frame, objects: this.shuffledObjects };
    }

    const evidence = this.window.push(frame);
    const summary = this.window.summary();
    this.perclos.push(frame.t, frame.face ? frame.face.eyeClosed : null);
    const pv = this.perclos.value();
    const perclos = pv.n >= this.params.perclosMinSamples ? pv.perclos : null;
    const longestClosureMs = this.perclos.longestClosureMs(this.window.windowStartT() ?? frame.t);

    let output: DaceWindowOutput | null = null;
    let effective: EffectiveOutput;
    if (this.opts.rules) {
      output = this.rules.evaluate({ summary, perclos, longestClosureMs });
      effective = { label: output.label, unsure: output.unsure, reason: output.reason, rule: output.rule };
    } else if (this.opts.evidence) {
      effective = evidenceOnlyOutput(summary);
    } else {
      effective = { label: null, unsure: false, reason: null, rule: 'none' };
    }
    return { frame, evidence, summary, perclos, perclosN: pv.n, longestClosureMs, output, effective };
  }

  reset(): void {
    this.window.reset();
    this.perclos.reset();
    this.rules.reset();
    this.shuffler.reset(this.opts.shuffleSeed ?? DEFAULT_SHUFFLE_SEED);
    this.shuffledObjects = [];
  }
}
