/**
 * Ordered rule layer T (Annex A.4, Stage 1) with Stage 1 abstention U (A.5).
 *
 * Each window is labelled by the FIRST rule that applies. Order (plan D11):
 *
 *   1  absent      no face in ≥ 80% of W                                  -> absent
 *   U3 tracking    face missing in ≥ 50% of W                              -> Unsure(U3)
 *   2  drowsy      gaze-corrected PERCLOS ≥ p_drowsy, or one continuous
 *                  closure ≥ 2 s                                          -> drowsy
 *   U4 conflict    phone in use + pen-grip motion in the other hand        -> Unsure(U4)
 *   3  phone       phone in use (A.2); label held for 10 s after the last
 *                  firing so repeated short glances accumulate             -> using_phone
 *   4  reading     looking down in most of W and (writing or book)         -> attentive_reading
 *   5  glance      in a downward episode of ≤ τ_g                          -> previous output kept
 *   6  no evidence downward episode > τ_g, no evidence                     -> Unsure(U2 if desk not
 *                                                                             visible, else U1)
 *   7  screen      not looking down                                        -> attentive_screen / distracted
 *
 * With abstention off (dace.abstain = 'off'): U3 and U4 are skipped (the
 * following rules decide) and rule 6 outputs `distracted` (legacy behaviour),
 * so ablations B/C can be run without "Unsure".
 */
import type { DaceParams } from './params';
import type { DaceWindowOutput, RuleId, WindowSummary, BehaviorLabel } from './types';
import { atLeastShare } from './params';
import { downwardUnsureReason } from './abstain';

export type RuleInput = {
  summary: WindowSummary;
  /** gaze-corrected PERCLOS (0-1) over 60 s, or null if unavailable */
  perclos: number | null;
  /** longest continuous closure (ms) among runs ending inside W */
  longestClosureMs: number;
};

export type RuleOptions = { abstain: 'off' | 'rules' };

export type RuleState = {
  prev: DaceWindowOutput | null;
  phoneHoldUntil: number | null;
};

export const initialRuleState = (): RuleState => ({ prev: null, phoneHoldUntil: null });

function label(summary: WindowSummary, l: BehaviorLabel, rule: RuleId): DaceWindowOutput {
  return { t: summary.t, label: l, unsure: false, reason: null, rule, summary };
}

/** Pure rule evaluation: returns the output and the next state. */
export function evaluateRules(
  input: RuleInput,
  state: RuleState,
  params: DaceParams,
  opts: RuleOptions
): { output: DaceWindowOutput; state: RuleState } {
  const s = input.summary;
  const N = params.windowFrames;
  const abstain = opts.abstain === 'rules';
  let phoneHoldUntil = state.phoneHoldUntil;

  const out = ((): DaceWindowOutput => {
    // 1. absent
    if (atLeastShare(s.noFaceCount, N, params.absentShare)) {
      phoneHoldUntil = null;
      return label(s, 'absent', 'absent');
    }
    // U3. poor tracking
    if (abstain && atLeastShare(s.noFaceCount, N, params.u3MissingShare)) {
      return { t: s.t, label: null, unsure: true, reason: 'U3', rule: 'U3', summary: s };
    }
    // 2. drowsy
    if ((input.perclos !== null && input.perclos >= params.pDrowsy) || input.longestClosureMs >= params.closureMs) {
      return label(s, 'drowsy', 'drowsy');
    }
    // U4. conflicting evidence
    if (abstain && s.conflict) {
      return { t: s.t, label: null, unsure: true, reason: 'U4', rule: 'U4', summary: s };
    }
    // 3. phone in use (+10 s hold)
    if (s.phoneInUse) {
      phoneHoldUntil = s.t + params.phoneHoldMs;
      return label(s, 'using_phone', 'phone');
    }
    if (phoneHoldUntil !== null && s.t < phoneHoldUntil) {
      return label(s, 'using_phone', 'phone_hold');
    }
    // 4. reading / writing with evidence
    if (s.lookingDownMost && (s.writing || s.book)) {
      return label(s, 'attentive_reading', 'reading');
    }
    // 5. brief glance: keep the previous output
    if (s.inDownEpisode && s.downEpisodeMs <= params.tauGlanceMs && state.prev) {
      const p = state.prev;
      return {
        t: s.t,
        label: p.label,
        unsure: p.unsure,
        reason: p.reason,
        rule: 'glance',
        keptFrom: p.rule === 'glance' ? p.keptFrom : p.rule,
        summary: s,
      };
    }
    // 6. long downward episode without evidence
    if (s.inDownEpisode && s.downEpisodeMs > params.tauGlanceMs) {
      if (abstain) {
        const reason = downwardUnsureReason(s.deskVisible);
        return { t: s.t, label: null, unsure: true, reason, rule: reason, summary: s };
      }
      return label(s, 'distracted', 'down_no_evidence');
    }
    // 7. not looking down: screen-attention scoring
    return label(s, s.screenAttentive ? 'attentive_screen' : 'distracted', 'screen');
  })();

  return { output: out, state: { prev: out, phoneHoldUntil } };
}

/** Stateful wrapper around evaluateRules. */
export class DaceRuleEngine {
  private state: RuleState = initialRuleState();

  constructor(
    private readonly params: DaceParams,
    private readonly opts: RuleOptions
  ) {}

  evaluate(input: RuleInput): DaceWindowOutput {
    const r = evaluateRules(input, this.state, this.params, this.opts);
    this.state = r.state;
    return r.output;
  }

  getState(): Readonly<RuleState> {
    return this.state;
  }

  reset(): void {
    this.state = initialRuleState();
  }
}
