/**
 * Start-up self-test and the FR4 context gate.
 *
 * FR4: "Load and self-test the object detector; use context-aware scoring
 * only when it is loaded." In DACE mode this means:
 *   - object evidence (phone/book boxes) is used ONLY when the detector is
 *     loaded AND its self-test passed (YOLO: zero tensor -> output0 dims
 *     [1,84,8400]; MediaPipe: blank-canvas detectForVideo);
 *   - the legacy "empty object list" upgraded path (computeObjectContext on
 *     [] etc.) is NEVER used: with evidence on, the DACE engine always passes
 *     explicit weights and DACE-mapped context features; metrics.detectedObjects
 *     is only set when the detector is available;
 *   - detector missing -> detectorAvailable = false; hand-only evidence
 *     (writing, desk visibility) still works; a long downward episode without
 *     evidence goes to Unsure (plan section 8).
 */
import type { DetectorStatus } from './detectors/types';
import type { ProviderSelfTest } from './measurement/provider';

export type ComponentReport = {
  name: 'measurement' | 'detector' | 'hands';
  ok: boolean;
  ms: number | null;
  detail?: string;
};

export type SelfTestReport = {
  components: ComponentReport[];
  detectorAvailable: boolean;
  handsAvailable: boolean;
  measurementOk: boolean;
};

export type ContextGate = {
  /** feed detector output into the window */
  useObjects: boolean;
  /** feed hand landmarks into the window */
  useHands: boolean;
  /** weight table for MLEngine.predict on the DACE path */
  weights: 'legacy' | 'upgraded';
  /** DACE context features (contextAwarePitch, objectContext, ...) are mapped and passed explicitly */
  mapContext: boolean;
  /** set metrics.detectedObjects (for display and the contradiction detector) */
  exposeObjects: boolean;
};

export function contextGate(evidence: boolean, detectorAvailable: boolean, handsAvailable: boolean): ContextGate {
  if (!evidence) {
    return { useObjects: false, useHands: false, weights: 'legacy', mapContext: false, exposeObjects: false };
  }
  return {
    useObjects: detectorAvailable,
    useHands: handsAvailable,
    weights: 'upgraded',
    mapContext: true,
    exposeObjects: detectorAvailable,
  };
}

export function buildSelfTestReport(input: {
  measurement: ProviderSelfTest | null;
  detector: DetectorStatus | null;
  hands: ProviderSelfTest | null;
}): SelfTestReport {
  const components: ComponentReport[] = [];
  const measurementOk = input.measurement ? input.measurement.ok : true;
  if (input.measurement) {
    components.push({ name: 'measurement', ok: input.measurement.ok, ms: input.measurement.ms, detail: input.measurement.error });
  }
  const detectorAvailable = !!input.detector && input.detector.loaded && input.detector.selfTestOk === true;
  if (input.detector) {
    components.push({
      name: 'detector',
      ok: detectorAvailable,
      ms: input.detector.selfTestMs,
      detail: input.detector.error ?? (input.detector.checksumVerified === false ? 'checksum mismatch' : undefined),
    });
  }
  const handsAvailable = !!input.hands && input.hands.ok;
  if (input.hands) components.push({ name: 'hands', ok: input.hands.ok, ms: input.hands.ms, detail: input.hands.error });
  return { components, detectorAvailable, handsAvailable, measurementOk };
}
