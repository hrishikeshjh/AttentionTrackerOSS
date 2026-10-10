/**
 * Browser factory for a DaceEngine from a resolved config: creates the
 * selected measurement provider (dynamic import: only that library loads),
 * the detector and hand landmarker when evidence is on, the eval logger, and
 * runs the start-up self-test (FR4).
 *
 * Measurement-provider init failure THROWS (the dashboard then falls back to
 * the legacy pipeline with a toast). Detector or hand-landmarker failures do
 * not throw: they leave the component unavailable and the FR4 gate routes
 * around it.
 */
import type { ResolvedDaceConfig } from './config';
import type { DaceParams } from '../scoring/dace/params';
import { makeParams } from '../scoring/dace/params';
import { DaceEngine } from './engine';
import { EvalLogger } from './eval-log';
import { LatencyProbe } from './latency';
import { createMeasurementProvider } from './measurement/provider';
import type { DaceObjectDetector } from './detectors/types';
import type { DaceHandSource } from './hands/hand-landmarker';

export type CreateEngineOptions = {
  params?: Partial<DaceParams>;
  /** expected SHA-256 of the YOLO model (optional; NEXT_PUBLIC_DACE_YOLO_SHA256) */
  yoloSha256?: string;
  sessionId?: string;
  /** force a log level (the replay page always logs) */
  logLevel?: 'off' | 'windows' | 'frames';
};

export async function createDaceEngine(resolved: ResolvedDaceConfig, opts: CreateEngineOptions = {}): Promise<{
  engine: DaceEngine;
  logger: EvalLogger | null;
  latency: LatencyProbe;
  warnings: string[];
}> {
  const { config, hash } = resolved;
  const params = makeParams(opts.params ?? {});
  const warnings: string[] = [...resolved.warnings];

  const measurement = await createMeasurementProvider(config.measurement, { pitchSign: params.pitchSign });
  await measurement.init();
  if (!measurement.isReady()) throw new Error('measurement provider failed to initialise');
  return finish(measurement);

  async function finish(m: typeof measurement) {
    let detector: DaceObjectDetector | null = null;
    let hands: DaceHandSource | null = null;
    if (config.evidence) {
      try {
        if (config.detector === 'mediapipe') {
          const { MediaPipeObjectDetector } = await import('./detectors/mediapipe-object-detector');
          detector = await MediaPipeObjectDetector.create({ scoreThreshold: Math.min(0.2, params.cMin) });
        } else {
          const { YoloDaceDetector } = await import('./detectors/yolo-detector');
          detector = await YoloDaceDetector.create({ sha256: opts.yoloSha256 });
        }
        if (!detector.isReady()) warnings.push(`object detector (${config.detector}) unavailable: ${detector.status().error ?? 'self-test failed'}`);
      } catch (e) {
        warnings.push(`object detector failed: ${e instanceof Error ? e.message : String(e)}`);
        detector = null;
      }
      try {
        const { MediaPipeHandLandmarker } = await import('./hands/hand-landmarker');
        const h = new MediaPipeHandLandmarker();
        await h.init();
        hands = h;
      } catch (e) {
        warnings.push(`hand landmarker failed: ${e instanceof Error ? e.message : String(e)}`);
        hands = null;
      }
    }
    const level = opts.logLevel ?? config.log;
    const logger = level === 'off' ? null : new EvalLogger({ level, configHash: hash, sessionId: opts.sessionId });
    const latency = new LatencyProbe();
    const engine = new DaceEngine({ config, params, measurement: m, detector, hands, logger, latency });
    await engine.selfTest();
    logger?.log('session', 0, {
      config,
      params,
      measurement: m.label,
      detector: detector?.status() ?? null,
      selfTest: engine.selfTestReport,
      warnings,
    });
    return { engine, logger, latency, warnings };
  }
}
