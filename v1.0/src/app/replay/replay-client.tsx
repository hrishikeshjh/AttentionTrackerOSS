'use client';

/**
 * Offline replay of one local recording through several ablation presets
 * (Annex D.6, Stage 1: A, A', A'', B, C, F, H, I).
 *
 *  - The file is opened with an object URL and stepped deterministically
 *    every 200 ms of media time (FileSource); every preset sees exactly the
 *    same frames with t = k * 200 ms.
 *  - DACE presets run the DaceEngine with the preset as the explicit config
 *    layer (env flags ignored); A / A' run the legacy pipeline exactly as the
 *    dashboard wires it (A: no object model, A': YOLO model loaded).
 *  - MLEngine.predict is called every 5th frame (the dashboard's 1 Hz).
 *  - Output: one JSONL eval log per preset (no images, random session id),
 *    downloaded manually. Feed them to scripts/dace/metrics.ts.
 */
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { DACE_PRESETS, type DacePresetId } from '@/lib/dace/presets';
import { configHash, resolveDaceConfig, DEFAULT_DACE_CONFIG } from '@/lib/dace/config';
import { EvalLogger } from '@/lib/dace/eval-log';
import { FileSource } from '@/lib/dace/replay/frame-source';
import { MLEngine } from '@/lib/ml-engine';
import type { ExtendedFaceMetrics } from '@/lib/types';

const PRESET_IDS = Object.keys(DACE_PRESETS) as DacePresetId[];
const PREDICT_EVERY = 5; // 200 ms frames -> 1 Hz predictions, as on the dashboard

type Output = { preset: DacePresetId; records: number; url: string; filename: string; error?: string };

const safeId = (id: string) => id.replace(/'/g, 'p');

async function runLegacy(preset: DacePresetId, file: File, level: 'windows' | 'frames', onProgress: (f: number) => void): Promise<EvalLogger> {
  const [{ LegacyFaceApiProvider }, { FaceMetricsAggregator, HandObjectProximityDetector }, { PrimaryFaceSelector }, { TemporalBehaviorClassifier }, { HandTracker }, { ObjectDetector }] =
    await Promise.all([
      import('@/lib/dace/measurement/legacy-faceapi'),
      import('@/lib/face-geometry'),
      import('@/lib/scoring/face-selection'),
      import('@/lib/temporal-classifier'),
      import('@/lib/hand-tracker'),
      import('@/lib/object-detector'),
    ]);
  const cfg = { ...DEFAULT_DACE_CONFIG, replay: { ...DEFAULT_DACE_CONFIG.replay } };
  const logger = new EvalLogger({ level, configHash: configHash(cfg) });
  const provider = new LegacyFaceApiProvider();
  await provider.loadModels();
  const od = new ObjectDetector();
  const objectModel = DACE_PRESETS[preset].objectModel === 'present';
  if (objectModel) {
    try {
      await od.load('/models/yolov8n-objects.onnx');
    } catch (e) {
      logger.log('event', 0, { event: 'object_model_missing', error: e instanceof Error ? e.message : String(e) });
    }
  }
  const ht = new HandTracker();
  await ht.initialize();
  const aggregator = new FaceMetricsAggregator();
  const proximity = new HandObjectProximityDetector();
  const selector = new PrimaryFaceSelector();
  const temporal = new TemporalBehaviorClassifier();
  const ml = new MLEngine();
  logger.log('session', 0, { preset, source: 'file', config: cfg, objectModelLoaded: od.isLoaded(), pipeline: 'legacy' });

  const src = await FileSource.open(file, 200);
  try {
    let k = 0;
    for (let f = await src.next(); f; f = await src.next(), k++) {
      const video = f.image as HTMLVideoElement;
      const dets = await provider.detectResized(video, { width: f.width, height: f.height });
      const primary = selector.select(dets.map((d) => ({ box: d.detection.box, detection: d })), f.width, f.height);
      if (!primary) {
        logger.log('window', f.t, { i: k, face: false, label: null, unsure: false, reason: null, rule: 'legacy_no_face' });
        onProgress(k / src.frameCount);
        continue;
      }
      const [objectResult, handResult] = await Promise.allSettled([od.detect(video), ht.detect(video)]);
      const detectedObjects = objectResult.status === 'fulfilled' ? objectResult.value : [];
      const handResults = handResult.status === 'fulfilled' ? handResult.value : [];
      const handObjectProximity = proximity.compute(handResults, detectedObjects);
      const base = aggregator.calculateExtendedMetrics(primary.detection.landmarks.positions, f.width, f.height, f.t);
      const m: ExtendedFaceMetrics = { ...base, detectedObjects, handObjectProximity };
      m.behaviorResult = temporal.update({ metrics: m, detectedObjects, handProximity: handObjectProximity });
      logger.log('window', f.t, {
        i: k,
        face: true,
        label: m.behaviorResult.label,
        unsure: false,
        reason: null,
        rule: `legacy_${m.behaviorResult.source}`,
        legacyPitch: m.pitch,
        objects: detectedObjects.map((o) => o.class),
        handObjectProximity,
      });
      if (k % PREDICT_EVERY === 0) {
        const p = ml.predict(m);
        logger.log('prediction', f.t, { i: k, score: p.rawScore, state: p.attentionState, label: m.behaviorResult.label, unsure: false });
      }
      onProgress(k / src.frameCount);
    }
  } finally {
    src.close();
  }
  return logger;
}

async function runDace(preset: DacePresetId, file: File, level: 'windows' | 'frames', onProgress: (f: number) => void): Promise<EvalLogger> {
  const { createDaceEngine } = await import('@/lib/dace/engine-factory');
  const resolved = resolveDaceConfig({ env: {}, explicit: { ...DACE_PRESETS[preset].override, log: level } });
  const { engine, logger } = await createDaceEngine(resolved, { logLevel: level });
  const log = logger as EvalLogger;
  log.log('session', 0, { preset, source: 'file', pipeline: 'dace' });
  const ml = new MLEngine();
  const src = await FileSource.open(file, engine.params.frameIntervalMs);
  try {
    let k = 0;
    for (let f = await src.next(); f; f = await src.next(), k++) {
      const r = await engine.processFrame(f);
      if (k % PREDICT_EVERY === 0 && r.scoreMetrics) {
        const p = r.predict ? ml.predict(r.scoreMetrics, { dace: r.predict }) : ml.predict(r.scoreMetrics);
        log.log('prediction', f.t, {
          i: k,
          score: p.rawScore,
          state: p.attentionState,
          label: p.daceLabel ?? null,
          unsure: p.unsure ?? false,
          reason: p.unsureReason ?? null,
          held: p.scoreHeld ?? false,
          fallback: r.fallback,
        });
      }
      onProgress(k / src.frameCount);
    }
    log.log('latency', src.frameCount * engine.params.frameIntervalMs, { summary: engine.latency.summary(), fallbacks: engine.getFallbackCount() });
  } finally {
    src.close();
    engine.dispose();
  }
  return log;
}

export default function ReplayClient() {
  const [file, setFile] = useState<File | null>(null);
  const [selected, setSelected] = useState<Set<DacePresetId>>(new Set(['A', "A'", 'F']));
  const [level, setLevel] = useState<'windows' | 'frames'>('windows');
  const [running, setRunning] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [outputs, setOutputs] = useState<Output[]>([]);

  const toggle = (id: DacePresetId) =>
    setSelected((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });

  const run = async () => {
    if (!file) return;
    outputs.forEach((o) => o.url && URL.revokeObjectURL(o.url));
    setOutputs([]);
    const base = file.name.replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9_-]+/g, '_');
    for (const id of PRESET_IDS.filter((p) => selected.has(p))) {
      setRunning(id);
      setProgress(0);
      try {
        const logger = DACE_PRESETS[id].override.enabled
          ? await runDace(id, file, level, setProgress)
          : await runLegacy(id, file, level, setProgress);
        const url = URL.createObjectURL(new Blob([logger.toJSONL()], { type: 'application/x-ndjson' }));
        setOutputs((o) => [...o, { preset: id, records: logger.size(), url, filename: `${base}__${safeId(id)}.jsonl` }]);
      } catch (e) {
        setOutputs((o) => [...o, { preset: id, records: 0, url: '', filename: '', error: e instanceof Error ? e.message : String(e) }]);
      }
    }
    setRunning(null);
  };

  return (
    <main className="mx-auto max-w-3xl p-6">
      <Card>
        <CardHeader>
          <CardTitle>DACE offline replay (research)</CardTitle>
          <CardDescription>
            Runs a local recording through the selected ablation presets. The file never leaves this browser. Output: one
            JSONL evaluation log per preset (no images). Preset I is F evaluated on desk-out-of-frame segments; select
            segments in scripts/dace/metrics.ts.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1">
            <Label htmlFor="video">Recording (local video file)</Label>
            <input
              id="video"
              type="file"
              accept="video/*"
              disabled={running !== null}
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="block text-sm"
            />
          </div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {PRESET_IDS.map((id) => (
              <label key={id} className="flex items-start gap-2 text-sm" title={DACE_PRESETS[id].title}>
                <Checkbox checked={selected.has(id)} onCheckedChange={() => toggle(id)} disabled={running !== null} />
                <span>
                  <span className="font-mono font-semibold">{id}</span>
                  <span className="block text-xs text-muted-foreground">{DACE_PRESETS[id].question}</span>
                </span>
              </label>
            ))}
          </div>
          <div className="flex items-center gap-4 text-sm">
            <span>Log level:</span>
            {(['windows', 'frames'] as const).map((l) => (
              <label key={l} className="flex items-center gap-1">
                <input type="radio" name="level" checked={level === l} onChange={() => setLevel(l)} disabled={running !== null} />
                {l}
              </label>
            ))}
            <span className="text-xs text-muted-foreground">(frames: needed for npm run dace:replay)</span>
          </div>
          <Button onClick={run} disabled={!file || selected.size === 0 || running !== null}>
            {running ? `Running ${running}…` : 'Run selected presets'}
          </Button>
          {running && <Progress value={Math.round(progress * 100)} className="h-2" />}
          {outputs.length > 0 && (
            <ul className="space-y-1 text-sm">
              {outputs.map((o) => (
                <li key={o.preset} className="flex items-center justify-between">
                  <span className="font-mono">{o.preset}</span>
                  {o.error ? (
                    <span className="text-red-500">{o.error}</span>
                  ) : (
                    <a className="underline" href={o.url} download={o.filename}>
                      {o.filename} ({o.records} records)
                    </a>
                  )}
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </main>
  );
}
