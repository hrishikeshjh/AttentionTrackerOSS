'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, Play, StopCircle, Video, VideoOff } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import type { SessionDataPoint, ExtendedFaceMetrics, MLPrediction } from '@/lib/types';
import { FaceMetricsAggregator, HandObjectProximityDetector } from '@/lib/face-geometry';
import { PrimaryFaceSelector } from '@/lib/scoring/face-selection';
import { getMLEngine, resetMLEngine, debugPrediction } from '@/lib/ml-engine';
import type { ObjectDetector } from '@/lib/object-detector';
import type { HandTracker } from '@/lib/hand-tracker';
import type { TemporalBehaviorClassifier } from '@/lib/temporal-classifier';
import { LegacyFaceApiProvider } from '@/lib/dace/measurement/legacy-faceapi';
import { resolveDaceConfig, type ResolvedDaceConfig } from '@/lib/dace/config';
import type { DaceEngine } from '@/lib/dace/engine';
import type { EvalLogger } from '@/lib/dace/eval-log';
import type { LatencySummary } from '@/lib/dace/latency';
import { FixedTickScheduler } from '@/lib/dace/scheduler';

/** DACE status for the dashboard (research badge, measurement label). */
export type DaceStatus = {
  /** DACE engine is running (false: legacy pipeline, including after a fallback) */
  active: boolean;
  requested: boolean;
  configHash: string;
  version: string;
  measurementLabel: string;
  detectorAvailable: boolean;
  fellBack: boolean;
  warnings: string[];
};

interface WebcamPanelProps {
  sessionActive: boolean;
  onStartSession: () => void;
  onStopSession: () => void;
  onNewDataPoint: (data: SessionDataPoint) => void;
  onNewFaceMetrics: (metrics: ExtendedFaceMetrics | null) => void;
  onNewPrediction: (prediction: MLPrediction | null) => void;
  onMultitask: () => void;
  /** DACE only (optional; never called with DACE off) */
  onDaceStatus?: (status: DaceStatus) => void;
  onNewLatency?: (summary: LatencySummary) => void;
  onEvalLog?: (logger: EvalLogger | null) => void;
}

type PipelineMode = 'pending' | 'legacy' | 'dace';

function readResolvedConfig(): ResolvedDaceConfig {
  let storage: Storage | null = null;
  try {
    storage = window.localStorage;
  } catch {
    storage = null;
  }
  return resolveDaceConfig({ search: window.location.search, storage });
}

export default function WebcamPanel({
  sessionActive,
  onStartSession,
  onStopSession,
  onNewDataPoint,
  onNewFaceMetrics,
  onNewPrediction,
  onMultitask,
  onDaceStatus,
  onNewLatency,
  onEvalLog,
}: WebcamPanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { toast } = useToast();
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [isWebcamOn, setIsWebcamOn] = useState(false);

  const animationFrameRef = useRef<number | null>(null);
  const metricsAggregatorRef = useRef<FaceMetricsAggregator>(new FaceMetricsAggregator());
  const lastDataPointTime = useRef<number>(0);
  const DATA_POINT_INTERVAL = 1000; // Record data point every 1 second

  // Context-aware inference modules (lazily loaded after face-api models are ready)
  const objectDetectorRef = useRef<ObjectDetector | null>(null);
  const handTrackerRef = useRef<HandTracker | null>(null);
  const proximityDetectorRef = useRef<HandObjectProximityDetector>(new HandObjectProximityDetector());
  const primaryFaceSelectorRef = useRef<PrimaryFaceSelector>(new PrimaryFaceSelector());
  const temporalClassifierRef = useRef<TemporalBehaviorClassifier | null>(null);

  // Legacy measurement (face-api.js, loaded with a dynamic import)
  const legacyProviderRef = useRef<LegacyFaceApiProvider>(new LegacyFaceApiProvider());

  // DACE (only used when dace.enabled resolves to true)
  const [mode, setMode] = useState<PipelineMode>('pending');
  const daceConfigRef = useRef<ResolvedDaceConfig | null>(null);
  const daceEngineRef = useRef<DaceEngine | null>(null);
  const daceLoggerRef = useRef<EvalLogger | null>(null);
  const schedulerRef = useRef<FixedTickScheduler | null>(null);
  const [calibrating, setCalibrating] = useState(false);
  const calibratingRef = useRef(false);
  const sessionActiveRef = useRef(sessionActive);
  const callbacksRef = useRef({ onNewDataPoint, onNewFaceMetrics, onNewPrediction, onNewLatency });
  useEffect(() => {
    sessionActiveRef.current = sessionActive;
  }, [sessionActive]);
  useEffect(() => {
    callbacksRef.current = { onNewDataPoint, onNewFaceMetrics, onNewPrediction, onNewLatency };
  }, [onNewDataPoint, onNewFaceMetrics, onNewPrediction, onNewLatency]);

  // Resolve the DACE config once (frozen for the page session). DACE off -> legacy, unchanged.
  useEffect(() => {
    try {
      const resolved = readResolvedConfig();
      daceConfigRef.current = resolved;
      if (resolved.warnings.length) console.info('[CognEn DACE] config:', resolved.warnings);
      setMode(resolved.config.enabled ? 'dace' : 'legacy');
    } catch (err) {
      console.warn('[CognEn DACE] config resolution failed — legacy pipeline:', err);
      setMode('legacy');
    }
  }, []);

  // Load face-api.js models (legacy pipeline)
  useEffect(() => {
    if (mode !== 'legacy') return;
    const loadModels = async () => {
      try {
        await legacyProviderRef.current.loadModels();
        setModelsLoaded(true);
        toast({
          title: 'Model Loaded',
          description: 'The facial analysis model is ready.',
        });
      } catch (error) {
        console.error('Failed to load models:', error);
        toast({
          variant: 'destructive',
          title: 'Model Error',
          description: 'Failed to load facial detection models. Please refresh the page.',
        });
      }
    };
    loadModels();
  }, [toast, mode]);

  // DACE: create the engine; MediaPipe/provider init failure -> automatic legacy fallback with a toast.
  useEffect(() => {
    if (mode !== 'dace') return;
    const resolved = daceConfigRef.current;
    if (!resolved) return;
    let cancelled = false;
    (async () => {
      try {
        const { createDaceEngine } = await import('@/lib/dace/engine-factory');
        const { engine, logger, warnings } = await createDaceEngine(resolved, {
          yoloSha256: process.env.NEXT_PUBLIC_DACE_YOLO_SHA256 || undefined,
        });
        if (cancelled) {
          engine.dispose();
          return;
        }
        daceEngineRef.current = engine;
        daceLoggerRef.current = logger;
        onEvalLog?.(logger);
        onDaceStatus?.({
          active: true,
          requested: true,
          configHash: resolved.hash,
          version: resolved.version,
          measurementLabel: engine.measurementLabel(),
          detectorAvailable: engine.isDetectorAvailable(),
          fellBack: false,
          warnings,
        });
        if (warnings.length) console.info('[CognEn DACE] warnings:', warnings);
        setModelsLoaded(true);
        toast({ title: 'Model Loaded', description: `DACE research mode: ${engine.measurementLabel()}.` });
      } catch (err) {
        if (cancelled) return;
        console.warn('[CognEn DACE] engine init failed — falling back to the legacy pipeline:', err);
        onDaceStatus?.({
          active: false,
          requested: true,
          configHash: resolved.hash,
          version: resolved.version,
          measurementLabel: 'face-api.js 68-pt (legacy fallback)',
          detectorAvailable: false,
          fellBack: true,
          warnings: [err instanceof Error ? err.message : String(err)],
        });
        toast({
          variant: 'destructive',
          title: 'DACE unavailable',
          description: 'MediaPipe could not start; using the standard analysis instead.',
        });
        setMode('legacy');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [mode, toast, onDaceStatus, onEvalLog]);

  // Dispose the DACE engine on unmount.
  useEffect(() => {
    return () => {
      schedulerRef.current?.stop();
      daceEngineRef.current?.dispose();
      daceEngineRef.current = null;
    };
  }, []);

  // Lazy-load context modules once face-api models are ready (legacy pipeline only)
  useEffect(() => {
    if (!modelsLoaded || mode !== 'legacy') return;
    let cancelled = false;

    const loadContextModules = async () => {
      try {
        const [{ ObjectDetector: OD }, { HandTracker: HT }, { TemporalBehaviorClassifier: TBC }] =
          await Promise.all([
            import('@/lib/object-detector'),
            import('@/lib/hand-tracker'),
            import('@/lib/temporal-classifier'),
          ]);
        if (cancelled) return;

        const od = new OD();
        const ht = new HT();
        const tbc = new TBC();

        // Load models in parallel; failures are non-fatal
        await Promise.allSettled([
          od.load('/models/yolov8n-objects.onnx'),
          ht.initialize(),
        ]);

        if (!cancelled) {
          objectDetectorRef.current = od;
          handTrackerRef.current = ht;
          temporalClassifierRef.current = tbc;
        }
      } catch (err) {
        console.warn('[CognEn] Context modules failed to load — running in legacy mode:', err);
      }
    };

    loadContextModules();
    return () => { cancelled = true; };
  }, [modelsLoaded, mode]);

  // Handle tab visibility for multitasking detection
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden && sessionActive) {
        onMultitask();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [sessionActive, onMultitask]);

  const stopAnalysis = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    schedulerRef.current?.stop();
    schedulerRef.current = null;
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }
  }, []);

  const stopWebcam = useCallback(() => {
    stopAnalysis();
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsWebcamOn(false);
  }, [stopAnalysis]);

  const startWebcam = useCallback(async () => {
    if (!modelsLoaded) {
      toast({
        title: 'Models not ready',
        description: 'Please wait for the models to finish loading.',
        variant: 'default',
      });
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          setIsWebcamOn(true);
        };
      }
    } catch (err) {
      console.error('Error accessing webcam:', err);
      toast({
        variant: 'destructive',
        title: 'Webcam Error',
        description: 'Could not access webcam. Please check permissions.',
      });
    }
  }, [toast, modelsLoaded]);

  // ── Legacy pipeline (DACE off): unchanged except that face-api calls go through the legacy provider ──
  const runAnalysis = useCallback(async () => {
    if (!modelsLoaded || !isWebcamOn || !videoRef.current || videoRef.current.readyState < 3) {
      animationFrameRef.current = requestAnimationFrame(runAnalysis);
      return;
    }

    if (!canvasRef.current || !videoRef.current) return;

    const video = videoRef.current;
    canvasRef.current.width = video.videoWidth;
    canvasRef.current.height = video.videoHeight;
    const displaySize = { width: video.videoWidth, height: video.videoHeight };

    try {
      const resizedDetections = await legacyProviderRef.current.detectResized(video, displaySize);

      const ctx = canvasRef.current.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

      // Lock onto a single primary face (largest + most centered, with
      // identity-continuity hysteresis) instead of trusting detector order —
      // detectAllFaces' array order is not a stable selection policy and can
      // flip between candidates frame-to-frame when a second person or a
      // photo/poster face enters frame.
      const primary = primaryFaceSelectorRef.current.select(
        resizedDetections.map((d) => ({ box: d.detection.box, detection: d })),
        displaySize.width,
        displaySize.height
      );

      if (primary) {
        const landmarks = primary.detection.landmarks.positions;
        legacyProviderRef.current.drawDetection(canvasRef.current, primary.detection);

        const timestamp = Date.now();

        // Run context detection in parallel with face metrics (non-blocking)
        const [objectResult, handResult] = await Promise.allSettled([
          objectDetectorRef.current?.detect(video) ?? Promise.resolve([]),
          handTrackerRef.current?.detect(video) ?? Promise.resolve([]),
        ]);

        const detectedObjects = objectResult.status === 'fulfilled' ? objectResult.value : [];
        const handResults    = handResult.status === 'fulfilled'   ? handResult.value  : [];

        // Compute hand-object proximity
        const handObjectProximity = proximityDetectorRef.current.compute(handResults, detectedObjects);

        // Build base face metrics
        const baseMetrics = metricsAggregatorRef.current.calculateExtendedMetrics(
          landmarks,
          video.videoWidth,
          video.videoHeight,
          timestamp
        );

        // Build full ExtendedFaceMetrics with context fields
        const extendedMetrics: ExtendedFaceMetrics = {
          ...baseMetrics,
          detectedObjects,
          handObjectProximity,
        };

        // Update temporal classifier once with complete metrics
        if (temporalClassifierRef.current) {
          extendedMetrics.behaviorResult = temporalClassifierRef.current.update({
            metrics: extendedMetrics,
            detectedObjects,
            handProximity: handObjectProximity,
          });
        }

        onNewFaceMetrics(extendedMetrics);

        // Only record data points at intervals when session is active
        if (sessionActive && timestamp - lastDataPointTime.current >= DATA_POINT_INTERVAL) {
          lastDataPointTime.current = timestamp;

          // Get ML prediction
          const mlEngine = getMLEngine();
          const prediction = mlEngine.predict(extendedMetrics);

          if (process.env.NODE_ENV === 'development') {
            debugPrediction(extendedMetrics, prediction);
          }

          onNewPrediction(prediction);

          onNewDataPoint({
            timestamp,
            attentionScore: prediction.rawScore,
            attentionState: prediction.attentionState,
          });
        }
      } else {
        onNewFaceMetrics(null);
        onNewPrediction(null);
      }
    } catch (e) {
      console.error("AI analysis failed", e);
    }
    if (isWebcamOn) {
      animationFrameRef.current = requestAnimationFrame(runAnalysis);
    }
  }, [modelsLoaded, isWebcamOn, sessionActive, onNewDataPoint, onNewFaceMetrics, onNewPrediction]);

  // ── DACE pipeline: fixed 200 ms ticks, no-face ticks fed to the window ──
  const runDaceTick = useCallback(async () => {
    const engine = daceEngineRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!engine || !video || !canvas || video.readyState < 2 || !video.videoWidth) return;
    const cb = callbacksRef.current;
    const logger = daceLoggerRef.current;

    if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
    const timestamp = Date.now();

    const r = await engine.processFrame({ image: video, width: video.videoWidth, height: video.videoHeight, t: timestamp });

    const ctx = canvas.getContext('2d');
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
    if (r.face) engine.drawFace(canvas, r.face);

    const isCal = r.calibrating && r.face !== null;
    if (isCal !== calibratingRef.current) {
      calibratingRef.current = isCal;
      setCalibrating(isCal);
    }
    if (r.reset === 'camera_move') {
      toast({ title: 'Camera moved', description: 'Re-calibrating: please look at the screen for 5 seconds.' });
    }
    if (r.i % 5 === 0) cb.onNewLatency?.(engine.latency.summary());

    if (!r.metrics && !r.scoreMetrics) {
      cb.onNewFaceMetrics(null);
      cb.onNewPrediction(null);
      return;
    }
    cb.onNewFaceMetrics(r.metrics);

    if (sessionActiveRef.current && r.scoreMetrics && timestamp - lastDataPointTime.current >= DATA_POINT_INTERVAL) {
      lastDataPointTime.current = timestamp;
      const mlEngine = getMLEngine();
      // r.predict === null: fallback frame -> unchanged legacy scoring path
      const prediction = r.predict ? mlEngine.predict(r.scoreMetrics, { dace: r.predict }) : mlEngine.predict(r.scoreMetrics);
      cb.onNewPrediction(prediction);
      logger?.log('prediction', timestamp, {
        i: r.i,
        score: prediction.rawScore,
        state: prediction.attentionState,
        unsure: prediction.unsure ?? false,
        reason: prediction.unsureReason ?? null,
        held: prediction.scoreHeld ?? false,
        label: prediction.daceLabel ?? null,
        fallback: r.fallback,
      });
      // Unsure on the very first tick: no score yet -> show "-" and record no point.
      if (prediction.unsure && !prediction.scoreHeld) return;
      cb.onNewDataPoint({
        timestamp,
        attentionScore: prediction.rawScore,
        attentionState: prediction.attentionState,
        unsure: prediction.unsure ?? false,
        ...(prediction.unsure && prediction.unsureReason ? { unsureReason: prediction.unsureReason } : {}),
        ...(prediction.daceLabel ? { behaviorLabel: prediction.daceLabel } : {}),
      });
    }
  }, [toast]);

  useEffect(() => {
    if (isWebcamOn && modelsLoaded && mode === 'legacy') {
      animationFrameRef.current = requestAnimationFrame(runAnalysis);
    } else if (isWebcamOn && modelsLoaded && mode === 'dace' && daceEngineRef.current) {
      const s = new FixedTickScheduler(runDaceTick, {
        intervalMs: daceEngineRef.current.params.frameIntervalMs,
        onDropped: (d) => {
          if (d.dropped % 25 === 1) console.debug(`[CognEn DACE] dropped tick ${d.i} (total ${d.dropped})`);
          daceLoggerRef.current?.log('event', Date.now(), { event: 'dropped_tick', i: d.i, dropped: d.dropped });
        },
        onError: (e) => console.error('[CognEn DACE] tick failed', e),
      });
      schedulerRef.current = s;
      s.start();
    } else {
      stopAnalysis();
    }
    return () => {
      stopAnalysis();
    };
  }, [isWebcamOn, modelsLoaded, mode, runAnalysis, runDaceTick, stopAnalysis]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, [stopWebcam]);

  const handleSessionToggle = () => {
    if (sessionActive) {
      onStopSession();
    } else {
      if (!isWebcamOn) {
        toast({ title: 'Webcam is off', description: 'Please turn on the webcam to start a session.' });
        return;
      }
      // Reset ML engine and metrics aggregator for new session
      resetMLEngine();
      metricsAggregatorRef.current = new FaceMetricsAggregator();
      temporalClassifierRef.current?.reset();
      lastDataPointTime.current = 0;
      if (mode === 'dace' && daceEngineRef.current) {
        // new session: re-estimate the pitch reference (5 s calibration) and start a fresh log
        daceEngineRef.current.reset();
        const logger = daceLoggerRef.current;
        const resolved = daceConfigRef.current;
        if (logger && resolved) {
          logger.clear();
          logger.log('session', 0, {
            config: resolved.config,
            params: daceEngineRef.current.params,
            measurement: daceEngineRef.current.measurementLabel(),
            selfTest: daceEngineRef.current.selfTestReport,
            source: 'webcam',
          });
        }
      }
      onStartSession();
    }
  }

  const handleWebcamToggle = async () => {
    if (isWebcamOn) {
      if (sessionActive) {
        onStopSession();
      }
      stopWebcam();
    } else {
      await startWebcam();
    }
  }

  const isLoading = !modelsLoaded;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Webcam Feed</CardTitle>
        <div className="flex items-center gap-2">
          <Button
            onClick={handleWebcamToggle}
            size="icon"
            variant="outline"
            disabled={isLoading}
            aria-label={isWebcamOn ? 'Turn off webcam' : 'Turn on webcam'}
          >
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : isWebcamOn ? <VideoOff className="h-4 w-4" /> : <Video className="h-4 w-4" />}
          </Button>
          <Button onClick={handleSessionToggle} disabled={!isWebcamOn || isLoading} className="w-32">
            {sessionActive ? <><StopCircle className="mr-2 h-4 w-4" /> Stop</> : <><Play className="mr-2 h-4 w-4" /> Start</>
            }
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="relative aspect-video w-full overflow-hidden rounded-md bg-muted">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="h-full w-full object-cover"
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 h-full w-full object-cover"
          />
          {isWebcamOn && mode === 'dace' && calibrating && (
            <div className="absolute inset-x-0 top-0 flex justify-center p-3">
              <div className="rounded-md bg-black/70 px-4 py-2 text-center text-sm text-white">
                <div className="font-semibold">Look at the screen</div>
                <div className="text-xs text-white/80">Calibrating head position (about 5 seconds)</div>
              </div>
            </div>
          )}
          {!isWebcamOn && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/50">
              {isLoading ? <Loader2 className="h-16 w-16 animate-spin text-muted-foreground" /> : <VideoOff className="h-16 w-16 text-muted-foreground" />}
              <p className="mt-2 text-muted-foreground">{isLoading ? 'Loading models...' : 'Webcam is off'}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
