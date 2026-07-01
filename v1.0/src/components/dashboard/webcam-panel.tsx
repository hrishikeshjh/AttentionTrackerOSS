'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, Play, StopCircle, Video, VideoOff } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import type { SessionDataPoint, ExtendedFaceMetrics, MLPrediction } from '@/lib/types';
import * as faceapi from 'face-api.js';
import { FaceMetricsAggregator, HandObjectProximityDetector } from '@/lib/face-geometry';
import { PrimaryFaceSelector } from '@/lib/scoring/face-selection';
import { getMLEngine, resetMLEngine, debugPrediction } from '@/lib/ml-engine';
import type { ObjectDetector } from '@/lib/object-detector';
import type { HandTracker } from '@/lib/hand-tracker';
import type { TemporalBehaviorClassifier } from '@/lib/temporal-classifier';

interface WebcamPanelProps {
  sessionActive: boolean;
  onStartSession: () => void;
  onStopSession: () => void;
  onNewDataPoint: (data: SessionDataPoint) => void;
  onNewFaceMetrics: (metrics: ExtendedFaceMetrics | null) => void;
  onNewPrediction: (prediction: MLPrediction | null) => void;
  onMultitask: () => void;
}

export default function WebcamPanel({
  sessionActive,
  onStartSession,
  onStopSession,
  onNewDataPoint,
  onNewFaceMetrics,
  onNewPrediction,
  onMultitask,
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

  // Load face-api.js models
  useEffect(() => {
    const loadModels = async () => {
      const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/';
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        ]);
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
  }, [toast]);

  // Lazy-load context modules once face-api models are ready
  useEffect(() => {
    if (!modelsLoaded) return;
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
  }, [modelsLoaded]);

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
      const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
      const resizedDetections = faceapi.resizeResults(detections, displaySize);

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
        faceapi.draw.drawFaceLandmarks(canvasRef.current, [primary.detection]);

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

  useEffect(() => {
    if (isWebcamOn && modelsLoaded) {
      animationFrameRef.current = requestAnimationFrame(runAnalysis);
    } else {
      stopAnalysis();
    }
    return () => {
      stopAnalysis();
    };
  }, [isWebcamOn, modelsLoaded, runAnalysis, stopAnalysis]);

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
          <Button onClick={handleWebcamToggle} size="icon" variant="outline" disabled={isLoading}>
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
