'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, Play, StopCircle, Video, VideoOff } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import type { SessionDataPoint, FaceMetrics } from '@/lib/types';
import * as faceapi from 'face-api.js';
import { analyzeFacialFeatures } from '@/ai/flows/analyze-facial-features';
import { calculateEAR, calculateMAR, calculateHeadPose } from '@/lib/face-geometry';

interface WebcamPanelProps {
  sessionActive: boolean;
  onStartSession: () => void;
  onStopSession: () => void;
  onNewDataPoint: (data: SessionDataPoint) => void;
  onNewFaceMetrics: (metrics: FaceMetrics | null) => void;
  onMultitask: () => void;
}

export default function WebcamPanel({
  sessionActive,
  onStartSession,
  onStopSession,
  onNewDataPoint,
  onNewFaceMetrics,
  onMultitask,
}: WebcamPanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { toast } = useToast();
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [isWebcamOn, setIsWebcamOn] = useState(false);
  
  const animationFrameRef = useRef<number | null>(null);

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
    if(animationFrameRef.current) {
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
    
    canvasRef.current.width = videoRef.current.videoWidth;
    canvasRef.current.height = videoRef.current.videoHeight;
    const displaySize = { width: videoRef.current.videoWidth, height: videoRef.current.videoHeight };

    try {
      const detections = await faceapi.detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
      const resizedDetections = faceapi.resizeResults(detections, displaySize);

      const ctx = canvasRef.current.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

      if (resizedDetections && resizedDetections.length > 0 && resizedDetections[0]) {
        const landmarks = resizedDetections[0].landmarks.positions;
        faceapi.draw.drawFaceLandmarks(canvasRef.current, resizedDetections);
      
        const ear = calculateEAR(landmarks);
        const mar = calculateMAR(landmarks);
        const { yaw, pitch, roll } = calculateHeadPose(landmarks);

        onNewFaceMetrics({ ear, mar, yaw, pitch, roll });

        if (sessionActive && (ear !== null && mar !== null && yaw !== null && pitch !== null && roll !== null)) {
            const { attentionState } = await analyzeFacialFeatures({
                faceLandmarks: JSON.stringify(landmarks), ear, mar, yaw, pitch, roll
            });
            
            let score = 50;
            if (attentionState === 'Attentive') score = Math.random() * 20 + 80;
            else if (attentionState === 'Distracted') score = Math.random() * 20 + 40;
            else if (attentionState === 'Multitasking') score = Math.random() * 20;

            onNewDataPoint({
              timestamp: Date.now(),
              attentionScore: Math.round(score),
              attentionState,
            });
        }
      } else {
        onNewFaceMetrics(null);
      }
    } catch(e) {
      console.error("AI analysis failed", e);
    }
    if (isWebcamOn) {
      animationFrameRef.current = requestAnimationFrame(runAnalysis);
    }
  }, [modelsLoaded, isWebcamOn, sessionActive, onNewDataPoint, onNewFaceMetrics]);
  
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
