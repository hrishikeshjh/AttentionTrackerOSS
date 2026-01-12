'use client';

import { useState, useMemo, useEffect, useCallback } from 'react';
import type { SessionDataPoint, SessionSummary, ExtendedFaceMetrics, MLPrediction } from '@/lib/types';
import Header from '@/components/dashboard/header';
import WebcamPanel from '@/components/dashboard/webcam-panel';
import VisualsPanel from '@/components/dashboard/visuals-panel';
import { resetMLEngine } from '@/lib/ml-engine';
import { resetAnalyticsEngine } from '@/lib/analytics';

export default function DashboardPage() {
  const [sessionActive, setSessionActive] = useState(false);
  const [sessionData, setSessionData] = useState<SessionDataPoint[]>([]);
  const [multitaskEvents, setMultitaskEvents] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [currentFaceMetrics, setCurrentFaceMetrics] = useState<ExtendedFaceMetrics | null>(null);
  const [currentPrediction, setCurrentPrediction] = useState<MLPrediction | null>(null);

  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (sessionActive && startTime) {
      timer = setInterval(() => {
        setElapsedTime(Date.now() - startTime);
      }, 1000);
    } else {
      setElapsedTime(0);
    }
    return () => clearInterval(timer);
  }, [sessionActive, startTime]);

  const handleStartSession = useCallback(() => {
    // Reset engines for fresh session
    resetMLEngine();
    resetAnalyticsEngine();

    setSessionData([]);
    setMultitaskEvents(0);
    setStartTime(Date.now());
    setCurrentFaceMetrics(null);
    setCurrentPrediction(null);
    setSessionActive(true);
  }, []);

  const handleStopSession = useCallback(() => {
    setSessionActive(false);
  }, []);

  const handleNewDataPoint = useCallback((data: SessionDataPoint) => {
    setSessionData(prev => [...prev, data]);
  }, []);

  const handleNewFaceMetrics = useCallback((metrics: ExtendedFaceMetrics | null) => {
    setCurrentFaceMetrics(metrics);
  }, []);

  const handleNewPrediction = useCallback((prediction: MLPrediction | null) => {
    setCurrentPrediction(prediction);
  }, []);

  const handleMultitask = useCallback(() => {
    setMultitaskEvents(prev => prev + 1);
  }, []);

  const currentScore = useMemo(() => {
    if (sessionData.length === 0) return 0;
    return sessionData[sessionData.length - 1].attentionScore;
  }, [sessionData]);

  const summary: SessionSummary = useMemo(() => {
    const totalPoints = sessionData.length;
    const avg = totalPoints === 0 ? 0 : Math.round(sessionData.reduce((sum, d) => sum + d.attentionScore, 0) / totalPoints);

    let engagementLevel: 'High' | 'Moderate' | 'Low';
    if (avg >= 75) engagementLevel = 'High';
    else if (avg >= 50) engagementLevel = 'Moderate';
    else engagementLevel = 'Low';

    const durationInSeconds = Math.round(elapsedTime / 1000);
    const minutes = String(Math.floor(durationInSeconds / 60)).padStart(2, '0');
    const seconds = String(durationInSeconds % 60).padStart(2, '0');
    const duration = `${minutes}:${seconds}`;

    return { averageAttention: avg, multitaskEvents, engagementLevel, duration };
  }, [sessionData, multitaskEvents, elapsedTime]);

  return (
    <div className="flex min-h-screen w-full flex-col bg-background">
      <Header />
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-8">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
          <div className="lg:col-span-2">
            <WebcamPanel
              sessionActive={sessionActive}
              onStartSession={handleStartSession}
              onStopSession={handleStopSession}
              onNewDataPoint={handleNewDataPoint}
              onNewFaceMetrics={handleNewFaceMetrics}
              onNewPrediction={handleNewPrediction}
              onMultitask={handleMultitask}
            />
          </div>
          <div className="lg:col-span-3">
            <VisualsPanel
              sessionData={sessionData}
              summary={summary}
              currentScore={currentScore}
              sessionActive={sessionActive}
              faceMetrics={currentFaceMetrics}
              prediction={currentPrediction}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
