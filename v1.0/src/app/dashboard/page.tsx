'use client';

import { useState, useMemo, useEffect, useCallback } from 'react';
import type { SessionDataPoint, SessionSummary, ExtendedFaceMetrics, MLPrediction } from '@/lib/types';
import Header from '@/components/dashboard/header';
import WebcamPanel, { type DaceStatus } from '@/components/dashboard/webcam-panel';
import VisualsPanel from '@/components/dashboard/visuals-panel';
import { resetMLEngine } from '@/lib/ml-engine';
import { resetAnalyticsEngine } from '@/lib/analytics';
import { summarizeUnsure } from '@/lib/scoring/dace/abstain';
import type { EvalLogger } from '@/lib/dace/eval-log';
import type { LatencySummary } from '@/lib/dace/latency';
import { Badge } from '@/components/ui/badge';

const RESEARCH_BUILD = process.env.NEXT_PUBLIC_DACE_RESEARCH === '1';

export default function DashboardPage() {
  const [sessionActive, setSessionActive] = useState(false);
  const [sessionData, setSessionData] = useState<SessionDataPoint[]>([]);
  const [multitaskEvents, setMultitaskEvents] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [currentFaceMetrics, setCurrentFaceMetrics] = useState<ExtendedFaceMetrics | null>(null);
  const [currentPrediction, setCurrentPrediction] = useState<MLPrediction | null>(null);
  // DACE (research; all null/unused when DACE is off)
  const [daceStatus, setDaceStatus] = useState<DaceStatus | null>(null);
  const [latency, setLatency] = useState<LatencySummary | null>(null);
  const [evalLogger, setEvalLogger] = useState<EvalLogger | null>(null);

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

  const handleDaceStatus = useCallback((s: DaceStatus) => setDaceStatus(s), []);
  const handleLatency = useCallback((l: LatencySummary) => setLatency(l), []);
  const handleEvalLog = useCallback((l: EvalLogger | null) => setEvalLogger(l), []);

  const currentScore = useMemo(() => {
    if (sessionData.length === 0) return 0;
    return sessionData[sessionData.length - 1].attentionScore;
  }, [sessionData]);

  /** DACE: the current score is a held value (window Unsure). */
  const currentHeld = sessionData.length > 0 && sessionData[sessionData.length - 1].unsure === true;

  const summary: SessionSummary = useMemo(() => {
    // DACE "Unsure" points hold the score and are excluded from the average (never a penalty).
    const scored = sessionData.filter((d) => !d.unsure);
    const totalPoints = scored.length;
    const avg = totalPoints === 0 ? 0 : Math.round(scored.reduce((sum, d) => sum + d.attentionScore, 0) / totalPoints);

    let engagementLevel: 'High' | 'Moderate' | 'Low';
    if (avg >= 75) engagementLevel = 'High';
    else if (avg >= 50) engagementLevel = 'Moderate';
    else engagementLevel = 'Low';

    const durationInSeconds = Math.round(elapsedTime / 1000);
    const minutes = String(Math.floor(durationInSeconds / 60)).padStart(2, '0');
    const seconds = String(durationInSeconds % 60).padStart(2, '0');
    const duration = `${minutes}:${seconds}`;

    const base: SessionSummary = { averageAttention: avg, multitaskEvents, engagementLevel, duration };
    if (!sessionData.some((d) => d.unsure !== undefined)) return base;
    const u = summarizeUnsure(sessionData);
    return { ...base, unsureShare: u.share, unsureReasons: u.counts, unsureText: u.text };
  }, [sessionData, multitaskEvents, elapsedTime]);

  return (
    <div className="flex min-h-screen w-full flex-col bg-background">
      <Header />
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-8">
        {(RESEARCH_BUILD || daceStatus?.requested) && (
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <Badge variant="outline">Research build</Badge>
            {daceStatus?.active && (
              <Badge variant="secondary">
                DACE {daceStatus.version} · cfg {daceStatus.configHash} · {daceStatus.measurementLabel}
                {daceStatus.detectorAvailable ? '' : ' · no object detector'}
              </Badge>
            )}
            {daceStatus?.fellBack && <Badge variant="destructive">DACE requested — fell back to legacy</Badge>}
            <span className="text-muted-foreground">Experimental scoring; results are not validated.</span>
          </div>
        )}
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
              onDaceStatus={handleDaceStatus}
              onNewLatency={handleLatency}
              onEvalLog={handleEvalLog}
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
              currentHeld={currentHeld}
              daceStatus={daceStatus}
              latency={latency}
              evalLogger={RESEARCH_BUILD ? evalLogger : null}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
