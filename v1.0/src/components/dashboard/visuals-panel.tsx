'use client';

import { useState, useEffect } from 'react';
import { useToast } from '@/hooks/use-toast';
import type { SessionDataPoint, SessionSummary, ExtendedFaceMetrics, MLPrediction, SessionAnalytics } from '@/lib/types';
import SummaryCards from './summary-cards';
import AttentionChart from './attention-chart';
import AttentionPieChart from './attention-pie-chart';
import ModelDataCard from './model-data-card';
import MLInsightsCard from './ml-insights-card';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '../ui/button';
import { Download, FileDown, Loader2 } from 'lucide-react';
import { generateReportAction } from '@/app/actions';
import { auth } from '@/lib/firebase';
import { getAnalyticsEngine, type AnomalyResult } from '@/lib/analytics';
import type { DaceStatus } from './webcam-panel';
import type { LatencySummary } from '@/lib/dace/latency';
import { downloadJsonl, type EvalLogger } from '@/lib/dace/eval-log';
import { UNSURE_REASON_TEXT } from '@/lib/scoring/dace/abstain';
import type { UnsureReason } from '@/lib/types';

interface VisualsPanelProps {
  sessionData: SessionDataPoint[];
  summary: SessionSummary;
  currentScore: number;
  sessionActive: boolean;
  faceMetrics: ExtendedFaceMetrics | null;
  prediction: MLPrediction | null;
  /** DACE: the current score is held (window Unsure) */
  currentHeld?: boolean;
  daceStatus?: DaceStatus | null;
  latency?: LatencySummary | null;
  /** research builds only: enables the manual eval-log download */
  evalLogger?: EvalLogger | null;
}

/** "U2: 12, U1: 3" -> "desk not visible (12), no phone, book or writing evidence (3)" */
function reasonsText(counts: Partial<Record<UnsureReason, number>> | undefined): string | undefined {
  if (!counts) return undefined;
  const parts = (Object.entries(counts) as Array<[UnsureReason, number]>)
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([r, n]) => `${UNSURE_REASON_TEXT[r]} (${n})`);
  return parts.length ? parts.join(', ') : undefined;
}

const LATENCY_ORDER = ['measure', 'detector', 'hands', 'rules', 'mapping', 'total'];

export default function VisualsPanel({
  sessionData,
  summary,
  currentScore,
  sessionActive,
  faceMetrics,
  prediction,
  currentHeld = false,
  daceStatus = null,
  latency = null,
  evalLogger = null,
}: VisualsPanelProps) {
  const { toast } = useToast();
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [analytics, setAnalytics] = useState<SessionAnalytics | null>(null);
  const [anomalies, setAnomalies] = useState<AnomalyResult | null>(null);

  // Update analytics when session data changes
  useEffect(() => {
    if (sessionData.length > 0) {
      const engine = getAnalyticsEngine();
      engine.setData(sessionData);
      setAnalytics(engine.analyze());
      setAnomalies(engine.getAnomalies());
    } else {
      setAnalytics(null);
      setAnomalies(null);
    }
  }, [sessionData]);

  const handleGenerateReport = async () => {
    if (sessionData.length === 0) {
      toast({
        title: 'Cannot Generate Report',
        description: 'No session data has been recorded.',
        variant: 'destructive',
      });
      return;
    }
    setIsGeneratingReport(true);
    try {
      const user = auth.currentUser;
      if (!user) throw new Error('Please sign in again to export a report.');
      const idToken = await user.getIdToken();
      const result = await generateReportAction(idToken, {
        averageAttention: summary.averageAttention,
        multitaskEvents: summary.multitaskEvents,
        engagementLevel: summary.engagementLevel,
        duration: summary.duration,
        ...(summary.unsureShare !== undefined
          ? { unsureShare: summary.unsureShare, unsureReasons: reasonsText(summary.unsureReasons)?.slice(0, 300) }
          : {}),
      });

      if (!result.success) {
        throw new Error(result.error || 'Unknown error');
      }
      if (result.data) {
        const link = document.createElement('a');
        link.href = result.data;
        link.download = `CognEn_Session_Report_${new Date().toISOString()}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        toast({
          title: 'Report Generated',
          description: 'Your session summary PDF has been downloaded.',
        });
      } else {
        throw new Error('Empty report');
      }
    } catch (error: unknown) {
      console.error('Failed to generate report:', error);
      toast({
        title: 'Report Generation Failed',
        description: error instanceof Error ? error.message : 'An unexpected error occurred.',
        variant: 'destructive',
      });
    } finally {
      setIsGeneratingReport(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <SummaryCards summary={summary} currentScore={currentScore} currentHeld={currentHeld} />
      <div className="grid gap-4 lg:grid-cols-2">
        <ModelDataCard faceMetrics={faceMetrics} measurementLabel={daceStatus?.active ? daceStatus.measurementLabel : undefined} />
        <MLInsightsCard
          prediction={prediction}
          analytics={analytics}
          anomalies={anomalies}
          sessionActive={sessionActive}
        />
      </div>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Session Analysis</CardTitle>
            <CardDescription>
              Real-time cognitive engagement metrics.
            </CardDescription>
          </div>
          <Button onClick={handleGenerateReport} disabled={sessionActive || isGeneratingReport || sessionData.length === 0}>
            {isGeneratingReport ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Download className="mr-2 h-4 w-4" />
            )}
            Export Report
          </Button>
        </CardHeader>
        <CardContent>
          {sessionData.length > 0 ? (
            <div className="grid gap-6 md:grid-cols-2">
              <div className="h-64">
                <h3 className="text-lg font-semibold mb-2">Attention Over Time</h3>
                <AttentionChart data={sessionData} />
              </div>
              <div className="h-64">
                <h3 className="text-lg font-semibold mb-2">Attention State Distribution</h3>
                <AttentionPieChart data={sessionData} />
              </div>
            </div>
          ) : (
            <div className="flex h-64 items-center justify-center text-muted-foreground">
              {sessionActive ? 'Waiting for session data...' : 'Start a session to see your analysis.'}
            </div>
          )}
        </CardContent>
      </Card>
      {evalLogger && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>DACE research tools</CardTitle>
              <CardDescription>
                Evaluation log (no images, no user id, random session id) and per-component latency. Downloaded only when you click.
              </CardDescription>
            </div>
            <Button
              variant="outline"
              onClick={() => downloadJsonl(evalLogger)}
              disabled={evalLogger.size() === 0}
            >
              <FileDown className="mr-2 h-4 w-4" />
              Download eval log
            </Button>
          </CardHeader>
          <CardContent>
            {latency && Object.keys(latency).length > 0 ? (
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="text-muted-foreground">
                    <th className="text-left font-normal">component</th>
                    <th className="text-right font-normal">n</th>
                    <th className="text-right font-normal">p50 ms</th>
                    <th className="text-right font-normal">p95 ms</th>
                    <th className="text-right font-normal">max ms</th>
                  </tr>
                </thead>
                <tbody>
                  {LATENCY_ORDER.filter((k) => latency[k]).map((k) => (
                    <tr key={k} className={k === 'total' && latency[k].p95 > 200 ? 'text-red-500' : ''}>
                      <td>{k}</td>
                      <td className="text-right">{latency[k].n}</td>
                      <td className="text-right">{latency[k].p50.toFixed(1)}</td>
                      <td className="text-right">{latency[k].p95.toFixed(1)}</td>
                      <td className="text-right">{latency[k].max.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-sm text-muted-foreground">No latency samples yet.</p>
            )}
            <p className="mt-2 text-xs text-muted-foreground">
              Target: p95 total ≤ 200 ms per frame (not yet measured on reference hardware). Log records: {evalLogger.size()}.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
