'use client';

import { useState } from 'react';
import { useToast } from '@/hooks/use-toast';
import type { SessionDataPoint, SessionSummary, FaceMetrics } from '@/lib/types';
import SummaryCards from './summary-cards';
import AttentionChart from './attention-chart';
import AttentionPieChart from './attention-pie-chart';
import ModelDataCard from './model-data-card';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '../ui/button';
import { Download, Loader2 } from 'lucide-react';
import { generateReportAction } from '@/app/actions';

interface VisualsPanelProps {
  sessionData: SessionDataPoint[];
  summary: SessionSummary;
  currentScore: number;
  sessionActive: boolean;
  faceMetrics: FaceMetrics | null;
}

export default function VisualsPanel({ sessionData, summary, currentScore, sessionActive, faceMetrics }: VisualsPanelProps) {
  const { toast } = useToast();
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

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
      const result = await generateReportAction({
        averageAttention: summary.averageAttention,
        multitaskEvents: summary.multitaskEvents,
        engagementLevel: summary.engagementLevel,
        duration: summary.duration,
      });

      if (result.success && result.data) {
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
        throw new Error(result.error || 'Unknown error');
      }
    } catch (error: any) {
      console.error('Failed to generate report:', error);
      toast({
        title: 'Report Generation Failed',
        description: error.message,
        variant: 'destructive',
      });
    } finally {
      setIsGeneratingReport(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <SummaryCards summary={summary} currentScore={currentScore} />
      <div className="grid gap-4 lg:grid-cols-2">
        <ModelDataCard faceMetrics={faceMetrics} />
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
      </div>
    </div>
  );
}
