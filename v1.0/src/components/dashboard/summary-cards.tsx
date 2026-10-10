'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { SessionSummary, UnsureReason } from '@/lib/types';
import { UNSURE_REASON_TEXT } from '@/lib/scoring/dace/abstain';
import { THRESHOLDS } from '@/lib/scoring/constants';
import { Activity, Gauge, HelpCircle, Hourglass, MousePointerClick } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SummaryCardsProps {
  summary: SessionSummary;
  currentScore: number;
  /** DACE: current score is held because the activity could not be determined */
  currentHeld?: boolean;
}

export default function SummaryCards({ summary, currentScore, currentHeld = false }: SummaryCardsProps) {
  const showUnsure = summary.unsureShare !== undefined;
  const dominant = (Object.entries(summary.unsureReasons ?? {}) as Array<[UnsureReason, number]>)
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1])[0]?.[0];
  const scoreColor =
    currentScore >= THRESHOLDS.ATTENTIVE_MIN
      ? 'text-accent'
      : currentScore >= THRESHOLDS.DISTRACTED_MIN
        ? 'text-yellow-400'
        : 'text-red-500';

  return (
    <div className={cn('grid gap-4 md:grid-cols-2', showUnsure ? 'lg:grid-cols-5' : 'lg:grid-cols-4')}>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Current Attention</CardTitle>
          <Gauge className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className={cn('text-2xl font-bold transition-colors', scoreColor)}>
            {currentScore}%
          </div>
          <p className="text-xs text-muted-foreground">
            {currentHeld ? 'Held (activity undetermined)' : 'Live engagement score'}
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Average Attention</CardTitle>
          <Activity className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{summary.averageAttention}%</div>
          <p className="text-xs text-muted-foreground">
            {showUnsure ? 'Session average (undetermined time excluded)' : 'Session average'}
          </p>
        </CardContent>
      </Card>
      {showUnsure && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Undetermined</CardTitle>
            <HelpCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{Math.round((summary.unsureShare ?? 0) * 100)}%</div>
            <p className="text-xs text-muted-foreground">
              {dominant ? `Mostly: ${UNSURE_REASON_TEXT[dominant]} · not a penalty` : 'Activity could not be determined · not a penalty'}
            </p>
          </CardContent>
        </Card>
      )}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Multitask Events</CardTitle>
          <MousePointerClick className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{summary.multitaskEvents}</div>
          <p className="text-xs text-muted-foreground">Tab switches detected</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Session Duration</CardTitle>
          <Hourglass className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{summary.duration}</div>
          <p className="text-xs text-muted-foreground">Total elapsed time</p>
        </CardContent>
      </Card>
    </div>
  );
}
