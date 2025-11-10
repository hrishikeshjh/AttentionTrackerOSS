'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { SessionSummary } from '@/lib/types';
import { Activity, Gauge, Hourglass, MousePointerClick } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SummaryCardsProps {
  summary: SessionSummary;
  currentScore: number;
}

export default function SummaryCards({ summary, currentScore }: SummaryCardsProps) {
  const scoreColor =
    currentScore >= 75 ? 'text-accent' : currentScore >= 50 ? 'text-yellow-400' : 'text-red-500';

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Current Attention</CardTitle>
          <Gauge className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className={cn('text-2xl font-bold transition-colors', scoreColor)}>
            {currentScore}%
          </div>
          <p className="text-xs text-muted-foreground">Live engagement score</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Average Attention</CardTitle>
          <Activity className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{summary.averageAttention}%</div>
          <p className="text-xs text-muted-foreground">Session average</p>
        </CardContent>
      </Card>
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
