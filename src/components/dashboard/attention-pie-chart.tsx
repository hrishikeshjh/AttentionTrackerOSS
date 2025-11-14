'use client';

import type { SessionDataPoint } from '@/lib/types';
import { useMemo } from 'react';
import { Pie, PieChart, Cell } from 'recharts';
import {
  ChartContainer,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from '@/components/ui/chart';

interface AttentionPieChartProps {
  data: SessionDataPoint[];
}

const chartConfig = {
  Attentive: {
    label: 'Attentive',
    color: 'hsl(var(--chart-1))',
  },
  Distracted: {
    label: 'Distracted',
    color: 'hsl(var(--chart-3))',
  },
  Multitasking: {
    label: 'Multitasking',
    color: 'hsl(var(--chart-5))',
  },
  Idle: {
    label: 'Idle',
    color: 'hsl(var(--muted))'
  },
};

export default function AttentionPieChart({ data }: AttentionPieChartProps) {
  const chartData = useMemo(() => {
    if (data.length === 0) {
      return [{ name: 'Idle', value: 1, fill: 'hsl(var(--muted))' }];
    }
    const stateCounts = data.reduce((acc, curr) => {
      acc[curr.attentionState] = (acc[curr.attentionState] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(stateCounts).map(([name, value]) => ({
      name,
      value,
      fill: chartConfig[name as keyof typeof chartConfig]?.color || 'hsl(var(--muted))',
    }));
  }, [data]);

  return (
    <ChartContainer config={chartConfig} className="h-full w-full flex flex-col items-center">
      <PieChart accessibilityLayer width={200} height={200}>
        <ChartTooltipContent
          cursor={false}
          content={<ChartTooltipContent hideLabel indicator="dot" />}
        />
        <Pie
          data={chartData}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          outerRadius={80}
          strokeWidth={2}
        >
          {chartData.map((entry) => (
            <Cell key={`cell-${entry.name}`} fill={entry.fill} />
          ))}
        </Pie>
      </PieChart>
      <ChartLegend content={<ChartLegendContent />} />
    </ChartContainer>
  );
}