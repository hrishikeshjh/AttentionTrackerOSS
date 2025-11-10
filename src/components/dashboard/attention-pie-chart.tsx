'use client';

import type { SessionDataPoint } from '@/lib/types';
import { useMemo } from 'react';
import { Pie, PieChart, ResponsiveContainer, Tooltip, Cell } from 'recharts';
import {
  ChartContainer,
  ChartTooltipContent,
} from '@/components/ui/chart';

interface AttentionPieChartProps {
  data: SessionDataPoint[];
}

const COLORS = {
  Attentive: 'hsl(var(--chart-1))',
  Distracted: 'hsl(var(--chart-3))',
  Multitasking: 'hsl(var(--chart-5))',
  Idle: 'hsl(var(--muted))'
};

export default function AttentionPieChart({ data }: AttentionPieChartProps) {
  const chartData = useMemo(() => {
    const stateCounts = data.reduce((acc, curr) => {
      acc[curr.attentionState] = (acc[curr.attentionState] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(stateCounts).map(([name, value]) => ({
      name,
      value,
      fill: COLORS[name as keyof typeof COLORS] || COLORS.Idle,
    }));
  }, [data]);

  return (
    <ChartContainer config={{}} className="h-full w-full">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Tooltip
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
      </ResponsiveContainer>
    </ChartContainer>
  );
}
