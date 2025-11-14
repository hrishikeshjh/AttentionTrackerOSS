'use client';

import type { SessionDataPoint } from '@/lib/types';
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { ChartContainer, ChartTooltipContent } from '@/components/ui/chart';
import { useMemo } from 'react';

interface AttentionChartProps {
  data: SessionDataPoint[];
}

const chartConfig = {
  score: {
    label: 'Attention Score',
    color: 'hsl(var(--chart-1))',
  },
};

export default function AttentionChart({ data }: AttentionChartProps) {
  const chartData = useMemo(() => data.map(d => ({
    time: new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    score: d.attentionScore,
  })), [data]);
  
  return (
    <ChartContainer config={chartConfig} className="h-full w-full">
      <LineChart
        accessibilityLayer
        data={chartData}
        margin={{
          top: 5,
          right: 10,
          left: 10,
          bottom: 5,
        }}
      >
        <XAxis
          dataKey="time"
          stroke="hsl(var(--muted-foreground))"
          fontSize={12}
          tickLine={false}
          axisLine={false}
          tickFormatter={(value, index) => {
            if (chartData.length < 2) return value;
            if (index % Math.ceil(chartData.length / 5) === 0) return value;
            return '';
          }}
        />
        <YAxis
          stroke="hsl(var(--muted-foreground))"
          fontSize={12}
          tickLine={false}
          axisLine={false}
          domain={[0, 100]}
        />
        <Tooltip
          cursor={{ stroke: 'hsl(var(--primary))', strokeWidth: 1, strokeDasharray: '3 3' }}
          content={<ChartTooltipContent
            formatter={(value) => `${value}%`}
            labelClassName="font-bold"
            indicator="dot"
             />}
        />
        <Line
          dataKey="score"
          type="monotone"
          stroke="hsl(var(--primary))"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ChartContainer>
  );
}