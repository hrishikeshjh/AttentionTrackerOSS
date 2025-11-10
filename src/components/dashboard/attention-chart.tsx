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
import { ChartTooltipContent } from '@/components/ui/chart';

interface AttentionChartProps {
  data: SessionDataPoint[];
}

export default function AttentionChart({ data }: AttentionChartProps) {
  const chartData = data.map(d => ({
    time: new Date(d.timestamp).toLocaleTimeString(),
    score: d.attentionScore,
  }));
  
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={chartData}>
        <XAxis
          dataKey="time"
          stroke="hsl(var(--muted-foreground))"
          fontSize={12}
          tickLine={false}
          axisLine={false}
          tickFormatter={(value, index) => index % Math.ceil(data.length / 5) === 0 ? value : ''}
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
          type="monotone"
          dataKey="score"
          stroke="hsl(var(--primary))"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
