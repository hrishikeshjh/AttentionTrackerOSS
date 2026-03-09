'use client';

import type { SessionDataPoint } from '@/lib/types';
import { useMemo } from 'react';
import { Pie, PieChart, Cell, Tooltip, Legend } from 'recharts';
import { ResponsiveContainer } from 'recharts';

interface AttentionPieChartProps {
  data: SessionDataPoint[];
}

const COLORS = {
  Attentive: '#00FF99',
  Distracted: '#FFA657',
  Multitasking: '#F85149',
  Idle: '#6E7681',
};

export default function AttentionPieChart({ data }: AttentionPieChartProps) {
  const chartData = useMemo(() => {
    if (data.length === 0) {
      return [{ name: 'Idle', value: 1, fill: COLORS.Idle }];
    }
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
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie
          data={chartData}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          outerRadius={70}
          strokeWidth={2}
          stroke="#0D1117"
        >
          {chartData.map((entry) => (
            <Cell key={`cell-${entry.name}`} fill={entry.fill} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            backgroundColor: '#161B22',
            border: '1px solid #30363D',
            borderRadius: '8px'
          }}
          labelStyle={{ color: '#F0F6FC' }}
        />
        <Legend
          verticalAlign="bottom"
          height={36}
          formatter={(value) => <span style={{ color: '#8B949E' }}>{value}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}