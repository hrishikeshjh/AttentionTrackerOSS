'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import type { FaceMetrics } from '@/lib/types';
import { Eye, FileJson, Gauge, GitBranch, Smile } from 'lucide-react';

interface ModelDataCardProps {
  faceMetrics: FaceMetrics | null;
}

const MetricDisplay = ({ label, value, icon }: { label: string; value: number | null; icon: React.ReactNode }) => (
  <div className="flex items-center justify-between">
    <div className="flex items-center gap-2 text-sm text-muted-foreground">
      {icon}
      <span>{label}</span>
    </div>
    <span className="font-mono text-sm font-medium">
      {value !== null && !isNaN(value) ? value.toFixed(2) : 'N/A'}
    </span>
  </div>
);

export default function ModelDataCard({ faceMetrics }: ModelDataCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Live Model Data</CardTitle>
        <CardDescription>Underlying metrics from the facial analysis model.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <MetricDisplay label="Eye Aspect Ratio (EAR)" value={faceMetrics?.ear ?? null} icon={<Eye className="h-4 w-4" />} />
        <MetricDisplay label="Mouth Aspect Ratio (MAR)" value={faceMetrics?.mar ?? null} icon={<Smile className="h-4 w-4" />} />
        <MetricDisplay label="Head Yaw" value={faceMetrics?.yaw ?? null} icon={<GitBranch className="h-4 w-4" />} />
        <MetricDisplay label="Head Pitch" value={faceMetrics?.pitch ?? null} icon={<Gauge className="h-4 w-4" />} />
        <MetricDisplay label="Head Roll" value={faceMetrics?.roll ?? null} icon={<FileJson className="h-4 w-4" />} />
      </CardContent>
    </Card>
  );
}
