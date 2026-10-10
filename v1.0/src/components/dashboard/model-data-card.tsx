'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import type { ExtendedFaceMetrics } from '@/lib/types';
import {
  Eye,
  Smile,
  GitBranch,
  Gauge,
  Activity,
  Zap,
  Target,
  Clock,
  AlertTriangle
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface ModelDataCardProps {
  faceMetrics: ExtendedFaceMetrics | null;
  /** DACE: which measurement provider produced the values */
  measurementLabel?: string;
}

const fmt = (v: number | null | undefined, digits = 1, unit = '') =>
  v === null || v === undefined || Number.isNaN(v) ? 'N/A' : `${v.toFixed(digits)}${unit}`;

const Flag = ({ label, on }: { label: string; on: boolean }) => (
  <span
    className={cn(
      'rounded px-1.5 py-0.5 text-[10px] font-medium',
      on ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground'
    )}
  >
    {label}
  </span>
);

type MetricStatus = 'good' | 'warning' | 'danger' | 'neutral';

const getStatus = (value: number | null, type: 'higher' | 'lower' | 'range', thresholds: { good: number; warning: number; max?: number; min?: number }): MetricStatus => {
  if (value === null || isNaN(value)) return 'neutral';

  if (type === 'higher') {
    if (value >= thresholds.good) return 'good';
    if (value >= thresholds.warning) return 'warning';
    return 'danger';
  } else if (type === 'lower') {
    if (value <= thresholds.good) return 'good';
    if (value <= thresholds.warning) return 'warning';
    return 'danger';
  } else {
    // range
    if (value >= (thresholds.min || 0) && value <= (thresholds.max || 100) && value >= thresholds.good) return 'good';
    if (value >= thresholds.warning) return 'warning';
    return 'danger';
  }
};

const statusColors: Record<MetricStatus, string> = {
  good: 'text-green-500',
  warning: 'text-yellow-500',
  danger: 'text-red-500',
  neutral: 'text-muted-foreground',
};

const statusBgColors: Record<MetricStatus, string> = {
  good: 'bg-green-500',
  warning: 'bg-yellow-500',
  danger: 'bg-red-500',
  neutral: 'bg-muted',
};

const MetricRow = ({
  label,
  value,
  unit = '',
  icon,
  status = 'neutral',
  showProgress = false,
  progressValue = 0,
}: {
  label: string;
  value: number | string | null;
  unit?: string;
  icon: React.ReactNode;
  status?: MetricStatus;
  showProgress?: boolean;
  progressValue?: number;
}) => (
  <div className="space-y-1">
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        {icon}
        <span>{label}</span>
      </div>
      <span className={cn('font-mono text-sm font-medium', statusColors[status])}>
        {value !== null && value !== undefined && !Number.isNaN(value)
          ? typeof value === 'number' ? `${value.toFixed(2)}${unit}` : value
          : 'N/A'}
      </span>
    </div>
    {showProgress && (
      <Progress
        value={progressValue}
        className="h-1.5"
        indicatorClassName={statusBgColors[status]}
      />
    )}
  </div>
);

const MetricSection = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <div className="space-y-2">
    <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{title}</h4>
    <div className="space-y-2">{children}</div>
  </div>
);

export default function ModelDataCard({ faceMetrics, measurementLabel }: ModelDataCardProps) {
  const dace = faceMetrics?.dace;
  const desk = faceMetrics?.deskEvidence;
  // Calculate derived values
  const earStatus = getStatus(faceMetrics?.ear ?? null, 'higher', { good: 0.25, warning: 0.2 });
  const perclosStatus = getStatus(faceMetrics?.perclos ?? null, 'lower', { good: 15, warning: 25 });
  const fatigueStatus = getStatus(faceMetrics?.fatigueIndex ?? null, 'lower', { good: 30, warning: 50 });
  const stabilityStatus = getStatus(faceMetrics?.headPoseStability ?? null, 'higher', { good: 70, warning: 50 });
  const fidgetStatus = getStatus(faceMetrics?.fidgetScore ?? null, 'lower', { good: 25, warning: 50 });

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Live Model Data</CardTitle>
        <CardDescription>
          Real-time biometric metrics from facial analysis.
          {measurementLabel ? <span className="block text-xs">Measurement: {measurementLabel}</span> : null}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {dace && (
          <MetricSection title="DACE measurement (degrees, + = head down)">
            <div className="grid grid-cols-4 gap-2 text-xs">
              <div className="text-center">
                <div className="text-muted-foreground">Pitch</div>
                <div className="font-mono font-medium">{fmt(dace.pitchDeg, 1, '°')}</div>
              </div>
              <div className="text-center">
                <div className="text-muted-foreground">Δ pitch</div>
                <div className="font-mono font-medium">{fmt(dace.relPitchDeg, 1, '°')}</div>
              </div>
              <div className="text-center">
                <div className="text-muted-foreground">Yaw</div>
                <div className="font-mono font-medium">{fmt(dace.yawDeg, 1, '°')}</div>
              </div>
              <div className="text-center">
                <div className="text-muted-foreground">Roll</div>
                <div className="font-mono font-medium">{fmt(dace.rollDeg, 1, '°')}</div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <MetricRow label="Eye gaze vertical (+down)" value={dace.gazeV} icon={<Target className="h-4 w-4" />} />
              <MetricRow label="Eye gaze horizontal" value={dace.gazeH} icon={<Target className="h-4 w-4" />} />
              <MetricRow
                label="Residual openness"
                value={dace.residualOpenness}
                icon={<Eye className="h-4 w-4" />}
                status={dace.residualOpenness === null ? 'neutral' : dace.residualOpenness >= 0.7 ? 'good' : 'danger'}
              />
              <MetricRow
                label="Corrected PERCLOS"
                value={dace.correctedPerclos === null ? null : dace.correctedPerclos * 100}
                unit="%"
                icon={<Clock className="h-4 w-4" />}
                status={dace.correctedPerclos === null ? 'neutral' : dace.correctedPerclos < 0.15 ? 'good' : 'danger'}
              />
            </div>
            <div className="flex flex-wrap gap-1">
              <Flag label="looking down" on={dace.lookingDown} />
              <Flag label={dace.gazeValid ? 'iris gaze valid' : 'gaze: head pitch only'} on={dace.gazeValid} />
              {dace.calibrating && <Flag label="calibrating" on />}
              {desk && (
                <>
                  <Flag label="phone in use" on={desk.phoneInUse} />
                  <Flag label="phone present" on={desk.phonePresent} />
                  <Flag label="book" on={desk.book} />
                  <Flag label="writing" on={desk.writing} />
                  <Flag label="desk visible" on={desk.deskVisible} />
                  {desk.conflict && <Flag label="conflict" on />}
                </>
              )}
            </div>
          </MetricSection>
        )}
        {/* Eye Metrics */}
        <MetricSection title="Eye Analysis">
          <MetricRow
            label="Eye Aspect Ratio (EAR)"
            value={faceMetrics?.ear ?? null}
            icon={<Eye className="h-4 w-4" />}
            status={earStatus}
          />
          <MetricRow
            label="PERCLOS"
            value={faceMetrics?.perclos ?? null}
            unit="%"
            icon={<Clock className="h-4 w-4" />}
            status={perclosStatus}
            showProgress
            progressValue={faceMetrics?.perclos ?? 0}
          />
          <MetricRow
            label="Blink Rate"
            value={faceMetrics?.blinkRate ?? null}
            unit="/min"
            icon={<Activity className="h-4 w-4" />}
            status={
              (faceMetrics?.blinkRate ?? 0) >= 12 && (faceMetrics?.blinkRate ?? 0) <= 20
                ? 'good'
                : (faceMetrics?.blinkRate ?? 0) > 0 ? 'warning' : 'neutral'
            }
          />
          {faceMetrics?.blinkDetected && (
            <div className="flex items-center gap-2 text-xs text-accent animate-pulse">
              <Zap className="h-3 w-3" />
              <span>Blink detected!</span>
            </div>
          )}
        </MetricSection>

        {/* Gaze Metrics */}
        <MetricSection title="Gaze Direction">
          <div className="grid grid-cols-2 gap-2">
            <MetricRow
              label="Horizontal"
              value={faceMetrics?.gazeX ?? null}
              icon={<Target className="h-4 w-4" />}
              status={Math.abs(faceMetrics?.gazeX ?? 0) <= 0.3 ? 'good' : 'warning'}
            />
            <MetricRow
              label="Vertical"
              value={faceMetrics?.gazeY ?? null}
              icon={<Target className="h-4 w-4" />}
              status={Math.abs(faceMetrics?.gazeY ?? 0) <= 0.3 ? 'good' : 'warning'}
            />
          </div>
        </MetricSection>

        {/* Head Pose */}
        <MetricSection title="Head Pose">
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="text-center">
              <div className="text-muted-foreground">Yaw</div>
              <div className="font-mono font-medium">
                {faceMetrics?.yaw?.toFixed(1) ?? 'N/A'}°
              </div>
            </div>
            <div className="text-center">
              <div className="text-muted-foreground">Pitch</div>
              <div className="font-mono font-medium">
                {faceMetrics?.pitch?.toFixed(1) ?? 'N/A'}°
              </div>
            </div>
            <div className="text-center">
              <div className="text-muted-foreground">Roll</div>
              <div className="font-mono font-medium">
                {faceMetrics?.roll?.toFixed(1) ?? 'N/A'}°
              </div>
            </div>
          </div>
          <MetricRow
            label="Head Stability"
            value={faceMetrics?.headPoseStability ?? null}
            unit="%"
            icon={<Gauge className="h-4 w-4" />}
            status={stabilityStatus}
            showProgress
            progressValue={faceMetrics?.headPoseStability ?? 0}
          />
        </MetricSection>

        {/* Fatigue & Behavior */}
        <MetricSection title="Engagement Indicators">
          <MetricRow
            label="Fatigue Index"
            value={faceMetrics?.fatigueIndex ?? null}
            unit="%"
            icon={<AlertTriangle className="h-4 w-4" />}
            status={fatigueStatus}
            showProgress
            progressValue={faceMetrics?.fatigueIndex ?? 0}
          />
          <MetricRow
            label="Fidget Score"
            value={faceMetrics?.fidgetScore ?? null}
            unit="%"
            icon={<GitBranch className="h-4 w-4" />}
            status={fidgetStatus}
            showProgress
            progressValue={faceMetrics?.fidgetScore ?? 0}
          />
          <MetricRow
            label="Mouth Openness"
            value={faceMetrics?.mouthOpenness ?? null}
            icon={<Smile className="h-4 w-4" />}
            status={(faceMetrics?.mouthOpenness ?? 0) > 0.3 ? 'warning' : 'good'}
          />
        </MetricSection>
      </CardContent>
    </Card>
  );
}
