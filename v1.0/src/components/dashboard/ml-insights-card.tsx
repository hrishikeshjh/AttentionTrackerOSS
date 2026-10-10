'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import type { MLPrediction, SessionAnalytics } from '@/lib/types';
import type { AnomalyResult } from '@/lib/analytics';
import {
    Brain,
    TrendingUp,
    TrendingDown,
    Minus,
    Sparkles,
    AlertCircle,
    CheckCircle2,
    BarChart3,
    Timer
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { UNSURE_REASON_TEXT } from '@/lib/scoring/dace/abstain';

const LABEL_TEXT: Record<string, string> = {
    attentive_screen: 'Attentive (screen)',
    attentive_reading: 'Attentive (reading / writing)',
    using_phone: 'Using phone',
    distracted: 'Distracted',
    drowsy: 'Drowsy',
    absent: 'Absent',
};

interface MLInsightsCardProps {
    prediction: MLPrediction | null;
    analytics: SessionAnalytics | null;
    /** sharp score drops / outliers (SessionAnalyticsEngine.getAnomalies) */
    anomalies?: AnomalyResult | null;
    sessionActive: boolean;
}

const ConfidenceMeter = ({ confidence }: { confidence: number }) => {
    const percentage = Math.round(confidence * 100);
    const color = confidence >= 0.7 ? 'bg-green-500' : confidence >= 0.4 ? 'bg-yellow-500' : 'bg-red-500';

    return (
        <div className="space-y-1">
            <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Model Confidence</span>
                <span className="font-mono font-medium">{percentage}%</span>
            </div>
            <Progress value={percentage} className="h-2" indicatorClassName={color} />
        </div>
    );
};

const TrendIndicator = ({ trend, strength }: { trend: 'improving' | 'declining' | 'stable'; strength?: number }) => {
    const config = {
        improving: { icon: TrendingUp, color: 'text-green-500', bg: 'bg-green-500/10', label: 'Improving' },
        declining: { icon: TrendingDown, color: 'text-red-500', bg: 'bg-red-500/10', label: 'Declining' },
        stable: { icon: Minus, color: 'text-yellow-500', bg: 'bg-yellow-500/10', label: 'Stable' },
    };

    const { icon: Icon, color, bg, label } = config[trend];

    return (
        <div className={cn('flex items-center gap-2 rounded-md px-3 py-2', bg)}>
            <Icon className={cn('h-5 w-5', color)} />
            <div>
                <div className={cn('font-medium', color)}>{label}</div>
                {strength !== undefined && (
                    <div className="text-xs text-muted-foreground">
                        Strength: {Math.round(strength * 100)}%
                    </div>
                )}
            </div>
        </div>
    );
};

const FeatureBar = ({ name, value, max = 0.2 }: { name: string; value: number; max?: number }) => {
    const percentage = Math.min(100, (Math.abs(value) / max) * 100);
    const isPositive = value >= 0;

    // Format name for display
    const displayName = name
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, str => str.toUpperCase())
        .trim();

    return (
        <div className="space-y-1">
            <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">{displayName}</span>
                <span className={cn(
                    'font-mono',
                    isPositive ? 'text-green-500' : 'text-red-500'
                )}>
                    {isPositive ? '+' : ''}{(value * 100).toFixed(1)}
                </span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                <div
                    className={cn(
                        'h-full rounded-full transition-all',
                        isPositive ? 'bg-green-500' : 'bg-red-500'
                    )}
                    style={{ width: `${percentage}%` }}
                />
            </div>
        </div>
    );
};

const InsightItem = ({
    icon: Icon,
    label,
    value,
    variant = 'default'
}: {
    icon: React.ComponentType<{ className?: string }>;
    label: string;
    value: string | number;
    variant?: 'default' | 'success' | 'warning' | 'danger';
}) => {
    const colors = {
        default: 'text-muted-foreground',
        success: 'text-green-500',
        warning: 'text-yellow-500',
        danger: 'text-red-500',
    };

    return (
        <div className="flex items-center justify-between py-1">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Icon className="h-4 w-4" />
                <span>{label}</span>
            </div>
            <span className={cn('text-sm font-medium', colors[variant])}>
                {value}
            </span>
        </div>
    );
};

export default function MLInsightsCard({ prediction, analytics, anomalies = null, sessionActive }: MLInsightsCardProps) {
    // Get top feature contributions
    const topFeatures = prediction
        ? Object.entries(prediction.featureContributions)
            .filter(([, value]) => Math.abs(value) > 0.01)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 5)
        : [];

    const hasData = prediction !== null;

    return (
        <Card>
            <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                    <Brain className="h-5 w-5 text-primary" />
                    <CardTitle className="text-lg">ML Insights</CardTitle>
                </div>
                <CardDescription>
                    Real-time analysis from the cognitive engagement model.
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                {!sessionActive && !hasData ? (
                    <div className="flex flex-col items-center justify-center py-8 text-center text-muted-foreground">
                        <Sparkles className="h-12 w-12 mb-2 opacity-50" />
                        <p>Start a session to see ML insights</p>
                    </div>
                ) : (
                    <>
                        {/* DACE behaviour label / rule / Unsure reason */}
                        {prediction?.scoringMode === 'dace' && (
                            <div className="space-y-1 rounded-md border px-3 py-2">
                                <div className="flex items-center justify-between text-sm">
                                    <span className="text-muted-foreground">Activity (DACE)</span>
                                    <span className={cn('font-medium', prediction.unsure ? 'text-yellow-500' : '')}>
                                        {prediction.unsure
                                            ? 'Undetermined'
                                            : prediction.daceLabel
                                                ? LABEL_TEXT[prediction.daceLabel] ?? prediction.daceLabel
                                                : '—'}
                                    </span>
                                </div>
                                {prediction.unsure && prediction.unsureReason && (
                                    <div className="text-xs text-muted-foreground">
                                        {prediction.unsureReason}: {UNSURE_REASON_TEXT[prediction.unsureReason]}
                                        {prediction.scoreHeld ? ' · score held, not penalised' : ' · no score yet'}
                                    </div>
                                )}
                                {prediction.daceRule && (
                                    <div className="text-xs text-muted-foreground font-mono">rule: {prediction.daceRule}</div>
                                )}
                            </div>
                        )}

                        {/* Confidence Meter */}
                        {prediction && (
                            <ConfidenceMeter confidence={prediction.confidence} />
                        )}

                        {/* Attention Trend */}
                        {analytics && (
                            <div className="space-y-2">
                                <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                                    Attention Trend
                                </h4>
                                <TrendIndicator
                                    trend={analytics.attentionTrend}
                                    strength={Math.abs(analytics.trendSlope) / 5}
                                />
                            </div>
                        )}

                        {/* Feature Contributions */}
                        {topFeatures.length > 0 && (
                            <div className="space-y-2">
                                <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                                    Top Feature Contributions
                                </h4>
                                <div className="space-y-2">
                                    {topFeatures.map(([name, value]) => (
                                        <FeatureBar key={name} name={name} value={value} />
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Session Insights */}
                        {analytics && (
                            <div className="space-y-2">
                                <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                                    Session Insights
                                </h4>
                                <div className="space-y-1">
                                    <InsightItem
                                        icon={BarChart3}
                                        label="Score Variability"
                                        value={`±${analytics.attentionVariability.toFixed(1)}`}
                                        variant={analytics.attentionVariability < 15 ? 'success' : analytics.attentionVariability < 25 ? 'warning' : 'danger'}
                                    />
                                    <InsightItem
                                        icon={CheckCircle2}
                                        label="Peak Periods"
                                        value={analytics.peakAttentionPeriods.length}
                                        variant={analytics.peakAttentionPeriods.length > 0 ? 'success' : 'default'}
                                    />
                                    <InsightItem
                                        icon={AlertCircle}
                                        label="Low Periods"
                                        value={analytics.lowAttentionPeriods.length}
                                        variant={analytics.lowAttentionPeriods.length === 0 ? 'success' : analytics.lowAttentionPeriods.length < 3 ? 'warning' : 'danger'}
                                    />
                                    {anomalies && (
                                        <InsightItem
                                            icon={TrendingDown}
                                            label="Sharp Score Drops"
                                            value={anomalies.scoreDropCount}
                                            variant={anomalies.scoreDropCount === 0 ? 'success' : anomalies.scoreDropCount < 3 ? 'warning' : 'danger'}
                                        />
                                    )}
                                    {analytics.fatigueOnset && (
                                        <InsightItem
                                            icon={Timer}
                                            label="Fatigue Detected"
                                            value="Yes"
                                            variant="warning"
                                        />
                                    )}
                                </div>
                            </div>
                        )}

                        {/* Raw Score */}
                        {prediction && (
                            <div className="pt-2 border-t">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm text-muted-foreground">Raw ML Score</span>
                                    <span className={cn(
                                        'text-2xl font-bold',
                                        prediction.rawScore >= 70 ? 'text-green-500' :
                                            prediction.rawScore >= 40 ? 'text-yellow-500' : 'text-red-500'
                                    )}>
                                        {prediction.unsure && !prediction.scoreHeld ? '—' : prediction.rawScore}
                                    </span>
                                </div>
                            </div>
                        )}
                    </>
                )}
            </CardContent>
        </Card>
    );
}
