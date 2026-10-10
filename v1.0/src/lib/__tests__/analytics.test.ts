import { describe, it, expect } from 'vitest';
import { SessionAnalyticsEngine, detectAnomalies } from '../analytics';
import type { SessionDataPoint } from '../types';

const pts = (scores: number[], extra: Partial<SessionDataPoint> = {}): SessionDataPoint[] =>
  scores.map((s, i) => ({ timestamp: 1000 * i, attentionScore: s, attentionState: 'Attentive', ...extra }));

describe('analytics anomalies', () => {
  const scores = [80, 82, 81, 40, 80, 79, 81, 10, 80, 81, 82, 80];

  it('detectAnomalies reports sharp score drops with timestamps (not blinks)', () => {
    const r = detectAnomalies(pts(scores));
    expect(r.scoreDropCount).toBe(2);
    expect(r.scoreDropTimestamps).toEqual([3000, 7000]);
    expect(r.outlierTimestamps).toEqual([7000]);
  });

  it('analyze().blinkAnomalies keeps holding the score-drop count (compatibility)', () => {
    const e = new SessionAnalyticsEngine();
    e.setData(pts(scores));
    expect(e.analyze().blinkAnomalies).toBe(e.getAnomalies().scoreDropCount);
  });

  it('getAnomalies excludes DACE Unsure points and needs 10 points', () => {
    const e = new SessionAnalyticsEngine();
    e.setData([...pts(scores.slice(0, 9)), { timestamp: 99_000, attentionScore: 0, attentionState: 'Idle', unsure: true }]);
    expect(e.getAnomalies()).toEqual({ scoreDropCount: 0, scoreDropTimestamps: [], outlierTimestamps: [] });
  });
});
