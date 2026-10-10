import { describe, it, expect } from 'vitest';
import { normalizeLabel, parseCvatTags, tagsToWindows, type GtWindow } from '../eval/cvat';
import { computeMetrics, metricsReport, predictionFor, windowRecords, type PredRecord } from '../eval/metrics';

const XML = `<?xml version="1.0"?>
<annotations>
  <tag frame="0" label="reading" source="manual"><attribute name="segment">1</attribute><attribute name="ambiguous">false</attribute></tag>
  <tag frame="30" label="attentive_reading" source="manual"><attribute name="segment">1</attribute></tag>
  <tag frame="95" label="phone" source="manual"><attribute name="segment">4</attribute><attribute name="ambiguous">true</attribute></tag>
  <tag frame="100" label="Using Phone" source="manual"><attribute name="segment">4</attribute></tag>
  <tag frame="200" label="banana" source="manual"/>
  <tag frame="300" label="distracted" source="manual"><attribute name="segment">10</attribute></tag>
</annotations>`;

describe('CVAT -> 3 s windows', () => {
  it('parses tags + attributes, maps aliases, majority label, ambiguous flag, skips unknown labels', () => {
    const tags = parseCvatTags(XML);
    expect(tags).toHaveLength(6);
    expect(tags[2]).toEqual({ frame: 95, label: 'phone', attributes: { segment: '4', ambiguous: 'true' } });
    expect(normalizeLabel('Using Phone')).toBe('using_phone');
    expect(normalizeLabel('writing')).toBe('attentive_reading');
    const { windows, unknownLabels } = tagsToWindows(tags, { fps: 30, recording: 'r1', person: 'p1', setup: 's1' });
    expect(unknownLabels).toEqual(['banana']);
    expect(windows.map((w) => [w.start_ms, w.label, w.segment, w.ambiguous, w.n])).toEqual([
      [0, 'attentive_reading', '1', false, 2],
      [3000, 'using_phone', '4', true, 2],
      [9000, 'distracted', '10', false, 1],
    ]);
  });
});

const gw = (start: number, label: GtWindow['label'], o: Partial<GtWindow> = {}): GtWindow => ({
  recording: 'r', person: 'p1', setup: 's', segment: '1', start_ms: start, end_ms: start + 3000, label, ambiguous: false, n: 10, ...o,
});
const pr = (t: number, label: PredRecord['label'], unsure = false, latencyTotal = 50): PredRecord => ({ t, label, unsure, latencyTotal });

describe('pilot metrics (D.5)', () => {
  it('takes the last window record inside [start, end)', () => {
    const preds = [pr(0, 'distracted'), pr(2800, 'attentive_reading'), pr(3000, 'drowsy')];
    expect(predictionFor({ start_ms: 0, end_ms: 3000 }, preds)).toBe('attentive_reading');
    expect(predictionFor({ start_ms: 6000, end_ms: 9000 }, preds)).toBeNull();
    expect(predictionFor({ start_ms: 0, end_ms: 3000 }, [pr(100, null, true)])).toBe('unsure');
  });

  it('FDR-DA (all / answered), phone recall + FNR, Unsure on desk-out, answer rate, ambiguity agreement', () => {
    const gt = [
      gw(0, 'attentive_reading'),
      gw(3000, 'attentive_reading'),
      gw(6000, 'attentive_reading'),
      gw(9000, 'attentive_reading'),
      gw(12000, 'using_phone'),
      gw(15000, 'using_phone'),
      gw(18000, 'distracted', { segment: '10', ambiguous: true }),
      gw(21000, 'attentive_screen', { segment: '10' }),
      gw(24000, 'drowsy'),
    ];
    const preds = [
      pr(2900, 'attentive_reading'),
      pr(5900, 'distracted'), // false distraction
      pr(8900, null, true), // unsure: not a false distraction, not answered
      pr(11900, 'using_phone'), // false distraction
      pr(14900, 'using_phone'),
      pr(17900, null, true), // phone missed by Unsure
      pr(20900, null, true),
      pr(23900, 'attentive_screen', false, 400),
      // drowsy window missing
    ];
    const m = computeMetrics(gt, preds);
    expect(m.missing).toBe(1);
    expect(m.readingWindows).toBe(4);
    expect(m.fdrDa).toBeCloseTo(2 / 4, 12);
    expect(m.fdrDaAnswered).toBeCloseTo(2 / 3, 12);
    expect(m.phoneRecall).toBeCloseTo(1 / 2, 12);
    expect(m.phoneFnr).toBeCloseTo(1 / 2, 12);
    expect(m.unsureRateDeskOut).toBeCloseTo(1 / 2, 12);
    expect(m.answerRate).toBeCloseTo(5 / 8, 12);
    expect(m.unsureAmbiguousPrecision).toBeCloseTo(1 / 3, 12);
    expect(m.unsureAmbiguousRecall).toBeCloseTo(1, 12);
    expect(m.drowsyRecall).toBeNull();
    const rep = metricsReport(gt, preds);
    expect(Object.keys(rep.bySegment)).toEqual(['1', '10']);
    expect(rep.latency).toMatchObject({ n: 8, p95: 400, max: 400 });
  });

  it('reads window records from parsed eval logs', () => {
    const recs = [
      { type: 'window', t: 200, label: 'drowsy', unsure: false, latency: { total: 12 } },
      { type: 'prediction', t: 200 },
      { type: 'window', t: 0, label: null, unsure: true, reason: 'U2' },
    ];
    expect(windowRecords(recs)).toEqual([
      { t: 0, label: null, unsure: true, reason: 'U2', latencyTotal: null },
      { t: 200, label: 'drowsy', unsure: false, reason: null, latencyTotal: 12 },
    ]);
  });
});
