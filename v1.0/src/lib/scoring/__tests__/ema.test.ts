import { describe, it, expect } from 'vitest';
import { ExponentialMovingAverage } from '../ema';

describe('ExponentialMovingAverage', () => {
  it('does not lag on the first sample (initializes directly to the first value)', () => {
    const ema = new ExponentialMovingAverage(0.15);
    expect(ema.update(80)).toBe(80);
  });

  it('converges toward a constant input over repeated updates', () => {
    const ema = new ExponentialMovingAverage(0.15);
    ema.update(0);
    let value = 0;
    for (let i = 0; i < 50; i++) value = ema.update(100);
    expect(value).toBeGreaterThan(99);
  });

  it('blends according to alpha on a single step', () => {
    const ema = new ExponentialMovingAverage(0.2);
    ema.update(50);
    const next = ema.update(100);
    expect(next).toBeCloseTo(0.2 * 100 + 0.8 * 50, 6);
  });

  it('reset() clears state so the next update behaves like the first sample', () => {
    const ema = new ExponentialMovingAverage(0.15);
    ema.update(80);
    ema.reset();
    expect(ema.update(20)).toBe(20);
  });
});
