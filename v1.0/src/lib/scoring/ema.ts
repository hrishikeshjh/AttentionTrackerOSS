export class ExponentialMovingAverage {
  private alpha: number;
  private value: number | null = null;

  constructor(alpha = 0.15) {
    this.alpha = alpha;
  }

  update(newValue: number): number {
    this.value = this.value === null ? newValue : this.alpha * newValue + (1 - this.alpha) * this.value;
    return this.value;
  }

  getValue(): number | null {
    return this.value;
  }

  reset(): void {
    this.value = null;
  }
}
