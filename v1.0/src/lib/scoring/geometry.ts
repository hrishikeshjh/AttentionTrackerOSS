export type Point2D = { x: number; y: number };

export const distance = (p1: Point2D, p2: Point2D): number => {
  return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
};

export const calculateVariance = (values: number[]): number => {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return values.reduce((sum, val) => sum + (val - mean) ** 2, 0) / values.length;
};

export const calculateStdDev = (values: number[]): number => {
  return Math.sqrt(calculateVariance(values));
};

export const clamp = (value: number, min: number, max: number): number => {
  return Math.max(min, Math.min(max, value));
};
