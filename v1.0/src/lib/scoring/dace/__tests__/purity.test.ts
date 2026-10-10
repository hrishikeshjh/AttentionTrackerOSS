/** scoring/dace must stay pure and node-testable: no DOM, no MediaPipe/ONNX, not on the legacy import path. */
import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const sources = readdirSync(dir).filter((f) => f.endsWith('.ts'));

describe('scoring/dace purity', () => {
  it.each(sources)('%s has no DOM / browser-runtime dependencies', (f) => {
    const src = readFileSync(join(dir, f), 'utf8');
    expect(src).not.toMatch(/['"]use client['"]/);
    expect(src).not.toMatch(/(from|import\()\s*['"](@mediapipe|onnxruntime|face-api)/);
    expect(src).not.toMatch(/\b(window|document|navigator|localStorage)\./);
    expect(src).not.toMatch(/from '\.\.\/\.\.\/(hand-tracker|object-detector|temporal-classifier|ml-engine)'/);
  });

  it('is not re-exported from scoring/index.ts', () => {
    const idx = readFileSync(resolve(dir, '..', 'index.ts'), 'utf8');
    expect(idx).not.toMatch(/dace/);
  });
});
