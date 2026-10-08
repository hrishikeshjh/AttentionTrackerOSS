/**
 * @deprecated import path — the actual implementation lives in
 * `./scoring/` (single source of truth, shared with the Chrome extension
 * bundle and kept in sync with v2.0's Python port via golden fixtures, see
 * scoring/EQUIVALENCE.md). This file is kept only so existing `@/lib/face-geometry`
 * imports keep working; prefer importing from `@/lib/scoring` directly in new code.
 */
export * from './scoring/geometry';
export * from './scoring/constants';
export * from './scoring/ear-mar';
export * from './scoring/pose/head-pose';
export * from './scoring/pose/gaze';
export * from './scoring/trackers';
export * from './scoring/basic-metrics';
export * from './scoring/aggregator';
export * from './scoring/face-selection';
