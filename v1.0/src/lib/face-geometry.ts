/**
 * @deprecated import path — the actual implementation lives in
 * `./scoring/` (the web app's single source of truth; the Chrome extension
 * and v2.0's Python port keep hand-maintained copies, see scoring/VERSION.ts).
 * This file is kept only so existing `@/lib/face-geometry`
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
