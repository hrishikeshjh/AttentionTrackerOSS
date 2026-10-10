#!/usr/bin/env node
// postinstall: adapt genkit's built-in telemetry to OpenTelemetry SDK 2.x.
//
// package.json `overrides` force the OpenTelemetry JS SDK onto the 2.x stable /
// 0.2xx experimental line to clear GHSA-8988-4f7v-96qf (@opentelemetry/core
// < 2.8.0). genkit 1.x (latest 1.42.0, and upstream main) is still written
// against OTel 1.x / 0.52 and hits two breaking changes in the 2.x SDK:
//
//  1. lib/tracing/exporter.{js,mjs} (TraceServerExporter, feeds the genkit Dev
//     UI) reads span.parentSpanId and span.instrumentationLibrary; 2.x renamed
//     them to span.parentSpanContext.spanId and span.instrumentationScope.
//     Unpatched, every span arrives as a root span (flat traces).
//  2. lib/tracing/node-telemetry-provider.{js,mjs} constructs
//     Simple/BatchLogRecordProcessor(exporter); sdk-logs 0.2xx takes
//     ({ exporter }). Unpatched, genkit log records are silently dropped.
//
// Both edits are textual, idempotent (marker comment) and fail the install
// loudly if genkit's code changes shape, so a genkit upgrade cannot silently
// lose them. Remove this script once genkit supports OTel 2.x natively.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const MARKER = '/* cognen:otel2-compat */';

const EDITS = {
  'exporter.js': [['_exportInfo(span) {', 1, spanShim]],
  'exporter.mjs': [['_exportInfo(span) {', 1, spanShim]],
  'node-telemetry-provider.js': [
    ['SimpleLogRecordProcessor(logExporter)', 1, (s) => s.replace('(logExporter)', `(${MARKER} { exporter: logExporter })`)],
    ['BatchLogRecordProcessor(logExporter)', 1, (s) => s.replace('(logExporter)', `(${MARKER} { exporter: logExporter })`)],
  ],
  'node-telemetry-provider.mjs': [
    ['SimpleLogRecordProcessor(logExporter)', 1, (s) => s.replace('(logExporter)', `(${MARKER} { exporter: logExporter })`)],
    ['BatchLogRecordProcessor(logExporter)', 1, (s) => s.replace('(logExporter)', `(${MARKER} { exporter: logExporter })`)],
  ],
};

function spanShim(anchor) {
  return (
    `${anchor}\n    ${MARKER} ` +
    'if (!("parentSpanId" in span)) Object.defineProperty(span, "parentSpanId", { value: span.parentSpanContext?.spanId, configurable: true }); ' +
    'if (!("instrumentationLibrary" in span)) Object.defineProperty(span, "instrumentationLibrary", { value: span.instrumentationScope, configurable: true });'
  );
}

const require = createRequire(path.join(process.cwd(), 'package.json'));
let coreEntry;
try {
  coreEntry = require.resolve('@genkit-ai/core');
} catch {
  console.log('[patch-genkit-otel2] @genkit-ai/core not installed; nothing to do.');
  process.exit(0);
}
const tracingDir = path.join(path.dirname(coreEntry), 'tracing');

let patched = 0;
for (const [file, edits] of Object.entries(EDITS)) {
  const p = path.join(tracingDir, file);
  let src = fs.readFileSync(p, 'utf8');
  if (src.includes(MARKER)) continue;
  for (const [anchor, expected, replace] of edits) {
    const count = src.split(anchor).length - 1;
    if (count !== expected) {
      console.error(
        `[patch-genkit-otel2] expected ${expected} occurrence(s) of "${anchor}" in ${p}, found ${count}. ` +
          'genkit changed its telemetry code: re-check OpenTelemetry 2.x compatibility and update or remove this script.'
      );
      process.exit(1);
    }
    src = src.replace(anchor, replace(anchor));
  }
  fs.writeFileSync(p, src);
  patched++;
}
console.log(`[patch-genkit-otel2] ${patched ? `patched ${patched} file(s)` : 'already patched'} in ${tracingDir}`);
