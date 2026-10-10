'use strict';
// Stand-in for extract-zip, wired in via package.json `overrides`.
//
// genkit-cli declares extract-zip ^2.0.1 but never loads it (no require/import
// of it anywhere in genkit-cli, @genkit-ai/tools-common or
// @genkit-ai/telemetry-server). Every extract-zip release (<= 2.0.1) is
// affected by GHSA-7pqw-9j4j-h8q3 and GHSA-jmr9-qjv8-65gv (symlink path
// traversal / arbitrary file write) with no patched version, so the vulnerable
// code is replaced by this stub instead of being installed unused.
// If a future genkit-cli starts calling it, fail loudly rather than extract.
module.exports = async function extractZip() {
  throw new Error(
    'extract-zip is stubbed out in this project (vulnerable, previously unused by genkit-cli). ' +
      'See v1.0/scripts/stubs/extract-zip/index.js and package.json overrides.'
  );
};
