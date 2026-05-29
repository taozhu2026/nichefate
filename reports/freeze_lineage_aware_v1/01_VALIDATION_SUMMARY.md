# Validation Summary

## Checks Passed

- `py_compile` passed for the new lineage and DARLIN facade modules and
  wrapper scripts.
- Focused pytest subset passed: 13/13.
- JSON parsing passed for the new benchmark and freeze summary files.
- TSV parsing passed for the key output manifest.
- H5AD schema acceptance passed on a tiny gene-expression layout fixture.

## Current Status

- The lineage-aware generic facade imports cleanly.
- The benchmark config loads cleanly.
- The claim-language audit is clean for the frozen docs.
- Git staging audit has not been run yet.
