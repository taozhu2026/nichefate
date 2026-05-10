# DARLIN-01E Execution Summary

Timestamp: 2026-05-10T11:26:47Z

## Result

Final decision category: `ta_failed_environment`

TA/Tigre was launched as real official DARLIN Part1 preprocessing from scratch staging. The run was interrupted after the first clear tool-wrapper failure.

RA/Rosa was not run because TA did not pass the execution gate.

## Selected Environment

`darlin-repro` was selected because it provides Snakemake, PEAR, FastQC, and MultiQC. MATLAB remains unavailable.

## Safety

- Original official code roots unchanged: `True`
- Scratch containment passed: `true`
- `/ssd` write detected: `false`
- Raw FASTQs copied: `false`
- Raw FASTQs modified: `False`
- Sailu-ST run: `false`
- NicheFate run: `false`

## Exact Next Step

Patch scratch run_pear.sh to use short PEAR options or compatible PEAR, remove/module-guard run_fastqc.sh module load, clean incomplete TA scratch outputs, then rerun TA only.
