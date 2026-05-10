# DARLIN-02B Protocol-Resolved Snakemake Safety Review

## Scope

This review covers future DARLIN-01C dry-run-only staging for Meiji E1 RA/TA barcode preprocessing. No Snakemake command was executed here.

## Main Risk

The official snakefile may perform template-copy or rsync-like setup during parse or dry-run preparation. Therefore, even dry-run-only execution should be treated as side-effect capable until proven otherwise.

## Required Containment

- Use a throwaway scratch root: `/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01c/`.
- Use separate working directories: `TA_Tigre/` and `RA_Rosa/`.
- Use explicit working directory before the first command.
- Set `TMPDIR`, `TMP`, and `TEMP` to `/home/zhutao/tmp` or a scratch-local temp directory.
- Inspect the final config before running any command.
- Keep all outputs under `/home/zhutao/scratch/nichefate/`.
- Do not write to `/ssd`.

## Status

The side-effect risk has been reviewed, but no dry-run has been executed. DARLIN-01C must still verify the dependency environment and MATLAB/Custom_CARLIN policy before running any Snakemake command.
