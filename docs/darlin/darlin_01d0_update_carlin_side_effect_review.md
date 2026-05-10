# DARLIN-01D0 update_CARLIN_dir Side-Effect Review

Timestamp: 2026-05-10T03:51:14Z

## Decision Context

DARLIN-01C staged Meiji E1 RA/TA configs and symlinks, but did not run Snakemake because `snakefile_matlab_DARLIN_Part1.py` calls `update_CARLIN_dir` at parse time.

## Key Finding

The official `darlin.settings` file sets `CARLIN_dir` to a sibling `CARLIN_pipeline` directory relative to whichever `darlin` Python package is imported. In the current `snakemake_darlin` environment, importing `darlin` without containment resolves to the original source tree:

`/home/zhutao/projects/darlin_cell_repro/code/snakemake_DARLIN/darlin/__init__.py`

With the 01D0 `PYTHONPATH` containment, importing `darlin` resolves to:

`/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01d0/code_copy/snakemake_DARLIN/darlin/__init__.py`

This changes the parse-time `rsync` target from the original code tree to:

`/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01d0/code_copy/CARLIN_pipeline/`

## Risk Summary

- `update_CARLIN_dir` performs template-specific `rsync` during Snakefile parsing.
- `Tigre_2022_v2` would create or update `Tigre_CARLIN_2022_v2` under the selected `CARLIN_pipeline` root.
- `Rosa_v2` would create or update `Rosa_CARLIN_v2` under the selected `CARLIN_pipeline` root.
- Part1 also has a parse-time cleanup of stale `DARLIN_analysis.done` files under the current working directory.
- Rule-time PEAR, FastQC, and MATLAB commands remain forbidden outside dry-run.

## Containment Strategy

Use the scratch-copy Snakefile and force the scratch-copy Python package with `PYTHONPATH=/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01d0/code_copy/snakemake_DARLIN`. Run future dry-runs only from the 01C scratch staging folders.

## Approval Result

The side-effect target is resolved for dry-run-only execution because parse-time CARLIN template copies are redirected to the 01D0 scratch copy, and output paths remain under 01C scratch staging.
