# DARLIN-01C Next Step Recommendation

## Current Decision

Staging is complete, but Snakemake dry-run is blocked pending explicit side-effect containment for `update_CARLIN_dir` parse-time `rsync`.

## Prepared Future Commands

These commands are prepared for later review only. They were not executed in DARLIN-01C.

TA / Tigre:

```bash
cd /home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01c/TA_Tigre && \
TMPDIR=/home/zhutao/tmp TMP=/home/zhutao/tmp TEMP=/home/zhutao/tmp \
conda run -n snakemake_darlin snakemake \
  -s /home/zhutao/projects/darlin_cell_repro/code/snakemake_DARLIN/snakefiles/snakefile_matlab_DARLIN_Part1.py \
  --configfile config.yaml \
  --cores 1 \
  --config sbatch=0 \
  --dry-run
```

RA / Rosa:

```bash
cd /home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01c/RA_Rosa && \
TMPDIR=/home/zhutao/tmp TMP=/home/zhutao/tmp TEMP=/home/zhutao/tmp \
conda run -n snakemake_darlin snakemake \
  -s /home/zhutao/projects/darlin_cell_repro/code/snakemake_DARLIN/snakefiles/snakefile_matlab_DARLIN_Part1.py \
  --configfile config.yaml \
  --cores 1 \
  --config sbatch=0 \
  --dry-run
```

## Required Approval Before Running

Before running either command, choose one of these containment strategies:

- Approve the official pipeline's parse-time `rsync` into `/home/zhutao/projects/darlin_cell_repro/code/CARLIN_pipeline/`.
- Patch or wrapper-isolate `CARLIN_dir` so template copies land inside a disposable scratch-only location.
- Run from a disposable copy of the official code tree and record all generated files before deleting it.

## Exact Next Step

Decide how to contain the official DARLIN `update_CARLIN_dir` side effect, then run only the prepared Snakemake dry-run command for one locus at a time.
