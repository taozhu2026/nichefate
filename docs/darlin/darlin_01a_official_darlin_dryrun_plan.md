# DARLIN-01A Official DARLIN Dry-Run Plan

Generated: 2026-05-09T03:57:32Z

This is a dry-run readiness plan only. No Snakemake, MATLAB, PEAR, FastQC, MultiQC, or official DARLIN preprocessing was run.

## Official Code Paths

- `snakemake_DARLIN`: `/home/zhutao/projects/darlin_cell_repro/code/snakemake_DARLIN`
- `Custom_CARLIN`: `/home/zhutao/projects/darlin_cell_repro/code/Custom_CARLIN`
- Primary snakefile reference: `/home/zhutao/projects/darlin_cell_repro/code/snakemake_DARLIN/snakefiles/snakefile_matlab_DARLIN_Part1.py`

## Required Inputs

- Reviewed project root for the dry-run.
- `config.yaml` with confirmed `SampleList`, `cfg_type`, `template`, `read_cutoff_UMI_override`, `CARLIN_memory_factor`, `sbatch`, and `CARLIN_max_run_time`.
- FASTQs staged under the official local snakefile pattern: `raw_fastq/{sample}_R1.fastq.gz` and `raw_fastq/{sample}_R2.fastq.gz`.
- DARLIN-01A barcode sample-sheet draft: `configs/darlin/meiji_e1_ra_ta_barcode_sample_sheet_draft.csv`.

DARLIN-01A does not create the project root, copy FASTQs, rename FASTQs, or create symlinks. That staging contract belongs to DARLIN-01B.

## Dependency Status

| dependency | current status |
| --- | --- |
| `snakemake` | not on current PATH |
| `MATLAB` | not on current PATH |
| `PEAR` | `/opt/miniforge/bin/pear` |
| `FastQC` | not on current PATH |
| `MultiQC` | not on current PATH |

## Dry-Run Readiness

Official DARLIN dry-run cannot start now.

Blockers:

- `cfg_type` is unresolved.
- `template` / locus / primer set is unresolved.
- Meiji FASTQ staging and naming contract is unresolved.
- Snakemake environment is not active on the current PATH.
- MATLAB is not available on the current PATH.
- Output root for dry-run artifacts has not been approved.

## DARLIN-01B Target

DARLIN-01B should resolve `cfg_type`, `template`, read cutoff defaults, environment activation, and staging contract. Only after that should an official Snakemake dry-run be considered.
