# DARLIN-01C Staging Report

Timestamp: 2026-05-10T03:40:07Z

Branch: `darlin-onboarding`

## Scope

This node created protocol-resolved throwaway scratch staging for Meiji E1 DARLIN barcode dry-run preparation only. It did not run Snakemake, DARLIN preprocessing, MATLAB, PEAR, FastQC, MultiQC, Sailu-ST, or NicheFate.

## Staging Root

Scratch root:

`/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01c/`

The scratch root was absent at the start of DARLIN-01C and was created under `/home/zhutao/scratch`, not `/ssd`.

## TA / Tigre

Run folder:

`/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01c/TA_Tigre/`

Config:

- `SampleList: [Brain_E1_all_TA]`
- `cfg_type: BulkRNA_Tigre_14UMI`
- `template: Tigre_2022_v2`
- `read_cutoff_UMI_override: [3, 10]`
- `CARLIN_memory_factor: 100`
- `sbatch: 0`
- `CARLIN_max_run_time: 12`

Symlinks:

- `raw_fastq/Brain_E1_all_TA_R1.fastq.gz` -> `/data/zhutao/nichefate_data/MeiJi/Brain_E1_all_TA.R1.raw.fastq.gz`
- `raw_fastq/Brain_E1_all_TA_R2.fastq.gz` -> `/data/zhutao/nichefate_data/MeiJi/Brain_E1_all_TA.R2.raw.fastq.gz`

## RA / Rosa

Run folder:

`/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01c/RA_Rosa/`

Config:

- `SampleList: [Brain_E1_all_RA]`
- `cfg_type: BulkRNA_Rosa_14UMI`
- `template: Rosa_v2`
- `read_cutoff_UMI_override: [3, 10]`
- `CARLIN_memory_factor: 100`
- `sbatch: 0`
- `CARLIN_max_run_time: 12`

Symlinks:

- `raw_fastq/Brain_E1_all_RA_R1.fastq.gz` -> `/data/zhutao/nichefate_data/MeiJi/Brain_E1_all_RA.R1.raw.fastq.gz`
- `raw_fastq/Brain_E1_all_RA_R2.fastq.gz` -> `/data/zhutao/nichefate_data/MeiJi/Brain_E1_all_RA.R2.raw.fastq.gz`

## Naming Decision

The scratch symlink names use the official Snakefile contract:

`raw_fastq/{sample}_R1.fastq.gz` and `raw_fastq/{sample}_R2.fastq.gz`

The earlier L001-style plan names were not created because `SampleList: [Brain_E1_all_TA]` and `SampleList: [Brain_E1_all_RA]` would make the official Snakefile look for the non-L001 filenames.

## Safety Notes

- Raw FASTQs were not copied.
- Raw FASTQs were not modified.
- Only scratch directories, scratch configs, and scratch symlinks were created.
- Existing untracked DARLIN/Sailu audit files were not modified.
