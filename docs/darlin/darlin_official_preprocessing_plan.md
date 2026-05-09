# DARLIN Official Preprocessing Plan

Generated: 2026-05-09T03:01:07Z

## Official / Lab-Standard Source

Use the ShouWen Wang Lab DARLIN stack as the first-line preprocessing route:

- `snakemake_DARLIN`: https://github.com/ShouWenWang-Lab/snakemake_DARLIN
- `Custom_CARLIN`: https://github.com/ShouWenWang-Lab/Custom_CARLIN

Local read-only references inspected:

- `/home/zhutao/projects/darlin_cell_repro/code/snakemake_DARLIN/README.md`
- `/home/zhutao/projects/darlin_cell_repro/code/Custom_CARLIN/README.md`
- `/home/zhutao/projects/darlin_cell_repro/code/snakemake_DARLIN/snakefiles/snakefile_matlab_DARLIN_Part1.py`

## Required Inputs

- Project root containing `config.yaml` and `raw_fastq/`.
- Paired FASTQs for each sample in the format expected by the local snakefile: `raw_fastq/{sample}_R1.fastq.gz` and `raw_fastq/{sample}_R2.fastq.gz`.
- `SampleList` in `config.yaml`, unless `raw_fastq/sample_info.csv` is generated and intentionally used.
- Confirmed `cfg_type` such as `BulkRNA_Tigre_14UMI`, `BulkRNA_Rosa_14UMI`, `BulkRNA_12UMI`, `scCamellia`, or `sc10xV3`.
- Confirmed `template` such as `Tigre_2022_v2`, `Rosa_v2`, `Tigre_2022`, `Rosa`, or `cCARLIN`.
- Read cutoff settings and memory/time parameters.
- For single-cell workflows, a valid transcriptome-derived barcode whitelist or approved built-in barcode reference.

## Expected Outputs

- Per-sample CARLIN/DARLIN outputs under `DARLIN/results_cutoff_override_*/{sample}/`.
- Expected downstream files include `Summary.mat`, `AlleleAnnotations.txt`, `AlleleColonies.txt`, `Results.txt`, `Warnings.txt`, `Log.txt`, per-sample allele tables, and merged reports/tables such as `merge_all/refined_results.csv` when Part2 succeeds.
- NicheFate must consume only standardized clone/barcode tables derived from these official outputs, not raw FASTQ directly.

## Dependency Readiness Probe

| dependency | current_path |
| --- | --- |
| snakemake | not on PATH |
| pear | /opt/miniforge/bin/pear |
| matlab | not on PATH |
| fastqc | not on PATH |
| multiqc | not on PATH |

Conda environment directories observed:

| environment | present |
| --- | --- |
| snakemake_darlin | true |
| darlin-repro | true |

## Current File-Format Readiness

- The official README expects MiSeq-style names like `sample_L001_R1_001.fastq.gz` and `sample_L001_R2_001.fastq.gz`.
- The local inspected snakefile reference in `snakefile_matlab_DARLIN_Part1.py` uses `raw_fastq/{sample}_R1.fastq.gz` and `raw_fastq/{sample}_R2.fastq.gz`.
- `/data/zhutao/darlin_data` has `config.yaml` and `raw_fastq/HSC_R1.fastq.gz`, `HSC_R2.fastq.gz`, `MPP_R1.fastq.gz`, `MPP_R2.fastq.gz`; this is compatible with the local snakefile reference, but it is a legacy/test context and not a matched NicheFate spatial subset.
- `/data/zhutao/nichefate_data/MeiJi` has `Brain_E1_all_RA.R1.raw.fastq.gz` and `Brain_E1_all_TA.R1.raw.fastq.gz` style names; these do not match the local snakefile pattern without a deliberate, reviewed staging/renaming plan.
- Sailu shared directories contain lane-level names such as `Lane01_ZHY_XBB_R_PE100_250_R1.fastq.gz`; these also need a reviewed sample sheet/staging contract before official preprocessing.
- No first-round NicheFate subset currently has confirmed CA/TA/RA demultiplexing, official `cfg_type/template`, and matched expression/spatial ID bridge.

## Manual Confirmations Required Before DARLIN-01

- Which subset is first: B1 CA/TA/RA shared PE100/250, E1 RA/TA plus ST, or another subset.
- Whether CA/TA/RA map to Tigre/Rosa/Col1a1 templates or another lab-standard naming convention for this experiment.
- Whether the shared lane FASTQs are already demultiplexed or need demultiplexing from sample indices not present in filenames.
- Whether a sample sheet exists outside inspected roots and whether FASTQ staging/renaming is allowed for an official dry-run.
- Whether MATLAB, PEAR, Snakemake, FastQC, and MultiQC should be provided through `snakemake_darlin` or another lab-standard environment.
