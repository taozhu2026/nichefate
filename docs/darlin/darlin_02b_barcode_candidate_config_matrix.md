# DARLIN-02B Barcode Candidate Config Matrix

Timestamp: 2026-05-09T13:06:39Z

Branch: `darlin-onboarding`

## Scope

This node prepares a candidate official DARLIN configuration matrix for the Meiji E1 barcode samples only.

Corrected mapping used:

- `Brain_E1_all_TA` → `Tigre` candidate.
- `Brain_E1_all_RA` → `Rosa` candidate.
- `Brain_E1_all_CA` → absent/deferred `Col1a1/cCARLIN` candidate.

The deprecated mapping `RA=Tigre / TA=Rosa` was not used.

## Official Evidence

Local code/config inspection found:

- Official templates: `Tigre`, `Tigre_2022`, `Tigre_2022_v2`, `Rosa`, `Rosa_v2`, `cCARLIN`.
- Tigre cfg files: `BulkRNA_Tigre`, `BulkDNA_Tigre`, `BulkRNA_Tigre_10UMI`, `BulkDNA_Tigre_10UMI`, `BulkRNA_Tigre_12UMI`, `BulkDNA_Tigre_12UMI`, `BulkRNA_Tigre_14UMI`, `BulkDNA_Tigre_14UMI`.
- Rosa cfg files: `BulkRNA_Rosa_12UMI`, `BulkDNA_Rosa_12UMI`, `BulkRNA_Rosa_14UMI`, `BulkDNA_Rosa_14UMI`.
- The README states short-primer templates include `Tigre_2022_v2` and `Rosa_v2`; long-primer templates include `Tigre_2022` and `Rosa`.
- The legacy `/data/zhutao/darlin_data/config.yaml` uses `BulkRNA_Tigre_14UMI` and `Tigre_2022_v2`, but it is not Meiji E1 evidence.

## Matrix Interpretation

The matrix is a candidate planning artifact, not production selection.

All candidates are marked conditional because these remain unresolved:

- exact long/short primer template variant
- exact DNA/RNA and UMI-length `cfg_type`
- `read_cutoff_UMI_override`
- safe staging root and symlink policy
- MATLAB/Custom_CARLIN execution policy
- Snakemake parse-time `rsync`/template-copy side effects

## Output Contract

Official Part1 expects:

- `config.yaml` in the working directory
- `raw_fastq/{sample}_R1.fastq.gz`
- `raw_fastq/{sample}_R2.fastq.gz`

Expected output root in a later dry-run-only staging node:

- `/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta/`

No staging directory, config file outside the repo, or symlink was created in DARLIN-02B.
