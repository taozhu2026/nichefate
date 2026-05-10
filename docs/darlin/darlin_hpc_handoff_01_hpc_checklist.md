# DARLIN-HPC-Handoff-01 HPC Checklist

Timestamp: 2026-05-10T11:46:15Z

## Data To Locate Or Transfer

Required raw FASTQs:

- `/data/zhutao/nichefate_data/MeiJi/Brain_E1_all_TA.R1.raw.fastq.gz`
- `/data/zhutao/nichefate_data/MeiJi/Brain_E1_all_TA.R2.raw.fastq.gz`
- `/data/zhutao/nichefate_data/MeiJi/Brain_E1_all_RA.R1.raw.fastq.gz`
- `/data/zhutao/nichefate_data/MeiJi/Brain_E1_all_RA.R2.raw.fastq.gz`

Recommended provider metadata:

- `/data/zhutao/nichefate_data/MeiJi/raw_md5.txt`
- `/data/zhutao/nichefate_data/MeiJi/others/rawdata.stat`
- `/data/zhutao/nichefate_data/MeiJi/others/report.pdf`

Recommended repo source:

- local repo: `/home/zhutao/projects/nichefate`
- GitHub remote: `https://github.com/taozhu2026/nichefate.git`

## Expected Staged FASTQ Names

Use a fresh HPC staging root and symlink raw FASTQs into each run folder.

TA/Tigre:

- `TA_Tigre/raw_fastq/Brain_E1_all_TA_R1.fastq.gz`
- `TA_Tigre/raw_fastq/Brain_E1_all_TA_R2.fastq.gz`

RA/Rosa:

- `RA_Rosa/raw_fastq/Brain_E1_all_RA_R1.fastq.gz`
- `RA_Rosa/raw_fastq/Brain_E1_all_RA_R2.fastq.gz`

These are symlink names only. Do not rename, copy, or modify raw FASTQs.

## MD5 Verification

Before running preprocessing on HPC:

1. Transfer or locate `raw_md5.txt`.
2. Verify each raw FASTQ against provider checksums if available.
3. If provider checksums are unavailable on HPC, compute and record new checksums before workflow execution.
4. Keep checksum records with the HPC run report.

## Required Tools

Confirm availability before launching Snakemake:

- `snakemake`
- `pear`
- `fastqc`
- `multiqc`
- `matlab`
- MATLAB Bioinformatics Toolbox
- MATLAB Image Processing Toolbox

## Environment Checks

Run these on HPC before preprocessing:

```bash
module avail matlab
module load matlab
which matlab
matlab -batch "ver"
pear --help
fastqc --version
multiqc --version
snakemake --version
```

If the official DARLIN environment is module-based, record the exact `module load` commands in the HPC run report.

## Output Path Safety

Use only HPC scratch or project storage for staging and outputs.

Required safety constraints:

- no `/ssd` output paths;
- no raw FASTQ modification;
- no raw FASTQ copying unless the HPC data-transfer plan explicitly requires one-time transfer from current server to HPC storage;
- symlink raw FASTQs into `raw_fastq/`;
- keep workflow outputs inside the HPC staging root;
- do not run Sailu-ST;
- do not run NicheFate;
- do not integrate barcode and ST outputs.

## Current Server Artifacts To Ignore

Do not transfer current-server partial runtime outputs as inputs:

- `TA_Tigre/DARLIN/`
- `TA_Tigre/pear_output/`
- `TA_Tigre/fastqc_before_pear/`
- `TA_Tigre/.snakemake/`
- 01E failed-run logs
- 01D0 scratch code copy
