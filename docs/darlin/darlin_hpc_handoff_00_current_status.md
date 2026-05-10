# DARLIN-HPC-Handoff-00 Current Server Status

Timestamp: 2026-05-10T11:46:15Z

## Project Goal

Move Meiji E1 RA/TA official DARLIN barcode preprocessing from the current server to the HPC. The current server reached official workflow execution but failed on local environment/tool-wrapper behavior before meaningful DARLIN output was produced.

This handoff covers only the DARLIN barcode preprocessing track. Sailu-ST, NicheFate, and barcode plus ST integration remain out of scope.

## Current Barcode Samples

| sample_id | corrected locus | official family | status |
|---|---:|---|---|
| `Brain_E1_all_TA` | `TA / TC` | Tigre | current preprocessing target |
| `Brain_E1_all_RA` | `RA / RC` | Rosa / Rosa26 | current preprocessing target after TA passes |
| `Brain_E1_all_CA` | `CA / CC` | Col1a1 / cCARLIN | absent/deferred |

The deprecated wrong mapping `RA=Tigre / TA=Rosa` must not be used.

## Protocol-Resolved Config Choices

| sample_id | cfg_type | template | read_cutoff_UMI_override |
|---|---|---|---|
| `Brain_E1_all_TA` | `BulkRNA_Tigre_14UMI` | `Tigre_2022_v2` | `[3, 10]` |
| `Brain_E1_all_RA` | `BulkRNA_Rosa_14UMI` | `Rosa_v2` | `[3, 10]` |

The prior DARLIN-02B cfg_type/template uncertainty is resolved for the current RA/TA samples by the Nature Protocols DARLIN parameter table.

## Current Server Staging Structure

01C created current-server scratch staging under:

`/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01c/`

Staged run folders:

- `TA_Tigre/`
- `RA_Rosa/`

The 01C staging report records symlink-only raw FASTQ layout. No raw FASTQs were copied or modified.

The confirmed official staged FASTQ names from 01C are:

| run folder | staged raw_fastq symlink |
|---|---|
| `TA_Tigre` | `raw_fastq/Brain_E1_all_TA_R1.fastq.gz` |
| `TA_Tigre` | `raw_fastq/Brain_E1_all_TA_R2.fastq.gz` |
| `RA_Rosa` | `raw_fastq/Brain_E1_all_RA_R1.fastq.gz` |
| `RA_Rosa` | `raw_fastq/Brain_E1_all_RA_R2.fastq.gz` |

01D0 created a scratch code copy under:

`/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01d0/code_copy/`

This copy contained possible `update_CARLIN_dir` / `rsync` side effects away from the original official source trees.

## Current Server Execution Status

01E attempted real official DARLIN preprocessing for `TA_Tigre` only. `RA_Rosa` was not run.

Selected environment:

`darlin-repro`

TA command status:

- exit code: `120`
- decision category: `ta_failed_environment`
- log path: `/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01e/logs/TA_Tigre_run.log`

Observed early failures:

- PEAR rejected `--min-assembly-length`.
- `run_fastqc.sh` failed at `module load fastqc` because `module` was unavailable.

MATLAB/CARLIN analysis was not reached.

## Failure Diagnosis

The 01E failure is best interpreted as current-server environment/tool-wrapper mismatch, not a biological sample, corrected mapping, cfg_type, or template failure.

Specific blockers on the current server:

- The available PEAR executable is incompatible with the official wrapper option spelling.
- The official FastQC wrapper expects an HPC-style module system.
- MATLAB is not available on the current `PATH`.
- Real preprocessing would require patching scratch wrappers or installing a compatible official toolchain, which is less defensible than moving to HPC.

## Why HPC Is Recommended

HPC is the appropriate continuation target because it is expected to provide:

- module system support for wrapper scripts;
- official or lab-standard PEAR, FastQC, MultiQC, and Snakemake environments;
- MATLAB with required toolboxes;
- scratch/project storage appropriate for workflow outputs;
- a cleaner provenance boundary than patching current-server scratch wrappers.

## What Must Not Be Reused From Current Server Scratch

Do not reuse current-server runtime artifacts on HPC:

- `TA_Tigre/DARLIN/`
- `TA_Tigre/pear_output/`
- `TA_Tigre/fastqc_before_pear/`
- `TA_Tigre/.snakemake/`
- `/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01e/logs/`
- the 01D0 scratch code copy as an execution source on HPC

The repo-tracked reports and config decisions may be used as audit documentation. The HPC run should build fresh staging and use a fresh official DARLIN code checkout or approved code copy on HPC.

## Exact Next Steps On HPC

1. Locate or transfer the four Meiji RA/TA raw FASTQs and provider metadata.
2. Verify raw FASTQ identity using provider `raw_md5.txt` or newly computed checksums.
3. Confirm module/tool availability: `snakemake`, `pear`, `fastqc`, `multiqc`, `matlab`.
4. Confirm MATLAB includes Bioinformatics Toolbox and Image Processing Toolbox.
5. Build a fresh HPC staging root under an HPC scratch/project path, not `/ssd`.
6. Create symlinks only, using the official staged names confirmed in 01C.
7. Write fresh `config.yaml` files using the protocol-resolved parameters above.
8. Run `TA_Tigre` first.
9. Run `RA_Rosa` only after `TA_Tigre` passes environment/tool setup and produces expected outputs.
