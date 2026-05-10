# DARLIN-HPC-Handoff-02 Execution Plan

Timestamp: 2026-05-10T11:46:15Z

## Scope

Run official DARLIN barcode preprocessing for Meiji E1 RA/TA on HPC using protocol-resolved parameters. This is not Sailu-ST processing and not NicheFate.

## Build Fresh HPC Staging

Choose an HPC scratch/project root, for example:

`<HPC_SCRATCH_OR_PROJECT>/darlin_meiji_e1_ra_ta_hpc/`

Create:

```text
<HPC_STAGING_ROOT>/
  TA_Tigre/
    raw_fastq/
    output/
    config.yaml
  RA_Rosa/
    raw_fastq/
    output/
    config.yaml
```

Do not reuse current-server scratch runtime artifacts.

## TA/Tigre Config

`TA_Tigre/config.yaml` should contain the official fields confirmed for the current server run:

```yaml
SampleList:
  - Brain_E1_all_TA
cfg_type: BulkRNA_Tigre_14UMI
template: Tigre_2022_v2
read_cutoff_UMI_override:
  - 3
  - 10
```

If HPC requires scheduler fields such as `sbatch`, set them according to the official HPC DARLIN instructions and record the values in the HPC run report.

Expected symlinks:

```text
TA_Tigre/raw_fastq/Brain_E1_all_TA_R1.fastq.gz -> <HPC_RAW_DATA>/Brain_E1_all_TA.R1.raw.fastq.gz
TA_Tigre/raw_fastq/Brain_E1_all_TA_R2.fastq.gz -> <HPC_RAW_DATA>/Brain_E1_all_TA.R2.raw.fastq.gz
```

## RA/Rosa Config

`RA_Rosa/config.yaml` should contain:

```yaml
SampleList:
  - Brain_E1_all_RA
cfg_type: BulkRNA_Rosa_14UMI
template: Rosa_v2
read_cutoff_UMI_override:
  - 3
  - 10
```

Expected symlinks:

```text
RA_Rosa/raw_fastq/Brain_E1_all_RA_R1.fastq.gz -> <HPC_RAW_DATA>/Brain_E1_all_RA.R1.raw.fastq.gz
RA_Rosa/raw_fastq/Brain_E1_all_RA_R2.fastq.gz -> <HPC_RAW_DATA>/Brain_E1_all_RA.R2.raw.fastq.gz
```

## Recommended Run Order

1. Validate raw FASTQ checksums.
2. Validate official DARLIN code location and environment modules.
3. Run tool checks: `snakemake`, `pear`, `fastqc`, `multiqc`, `matlab -batch "ver"`.
4. Run `TA_Tigre` first.
5. Inspect `TA_Tigre` outputs and logs.
6. Run `RA_Rosa` only after TA passes environment/tool setup and produces expected outputs.
7. Keep each run log under the HPC staging root.

## Failure Handling

If PEAR fails:

- record the exact `pear --help` output and PEAR version;
- compare wrapper options against the installed PEAR CLI;
- prefer loading the official DARLIN PEAR module/environment over editing source code.

If FastQC or MultiQC fails:

- verify module system availability;
- record `module list`, `which fastqc`, and `which multiqc`;
- prefer official module loads over local wrapper edits.

If MATLAB fails:

- record `which matlab`;
- record `matlab -batch "ver"`;
- confirm Bioinformatics Toolbox and Image Processing Toolbox;
- do not treat MATLAB failure as a sample/config failure until toolbox availability is resolved.

If Snakemake path/CARLIN copy behavior appears to target source directories:

- stop;
- rerun from a throwaway code copy under HPC scratch/project storage;
- confirm original official code roots remain unchanged.

## Expected Output Inventory To Check

For each locus, record:

- Snakemake exit code and log path;
- PEAR assembled FASTQ outputs;
- FastQC and MultiQC outputs;
- DARLIN result directories for cutoff overrides `3` and `10`;
- completion sentinels such as `DARLIN_analysis.done`, if produced;
- MATLAB `.mat` summaries such as `Summary.mat`, if produced;
- any error logs under `.snakemake/log/`.

## Completion Criteria

The HPC run is successful only if both current samples complete official preprocessing with scratch-contained outputs and verified raw FASTQ integrity:

- `Brain_E1_all_TA` as Tigre;
- `Brain_E1_all_RA` as Rosa;
- no `/ssd` writes;
- no raw FASTQ modification;
- no Sailu-ST or NicheFate execution.
