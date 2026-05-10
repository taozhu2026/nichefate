# DARLIN-01C Protocol-Resolved Dry-Run Staging Plan

## Scope

This is a future DARLIN-01C plan for Meiji E1 RA/TA barcode preprocessing dry-run-only staging. No staging directory, symlink, scratch config, Snakemake command, MATLAB command, or preprocessing was executed in this task.

## Scratch Root

Planned root:

`/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01c/`

## Planned Layout

```text
/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01c/
├── TA_Tigre/
│   ├── raw_fastq/
│   │   ├── Brain_E1_all_TA_L001_R1_001.fastq.gz -> /data/zhutao/nichefate_data/MeiJi/Brain_E1_all_TA.R1.raw.fastq.gz
│   │   └── Brain_E1_all_TA_L001_R2_001.fastq.gz -> /data/zhutao/nichefate_data/MeiJi/Brain_E1_all_TA.R2.raw.fastq.gz
│   ├── config.yaml
│   └── output/
└── RA_Rosa/
    ├── raw_fastq/
    │   ├── Brain_E1_all_RA_L001_R1_001.fastq.gz -> /data/zhutao/nichefate_data/MeiJi/Brain_E1_all_RA.R1.raw.fastq.gz
    │   └── Brain_E1_all_RA_L001_R2_001.fastq.gz -> /data/zhutao/nichefate_data/MeiJi/Brain_E1_all_RA.R2.raw.fastq.gz
    ├── config.yaml
    └── output/
```

## Planned Config Contents

`TA_Tigre/config.yaml`:

```yaml
SampleList: [Brain_E1_all_TA]
cfg_type: BulkRNA_Tigre_14UMI
template: Tigre_2022_v2
read_cutoff_UMI_override: [3, 10]
sbatch: choose_later_based_on_environment
```

`RA_Rosa/config.yaml`:

```yaml
SampleList: [Brain_E1_all_RA]
cfg_type: BulkRNA_Rosa_14UMI
template: Rosa_v2
read_cutoff_UMI_override: [3, 10]
sbatch: choose_later_based_on_environment
```

## Execution Boundary

- No symlinks are created in this task.
- No config files are written to scratch in this task.
- No Snakemake command is executed in this task.
- No raw data is moved, copied, renamed, or modified.
- This is a plan for DARLIN-01C only.
