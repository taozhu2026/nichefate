# DARLIN-02B Future DARLIN-01C Staging and Dry-Run Plan

## Status

Plan only. Do not execute in DARLIN-02B.

## Future Staging Root

Candidate root:

`/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta/`

Do not use `/ssd`.

## Future Layout

Use separate sample/config roots by default because TA and RA use different locus/template families:

- `ta_tigre/`
- `ra_rosa/`

Each root should contain:

- `config.yaml`
- `raw_fastq/`
- future symlinks only, not copies

Future expected symlink names:

- `ta_tigre/raw_fastq/Brain_E1_all_TA_R1.fastq.gz`
- `ta_tigre/raw_fastq/Brain_E1_all_TA_R2.fastq.gz`
- `ra_rosa/raw_fastq/Brain_E1_all_RA_R1.fastq.gz`
- `ra_rosa/raw_fastq/Brain_E1_all_RA_R2.fastq.gz`

## Future Config Shape

Each future `config.yaml` should include:

- `SampleList`
- `cfg_type`
- `template`
- `read_cutoff_UMI_override`
- `CARLIN_memory_factor`
- `sbatch`
- `CARLIN_max_run_time`

Exact `cfg_type`, `template`, and cutoff values are intentionally unresolved in 02B.

## Future Dry-Run Command Shape

Do not execute now.

Candidate command shape after confirmation:

`conda run -n darlin-repro snakemake -s /home/zhutao/projects/darlin_cell_repro/code/snakemake_DARLIN/snakefiles/snakefile_matlab_DARLIN_Part1.py --configfile config.yaml --cores 1 --dry-run`

## Blocker

Because the snakefile may call `update_CARLIN_dir(template)` and perform `rsync` during parse, 01C must explicitly approve where template-copy side effects are allowed before running even a dry-run.
