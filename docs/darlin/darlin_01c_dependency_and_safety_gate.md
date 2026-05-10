# DARLIN-01C Dependency and Safety Gate

Timestamp: 2026-05-10T03:40:07Z

## Dependency Status

- `snakemake` is not on the current PATH.
- `snakemake_darlin` has Python 3.9.23 and Snakemake 7.24.0.
- `darlin-repro` has Python 3.9.23 and Snakemake 7.24.0.
- `matlab` is not on the current PATH and was not run.
- `pear` is on the current PATH at `/opt/miniforge/bin/pear`, but PEAR processing was not run.
- `fastqc` and `multiqc` are not on the current PATH and were not run.

## Config Field Safety

The scratch `config.yaml` files include only fields confirmed from official/example DARLIN configs:

- `SampleList`
- `cfg_type`
- `template`
- `read_cutoff_UMI_override`
- `CARLIN_memory_factor`
- `sbatch`
- `CARLIN_max_run_time`

No unverified operational fields were added.

## Snakemake Parse-Time Side Effect

`snakefile_matlab_DARLIN_Part1.py` calls `hf.update_CARLIN_dir(CARLIN_dir, config['template'])` at top level. That helper runs `rsync` into `/home/zhutao/projects/darlin_cell_repro/code/CARLIN_pipeline/` for template-specific CARLIN directories.

Observed implications:

- `Tigre_2022_v2` would rsync into `Tigre_CARLIN_2022_v2`.
- `Rosa_v2` would rsync into `Rosa_CARLIN_v2`.
- These destinations are outside the 01C scratch staging root.
- Therefore, a Snakemake workflow dry-run is not treated as side-effect-free.

## Gate Decision

Decision:

`staging_complete_dryrun_blocked_pending_snakemake_side_effect_review`

The candidate Snakemake environment exists, but the dry-run was not executed because Snakefile parsing can write outside scratch via `rsync`.
