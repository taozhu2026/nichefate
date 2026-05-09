# DARLIN-02B Dry-Run Safety Review

## Decision

DARLIN-01C dry-run staging is not ready yet.

Primary decision:

`blocked_pending_exact_template_variant`

Secondary blockers:

- `blocked_pending_cfg_type`
- `blocked_pending_dependency_env`
- `blocked_pending_matlab_policy`
- `blocked_pending_safe_staging_root`
- `blocked_pending_snakemake_side_effect_review`

## Main Safety Issue

The official Part1 snakefile imports and calls template setup logic at top level. The helper `update_CARLIN_dir(template)` uses `rsync` to copy `Custom_CARLIN` into template-specific directories.

Therefore, a Snakemake dry-run may still produce code-copy side effects during parse unless run in an approved staging/code-copy location.

## Safe 01C Preconditions

Before DARLIN-01C:

1. Confirm exact TA/Tigre template.
2. Confirm exact RA/Rosa template.
3. Confirm exact `cfg_type` values.
4. Confirm `read_cutoff_UMI_override`.
5. Approve scratch staging root.
6. Confirm symlink-only staging.
7. Confirm MATLAB/module/toolbox policy.
8. Confirm whether `update_CARLIN_dir` side effects are acceptable in the chosen location.

## What 02B Did Not Do

- No Snakemake command.
- No MATLAB command.
- No PEAR/FastQC/MultiQC command.
- No symlinks.
- No raw data movement.
- No `/ssd` write.
- No preprocessing.
