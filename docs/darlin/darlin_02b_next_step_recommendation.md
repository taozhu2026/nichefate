# DARLIN-02B Next-Step Recommendation

## Corrected Mapping Used

- `Brain_E1_all_TA` → Tigre.
- `Brain_E1_all_RA` → Rosa.
- Deprecated `RA=Tigre / TA=Rosa` mapping was not used.

## Dry-Run Readiness

Current decision:

`blocked_pending_exact_template_variant`

DARLIN-01C should not start yet.

## Required Answers Before DARLIN-01C

For `Brain_E1_all_TA`:

- Exact Tigre template: `Tigre_2022`, `Tigre_2022_v2`, or another official value.
- Exact Tigre `cfg_type`.
- `read_cutoff_UMI_override`.

For `Brain_E1_all_RA`:

- Exact Rosa template: `Rosa`, `Rosa_v2`, or another official value.
- Exact Rosa `cfg_type`.
- `read_cutoff_UMI_override`.

Shared:

- Approve separate staging roots under `/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta/`.
- Approve symlink-only staging.
- Confirm MATLAB/module/toolbox policy.
- Confirm whether Snakemake parse-time `rsync`/template-copy side effects are acceptable.

## Exact Next Step

Ask provider/lab contact for exact per-sample official config values. After confirmation, start DARLIN-01C dry-run-only staging/config preparation. Do not start full preprocessing in DARLIN-01C.
