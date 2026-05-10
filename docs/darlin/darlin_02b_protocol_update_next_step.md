# DARLIN-02B Protocol Update Next Step

## Protocol Resolution

The cfg_type/template blocker is resolved for current Meiji RA/TA:

- `Brain_E1_all_TA`: `BulkRNA_Tigre_14UMI` + `Tigre_2022_v2` + `[3, 10]`.
- `Brain_E1_all_RA`: `BulkRNA_Rosa_14UMI` + `Rosa_v2` + `[3, 10]`.

## Current Readiness

DARLIN-01C staging plan is ready for review, but actual dry-run execution is not ready.

Decision: `blocked_pending_dependency_env`.

## Exact Next Step

Start DARLIN-01C only after review: create throwaway scratch staging roots and symlink-only `raw_fastq/` layout using the protocol-resolved configs, then verify dependency/MATLAB/Snakemake side-effect safety before any dry-run command.

Do not start DARLIN-01C from this task.
