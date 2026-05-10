# DARLIN-02B to DARLIN-01C Updated Recommendation

## Protocol Parameters

The Nature Protocols table resolves the previous cfg_type/template/cutoff blocker for current Meiji RA/TA samples:

- `Brain_E1_all_TA`: `BulkRNA_Tigre_14UMI`, `Tigre_2022_v2`, `read_cutoff_UMI_override = [3, 10]`.
- `Brain_E1_all_RA`: `BulkRNA_Rosa_14UMI`, `Rosa_v2`, `read_cutoff_UMI_override = [3, 10]`.

## Updated Readiness Decision

Decision: `blocked_pending_dependency_env`.

DARLIN-01C dry-run-only staging can now be prepared as a plan, but actual staging/dry-run execution remains blocked until the dependency environment, MATLAB policy, and Snakemake side-effect containment are explicitly reviewed.

## Remaining Blockers

- Safe scratch staging root must be approved before creating directories or symlinks.
- Symlink-only raw_fastq naming must be executed only in DARLIN-01C.
- Dependency environment must expose Snakemake and required tools.
- MATLAB / Custom_CARLIN availability and policy must be confirmed.
- Snakemake parse-time template-copy/rsync side effects must be contained in a throwaway scratch working directory.
- Output root must stay under `/home/zhutao/scratch/nichefate/`; no `/ssd` output.

Do not start DARLIN-01C from this task.
