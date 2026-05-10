# DARLIN-01D0 Next Step Recommendation

## Approval Decision

`approved_for_01D_contained_dryrun`

## Evidence

- Scratch code copy exists at `/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01d0/code_copy`.
- `CARLIN_pipeline/Custom_CARLIN` exists under the scratch code copy, matching the official `settings.py` path model.
- Original `snakemake_DARLIN` and `Custom_CARLIN` inventories are unchanged between precheck and postcheck.
- Snakemake 7.24.0 is available in `snakemake_darlin`.
- MATLAB is not available, but MATLAB is not required for Snakemake dry-run parsing. It remains required before real preprocessing.
- Prepared commands use scratch-copy Snakefile and scratch-copy Python import path.

## Exact Next Step

Start DARLIN-01D as a dry-run-only node. Run the prepared TA/Tigre command first. If it remains scratch-contained and does not start rule execution, run the prepared RA/Rosa command next.

Do not start full preprocessing.
