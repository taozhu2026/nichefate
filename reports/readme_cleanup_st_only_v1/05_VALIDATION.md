# README Cleanup Validation

Decision: `PASS`

## Checks

- `pass` production facade import with `PYTHONPATH=src`:
  `nichefate.planA_st_only`, `module_registry`, `spatial_dataset_adapter`, and
  `fate_probability`.
- `pass` Markdown link sanity for staged README/docs cleanup files.
- `pass` JSON validation for all new `reports/readme_cleanup_st_only_v1/*.json`
  files.
- `pass` TSV validation for `03_LEGACY_CLEANUP_PROPOSAL.tsv`.
- `pass` claim-boundary grep for current-facing README/docs cleanup files.
- `pass` `git diff --check`.
- `skip` `python -m py_compile`: no code files were changed.
- `skip` focused pytest: no code files were changed.

## Behavioral Scope

- Numerical behavior changed: `False`
- Analysis rerun: `False`
- M2.5 rerun: `False`
- Kmix_A rerun: `False`
- GPCCA rerun: `False`
- Fate-probability rerun: `False`
- DARLIN/barcode processing: `False`
- Raw data modified: `False`
