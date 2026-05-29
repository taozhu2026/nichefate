# Git Staging Audit

## Result

- Staging audit status: `PASS`
- Staged files after adding this audit: `16`
- Staged forbidden-data files: `0`
- `git diff --cached --check`: passed

## Largest Staged Files

- `configs/module_registry/nichefate_module_registry.json` - 10031 bytes
- `docs/pipeline_module_index.md` - 6521 bytes
- `README.md` - 6102 bytes

## Included

- Concrete algorithm module registry config
- Public registry and pipeline docs
- Lineage insertion point documentation
- Script-to-module mapping docs
- Concrete module audit, validation, and staging audit reports
- Focused registry tests

## Excluded

- Raw data inputs
- h5ad, FASTQ, mtx, npz, parquet, and large TSV artifacts
- Processed outputs
- Figure directories
- Scratch outputs
- Unrelated untracked files from prior work

## Notes

No algorithm code was staged. The changes are documentation, registry config,
tests, and small report files only.
