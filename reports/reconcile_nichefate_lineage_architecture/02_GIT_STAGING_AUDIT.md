# Git Staging Audit

## Result

- Staging audit status: `PASS`
- Staged files: `18`
- Staged forbidden-data files: `0`
- `git diff --cached --check`: passed

## Largest Staged Files

- `README.md` - 5077 bytes
- `docs/nichefate_architecture.md` - 2538 bytes
- `reports/reconcile_nichefate_lineage_architecture/00_ARCHITECTURE_AUDIT.md` - 2006 bytes

## Included

- README and public architecture docs
- Module index and lineage module pages
- L126 benchmark page
- Architecture audit and validation reports

## Excluded

- Raw data inputs
- Processed outputs
- Scratch outputs
- h5ad, FASTQ, mtx, npz, parquet, and large TSV artifacts
- Unrelated untracked files from earlier work

## Notes

The staged set is limited to the intended public documentation and report
files. No code or data artifacts were staged for this reconciliation.
