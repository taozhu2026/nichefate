# DARLIN-01A Remaining Blockers

Generated: 2026-05-09T03:57:32Z

## Blocks Official DARLIN Dry-Run

- Confirm official DARLIN `cfg_type` for Meiji E1 RA/TA.
- Confirm official DARLIN `template` / locus / primer set for Meiji E1 RA/TA.
- Confirm staging contract for Meiji FASTQs. Current names do not match the local snakefile pattern `raw_fastq/{sample}_R1.fastq.gz` and `raw_fastq/{sample}_R2.fastq.gz`.
- Confirm output root for dry-run artifacts. DARLIN-01A does not create directories, symlinks, or copied FASTQs.
- Confirm whether MATLAB should be made available through module, conda environment, or another lab-standard route.
- Confirm whether Snakemake should run from `snakemake_darlin` conda environment or another environment.

## Blocks Sailu ST Processing

- Find or confirm the E1/E2 split key for the shared Sailu folder.
- Identify the user's official ST script entry point.
- Define expected ST inputs, output matrix path, metadata path, coordinate path, QC report path, and sample ID naming.
- Confirm whether the ST script consumes lane-level FASTQs directly or expects a provider-demultiplexed export.

## Blocks Barcode-Informed NicheFate

- Official DARLIN preprocessing output table is not available yet for Meiji E1 RA/TA.
- ST metadata table is not available yet for the confirmed E1 route.
- Matching key between processed barcode outputs and ST metadata is not found.
- Matching level is not yet decided: sample, library, spot, cell, or spatial barcode.

## Not Blocked

- Meiji RA and TA can be represented as draft barcode sample-sheet rows.
- Meiji R1/R2 pairing can be recorded from filenames and md5 manifest evidence.
- Sailu shared-folder ST route candidate can be documented without processing.
