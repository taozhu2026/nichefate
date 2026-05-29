# Lineage Input Contract

The lineage input contract is part of the lineage-aware evidence regime. It
defines how processed barcode evidence joins to the shared NicheFate spatial
units.

## Required Contract

- Primary join key: `sample_id + slice_id + cellbin_id`
- Assays: `CA`, `TA`, `RA`
- Processed lineage evidence tables must be available before clone calling
- Allele annotation is QC metadata and must not inflate primary counts

## What It Guarantees

- Stable cellbin identity across evidence, clone, and niche tables
- Distinct CA/TA/RA loci
- Benchmark-specific sample names and paths stay outside the generic module
  names
- Compatibility with the shared M0-M2.5 substrate rather than a separate
  lineage-only workflow
