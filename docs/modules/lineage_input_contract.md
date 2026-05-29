# Lineage Input Contract

NF-L0 defines the join contract for lineage-aware NicheFate.

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
