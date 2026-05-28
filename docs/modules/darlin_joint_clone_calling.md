# DARLIN-Style Joint Clone Calling

NF-L2 builds the primary clone unit: a validated DARLIN-style joint clone.

## Clone Definition

- Joint clone identity is based on validated lineage alleles passing a unified
  DARLIN-style validity filter.
- Reference-mapped status and de novo status are allele-level QC annotations,
  not clone classes.
- Reference-only clone calling remains a conservative QC benchmark.

## Frozen Policy

- Selected policy: `gr + mapped_rare_plus_empirical_denovo`
- Primary role: validated joint clones
- QC fields: reference support, de novo allele fraction, clone size fraction,
  locus support, and QC status flags
