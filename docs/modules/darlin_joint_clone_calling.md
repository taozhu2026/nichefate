# DARLIN-Style Joint Clone Calling

DARLIN-style joint clone calling is the E2 lineage-aware evidence module. It
builds the primary clone unit used by NicheFate lineage-aware mode:
a validated DARLIN-style joint clone.

## Clone Definition

- Joint clone identity is based on validated CA/TA/RA lineage alleles passing a
  unified DARLIN-style validity filter.
- Reference-mapped status and empirical de novo status are allele-level QC
  annotations, not clone classes.
- Reference-only clone calling remains a conservative QC benchmark and
  reference-bank coverage diagnostic.

## Role In NicheFate

The module produces cellbin-level joint clone variables upstream of niche-level
lineage composition. It does not perform niche construction, dynamics
inference, or raw FASTQ processing.

The output is consumed by clone x niche integration, where joint clone
composition can be summarized over tiles, groups, and metaniches.

## Frozen L126 Benchmark Policy

The first benchmark policy selected for L126 spatio-DARLIN was:

- reference bank policy: `gr`
- allele inclusion policy: `mapped_rare_plus_empirical_denovo`
- clone unit: validated DARLIN-style joint clone
- QC fields: reference support, de novo allele fraction, clone size fraction,
  locus support, and QC status flags

This is a benchmark-backed policy, not a claim that L126 is the method name.
