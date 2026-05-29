# L126 Spatio-DARLIN Benchmark

L126 spatio-DARLIN is the first benchmark dataset used to validate the
lineage-aware NicheFate evidence regime.

It is a benchmark, not a method name.

## Benchmark Facts

- The dataset contains three serial brain sections.
- Each section has matched ST expression and CA/TA/RA lineage evidence.
- spatio_DARLIN does not provide a precomputed cross-locus joint clone table.
- DARLIN-style joint clone calling was added as a NicheFate lineage-aware
  evidence module.

## Selected Operational Policy

- Reference bank policy: `gr`
- Allele inclusion policy: `mapped_rare_plus_empirical_denovo`
- Primary clone unit: validated DARLIN-style joint clone
- Reference/de novo status: allele-level QC metadata only

## Benchmark Result

- The joint clone layer recovered substantial spatial clone coverage.
- Reference-only calling was too conservative for this benchmark.
- Clone x niche summaries support spatial clone and niche characterization.

## Limitations

L126 sections are serial sections, not timepoints. They do not establish
temporal directionality or complete lineage-aware dynamics.
