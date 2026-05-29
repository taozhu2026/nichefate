# L126 Spatio-DARLIN Benchmark

L126 is the first benchmark dataset used to validate the lineage-aware
NicheFate module surface.

## Benchmark Facts

- The dataset contains three serial brain sections.
- Each section has matched ST expression and CA/TA/RA lineage evidence.
- spatio_DARLIN does not provide a precomputed cross-locus joint clone table.
- DARLIN-style joint clone calling was added as a NicheFate lineage module.

## Selected Operational Policy

- Reference bank policy: `gr`
- Allele inclusion policy: `mapped_rare_plus_empirical_denovo`
- Reference/de novo status remains allele-level QC metadata

## Benchmark Result

- Joint clone layer recovered substantial spatial clone coverage.
- Reference-only calling was too conservative for the benchmark.
- The benchmark supports spatial clone and niche characterization, not future
  dynamics inference.
