# NicheFate Lineage-Aware Baseline v1

NicheFate now ships a lineage-aware module set for barcode-supported clone and
niche analysis.

## What This Baseline Does

- Validates a lineage input contract.
- Adapts processed lineage evidence into canonical cellbin-level tables.
- Calls validated DARLIN-style joint clones with allele-level QC metadata.
- Integrates joint clones into cellbin, tile, group, and metaniche summaries.
- Exposes a design-only interface for future dynamics modules.

## What It Does Not Do

- It does not process raw FASTQ.
- It does not rerun spatio_DARLIN.
- It does not infer future dynamics from serial sections.
- It does not expose reference/de novo as main clone classes.

## Public Module Map

- NF-L0: lineage input contract
- NF-L1: lineage evidence adapter
- NF-L2: DARLIN-style joint clone calling
- NF-L3: clone x niche integration
- NF-L4: lineage-aware spatial niche characterization
- NF-L5: dynamics interface design

The public module documentation lives in `docs/modules/`. The first benchmark
for this freeze is L126 spatio-DARLIN, documented in
`docs/benchmarks/l126_spatiodarlin.md`.

## Benchmark Positioning

L126 is the first benchmark dataset used to validate the lineage-aware module
surface. It provides matched ST and CA/TA/RA lineage evidence for three serial
sections, but it is a benchmark for spatial clone and niche characterization,
not a basis for future-dynamics claims.

## Legacy And Provenance

Legacy L126-specific scripts and reports are retained for provenance under
`docs/legacy/` and `reports/`. The frozen public names are the generic
lineage-aware and DARLIN-aware module names above.
