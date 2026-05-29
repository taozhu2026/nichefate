# Lineage-Aware NicheFate

The frozen baseline is organized as six public module layers:

| Layer | Responsibility |
|---|---|
| NF-L0 | Validate the lineage input contract and required join keys. |
| NF-L1 | Convert processed lineage evidence into canonical cellbin-level tables. |
| NF-L2 | Call validated DARLIN-style joint clones with allele-level QC metadata. |
| NF-L3 | Aggregate joint clones into cellbin, tile, group, and metaniche units. |
| NF-L4 | Summarize spatial clone composition, diversity, and QC signals. |
| NF-L5 | Expose a design-only interface for future dynamics modules. |

L126 spatio-DARLIN is the first benchmark dataset used to validate this module
surface. It is a benchmark for clone and niche characterization, not a basis
for future-dynamics claims.
