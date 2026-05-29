# Clone x Niche Integration

Clone x niche integration is the E3 lineage-aware representation module. It
maps validated DARLIN-style joint clones onto NicheFate spatial units produced
by the shared M0-M2.5 substrate.

## Spatial Units

- Cellbin
- Non-overlapping tile
- Local group
- Metaniche / niche-state category

Tiles are the primary spatial summary because they are non-overlapping. Groups
are local-context summaries and should not be summed as tissue abundance.
Metaniches are descriptive niche-state categories.

## Outputs

- Clone composition matrices
- Clone richness, entropy, and dominance summaries
- QC-aware coverage summaries
- Reference-support and de novo allele fraction summaries as QC metadata

Clone x niche composition is a representation layer. It can later become an
input to PlanA or PlanB designs, but it is not a standalone dynamics result.

## Framework Role

This module connects lineage-aware evidence to the same spatial units used by
ST-only NicheFate. It makes lineage information comparable with the shared
spatial niche substrate without replacing ST-only modules.
