# Lineage-Aware NicheFate Mode

Lineage-aware NicheFate is one evidence regime of the NicheFate spatial niche
dynamics framework. It adds barcode-supported lineage variables to the shared
M0-M2.5 spatial niche substrate.

It is not a standalone replacement for NicheFate, and it does not supersede the
ST-only / lineage-free mode.

## Module Layers

| Layer | Responsibility |
|---|---|
| E1 / NF-L1 | Convert processed lineage evidence into canonical cellbin-level tables. |
| E2 / NF-L2 | Call validated DARLIN-style joint clones with allele-level QC metadata. |
| E3 / NF-L3 | Aggregate joint clones into cellbin, tile, group, and metaniche units. |
| NF-L4 | Summarize spatial clone composition, diversity, and QC signals. |
| NF-L5 | Expose a design-only interface for future lineage-aware dynamics. |

The lineage input contract remains explicit because barcode evidence may come
from DARLIN or from another lineage barcode system with compatible processed
allele tables.

## Relationship To The Shared Substrate

The lineage-aware mode assumes spatial transcriptomics data can be represented
through the shared M0-M2.5 substrate. Lineage evidence then adds clone state
variables that can be joined to cellbins, tiles, groups, and metaniches.

Validated joint clones are cellbin-level lineage variables. Clone x niche
composition is a representation layer, not a dynamics result by itself.

## Benchmark

L126 spatio-DARLIN is the first benchmark dataset used to validate this module
surface. It provides matched ST expression and CA/TA/RA lineage evidence for
serial sections. It supports spatial clone and niche characterization, not
temporal directionality claims.
