# NicheFate

Spatial niche-fate dynamics for spatial transcriptomics, with optional
lineage/barcode evidence.

## What Is NicheFate?

NicheFate is a spatial niche dynamics framework. It builds spatial niche
representations from spatial transcriptomics data, then connects those
representations to dynamics engines when appropriate evidence is available.

The repository supports two evidence regimes:

- ST-only / lineage-free NicheFate for spatial transcriptomics data without
  barcode lineage evidence.
- ST + lineage-aware NicheFate for matched spatial transcriptomics and
  processed lineage/barcode evidence.

These regimes are parallel modes of the same framework. The lineage-aware mode
does not replace the ST-only baseline, and the ST-only baseline is not legacy.

## Shared Spatial-Niche Substrate

The shared substrate is used before evidence-specific interpretation:

- M0: ST/sample/input contract and standardized spatial objects.
- M1: spatial niche construction from local neighborhoods.
- M2: niche representation matrices and feature schemas.
- M2.5: niche-state / metaniche coarsening.

This substrate is the common handoff layer for both ST-only and lineage-aware
analysis.

## Evidence Regimes

### E0: ST-Only / Lineage-Free Mode

The ST-only mode uses gene expression, spatial coordinates, metadata, and
derived niche representations. It was developed first on MERFISH/Cadinu/Moffitt
style spatial transcriptomics benchmarks and remains the first stable baseline.

### E1: ST + Lineage-Aware / Barcode-Supported Mode

The lineage-aware mode adds processed barcode evidence after the shared spatial
substrate is available. It introduces:

- a lineage evidence input contract,
- DARLIN-style validated joint clone calling,
- clone x niche composition summaries,
- QC metadata for reference-mapped and empirical de novo alleles,
- design-only interfaces for future lineage-aware dynamics.

Reference/de novo status is allele-level QC metadata, not a biological clone
class. The primary clone unit is a validated DARLIN-style joint clone.

## Dynamics Engines

- PlanA Markov-GPCCA baseline: the mature ST-only dynamics baseline built on
  the shared substrate and PlanA-K outputs.
- PlanA lineage-aware mode: design/in-progress interface for adding clone
  state variables to candidate dynamics analyses.
- PlanB branch/clone-aware niche dynamics: in progress. It is not frozen as a
  completed production engine.

PlanA and PlanB consume the shared substrate plus whichever evidence regime is
available.

## How Lineage Changes The Workflow

Lineage-aware NicheFate adds barcode evidence as an additional evidence regime:

1. Validate processed lineage evidence and cellbin join keys.
2. Adapt CA/TA/RA lineage alleles into canonical lineage tables.
3. Call validated DARLIN-style joint clones.
4. Aggregate clone composition into cellbins, tiles, groups, and metaniches.
5. Expose clone state variables for future dynamics interfaces.

The clone layer supports spatial clone/niche characterization. It is not, by
itself, a temporal dynamics result.

## Benchmarks

- MERFISH/Cadinu/Moffitt: ST-only baseline and PlanA/ST development benchmark.
- L126 spatio-DARLIN: first lineage-aware benchmark with matched ST expression
  and CA/TA/RA lineage evidence.

Benchmarks validate module contracts and outputs. They are not method names.

## Current Status

- ST-only M0-M2.5 substrate: frozen as the stable spatial niche backbone.
- ST-only PlanA / PlanA-K baseline: frozen as the first stable dynamics
  baseline.
- Lineage-aware clone/niche layer: ready with QC warnings from the L126
  benchmark.
- Lineage-aware dynamics: design interface only; not a frozen production
  dynamics engine.

## What Is Not Claimed

- L126 serial sections do not support temporal fate inference.
- PlanB is not completed or frozen as a production engine.
- This branch does not process raw FASTQ or rerun spatio_DARLIN.
- Lineage evidence does not by itself determine directionality.

## Repository Structure

- `src/nichefate/`: core package modules.
- `src/nichefate/planA_st_only/`: ST-only PlanA facade modules.
- `src/nichefate/planA_k/`: shared substrate and PlanA-K implementation
  surfaces.
- `src/nichefate/lineage/`: generic lineage-aware evidence and clone/niche
  interfaces.
- `src/nichefate/darlin/`: DARLIN-style allele and joint clone calling
  interfaces.
- `docs/`: public architecture, module, benchmark, and legacy/provenance docs.
- `configs/`: generic schemas and benchmark configs.
- `scripts/`: reproducible entrypoints and compatibility scripts.

## Quick Module Map

| Layer | Meaning |
|---|---|
| M0-M2.5 | Shared spatial niche substrate |
| E0 | ST-only / lineage-free evidence regime |
| E1 | Lineage evidence adapter |
| E2 | DARLIN-style joint clone calling |
| E3 | Clone x niche representation |
| PlanA | Markov-GPCCA baseline and lineage-aware design interface |
| PlanB | Branch/clone-aware niche dynamics, in progress |

Start with `docs/nichefate_architecture.md` and
`docs/pipeline_module_index.md` for the integrated architecture.
