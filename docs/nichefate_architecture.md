# NicheFate Architecture

NicheFate is one spatial niche dynamics framework with multiple evidence
regimes. The framework separates shared spatial niche construction from the
evidence available for downstream interpretation.

## One Framework, Multiple Evidence Regimes

The shared substrate is M0-M2.5:

- M0 validates and standardizes spatial transcriptomics inputs.
- M1 builds local spatial niche features.
- M2 creates niche representation matrices.
- M2.5 coarsens anchors into metaniche / niche-state units.

After M2.5, NicheFate can operate in either ST-only mode or lineage-aware mode.
These modes are parallel regimes, not parent/child modules.

## ST-Only / Lineage-Free Mode

The ST-only mode uses expression, spatial context, metadata, and niche
representations without barcode lineage evidence. It is the first stable
baseline and remains a primary NicheFate mode.

The MERFISH/Cadinu/Moffitt benchmark lineage validates this regime and the
PlanA-ST / PlanA-K baseline.

## Lineage-Aware / Barcode-Supported Mode

The lineage-aware mode adds processed barcode evidence to the shared substrate.
It introduces a lineage input contract, lineage evidence adapter,
DARLIN-style joint clone calling, and clone x niche representation.

The primary clone unit is a validated DARLIN-style joint clone. Reference and
de novo status are allele-level QC metadata, not clone classes.

Clone calling is upstream of niche-level lineage composition:

1. Processed CA/TA/RA alleles are normalized into lineage evidence tables.
2. Validated joint clones are called at cellbin level.
3. Clone assignments are aggregated into tiles, groups, and metaniches.
4. Clone composition becomes an additional niche state variable.

## Dynamics Interfaces

PlanA and PlanB consume the shared substrate plus whichever evidence regime is
available:

- PlanA-ST consumes M0-M2.5 plus E0 evidence.
- PlanA lineage-aware designs can consume M0-M2.5 plus clone state variables.
- PlanB branch/clone-aware dynamics remains design / in progress.

Lineage evidence can support clone overlap terms, confidence weights, and clone
composition regularization. Direction still requires time, perturbation, or an
explicit biological prior.

## Benchmarks

MERFISH/Cadinu/Moffitt and L126 spatio-DARLIN are benchmarks only:

- MERFISH/Cadinu/Moffitt validates the ST-only baseline.
- L126 spatio-DARLIN validates the lineage-aware clone/niche layer.

L126 serial sections support spatial clone and niche characterization. They do
not establish temporal directionality.
