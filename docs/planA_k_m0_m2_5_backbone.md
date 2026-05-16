# PlanA-K M0-M2.5 Spatial Niche Backbone

This document defines the stable M0-M2.5 backbone for NicheFate. The backbone
ends at metaniche / niche-state construction. It does not include Kmix_A,
GPCCA, terminal-state selection, absorption probability, final fate inference,
or DARLIN / barcode integration.

## Frozen Backbone

The M0-M2.5 backbone is frozen as the stable spatial niche construction and
representation layer:

- M0 prepares time-resolved spatial transcriptomics inputs, performs input QC,
  computes reusable embeddings, builds within-slice spatial graphs, and exports
  standardized slice-level objects.
- M1 constructs anchor-centered multi-scale micro-niche feature tables from M0
  objects. Rows are anchor and scale indexed.
- M2 pivots M1 multi-scale niche features into one row per anchor with stable
  metadata and numeric feature groups.
- M2.5 coarsens M2 anchors into metaniches / niche-states, preserving anchor
  provenance, timepoint, slice, feature centroids, composition summaries, and
  coordinate join information for within-slice interpretation.

## Inputs

M0 expects source spatial transcriptomics inputs with expression, cell metadata,
slice or sample identifiers, timepoint labels, and spatial coordinates. New ST
datasets may require an input or feature adapter before they can enter the
M1/M2 contracts.

M1 expects standardized M0 outputs, including spatial coordinates, cell labels
when available, embeddings, and within-slice spatial graphs.

M2 expects M1 by-slice niche feature tables with:

- `slice_id`, `slice_file`, `time`, `time_day`, `mouse_id`
- `anchor_index`, `anchor_cell_id`
- cell type labels when available
- `scale`
- numeric feature groups for composition, entropy, embedding summaries,
  spatial summaries, and graph topology

M2.5 expects M2 by-slice representation tables, an M2 feature schema, and M1
coordinates for coordinate rescue by `slice_id`, `anchor_index`, and
`anchor_cell_id`.

## Outputs

M0 outputs standardized AnnData-derived objects, spatial graphs, embeddings, QC
summaries, and exported slice-level objects.

M1 outputs by-slice anchor-centered niche feature tables.

M2 outputs by-slice representation matrices with one row per anchor and a
stable feature schema.

M2.5 outputs:

- anchor-to-metaniche assignment tables
- metaniche state tables
- feature centroid matrices
- metaniche coordinate summaries
- metaniche composition summaries
- QC summaries for compactness, rare-state preservation, and coordinate joins
- production manifests for reproducibility

## Coordinate Assumptions

Spatial coordinates are valid for within-slice niche construction, within-slice
coordinate joins, and per-slice visualization. Raw x/y coordinates must not be
used as cross-slice or cross-time distances unless an explicit registration
contract exists. Downstream transition construction must not use unregistered
cross-slice raw x/y distance.

M1 duplicate coordinate keys can occur through scale replicates. M2.5 coordinate
joins must deduplicate these scale rows and retain a single valid anchor
coordinate per `slice_id`, `anchor_index`, and `anchor_cell_id`.

## Not Included

This freeze does not include:

- Kmix_A or any transition kernel production
- GPCCA or macrostate inference
- macrostate annotation
- source / terminal role scoring
- absorption or fate probability computation
- final PlanA-K fate / terminal / absorption logic
- DARLIN / barcode integration
- raw data, scratch outputs, production matrices, or large generated artifacts

The previous and future fate layers remain separate from this backbone freeze.
They must consume M2.5 outputs through explicit, audited contracts.

## Reuse

For new ST datasets, reuse M0-M2.5 as the spatial niche construction and
representation backbone after adding any dataset-specific adapter required to
match the M0/M1 inputs. DARLIN / barcode integration should begin after the ST
dataset can produce compatible M0-M2.5 outputs and should not be treated as part
of this backbone freeze.
