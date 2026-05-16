# M2.5 Metaniche Pilot Protocol

## Biological Unit

An anchor-indexed micro-niche is a sampled local microenvironment, not
a cell-level state. The anchor is an indexing point. A metaniche or
niche-state is an aggregation of similar micro-niches in M2
representation space. GPCCA macrostates will later be inferred from
directed sparse transitions among metaniches.

## First Pilot Scope

Use a small, safe subset only: at most four representative M2 slices
and at most 5,000 anchors per slice. The pilot must not use the full
M2 dataset and must not run GPCCA.

## Coarsening Strategy

- Sample anchors stratified by timepoint/slice.
- Select safe feature groups: composition, entropy, and embedding mean.
- Standardize features.
- Reduce dimensionality with PCA, capped at 30 components.
- Cluster with Leiden when explicitly requested and available, or use
  MiniBatchKMeans for a capped reproducible pilot.
- Produce metaniche centroids and an anchor-to-metaniche map.

## Coarsening Constraints

- Preserve timepoint and slice labels.
- Do not mix incompatible conditions without recording it.
- Preserve spatial centroid only when coordinates exist in M2 metadata.
- Preserve cell-type composition summaries when labels exist.
- Track rare-state diagnostics so rare states are not silently erased.

## Output Contract

- `anchor_to_metaniche.tsv`
- `metaniche_table.tsv`
- `metaniche_feature_centroids.csv`
- `metaniche_composition.tsv`
- `metaniche_qc.json`
- `pilot_summary.md`

## QC

- Number of anchors sampled.
- Number of metaniches.
- Anchors per metaniche distribution.
- Timepoint and slice purity.
- Spatial compactness if coordinates exist.
- Feature compactness.
- Rare-state loss warning.
- Whether the metaniche count is suitable for sparse K and GPCCA pilot.

## Failure Modes

- Too few metaniches.
- Too many tiny metaniches.
- Mixed timepoint artifacts.
- Spatially incoherent clusters.
- Feature group domination.
- Rare states collapsed away.
- Missing metadata.
