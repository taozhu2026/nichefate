# M2.5 Niche-State Coarsening Design

Coarsening is needed because the anchor-level representation is too redundant for stable GPCCA on the mainline.

## Why Coarsening Is Needed

- M2 already has 10 metadata columns and 765 numeric features per anchor.
- The three-scale representation is biologically useful, but one row per anchor makes the state space too fine for clean macrostate inference.
- GPCCA should operate on meaningful niche-states, not millions of nearly duplicated anchor rows.

## What Must Be Preserved

- Timepoint structure.
- Slice structure.
- Spatial coordinates.
- Cell-type composition.
- M2 feature averages.
- Biological annotation summaries.

## Candidate Methods

- Leiden clustering in M2 latent space.
- Metacell / metaniche aggregation.
- Spatially constrained clustering.
- Timepoint-wise coarsening followed by cross-time alignment.

## Recommended First Pilot

- Use the reduced M2 embedding.
- Coarsen within each timepoint or within each timepoint/slice first.
- Preserve an anchor_id -> metaniche_id mapping.
- Keep the pilot conservative so rare states are not erased.

## Input Contract

- M2 by-slice representation.
- M2 feature schema.
- Anchor metadata with timepoint and slice labels.

## Output Contract

- Metaniche table.
- Anchor-to-metaniche map.
- Metaniche feature centroid matrix.
- Metaniche coordinates.
- Metaniche composition table.
- QC summary.

## Risks

- Over-coarsening.
- Losing rare states.
- Mixing timepoints.
- Mixing spatially distant niches.
- Circular endpoint definitions.
