# DARLIN-ST-00 Putative Tissue Annotation Plan

Generated: 2026-05-09T05:37:43Z

## Purpose

This plan defines a future post-processing strategy for annotating putative Brain-like and Pancreas-like regions after `Sailu_0313_19A_mixed_ST` has a processed expression matrix and spatial metadata.

No marker analysis is run in DARLIN-ST-00. No Brain mask is created in DARLIN-ST-00.

## Marker-Based Strategy

After mixed ST processing succeeds, use marker expression to assign exploratory labels:

- Brain-like region candidates: enriched neuronal, glial, and brain-region marker programs.
- Pancreas-like region candidates: enriched endocrine, exocrine, ductal, acinar, islet, and pancreas stromal marker programs.
- Ambiguous regions: retain as `mixed_or_unresolved` if marker programs conflict or spatial continuity is weak.

Marker lists should be selected and documented before analysis. They should be treated as annotation aids, not as provider demultiplexing keys.

## Clustering and Spatial Continuity Checks

Future annotation should require:

- Clusters with coherent marker enrichment.
- Spatially contiguous regions rather than isolated marker-positive spots.
- QC review for low-quality spots that could create false tissue signals.
- Consistency between expression clusters and spatial coordinate neighborhoods.

## Criteria for a Future Putative Brain Mask

A future putative Brain mask may be drafted only after:

1. Mixed ST processing produces a valid expression matrix and coordinate table.
2. Brain-like marker programs are enriched in coherent spatial clusters.
3. Pancreas-like marker programs are absent or clearly separated from the candidate region.
4. The mask is explicitly labeled as `putative_brain_like`, not provider-confirmed Brain `031319A-E1`.
5. The mask is reviewed before any Brain-specific NicheFate interpretation.

## Interpretation Warning

Marker annotation is not provider demultiplexing ground truth. It cannot replace a Sailu sample sheet, well map, spatial barcode-to-sample table, or Brain/Pancreas region map.

The output of this future step may support exploratory QC and hypothesis generation. It must not be used for formal Brain-specific inference, barcode-informed NicheFate, or final benchmarking without review.

## Blocked Actions in DARLIN-ST-00

- No marker analysis.
- No clustering execution.
- No Brain mask creation.
- No Pancreas mask creation.
- No NicheFate M0-M5.
- No barcode-informed transition evidence.
