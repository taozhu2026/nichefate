# DARLIN Expression / Spatial Matching Contract

Generated: 2026-05-09T03:01:07Z

This is a design-only contract. DARLIN-00 does not implement matching and does not load expression matrices.

## Required Standardized Expression / Spatial Metadata

- `cell_id`
- `sample_id`
- `mouse_id`
- `tissue`
- `time_point`
- `slice_id` if spatial
- `x` coordinate if spatial
- `y` coordinate if spatial
- `cell_type_annotation`
- `expression_matrix_path`
- `metadata_path`
- `barcode_match_key`

## Matching Status From DARLIN-00

- Direct `cell_id` matching: not confirmed.
- ID conversion needed: likely for single-cell or spatial data, but conversion table was not found.
- Sample-level matching only: possible for some candidates after sample-sheet confirmation, but insufficient for barcode-informed niche transitions.
- Current status: barcode and expression/spatial modalities are unmatched for first-round NicheFate integration.

## Matching Requirements Before Barcode-Informed NicheFate

- Confirm sample-level identity: mouse, tissue, time point, lane/library, and DARLIN locus.
- Confirm cell/spot-level bridge: raw barcode cell barcode, expression cell barcode, or spatial anchor ID.
- Define whether barcode evidence is bulk-level, single-cell-level, or spatial-anchor-level.
- If only bulk barcode evidence exists, do not treat it as per-cell fate evidence; use it only for sample-level validation or a separately approved method.
