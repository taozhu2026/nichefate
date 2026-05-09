# NicheFate DARLIN Barcode Adapter Contract

Generated: 2026-05-09T03:01:07Z

This is a design-only contract. DARLIN-00 does not implement an adapter and does not parse raw barcode reads.

## Required Standardized Barcode Columns

- `cell_id`
- `sample_id`
- `mouse_id`
- `tissue`
- `time_point`
- `barcode_id`
- `clone_id`
- `allele_sequence` or `allele_id`
- `edited_status`
- `read_count` or `UMI_count`
- `confidence` / `QC flag`
- `source_file`
- `preprocessing_method`

## Optional Columns

- `library_id`
- `molecule_count`
- `barcode_family`
- `rare_allele_flag`
- `germline_flag`
- `homoplasy_risk_flag`
- `clone_size`
- `matched_expression_cell_id`
- `matched_spatial_anchor_id`

## Entry Into NicheFate

- `DatasetAdapter`: loads expression/spatial metadata and exposes standardized cells/spots/anchors without lineage assumptions.
- `BarcodeEvidenceAdapter`: loads official DARLIN/CARLIN clone tables and normalizes allele/clone/QC fields to this contract.
- `TransitionEvidence[barcode]`: contributes barcode-derived coupling only after clone confidence, sample matching, and time/tissue comparability pass review.
- Hybrid transition evidence: compare barcode-informed transition evidence against frozen `pseudo_broad` and `pseudo_sharpened` controls rather than replacing them silently.

## Interpretation Rules

- Raw DARLIN FASTQ is not lineage evidence until official/lab-standard preprocessing produces allele/clone assignments.
- Clone membership is not automatically fate; NicheFate outputs must distinguish clone, lineage, fate probability, and pseudo-transition.
- Ambiguous barcode-to-expression matching must block barcode-informed transition evidence and remain pseudo-only until confirmed.
