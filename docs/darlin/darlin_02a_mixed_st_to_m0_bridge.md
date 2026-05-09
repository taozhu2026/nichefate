# DARLIN-02A Mixed ST to M0 Bridge

## Required M0 Inputs

For `Sailu_0313_19A_mixed_ST`, M0 should receive processed spatial expression objects, not raw FASTQs.

Required fields/files:

- `expression_matrix_path`
- `spot_or_spatial_barcode_id`
- `coordinate_table_path` if available
- `spot_metadata_path`
- `gene_metadata_path`
- `h5ad_path` if generated
- `sample_id = Sailu_0313_19A_mixed_ST`
- `tissue_status = mixed_unresolved`
- `putative_tissue_label = unknown`

## Current Readiness

M0 input is not ready.

Reasons:

- No expression matrix has been generated.
- No confirmed barcode-to-coordinate or spot coordinate table is ready.
- No spot metadata or gene metadata exists for this mixed ST sample.
- No `.h5ad` exists for this mixed ST sample.
- Raw ST processing implementation is not confirmed.

## Interpretation Boundary

M0-M3 can begin only after mixed ST expression and spatial metadata outputs exist.

M0-M3 should not parse raw FASTQ directly.

Putative Brain-like or Pancreas-like annotation is future post-processing and is not provider demultiplexing ground truth.
