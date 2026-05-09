# DARLIN-01A Sailu E1 ST Processing Plan

Generated: 2026-05-09T03:57:32Z

DARLIN-01A records a processing plan only. It does not run ST processing, Snakemake, MATLAB, NicheFate, GPCCA/CellRank, or any data movement.

## Route-Level ST Candidate

- Route component: `Sailu_Brain_031319A_E1_ST`
- Current local folder: `/data/zhutao/nichefate_data/202604271751_2211180001_B_1_ZHY_XBB_E_PE100_100`
- Current status: `shared_folder_route_level_candidate`
- Detected FASTQs: four lane-level R1/R2 pairs under `Res/Lane01` through `Res/Lane04`
- Detected run metadata: `RunInfo.xml`

The lane FASTQs must not be labeled as E1-only. The manual relationship table maps this same Sailu folder to both Brain `031319A-E1` ST and Pancreas `031319A-E2` ST. No local document inspected through DARLIN-01A provides the E1/E2 split key.

## Processing Readiness

ST processing cannot start safely in DARLIN-01A.

Required before a dry-run:

- E1/E2 split key for the shared Sailu folder.
- User's official ST script entry point.
- Expected input layout for that script.
- Expected output paths for expression matrix, spatial coordinates, metadata, and QC report.
- Confirmed sample naming convention linking Sailu ST output back to `Brain_031319A_E1`.

## Planned Contract

The ST processing plan should produce or identify a standardized metadata table with at least:

- `sample_id`
- `tissue`
- `chip_position`
- `group_label`
- `slice_id` if applicable
- `x` coordinate if applicable
- `y` coordinate if applicable
- `cell_or_spot_id`
- `expression_matrix_path`
- `metadata_path`
- `barcode_match_key`

Until those fields exist and the E1/E2 split is resolved, ST remains a route-level planning component only.
