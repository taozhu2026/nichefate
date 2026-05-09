# DARLIN-02A Mixed ST Raw FASTQ Strategy

## Sample Identity

Use only:

- `sample_id = Sailu_0313_19A_mixed_ST`
- `tissue_status = mixed_unresolved`

Do not label this sample as `Brain_E1_ST`, `E1-only`, or Brain-only.

## Interpretation

Brain/Pancreas split is not required before preliminary raw ST processing.

The ST track question is:

Can Sailu BC + E generate a mixed ST expression matrix plus spatial coordinates/metadata for M0?

This is not barcode-informed NicheFate and does not integrate Meiji RA/TA DARLIN barcode data.

## Current Observations

- E folder exists and contains lane-level PE100 R1/R2 FASTQs.
- E `RunInfo.xml` reports `Read1-Read2`, `R1LenShow=100`, `LaneNum=4`, and `IndexLen=0`.
- BC folder exists and contains Salus call/log/table artifacts.
- No BC FASTQ file was found during targeted all-depth FASTQ search in the BC folder.
- BC `RunInfo.xml` reports `CustomSeq`, `CycleNumShow=30`, `LaneNum=2`, `RowMax=14`, and `ColumnMax=130`.
- BC `SalusCallFile/ListTable.csv` has two rows and 1821 columns in a lightweight CSV header/sample probe, consistent with a called spatial-structure artifact but not yet validated as an M0 coordinate table.

## Required Output Before M0

M0 should consume processed spatial expression inputs, not raw FASTQ. Required future outputs include:

- expression matrix
- spot or spatial barcode identifiers
- coordinate/spatial table
- spot metadata
- gene metadata
- optional AnnData `.h5ad`

## Boundary

No ST processing, alignment, counting, matrix generation, Brain/Pancreas annotation, NicheFate M0-M5, or barcode/ST integration was run in DARLIN-02A.
