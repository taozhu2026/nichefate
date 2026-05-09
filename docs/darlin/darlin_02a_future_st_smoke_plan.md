# DARLIN-02A Future Mixed ST Smoke Plan

## Status

Plan only. Do not execute in DARLIN-02A.

## Sample

- `sample_id = Sailu_0313_19A_mixed_ST`
- `tissue_status = mixed_unresolved`

## Allowed Future Smoke Root

Use only:

`/home/zhutao/scratch/nichefate/darlin_st_sailu_mixed/`

Do not write to `/ssd`.

## Future Smoke Objective

Test whether Sailu BC + E can generate a tiny mixed ST expression matrix plus spatial coordinates/metadata for M0.

## Proposed Future Smoke Steps

1. Create tiny read subsets from E lane R1/R2 files.
2. Parse BC `RunInfo.xml` and `SalusCallFile/ListTable.csv`.
3. Confirm whether `ListTable.csv` is a barcode-to-coordinate, coordinate-grid, or other Salus call artifact.
4. Confirm whether E reads contain or can be linked to spatial barcode identifiers.
5. If a parser and reference are available, run a tiny alignment/counting smoke test.
6. Write minimal matrix, coordinate, spot metadata, gene metadata, and optional `.h5ad`.
7. Validate that generated outputs match the M0 bridge contract.

## Current Blockers

- No raw ST processing entrypoint confirmed.
- BC called table semantics are not confirmed.
- E/BC linking rule is not confirmed.
- Reference genome/transcriptome and gene annotation are not specified.
- No M0-compatible matrix or coordinate table exists yet.

## Boundary

This future smoke plan is not Brain/Pancreas demultiplexing and is not barcode-informed NicheFate.
