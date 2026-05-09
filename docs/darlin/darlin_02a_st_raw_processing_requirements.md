# DARLIN-02A Raw ST Processing Requirements

## Goal

For `Sailu_0313_19A_mixed_ST`, determine whether BC + E can produce a mixed spatial expression matrix and spatial metadata suitable for NicheFate M0.

Brain/Pancreas split is not a prerequisite for this preliminary raw ST processing track.

## Required Components

- BC parser or called barcode-table parser.
- E read parser for PE100 R1/R2 reads.
- UMI rule if UMIs are present in the Sailu/Salus chemistry.
- Spatial barcode or spot identifier rule.
- Barcode-to-position or coordinate construction.
- Reference genome or transcriptome.
- Gene annotation.
- Aligner/counting method.
- Expression matrix writer.
- Coordinate/spatial metadata writer.
- AnnData writer if `.h5ad` is the bridge format.
- M0-compatible metadata writer.

## Current Status

- E FASTQs are present and look like lane-level PE100 paired-end reads in tiny probes.
- BC raw FASTQs were not found.
- BC called/log artifacts exist, including `SalusCallFile/ListTable.csv` and `LogFiles/RunInfo.xml`.
- BC `RunInfo.xml` has `RowMax=14`, `ColumnMax=130`, and `CycleNumShow=30`.
- `ListTable.csv` is a spatial-structure candidate but is not yet confirmed as a barcode-to-coordinate table.
- No local raw ST processing entrypoint was confirmed in DARLIN-02A.

## Missing Contract

The missing Sailu/Salus raw ST contract must define:

- Which BC artifact contains usable spatial barcodes or coordinates.
- How BC artifacts connect to E FASTQ reads.
- Which read contains expression sequence, UMI, and/or spatial barcode.
- Whether lanes should be merged before counting.
- Which reference and annotation should be used.
- What output schema should be considered M0-ready.

## Boundary

Do not run ST processing until this contract is confirmed.
