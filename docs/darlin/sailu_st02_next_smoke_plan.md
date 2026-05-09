# Sailu-ST-02 Next Smoke Plan

## Exact Next Smoke Concept

Do not run the smoke test until explicitly approved.

A future Sailu-ST-03 smoke may:

- read a tiny bounded subset from the same E_PE100_100 FASTQs;
- extract candidate barcode/UMI fields only if candidate positions are selected from ST-02 evidence or provider documentation;
- optionally align or pseudoalign a tiny transcript-read subset only if reference and annotation paths are available;
- write exploratory outputs only under `/home/zhutao/scratch/nichefate/darlin_st_sailu_mixed/`;
- avoid `/ssd` and avoid modifying raw files.

## Required Before Smoke

- Decide whether ST-02 candidate windows are sufficient for a parser attempt or whether provider chemistry documentation is required first.
- Provide barcode whitelist/coordinate map if barcode extraction is expected to produce spatial coordinates.
- Provide reference genome/transcriptome and gene annotation if any count matrix is attempted.

## Non-Goals

The smoke output would not be final ST output, not Brain-only, not E1-only, and not barcode-informed NicheFate.
