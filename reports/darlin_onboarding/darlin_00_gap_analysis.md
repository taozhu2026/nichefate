# DARLIN-00 Gap Analysis

Generated: 2026-05-09T03:01:07Z

## What Exists

- Local candidate DARLIN/spatial data under `/data/zhutao/nichefate_data`, including MeiJi RA/TA FASTQs, Sailu shared PE100/100 and PE100/250 lane FASTQs, metadata/logs/QC reports, and Salus spatial delivery artifacts.
- Legacy/test DARLIN pipeline context under `/data/zhutao/darlin_data`, including HSC/MPP FASTQs, `config.yaml`, Snakemake metadata, and `DARLIN_analysis.done` flags.
- Public/reference DARLIN reproduction metadata and official code under `/home/zhutao/projects/darlin_cell_repro`.

## What Is Missing Or Unconfirmed

- CA/TA/RA demultiplexing/sample sheet for shared Sailu directories was not found in inspected files.
- Official DARLIN cfg_type/template/locus selection for the new local TA/RA/CA data requires user or lab confirmation.
- No standardized clone/barcode output table such as refined_results.csv or Summary-derived table was found for the local onboarding subset.
- No direct cell_id/spot_id bridge between DARLIN barcode reads and expression/spatial metadata was confirmed.
- Existing /data/zhutao/darlin_data HSC/MPP run has raw FASTQs and done flags but no matched NicheFate spatial/scRNA context.

## First Subset Decision

No first subset is automatically selected. The recommended candidate is `Brain_031319A_B1_CA_TA_RA_shared_PE100_250` only if the user/lab confirms sample sheet, I7/demultiplexing, and official DARLIN config. Otherwise use manual confirmation before DARLIN-01.

## Official Preprocessing Readiness

- Can official preprocessing start for the NicheFate first subset: **no / uncertain**.
- Can a sample sheet be constructed from inspected files alone: **not reliably**.
- FASTQ naming is not uniform across sources: the official README expects MiSeq-style `_L001_...` names, while the local reference snakefile uses `_R1/_R2`; the new local data still needs a reviewed staging/renaming or demultiplexing contract.
- Existing barcode outputs: **no usable standardized clone table found**.

## Expression / Spatial Matching Readiness

- Direct cell/spot ID matching: not confirmed.
- Sample-level matching: possible only after user/lab confirms sample identities.
- Barcode-informed NicheFate transition evidence: blocked until official clone table and match key exist.

## Recommended Next Command

Do not run preprocessing yet. The next node-level task should be manual subset confirmation and sample-sheet contract review, followed only then by DARLIN-01 sample-sheet construction and official preprocessing dry-run for one selected subset.
