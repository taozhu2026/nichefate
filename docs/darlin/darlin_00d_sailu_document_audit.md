# DARLIN-00D Sailu Document Audit

Generated: 2026-05-09T03:47:54Z

This audit searched only previously inspected Sailu/local DARLIN roots for lightweight document or sample-sheet-like files. FASTQs were not opened, archives were not extracted, and no preprocessing was run.

## Roots Audited

- `/data/zhutao/nichefate_data/202603131741_Pro019_A_SE30_1_0313_19A_BC`
- `/data/zhutao/nichefate_data/202604271751_2211180001_B_1_ZHY_XBB_E_PE100_100`
- `/data/zhutao/nichefate_data/202604271754_C2302270016_A_1_ZHY_XBB_R_PE100_250`
- `/data/zhutao/nichefate_data/MeiJi`

## Candidate Summary

Sixteen candidate text/table/log files were found in the audited roots. Fifteen are Sailu-side files from the `0313_19A_BC` delivery and one is the Meiji checksum manifest for E1 RA/TA FASTQs.

The Sailu-side candidates appear to be instrument/basecall/runtime logs rather than biological sample sheets:

- `LogFiles/AInfo.txt` records run configuration such as `nStrExpName=0313_19A_BC`, sequencing mode, lane activity, and `Index1=0` / `Index2=0`. It does not provide Brain/Pancreas, B1/E1/E2, RA/TA/CA/ST, or a sample-to-well mapping.
- `LogFiles/ASalusCall.csv` records time and lane count summaries. It does not contain biological sample mapping.
- `SalusCallFile/ListTable.csv` is a numeric instrument table without headers describing sample identity, chip position, or well mapping.
- `SalusCallFile/log.txt` and `LogFiles/saluscallOutput.txt` are basecall/process logs. They mention index settings and lane events but do not contain a usable sample sheet.
- The remaining Sailu logs (`AAbnormal.txt`, `AError.txt`, `APerformance.csv`, `AFluid.txt`, `ATemperature.csv`, `AFocus.txt`, `AStage.txt`, `AQuality.txt`, `ADebug.txt`, `uuid.txt`) are instrument QC/runtime artifacts.

The Meiji `raw_md5.txt` confirms local RA/TA E1 FASTQ filenames and checksums. It is useful for DARLIN-01A barcode sample-sheet construction, but it does not link Meiji barcode data to Sailu ST data.

## Matching Evidence

- Supports E1 route at provider-confirmed route level: yes, from user/provider confirmation.
- Supports E1 route from machine-readable Sailu document evidence: no.
- Supports B1 as separate from E1: no direct Sailu document proof was found in this audit; B1 separation remains based on provider confirmation plus chip-position logic from DARLIN-00C.
- Exact Meiji RA/TA to Sailu E1 ST matching key found: no.
- Useful Sailu documents for DARLIN-01A sample-sheet construction: none found in the audited Sailu files.
- Useful non-Sailu document for DARLIN-01A: Meiji `raw_md5.txt`, for RA/TA FASTQ filename and checksum confirmation.

## Consequence

DARLIN-01A can proceed as sample-sheet construction and dry-run planning for the confirmed E1 route, but it should explicitly carry an unresolved matching-field requirement. Barcode-informed NicheFate integration must wait until official DARLIN outputs and Sailu ST metadata expose or are assigned a reviewed matching key.
