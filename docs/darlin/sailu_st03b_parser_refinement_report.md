# Sailu-ST-03B Parser Refinement Report

## Decision

Parser refinement decision: `stable_nonoverlap_parser_candidate_needs_whitelist_coordinate`.

## ST03 Overlap Correction

The ST03 best barcode/UMI pair overlaps by 4 bp and is not a final joint parser.

## Best Non-Overlapping Candidate

- Structure: `NONOVERLAP_R1_BC1_8_UMI9_20`
- Barcode window: `R1:1-8`
- UMI window: `R1:9-20`
- Score: `9.0`
- Decision: `candidate_parser_needs_whitelist_coordinate`

## Negative Controls

Negative controls tested: R1 `31-40`, R1 `41-60`, R1 `61-80`, R2 `1-20`, and R2 `20-40`. These controls are not selected as final parser windows.

## Readiness

Coordinate map ready: `false`.

Expression matrix ready: `false`.

M0 input ready: `false`.
