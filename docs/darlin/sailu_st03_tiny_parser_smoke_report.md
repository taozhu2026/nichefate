# Sailu-ST-03 Tiny Parser Smoke Report

## Sample

- `sample_id`: `Sailu_0313_19A_mixed_ST`
- `tissue_status`: `mixed_unresolved`
- Brain/Pancreas split required before processing: `false`

## Parser Scope

This task parsed the first 100,000 read pairs per lane from E_PE100_100 R1/R2 FASTQs. It did not run alignment, gene counting, ST processing, NicheFate, DARLIN preprocessing, or matrix production.

## Candidate Windows Tested

- `CAND_R1_BC_1_8`: R1 positions 1-8
- `CAND_R1_BC_1_10`: R1 positions 1-10
- `CAND_R1_ALT_8_18`: R1 positions 8-18
- `CAND_R1_ALT_10_20`: R1 positions 10-20
- `CAND_R1_UMI_15_30`: R1 positions 15-30
- `CAND_R2_ALT_1_20`: R2 positions 1-20 control

## Results

- Records parsed per lane: `{'Lane01': 100000, 'Lane02': 100000, 'Lane03': 100000, 'Lane04': 100000}`
- Read-pair header mismatches: `0`
- Best R1 barcode-like candidate: `CAND_R1_ALT_8_18`
- Best UMI-like candidate: `CAND_R1_UMI_15_30`
- R2 control signal: `CAND_R2_ALT_1_20` was stable but not selected as barcode because R2 remains `mixed_or_unknown`
- Lane consistency status: `consistent_for_bounded_parser_smoke`
- BC/E linkage status: `unresolved_no_direct_bc_e_linkage`

## Decision

Parser smoke decision: `parser_candidate_stable_needs_whitelist_coordinate`.

The parser can extract candidate windows in a bounded, reproducible way. This does not define final barcode/UMI positions. Coordinate map, whitelist, chemistry contract, reference/annotation, and expression counting remain unresolved.
