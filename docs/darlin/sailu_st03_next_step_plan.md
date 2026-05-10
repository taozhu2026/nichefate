# Sailu-ST-03 Next Step Plan

## Current Decision

Parser smoke decision: `parser_candidate_stable_needs_whitelist_coordinate`.

## If Proceeding

Before Sailu-ST-04, resolve:

1. barcode whitelist or coordinate map for Sailu/Salus spatial barcodes;
2. chemistry/read-structure contract confirming barcode and UMI positions;
3. reference genome/transcriptome and gene annotation paths;
4. toy aligner/counting command boundaries;
5. output schema for spot x gene matrix and M0 metadata.

## Recommended Next Step

Obtain or construct a barcode whitelist / coordinate map and confirm whether `CAND_R1_ALT_8_18` plus `CAND_R1_UMI_15_30` are valid parser fields. Only after that review should Sailu-ST-04 plan toy alignment/counting.

Do not start Sailu-ST-04 from this task.
