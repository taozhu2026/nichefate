# Sailu-ST-03B ST03 Quantitative Review

## Scope

Sample: `Sailu_0313_19A_mixed_ST`, `tissue_status = mixed_unresolved`.

ST03 parsed 100,000 read pairs per lane and selected:

- barcode-like candidate: `CAND_R1_ALT_8_18`
- UMI-like candidate: `CAND_R1_UMI_15_30`

## Coordinate Convention

All reported windows use 1-based inclusive coordinates. Python slicing converts these to 0-based half-open slices as `seq[start-1:end]`. For example, R1 `8-18` is implemented as `seq[7:18]`.

## Key Issue

The ST03 best pair overlaps by 4 bp: R1 `8-18` and R1 `15-30` share positions 15-18. This pair is therefore not a final barcode+UMI parser.

## Quantitative Interpretation

ST03 showed stable bounded extraction for several R1 internal windows, but stability alone is not enough. ST03B adds non-overlap checks, negative controls, top1/top10 dominance review, barcode scale comparison, and explicit whitelist/coordinate requirements.
