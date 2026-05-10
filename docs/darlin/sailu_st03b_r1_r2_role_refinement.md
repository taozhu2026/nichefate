# Sailu-ST-03B R1/R2 Role Refinement

## R1

Refined role: `barcode_umi_candidate_read_low_confidence`.

R1 remains the best candidate read for barcode/UMI parsing because all tested non-overlapping parser candidates are R1-based and the best scorecard candidate is `NONOVERLAP_R1_BC1_8_UMI9_20`.

## R2

Refined role: `mixed_or_possible_transcript_control_not_barcode_selected`.

R2 negative/control windows were tested, including R2 `1-20` and R2 `20-40`, but R2 is not selected as the barcode/UMI parser read because prior ST02/ST03 evidence classified R2 as `mixed_or_unknown` and no chemistry contract supports R2 barcode/UMI use.

## Missing Evidence

- Sailu/Salus read-structure contract.
- Barcode whitelist.
- Coordinate map.
- Reference/annotation and alignment/counting policy for a later toy count smoke.
