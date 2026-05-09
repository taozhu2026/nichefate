# DARLIN-00D Updated First-Round Plan

Generated: 2026-05-09T03:47:54Z

## Route

First-round development uses `Brain_031319A_E1_RA_TA_ST`.

- ST component: Sailu Brain `031319A-E1` ST, processed and QCed with the user's existing ST scripts.
- Barcode component: Meiji Brain `031319A-E1` RA/TA DARLIN reads, processed with the ShouWen Wang lab official/lab-standard DARLIN software/pipeline.
- Missing component: E1 CA.
- Deferred components: Sailu Brain B1 barcode-only data, Pancreas E2 ST, public/legacy references.

## Barcode Preprocessing Track

- Input scope for DARLIN-01A: Meiji E1 RA/TA paired FASTQs only.
- Official route: ShouWen Wang lab DARLIN/CARLIN pipeline or lab-standard equivalent.
- First action: construct a reviewed sample sheet and staging plan; do not preprocess in DARLIN-00D.
- Naming issue: local Meiji names use `Brain_E1_all_RA.R1.raw.fastq.gz` / `Brain_E1_all_RA.R2.raw.fastq.gz` style and will need an explicit dry-run staging or naming contract before official execution.
- Required unresolved fields: DARLIN `cfg_type`, template/locus, sample IDs, output destination, and whether RA/TA labels represent biological groups, technical captures, or library roles.

## ST Processing Track

- Input scope for DARLIN-01A: Sailu Brain E1 ST data under the provider-confirmed E1 route.
- Processing route: user's existing ST processing and QC scripts.
- First action: document the ST script entry point, expected inputs, expected outputs, metadata path, expression matrix path, and QC summary contract.
- No ST processing is run in DARLIN-00D.

## Integration Track

- Integration starts only after official DARLIN output tables and ST metadata are available.
- The BarcodeEvidenceAdapter must consume standardized clone/barcode tables derived from official outputs, not raw FASTQ.
- The DatasetAdapter must expose ST cell/spot/anchor metadata with a reviewed `barcode_match_key`.
- If only sample-level matching is available, first-round NicheFate remains pseudo-only or sample-level validation only. It must not be interpreted as cell/spot-level barcode-informed fate inference.

## DARLIN-01A Output Goal

The next node should produce an E1 sample-sheet and dry-run plan for Meiji RA/TA barcode preprocessing plus a Sailu E1 ST processing plan. It should not run preprocessing, ST processing, or NicheFate until the sample sheet, matching key, and output contracts are reviewed.
