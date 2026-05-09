# Sailu-ST-02 R1/R2 Role Inference

## Sample

- `sample_id`: `Sailu_0313_19A_mixed_ST`
- `tissue_status`: `mixed_unresolved`
- Input folder: `/data/zhutao/nichefate_data/202604271751_2211180001_B_1_ZHY_XBB_E_PE100_100`

## R1 Decision

- Inference: `likely_barcode_umi_read`
- Confidence: `low`
- Evidence: Leading short windows are barcode/UMI-like and low-diversity motif fraction is 18.00%.

## R2 Decision

- Inference: `mixed_or_unknown`
- Confidence: `low`
- Evidence: Some leading windows are barcode/UMI-like, but full-read profile is not diagnostic.

## Boundary

This role inference uses only bounded sequence statistics from the first 10,000 read pairs per lane. It does not use alignment, whitelist matching, coordinate metadata, or provider chemistry documentation.
