# DARLIN-02A Barcode Track Next Step

## Corrected State

The barcode track is no longer blocked by the meaning of RA/TA.

Corrected mapping:

- `Brain_E1_all_TA` → `Tigre` candidate track.
- `Brain_E1_all_RA` → `Rosa` candidate track.
- `Brain_E1_all_CA` → absent/deferred `Col1a1/cCARLIN` track.

The previous `RA=Tigre / TA=Rosa` mapping is wrong and deprecated as:

`deprecated_wrong_mapping_from_prior_plan`

## Official Support Found

Local official code supports the relevant families:

- Tigre templates: `Tigre`, `Tigre_2022`, `Tigre_2022_v2`.
- Rosa templates: `Rosa`, `Rosa_v2`.
- cCARLIN template: `cCARLIN`.
- Tigre cfg candidates: `BulkDNA_Tigre*`, `BulkRNA_Tigre*`.
- Rosa cfg candidates: `BulkDNA_Rosa*`, `BulkRNA_Rosa*`.

## Still Blocked Before DARLIN-01C

Do not start the official dry-run until these are confirmed:

1. Exact template for `Brain_E1_all_TA`: `Tigre_2022`, `Tigre_2022_v2`, or another official Tigre value.
2. Exact template for `Brain_E1_all_RA`: `Rosa`, `Rosa_v2`, or another official Rosa value.
3. Exact `cfg_type` for each sample.
4. `read_cutoff_UMI_override` and any allele/QC thresholds.
5. Whether RA and TA require separate staging/config roots because they use different templates.
6. Safe staging root and symlink policy.
7. MATLAB/module/toolbox availability.
8. Approval for Snakemake parse-time template-copy side effects.

## Exact Recommended Barcode Next Step

Ask the provider/lab contact for exact `template`, exact `cfg_type`, and dry-run cutoff values for:

- `Brain_E1_all_TA` as Tigre.
- `Brain_E1_all_RA` as Rosa.

After those values are confirmed, start DARLIN-01C as a dry-run-only staging/config preparation node. Do not run full preprocessing in DARLIN-01C.
