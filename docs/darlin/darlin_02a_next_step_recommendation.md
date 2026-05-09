# DARLIN-02A Next-Step Recommendation

## Barcode Track

Corrected mapping is now:

- `Brain_E1_all_TA` → Tigre candidate.
- `Brain_E1_all_RA` → Rosa candidate.
- `Brain_E1_all_CA` → absent/deferred Col1a1/cCARLIN candidate.

The previous `RA=Tigre / TA=Rosa` mapping is wrong and deprecated.

Exact next barcode step:

Confirm the exact official preprocessing values before DARLIN-01C:

- `Brain_E1_all_TA`: exact Tigre template, exact `cfg_type`, and read/UMI cutoff.
- `Brain_E1_all_RA`: exact Rosa template, exact `cfg_type`, and read/UMI cutoff.
- Whether RA and TA require separate config/staging roots.
- Safe non-`/ssd` staging root and symlink policy.
- MATLAB/module/toolbox policy.
- Whether Snakemake dry-run is acceptable despite parse-time template-copy side effects.

Do not start DARLIN-01C until these are confirmed.

## ST Track

Use only:

- `sample_id = Sailu_0313_19A_mixed_ST`
- `tissue_status = mixed_unresolved`

Brain/Pancreas split is not required before preliminary raw ST processing.

Exact next ST step:

Identify or provide the Sailu/Salus raw ST processing entrypoint or vendor output contract that maps:

- BC Salus artifacts
- E PE100 R1/R2 reads

into:

- mixed expression matrix
- spatial coordinate table
- spot metadata
- gene metadata
- optional `.h5ad`

Only after that contract exists should a DARLIN-ST-01 tiny mixed ST smoke plan be prepared under `/home/zhutao/scratch/nichefate/darlin_st_sailu_mixed/`.

## M0 Bridge

M0 is not ready.

M0 should start only after processed mixed ST matrix and coordinate/metadata outputs exist. M0 should not parse raw FASTQs directly.
