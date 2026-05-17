# Staging Audit

Decision: `PASS`

## Checks

- `pass` only_approved_paths_staged: all staged paths are within the approved reorg scope
- `pass` no_raw_data_staged: raw/external data roots are absent from the index
- `pass` no_h5ad_fastq_staged: h5ad and FASTQ files are absent from the index
- `pass` no_production_matrix_tables_staged: parquet, npz, npy, and large matrix formats are absent from the index
- `pass` no_scratch_outputs_staged: scratch outputs are absent from staged paths
- `pass` no_darlin_evidence_staged: DARLIN/barcode evidence reports are absent from the index
- `pass` no_figure_binaries_staged: figure binaries are absent from the index
- `pass` no_ssd_paths_in_staged_text: staged text files and staged paths contain no SSD-root references

## Staged Files

- `README.md`
- `docs/pipeline_module_index.md`
- `docs/planA_st_only_v1_production_modules.md`
- `reports/git_update_planA_st_only_modules/00_GIT_PREFLIGHT.json`
- `reports/git_update_planA_st_only_modules/00_GIT_PREFLIGHT.md`
- `reports/git_update_planA_st_only_modules/00_PLAN_A_ST_ONLY_MODULE_REORG_SUMMARY.json`
- `reports/git_update_planA_st_only_modules/00_PLAN_A_ST_ONLY_MODULE_REORG_SUMMARY.md`
- `reports/git_update_planA_st_only_modules/01_PLAN_A_FILE_INVENTORY.json`
- `reports/git_update_planA_st_only_modules/01_PLAN_A_FILE_INVENTORY.md`
- `reports/git_update_planA_st_only_modules/01_PLAN_A_FILE_INVENTORY.tsv`
- `reports/git_update_planA_st_only_modules/02_LEGACY_TO_PRODUCTION_MODULE_MAP.json`
- `reports/git_update_planA_st_only_modules/02_LEGACY_TO_PRODUCTION_MODULE_MAP.md`
- `reports/git_update_planA_st_only_modules/02_LEGACY_TO_PRODUCTION_MODULE_MAP.tsv`
- `reports/git_update_planA_st_only_modules/03_VALIDATION.json`
- `reports/git_update_planA_st_only_modules/03_VALIDATION.md`
- `reports/git_update_planA_st_only_modules/04_PROPOSED_STAGING_LIST.json`
- `reports/git_update_planA_st_only_modules/04_PROPOSED_STAGING_LIST.md`
- `reports/git_update_planA_st_only_modules/04_PROPOSED_STAGING_LIST.tsv`
- `reports/git_update_planA_st_only_modules/05_STAGING_AUDIT.json`
- `reports/git_update_planA_st_only_modules/05_STAGING_AUDIT.md`
- `reports/planA_st_only_v1_index/00_PLAN_A_ST_ONLY_V1_INDEX.json`
- `reports/planA_st_only_v1_index/00_PLAN_A_ST_ONLY_V1_INDEX.md`
- `reports/planA_st_only_v1_index/01_MODULE_REGISTRY.tsv`
- `reports/planA_st_only_v1_index/02_FINAL_RESULT_MANIFEST.tsv`
- `reports/planA_st_only_v1_index/03_FINAL_FIGURE_INDEX.md`
- `reports/planA_st_only_v1_index/03_FINAL_FIGURE_INDEX.tsv`
- `reports/planA_st_only_v1_index/04_CLAIM_BOUNDARY.json`
- `reports/planA_st_only_v1_index/04_CLAIM_BOUNDARY.md`
- `scripts/planA_st_only_00_module_inventory.py`
- `scripts/planA_st_only_01_validate_frozen_outputs.py`
- `scripts/planA_st_only_02_build_result_index.py`
- `src/nichefate/planA_st_only/__init__.py`
- `src/nichefate/planA_st_only/biological_annotation.py`
- `src/nichefate/planA_st_only/endpoint_markov_inference.py`
- `src/nichefate/planA_st_only/fate_probability.py`
- `src/nichefate/planA_st_only/gpcca_macrostate_inference.py`
- `src/nichefate/planA_st_only/kernel_assembly.py`
- `src/nichefate/planA_st_only/metaniche_coarsener.py`
- `src/nichefate/planA_st_only/module_registry.py`
- `src/nichefate/planA_st_only/niche_builder.py`
- `src/nichefate/planA_st_only/niche_encoder.py`
- `src/nichefate/planA_st_only/result_package.py`
- `src/nichefate/planA_st_only/result_visualization.py`
- `src/nichefate/planA_st_only/spatial_dataset_adapter.py`
- `src/nichefate/planA_st_only/transition_evidence.py`
- `tests/test_planA_st_only_facades.py`
