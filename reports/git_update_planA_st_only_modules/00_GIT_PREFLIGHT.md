# Git Preflight

- Hostname: `c461c74de0a4`
- Date UTC: `2026-05-17T02:16:58.881891+00:00`
- PWD: `/home/zhutao/projects/nichefate`
- Branch: `refactor/planA-st-only-production-modules`
- HEAD: `57cbb3b82abbe908a4de31722cb0603d9a9df968`
- origin/main contains frozen backbone commit: `True`
- Untracked files: `1363`
- Changed tracked files: `46`

## Git Status

```text
## refactor/planA-st-only-production-modules...origin/main
M  README.md
M  docs/pipeline_module_index.md
A  docs/planA_st_only_v1_production_modules.md
A  reports/git_update_planA_st_only_modules/00_GIT_PREFLIGHT.json
A  reports/git_update_planA_st_only_modules/00_GIT_PREFLIGHT.md
A  reports/git_update_planA_st_only_modules/00_PLAN_A_ST_ONLY_MODULE_REORG_SUMMARY.json
A  reports/git_update_planA_st_only_modules/00_PLAN_A_ST_ONLY_MODULE_REORG_SUMMARY.md
A  reports/git_update_planA_st_only_modules/01_PLAN_A_FILE_INVENTORY.json
A  reports/git_update_planA_st_only_modules/01_PLAN_A_FILE_INVENTORY.md
A  reports/git_update_planA_st_only_modules/01_PLAN_A_FILE_INVENTORY.tsv
A  reports/git_update_planA_st_only_modules/02_LEGACY_TO_PRODUCTION_MODULE_MAP.json
A  reports/git_update_planA_st_only_modules/02_LEGACY_TO_PRODUCTION_MODULE_MAP.md
A  reports/git_update_planA_st_only_modules/02_LEGACY_TO_PRODUCTION_MODULE_MAP.tsv
A  reports/git_update_planA_st_only_modules/03_VALIDATION.json
A  reports/git_update_planA_st_only_modules/03_VALIDATION.md
A  reports/git_update_planA_st_only_modules/04_PROPOSED_STAGING_LIST.json
A  reports/git_update_planA_st_only_modules/04_PROPOSED_STAGING_LIST.md
A  reports/git_update_planA_st_only_modules/04_PROPOSED_STAGING_LIST.tsv
A  reports/git_update_planA_st_only_modules/05_STAGING_AUDIT.json
A  reports/git_update_planA_st_only_modules/05_STAGING_AUDIT.md
A  reports/planA_st_only_v1_index/00_PLAN_A_ST_ONLY_V1_INDEX.json
A  reports/planA_st_only_v1_index/00_PLAN_A_ST_ONLY_V1_INDEX.md
A  reports/planA_st_only_v1_index/01_MODULE_REGISTRY.tsv
A  reports/planA_st_only_v1_index/02_FINAL_RESULT_MANIFEST.tsv
A  reports/planA_st_only_v1_index/03_FINAL_FIGURE_INDEX.md
A  reports/planA_st_only_v1_index/03_FINAL_FIGURE_INDEX.tsv
A  reports/planA_st_only_v1_index/04_CLAIM_BOUNDARY.json
A  reports/planA_st_only_v1_index/04_CLAIM_BOUNDARY.md
A  scripts/planA_st_only_00_module_inventory.py
A  scripts/planA_st_only_01_validate_frozen_outputs.py
A  scripts/planA_st_only_02_build_result_index.py
A  src/nichefate/planA_st_only/__init__.py
A  src/nichefate/planA_st_only/biological_annotation.py
A  src/nichefate/planA_st_only/endpoint_markov_inference.py
A  src/nichefate/planA_st_only/fate_probability.py
A  src/nichefate/planA_st_only/gpcca_macrostate_inference.py
A  src/nichefate/planA_st_only/kernel_assembly.py
A  src/nichefate/planA_st_only/metaniche_coarsener.py
A  src/nichefate/planA_st_only/module_registry.py
A  src/nichefate/planA_st_only/niche_builder.py
A  src/nichefate/planA_st_only/niche_encoder.py
A  src/nichefate/planA_st_only/result_package.py
A  src/nichefate/planA_st_only/result_visualization.py
A  src/nichefate/planA_st_only/spatial_dataset_adapter.py
A  src/nichefate/planA_st_only/transition_evidence.py
A  tests/test_planA_st_only_facades.py
?? configs/darlin/meiji_e1_ra_ta_staging_plan.yaml
?? configs/planA_k/full_kmix_A.draft.yaml
?? docs/README.md
?? docs/darlin/darlin_01a_bc_spatial_barcode_audit.md
?? docs/darlin/darlin_01a_bc_st_split_feasibility.md
?? docs/darlin/darlin_01b_cfg_type_template_review.md
?? docs/darlin/darlin_01b_dependency_resolution.md
?? docs/darlin/darlin_01b_next_step_recommendation.md
?? docs/darlin/darlin_01b_official_config_contract.md
?? docs/darlin/darlin_01b_provider_confirmation_questions.md
?? docs/darlin/darlin_01b_staging_contract.md
?? docs/darlin/darlin_hpc02_completion_check_report.md
?? docs/darlin/darlin_sailu_pe100_100_split_key_inspection.md
?? docs/planA_k_gpcca/01_planA_mainline_reframe.md
?? docs/planA_k_gpcca/02_niche_definition.md
?? docs/planA_k_gpcca/03_sparse_markov_kernel_design.md
?? docs/planA_k_gpcca/06_minimal_gpcca_pilot_plan.md
?? docs/planA_k_gpcca/07_cellrank_like_kernel_stabilization.md
?? docs/planA_st_only_v1_freeze.md
?? docs/planB_branchsbm/
?? reports/darlin_onboarding/darlin_01a_bc_e_r_relationship_probe.csv
?? reports/darlin_onboarding/darlin_01a_bc_fastq_header_probe.csv
?? reports/darlin_onboarding/darlin_01a_bc_file_inventory.csv
?? reports/darlin_onboarding/darlin_01a_bc_remaining_questions.md
?? reports/darlin_onboarding/darlin_01a_bc_saluscallfile_probe.csv
?? reports/darlin_onboarding/darlin_01a_bc_split_decision.json
?? reports/darlin_onboarding/darlin_01b_cfg_type_template_candidates.csv
?? reports/darlin_onboarding/darlin_01b_config_fields_inventory.csv
?? reports/darlin_onboarding/darlin_01b_confirmation_decision_table.csv
?? reports/darlin_onboarding/darlin_01b_confirmation_summary.json
?? reports/darlin_onboarding/darlin_01b_dependency_status.csv
?? reports/darlin_onboarding/darlin_01b_dryrun_readiness_decision.csv
?? reports/darlin_onboarding/darlin_01b_staging_file_map.csv
?? reports/darlin_onboarding/darlin_01b_summary.json
?? reports/darlin_onboarding/darlin_hpc02_completion_gate_check.csv
?? reports/darlin_onboarding/darlin_hpc02_job_status.csv
?? reports/darlin_onboarding/darlin_hpc02_output_inventory.csv
?? reports/darlin_onboarding/darlin_hpc02_summary.json
?? reports/darlin_onboarding/sailu_pe100_100_fastq_header_probe.csv
?? reports/darlin_onboarding/sailu_pe100_100_html_report_probe.csv
?? reports/darlin_onboarding/sailu_pe100_100_metadata_inventory.csv
?? reports/darlin_onboarding/sailu_pe100_100_split_decision.json
?? reports/git_freeze_m0_m2_5/00_M0_M2_5_GITHUB_FREEZE_SUMMARY.json
?? reports/git_freeze_m0_m2_5/00_M0_M2_5_GITHUB_FREEZE_SUMMARY.md
?? reports/planA_k_cellrank_aligned_fate_freeze/
?? reports/planA_k_final_result_package/
?? reports/planA_k_full_gpcca/
?? reports/planA_k_full_gpcca_feature_only/
?? reports/planA_k_full_gpcca_feature_only_smoke/
?? reports/planA_k_full_kmix_A/
?? reports/planA_k_full_kmix_A_feature_only/
?? reports/planA_k_full_m2_5_implementation/
?? reports/planA_k_full_m2_5_production/
?? reports/planA_k_full_m2_5_qc/
?? reports/planA_k_full_macrostate_annotation/
?? reports/planA_k_full_macrostate_annotation_feature_only/
?? reports/planA_k_full_result_packet/
?? reports/planA_k_full_result_visualization/
?? reports/planA_k_full_result_visualization_feature_only/
?? reports/planA_k_gpcca_redesign/
?? reports/planA_k_gpcca_stabilization/
?? reports/planA_k_macrostate_annotation_probe/
?? reports/planA_k_metaniche_hardening/
?? reports/planA_k_metaniche_pilot/
?? reports/planA_k_production_preflight/
?? reports/planA_k_refactor/
?? reports/planA_k_source_terminal_absorption/
?? reports/planA_k_sparse_kernel_pilot/
?? reports/planA_k_spatial_kernel_integrity_audit/
?? reports/planA_k_tiny_gpcca_probe/
?? reports/planA_readiness/
?? reports/planA_readiness_patch_sprint/
?? reports/planB_branchsbm/
?? scripts/planA_01_readiness_audit.py
?? scripts/planA_02_advisor_patch_sprint.py
?? scripts/planA_k_00_index_existing_kernels.py
?? scripts/planA_k_01_kernel_qc_inspect.py
?? scripts/planA_k_02_metaniche_coarsening_probe.py
?? scripts/planA_k_03_sparse_kernel_design_probe.py
?? scripts/planA_k_10_sparse_kernel_pilot.py
?? scripts/planA_k_11_sparse_kernel_qc.py
?? scripts/planA_k_12_gpcca_readiness_probe.py
?? scripts/planA_k_13_tiny_gpcca_probe.py
?? scripts/planA_k_14_gpcca_probe_qc.py
?? scripts/planA_k_15_build_within_time_kernel.py
?? scripts/planA_k_16_build_stabilized_gpcca_kernels.py
?? scripts/planA_k_17_tiny_gpcca_retry_stabilized.py
?? scripts/planA_k_18_macrostate_annotation_probe.py
?? scripts/planA_k_19_macrostate_triviality_qc.py
?? scripts/planA_k_20_macrostate_figures.py
?? scripts/planA_k_25_build_full_kmix_A.py
?? scripts/planA_k_26_full_kernel_qc.py
?? scripts/planA_k_27_run_full_gpcca.py
?? scripts/planA_k_28_annotate_full_macrostates.py
?? scripts/planA_k_29_full_result_packet.py
?? scripts/planA_k_30_full_result_visualization.py
?? scripts/planA_k_31_source_terminal_role_scoring.py
?? scripts/planA_k_31_spatial_kernel_integrity_audit.py
?? scripts/planA_k_35_cellrank_aligned_terminal_audit.py
?? scripts/planA_k_36_compute_cellrank_aligned_absorption.py
?? scripts/planA_k_37_compute_kforward_absorption_sensitivity.py
?? scripts/planA_k_38_visualize_cellrank_aligned_absorption.py
?? scripts/planB_00_branchsbm_scaffold.py
?? src/nichefate/planA_k/absorption_fate.py
?? src/nichefate/planA_k/cellrank_aligned_terminal.py
?? src/nichefate/planA_k/figures.py
?? src/nichefate/planA_k/full_gpcca.py
?? src/nichefate/planA_k/full_kmix_a.py
?? src/nichefate/planA_k/full_macrostate_annotation.py
?? src/nichefate/planA_k/full_result_packet.py
?? src/nichefate/planA_k/full_result_visualization.py
?? src/nichefate/planA_k/gpcca_probe.py
?? src/nichefate/planA_k/gpcca_stabilization.py
?? src/nichefate/planA_k/legacy.py
?? src/nichefate/planA_k/macrostate_annotation.py
?? src/nichefate/planA_k/source_terminal_roles.py
?? src/nichefate/planA_k/sparse_kernel.py
?? src/nichefate/planA_k/spatial_kernel_integrity_audit.py
?? tests/test_planA_advisor_patch_sprint.py
?? tests/test_planA_k_absorption_fate.py
?? tests/test_planA_k_cellrank_aligned_terminal.py
?? tests/test_planA_k_full_gpcca.py
?? tests/test_planA_k_full_kmix_A.py
?? tests/test_planA_k_full_macrostate_annotation.py
?? tests/test_planA_k_full_result_packet.py
?? tests/test_planA_k_full_result_visualization.py
?? tests/test_planA_k_gpcca_stabilization.py
?? tests/test_planA_k_kernel_qc.py
?? tests/test_planA_k_macrostate_annotation_probe.py
?? tests/test_planA_k_scaffold.py
?? tests/test_planA_k_source_terminal_roles.py
?? tests/test_planA_k_sparse_kernel_pilot.py
?? tests/test_planA_k_spatial_kernel_integrity_audit.py
?? tests/test_planA_k_tiny_gpcca_probe.py
?? tests/test_planA_readiness_audit.py
?? tests/test_planB_branchsbm_scaffold.py
```

## Remotes

```text
origin	https://github.com/taozhu2026/nichefate.git (fetch)
origin	https://github.com/taozhu2026/nichefate.git (push)
```

## Large Untracked Files

|path|file_size|
|---|---|
|reports/planA_k_full_result_visualization/tables/anchor_spatial_plot_sample.tsv|74408677|
|reports/planA_k_full_result_visualization_feature_only/tables/anchor_spatial_plot_sample.tsv|73934660|
|reports/planA_k_final_result_package/tables/per_slice_absorption_plot_table.tsv|11831980|
|reports/planA_k_full_result_visualization_feature_only/key_result_figures/tables/per_slice_spatial_plot_table.tsv|9577633|
|reports/planA_k_metaniche_hardening/coordinate_join_preview/anchor_coordinates.preview.tsv|7931981|
|reports/planA_k_metaniche_hardening/stratified_pilot_outputs/coordinate_join_preview/anchor_coordinates.preview.tsv|7931981|
|reports/planA_k_metaniche_hardening/stratified_pilot_outputs/anchor_to_metaniche.tsv|4941973|
|reports/planA_k_metaniche_pilot/pilot_outputs/anchor_to_metaniche.tsv|4941973|
|reports/planA_k_full_result_visualization_feature_only/key_result_figures/tables/metaniche_state_table_for_key_figures.tsv|3559266|
|reports/planA_k_cellrank_aligned_fate_freeze/figures/spatial_absorption_maps/cellrank_aligned_spatial_absorption_maps.png|3164673|
|reports/planA_k_final_result_package/figures/Figure_2_GPCCA_k6_macrostate_atlas.pdf|3066620|
|reports/planA_k_final_result_package/figures/supplementary_figures/Figure_S1_original_PCA_state_space_atlas.pdf|3066620|
|reports/planA_k_final_result_package/figures/main_figures/Figure_2_UMAP_GPCCA_k6_macrostate_atlas.pdf|3064321|
|reports/planA_k_full_result_visualization_feature_only/key_result_figures/tables/macrostate_membership_long.tsv|2944296|
|reports/planA_k_final_result_package/figures/supplementary_figures/Figure_S3_membership_vs_absorption_probability_to_M5.png|2827761|
|reports/planA_k_full_result_visualization_feature_only/figures/key_results/Figure_3_membership_probability_maps.png|2791426|
|reports/planA_k_full_result_visualization/tables/metaniche_spatial_plot_table.tsv|2758944|
|reports/planA_k_full_result_visualization_feature_only/tables/metaniche_spatial_plot_table.tsv|2744372|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/100221_D9_m5_2_slice_1_membership_probabilities.png|2183782|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/062921_D9_m5_1_slice_1_membership_probabilities.png|1888644|
|reports/planA_k_metaniche_pilot/pilot_outputs/metaniche_feature_centroids.csv|1879250|
|reports/planA_k_metaniche_hardening/stratified_pilot_outputs/metaniche_feature_centroids.csv|1840556|
|reports/planA_k_cellrank_aligned_fate_freeze/figures/spatial_absorption_maps/cellrank_aligned_spatial_absorption_maps.pdf|1822794|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/082421_D21_m1_1_slice_2_membership_probabilities.png|1665326|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/062921_D0_m3a_1_slice_2_membership_probabilities.png|1580987|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/092421_D3_m1_1_slice_2_membership_probabilities.png|1576828|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/092421_D3_m1_1_slice_3_membership_probabilities.png|1568975|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/100221_D9_m5_2_slice_1_single_slice_sanity.png|1556738|
|reports/planA_k_final_result_package/figures/main_figures/Figure_2_UMAP_GPCCA_k6_macrostate_atlas.png|1517997|
|reports/planA_k_final_result_package/figures/Figure_2_GPCCA_k6_macrostate_atlas.png|1517228|
|reports/planA_k_final_result_package/figures/supplementary_figures/Figure_S1_original_PCA_state_space_atlas.png|1517228|
|reports/planA_k_final_result_package/figures/Figure_3_representative_per_slice_macrostate_maps.png|1497588|
|reports/planA_k_final_result_package/figures/main_figures/Figure_3_representative_per_slice_macrostate_maps.png|1497588|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/062921_D0_m3a_1_slice_1_membership_probabilities.png|1492360|
|reports/planA_k_full_result_visualization_feature_only/figures/key_results/Figure_2_per_slice_macrostate_maps.png|1479401|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/062921_D9_m5_1_slice_1_single_slice_sanity.png|1375924|
|reports/planA_k_final_result_package/figures/Figure_5_terminal_M5_absorption_fate_probability_maps.png|1354502|
|reports/planA_k_final_result_package/figures/main_figures/Figure_5_PlanA_inferred_absorption_fate_probability_to_M5.png|1354502|
|reports/planA_k_full_result_visualization_feature_only/figures/key_results/Figure_1_metaniche_state_atlas.png|1303929|
|reports/planA_k_final_result_package/tables/umap_macrostate_atlas_table.tsv|1273161|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/082421_D21_m1_1_slice_2_single_slice_sanity.png|1187297|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/092421_D3_m1_1_slice_2_single_slice_sanity.png|1168562|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/092421_D3_m1_1_slice_3_single_slice_sanity.png|1154385|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/062921_D0_m3a_1_slice_2_single_slice_sanity.png|1152711|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/082421_D21_m1_1_slice_1_membership_probabilities.png|1145934|
|reports/planA_k_spatial_kernel_integrity_audit/figures/per_slice/062921_D0_m3a_1_slice_1_single_slice_sanity.png|1096540|
