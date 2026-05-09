# DARLIN-ST-00 Mixed Sailu ST Processing Plan

Generated: 2026-05-09T05:37:43Z

## Scope

- Mixed ST sample ID: `Sailu_0313_19A_mixed_ST`
- BC folder: `/data/zhutao/nichefate_data/202603131741_Pro019_A_SE30_1_0313_19A_BC`
- E/ST folder: `/data/zhutao/nichefate_data/202604271751_2211180001_B_1_ZHY_XBB_E_PE100_100`
- R/DARLIN folder recorded for context only: `/data/zhutao/nichefate_data/202604271754_C2302270016_A_1_ZHY_XBB_R_PE100_250`
- Meiji RA/TA barcode data is excluded from this ST node.

This node is feasibility and smoke-run planning only. It does not run ST processing, DARLIN preprocessing, Snakemake, NicheFate, GPCCA/CellRank, fate probability computation, data movement, symlink creation, `/ssd` writes, or Git commits.

## Updated Interpretation

The mixed ST dataset is not rejected. `Sailu_0313_19A_mixed_ST` is conceptually allowed as an exploratory mixed/unresolved ST input because a preliminary ST processing/QC pass can be useful before Brain/Pancreas demultiplexing is resolved.

The dataset must not be labeled as `Brain_E1_ST`, `E1-only_ST`, or any final Brain-specific ST input. Local split-key audits still show that Brain `031319A-E1` and Pancreas `031319A-E2` cannot be separated from the currently available local files.

## Current Feasibility Decision

- BC folder found: yes.
- E/ST folder found: yes.
- Existing Sailu/Salus raw ST processing entrypoint found: no.
- Existing output contract found: no.
- Mixed ST smoke-run ready: no.

The blocker is practical, not conceptual: the Sailu/Salus raw ST processing script entrypoint and input/output contract are missing. Generic NicheFate spatial graph utilities are downstream tools that expect processed AnnData/spatial coordinates; they are not raw Sailu ST preprocessing entrypoints.

## Expected Use of BC and E Folders

A valid mixed ST processing script should treat the BC and E folders as a single unresolved sample:

1. Use `BC_SE30` run outputs for spatial barcode/chip barcode handling if the Sailu/Salus ST pipeline requires them.
2. Use `E_PE100_100` lane FASTQs as expression/ST sequencing input.
3. Preserve sample-level label `Sailu_0313_19A_mixed_ST` in all output metadata.
4. Do not split or relabel spots as Brain/Pancreas during the smoke run.
5. Do not integrate `R_PE100_250` or Meiji RA/TA barcode evidence in this node.

## Expected Output Objects

The eventual smoke run should produce, at minimum:

- QC report for raw input parsing and spatial barcode/expression read handling.
- Expression matrix or count matrix.
- Spot/cell metadata table.
- Coordinate table or spatial barcode-to-coordinate table if supported by the ST pipeline.
- Processing log with exact command, software version, and input/output paths.

If clustering or marker outputs are produced by the user ST scripts, they are exploratory and must remain under the mixed sample label.

## QC Metrics to Inspect Later

- FASTQ lane detection and read pair completeness.
- Read structure recognition for BC and E components.
- Spatial barcode parsing/calling success rate.
- Fraction of reads/spots retained after filtering.
- UMI/read count distributions per spot.
- Detected gene counts per spot.
- Mitochondrial/ribosomal or equivalent QC metrics if available.
- Spatial coordinate completeness and duplicate coordinate checks.
- Whether output metadata contains enough fields to trace back to BC and E inputs.

## Exploratory Tissue Annotation Boundary

Brain/Pancreas annotation is future post-processing only. DARLIN-ST-00 does not run marker analysis, does not create a Brain mask, and does not claim provider-level demultiplexing truth.

After a mixed processed matrix exists, putative tissue regions may be annotated using expression markers, clustering, and spatial continuity. Those labels must be named `putative_brain_like` and `putative_pancreas_like` or equivalent, not provider-confirmed Brain/Pancreas labels.

## Pre-Write Git State

Before writing ST00 outputs, `git status --short` showed the following existing untracked DARLIN outputs. These files were not modified or overwritten by this node:

```text
?? configs/darlin/meiji_e1_ra_ta_staging_plan.yaml
?? docs/darlin/darlin_01a_bc_spatial_barcode_audit.md
?? docs/darlin/darlin_01a_bc_st_split_feasibility.md
?? docs/darlin/darlin_01b_cfg_type_template_review.md
?? docs/darlin/darlin_01b_dependency_resolution.md
?? docs/darlin/darlin_01b_next_step_recommendation.md
?? docs/darlin/darlin_01b_official_config_contract.md
?? docs/darlin/darlin_01b_staging_contract.md
?? docs/darlin/darlin_sailu_pe100_100_split_key_inspection.md
?? reports/darlin_onboarding/darlin_01a_bc_e_r_relationship_probe.csv
?? reports/darlin_onboarding/darlin_01a_bc_fastq_header_probe.csv
?? reports/darlin_onboarding/darlin_01a_bc_file_inventory.csv
?? reports/darlin_onboarding/darlin_01a_bc_remaining_questions.md
?? reports/darlin_onboarding/darlin_01a_bc_saluscallfile_probe.csv
?? reports/darlin_onboarding/darlin_01a_bc_split_decision.json
?? reports/darlin_onboarding/darlin_01b_cfg_type_template_candidates.csv
?? reports/darlin_onboarding/darlin_01b_config_fields_inventory.csv
?? reports/darlin_onboarding/darlin_01b_dependency_status.csv
?? reports/darlin_onboarding/darlin_01b_dryrun_readiness_decision.csv
?? reports/darlin_onboarding/darlin_01b_staging_file_map.csv
?? reports/darlin_onboarding/darlin_01b_summary.json
?? reports/darlin_onboarding/sailu_pe100_100_fastq_header_probe.csv
?? reports/darlin_onboarding/sailu_pe100_100_html_report_probe.csv
?? reports/darlin_onboarding/sailu_pe100_100_metadata_inventory.csv
?? reports/darlin_onboarding/sailu_pe100_100_split_decision.json
```

## Exact Next Step

- ST track: provide the Sailu/Salus raw ST processing script entrypoint and input/output contract, then start `DARLIN-ST-01` mixed ST smoke planning.
- Barcode track: continue `DARLIN-01B` official DARLIN `cfg_type`/template/staging-contract resolution for Meiji RA/TA.
