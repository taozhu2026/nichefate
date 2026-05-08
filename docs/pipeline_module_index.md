# Nichefate Pipeline Module Index

Generated for ReviewPack-01 on 2026-05-08.

This index maps the current milestone-style repository into future production
module names. Heavy outputs are external paths and should be documented, not
uploaded to GitHub.

## Legacy To Production Mapping

| Legacy milestone | Future production module |
|---|---|
| M0 | SpatialDatasetAdapter |
| M1 | NicheBuilder |
| M2 | NicheEncoder |
| M3-v1 | TransitionEvidence[pseudo_broad] |
| M3-v2 | TransitionEvidence[pseudo_sharpened] |
| M4A | KernelAssembly |
| M4C | EndpointMarkovInference / P_fate |
| M4E | BiologicalAnnotation |
| K_gpcca | GPCCAMacrostateInference experimental branch |

## Module Index

| Module | Main scripts | Configs | Key reports and summaries | External output roots |
|---|---|---|---|---|
| M0 | `scripts/m0_00*.py` through `scripts/m0_06*.py` | `configs/m0_merfish_colitis.yaml`, `configs/m0_merfish_colitis_home.yaml` | `/home/zhutao/scratch/nichefate/m0/reports/m0_report.md` | `/home/zhutao/scratch/nichefate/m0`, `/data/zhutao/datasets/merfish_colitis_moffitt_2024/raw` |
| M1 | `scripts/m1_00*.py` through `scripts/m1_04*.py` | `configs/m1_niche_construction.yaml` | `/home/zhutao/scratch/nichefate/m1/reports/m1_full_by_slice_summary.md` | `/home/zhutao/scratch/nichefate/m1/by_slice` |
| M2 | `scripts/m2_00*.py` through `scripts/m2_02*.py` | `configs/m2_niche_representation.yaml` | `/home/zhutao/scratch/nichefate/m2/reports/m2_full_by_slice_summary.md`, `m2_full_feature_schema.json` | `/home/zhutao/scratch/nichefate/m2/by_slice` |
| M3-v1 | `scripts/m3_00*.py` through `scripts/m3_16*.py` | `configs/m3_transition_kernel.yaml` | `/home/zhutao/scratch/nichefate/m3/reports/m3_full_m3_final_freeze_manifest.json`, `m3_full_m3_run_summary.md` | `/home/zhutao/scratch/nichefate/m3/full_by_shard` |
| M3-v2 | `scripts/m3_v2_00*.py` through `scripts/m3_v2_08*.py` | `configs/m3_v2_pilot.yaml`, `configs/m3_v2_full_production.yaml` | `/home/zhutao/scratch/nichefate/m3_v2/reports/m3_v2_full_production_report.md`, `/home/zhutao/scratch/nichefate/m3_v2_benchmark/m3_v1_vs_v2_edge_benchmark_summary.json` | `/home/zhutao/scratch/nichefate/m3_v2/full_by_shard` |
| M4A | `scripts/m4a_01_assemble_global_transition_object.py`, `scripts/m4a_v2_*.py` | `configs/m4a_markov_assembly.yaml`, `configs/m4a_v2_assembly.yaml` | `/home/zhutao/scratch/nichefate/m4a/reports/m4a_assembly_report.md`, `/home/zhutao/scratch/nichefate/m4a_v2/reports/m4a_v2_02_full_assembly_report.md` | `/home/zhutao/scratch/nichefate/m4a`, `/home/zhutao/scratch/nichefate/m4a_v2` |
| M4B | `scripts/m4b_01_design_terminal_macrostates.py`, `scripts/m4b_02_review_markov_gpcca_feasibility.py` | `configs/m4b_markov_terminal_design.yaml` | `/home/zhutao/scratch/nichefate/m4b/reports/m4b_terminal_macrostate_design_summary.json`, `m4b_markov_gpcca_feasibility_summary.json` | `/home/zhutao/scratch/nichefate/m4b` |
| M4C / P_fate | `scripts/m4c_*.py`, `scripts/m4c_v2_*.py`, `scripts/planA_00_freeze_p_fate_branch.py` | `configs/m4c_fate_probability.yaml`, `configs/m4c_v2_fate_propagation.yaml` | `/home/zhutao/scratch/nichefate/m4c/reports/m4c_markov_fate_final_freeze_summary.json`, `/home/zhutao/scratch/nichefate/planA_freeze/planA_freeze_summary.json` | `/home/zhutao/scratch/nichefate/m4c`, `/home/zhutao/scratch/nichefate/m4c_v2`, `/home/zhutao/scratch/nichefate/planA_freeze` |
| M4D | `scripts/m4d_*.py`, `scripts/m4v_*.py` | `configs/m4d_markov_macrostate_visualization.yaml` | `/home/zhutao/scratch/nichefate/m4d/reports/m4d_standard_gpcca_environment_report.md`, `m4d_supernode_qc_summary.json` | `/home/zhutao/scratch/nichefate/m4d` |
| M4E | `scripts/m4e_*.py` | Configured through existing M4C/M4E paths | `/home/zhutao/scratch/nichefate/m4e/reports/m4e_endpoint_biological_annotation_report.md`, `m4e_neighborhood_annotation_report.md` | `/home/zhutao/scratch/nichefate/m4e` |
| K_gpcca | `scripts/k_gpcca_00_design.py` through `scripts/k_gpcca_04_kernel_revision_pilot.py` | `configs/k_gpcca_pilot.yaml`, `configs/k_gpcca_revision.yaml` | `/home/zhutao/scratch/nichefate/k_gpcca_revision/k_gpcca_04_summary.json`, `/home/zhutao/scratch/nichefate/k_gpcca_revision/reports/k_gpcca_04_completion_check.md` | `/home/zhutao/scratch/nichefate/k_gpcca_*` |
| DARLIN / barcode adapter | None in production pipeline | Future adapter only | PlanA positioning reports | Future external barcode/DARLIN roots |
| BranchSBM / Plan B | None in production pipeline | Future branch only | PlanA positioning reports | Future external BranchSBM root |

## Packaging Notes

- `scripts/`, `configs/`, `docs/`, `tests/`, `src/`, `README.md`,
  `pyproject.toml`, `requirements.txt`, and `environment.yml` are lightweight
  repository candidates.
- Scratch roots listed above are external artifacts.
- `.h5ad`, `.npz`, `.parquet`, generated `.csv`, raw data, and working
  directories should remain out of GitHub unless deliberately curated as tiny
  metadata examples.

