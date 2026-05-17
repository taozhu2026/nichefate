# nichefate

`nichefate` is a spatial transcriptomics engineering project for niche-level
fate inference. The current PlanA-ST-only v1 pipeline is barcode-free and uses
spatial transcriptomics data only. Large raw data, scratch outputs, production
matrices, and generated scientific artifacts are external to the repository.

## PlanA-ST-only v1 status

- The M0-M2.5 spatial niche backbone is frozen and merged.
- The barcode-free PlanA-ST-only v1 pipeline has completed through corrected
  feature-only Kmix_A, corrected full GPCCA k=6, macrostate annotation,
  terminal audit, PlanA-inferred terminal/sink macrostate M5, Kmix_A
  absorption/fate probability to M5, final result packaging, and final
  visualization package QA.
- Production-style module organization is documented in
  [PlanA-ST-only v1 production modules](docs/planA_st_only_v1_production_modules.md).
- Early backbone facades re-export existing modules; later frozen v1 result
  stages are indexed as documented-only boundaries pending a clean standalone
  PlanA-K refactor.
- The final result package is under `reports/planA_k_final_result_package/`.
- This is ST-only / barcode-free. DARLIN/barcode integration is the next
  separate development stage and is not part of this freeze.

## Current Project Index

Start with the PlanA-ST-only v1 module and result indexes:

- [Pipeline module index](docs/pipeline_module_index.md)
- [PlanA-ST-only v1 production modules](docs/planA_st_only_v1_production_modules.md)
- [PlanA-ST-only v1 result index](reports/planA_st_only_v1_index/00_PLAN_A_ST_ONLY_V1_INDEX.md)
- [PlanA-ST-only v1 claim boundary](reports/planA_st_only_v1_index/04_CLAIM_BOUNDARY.md)
- [PlanA-ST-only module reorg validation](reports/git_update_planA_st_only_modules/03_VALIDATION.md)
- [Project status checkpoint](docs/project_status_checkpoint.md)
- [Reproducibility guide](docs/reproducibility_guide.md)
- [Environment and dependencies](docs/environment_and_dependencies.md)
- [GitHub packaging plan](docs/github_packaging_plan.md)
- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Git initialization plan](docs/git_initialization_plan.md)
- [PlanA-K M0-M2.5 spatial niche backbone](docs/planA_k_m0_m2_5_backbone.md)

Checkpoint status:

- Frozen backbone: M0-M2.5 spatial niche construction.
- Completed ST-only v1: corrected feature-only Kmix_A, corrected GPCCA k=6,
  PlanA-inferred terminal/sink M5, absorption/fate probability to M5, result
  package, and visualization package QA.
- Deferred: DARLIN / barcode adapter and BranchSBM / Plan B.
- External: large data and generated outputs under `/data` and scratch roots.

## Server Layout

- Code root: `/home/zhutao/projects/nichefate`
- Raw dataset root: `/data/zhutao/datasets/merfish_colitis_moffitt_2024/raw`
- Temporary M0 output root: `/data/zhutao/work/nichefate/m0`

Keep raw data and temporary working outputs outside the project code root. Do
not write large raw data, intermediate files, caches, or model outputs directly
into the repository.

## Raw Data Policy

For the first M0 workflow, place only the required Dryad files in the raw data
directory:

- `adata.h5ad`
- `adata_day35.h5ad`
- `README.md`

Optional:

- `ligand_receptor_pair_masterlist.csv`

Dryad lists `adata.h5ad` as 17.96 GB and `adata_day35.h5ad` as 1.51 GB, so the
required core download is about 19.5 GB plus tiny README/LR files. Do not fully
download the complete Dryad dataset unless it is explicitly needed; Dryad lists
the full dataset at 108.70 GB. Avoid downloading `X.csv`, `X_raw.csv`,
transcript-level RNA metadata CSVs, or all supplementary CSVs for the initial M0
work; they are large and mostly redundant for this workflow.

## Environment

Historical execution used the existing `omicverse` conda environment:

- Environment path: `/home/zhutao/software/conda_envs/omicverse`
- Python: `3.10.14`
- `scanpy`, `anndata`, `sklearn`, `scipy`, and core plotting/data packages are
  available. The `omicverse` package is available in that historical
  environment, but the environment name does not define the project identity.

Do not modify the existing `omicverse` environment as part of packaging.
`squidpy`, `spatialdata`, and `harmonypy` are not M0 v1 dependencies.

M4D-01b uses a separate isolated `nichefate-gpcca` environment for standard
GPCCA backend validation. That environment is only for pyGPCCA/CellRank
interface checks and must not be used to modify `omicverse`.

Run commands with:

```bash
cd /home/zhutao/projects/nichefate
conda run -n omicverse python scripts/m0_00_check_environment.py --config configs/m0_merfish_colitis.yaml
```

`environment.yml` is kept as a historical future reproducible environment
specification. Draft environments under `envs/` are not fully locked until
regression validation is complete.

## Pipeline Entry Points

The scripts are staged entry points for reproducibility. Do not use README
commands as a quickstart for full production runs. See
[Quickstart](docs/quickstart.md) for lightweight inspection and test guidance.

```bash
cd /home/zhutao/projects/nichefate
conda run -n omicverse python scripts/m0_00_download_core_dryad_files.py --config configs/m0_merfish_colitis.yaml --no-download
conda run -n omicverse python scripts/m0_00_check_environment.py --config configs/m0_merfish_colitis.yaml
conda run -n omicverse python scripts/m0_01_inspect_raw_anndata.py --config configs/m0_merfish_colitis.yaml
conda run -n omicverse python scripts/m0_02_build_m0_anndata.py --config configs/m0_merfish_colitis.yaml
conda run -n omicverse python scripts/m0_03_compute_embeddings.py --config configs/m0_merfish_colitis.yaml
conda run -n omicverse python scripts/m0_04_build_spatial_graphs.py --config configs/m0_merfish_colitis.yaml
conda run -n omicverse python scripts/m0_05_export_m0_objects.py --config configs/m0_merfish_colitis.yaml
conda run -n omicverse python scripts/m0_06_make_qc_report.py --config configs/m0_merfish_colitis.yaml
```

Do not claim true lineage for this dataset. It has no lineage barcode; M0 only
prepares time-anchored spatial transcriptomics data for later pseudo-lineage
inference.

## Markov Macrostate Routes

- M4C is Markov baseline v1 using final-time clustering targets.
- M4D is the standard GPCCA/CellRank-inspired Markov route.
- `scipy_pcca_like_diagnostic_fallback` is diagnostic only and is not the main
  macrostate algorithm.

## Development Checks

```bash
cd /home/zhutao/projects/nichefate
conda run -n omicverse python scripts/m0_00_check_environment.py --config configs/m0_merfish_colitis.yaml
conda run -n omicverse python -m pytest tests
```
