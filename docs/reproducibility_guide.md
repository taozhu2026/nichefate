# Nichefate Reproducibility Guide

Generated for ReviewPack-01 on 2026-05-08.

This guide records how existing outputs were produced or can be reproduced in a
controlled rerun. ReviewPack-01 itself must not run these commands. During
ReviewPack, the listed modules are safe to inspect only.

## ReviewPack Rule

Do not run during ReviewPack-01:

- M0, M1, M2, M3, M4, or M5 runners
- GPCCA, pyGPCCA, or CellRank
- terminal-state design
- fate-probability propagation
- P_fate propagation
- BranchSBM
- barcode or DARLIN preprocessing
- data movement or writes to `/ssd`

## Module Commands And Artifacts

| Module | Repro command family | Inputs | Expected outputs | ReviewPack mode |
|---|---|---|---|---|
| M0 | `conda run -n omicverse python scripts/m0_00_check_environment.py --config configs/m0_merfish_colitis.yaml`; then `m0_01` through `m0_06` | Dryad h5ad files under `/data/zhutao/datasets/merfish_colitis_moffitt_2024/raw` or configured home raw root | `/home/zhutao/scratch/nichefate/m0`, M0 reports | Inspect only |
| M1 | `conda run -n omicverse python scripts/m1_04_build_niche_full_by_slice.py --config configs/m1_niche_construction.yaml` | M0 by-slice outputs | `/home/zhutao/scratch/nichefate/m1/by_slice`, M1 reports | Inspect only |
| M2 | `conda run -n omicverse python scripts/m2_02_build_full_representation_by_slice.py --config configs/m2_niche_representation.yaml` | M1 by-slice outputs and global schema | `/home/zhutao/scratch/nichefate/m2/by_slice`, feature schema and summary reports | Inspect only |
| M3-v1 | `conda run -n omicverse python scripts/m3_15_run_full_m3_by_shard.py --config configs/m3_transition_kernel.yaml`; freeze with `m3_16` | M2 by-slice representation | `/home/zhutao/scratch/nichefate/m3/full_by_shard`, final freeze manifest | Inspect only |
| M3-v2 | `conda run -n omicverse python scripts/m3_v2_06_run_full_by_shard.py --config configs/m3_v2_full_production.yaml`; benchmark with `m3_v2_08` | M3-v1 edges, M2 features, M4A/M4C/M4E context | `/home/zhutao/scratch/nichefate/m3_v2/full_by_shard`, v2 benchmark reports | Inspect only |
| M4A | `conda run -n omicverse python scripts/m4a_01_assemble_global_transition_object.py --config configs/m4a_markov_assembly.yaml`; v2 with `m4a_v2_01` | M3/M3-v2 edge shards, M2 metadata | Markov sparse matrices and node tables under `/home/zhutao/scratch/nichefate/m4a*` | Inspect only |
| M4B | `conda run -n omicverse python scripts/m4b_01_design_terminal_macrostates.py --config configs/m4b_markov_terminal_design.yaml`; feasibility review with `m4b_02` | M4A matrices and node table | terminal macrostate design and feasibility reports | Inspect only |
| M4C / P_fate | `conda run -n omicverse python scripts/m4c_01_compute_markov_fate_probabilities.py --config configs/m4c_fate_probability.yaml`; freeze with `m4c_02`; PlanA freeze with `planA_00` | M4A absorbing matrix, M4B terminal assignments | `/home/zhutao/scratch/nichefate/m4c/fate_probabilities`, PlanA freeze reports | Inspect only |
| M4C-v2 | `conda run -n omicverse python scripts/m4c_v2_01_run_fate_propagation.py --config configs/m4c_v2_fate_propagation.yaml`; benchmark with `m4c_v2_03` | M4A-v2 matrices, endpoint mapping, M4C v1 baseline | `/home/zhutao/scratch/nichefate/m4c_v2`, benchmark reports | Inspect only |
| M4E | `conda run -n omicverse python scripts/m4e_01_endpoint_annotation_review.py`; `m4e_02`; `m4e_03` | M4C/P_fate outputs and M2/M4A node context | endpoint and neighborhood annotation reports | Inspect only |
| K_gpcca | `conda run -n omicverse python scripts/k_gpcca_00_design.py`; pilot/revision scripts through `k_gpcca_04_kernel_revision_pilot.py` | M2, M3/M3-v2, M4A/M4E context | design, pilot, benchmark, and bounded revision outputs under `/home/zhutao/scratch/nichefate/k_gpcca_*` | Inspect only |
| DARLIN / barcode adapter | No ReviewPack command | Future official/lab-standard DARLIN preprocessing outputs | Future adapter inputs | Deferred |
| BranchSBM / Plan B | No ReviewPack command | Future branch-level evidence | Future model outputs | Deferred |

## Existing Scratch Output Paths

- `/home/zhutao/scratch/nichefate/m0`
- `/home/zhutao/scratch/nichefate/m1`
- `/home/zhutao/scratch/nichefate/m2`
- `/home/zhutao/scratch/nichefate/m3`
- `/home/zhutao/scratch/nichefate/m3_v2`
- `/home/zhutao/scratch/nichefate/m4a`
- `/home/zhutao/scratch/nichefate/m4a_v2`
- `/home/zhutao/scratch/nichefate/m4b`
- `/home/zhutao/scratch/nichefate/m4c`
- `/home/zhutao/scratch/nichefate/m4c_v2`
- `/home/zhutao/scratch/nichefate/m4d`
- `/home/zhutao/scratch/nichefate/m4e`
- `/home/zhutao/scratch/nichefate/planA_freeze`
- `/home/zhutao/scratch/nichefate/k_gpcca_design`
- `/home/zhutao/scratch/nichefate/k_gpcca_pilot`
- `/home/zhutao/scratch/nichefate/k_gpcca_pilot_benchmark`
- `/home/zhutao/scratch/nichefate/k_gpcca_revision`

