# Script To Module Mapping

This page maps older milestone or benchmark scripts to the concrete public
module registry. Script names are retained for provenance and reproducibility;
they are not public algorithm names.

## Shared Core And PlanA Scripts

| Script family | Concrete module |
|---|---|
| `scripts/m0_*` | M0 SpatialInputContract |
| `scripts/m1_*` | M1 NicheConstructor |
| `scripts/m2_*` | M2 NicheEncoder / NicheRepresentationBuilder |
| `scripts/planA_k_04_*` through `scripts/planA_k_23_*` | M2.5 MetaNicheCoarsener |
| `scripts/m3_*`, `scripts/m3_v2_*`, `scripts/planA_k_25_*`, `scripts/planA_k_26_*` | M3 TransitionKernelBuilder |
| `scripts/planA_k_27_*` through `scripts/planA_k_39_*` | PlanA Markov-GPCCA and result packaging modules |

## Lineage Benchmark Scripts

| Script family | Concrete module |
|---|---|
| `scripts/planC_l126_barcode_adapter_*` | E1 LineageEvidenceAdapter benchmark wrapper |
| `scripts/planC_l126_darlin_style_clone_calling_*` | E2 DARLINJointCloneCaller benchmark wrapper |
| `scripts/planC_l126_darlin_joint_clone_niche_v1.py` | E2 DARLINJointCloneCaller and E3 CloneNicheIntegrator benchmark wrapper |
| `scripts/nichefate_lineage_*` | Generic lineage-aware wrapper scripts |
| `scripts/nichefate_darlin_03_call_joint_clones.py` | Generic DARLINJointCloneCaller wrapper |

## Probes And Ablations

| Script family | Classification |
|---|---|
| `scripts/planC_l126_darlin_clone_integration_round1.py` | legacy_or_ablation |
| `scripts/planC_l126_darlin_clone_signature_round2.py` | superseded |
| `scripts/planC_l126_clone_membership_rescue_round2_1.py` | superseded |
| `scripts/planC_l126_planA_lineage_*` | legacy_or_ablation |

The current public registry is `docs/algorithm_module_registry.md`.
