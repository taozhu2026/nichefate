# NicheFate Algorithm Module Registry

This registry makes the concrete algorithm modules visible across the
NicheFate framework. It complements the layer-level architecture in
`docs/pipeline_module_index.md`.

## Shared Core Algorithms

| Module | Public name | Code surface | Input | Output | Status |
|---|---|---|---|---|---|
| M0 | SpatialInputContract | `src/nichefate/planA_st_only/spatial_dataset_adapter.py`; `src/nichefate/io.py`; `src/nichefate/spatial.py` | ST sample package | Standardized spatial objects | frozen |
| M1 | NicheConstructor | `src/nichefate/planA_st_only/niche_builder.py`; `src/nichefate/niche.py` | M0 objects and graphs | Multi-scale niche feature tables | frozen |
| M2 | NicheEncoder | `src/nichefate/planA_st_only/niche_encoder.py`; `src/nichefate/representation.py`; `src/nichefate/embedding.py` | M1 feature tables | Anchor-level representation matrix | frozen |
| M2_REP | NicheRepresentationBuilder | `src/nichefate/representation.py` | Schema-aligned M1 rows | Deterministic M2 feature matrix | frozen |
| M2.5 | MetaNicheCoarsener | `src/nichefate/planA_st_only/metaniche_coarsener.py`; `src/nichefate/planA_k/metaniche.py` | M2 representations | Metaniche assignments and centroids | frozen |
| M3 | TransitionKernelBuilder | `src/nichefate/planA_st_only/transition_evidence.py`; `src/nichefate/planA_k/sparse_kernel.py` | M2.5 states | Transition evidence and kernels | frozen |

`NicheEncoder` is present as a public facade. It is not a monolithic class:
the facade re-exports the M2 implementation components in
`nichefate.embedding` and `nichefate.representation`.

## Evidence-Specific Modules

| Module | Public name | Code surface | Input | Output | Status |
|---|---|---|---|---|---|
| E0 | STOnlyEvidenceBuilder | `src/nichefate/planA_st_only/` | M0-M2.5 without barcode evidence | ST-only state evidence | frozen |
| E1 | LineageEvidenceAdapter | `src/nichefate/lineage/evidence.py`; `src/nichefate/lineage/input_contract.py` | Processed lineage evidence | Canonical lineage tables | ready_with_warnings |
| E2 | DARLINJointCloneCaller | `src/nichefate/darlin/joint_clone_calling.py` | Valid CA/TA/RA lineage alleles | Validated joint clone assignments | ready_with_warnings |
| E3 | CloneNicheIntegrator | `src/nichefate/lineage/clone_niche.py` | Joint clones and spatial unit maps | Clone x niche summaries | ready_with_warnings |
| E4 | LineageAwareFeatureBuilder | `src/nichefate/lineage/clone_matrix.py`; `src/nichefate/planA_l/representation.py` | Substrate features plus clone variables | Lineage-aware representation channels | design |

Lineage-aware evidence adds representation channels. It does not replace
NicheEncoder or the shared M0-M2.5 substrate.

## Dynamics Modules

| Module | Public name | Code surface | Input | Output | Status |
|---|---|---|---|---|---|
| PlanA | PlanA Markov-GPCCA Engine | `src/nichefate/planA_st_only/gpcca_macrostate_inference.py`; `src/nichefate/planA_k/full_gpcca.py` | M2.5 states and kernels | PlanA ST-only macrostate outputs | frozen |
| PlanA-L | PlanA Lineage-Aware Design Interface | `src/nichefate/lineage/dynamics_interface.py`; `src/nichefate/planA_l/` | M2.5 states plus clone matrices | Design-only lineage-aware dynamics contract | design |
| PlanB | PlanB Branch/Clone-Aware Engine | `src/nichefate/planB_nichebranchsbm/`; `src/nichefate/nichebranchsbm/` | M2.5 states with optional clone variables | In-progress branch/clone-aware artifacts | design |

Directionality still requires time, perturbation, or another valid prior.
L126 serial sections are not used to claim temporal directionality.

## Benchmark Wrappers

| Wrapper | Maps to | Code surface | Benchmark |
|---|---|---|---|
| MERFISH/Cadinu/Moffitt workflow scripts | M0-M2.5, E0, PlanA | `scripts/m0_*`; `scripts/m1_*`; `scripts/m2_*`; `scripts/planA_k_*`; `scripts/planA_st_only_*` | ST-only baseline |
| L126 barcode adapter scripts | E1 LineageEvidenceAdapter | `scripts/planC_l126_barcode_adapter_round1.py`; `scripts/nichefate_lineage_01_build_evidence_adapter.py` | L126 spatio-DARLIN |
| L126 DARLIN joint clone scripts | E2 DARLINJointCloneCaller | `scripts/planC_l126_darlin_style_clone_calling_audit.py`; `scripts/nichefate_darlin_03_call_joint_clones.py` | L126 spatio-DARLIN |
| L126 clone x niche scripts | E3 CloneNicheIntegrator | `scripts/planC_l126_darlin_joint_clone_niche_v1.py`; `scripts/nichefate_lineage_04_integrate_clones_to_niches.py` | L126 spatio-DARLIN |

The machine-readable registry lives at
`configs/module_registry/nichefate_module_registry.json`.
