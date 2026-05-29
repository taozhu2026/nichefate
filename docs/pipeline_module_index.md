# NicheFate Pipeline Module Index

This index maps NicheFate at two levels:

1. Layer-level architecture: shared substrate, evidence regimes, and dynamics
   engines.
2. Concrete algorithm modules: public module names and current code surfaces.

The machine-readable registry is
`configs/module_registry/nichefate_module_registry.json`.

## Layer-Level Architecture

### Shared Spatial-Niche Substrate

M0-M2.5 are shared substrate modules. They are not ST-only internals and they
are not lineage-specific modules.

| Module | Responsibility | Current surface |
|---|---|---|
| M0 | ST/sample/input contract and standardized spatial objects | `docs/m0_design.md`; `src/nichefate/planA_st_only/spatial_dataset_adapter.py` |
| M1 | Spatial niche construction | `src/nichefate/planA_st_only/niche_builder.py`; `src/nichefate/niche.py` |
| M2 | Niche encoding and representation | `src/nichefate/planA_st_only/niche_encoder.py`; `src/nichefate/representation.py` |
| M2.5 | Metaniche / niche-state coarsening | `src/nichefate/planA_st_only/metaniche_coarsener.py`; `src/nichefate/planA_k/metaniche.py` |
| M3 | Transition evidence / kernel construction | `src/nichefate/planA_st_only/transition_evidence.py`; `src/nichefate/planA_k/sparse_kernel.py` |

### Evidence Regimes

| Module | Evidence regime | Responsibility | Current surface |
|---|---|---|---|
| E0 | ST-only / lineage-free | Use expression, spatial coordinates, metadata, and niche representations without barcode lineage evidence | `src/nichefate/planA_st_only/` |
| E1 | Lineage-aware | Adapt processed lineage evidence into NicheFate cellbin-level contracts | `src/nichefate/lineage/evidence.py` |
| E2 | Lineage-aware | Call validated DARLIN-style joint clones | `src/nichefate/darlin/joint_clone_calling.py` |
| E3 | Lineage-aware | Build clone x niche representation summaries | `src/nichefate/lineage/clone_niche.py` |
| E4 | Lineage-aware design | Add clone composition variables as representation channels | `src/nichefate/lineage/clone_matrix.py`; `src/nichefate/planA_l/representation.py` |

E1-E4 add lineage evidence and clone state variables. They do not replace the
shared M0-M2.5 substrate.

### Dynamics Engines

| Engine | Status | Consumes |
|---|---|---|
| PlanA-ST | Frozen baseline | M0-M2.5 plus E0 |
| PlanA lineage-aware | Design / in progress | M0-M2.5 plus E1-E4 |
| PlanB branch/clone-aware dynamics | Design / in progress | M0-M2.5 plus E0 or E1-E4, depending on dataset |

Directionality in dynamics engines must come from time, perturbation, or an
explicit biological prior. Clone evidence can add state variables or support
terms when the dataset supports that interpretation.

## Concrete Algorithm Module Table

| Public module | Layer | Code surface | Input | Output | Status | Benchmark examples |
|---|---|---|---|---|---|---|
| SpatialInputContract | M0 | `src/nichefate/planA_st_only/spatial_dataset_adapter.py`; `src/nichefate/io.py`; `src/nichefate/spatial.py` | ST sample package | Standardized spatial objects | frozen | MERFISH/Cadinu/Moffitt; L126 ST side |
| NicheConstructor | M1 | `src/nichefate/planA_st_only/niche_builder.py`; `src/nichefate/niche.py` | M0 objects and graphs | Multi-scale niche feature tables | frozen | MERFISH/Cadinu/Moffitt |
| NicheEncoder | M2 | `src/nichefate/planA_st_only/niche_encoder.py`; `src/nichefate/embedding.py`; `src/nichefate/representation.py` | M1 feature tables | Anchor-level representation matrix | frozen | MERFISH/Cadinu/Moffitt; L126 shared substrate |
| NicheRepresentationBuilder | M2 | `src/nichefate/representation.py` | Schema-aligned M1 rows | Pivoted M2 feature matrix | frozen | MERFISH/Cadinu/Moffitt |
| MetaNicheCoarsener | M2.5 | `src/nichefate/planA_st_only/metaniche_coarsener.py`; `src/nichefate/planA_k/metaniche.py` | M2 representations | Metaniche assignments and centroids | frozen | MERFISH/Cadinu/Moffitt; L126 spatial units |
| TransitionKernelBuilder | M3 | `src/nichefate/planA_st_only/transition_evidence.py`; `src/nichefate/planA_k/sparse_kernel.py` | M2.5 states | Transition evidence and kernels | frozen | PlanA-ST baseline |
| STOnlyEvidenceBuilder | E0 | `src/nichefate/planA_st_only/` | M0-M2.5 without lineage evidence | ST-only state evidence | frozen | MERFISH/Cadinu/Moffitt |
| LineageEvidenceAdapter | E1 | `src/nichefate/lineage/evidence.py`; `src/nichefate/lineage/input_contract.py` | Processed lineage evidence | Canonical lineage tables | ready_with_warnings | L126 spatio-DARLIN |
| DARLINJointCloneCaller | E2 | `src/nichefate/darlin/joint_clone_calling.py` | Valid CA/TA/RA lineage alleles | Validated joint clone assignments | ready_with_warnings | L126 spatio-DARLIN |
| CloneNicheIntegrator | E3 | `src/nichefate/lineage/clone_niche.py` | Joint clones and spatial unit maps | Clone x niche summaries | ready_with_warnings | L126 spatio-DARLIN |
| LineageAwareFeatureBuilder | E4 | `src/nichefate/lineage/clone_matrix.py`; `src/nichefate/planA_l/representation.py` | Substrate features plus clone variables | Lineage-aware representation channels | design | L126 design interface |
| PlanA Markov-GPCCA Engine | PlanA | `src/nichefate/planA_st_only/gpcca_macrostate_inference.py`; `src/nichefate/planA_k/full_gpcca.py` | M2.5 states and kernels | PlanA ST-only macrostate outputs | frozen | MERFISH/Cadinu/Moffitt |
| PlanA Lineage-Aware Design Interface | PlanA-L | `src/nichefate/lineage/dynamics_interface.py`; `src/nichefate/planA_l/` | M2.5 states plus clone matrices | Design-only lineage-aware dynamics contract | design | L126 design only |
| PlanB Branch/Clone-Aware Engine | PlanB | `src/nichefate/planB_nichebranchsbm/`; `src/nichefate/nichebranchsbm/` | M2.5 states with optional clone variables | In-progress branch/clone-aware artifacts | design | PlanB pilots |

## Benchmarks

| Benchmark | Regime | Role |
|---|---|---|
| MERFISH/Cadinu/Moffitt | ST-only / lineage-free | First stable M0-M2.5 and PlanA-ST baseline |
| L126 spatio-DARLIN | ST + lineage-aware | First benchmark for DARLIN-style joint clone calling and clone x niche representation |

Benchmarks validate implementation contracts. They are not algorithm names.

## Public Naming Rules

- Use "NicheFate" for the overall framework.
- Use "ST-only / lineage-free mode" for E0.
- Use "lineage-aware NicheFate mode" or "barcode-supported evidence regime"
  for E1-E4.
- Use "L126 benchmark" only for dataset-specific results.
- Keep old `planC_l126_*` script names only as provenance or compatibility
  entrypoints, not public method names.
