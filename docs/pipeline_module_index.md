# NicheFate Pipeline Module Index

This index maps NicheFate as one framework with shared substrate modules,
parallel evidence regimes, and dynamics engines that consume whichever evidence
regime is available.

## Shared Spatial-Niche Substrate

M0-M2.5 are shared substrate modules. They are not ST-only internals and they
are not lineage-specific modules.

| Module | Responsibility | Current surface |
|---|---|---|
| M0 | ST/sample/input contract and standardized spatial objects | `docs/m0_design.md` |
| M1 | Spatial niche construction | `src/nichefate/niche.py` and M1 scripts |
| M2 | Niche representation matrices | `src/nichefate/representation.py` |
| M2.5 | Metaniche / niche-state coarsening | `src/nichefate/planA_k/metaniche.py` |

## Evidence Regimes

| Module | Evidence regime | Responsibility | Current surface |
|---|---|---|---|
| E0 | ST-only / lineage-free | Use expression, spatial coordinates, metadata, and niche representations without barcode lineage evidence | `src/nichefate/planA_st_only/` |
| E1 | Lineage-aware | Adapt processed lineage evidence into NicheFate cellbin-level contracts | `src/nichefate/lineage/evidence.py` |
| E2 | Lineage-aware | Call validated DARLIN-style joint clones | `src/nichefate/darlin/joint_clone_calling.py` |
| E3 | Lineage-aware | Build clone x niche representation summaries | `src/nichefate/lineage/clone_niche.py` |

E1-E3 add lineage evidence and clone state variables. They do not replace the
shared M0-M2.5 substrate.

## Dynamics Engines

| Engine | Status | Consumes |
|---|---|---|
| PlanA-ST | Frozen baseline | M0-M2.5 plus E0 |
| PlanA lineage-aware | Design / in progress | M0-M2.5 plus E1-E3 |
| PlanB branch/clone-aware dynamics | Design / in progress | M0-M2.5 plus E0 or E1-E3, depending on dataset |

Directionality in dynamics engines must come from time, perturbation, or an
explicit biological prior. Clone evidence can add state variables or support
terms when the dataset supports that interpretation.

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
  for E1-E3.
- Use "L126 benchmark" only for dataset-specific results.
- Keep old `planC_l126_*` script names only as provenance or compatibility
  entrypoints, not public method names.
