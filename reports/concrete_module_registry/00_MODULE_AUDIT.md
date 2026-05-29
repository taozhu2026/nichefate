# Concrete Module Audit

## Result

- Decision label: `CONCRETE_MODULE_REGISTRY_READY`
- Target branch: `docs/concrete-module-registry`
- Scope: documentation, registry config, and validation only

## Key Findings

- `NicheEncoder` exists as a public facade at
  `src/nichefate/planA_st_only/niche_encoder.py`.
- `NicheEncoder` is not a monolithic class. It re-exports the equivalent M2
  implementation components from `nichefate.embedding` and
  `nichefate.representation`.
- ST-only modules are already represented in
  `src/nichefate/planA_st_only/module_registry.py`, but the public docs did
  not integrate them with the lineage-aware module set.
- Lineage-aware modules exist as generic facades under `src/nichefate/lineage/`
  and `src/nichefate/darlin/`.
- L126-specific scripts are benchmark wrappers and provenance, not algorithm
  names.

## Module Classification

| Item | Surface | Classification | Public module |
|---|---|---|---|
| Spatial input adapter | `src/nichefate/planA_st_only/spatial_dataset_adapter.py`; `src/nichefate/io.py`; `src/nichefate/spatial.py` | shared_core_algorithm | SpatialInputContract |
| Niche construction | `src/nichefate/planA_st_only/niche_builder.py`; `src/nichefate/niche.py`; `src/nichefate/niche_qc.py` | shared_core_algorithm | NicheConstructor |
| Niche encoding | `src/nichefate/planA_st_only/niche_encoder.py`; `src/nichefate/representation.py`; `src/nichefate/embedding.py` | shared_core_algorithm | NicheEncoder |
| M2 representation helpers | `src/nichefate/representation.py` | shared_core_algorithm | NicheRepresentationBuilder |
| Metaniche coarsening | `src/nichefate/planA_st_only/metaniche_coarsener.py`; `src/nichefate/planA_k/metaniche.py` | shared_core_algorithm | MetaNicheCoarsener |
| Transition evidence / sparse kernels | `src/nichefate/planA_st_only/transition_evidence.py`; `src/nichefate/planA_k/sparse_kernel.py` | shared_core_algorithm | TransitionKernelBuilder |
| ST-only facade registry | `src/nichefate/planA_st_only/module_registry.py` | evidence_specific_st_only | STOnlyEvidenceBuilder |
| Barcode adapter utilities | `src/nichefate/barcode_adapter/` | evidence_specific_lineage | LineageEvidenceAdapter |
| Lineage facade | `src/nichefate/lineage/` | evidence_specific_lineage | LineageEvidenceAdapter / CloneNicheIntegrator |
| DARLIN facade | `src/nichefate/darlin/` | evidence_specific_lineage | DARLINJointCloneCaller |
| L126 joint clone package | `src/nichefate/darlin_joint_clone_niche_v1.py` | benchmark_wrapper | L126 DARLIN wrapper |
| PlanA ST-only facades | `src/nichefate/planA_st_only/` | dynamics_engine | PlanA Markov-GPCCA Engine |
| PlanA-K implementation | `src/nichefate/planA_k/` | dynamics_engine | PlanA Markov-GPCCA Engine |
| PlanA-L probes | `src/nichefate/planA_l/`; `scripts/planC_l126_planA_lineage_*` | legacy_or_ablation | PlanA lineage-aware design interface |
| PlanB implementation | `src/nichefate/planB_nichebranchsbm/`; `src/nichefate/nichebranchsbm/` | dynamics_engine | PlanB Branch/Clone-Aware Engine |
| L126 PlanC wrappers | `scripts/planC_l126_*.py` | benchmark_wrapper | E1/E2/E3 wrappers |
| CloneSignature probes | `src/nichefate/darlin_clone_signature/`; `scripts/planC_l126_darlin_clone_signature_round2.py` | superseded | lineage clone ablation |

## Repair Actions

- Add a machine-readable module registry under `configs/module_registry/`.
- Add a human-readable algorithm registry page under `docs/`.
- Update README and pipeline index to show concrete algorithm modules.
- Add lineage insertion point documentation.
- Expand legacy script-to-module mapping.
