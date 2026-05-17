# PlanA-ST-only v1 Production Modules

PlanA-ST-only v1 is the completed barcode-free spatial-transcriptomics-only
pipeline for niche-level macrostate and fate-probability inference. Legacy
M-number names are retained only as development provenance. The production
interface uses functional module names that describe what each stage does.

## Why Production Names

The M0, M1, M2, M2.5, M3, and M4 labels were useful during milestone-driven
development, but they do not communicate stable algorithm boundaries. The v1
repository now exposes production-style facades under
`nichefate.planA_st_only` while preserving the validated legacy `planA_k`
modules and scripts for compatibility. Facades re-export modules only when the
underlying implementation is already safe in the frozen backbone; later PlanA-K
v1 outputs are indexed as documented-only boundaries until they receive a clean
standalone refactor.

## Module Map

| Legacy milestone | Production module | Status |
|---|---|---|
| M0 | SpatialDatasetAdapter | stable re-export |
| M1 | NicheBuilder | stable re-export |
| M2 | NicheEncoder | stable re-export |
| M2.5 | MetanicheCoarsener / NicheStateCoarsener | stable re-export |
| M3-v1 | TransitionEvidence[pseudo_broad] | legacy-compatible re-export |
| M3-v2 | TransitionEvidence[pseudo_sharpened] | legacy-compatible re-export |
| M4A | KernelAssembly | documented-only; frozen output indexed |
| K_gpcca historical label | GPCCAMacrostateInference | documented-only; corrected GPCCA k=6 frozen output indexed |
| M4C historical endpoint baseline | EndpointMarkovInference / FateProbability | documented-only frozen context plus active Kmix_A absorption |
| M4E | BiologicalAnnotation | documented-only; frozen output indexed |
| Visualization scripts | ResultVisualization | documented-only; frozen output indexed |
| Final result package | ResultPackage / FreezePackage | documented-only; frozen output indexed |
| Future DARLIN adapter | BarcodeEvidenceAdapter | future extension, excluded |

## End-to-End ST-only v1 Pipeline

1. `SpatialDatasetAdapter` prepares spatial transcriptomics inputs for the M0
   contract.
2. `NicheBuilder` constructs anchor-centered multi-scale niche features.
3. `NicheEncoder` creates aligned M2 anchor-level representation tables.
4. `MetanicheCoarsener` coarsens anchors into metaniche/niche-state units.
5. `KernelAssembly` builds the corrected feature-only Kmix_A transition kernel.
6. `GPCCAMacrostateInference` runs corrected full GPCCA and selects k=6.
7. `BiologicalAnnotation` annotates macrostates and scores source/terminal
   roles.
8. `FateProbability` computes Kmix_A absorption/fate probability to M5.
9. `ResultVisualization` and `ResultPackage` create the final figures, QA, and
   result package.

Final algorithmic interpretation:

- M5 is the PlanA-inferred terminal/sink macrostate with structural/stromal
  context.
- Kmix_A absorption to M5 is the PlanA-inferred absorption/fate probability.
- M4 is a D35-enriched non-terminal comparator.
- M2/M3 are intermediate/transient macrostates with source tendency.
- No primary initial macrostate was selected.

## Barcode Boundary

This freeze is ST-only / barcode-free. It does not include DARLIN preprocessing,
barcode evidence, barcode-backed transitions, clone-supported fate inference, or
barcode validation. `BarcodeEvidenceAdapter` is a future module boundary and is
not part of the PlanA-ST-only v1 release.

## Results And Inspection

- Final result package: `reports/planA_k_final_result_package/`
- Main figures: `reports/planA_k_final_result_package/figures/main_figures/`
- Supplementary figures:
  `reports/planA_k_final_result_package/figures/supplementary_figures/`
- Figure provenance:
  `reports/planA_k_final_result_package/03_FINAL_FIGURE_SOURCE_PROVENANCE.tsv`
- Visualization QA:
  `reports/planA_k_final_result_package/06_FINAL_VISUALIZATION_QA.md`
- Production registry:
  `src/nichefate/planA_st_only/module_registry.py`

Use the lightweight inspection scripts for repository metadata and frozen output
checks:

```bash
python scripts/planA_st_only_00_module_inventory.py
python scripts/planA_st_only_02_build_result_index.py
python scripts/planA_st_only_01_validate_frozen_outputs.py
```

These scripts do not rerun production computation.
