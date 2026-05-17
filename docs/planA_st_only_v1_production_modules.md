# NicheFate ST-only Workflow v1 Production Modules

NicheFate ST-only workflow v1 is the completed barcode-free
spatial-transcriptomics-only baseline for niche-level macrostate and
fate-probability inference. Legacy M-stage and PlanA names are retained only as
development provenance. The public workflow uses functional module names that
describe what each stage does.

## Why Production Names

The M0, M1, M2, M2.5, M3, and M4 labels were useful during milestone-driven
development, but they do not communicate stable algorithm boundaries. The v1
repository now exposes production-style facades under
`nichefate.planA_st_only` while preserving validated legacy modules and scripts
for compatibility. Facades re-export modules only when the underlying
implementation is already safe in the frozen backbone; later frozen v1 outputs
are indexed as documented-only boundaries until they receive clean standalone
module implementations.

## Module Map

| Legacy milestone | Production module | Status |
|---|---|---|
| M0 | SpatialDatasetAdapter | stable re-export |
| M1 | NicheBuilder | stable re-export |
| M2 | NicheEncoder | stable re-export |
| M2.5 | NicheStateCoarsener | stable re-export through the metaniche coarsener facade |
| M3-v1 | TransitionEvidence[pseudo_broad] | legacy-compatible re-export |
| M3-v2 | TransitionEvidence[pseudo_sharpened] | legacy-compatible re-export |
| Kmix_A | TransitionKernelAssembly | documented-only; corrected feature-only kernel output indexed |
| GPCCA | MacrostateInference | documented-only; corrected GPCCA k=6 output indexed |
| terminal audit | TerminalStateInference | documented-only; source/terminal role output indexed |
| absorption | FateProbability | documented-only; Kmix_A absorption to M5 indexed |
| annotation | BiologicalContextAnnotation | documented-only; biological context output indexed |
| visualization | ResultVisualization | documented-only; final figures and QA indexed |
| final result package | ResultPackage | documented-only; frozen result package indexed |
| Future DARLIN adapter | BarcodeEvidenceAdapter | future extension, excluded |

## End-to-End ST-only v1 Pipeline

1. `SpatialDatasetAdapter` prepares spatial transcriptomics inputs for the
   frozen input contract.
2. `NicheBuilder` constructs anchor-centered multi-scale niche features.
3. `NicheEncoder` creates aligned M2 anchor-level representation tables.
4. `NicheStateCoarsener` coarsens anchors into metaniche/niche-state units.
5. `TransitionKernelAssembly` builds the corrected feature-only Kmix_A
   transition kernel.
6. `MacrostateInference` runs corrected GPCCA and selects k=6.
7. `TerminalStateInference` scores source, transient, and terminal/sink roles.
8. `FateProbability` computes Kmix_A absorption/fate probability to M5.
9. `BiologicalContextAnnotation` adds ST-only biological context.
10. `ResultVisualization` and `ResultPackage` create the final figures, QA,
    and result package.

Final algorithmic interpretation:

- M5 is the inferred terminal/sink macrostate with structural/stromal context.
- Kmix_A absorption to M5 is the inferred absorption/fate probability.
- M4 is a D35-enriched non-terminal comparator.
- M2/M3 are intermediate/transient macrostates with source tendency.
- No primary initial macrostate was selected.

## Barcode Boundary

This freeze is ST-only / barcode-free. It does not include DARLIN preprocessing,
barcode evidence, barcode-derived transition support, clone-matrix fate
validation, or barcode validation. `BarcodeEvidenceAdapter` is a future module
boundary and is not part of the ST-only v1 release.

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
