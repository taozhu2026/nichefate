# NicheFate: Spatial Niche-Fate Inference for Time-Resolved Spatial Transcriptomics

NicheFate models spatial niches rather than isolated cells. It builds
anchor-centered niche-state representations from time-resolved spatial
transcriptomics, infers coarse-grained niche macrostates, and estimates
terminal-state absorption/fate probabilities in a frozen barcode-free ST-only
workflow.

## Current Status

The NicheFate ST-only workflow v1 is frozen as a reproducible barcode-free
baseline.

- The M0-M2.5 spatial niche backbone is frozen.
- The corrected feature-only transition kernel is the mainline kernel.
- Corrected GPCCA macrostate inference is complete with k=6 niche macrostates.
- M5 is the inferred terminal/sink macrostate with structural/stromal context.
- Kmix_A absorption to M5 is the inferred absorption/fate probability.
- The final result visualization package is available.
- DARLIN/barcode integration is future work and is not part of this freeze.

The frozen baseline is an algorithm-defined ST-only result. It does not claim
lineage-validated fate, barcode-derived transition support, or clone-matrix
endpoint validation.

## Module Workflow

The public workflow is organized by functional modules.

| Module | Purpose | Input | Output |
|---|---|---|---|
| SpatialDatasetAdapter | Adapts time-resolved spatial transcriptomics inputs into the project contract. | Raw or prepared spatial transcriptomics objects plus sample metadata. | Standardized slice-level spatial data and metadata. |
| NicheBuilder | Builds anchor-centered multi-scale neighborhood features. | Standardized spatial data, coordinates, and cell annotations. | Per-anchor niche feature tables. |
| NicheEncoder | Encodes niche features into aligned representation matrices. | Per-anchor niche feature tables. | Comparable anchor-level niche representation tables. |
| NicheStateCoarsener | Coarsens anchors into stable niche/metaniche states. | Niche representation tables. | Metaniche/niche-state assignments and summaries. |
| TransitionKernelAssembly | Assembles the corrected feature-only transition kernel. | Niche-state features and transition evidence. | Kmix_A transition kernel and QC summaries. |
| MacrostateInference | Infers coarse-grained niche macrostates. | Corrected transition kernel and metaniche state space. | GPCCA k=6 macrostate assignments and memberships. |
| TerminalStateInference | Scores source, transient, and terminal/sink roles. | Macrostate assignments, transition structure, and composition summaries. | Terminal/sink macrostate call with M5 selected as the inferred terminal state. |
| FateProbability | Computes absorption/fate probability to the terminal state. | Kmix_A transition operator and terminal macrostate definition. | Absorption/fate probability to M5. |
| BiologicalContextAnnotation | Adds biological context for interpreting macrostates. | Macrostate roles, composition tables, and marker/context summaries. | ST-only biological context annotations and role summaries. |
| ResultVisualization | Builds final figures and QA outputs. | Final macrostate, fate-probability, and annotation tables. | Main and supplementary figure package with provenance. |
| ResultPackage | Collects frozen reports, tables, figures, and claim boundaries. | Final reports, figure manifests, and validation notes. | Reviewable frozen ST-only result package. |

## Repository Layout

- `src/nichefate/`: reusable package modules for spatial data handling,
  niche construction, representation, transition evidence, and downstream
  workflow support.
- `src/nichefate/planA_st_only/`: metadata-only production facade for the
  frozen barcode-free ST-only workflow and its legacy provenance.
- `scripts/`: reproducibility and inspection entry points. Full production
  analysis scripts are not quickstart commands.
- `configs/`: lightweight historical and workflow configuration files.
- `docs/`: public documentation, workflow indexes, reproducibility notes, and
  legacy provenance.
- `reports/`: curated result indexes, validation reports, and small text/table
  summaries. Large generated outputs remain external.
- `tests/`: lightweight tests for package contracts, path policies, and focused
  workflow helpers.

## Frozen Results And Reports

The frozen ST-only result package is indexed in the repository. Figure binaries
are linked or indexed only when already present; do not add new figure binaries
as part of documentation cleanup.

- Final result package: `reports/planA_k_final_result_package/`
- Main figures: `reports/planA_k_final_result_package/figures/main_figures/`
- Supplementary figures:
  `reports/planA_k_final_result_package/figures/supplementary_figures/`
- ST-only v1 result index: `reports/planA_st_only_v1_index/`
- Production module mapping:
  `docs/planA_st_only_v1_production_modules.md`
- Final result summary:
  `reports/planA_k_final_result_package/00_PLAN_A_ST_ONLY_V1_FINAL_RESULT_SUMMARY.md`
- Final visualization QA:
  `reports/planA_k_final_result_package/06_FINAL_VISUALIZATION_QA.md`

## Legacy Provenance

Legacy development-stage names are retained for provenance. New readers should
use the functional module names above. See
`docs/planA_st_only_v1_production_modules.md` for the full mapping.

| Legacy name | Functional module |
|---|---|
| M0 | SpatialDatasetAdapter |
| M1 | NicheBuilder |
| M2 | NicheEncoder |
| M2.5 | NicheStateCoarsener |
| Kmix_A | TransitionKernelAssembly |
| GPCCA | MacrostateInference |
| terminal audit | TerminalStateInference |
| absorption | FateProbability |
| visualization | ResultVisualization |

## What Is Not Included

- No DARLIN/barcode-supported fate inference yet.
- No clone matrix integration yet.
- No raw data are included in GitHub.
- No production matrices are included in GitHub.
- No regulator discovery completion claim is made.
- No lineage-backed transition claim is made.

## Next Stage

DARLIN/barcode integration will be developed separately through a
`BarcodeEvidenceAdapter` and lineage-supported fate benchmarking. That stage
will compare the frozen ST-only baseline against barcode-derived evidence
without retroactively changing the ST-only freeze.

## Development And Validation

The production module reorganization was documentation/index oriented and did
not change validated numerical behavior. Focused validation in the module reorg
included production facade imports, lightweight support tests, JSON/TSV
consistency checks, figure-index checks, claim-boundary checks, and staging
audits. Large data, scratch outputs, raw files, production matrices, DARLIN
evidence, and new figure binaries are excluded from GitHub.
