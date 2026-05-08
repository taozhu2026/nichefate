# Nichefate Project Status Checkpoint

Generated for ReviewPack-01 on 2026-05-08.

This checkpoint freezes the current workspace for review and packaging. It is
documentation-only: no new biological analysis, GPCCA, CellRank, terminal-state
design, fate propagation, BranchSBM, barcode, or DARLIN preprocessing is part of
this stage.

## Status Table

| Module | Future production module | Status | Current checkpoint | Key existing outputs |
|---|---|---:|---|---|
| M0 | SpatialDatasetAdapter | completed | MERFISH colitis data adapter, metadata harmonization, embeddings, spatial graphs, and M0 reports exist. | `/home/zhutao/scratch/nichefate/m0` |
| M1 | NicheBuilder | completed | Anchor-centered multi-scale niche construction completed by slice. | `/home/zhutao/scratch/nichefate/m1/by_slice` |
| M2 | NicheEncoder | completed | Niche representation matrix and feature schema completed by slice. | `/home/zhutao/scratch/nichefate/m2/by_slice` |
| M3-v1 | TransitionEvidence[pseudo_broad] | frozen | Broad pseudo-lineage transition evidence was built and frozen as the v1 baseline. | `/home/zhutao/scratch/nichefate/m3/full_by_shard` |
| M3-v2 | TransitionEvidence[pseudo_sharpened] | completed | Constrained v1-prior sharpening was designed, piloted, validated, and run as a v2 branch. | `/home/zhutao/scratch/nichefate/m3_v2/full_by_shard` |
| M4A | KernelAssembly | completed | M3-v1 transition evidence was assembled into Markov transition matrices. | `/home/zhutao/scratch/nichefate/m4a` |
| M4A-v2 | KernelAssembly | completed | M3-v2 transition evidence was assembled into v2 Markov transition matrices. | `/home/zhutao/scratch/nichefate/m4a_v2` |
| M4B | TerminalStateDesign | pilot | Terminal macrostate design and Markov feasibility checks exist; full GPCCA was not promoted as a production dependency. | `/home/zhutao/scratch/nichefate/m4b` |
| M4C | EndpointMarkovInference / P_fate | frozen | Baseline endpoint Markov fate probabilities were computed and frozen as the interpretable P_fate branch. | `/home/zhutao/scratch/nichefate/m4c` |
| M4C-v2 | EndpointMarkovInference / P_fate | completed | v2 propagation and benchmarks exist as a comparison branch, not a replacement claim in this checkpoint. | `/home/zhutao/scratch/nichefate/m4c_v2` |
| M4E | BiologicalAnnotation | completed | Endpoint and neighborhood annotation reports exist for interpreting P_fate outputs. | `/home/zhutao/scratch/nichefate/m4e` |
| P_fate | EndpointMarkovInference / P_fate | frozen | Primary reviewable fate-inference branch for this checkpoint. | `/home/zhutao/scratch/nichefate/planA_freeze` |
| K_gpcca | GPCCAMacrostateInference experimental branch | experimental | Bounded GPCCA pilots and revisions exist; no production K_gpcca matrices or downstream fate outputs are claimed. | `/home/zhutao/scratch/nichefate/k_gpcca_*` |
| DARLIN / barcode adapter | BarcodeEvidenceAdapter | deferred | Barcode integration is positioned for future official/lab-standard DARLIN preprocessing. | Documented in PlanA reports only |
| BranchSBM / Plan B | BranchingStructureModel | deferred | BranchSBM remains a future branch-level modeling direction and is not implemented in this checkpoint. | Documented in PlanA reports only |

## Review Position

- `P_fate` is the frozen checkpoint branch for biological review.
- `K_gpcca` remains experimental and should not block ReviewPack.
- Existing GPCCA-related outputs are completed/pilot artifacts only.
- Large scratch outputs are external artifacts and should not be uploaded to
  GitHub.

