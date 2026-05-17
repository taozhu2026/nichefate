"""Production registry for the barcode-free PlanA-ST-only v1 pipeline.

This module is intentionally metadata-only. It records the production-facing
module order and the legacy milestone provenance without running any pipeline
stage or importing heavy numerical dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_RESULT_PACKAGE = Path("reports/planA_k_final_result_package")
FINAL_INDEX_ROOT = Path("reports/planA_st_only_v1_index")


@dataclass(frozen=True)
class ProductionModule:
    order: int
    module_name: str
    legacy_name: str
    facade: str
    status: str
    role: str
    primary_legacy_modules: tuple[str, ...] = ()
    primary_legacy_scripts: tuple[str, ...] = ()


PRODUCTION_PIPELINE: tuple[ProductionModule, ...] = (
    ProductionModule(
        0,
        "SpatialDatasetAdapter",
        "M0",
        "nichefate.planA_st_only.spatial_dataset_adapter",
        "stable-reexport",
        "Adapt spatial transcriptomics inputs into the M0 slice-level contract.",
        ("nichefate.io", "nichefate.metadata", "nichefate.spatial"),
        ("scripts/m0_00_check_environment.py", "scripts/m0_01_inspect_raw_anndata.py"),
    ),
    ProductionModule(
        1,
        "NicheBuilder",
        "M1",
        "nichefate.planA_st_only.niche_builder",
        "stable-reexport",
        "Build anchor-centered multi-scale niche feature tables.",
        ("nichefate.niche", "nichefate.niche_qc"),
        ("scripts/m1_03_build_niche_full.py", "scripts/m1_04_build_niche_full_by_slice.py"),
    ),
    ProductionModule(
        2,
        "NicheEncoder",
        "M2",
        "nichefate.planA_st_only.niche_encoder",
        "stable-reexport",
        "Encode M1 niche features into aligned per-anchor representation matrices.",
        ("nichefate.representation", "nichefate.embedding"),
        ("scripts/m2_01_prepare_representation_matrix.py", "scripts/m2_02_build_full_representation_by_slice.py"),
    ),
    ProductionModule(
        3,
        "MetanicheCoarsener",
        "M2.5",
        "nichefate.planA_st_only.metaniche_coarsener",
        "stable-reexport",
        "Coarsen M2 anchors into metaniche states while preserving provenance.",
        ("nichefate.planA_k.metaniche", "nichefate.planA_k.full_m2_5_production"),
        ("scripts/planA_k_23_run_full_m2_5_production.py",),
    ),
    ProductionModule(
        4,
        "TransitionEvidence",
        "M3-v1 / M3-v2",
        "nichefate.planA_st_only.transition_evidence",
        "legacy-compatible-reexport",
        "Represent pseudo-broad and pseudo-sharpened transition evidence.",
        ("nichefate.transition", "nichefate.m3_v2_kernel"),
        ("scripts/m3_15_run_full_m3_by_shard.py", "scripts/m3_v2_06_run_full_by_shard.py"),
    ),
    ProductionModule(
        5,
        "KernelAssembly",
        "M4A / Kmix_A",
        "nichefate.planA_st_only.kernel_assembly",
        "documented-only; pending clean re-export",
        "Assemble corrected feature-only Kmix_A transition kernels from metaniche states.",
        ("nichefate.planA_k.full_kmix_a", "nichefate.planA_k.sparse_kernel"),
        ("scripts/planA_k_25_build_full_kmix_A.py", "scripts/planA_k_26_full_kernel_qc.py"),
    ),
    ProductionModule(
        6,
        "GPCCAMacrostateInference",
        "K_gpcca",
        "nichefate.planA_st_only.gpcca_macrostate_inference",
        "documented-only; pending clean re-export",
        "Run corrected full GPCCA and select the k=6 macrostate model.",
        ("nichefate.planA_k.full_gpcca", "nichefate.planA_k.gpcca_probe"),
        ("scripts/planA_k_27_run_full_gpcca.py",),
    ),
    ProductionModule(
        7,
        "BiologicalAnnotation",
        "M4E",
        "nichefate.planA_st_only.biological_annotation",
        "documented-only; pending clean re-export",
        "Annotate macrostates and score source, transient, and terminal roles.",
        ("nichefate.planA_k.full_macrostate_annotation", "nichefate.planA_k.source_terminal_roles"),
        ("scripts/planA_k_28_annotate_full_macrostates.py", "scripts/planA_k_31_source_terminal_role_scoring.py"),
    ),
    ProductionModule(
        8,
        "EndpointMarkovInference",
        "M4C / P_fate",
        "nichefate.planA_st_only.endpoint_markov_inference",
        "documented-only; frozen baseline",
        "Document the frozen endpoint Markov/P_fate baseline used only as historical context.",
        (),
        ("scripts/m4c_01_compute_markov_fate_probabilities.py", "scripts/planA_00_freeze_p_fate_branch.py"),
    ),
    ProductionModule(
        9,
        "FateProbability",
        "M4C absorption",
        "nichefate.planA_st_only.fate_probability",
        "documented-only; pending clean re-export",
        "Compute Kmix_A absorption/fate probability to terminal macrostate M5.",
        ("nichefate.planA_k.absorption_fate", "nichefate.planA_k.cellrank_aligned_terminal"),
        ("scripts/planA_k_35_cellrank_aligned_terminal_audit.py", "scripts/planA_k_36_compute_cellrank_aligned_absorption.py"),
    ),
    ProductionModule(
        10,
        "ResultVisualization",
        "Visualization scripts",
        "nichefate.planA_st_only.result_visualization",
        "documented-only; pending clean re-export",
        "Build final ST-only figures and visualization QA artifacts.",
        ("nichefate.planA_k.full_result_visualization", "nichefate.planA_k.figures"),
        ("scripts/planA_k_30_full_result_visualization.py", "scripts/planA_k_38_visualize_cellrank_aligned_absorption.py"),
    ),
    ProductionModule(
        11,
        "ResultPackage",
        "Final result package",
        "nichefate.planA_st_only.result_package",
        "documented-only; pending clean re-export",
        "Package final roles, figure provenance, interpretation, validation, and QA.",
        ("nichefate.planA_k.full_result_packet",),
        ("scripts/planA_k_29_full_result_packet.py",),
    ),
)


LEGACY_TO_PRODUCTION: tuple[tuple[str, str, str], ...] = (
    ("M0", "SpatialDatasetAdapter", "Stable ST input adapter and direct re-export."),
    ("M1", "NicheBuilder", "Stable anchor-centered niche construction re-export."),
    ("M2", "NicheEncoder", "Stable aligned niche representation re-export."),
    ("M2.5", "MetanicheCoarsener / NicheStateCoarsener", "Stable metaniche state coarsening re-export."),
    ("M3-v1", "TransitionEvidence[pseudo_broad]", "Legacy broad pseudo-transition evidence re-export."),
    ("M3-v2", "TransitionEvidence[pseudo_sharpened]", "Legacy sharpened pseudo-transition evidence re-export."),
    ("M4A", "KernelAssembly", "Frozen corrected feature-only Kmix_A outputs indexed; clean re-export pending."),
    ("K_gpcca", "GPCCAMacrostateInference", "Frozen corrected full GPCCA outputs indexed; clean re-export pending."),
    ("M4C / P_fate", "EndpointMarkovInference / FateProbability", "Frozen P_fate context plus Kmix_A absorption outputs indexed."),
    ("M4E", "BiologicalAnnotation", "Frozen annotation and role diagnostics indexed; clean re-export pending."),
    ("Visualization scripts", "ResultVisualization", "Frozen figure generation and QA outputs indexed; clean re-export pending."),
    ("Final result package", "ResultPackage / FreezePackage", "Frozen final ST-only result package indexed; clean re-export pending."),
    ("Future DARLIN adapter", "BarcodeEvidenceAdapter", "Future extension, excluded from this freeze."),
)


CLAIM_GUARDRAILS = {
    "terminal_macrostate": "M5 is the PlanA-inferred terminal/sink macrostate with structural/stromal context.",
    "fate_probability": "Kmix_A absorption to M5 is the PlanA-inferred absorption/fate probability.",
    "comparator": "M4 is a D35-enriched non-terminal comparator.",
    "intermediate_states": "M2/M3 are intermediate/transient macrostates with source tendency.",
    "initial_state": "No primary initial macrostate was selected.",
    "barcode_boundary": "This is ST-only / barcode-free; DARLIN/barcode validation is future work.",
}


FORBIDDEN_MAIN_CLAIMS = (
    "warning-level terminal-like candidate",
    "maybe terminal",
    "validated biological endpoint",
    "DARLIN-supported fate",
    "barcode-backed transition",
    "final clone-supported fate",
)


def production_rows() -> list[dict[str, object]]:
    return [asdict(row) for row in PRODUCTION_PIPELINE]


def legacy_mapping_rows() -> list[dict[str, str]]:
    return [
        {"legacy_milestone": legacy, "production_module": module, "status_or_note": note}
        for legacy, module, note in LEGACY_TO_PRODUCTION
    ]


def module_by_name(name: str) -> ProductionModule:
    for module in PRODUCTION_PIPELINE:
        if module.module_name == name:
            return module
    raise KeyError(name)
