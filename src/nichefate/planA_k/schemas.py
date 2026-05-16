from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRATCH_ROOT = Path("/home/zhutao/scratch/nichefate")
DOC_ROOT = PROJECT_ROOT / "docs" / "planA_k_gpcca"
REPORT_ROOT = PROJECT_ROOT / "reports" / "planA_k_gpcca_redesign"
METANICHE_REPORT_ROOT = PROJECT_ROOT / "reports" / "planA_k_metaniche_pilot"
METANICHE_HARDENING_ROOT = PROJECT_ROOT / "reports" / "planA_k_metaniche_hardening"
SPARSE_KERNEL_REPORT_ROOT = PROJECT_ROOT / "reports" / "planA_k_sparse_kernel_pilot"
TINY_GPCCA_REPORT_ROOT = PROJECT_ROOT / "reports" / "planA_k_tiny_gpcca_probe"
GPCCA_STABILIZATION_ROOT = PROJECT_ROOT / "reports" / "planA_k_gpcca_stabilization"
MACROSTATE_ANNOTATION_ROOT = PROJECT_ROOT / "reports" / "planA_k_macrostate_annotation_probe"
PRODUCTION_PREFLIGHT_ROOT = PROJECT_ROOT / "reports" / "planA_k_production_preflight"
PLAN_A_K_PRODUCTION_SCRATCH_ROOT = SCRATCH_ROOT / "planA_k_production"
FULL_KMIX_A_PRODUCTION_ROOT = PLAN_A_K_PRODUCTION_SCRATCH_ROOT / "full_kmix_A"
FULL_KMIX_A_REPORT_ROOT = PROJECT_ROOT / "reports" / "planA_k_full_kmix_A"
FULL_KMIX_A_FEATURE_ONLY_PRODUCTION_ROOT = PLAN_A_K_PRODUCTION_SCRATCH_ROOT / "full_kmix_A_feature_only"
FULL_KMIX_A_FEATURE_ONLY_REPORT_ROOT = PROJECT_ROOT / "reports" / "planA_k_full_kmix_A_feature_only"
FULL_GPCCA_PRODUCTION_ROOT = PLAN_A_K_PRODUCTION_SCRATCH_ROOT / "full_gpcca"
FULL_GPCCA_REPORT_ROOT = PROJECT_ROOT / "reports" / "planA_k_full_gpcca"
FULL_GPCCA_FEATURE_ONLY_PRODUCTION_ROOT = PLAN_A_K_PRODUCTION_SCRATCH_ROOT / "full_gpcca_feature_only"
FULL_GPCCA_FEATURE_ONLY_REPORT_ROOT = PROJECT_ROOT / "reports" / "planA_k_full_gpcca_feature_only"
FULL_GPCCA_FEATURE_ONLY_SMOKE_PRODUCTION_ROOT = PLAN_A_K_PRODUCTION_SCRATCH_ROOT / "full_gpcca_feature_only_smoke"
FULL_GPCCA_FEATURE_ONLY_SMOKE_REPORT_ROOT = PROJECT_ROOT / "reports" / "planA_k_full_gpcca_feature_only_smoke"
FULL_MACROSTATE_ANNOTATION_PRODUCTION_ROOT = PLAN_A_K_PRODUCTION_SCRATCH_ROOT / "full_macrostate_annotation"
FULL_MACROSTATE_ANNOTATION_REPORT_ROOT = PROJECT_ROOT / "reports" / "planA_k_full_macrostate_annotation"
M2_BY_SLICE_ROOT = SCRATCH_ROOT / "m2" / "by_slice"
M1_BY_SLICE_ROOT = SCRATCH_ROOT / "m1" / "by_slice"
M2_SCHEMA_PATH = SCRATCH_ROOT / "m2" / "reports" / "m2_full_feature_schema.json"
M2_COMPLETED_SLICES_PATH = M2_BY_SLICE_ROOT / "completed_slices.csv"
PILOT_OUTPUT_ROOT = METANICHE_REPORT_ROOT / "pilot_outputs"
STRATIFIED_METANICHE_ROOT = METANICHE_HARDENING_ROOT / "stratified_pilot_outputs"

KEY_FILES: list[Path] = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "AGENTS.md",
    PROJECT_ROOT / "docs" / "project_status_checkpoint.md",
    PROJECT_ROOT / "reports" / "planA_readiness" / "06_final_readiness_summary.md",
    PROJECT_ROOT / "reports" / "planA_readiness_patch_sprint" / "00_PATCH_SPRINT_SUMMARY.md",
    PROJECT_ROOT / "reports" / "planB_branchsbm" / "05_planB_guardrails_for_report.md",
    SCRATCH_ROOT / "k_gpcca_design" / "reports" / "p_fate_vs_k_gpcca_design_distinction.md",
    SCRATCH_ROOT / "k_gpcca_pilot_benchmark" / "reports" / "k_gpcca_03_benchmark_decision_report.md",
    SCRATCH_ROOT / "k_gpcca_revision" / "reports" / "k_gpcca_04_completion_check.md",
    SCRATCH_ROOT / "m4d" / "reports" / "m4d_standard_gpcca_next_step_recommendation.md",
    SCRATCH_ROOT / "m4a" / "reports" / "m4a_assembly_report.md",
    SCRATCH_ROOT / "m4a_v2" / "reports" / "m4a_v2_02_full_assembly_report.md",
]

MATRIX_NNZ_COMPONENT_LIMIT = 20_000_000
MATRIX_STATE_COMPONENT_LIMIT = 200_000
EDGE_PREVIEW_LIMIT = 3


@dataclass(frozen=True)
class MatrixSpec:
    path: Path
    branch: str
    artifact_role: str
    node_table: Path | None = None


@dataclass(frozen=True)
class EdgeSpec:
    path: Path
    branch: str
    artifact_role: str = "edge_evidence_table"


__all__ = [name for name in globals() if not name.startswith("__")]
