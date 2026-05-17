"""ResultVisualization facade for final PlanA-ST-only v1 figures and QA.

This facade is metadata-only in the GitHub module reorg commit. It records the
figure and QA boundary while indexes reference the frozen local figure files.
"""

DOCUMENTED_ONLY_FACADE = True
PENDING_REFACTOR_REASON = (
    "The validated figure and QA implementation is retained as frozen PlanA-K "
    "provenance, but it is outside the approved staging scope for this "
    "production-module reorg commit."
)
LEGACY_MODULES = (
    "nichefate.planA_k.figures",
    "nichefate.planA_k.full_result_visualization",
    "nichefate.planA_k.spatial_kernel_integrity_audit",
)
LEGACY_ENTRYPOINTS = (
    "scripts/planA_k_30_full_result_visualization.py",
    "scripts/planA_k_38_visualize_cellrank_aligned_absorption.py",
)
FROZEN_OUTPUTS = (
    "reports/planA_k_final_result_package/03_FINAL_FIGURE_MANIFEST.tsv",
    "reports/planA_k_final_result_package/06_FINAL_VISUALIZATION_QA.md",
)

__all__ = [
    "DOCUMENTED_ONLY_FACADE",
    "FROZEN_OUTPUTS",
    "LEGACY_ENTRYPOINTS",
    "LEGACY_MODULES",
    "PENDING_REFACTOR_REASON",
]
