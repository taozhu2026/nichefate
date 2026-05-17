"""BiologicalAnnotation facade for macrostate annotation and role scoring.

This facade is metadata-only in the GitHub module reorg commit. The validated
annotation outputs are indexed under ``reports/planA_st_only_v1_index``; the
algorithm modules remain legacy PlanA-K provenance and are not rewritten here.
"""

DOCUMENTED_ONLY_FACADE = True
PENDING_REFACTOR_REASON = (
    "The validated macrostate annotation implementation is retained as frozen "
    "PlanA-K provenance, but it is outside the approved staging scope for this "
    "production-module reorg commit."
)
LEGACY_MODULES = (
    "nichefate.planA_k.full_macrostate_annotation",
    "nichefate.planA_k.macrostate_annotation",
    "nichefate.planA_k.source_terminal_roles",
    "nichefate.planA_k.cellrank_aligned_terminal",
)
LEGACY_ENTRYPOINTS = (
    "scripts/planA_k_28_annotate_full_macrostates.py",
    "scripts/planA_k_31_source_terminal_role_scoring.py",
    "scripts/planA_k_35_cellrank_aligned_terminal_audit.py",
)
FROZEN_OUTPUTS = (
    "reports/planA_k_final_result_package/02_FINAL_MACROSTATE_ROLES.tsv",
    "reports/planA_k_final_result_package/02_FINAL_MACROSTATE_ROLES.md",
)

__all__ = [
    "DOCUMENTED_ONLY_FACADE",
    "FROZEN_OUTPUTS",
    "LEGACY_ENTRYPOINTS",
    "LEGACY_MODULES",
    "PENDING_REFACTOR_REASON",
]
