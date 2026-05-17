"""FateProbability facade for Kmix_A absorption to terminal macrostate M5.

This facade is metadata-only in the GitHub module reorg commit. It records the
production boundary for PlanA-inferred absorption/fate probability without
rewriting or rerunning the validated PlanA-K absorption implementation.
"""

DOCUMENTED_ONLY_FACADE = True
PENDING_REFACTOR_REASON = (
    "The validated Kmix_A absorption implementation is retained as frozen "
    "PlanA-K provenance, but it is outside the approved staging scope for this "
    "production-module reorg commit."
)
LEGACY_MODULES = (
    "nichefate.planA_k.absorption_fate",
    "nichefate.planA_k.cellrank_aligned_terminal",
)
LEGACY_ENTRYPOINTS = (
    "scripts/planA_k_35_cellrank_aligned_terminal_audit.py",
    "scripts/planA_k_36_compute_cellrank_aligned_absorption.py",
    "scripts/planA_k_37_compute_kforward_absorption_sensitivity.py",
)
FROZEN_OUTPUTS = (
    "reports/planA_k_final_result_package/figures/main_figures/"
    "Figure_5_PlanA_inferred_absorption_fate_probability_to_M5.png",
    "reports/planA_k_final_result_package/tables/kmix_vs_kforward_absorption_comparison.tsv",
)

__all__ = [
    "DOCUMENTED_ONLY_FACADE",
    "FROZEN_OUTPUTS",
    "LEGACY_ENTRYPOINTS",
    "LEGACY_MODULES",
    "PENDING_REFACTOR_REASON",
]
