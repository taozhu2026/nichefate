"""KernelAssembly facade for corrected feature-only Kmix_A construction.

This facade is metadata-only in the GitHub module reorg commit. It preserves
the Kmix_A production module boundary and points to frozen outputs without
rewriting kernel assembly logic.
"""

DOCUMENTED_ONLY_FACADE = True
PENDING_REFACTOR_REASON = (
    "The validated Kmix_A implementation is retained as frozen PlanA-K "
    "provenance, but the full kernel assembly modules are outside the approved "
    "staging scope for this production-module reorg commit."
)
LEGACY_MODULES = (
    "nichefate.planA_k.full_kmix_a",
    "nichefate.planA_k.gpcca_stabilization",
    "nichefate.planA_k.kernel_qc",
    "nichefate.planA_k.sparse_kernel",
)
LEGACY_ENTRYPOINTS = (
    "scripts/planA_k_25_build_full_kmix_A.py",
    "scripts/planA_k_26_full_kernel_qc.py",
)
FROZEN_OUTPUTS = (
    "reports/planA_k_final_result_package/03_FINAL_FIGURE_SOURCE_PROVENANCE.tsv",
    "reports/planA_k_final_result_package/05_VALIDATION.md",
)

__all__ = [
    "DOCUMENTED_ONLY_FACADE",
    "FROZEN_OUTPUTS",
    "LEGACY_ENTRYPOINTS",
    "LEGACY_MODULES",
    "PENDING_REFACTOR_REASON",
]
