"""GPCCAMacrostateInference facade for corrected full GPCCA.

This facade is metadata-only in the GitHub module reorg commit. It preserves
the production module boundary and points to frozen k=6 GPCCA outputs without
changing numerical behavior.
"""

DOCUMENTED_ONLY_FACADE = True
PENDING_REFACTOR_REASON = (
    "The validated full GPCCA implementation is retained as frozen PlanA-K "
    "provenance, but it is outside the approved staging scope for this "
    "production-module reorg commit."
)
LEGACY_MODULES = (
    "nichefate.planA_k.full_gpcca",
    "nichefate.planA_k.gpcca_probe",
)
LEGACY_ENTRYPOINTS = (
    "scripts/planA_k_27_run_full_gpcca.py",
)
FROZEN_OUTPUTS = (
    "reports/planA_k_final_result_package/figures/main_figures/"
    "Figure_2_UMAP_GPCCA_k6_macrostate_atlas.png",
    "reports/planA_k_final_result_package/tables/umap_macrostate_atlas_table.tsv",
)

__all__ = [
    "DOCUMENTED_ONLY_FACADE",
    "FROZEN_OUTPUTS",
    "LEGACY_ENTRYPOINTS",
    "LEGACY_MODULES",
    "PENDING_REFACTOR_REASON",
]
