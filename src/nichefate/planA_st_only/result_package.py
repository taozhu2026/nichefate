"""ResultPackage facade for the final PlanA-ST-only v1 freeze package.

This facade is metadata-only in the GitHub module reorg commit. It records the
final package boundary and indexes frozen artifacts without rerunning package
construction.
"""

DOCUMENTED_ONLY_FACADE = True
PENDING_REFACTOR_REASON = (
    "The validated final result packaging implementation is retained as frozen "
    "PlanA-K provenance, but it is outside the approved staging scope for this "
    "production-module reorg commit."
)
LEGACY_MODULES = ("nichefate.planA_k.full_result_packet",)
LEGACY_ENTRYPOINTS = ("scripts/planA_k_29_full_result_packet.py",)
FROZEN_OUTPUTS = (
    "reports/planA_k_final_result_package/00_PLAN_A_ST_ONLY_V1_FINAL_RESULT_SUMMARY.md",
    "reports/planA_k_final_result_package/04_PLAN_A_ST_ONLY_V1_FINAL_INTERPRETATION.md",
)

__all__ = [
    "DOCUMENTED_ONLY_FACADE",
    "FROZEN_OUTPUTS",
    "LEGACY_ENTRYPOINTS",
    "LEGACY_MODULES",
    "PENDING_REFACTOR_REASON",
]
