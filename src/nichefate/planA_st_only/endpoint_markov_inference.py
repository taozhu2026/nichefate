"""EndpointMarkovInference facade for the frozen M4C/P_fate baseline.

The frozen P_fate baseline is retained as historical context and is not rerun
as part of PlanA-ST-only v1. The active v1 fate probability is exposed through
``nichefate.planA_st_only.fate_probability``.
"""

DOCUMENTED_ONLY_FACADE = True
PENDING_REFACTOR_REASON = (
    "The frozen P_fate endpoint Markov baseline is retained as historical "
    "context and is not part of the active PlanA-ST-only v1 absorption result."
)
LEGACY_ENTRYPOINTS = (
    "scripts/m4c_01_compute_markov_fate_probabilities.py",
    "scripts/m4c_02_review_and_freeze_markov_fate_results.py",
    "scripts/planA_00_freeze_p_fate_branch.py",
)
FROZEN_OUTPUTS = (
    "reports/planA_st_only_v1_index/04_CLAIM_BOUNDARY.md",
)

__all__ = [
    "DOCUMENTED_ONLY_FACADE",
    "FROZEN_OUTPUTS",
    "LEGACY_ENTRYPOINTS",
    "PENDING_REFACTOR_REASON",
]
