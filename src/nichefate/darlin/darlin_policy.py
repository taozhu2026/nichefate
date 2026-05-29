from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from nichefate.darlin_joint_clone_niche_v1 import (
    MIN_CELLBINS_PER_ALLELE,
    NORMALIZED_COUNT_CUTOFF,
    SAMPLE_COUNT_CUTOFF,
    SELECTED_ALLELE_POLICY,
    SELECTED_REFERENCE_BANK_POLICY,
    SELECTED_THRESHOLD_LABEL,
)


@dataclass(frozen=True)
class FrozenDarlinJointClonePolicy:
    reference_bank_policy: str
    allele_policy: str
    threshold_label: str
    normalized_count_cutoff: float
    sample_count_cutoff: int
    min_cellbins_per_allele: int


def freeze_selected_joint_clone_policy() -> dict[str, Any]:
    policy = FrozenDarlinJointClonePolicy(
        reference_bank_policy=SELECTED_REFERENCE_BANK_POLICY,
        allele_policy=SELECTED_ALLELE_POLICY,
        threshold_label=SELECTED_THRESHOLD_LABEL,
        normalized_count_cutoff=NORMALIZED_COUNT_CUTOFF,
        sample_count_cutoff=SAMPLE_COUNT_CUTOFF,
        min_cellbins_per_allele=MIN_CELLBINS_PER_ALLELE,
    )
    return asdict(policy)


__all__ = ["FrozenDarlinJointClonePolicy", "freeze_selected_joint_clone_policy"]
