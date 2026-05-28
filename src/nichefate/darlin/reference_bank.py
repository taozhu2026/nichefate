from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .allele_schema import REFERENCE_BANK_POLICIES

DEFAULT_REFERENCE_BANK_POLICY = "union"
REFERENCE_BANK_POLICY_PRIORITY: tuple[str, ...] = ("union", "plain", "gr")


@dataclass(frozen=True)
class ReferenceBankPolicySummary:
    policy_name: str
    description: str


def describe_reference_bank_policy(policy_name: str) -> dict[str, Any]:
    policy_name = str(policy_name)
    return {
        "policy_name": policy_name,
        "is_supported": policy_name in REFERENCE_BANK_POLICIES,
        "priority_rank": (
            REFERENCE_BANK_POLICY_PRIORITY.index(policy_name)
            if policy_name in REFERENCE_BANK_POLICY_PRIORITY
            else None
        ),
        "description": {
            "plain": "plain reference bank",
            "gr": "gr reference bank",
            "union": "union of plain and gr reference banks",
        }.get(policy_name, "unknown reference bank policy"),
    }


__all__ = [
    "DEFAULT_REFERENCE_BANK_POLICY",
    "REFERENCE_BANK_POLICY_PRIORITY",
    "ReferenceBankPolicySummary",
    "describe_reference_bank_policy",
]
