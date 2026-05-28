from __future__ import annotations

from nichefate.darlin_joint_clone_niche_v1 import REQUIRED_CLONE_QC_COLUMNS

ASSAYS: tuple[str, ...] = ("CA", "TA", "RA")
ASSAY_SCOPED_FEATURE_SEPARATOR = "::"
REFERENCE_BANK_POLICIES: tuple[str, ...] = ("plain", "gr", "union")
ALLELE_REFERENCE_STATUS_VALUES: tuple[str, ...] = (
    "unknown",
    "reference_mapped_only",
    "de_novo_only",
    "mixed_reference_and_de_novo",
)
DARLIN_JOINT_CLONE_QC_FIELDS: tuple[str, ...] = tuple(REQUIRED_CLONE_QC_COLUMNS)


def joint_clone_qc_field_names() -> tuple[str, ...]:
    return DARLIN_JOINT_CLONE_QC_FIELDS


__all__ = [
    "ALLELE_REFERENCE_STATUS_VALUES",
    "ASSAY_SCOPED_FEATURE_SEPARATOR",
    "ASSAYS",
    "DARLIN_JOINT_CLONE_QC_FIELDS",
    "REFERENCE_BANK_POLICIES",
    "joint_clone_qc_field_names",
]
