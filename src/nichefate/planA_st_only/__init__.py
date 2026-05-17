"""Barcode-free PlanA-ST-only v1 production facades.

The facade package exposes production-style functional module names while
retaining the legacy PlanA-K milestone modules as implementation provenance.
Importing this package does not run production steps or import heavy numerical
backends.
"""

from .module_registry import (
    CLAIM_GUARDRAILS,
    FINAL_INDEX_ROOT,
    FINAL_RESULT_PACKAGE,
    FORBIDDEN_MAIN_CLAIMS,
    LEGACY_TO_PRODUCTION,
    PRODUCTION_PIPELINE,
    legacy_mapping_rows,
    module_by_name,
    production_rows,
)

__all__ = [
    "CLAIM_GUARDRAILS",
    "FINAL_INDEX_ROOT",
    "FINAL_RESULT_PACKAGE",
    "FORBIDDEN_MAIN_CLAIMS",
    "LEGACY_TO_PRODUCTION",
    "PRODUCTION_PIPELINE",
    "legacy_mapping_rows",
    "module_by_name",
    "production_rows",
]
