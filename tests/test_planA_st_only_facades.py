"""Import and registry checks for PlanA-ST-only production facades."""

from __future__ import annotations

import importlib


FACADE_MODULES = (
    "nichefate.planA_st_only",
    "nichefate.planA_st_only.spatial_dataset_adapter",
    "nichefate.planA_st_only.niche_builder",
    "nichefate.planA_st_only.niche_encoder",
    "nichefate.planA_st_only.metaniche_coarsener",
    "nichefate.planA_st_only.transition_evidence",
    "nichefate.planA_st_only.kernel_assembly",
    "nichefate.planA_st_only.gpcca_macrostate_inference",
    "nichefate.planA_st_only.endpoint_markov_inference",
    "nichefate.planA_st_only.fate_probability",
    "nichefate.planA_st_only.biological_annotation",
    "nichefate.planA_st_only.result_visualization",
    "nichefate.planA_st_only.result_package",
    "nichefate.planA_st_only.module_registry",
)


def test_planA_st_only_facades_import() -> None:
    for module_name in FACADE_MODULES:
        importlib.import_module(module_name)


def test_planA_st_only_registry_contains_claim_boundary() -> None:
    registry = importlib.import_module("nichefate.planA_st_only.module_registry")
    rows = registry.production_rows()
    assert len(rows) == 12
    assert registry.module_by_name("FateProbability").legacy_name == "M4C absorption"
    assert "ST-only" in registry.CLAIM_GUARDRAILS["barcode_boundary"]
