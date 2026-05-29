from __future__ import annotations

import importlib
import json
from pathlib import Path

from nichefate.darlin_clone_signature.reporting import positive_claim_hits


REGISTRY_PATH = Path("configs/module_registry/nichefate_module_registry.json")


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_concrete_registry_contains_niche_encoder_mapping() -> None:
    registry = load_registry()
    modules = {row["public_name"]: row for row in registry["modules"]}

    encoder = modules["NicheEncoder"]
    assert encoder["classification"] == "shared_core_algorithm"
    assert "src/nichefate/planA_st_only/niche_encoder.py" in encoder["current_code_surface"]
    assert "src/nichefate/representation.py" in encoder["current_code_surface"]
    assert "conceptual module backed by facade" in encoder["relation"]

    importlib.import_module("nichefate.planA_st_only.niche_encoder")


def test_lineage_modules_are_evidence_specific_not_substrate_replacements() -> None:
    registry = load_registry()
    modules = {row["public_name"]: row for row in registry["modules"]}

    for public_name in [
        "LineageEvidenceAdapter",
        "DARLINJointCloneCaller",
        "CloneNicheIntegrator",
    ]:
        assert modules[public_name]["classification"] == "evidence_specific_lineage"
        assert modules[public_name]["relation"].startswith("E")

    m2 = modules["NicheEncoder"]
    assert "E1_lineage_aware" in m2["evidence_regime_compatibility"]


def test_benchmark_wrappers_are_not_generic_algorithm_modules() -> None:
    registry = load_registry()
    wrappers = [row for row in registry["modules"] if row["classification"] == "benchmark_wrapper"]
    assert len(wrappers) == 1
    assert wrappers[0]["public_name"] == "L126 Benchmark Wrapper Set"
    assert wrappers[0]["relation"] == "benchmark-specific wrapper, not an algorithm module"


def test_registry_docs_have_no_forbidden_l126_claims() -> None:
    paths = [
        Path("README.md"),
        Path("docs/algorithm_module_registry.md"),
        Path("docs/pipeline_module_index.md"),
        Path("docs/modules/lineage_insertion_points.md"),
        Path("reports/concrete_module_registry/00_MODULE_AUDIT.md"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert positive_claim_hits(text) == []
