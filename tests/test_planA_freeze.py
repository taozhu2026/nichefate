import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "planA_00_freeze_p_fate_branch.py"
SPEC = importlib.util.spec_from_file_location("planA_freeze", SCRIPT_PATH)
planA_freeze = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planA_freeze
SPEC.loader.exec_module(planA_freeze)


def test_output_root_rejects_protected_and_ssd_paths(tmp_path: Path) -> None:
    safe = tmp_path / "planA_freeze"

    assert planA_freeze.validate_output_root(safe) == safe.resolve()

    with pytest.raises(ValueError, match="protected production root"):
        planA_freeze.validate_output_root(
            Path("/home/zhutao/scratch/nichefate/m4a_v2/reports")
        )

    with pytest.raises(ValueError, match="Refusing /ssd path"):
        planA_freeze.validate_output_root(Path("/ssd/nichefate/planA_freeze"))


def test_artifact_inventory_schema_and_read_only_policy() -> None:
    inventory = planA_freeze.build_artifact_inventory()

    assert {
        "stage",
        "version",
        "path",
        "role",
        "status",
        "frozen_or_reference",
        "safe_to_reuse_for_darlin_adapter",
        "read_only_in_future_workflows",
    } <= set(inventory.columns)
    assert {"M1", "M2", "M3", "M4A", "M4C", "M4E"} <= set(inventory["stage"])
    assert inventory["read_only_in_future_workflows"].all()


def test_stage_status_matrix_carries_required_status_categories() -> None:
    stage_status = planA_freeze.build_stage_status_matrix()

    assert {
        "frozen",
        "needs_design",
        "needs_pilot",
        "future_after_darlin",
    } <= set(stage_status["status_category"])
    assert "P_fate" in set(stage_status["branch"])
    assert "K_gpcca" in set(stage_status["branch"])
    assert "Plan B" in set(stage_status["branch"])


def test_remaining_tasks_include_pre_darlin_contract_items() -> None:
    tasks = planA_freeze.build_remaining_tasks()
    report = planA_freeze.build_pre_darlin_checklist_report(tasks)

    assert {"completed", "frozen", "needs_design", "needs_pilot", "future_after_darlin"} <= set(
        tasks["status_category"]
    )
    assert tasks["required_before_darlin"].any()
    assert tasks["task"].str.contains("barcode adapter", case=False).any()
    assert tasks["task"].str.contains("DARLIN data inventory", case=False).any()
    assert "| task_id | task | branch |" in report


def test_distinction_reports_avoid_forbidden_final_result_terms() -> None:
    body = "\n".join(
        [
            planA_freeze.build_distinction_report(),
            planA_freeze.build_standard_policy_report(),
            planA_freeze.build_k_requirements_report(),
        ]
    )

    forbidden_phrases = [
        "custom GPCCA",
        "surrogate GPCCA",
        "GPCCA-like final result",
        "lineage-validated fate",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in body
    assert "P_fate and K_gpcca are not the same matrix" in body
    assert "standard pyGPCCA" in body


def test_endpoint_taxonomy_counts_preserve_raw_columns() -> None:
    mapping = pd.DataFrame(
        {
            "raw_terminal_macrostate": [0, 1, 2],
            "refined_endpoint_id": ["a", "b", "b"],
            "confidence_tier_after_refinement": [
                "high_confidence_biological_endpoint",
                "plausible_but_mixed_endpoint",
                "rare_biological_endpoint",
            ],
        }
    )

    counts = planA_freeze.endpoint_taxonomy_counts(mapping)

    assert counts["raw_terminal_columns"] == 3
    assert counts["unique_refined_endpoint_ids"] == 2
    assert counts["high_confidence"] == 1
    assert counts["plausible_but_mixed"] == 1
    assert counts["low_size_or_low_mass"] == 1


def test_barcode_and_branchsbm_positioning_are_separate() -> None:
    barcode = planA_freeze.build_barcode_positioning_report()
    branchsbm = planA_freeze.build_branchsbm_positioning_report()

    assert "official/lab-standard DARLIN pipeline" in barcode
    assert "processed clone/barcode tables" in barcode
    assert "P_barcode" in barcode
    assert "Plan B" in branchsbm
    assert "not required before DARLIN onboarding" in branchsbm
