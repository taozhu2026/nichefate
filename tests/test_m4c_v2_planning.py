import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m4c_v2_00_planning_and_handoff.py"
SPEC = importlib.util.spec_from_file_location("m4c_v2_plan", SCRIPT_PATH)
m4c_v2_plan = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m4c_v2_plan
SPEC.loader.exec_module(m4c_v2_plan)


def test_output_root_rejects_protected_and_ssd_paths(tmp_path: Path) -> None:
    safe = tmp_path / "m4c_v2_plan"
    assert m4c_v2_plan.validate_output_root(safe) == safe.resolve()

    with pytest.raises(ValueError, match="overlaps protected"):
        m4c_v2_plan.validate_output_root(
            Path("/home/zhutao/scratch/nichefate/m4a_v2/reports")
        )

    with pytest.raises(ValueError, match="Refusing /ssd path"):
        m4c_v2_plan.validate_output_root(Path("/ssd/nichefate/m4c_v2_plan"))


def test_planned_fate_shape_uses_node_and_endpoint_counts() -> None:
    assert m4c_v2_plan.planned_fate_shape(12) == "1439542x12"
    assert m4c_v2_plan.planned_fate_shape(3, node_count=10) == "10x3"


def test_endpoint_taxonomy_summary_groups_reuse_categories() -> None:
    mapping = pd.DataFrame(
        {
            "refined_endpoint_id": ["a", "b", "c", "d"],
            "confidence_tier_after_refinement": [
                "high_confidence_biological_endpoint",
                "plausible_but_mixed_endpoint",
                "rare_biological_endpoint",
                "slice_or_mouse_associated_endpoint",
            ],
        }
    )
    summary = m4c_v2_plan.summarize_endpoint_taxonomy(mapping)

    counts = dict(zip(summary["reuse_category"], summary["n_endpoints"], strict=True))
    assert counts["high_confidence"] == 1
    assert counts["plausible_but_mixed"] == 1
    assert counts["low_size_or_low_mass"] == 1
    assert counts["slice_or_mouse_associated"] == 1


def test_endpoint_count_preserves_raw_terminal_columns_for_merge_candidates() -> None:
    mapping = pd.DataFrame(
        {
            "raw_terminal_macrostate": [0, 1, 2],
            "refined_endpoint_id": ["a", "b", "b"],
        }
    )

    assert m4c_v2_plan.endpoint_count_from_mapping(mapping) == 3


def test_required_input_inventory_reports_missing_required_inputs(
    tmp_path: Path,
) -> None:
    inventory = m4c_v2_plan.build_required_input_inventory(tmp_path)

    assert {"input_name", "path", "status", "read_only"} <= set(inventory.columns)
    assert bool(inventory["required"].all())
    assert set(inventory["status"]) == {"FAIL"}
    assert bool(inventory["read_only"].all())


def test_planned_outputs_are_contract_only() -> None:
    outputs = m4c_v2_plan.build_planned_output_inventory(
        Path("/home/zhutao/scratch/nichefate/m4c_v2"),
        endpoint_count=12,
    )

    assert not bool(outputs["production_created_in_this_task"].any())
    assert outputs["planned_path"].str.contains("/m4c_v2/").all()
    assert not outputs["planned_path"].str.contains("gpcca|branchsbm|barcode").any()
    fate = outputs.query("output_name == 'fate_probability_matrix_v2'").iloc[0]
    assert fate["expected_shape_or_rows"] == "1439542x12"


def test_recommendation_requires_inputs_and_readiness() -> None:
    state = {
        "input_inventory": pd.DataFrame(
            {"required": [True, True], "status": ["PASS", "PASS"]}
        ),
        "m4a_v2_summary": {
            "full_qc_status": "PASS",
            "m4c_v2_readiness_status": "PASS",
        },
        "m4a_v2_readiness": pd.DataFrame({"status": ["PASS", "PASS"]}),
    }
    recommendation, reason = m4c_v2_plan.choose_execution_recommendation(state)

    assert recommendation == "implement_m4c_v2_runner_dryrun_preflight_only"
    assert "dry-run" in reason

    state["m4a_v2_readiness"] = pd.DataFrame({"status": ["PASS", "FAIL"]})
    recommendation, _ = m4c_v2_plan.choose_execution_recommendation(state)
    assert recommendation == "repair_planning_inputs_before_m4c_v2_runner"


def test_planning_checklist_carries_safety_statuses() -> None:
    state = {
        "input_inventory": pd.DataFrame(
            {"required": [True], "status": ["PASS"]}
        ),
        "m4a_v2_summary": {
            "full_qc_status": "PASS",
            "m4c_v2_readiness_status": "PASS",
        },
        "m4a_v2_readiness": pd.DataFrame({"status": ["PASS"]}),
        "m4c_v1": {"uses_time_layered_backward_propagation": True},
        "endpoint_count": 12,
    }
    checklist = m4c_v2_plan.build_planning_checklist(
        state,
        {
            "upstream_metadata_diff_count": 0,
            "forbidden_downstream_diff_count": 0,
            "ssd_output_count": 0,
        },
    )

    assert set(checklist["status"]) == {"PASS"}
    assert "upstream_metadata_diff_zero" in set(checklist["check"])
