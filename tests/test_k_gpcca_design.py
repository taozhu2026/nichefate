import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "k_gpcca_00_design.py"
SPEC = importlib.util.spec_from_file_location("k_gpcca_design", SCRIPT_PATH)
k_gpcca_design = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = k_gpcca_design
SPEC.loader.exec_module(k_gpcca_design)


def test_output_root_rejects_protected_forbidden_and_ssd_paths(tmp_path: Path) -> None:
    safe = tmp_path / "k_gpcca_design"

    assert k_gpcca_design.validate_output_root(safe) == safe.resolve()

    with pytest.raises(ValueError, match="protected production root"):
        k_gpcca_design.validate_output_root(
            Path("/home/zhutao/scratch/nichefate/m4a_v2/reports")
        )

    with pytest.raises(ValueError, match="forbidden execution root"):
        k_gpcca_design.validate_output_root(
            Path("/home/zhutao/scratch/nichefate/k_gpcca/reports")
        )

    with pytest.raises(ValueError, match="Refusing /ssd path"):
        k_gpcca_design.validate_output_root(Path("/ssd/nichefate/k_gpcca_design"))


def test_output_paths_include_all_required_design_artifacts(tmp_path: Path) -> None:
    paths = k_gpcca_design.output_paths(tmp_path / "k_gpcca_design")

    for name in k_gpcca_design.REPORT_NAMES:
        assert name in paths
        assert paths[name].parent.name == "reports"
    for name in k_gpcca_design.CSV_NAMES:
        assert name in paths
        assert paths[name].parent == (tmp_path / "k_gpcca_design").resolve()
    assert paths["summary"].name == "k_gpcca_design_summary.json"


def test_candidate_parameter_grid_has_expected_schema_and_values() -> None:
    grid = k_gpcca_design.build_candidate_parameter_grid()

    assert {
        "grid_id",
        "route",
        "cross_time_source",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "within_time_k",
        "similarity_metric",
        "scope",
        "priority",
        "rationale",
    } <= set(grid.columns)
    assert {0.01, 0.03, 0.05, 0.10} <= set(grid["gamma"])
    assert {"M3-v1", "M3-v2", "M3-v1_v2_mixed"} <= set(grid["cross_time_source"])
    assert {"full_resolution_subset", "supernode", "future_barcode"} <= set(grid["route"])
    assert set(grid["within_time_k"]) <= {30, 50}
    assert "future_barcode_placeholder" in set(grid["grid_id"])


def test_design_checklist_requires_design_only_and_pygpcca_policy() -> None:
    checklist = k_gpcca_design.build_design_checklist()

    assert set(checklist["status"]) == {"PASS"}
    assert "design_only_scope" in set(checklist["check"])
    assert "pygpcca_only_policy_defined" in set(checklist["check"])
    assert checklist["failure_behavior"].str.contains("heuristic", case=False).any()


def test_planned_output_inventory_creates_nothing_in_design_task() -> None:
    outputs = k_gpcca_design.build_planned_output_inventory()

    assert not outputs["created_in_this_task"].any()
    assert outputs["planned_path"].str.contains("/home/zhutao/scratch/nichefate/k_gpcca/").all()
    assert {"future_kernel", "future_gpcca", "future_after_darlin"} <= set(outputs["category"])


def test_distinction_and_policy_reports_are_decisive() -> None:
    body = "\n".join(
        [
            k_gpcca_design.build_distinction_report(),
            k_gpcca_design.build_pygpcca_policy_report(),
        ]
    )

    assert "K_gpcca is not the same as P_fate" in body
    assert "Strictly time-forward P_fate should not be forced into pyGPCCA" in body
    assert "Custom GPCCA-like code must not be used as formal GPCCA output" in body
    assert "standard pyGPCCA" in body
    assert "heuristic macrostate fallback as a final result" in body


def test_input_contract_contains_required_barcode_future_fields() -> None:
    contract = k_gpcca_design.build_input_contract_rows()
    barcode = contract.query("input_name == 'darlin_processed_clone_tables'").iloc[0]

    assert barcode["required_status"] == "future_optional"
    assert "clone_id" in barcode["required_columns_or_objects"]
    assert "barcode_id" in barcode["required_columns_or_objects"]
    assert "not used before DARLIN onboarding" in barcode["failure_behavior"]


def test_risk_register_has_mitigations_for_all_risks() -> None:
    risks = k_gpcca_design.build_risk_register()

    assert len(risks) >= 8
    assert risks["risk_id"].is_unique
    assert risks["mitigation"].str.len().min() > 20
    assert "pygpcca_failure" in set(risks["risk_id"])
    assert "runtime_memory_infeasible" in set(risks["risk_id"])
