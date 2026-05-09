import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "k_gpcca_01_pilot_kernel_preflight.py"
SPEC = importlib.util.spec_from_file_location("k_gpcca_pilot", SCRIPT_PATH)
k_gpcca_pilot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = k_gpcca_pilot
SPEC.loader.exec_module(k_gpcca_pilot)


def minimal_config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "output_root": str(tmp_path / "k_gpcca_pilot"),
            "reports_dir": str(tmp_path / "k_gpcca_pilot" / "reports"),
        },
        "protected_roots": [
            "/home/zhutao/scratch/nichefate/m3",
            "/home/zhutao/scratch/nichefate/m4a_v2",
        ],
        "forbidden_downstream_roots": [
            "/home/zhutao/scratch/nichefate/k_gpcca",
        ],
    }


def toy_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_node_index": list(range(12)),
            "anchor_id": [f"s{i % 3}::{i}" for i in range(12)],
            "slice_id": [f"s{i % 3}" for i in range(12)],
            "mouse_id": [f"m{i % 2}" for i in range(12)],
            "time": ["D9"] * 4 + ["D21"] * 4 + ["D35"] * 4,
            "time_day": [9] * 4 + [21] * 4 + [35] * 4,
            "cell_type_l3": [f"c{i % 2}" for i in range(12)],
        }
    )


def test_config_parsing_reads_yaml() -> None:
    config = k_gpcca_pilot.load_config(PROJECT_ROOT / "configs" / "k_gpcca_pilot.yaml")

    assert config["project"]["stage"] == "K_gpcca-01"
    assert config["pilot"]["preferred_time_points"] == ["D9", "D21", "D35"]
    assert 30 in config["kernel"]["within_time_k_values"]


def test_output_root_rejects_protected_forbidden_and_ssd_paths(tmp_path: Path) -> None:
    config = minimal_config(tmp_path)
    assert k_gpcca_pilot.validate_output_root(config) == (tmp_path / "k_gpcca_pilot").resolve()

    protected = minimal_config(tmp_path)
    protected["paths"]["output_root"] = "/home/zhutao/scratch/nichefate/m4a_v2/reports"
    with pytest.raises(ValueError, match="protected production root"):
        k_gpcca_pilot.validate_output_root(protected)

    forbidden = minimal_config(tmp_path)
    forbidden["paths"]["output_root"] = "/home/zhutao/scratch/nichefate/k_gpcca/reports"
    with pytest.raises(ValueError, match="forbidden downstream root"):
        k_gpcca_pilot.validate_output_root(forbidden)

    ssd = minimal_config(tmp_path)
    ssd["paths"]["output_root"] = "/ssd/nichefate/k_gpcca_pilot"
    with pytest.raises(ValueError, match="Refusing /ssd path"):
        k_gpcca_pilot.validate_output_root(ssd)


def test_candidate_grid_loading_and_statuses(tmp_path: Path) -> None:
    design_root = tmp_path / "design"
    design_root.mkdir()
    frame = pd.DataFrame(
        {
            "grid_id": ["v1", "mixed", "barcode"],
            "route": ["full_resolution_subset", "full_resolution_subset", "future_barcode"],
            "cross_time_source": ["M3-v1", "M3-v1_v2_mixed", "M3-v2_plus_barcode"],
            "alpha": [0.6, 0.6, 0.5],
            "beta": [0.35, 0.35, 0.3],
            "gamma": [0.05, 0.05, 0.05],
            "delta": [0.0, 0.0, 0.15],
            "within_time_k": [30, 30, 30],
            "similarity_metric": ["cosine", "cosine", "cosine"],
            "scope": ["pilot", "pilot", "future"],
            "priority": ["default", "review_only", "future_after_darlin"],
            "rationale": ["ok", "review", "future"],
        }
    )
    frame.to_csv(design_root / "k_gpcca_candidate_parameter_grid.csv", index=False)

    grid = k_gpcca_pilot.load_candidate_grid(design_root)
    summary = k_gpcca_pilot.build_candidate_preflight_summary(grid)

    assert len(grid) == 3
    assert set(summary["status"]) == {
        "DRYRUN_PREFLIGHT",
        "REVIEW_PLANNING_ONLY",
        "SKIP_FUTURE_BARCODE",
    }


def test_subset_selection_is_deterministic_and_time_bounded() -> None:
    left = k_gpcca_pilot.deterministic_select_nodes(toy_nodes(), ["D9", "D21", "D35"], 6)
    right = k_gpcca_pilot.deterministic_select_nodes(toy_nodes(), ["D9", "D21", "D35"], 6)

    assert left["global_node_index"].tolist() == right["global_node_index"].tolist()
    assert len(left) == 6
    assert set(left["time"]) == {"D9", "D21", "D35"}


def test_within_time_graph_toy_planning_counts_edges() -> None:
    selected = toy_nodes()
    plan = k_gpcca_pilot.within_time_graph_plan(selected, [2, 10], bytes_per_nnz=16)

    k2 = plan.query("k == 2").iloc[0]
    k10 = plan.query("k == 10").iloc[0]
    assert k2["expected_nnz"] == 24
    assert k10["expected_nnz"] == 36
    assert bool(k2["row_coverage_expected"]) is True


def test_cross_time_schema_validation_toy_case() -> None:
    columns = [
        "source_anchor_id",
        "target_anchor_id",
        "source_time",
        "target_time",
        "row_normalized_transition_prob",
    ]

    assert k_gpcca_pilot.validate_cross_time_schema(columns, "row_normalized_transition_prob") == []
    missing = k_gpcca_pilot.validate_cross_time_schema(columns[:-1], "row_normalized_transition_prob")
    assert missing == ["row_normalized_transition_prob"]


def test_self_loop_count_logic_flags_high_gamma() -> None:
    plan = k_gpcca_pilot.self_loop_plan(12, [0.01, 0.10])

    assert set(plan["expected_self_loop_nnz"]) == {12}
    assert bool(plan.query("gamma == 0.1").iloc[0]["dominance_warning"]) is True
    assert bool(plan.query("gamma == 0.01").iloc[0]["dominance_warning"]) is False


def test_no_npz_written_check_and_output_schemas(tmp_path: Path) -> None:
    output_root = tmp_path / "k_gpcca_pilot"
    output_root.mkdir()
    assert k_gpcca_pilot.count_npz_outputs(output_root) == 0

    candidate = pd.DataFrame(
        {
            "grid_id": ["v1"],
            "route": ["full_resolution_subset"],
            "cross_time_source": ["M3-v1"],
            "alpha": [0.6],
            "beta": [0.35],
            "gamma": [0.05],
            "delta": [0.0],
            "within_time_k": [30],
            "similarity_metric": ["cosine"],
            "status": ["DRYRUN_PREFLIGHT"],
            "note": ["ok"],
        }
    )
    within = k_gpcca_pilot.within_time_graph_plan(toy_nodes(), [30])
    cross = pd.DataFrame(
        {
            "source": ["M3-v1"],
            "estimated_in_pilot_edges": [20],
        }
    )
    self_plan = k_gpcca_pilot.self_loop_plan(12, [0.05])
    inventory = k_gpcca_pilot.build_expected_kernel_inventory(
        candidate,
        12,
        within,
        cross,
        self_plan,
        16,
    )

    assert {
        "candidate_id",
        "future_matrix_object",
        "created_in_this_task",
        "estimated_sparse_memory_mb",
    } <= set(inventory.columns)
    assert not inventory["created_in_this_task"].any()
    assert inventory.iloc[0]["future_matrix_object"].endswith(".npz")
