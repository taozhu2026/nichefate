import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m4b_01_design_terminal_macrostates.py"
SPEC = importlib.util.spec_from_file_location("m4b_terminal_design", SCRIPT_PATH)
m4b = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m4b
SPEC.loader.exec_module(m4b)


def toy_node_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_node_index": [0, 1, 2, 3],
            "anchor_id": ["a::0", "a::1", "b::0", "b::1"],
            "slice_id": ["a", "a", "b", "b"],
            "anchor_index": [0, 1, 0, 1],
            "anchor_cell_id": ["c0", "c1", "c2", "c3"],
            "time": ["early", "early", "late", "late"],
            "time_day": [0.0, 0.0, 10.0, 10.0],
            "mouse_id": ["m0", "m0", "m1", "m1"],
            "cell_type_l1": ["x", "x", "y", "z"],
            "cell_type_l2": ["x2", "x2", "y2", "z2"],
            "cell_type_l3": ["x3", "x3", "y3", "z3"],
            "is_final_time": [False, False, True, True],
        }
    )


def toy_terminal_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_node_index": [2, 3],
            "anchor_id": ["b::0", "b::1"],
            "slice_id_node": ["b", "b"],
            "anchor_index": [0, 1],
            "anchor_cell_id": ["c2", "c3"],
            "time_node": ["late", "late"],
            "time_day_node": [10.0, 10.0],
            "mouse_id_node": ["m1", "m1"],
            "cell_type_l1": ["y", "z"],
            "cell_type_l2": ["y2", "z2"],
            "cell_type_l3": ["y3", "z3"],
            "f1": [1.0, 3.0],
            "f2": [5.0, 5.0],
        }
    )


def test_final_time_inferred_from_max_time_day() -> None:
    day, label = m4b.infer_final_time(toy_node_table())
    terminal = m4b.select_terminal_nodes(toy_node_table())

    assert day == pytest.approx(10.0)
    assert label == "late"
    assert terminal["anchor_id"].tolist() == ["b::0", "b::1"]


def test_feature_group_resolution_uses_m2_schema_and_fails_cleanly() -> None:
    m2_schema = {"numeric_feature_columns": ["f1", "f2", "unused"]}
    m3_groups = {"feature_groups": {"group_a": ["f1", "f2"], "bad": ["missing"]}}

    columns, mapping = m4b.resolve_feature_columns(m2_schema, m3_groups, ["group_a"])

    assert columns == ["f1", "f2"]
    assert mapping.loc[0, "mapping_status"] == "mapped_to_m2_schema"
    with pytest.raises(KeyError, match="does not map cleanly"):
        m4b.resolve_feature_columns(m2_schema, m3_groups, ["bad"])


def test_robust_standardization_handles_near_constant_features_safely() -> None:
    matrix, report = m4b.robust_standardize_features(toy_terminal_data(), ["f1", "f2"], 1e-6)

    assert matrix.shape == (2, 2)
    assert report.loc[report["feature"] == "f2", "near_constant"].iloc[0]
    assert np.allclose(matrix[:, 1], 0.0)


def test_toy_terminal_macrostate_assignments_have_expected_columns() -> None:
    assignments = m4b.build_assignments(
        toy_terminal_data(),
        np.array([0, 1], dtype=np.int32),
        selected_k=2,
        feature_matrix=np.zeros((2, 2), dtype=np.float32),
    )
    assignments["incoming_degree_structural"] = [2, 3]
    assignments["incoming_mass_structural"] = [0.5, 1.5]
    summary = m4b.macrostate_summary(assignments)

    assert {"global_node_index", "terminal_macrostate_id", "terminal_macrostate_label"} <= set(assignments.columns)
    assert int(summary["incoming_degree_sum_structural"].sum()) == 5
    assert summary["incoming_mass_sum_structural"].sum() == pytest.approx(2.0)


def test_candidate_k_selection_keeps_default_when_valid() -> None:
    rng = np.random.default_rng(1)
    matrix = np.vstack(
        [
            rng.normal(loc=-3, scale=0.1, size=(30, 2)),
            rng.normal(loc=3, scale=0.1, size=(30, 2)),
        ]
    ).astype(np.float32)

    selected, labels_by_k, qc = m4b.run_candidate_clusterings(matrix, [2, 3], 2, 1, 0.01)

    assert selected == 2
    assert set(labels_by_k) == {2, 3}
    assert qc.loc[qc["n_macrostates"] == 2, "selected_default"].iloc[0]


def test_m4c_handoff_json_contains_required_route_and_barcode_notes(tmp_path: Path) -> None:
    paths = {
        "output_root": tmp_path / "m4b",
        "p_forward": tmp_path / "m4a" / "transition_objects" / "P_forward_no_terminal_selfloops.npz",
        "p_absorbing": tmp_path / "m4a" / "transition_objects" / "P_absorbing_terminal_selfloops.npz",
        "node_table": tmp_path / "m4a" / "node_table" / "global_node_table.parquet",
    }

    payload = m4b.m4c_inputs_payload(paths, selected_k=2, final_time="late", final_time_day=10.0)

    assert payload["recommended_fate_computation"] == "time-layered backward propagation to terminal macrostate labels"
    assert "barcode-aware M3" in payload["barcode_compatibility_note"]
    assert "M4C is Markov baseline v1" in payload["route_compatibility_note"]
    assert "M4D is the standard GPCCA/CellRank-inspired Markov route" in payload["route_compatibility_note"]
    assert "Branched NicheFlow / BranchSBM" in payload["route_compatibility_note"]


def test_no_downstream_or_regulator_outputs_are_declared() -> None:
    forbidden = ["gpcca_result", "fate_probability", "absorption_probability", "branched_nicheflow", "m5", "regulator"]

    assert all(value is True for value in m4b.NO_DOWNSTREAM_FLAGS.values())
    assert not any(token in "terminal_macrostate_assignments.parquet" for token in forbidden)
    assert "not absorption probabilities or fate probabilities" in m4b.STRUCTURAL_DIAGNOSTIC_NOTE


def test_read_only_inputs_are_not_output_targets(tmp_path: Path) -> None:
    paths = {
        "output_root": tmp_path / "m4b",
        "p_forward": tmp_path / "m4a" / "transition_objects" / "P_forward_no_terminal_selfloops.npz",
        "p_absorbing": tmp_path / "m4a" / "transition_objects" / "P_absorbing_terminal_selfloops.npz",
        "node_table": tmp_path / "m4a" / "node_table" / "global_node_table.parquet",
    }
    payload = m4b.m4c_inputs_payload(paths, selected_k=2, final_time="late", final_time_day=10.0)

    assert str(tmp_path / "m4b") in payload["terminal_macrostate_assignments"]
    assert payload["node_table"] == str(paths["node_table"])
