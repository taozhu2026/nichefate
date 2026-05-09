import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m4c_01_compute_markov_fate_probabilities.py"
SPEC = importlib.util.spec_from_file_location("m4c_fate_probability", SCRIPT_PATH)
m4c = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m4c
SPEC.loader.exec_module(m4c)


def toy_node_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_node_index": [0, 1, 2, 3, 4],
            "anchor_id": ["d0::0", "d0::1", "d1::0", "d1::1", "d2::0"],
            "slice_id": ["s0", "s0", "s1", "s1", "s2"],
            "anchor_index": [0, 1, 0, 1, 0],
            "anchor_cell_id": ["c0", "c1", "c2", "c3", "c4"],
            "time": ["early", "early", "mid", "mid", "late"],
            "time_day": [0.0, 0.0, 5.0, 5.0, 10.0],
            "mouse_id": ["m0", "m0", "m1", "m1", "m2"],
            "cell_type_l1": ["a", "a", "b", "b", "c"],
            "cell_type_l2": ["a2", "a2", "b2", "b2", "c2"],
            "cell_type_l3": ["a3", "a3", "b3", "b3", "c3"],
        }
    )


def toy_assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_node_index": [4],
            "terminal_macrostate_id": [0],
            "terminal_macrostate_label": ["terminal_macrostate_00"],
        }
    )


def toy_forward() -> sp.csr_matrix:
    rows = np.array([0, 0, 1, 1, 2, 3], dtype=np.int64)
    cols = np.array([2, 3, 2, 3, 4, 4], dtype=np.int64)
    data = np.array([0.25, 0.75, 0.6, 0.4, 1.0, 1.0], dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(5, 5))


def test_toy_time_layered_backward_propagation_and_final_one_hot() -> None:
    node_table, final_mask, final_day, final_time = m4c.validate_global_node_table(toy_node_table(), 5)
    assignments, macro_ids, labels = m4c.validate_terminal_assignments(
        node_table,
        final_mask,
        toy_assignments(),
        "terminal_macrostate_id",
        1,
        1,
    )

    fate, steps = m4c.compute_fate_probabilities(toy_forward(), node_table, assignments, 1, np.dtype("float32"))
    qc = m4c.validate_fate_matrix(fate, final_mask, 1e-6, True, True)

    assert final_day == pytest.approx(10.0)
    assert final_time == "late"
    assert fate.shape == (5, 1)
    assert fate[:, 0].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0])
    assert fate[final_mask].tolist() == [[1.0]]
    assert macro_ids.tolist() == [0]
    assert labels == ["terminal_macrostate_00"]
    assert [step["source_time"] for step in steps] == ["early", "mid"]
    assert qc["final_onehot_error_max"] == pytest.approx(0.0)


def test_final_time_is_inferred_by_max_time_day_without_hard_coded_label() -> None:
    frame = toy_node_table()
    frame["time"] = ["D3", "D3", "D8", "D8", "not_D35"]
    frame["time_day"] = [3.0, 3.0, 8.0, 8.0, 21.0]

    node_table, final_mask, final_day, final_time = m4c.validate_global_node_table(frame, 5)

    assert final_day == pytest.approx(21.0)
    assert final_time == "not_D35"
    assert node_table.loc[final_mask, "global_node_index"].tolist() == [4]


def test_row_sum_validation_failure_is_reported() -> None:
    fate = np.array([[0.2, 0.2], [1.0, 0.0]], dtype=np.float32)
    final_mask = np.array([False, True])

    with pytest.raises(ValueError, match="non-final fate rows exceed"):
        m4c.validate_fate_matrix(fate, final_mask, 1e-6, True, True)


def test_entropy_normalized_plasticity_dominant_fate_and_margin() -> None:
    fate = np.array([[0.25, 0.75], [1.0, 0.0], [1.0000001, 0.0]], dtype=np.float32)

    metrics = m4c.fate_metrics(
        fate,
        np.array([10, 20], dtype=np.int32),
        ["left", "right"],
    )

    assert metrics.loc[0, "plasticity_entropy"] == pytest.approx(
        -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
    )
    assert metrics.loc[0, "normalized_plasticity_entropy"] == pytest.approx(
        metrics.loc[0, "plasticity_entropy"] / np.log(2.0)
    )
    assert metrics["plasticity_entropy"].min() >= 0.0
    assert metrics["normalized_plasticity_entropy"].between(0.0, 1.0).all()
    assert metrics["dominant_fate"].tolist() == [20, 10, 10]
    assert metrics["dominant_fate_probability"].tolist() == pytest.approx([0.75, 1.0, 1.0000001])
    assert metrics["fate_margin_top1_minus_top2"].tolist() == pytest.approx([0.5, 1.0, 1.0000001])


def test_m4b_terminal_macrostate_ids_resolve_to_canonical_m4c_fields() -> None:
    frame = toy_node_table()
    frame.loc[3, ["time", "time_day"]] = ["late", 10.0]
    node_table, final_mask, _, _ = m4c.validate_global_node_table(frame, 5)
    assignments = pd.DataFrame({"global_node_index": [3, 4], "terminal_macrostate_id": [0, 1]})

    resolved, macro_ids, labels = m4c.validate_terminal_assignments(
        node_table,
        final_mask,
        assignments,
        "terminal_macrostate_id",
        2,
        2,
    )

    assert resolved["terminal_macrostate"].tolist() == [0, 1]
    assert resolved["terminal_macrostate_label"].tolist() == ["terminal_macrostate_00", "terminal_macrostate_01"]
    assert macro_ids.tolist() == [0, 1]
    assert labels == ["terminal_macrostate_00", "terminal_macrostate_01"]


def test_terminal_summary_must_match_m4b_assignment_fields() -> None:
    summary = pd.DataFrame(
        {
            "terminal_macrostate_id": [0, 1],
            "terminal_macrostate_label": ["terminal_macrostate_00", "terminal_macrostate_01"],
            "n_nodes": [1, 1],
            "time": ["late", "late"],
            "time_day": [10.0, 10.0],
        }
    )

    resolved = m4c.validate_terminal_summary(
        summary,
        np.array([0, 1], dtype=np.int32),
        ["terminal_macrostate_00", "terminal_macrostate_01"],
        expected_terminal_nodes=2,
        expected_terminal_macrostates=2,
        final_time_day=10.0,
        final_time="late",
    )

    assert resolved["terminal_macrostate_id"].tolist() == [0, 1]


def test_barcode_compatible_evidence_metadata_is_preserved() -> None:
    node_table, final_mask, _, _ = m4c.validate_global_node_table(toy_node_table(), 5)
    assignments, _, _ = m4c.validate_terminal_assignments(
        node_table,
        final_mask,
        toy_assignments(),
        "terminal_macrostate_id",
        1,
        1,
    )
    metrics = m4c.fate_metrics(np.ones((5, 1), dtype=np.float32), np.array([0], dtype=np.int32), ["terminal"])

    summary = m4c.build_node_summary(
        node_table,
        assignments,
        metrics,
        "pseudo_lineage_time_coupled_transition",
        True,
    )

    assert summary["directionality_evidence_source"].unique().tolist() == ["pseudo_lineage_time_coupled_transition"]
    assert summary["barcode_compatible_contract"].unique().tolist() == [True]
    assert summary["global_node_index"].tolist() == list(range(5))


def test_summary_aggregation_fields_include_normalized_and_unnormalized_values() -> None:
    node_summary = pd.DataFrame(
        {
            "global_node_index": [0, 1],
            "time_day": [0.0, 0.0],
            "time": ["early", "early"],
            "dominant_fate": [0, 1],
        }
    )
    fate = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)

    summary = m4c.group_probability_summary(
        node_summary,
        fate,
        ["time_day", "time"],
        np.array([0, 1], dtype=np.int32),
        ["a", "b"],
    ).sort_values("terminal_macrostate")

    assert {"mean_probability", "sum_probability", "normalized_mass_fraction", "dominant_fate_fraction"} <= set(
        summary.columns
    )
    assert summary["mean_probability"].tolist() == pytest.approx([0.4, 0.6])
    assert summary["sum_probability"].tolist() == pytest.approx([0.8, 1.2])
    assert summary["normalized_mass_fraction"].tolist() == pytest.approx([0.4, 0.6])
    assert summary["dominant_fate_fraction"].tolist() == pytest.approx([0.5, 0.5])


def test_no_forbidden_downstream_outputs_are_declared(tmp_path: Path) -> None:
    paths = {"output_root": tmp_path / "m4c", "reports_dir": tmp_path / "m4c" / "reports"}
    outputs = m4c.output_paths(paths)
    forbidden = ["gpcca", "branched_nicheflow", "branchsbm", "m5", "regulator"]

    m4c.validate_no_forbidden_output_paths(outputs)

    assert m4c.NO_DOWNSTREAM_FLAGS["no_gpcca"] is True
    assert m4c.NO_DOWNSTREAM_FLAGS["no_branched_nicheflow_training"] is True
    assert m4c.NO_DOWNSTREAM_FLAGS["no_m5"] is True
    assert m4c.NO_DOWNSTREAM_FLAGS["no_regulator_analysis"] is True
    assert not any(token in str(path).lower() for path in outputs.values() for token in forbidden)


def test_upstream_read_only_paths_are_separate_from_m4c_outputs(tmp_path: Path) -> None:
    config = {
        "paths": {
            "m4a_node_table": str(tmp_path / "m4a" / "node_table" / "global_node_table.parquet"),
            "p_forward": str(tmp_path / "m4a" / "transition_objects" / "P_forward_no_terminal_selfloops.npz"),
            "p_absorbing": str(tmp_path / "m4a" / "transition_objects" / "P_absorbing_terminal_selfloops.npz"),
            "terminal_assignments": str(tmp_path / "m4b" / "terminal_states" / "terminal_macrostate_assignments.parquet"),
            "terminal_summary": str(tmp_path / "m4b" / "terminal_states" / "terminal_macrostate_summary.csv"),
            "output_root": str(tmp_path / "m4c"),
            "reports_dir": str(tmp_path / "m4c" / "reports"),
            "figures_dir": str(tmp_path / "m4c" / "reports" / "figures"),
        }
    }

    paths = m4c.configured_paths(config)
    outputs = m4c.output_paths(paths)

    upstream = {paths[key] for key in ["m4a_node_table", "p_forward", "p_absorbing", "terminal_assignments", "terminal_summary"]}
    assert not upstream.intersection(outputs.values())
    assert all(str(path).startswith(str(tmp_path / "m4c")) for path in outputs.values())


def test_config_validation_rejects_ssd_paths() -> None:
    with pytest.raises(ValueError, match="Refusing to use /ssd"):
        m4c.configured_paths({"paths": {"output_root": "/ssd/forbidden"}})


def test_parse_args_default_config_points_to_m4c_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["script"])

    args = m4c.parse_args()

    assert isinstance(args, argparse.Namespace)
    assert args.config == "configs/m4c_fate_probability.yaml"
