import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m4a_01_assemble_global_transition_object.py"
SPEC = importlib.util.spec_from_file_location("m4a_assembly", SCRIPT_PATH)
m4a = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m4a
SPEC.loader.exec_module(m4a)


def toy_node_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slice_id": ["slice_a", "slice_a", "slice_b", "slice_b"],
            "slice_file": ["a.h5ad", "a.h5ad", "b.h5ad", "b.h5ad"],
            "time": ["T0", "T0", "T1", "T1"],
            "time_day": [0.0, 0.0, 1.0, 1.0],
            "mouse_id": ["m0", "m0", "m1", "m1"],
            "anchor_index": [0, 1, 0, 1],
            "anchor_cell_id": ["c0", "c1", "c2", "c3"],
            "cell_type_l1": ["a", "a", "b", "b"],
            "cell_type_l2": ["a2", "a2", "b2", "b2"],
            "cell_type_l3": ["a3", "a3", "b3", "b3"],
        }
    )


def toy_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_anchor_id": ["slice_a::0", "slice_a::0", "slice_a::1", "slice_a::1"],
            "target_anchor_id": ["slice_b::0", "slice_b::1", "slice_b::0", "slice_b::1"],
            "source_time": ["T0", "T0", "T0", "T0"],
            "target_time": ["T1", "T1", "T1", "T1"],
            "source_slice_id": ["slice_a", "slice_a", "slice_a", "slice_a"],
            "target_slice_id": ["slice_b", "slice_b", "slice_b", "slice_b"],
            "source_mouse_id": ["m0", "m0", "m0", "m0"],
            "target_mouse_id": ["m1", "m1", "m1", "m1"],
            "row_normalized_transition_prob": [0.25, 0.75, 0.4, 0.6],
            "raw_edge_weight": [2.0, 6.0, 4.0, 6.0],
            "mass_adjusted_weight": [1.0, 3.0, 2.0, 3.0],
        }
    )


def toy_config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "m3_edge_root": str(tmp_path),
            "m3_manifest": str(tmp_path / "manifest.csv"),
            "m3_schema": str(tmp_path / "schema.json"),
            "m2_by_slice_root": str(tmp_path),
            "output_root": str(tmp_path / "m4a"),
            "reports_dir": str(tmp_path / "m4a" / "reports"),
            "figures_dir": str(tmp_path / "m4a" / "reports" / "figures"),
        },
        "assembly": {
            "edge_probability_column": "row_normalized_transition_prob",
            "raw_weight_column": "raw_edge_weight",
            "mass_adjusted_weight_column": "mass_adjusted_weight",
            "source_id_column": "source_anchor_id",
            "target_id_column": "target_anchor_id",
            "terminal_time_policy": "final_time_no_outgoing",
            "write_absorbing_terminal_variant": True,
            "final_time_self_loop_weight": 1.0,
            "dtype": "float32",
            "index_dtype": "int64",
        },
        "validation": {
            "expected_edge_rows": 4,
            "expected_source_candidate_k": 2,
            "row_sum_tolerance": 1e-6,
            "fail_on_nan": True,
            "fail_on_negative_probability": True,
            "collapse_warnings_are_failure": False,
        },
        "visualization": {
            "make_figures": False,
            "figure_failure_is_warning": True,
            "use_r_if_available": False,
        },
    }


def mapped_toy_arrays() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    node_table = m4a.build_node_table_from_frames([toy_node_frame()])
    anchor_index = pd.Index(node_table["anchor_id"].astype(str))
    source, target = m4a.map_edge_batch(toy_edges(), anchor_index, "source_anchor_id", "target_anchor_id")
    return node_table, source, target


def test_node_index_construction_infers_final_time_without_d35() -> None:
    node_table = m4a.build_node_table_from_frames([toy_node_frame()])

    final_day, final_time = m4a.infer_final_time(node_table)

    assert final_day == pytest.approx(1.0)
    assert final_time == "T1"
    assert node_table["global_node_index"].tolist() == [0, 1, 2, 3]
    assert node_table["is_final_time"].tolist() == [False, False, True, True]


def test_sparse_matrix_assembly_and_absorbing_policy() -> None:
    node_table, source, target = mapped_toy_arrays()
    edges = toy_edges()
    final_indices = node_table.loc[node_table["is_final_time"], "global_node_index"].to_numpy(dtype=np.int64)

    matrices = m4a.assemble_sparse_matrices(
        source,
        target,
        edges["row_normalized_transition_prob"].to_numpy(dtype=np.float32),
        edges["raw_edge_weight"].to_numpy(dtype=np.float32),
        edges["mass_adjusted_weight"].to_numpy(dtype=np.float32),
        len(node_table),
        final_indices,
        "float32",
        1.0,
        True,
    )
    qc = m4a.row_sum_qc(
        matrices["P_forward_no_terminal_selfloops"],
        matrices["P_absorbing_terminal_selfloops"],
        node_table,
        1e-6,
    )

    forward_sums = np.asarray(matrices["P_forward_no_terminal_selfloops"].sum(axis=1)).ravel()
    absorbing_sums = np.asarray(matrices["P_absorbing_terminal_selfloops"].sum(axis=1)).ravel()
    assert matrices["P_forward_no_terminal_selfloops"].shape == (4, 4)
    assert matrices["P_forward_no_terminal_selfloops"].nnz == 4
    assert matrices["P_absorbing_terminal_selfloops"].nnz == 6
    assert forward_sums.tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert absorbing_sums.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert qc["forward_nonfinal_row_sum_error"]["rows_exceeding_1e_6"] == 0
    assert qc["forward_nonfinal_row_sum_error"]["rows_exceeding_1e_5"] == 0


def test_raw_and_mass_weight_matrices_align_with_forward_pattern() -> None:
    node_table, source, target = mapped_toy_arrays()
    edges = toy_edges()
    final_indices = node_table.loc[node_table["is_final_time"], "global_node_index"].to_numpy(dtype=np.int64)
    matrices = m4a.assemble_sparse_matrices(
        source,
        target,
        edges["row_normalized_transition_prob"].to_numpy(dtype=np.float32),
        edges["raw_edge_weight"].to_numpy(dtype=np.float32),
        edges["mass_adjusted_weight"].to_numpy(dtype=np.float32),
        len(node_table),
        final_indices,
        "float32",
        1.0,
        True,
    )

    p_forward = matrices["P_forward_no_terminal_selfloops"]
    assert m4a.sparse_patterns_equal(p_forward, matrices["W_raw_edge_weight"])
    assert m4a.sparse_patterns_equal(p_forward, matrices["W_mass_adjusted_weight"])


def test_absorbing_variant_preserves_zero_probability_edge_pattern() -> None:
    node_table, source, target = mapped_toy_arrays()
    edges = toy_edges()
    probabilities = np.array([0.0, 1.0, 0.4, 0.6], dtype=np.float32)
    final_indices = node_table.loc[node_table["is_final_time"], "global_node_index"].to_numpy(dtype=np.int64)

    matrices = m4a.assemble_sparse_matrices(
        source,
        target,
        probabilities,
        edges["raw_edge_weight"].to_numpy(dtype=np.float32),
        edges["mass_adjusted_weight"].to_numpy(dtype=np.float32),
        len(node_table),
        final_indices,
        "float32",
        1.0,
        True,
    )

    assert matrices["P_forward_no_terminal_selfloops"].nnz == 4
    assert matrices["P_absorbing_terminal_selfloops"].nnz == 6
    m4a.row_sum_qc(
        matrices["P_forward_no_terminal_selfloops"],
        matrices["P_absorbing_terminal_selfloops"],
        node_table,
        1e-6,
    )


def test_duplicate_node_ids_are_rejected() -> None:
    frame = toy_node_frame()
    frame.loc[1, "anchor_index"] = 0

    with pytest.raises(ValueError, match="Duplicate anchor IDs"):
        m4a.build_node_table_from_frames([frame])


def test_duplicate_edge_pairs_are_rejected_before_sparse_conversion() -> None:
    source = np.array([0, 0], dtype=np.int64)
    target = np.array([1, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="Duplicate .* pairs"):
        m4a.check_duplicate_edge_pairs(source, target, 4)


def test_nan_and_negative_probabilities_fail() -> None:
    config = toy_config(Path("/tmp"))
    frame = toy_edges()
    frame.loc[0, "row_normalized_transition_prob"] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        m4a.validate_edge_values(
            frame,
            "row_normalized_transition_prob",
            "raw_edge_weight",
            "mass_adjusted_weight",
            config,
        )

    frame = toy_edges()
    frame.loc[0, "row_normalized_transition_prob"] = -0.1
    with pytest.raises(ValueError, match="negative probabilities"):
        m4a.validate_edge_values(
            frame,
            "row_normalized_transition_prob",
            "raw_edge_weight",
            "mass_adjusted_weight",
            config,
        )


def test_endpoint_preflight_fails_on_inconsistent_anchor_ids(tmp_path: Path) -> None:
    node_table = m4a.build_node_table_from_frames([toy_node_frame()])
    bad_edges = toy_edges()
    bad_edges.loc[0, "target_anchor_id"] = "missing::0"
    shard_path = tmp_path / "candidate_edges.parquet"
    bad_edges.to_parquet(shard_path, index=False)
    manifest = pd.DataFrame(
        {
            "shard_id": ["toy"],
            "observed_edge_rows": [len(bad_edges)],
            "output_parquet": [str(shard_path)],
            "m3_16_status": ["FINAL_QC_VALIDATED"],
        }
    )

    with pytest.raises(ValueError, match="do not map"):
        m4a.preflight_edge_endpoint_mapping(
            manifest,
            pd.Index(node_table["anchor_id"].astype(str)),
            toy_config(tmp_path),
            rows_per_shard=100,
        )


def test_row_sum_validation_reports_exceedance_counts() -> None:
    node_table, source, target = mapped_toy_arrays()
    edges = toy_edges()
    probabilities = edges["row_normalized_transition_prob"].to_numpy(dtype=np.float32)
    probabilities[0] = 0.1
    final_indices = node_table.loc[node_table["is_final_time"], "global_node_index"].to_numpy(dtype=np.int64)
    matrices = m4a.assemble_sparse_matrices(
        source,
        target,
        probabilities,
        edges["raw_edge_weight"].to_numpy(dtype=np.float32),
        edges["mass_adjusted_weight"].to_numpy(dtype=np.float32),
        len(node_table),
        final_indices,
        "float32",
        1.0,
        True,
    )

    with pytest.raises(ValueError, match="not row-stochastic"):
        m4a.row_sum_qc(
            matrices["P_forward_no_terminal_selfloops"],
            matrices["P_absorbing_terminal_selfloops"],
            node_table,
            1e-6,
        )


def test_no_downstream_output_paths_are_declared() -> None:
    forbidden = ["gpcca", "fate", "branched", "m5", "regulator"]
    output_names = [
        "P_forward_no_terminal_selfloops.npz",
        "P_absorbing_terminal_selfloops.npz",
        "W_raw_edge_weight.npz",
        "W_mass_adjusted_weight.npz",
        "global_node_table.parquet",
        "m4a_edge_shard_manifest.csv",
        "m4a_assembly_report.md",
        "m4a_assembly_qc_summary.csv",
        "m4a_transition_object_schema.json",
    ]

    assert all(value is True for value in m4a.NO_DOWNSTREAM_FLAGS.values())
    assert not any(token in name.lower() for name in output_names for token in forbidden)
