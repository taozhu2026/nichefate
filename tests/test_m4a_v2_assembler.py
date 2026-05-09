import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m4a_v2_01_assemble_sparse_matrices.py"
CONFIG_PATH = PROJECT_ROOT / "configs" / "m4a_v2_assembly.yaml"
SPEC = importlib.util.spec_from_file_location("m4a_v2_assembler", SCRIPT_PATH)
m4a_v2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m4a_v2
SPEC.loader.exec_module(m4a_v2)


def toy_config(tmp_path: Path) -> dict:
    output_root = tmp_path / "m4a_v2"
    return {
        "paths": {
            "output_root": str(output_root),
            "reports_dir": str(output_root / "reports"),
            "tmp_dir": str(output_root / "tmp"),
            "transition_objects_dir": str(output_root / "transition_objects"),
            "node_table_dir": str(output_root / "node_table"),
        },
        "inputs": {
            "m3_v2_edge_root": str(tmp_path / "m3_v2" / "full_by_shard"),
            "m3_v2_qc_summary": str(tmp_path / "m3_v2" / "reports" / "qc.csv"),
            "m3_v2_benchmark_summary": str(
                tmp_path / "m3_v2_benchmark" / "summary.json"
            ),
            "m4a_v1_node_table": str(
                tmp_path / "m4a" / "node_table" / "global_node_table.parquet"
            ),
        },
        "assembly": {
            "source_id_column": "source_anchor_id",
            "target_id_column": "target_anchor_id",
            "probability_column": "v2_row_normalized_transition_prob",
            "weight_column": "v2_unnormalized_weight",
            "terminal_time_policy": "final_time_no_outgoing",
            "write_absorbing_terminal_variant": True,
            "final_time_label": "D35",
            "final_time_self_loop_weight": 1.0,
            "dtype": "float32",
            "index_dtype": "int64",
        },
        "expected": {
            "shards": 52,
            "nodes": 4,
            "final_time_nodes": 2,
            "source_rows": 2,
            "retained_v2_edges": 4,
            "absorbing_nnz": 6,
        },
        "validation": {
            "row_sum_tolerance": 1e-6,
            "batch_rows": 1000,
            "full_duplicate_check": True,
            "fail_on_nan": True,
            "fail_on_negative_probability": True,
            "fail_on_negative_weight": True,
        },
        "protected_roots": [
            str(tmp_path / "m3"),
            str(tmp_path / "m3_v2"),
            str(tmp_path / "m4a"),
            str(tmp_path / "m4b"),
            str(tmp_path / "m4c"),
        ],
        "forbidden_downstream_roots": [str(output_root / "m4c_v2")],
    }


def toy_forward_matrix() -> sp.csr_matrix:
    return m4a_v2.construct_sparse_matrix(
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([2, 3, 2, 3], dtype=np.int64),
        np.array([0.25, 0.75, 0.4, 0.6], dtype=np.float32),
        4,
        "float32",
    )


def test_config_parses_expected_m4a_v2_values() -> None:
    config = m4a_v2.load_yaml_config(CONFIG_PATH)
    paths = m4a_v2.validate_config(config)

    assert paths["output_root"] == Path("/home/zhutao/scratch/nichefate/m4a_v2")
    assert config["expected"]["nodes"] == 1_439_542
    assert config["expected"]["final_time_nodes"] == 90_960
    assert config["expected"]["source_rows"] == 1_348_582
    assert config["expected"]["retained_v2_edges"] == 13_485_820
    assert config["expected"]["absorbing_nnz"] == 13_576_780
    assert (
        config["assembly"]["probability_column"]
        == "v2_row_normalized_transition_prob"
    )
    assert config["assembly"]["weight_column"] == "v2_unnormalized_weight"


def test_output_paths_reject_protected_roots(tmp_path: Path) -> None:
    config = toy_config(tmp_path)
    output_root = tmp_path / "m3" / "m4a_v2"
    config["paths"]["output_root"] = str(output_root)
    config["paths"]["reports_dir"] = str(output_root / "reports")
    config["paths"]["tmp_dir"] = str(output_root / "tmp")
    config["paths"]["transition_objects_dir"] = str(output_root / "transition_objects")
    config["paths"]["node_table_dir"] = str(output_root / "node_table")

    with pytest.raises(ValueError, match="overlaps protected root"):
        m4a_v2.validate_config(config)


def test_output_paths_reject_ssd(tmp_path: Path) -> None:
    config = toy_config(tmp_path)
    config["paths"]["output_root"] = "/ssd/nichefate/m4a_v2"
    config["paths"]["reports_dir"] = "/ssd/nichefate/m4a_v2/reports"
    config["paths"]["tmp_dir"] = "/ssd/nichefate/m4a_v2/tmp"
    config["paths"]["transition_objects_dir"] = "/ssd/nichefate/m4a_v2/transition_objects"
    config["paths"]["node_table_dir"] = "/ssd/nichefate/m4a_v2/node_table"

    with pytest.raises(ValueError, match="Refusing /ssd path"):
        m4a_v2.validate_config(config)


def test_existing_production_outputs_require_overwrite(tmp_path: Path) -> None:
    config = toy_config(tmp_path)
    paths = m4a_v2.validate_config(config)
    p_forward = m4a_v2.planned_production_outputs(paths)["p_forward"]
    p_forward.parent.mkdir(parents=True)
    p_forward.touch()

    with pytest.raises(FileExistsError, match="--overwrite"):
        m4a_v2.validate_no_existing_production_outputs(paths, overwrite=False, resume=True)

    m4a_v2.validate_no_existing_production_outputs(paths, overwrite=True, resume=False)


def test_smoke_mode_requires_production_max_shards() -> None:
    assert m4a_v2.is_smoke_mode(Namespace(dry_run=False, max_shards=1))
    assert not m4a_v2.is_smoke_mode(Namespace(dry_run=True, max_shards=1))
    assert not m4a_v2.is_smoke_mode(Namespace(dry_run=False, max_shards=None))


def test_m4a_v2_02_output_names_exclude_smoke_report(tmp_path: Path) -> None:
    paths = m4a_v2.validate_config(toy_config(tmp_path))
    outputs = m4a_v2.planned_production_outputs(paths)

    assert outputs["assembly_report"].name == "m4a_v2_02_full_assembly_report.md"
    assert outputs["qc_summary"].name == "m4a_v2_02_qc_summary.csv"
    assert outputs["output_inventory"].name == "m4a_v2_02_output_inventory.csv"
    assert outputs["matrix_comparison"].name == "m4a_v2_02_v1_v2_matrix_comparison.csv"
    assert outputs["next_step"].name == "m4a_v2_02_next_step_recommendation.md"
    assert m4a_v2.SMOKE_REPORT_NAME not in {path.name for path in outputs.values()}


def test_duplicate_source_target_coordinates_are_detected() -> None:
    source = np.array([0, 0, 1], dtype=np.int64)
    target = np.array([1, 1, 2], dtype=np.int64)

    assert m4a_v2.duplicate_coordinate_count(source.copy(), target, 4) == 1
    with pytest.raises(ValueError, match="Duplicate source-target matrix coordinates"):
        m4a_v2.assert_no_duplicate_coordinates(source, target, 4)


def test_toy_sparse_coordinate_construction() -> None:
    matrix = toy_forward_matrix()

    assert matrix.shape == (4, 4)
    assert matrix.nnz == 4
    assert np.asarray(matrix.sum(axis=1)).ravel().tolist() == pytest.approx(
        [1.0, 1.0, 0.0, 0.0]
    )
    assert matrix[0, 2] == pytest.approx(0.25)
    assert matrix[1, 3] == pytest.approx(0.6)


def test_final_self_loops_are_added_and_structural_conflicts_fail() -> None:
    forward = toy_forward_matrix()
    absorbing = m4a_v2.add_final_self_loops(forward, np.array([2, 3], dtype=np.int64))

    assert absorbing.nnz == 6
    assert absorbing[2, 2] == pytest.approx(1.0)
    assert absorbing[3, 3] == pytest.approx(1.0)

    explicit_zero_self_loop = sp.csr_matrix(
        (
            np.array([0.0], dtype=np.float32),
            np.array([2], dtype=np.int32),
            np.array([0, 0, 0, 1, 1], dtype=np.int32),
        ),
        shape=(4, 4),
    )
    assert (
        m4a_v2.stored_diagonal_coordinate_count(
            explicit_zero_self_loop, np.array([2], dtype=np.int64)
        )
        == 1
    )
    with pytest.raises(ValueError, match="Final-time self-loop conflict count"):
        m4a_v2.add_final_self_loops(
            explicit_zero_self_loop, np.array([2], dtype=np.int64)
        )


def test_row_sum_qc_for_toy_forward_and_absorbing_matrices() -> None:
    final_mask = np.array([False, False, True, True])
    forward = toy_forward_matrix()
    absorbing = m4a_v2.add_final_self_loops(forward, np.array([2, 3], dtype=np.int64))

    qc = m4a_v2.row_sum_qc(forward, absorbing, final_mask, 1e-6)

    assert qc["forward_nonfinal_rows_exceeding_tolerance"] == 0
    assert qc["forward_final_rows_exceeding_tolerance"] == 0
    assert qc["absorbing_rows_exceeding_tolerance"] == 0


def test_row_sum_qc_rejects_bad_forward_rows() -> None:
    final_mask = np.array([False, False, True, True])
    bad_forward = m4a_v2.construct_sparse_matrix(
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([2, 3, 2, 3], dtype=np.int64),
        np.array([0.2, 0.7, 0.4, 0.6], dtype=np.float32),
        4,
        "float32",
    )
    absorbing = m4a_v2.add_final_self_loops(
        bad_forward, np.array([2, 3], dtype=np.int64)
    )

    with pytest.raises(ValueError, match="Forward non-final row sums exceed tolerance"):
        m4a_v2.row_sum_qc(bad_forward, absorbing, final_mask, 1e-6)
