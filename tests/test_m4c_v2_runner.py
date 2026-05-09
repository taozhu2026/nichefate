import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m4c_v2_01_run_fate_propagation.py"
CONFIG_PATH = PROJECT_ROOT / "configs" / "m4c_v2_fate_propagation.yaml"
SPEC = importlib.util.spec_from_file_location("m4c_v2_runner", SCRIPT_PATH)
m4c_v2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m4c_v2
SPEC.loader.exec_module(m4c_v2)


def toy_config(tmp_path: Path) -> dict:
    root = tmp_path / "m4c_v2"
    return {
        "paths": {
            "output_root": str(root),
            "reports_dir": str(root / "reports"),
            "tmp_dir": str(root / "tmp"),
            "fate_dir": str(root / "fate_probabilities"),
        },
        "inputs": {
            "p_absorbing": str(tmp_path / "m4a_v2" / "p_absorbing.npz"),
            "p_forward": str(tmp_path / "m4a_v2" / "p_forward.npz"),
            "m4a_v2_node_table": str(tmp_path / "m4a_v2" / "nodes.parquet"),
            "m4a_v2_qc_summary": str(tmp_path / "m4a_v2" / "qc.csv"),
            "endpoint_mapping": str(tmp_path / "m4e" / "mapping.csv"),
            "endpoint_node_annotation": str(tmp_path / "m4e" / "endpoint_nodes.parquet"),
            "neighborhood_annotation": str(tmp_path / "m4e" / "neighborhood.parquet"),
            "m4c_v1_fate_matrix": str(tmp_path / "m4c" / "fate.npz"),
            "m4c_v1_node_summary": str(tmp_path / "m4c" / "nodes.parquet"),
            "m3_v2_benchmark_summary": str(tmp_path / "m3_v2_benchmark" / "summary.json"),
        },
        "fate": {
            "method": "time_layered_backward_propagation",
            "probability_dtype": "float32",
            "endpoint_macrostate_column": "candidate_endpoint",
            "raw_terminal_column": "raw_terminal_macrostate",
            "preserve_raw_terminal_columns": True,
            "endpoint_count": 2,
            "final_time_label": "D2",
            "directionality_evidence_source": "toy",
            "barcode_compatible_contract": False,
        },
        "validation": {
            "expected_global_nodes": 5,
            "expected_terminal_nodes": 2,
            "expected_source_rows": 3,
            "expected_endpoint_count": 2,
            "expected_forward_nnz": 5,
            "expected_absorbing_nnz": 7,
            "row_sum_tolerance": 1e-6,
            "fail_on_nan": True,
            "fail_on_negative_probability": True,
        },
        "smoke": {"default_max_sources": 2},
        "protected_roots": [
            str(tmp_path / "m3"),
            str(tmp_path / "m3_v2"),
            str(tmp_path / "m4a"),
            str(tmp_path / "m4a_v2"),
            str(tmp_path / "m4b"),
            str(tmp_path / "m4c"),
        ],
        "forbidden_downstream_roots": [str(root / "gpcca")],
    }


def toy_node_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_node_index": [0, 1, 2, 3, 4],
            "anchor_id": ["n0", "n1", "n2", "n3", "n4"],
            "time": ["D0", "D0", "D1", "D2", "D2"],
            "time_day": [0.0, 0.0, 1.0, 2.0, 2.0],
            "is_final_time": [False, False, False, True, True],
        }
    )


def toy_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "raw_terminal_macrostate": [0, 1],
            "raw_terminal_macrostate_label": ["terminal_macrostate_00", "terminal_macrostate_01"],
            "refined_endpoint_id": ["same_refined", "same_refined"],
            "refined_endpoint_label": ["Same refined endpoint", "Same refined endpoint"],
            "confidence_tier_after_refinement": ["high", "merge_candidate_needs_manual_review"],
        }
    )


def toy_assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_node_index": [3, 4],
            "anchor_id": ["n3", "n4"],
            "candidate_endpoint": [0, 1],
        }
    )


def toy_forward() -> sp.csr_matrix:
    rows = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    cols = np.array([2, 2, 2, 3, 4], dtype=np.int64)
    data = np.array([0.5, 0.5, 1.0, 0.25, 0.75], dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(5, 5))


def test_config_parses_expected_m4c_v2_values() -> None:
    config = m4c_v2.load_config(CONFIG_PATH)
    paths = m4c_v2.validate_config(config)

    assert paths["output_root"] == Path("/home/zhutao/scratch/nichefate/m4c_v2")
    assert config["validation"]["expected_global_nodes"] == 1_439_542
    assert config["validation"]["expected_terminal_nodes"] == 90_960
    assert config["validation"]["expected_endpoint_count"] == 12
    assert config["fate"]["preserve_raw_terminal_columns"] is True


def test_output_paths_reject_protected_roots_and_ssd(tmp_path: Path) -> None:
    config = toy_config(tmp_path)
    safe_paths = m4c_v2.validate_config(config)
    assert safe_paths["output_root"] == (tmp_path / "m4c_v2").resolve()

    config["paths"]["output_root"] = str(tmp_path / "m4a_v2" / "bad")
    config["paths"]["reports_dir"] = str(tmp_path / "m4a_v2" / "bad" / "reports")
    config["paths"]["tmp_dir"] = str(tmp_path / "m4a_v2" / "bad" / "tmp")
    config["paths"]["fate_dir"] = str(tmp_path / "m4a_v2" / "bad" / "fate")
    with pytest.raises(ValueError, match="overlaps protected root"):
        m4c_v2.validate_config(config)

    config = toy_config(tmp_path)
    config["paths"]["output_root"] = "/ssd/nichefate/m4c_v2"
    config["paths"]["reports_dir"] = "/ssd/nichefate/m4c_v2/reports"
    config["paths"]["tmp_dir"] = "/ssd/nichefate/m4c_v2/tmp"
    config["paths"]["fate_dir"] = "/ssd/nichefate/m4c_v2/fate"
    with pytest.raises(ValueError, match="Refusing /ssd"):
        m4c_v2.validate_config(config)


def test_existing_production_outputs_require_overwrite(tmp_path: Path) -> None:
    paths = m4c_v2.validate_config(toy_config(tmp_path))
    outputs = m4c_v2.production_outputs(paths)
    outputs["fate_matrix"].parent.mkdir(parents=True)
    outputs["fate_matrix"].touch()

    with pytest.raises(FileExistsError, match="--overwrite"):
        m4c_v2.validate_no_existing_production_outputs(outputs, overwrite=False)

    m4c_v2.validate_no_existing_production_outputs(outputs, overwrite=True)


def test_endpoint_mapping_preserves_raw_columns_despite_refined_merge() -> None:
    mapping = m4c_v2.validate_endpoint_mapping(toy_mapping(), expected_endpoint_count=2)

    assert mapping["raw_terminal_macrostate"].tolist() == [0, 1]
    assert mapping["refined_endpoint_id"].nunique() == 1


def test_toy_backward_propagation_and_qc() -> None:
    node_table, final_mask = m4c_v2.validate_node_table(toy_node_table(), 5, 2, "D2")
    mapping = m4c_v2.validate_endpoint_mapping(toy_mapping(), 2)
    assignments, _ = m4c_v2.validate_endpoint_assignments(
        toy_assignments(),
        node_table,
        final_mask,
        mapping,
        "candidate_endpoint",
    )

    fate, steps = m4c_v2.compute_fate_probabilities(
        toy_forward(),
        node_table,
        assignments,
        2,
        np.dtype("float32"),
    )
    qc = m4c_v2.validate_fate_matrix(fate, final_mask, 1e-6)
    metrics = m4c_v2.fate_metrics(fate, mapping)

    assert fate.shape == (5, 2)
    assert fate[0].tolist() == pytest.approx([0.25, 0.75])
    assert fate[1].tolist() == pytest.approx([0.25, 0.75])
    assert fate[2].tolist() == pytest.approx([0.25, 0.75])
    assert fate[3].tolist() == pytest.approx([1.0, 0.0])
    assert [step["source_time"] for step in steps] == ["D0", "D1"]
    assert qc["row_sum_max_error"] == pytest.approx(0.0)
    assert metrics["dominant_endpoint"].tolist() == [1, 1, 1, 0, 1]


def test_smoke_subset_uses_latest_nonfinal_sources_and_final_targets() -> None:
    node_table, final_mask = m4c_v2.validate_node_table(toy_node_table(), 5, 2, "D2")
    mapping = m4c_v2.validate_endpoint_mapping(toy_mapping(), 2)
    assignments, _ = m4c_v2.validate_endpoint_assignments(
        toy_assignments(),
        node_table,
        final_mask,
        mapping,
        "candidate_endpoint",
    )
    fate, subset_indices, subset_final_mask, summary = m4c_v2.compute_smoke_fate(
        toy_forward(),
        node_table,
        assignments,
        mapping,
        max_sources=1,
        max_nodes=None,
        dtype=np.dtype("float32"),
    )

    assert subset_indices.tolist() == [2, 3, 4]
    assert subset_final_mask.tolist() == [False, True, True]
    assert fate[0].tolist() == pytest.approx([0.25, 0.75])
    assert summary["smoke_source_rows"] == 1
    assert summary["smoke_local_nnz"] == 2


def test_fate_qc_rejects_invalid_rows() -> None:
    bad = np.array([[0.2, 0.2], [1.0, 0.0]], dtype=np.float32)
    final_mask = np.array([False, True])

    with pytest.raises(ValueError, match="row-sum tolerance"):
        m4c_v2.validate_fate_matrix(bad, final_mask, 1e-6)


def test_output_names_are_versioned_and_under_m4c_v2(tmp_path: Path) -> None:
    paths = m4c_v2.validate_config(toy_config(tmp_path))
    outputs = m4c_v2.production_outputs(paths)

    assert outputs["fate_matrix"].name == "fate_probability_matrix_v2.npz"
    assert outputs["node_summary"].name == "node_fate_summary_v2.parquet"
    assert outputs["qc_summary"].name == "m4c_v2_02_qc_summary.csv"
    assert all(paths["output_root"] in output.parents for output in outputs.values())
