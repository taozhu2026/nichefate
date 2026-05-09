import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m4a_v2_03_full_qc_and_benchmark.py"
SPEC = importlib.util.spec_from_file_location("m4a_v2_benchmark", SCRIPT_PATH)
m4a_v2_benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m4a_v2_benchmark
SPEC.loader.exec_module(m4a_v2_benchmark)


def toy_node_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_node_index": [0, 1, 2, 3],
            "anchor_id": ["s0", "s1", "t0", "t1"],
            "time": ["D0", "D0", "D3", "D35"],
            "time_day": [0, 0, 3, 35],
            "is_final_time": [False, False, False, True],
        }
    )


def toy_forward() -> sp.csr_matrix:
    return sp.coo_matrix(
        (
            np.array([0.25, 0.75, 1.0], dtype=np.float32),
            (np.array([0, 0, 1]), np.array([2, 3, 2])),
        ),
        shape=(4, 4),
    ).tocsr()


def test_output_root_rejects_protected_and_ssd_paths() -> None:
    m4a_v2_benchmark.validate_output_root(
        Path("/home/zhutao/scratch/nichefate/m4a_v2_benchmark_test")
    )
    with pytest.raises(ValueError):
        m4a_v2_benchmark.validate_output_root(
            Path("/home/zhutao/scratch/nichefate/m4a_v2/reports")
        )
    with pytest.raises(ValueError):
        m4a_v2_benchmark.validate_output_root(Path("/ssd/nichefate/m4a_v2_benchmark"))


def test_sparse_row_activity_and_distribution_summary() -> None:
    node = toy_node_table()
    final_mask = node["is_final_time"].to_numpy(dtype=bool)
    matrix = toy_forward()

    activity = m4a_v2_benchmark.row_activity(matrix, final_mask)
    dist = m4a_v2_benchmark.matrix_distribution(matrix)
    qc = m4a_v2_benchmark.row_sum_qc(matrix, final_mask, "forward")

    assert activity["source_row_count"] == 2
    assert activity["final_time_zero_outgoing_rows"] == 1
    assert activity["non_final_zero_outgoing_rows"] == 1
    assert dist["top1_per_row_mean"] == pytest.approx(0.875)
    assert qc["row_sum_max_error"] == pytest.approx(1.0)
    assert qc["rows_exceeding_tolerance"] == 1


def test_time_pair_coverage_reports_source_and_target_rows() -> None:
    coverage = m4a_v2_benchmark.time_pair_coverage(
        toy_forward(),
        toy_node_table(),
        "toy",
        chunk_rows=2,
    )
    d0_to_d3 = coverage.query("time_pair == 'D0_to_D3'").iloc[0]
    d0_to_d35 = coverage.query("time_pair == 'D0_to_D35'").iloc[0]

    assert d0_to_d3["source_rows"] == 2
    assert d0_to_d3["target_rows"] == 1
    assert d0_to_d3["edge_count"] == 2
    assert d0_to_d35["source_rows"] == 1
    assert d0_to_d35["target_rows"] == 1
    assert d0_to_d35["edge_count"] == 1


def test_by_time_pair_comparison_derives_deltas() -> None:
    v1 = pd.DataFrame(
        {
            "version": ["v1"],
            "time_pair": ["D0_to_D3"],
            "source_time": ["D0"],
            "target_time": ["D3"],
            "source_rows": [2],
            "target_rows": [2],
            "edge_count": [6],
            "probability_mass": [2.0],
            "edge_density_within_observed_rows": [1.5],
        }
    )
    v2 = v1.copy()
    v2["version"] = "v2"
    v2["edge_count"] = 4
    comparison = m4a_v2_benchmark.build_by_time_pair_comparison(v1, v2)

    assert comparison.loc[0, "edge_count_delta_v2_minus_v1"] == -2
    assert bool(comparison.loc[0, "source_coverage_preserved"])
    assert bool(comparison.loc[0, "v2_sparser"])


def test_decision_logic_recommends_m4c_planning_when_checks_pass() -> None:
    full_qc = {"full_qc_status": "PASS"}
    global_comparison = pd.DataFrame(
        [
            {
                "source_coverage_preserved": True,
                "non_final_zero_outgoing_rows_v2": 0,
            }
        ]
    )
    readiness = pd.DataFrame({"status": ["PASS", "PASS"]})

    decision, reasons = m4a_v2_benchmark.choose_decision(
        full_qc,
        global_comparison,
        readiness,
    )

    assert decision == "proceed_to_m4c_v2_planning"
    assert reasons


def test_decision_logic_blocks_structural_artifacts() -> None:
    full_qc = {"full_qc_status": "PASS"}
    global_comparison = pd.DataFrame(
        [
            {
                "source_coverage_preserved": True,
                "non_final_zero_outgoing_rows_v2": 3,
            }
        ]
    )
    readiness = pd.DataFrame({"status": ["PASS"]})

    decision, reasons = m4a_v2_benchmark.choose_decision(
        full_qc,
        global_comparison,
        readiness,
    )

    assert decision == "revise_m4a_v2_assembly"
    assert reasons
