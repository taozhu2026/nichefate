import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m4c_v2_03_fate_benchmark.py"
SPEC = importlib.util.spec_from_file_location("m4c_v2_benchmark", SCRIPT_PATH)
m4c_v2_benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m4c_v2_benchmark
SPEC.loader.exec_module(m4c_v2_benchmark)


def toy_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "raw_terminal_macrostate": [0, 1, 2],
            "raw_terminal_macrostate_label": ["e0", "e1", "e2"],
            "refined_endpoint_id": ["r0", "r1", "r1"],
            "refined_endpoint_label": ["R0", "R1", "R1"],
            "confidence_tier_after_refinement": [
                "high_confidence_biological_endpoint",
                "rare_biological_endpoint",
                "merge_candidate_needs_manual_review",
            ],
        }
    )


def toy_node_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_node_index": [0, 1, 2],
            "time": ["D0", "D0", "D3"],
            "time_day": [0, 0, 3],
            "slice_id": ["s0", "s0", "s1"],
            "mouse_id": ["m0", "m0", "m1"],
            "leiden_neigh": ["n0", "n1", "n1"],
            "cell_type_l3": ["c0", "c1", "c1"],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
        }
    )


def test_output_root_rejects_protected_and_ssd_paths(tmp_path: Path) -> None:
    assert m4c_v2_benchmark.validate_output_root(tmp_path / "bench") == (tmp_path / "bench").resolve()
    with pytest.raises(ValueError):
        m4c_v2_benchmark.validate_output_root(Path("/home/zhutao/scratch/nichefate/m4c_v2/reports"))
    with pytest.raises(ValueError):
        m4c_v2_benchmark.validate_output_root(Path("/ssd/nichefate/m4c_v2_benchmark"))


def test_js_divergence_zero_for_identical_vectors() -> None:
    probs = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    js = m4c_v2_benchmark.jensen_shannon_rows(probs, probs)

    assert js.tolist() == pytest.approx([0.0, 0.0])


def test_row_pearson_reports_vector_similarity() -> None:
    left = np.array([[0.2, 0.8, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    right = np.array([[0.2, 0.8, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    corr = m4c_v2_benchmark.row_pearson(left, right)

    assert corr[0] == pytest.approx(1.0)
    assert corr[1] < 0.0


def test_endpoint_mapping_check_uses_explicit_ids_not_column_order() -> None:
    v1_payload = {
        "terminal_macrostate_ids": np.array([2, 1, 0], dtype=np.int32),
    }
    v2_payload = {
        "terminal_macrostate_ids": np.array([0, 1, 2], dtype=np.int32),
    }
    check = m4c_v2_benchmark.endpoint_mapping_check(v1_payload, v2_payload, toy_mapping())

    assert set(check["status"]) == {"PASS"}
    row0 = check.query("terminal_macrostate == 0").iloc[0]
    assert row0["column_index_v1"] == 2
    assert row0["column_index_v2"] == 0


def test_build_node_metrics_derives_agreement_and_sharpness() -> None:
    v1 = np.array([[0.6, 0.4, 0.0], [0.4, 0.6, 0.0], [0.2, 0.3, 0.5]], dtype=np.float32)
    v2 = np.array([[0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.1, 0.2, 0.7]], dtype=np.float32)
    metrics = m4c_v2_benchmark.build_node_metrics(v1, v2, toy_node_table(), toy_mapping())

    assert metrics["dominant_endpoint_agreement"].tolist() == [True, True, True]
    assert metrics["dominant_refined_endpoint_agreement"].tolist() == [True, True, True]
    assert metrics["top1_delta_v2_minus_v1"].mean() > 0.0
    assert metrics["entropy_delta_v2_minus_v1"].mean() < 0.0


def test_endpoint_mass_flags_detect_low_size_inflation() -> None:
    v1 = np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]], dtype=np.float32)
    v2 = np.array([[0.4, 0.1, 0.5], [0.4, 0.1, 0.5]], dtype=np.float32)
    raw, refined = m4c_v2_benchmark.endpoint_mass_comparison(v1, v2, toy_mapping())
    flags, refined_flags = m4c_v2_benchmark.endpoint_shift_flags(raw, refined)

    endpoint2 = flags.query("terminal_macrostate == 2").iloc[0]
    assert endpoint2["severity"] == "WARN"
    assert "low_size_endpoint_inflation" in endpoint2["flag_category"]
    assert not refined_flags.empty


def test_stratified_benchmark_reports_group_mass_shift() -> None:
    v1 = np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.2, 0.3, 0.5]], dtype=np.float32)
    v2 = np.array([[0.4, 0.1, 0.5], [0.4, 0.1, 0.5], [0.1, 0.2, 0.7]], dtype=np.float32)
    metrics = m4c_v2_benchmark.build_node_metrics(v1, v2, toy_node_table(), toy_mapping())
    by_time = m4c_v2_benchmark.stratified_benchmark(metrics, v1, v2, toy_mapping(), ["time"])

    d0 = by_time.query("time == 'D0'").iloc[0]
    assert d0["n_nodes"] == 2
    assert d0["endpoint_mass_shift_max_abs"] == pytest.approx(0.5)


def test_decision_keeps_complementary_branch_when_sharper_and_safe() -> None:
    global_frame = pd.DataFrame(
        [
            {
                "top1_delta_v2_minus_v1": 0.1,
                "entropy_delta_v2_minus_v1": -0.2,
                "v2_row_sum_max_error": 1e-7,
                "v2_nonfinite_values": 0,
                "v2_negative_values": 0,
            }
        ]
    )
    artifacts = pd.DataFrame(
        {
            "artifact": ["endpoint_collapse", "slice_artifact"],
            "status": ["PASS", "PASS"],
        }
    )
    decision, reasons = m4c_v2_benchmark.choose_decision(global_frame, artifacts)

    assert decision == "keep_v1_and_v2_as_complementary_p_fate_branch"
    assert reasons
