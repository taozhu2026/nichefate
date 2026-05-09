import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m3_v2_08_full_qc_and_benchmark.py"
SPEC = importlib.util.spec_from_file_location("m3_v2_benchmark", SCRIPT_PATH)
m3_v2_benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m3_v2_benchmark
SPEC.loader.exec_module(m3_v2_benchmark)


def toy_annotations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_node_index": [0, 1, 2, 3],
            "anchor_id": ["s0", "s1", "t0", "t1"],
            "slice_id": ["a", "a", "b", "b"],
            "anchor_cell_id": ["cs0", "cs1", "ct0", "ct1"],
            "leiden_neigh": ["n0", "n1", "n0", "n1"],
            "cell_type_l3": ["c0", "c1", "c0", "c1"],
            "refined_endpoint_id": ["e0", "e1", "e0", "e1"],
        }
    )


def toy_edges(prob_col: str = "prob") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_anchor_id": ["s0", "s0", "s1", "s1"],
            "target_anchor_id": ["t0", "t1", "t0", "t1"],
            "source_slice_id": ["a", "a", "a", "a"],
            "target_slice_id": ["b", "b", "b", "b"],
            "source_mouse_id": ["m0", "m0", "m0", "m0"],
            "target_mouse_id": ["m1", "m1", "m2", "m2"],
            prob_col: [0.8, 0.2, 0.3, 0.7],
        }
    )


def test_detect_probability_column_requires_unambiguous_v1_schema() -> None:
    assert (
        m3_v2_benchmark.detect_v1_probability_column(["row_normalized_transition_prob"])
        == "row_normalized_transition_prob"
    )
    with pytest.raises(ValueError):
        m3_v2_benchmark.detect_v1_probability_column(["foo"])
    with pytest.raises(ValueError):
        m3_v2_benchmark.detect_v1_probability_column(
            ["row_normalized_transition_prob", "other_transition_probability"]
        )
    assert m3_v2_benchmark.detect_v1_probability_column(["p"], configured="p") == "p"


def test_probability_values_fail_on_nonfinite_or_negative() -> None:
    good = pd.DataFrame({"p": [0.2, 0.8]})
    assert m3_v2_benchmark.validate_probability_values(good, "p", "toy")["toy_probability_negative_count"] == 0
    with pytest.raises(ValueError):
        m3_v2_benchmark.validate_probability_values(pd.DataFrame({"p": [0.2, -1.0]}), "p", "toy")
    with pytest.raises(ValueError):
        m3_v2_benchmark.validate_probability_values(pd.DataFrame({"p": [0.2, np.inf]}), "p", "toy")


def test_annotation_join_prefers_global_then_anchor() -> None:
    ann = toy_annotations()
    join = m3_v2_benchmark.choose_annotation_join(
        {"source_global_node_index", "target_global_node_index"},
        ann,
    )
    assert join.name == "global_node_index"
    join = m3_v2_benchmark.choose_annotation_join({"source_anchor_id", "target_anchor_id"}, ann)
    assert join.name == "anchor_id"


def test_method_metrics_reports_missing_rates_and_plausibility() -> None:
    ann = toy_annotations()
    join = m3_v2_benchmark.choose_annotation_join({"source_anchor_id", "target_anchor_id"}, ann)
    lookup = m3_v2_benchmark.annotation_lookup(ann, join)
    metrics, dists = m3_v2_benchmark.method_metrics(toy_edges(), "prob", lookup, join, "v1")

    assert metrics["v1_source_annotation_missing_rate"] == pytest.approx(0.0)
    assert metrics["v1_target_annotation_missing_rate"] == pytest.approx(0.0)
    assert metrics["v1_refined_endpoint_plausibility"] == pytest.approx(1.0)
    assert "leiden" in dists


def test_delta_and_decision_logic_uses_per_time_pair_failures() -> None:
    good = pd.DataFrame(
        [
            {
                "time_pair": "global",
                "endpoint_delta_v2_minus_v1": 0.0,
                "leiden_delta_v2_minus_v1": 0.0,
                "entropy_delta_v2_minus_v1": -0.5,
                "top1_delta_v2_minus_v1": 0.2,
                "collapse_delta_v2_minus_v1": 0.0,
                "diversity_delta_v2_minus_v1": 0.0,
                "v1_source_annotation_missing_rate": 0.0,
                "v2_source_annotation_missing_rate": 0.0,
                "v1_target_annotation_missing_rate": 0.0,
                "v2_target_annotation_missing_rate": 0.0,
            }
        ]
    )
    bad_pair = good.copy()
    bad_pair["time_pair"] = "D0_to_D3"
    bad_pair["collapse_delta_v2_minus_v1"] = 0.5

    decision, reasons = m3_v2_benchmark.choose_decision(good, bad_pair)

    assert decision == "revise_v2_and_repeat_full_or_partial"
    assert reasons


def test_aggregate_outputs_derives_distribution_dependent_deltas() -> None:
    shard_rows = pd.DataFrame(
        [
            {
                "time_pair": "D0_to_D3",
                "source_count_v1": 2,
                "v1_candidate_edges": 4,
                "v2_retained_edges": 4,
                "annotation_join_key_used": "anchor_id",
                "v1_source_annotation_missing_rate": 0.0,
                "v2_source_annotation_missing_rate": 0.0,
                "v1_target_annotation_missing_rate": 0.0,
                "v2_target_annotation_missing_rate": 0.0,
                "mean_js_divergence_from_v1": 0.1,
                "v1_entropy_mean": 0.8,
                "v2_entropy_mean": 0.5,
                "v1_top1_probability_mean": 0.6,
                "v2_top1_probability_mean": 0.8,
                "v1_leiden_match_count": 1,
                "v1_leiden_valid_count": 2,
                "v2_leiden_match_count": 2,
                "v2_leiden_valid_count": 2,
                "v1_fine_match_count": 1,
                "v1_fine_valid_count": 2,
                "v2_fine_match_count": 1,
                "v2_fine_valid_count": 2,
                "v1_endpoint_match_count": 1,
                "v1_endpoint_valid_count": 2,
                "v2_endpoint_match_count": 2,
                "v2_endpoint_valid_count": 2,
            }
        ]
    )
    dist_acc = {
        ("D0_to_D3", "v1", "leiden"): pd.Series({"n0": 1.0, "n1": 1.0}),
        ("D0_to_D3", "v2", "leiden"): pd.Series({"n0": 1.7, "n1": 0.3}),
        ("D0_to_D3", "v1", "slice"): pd.Series({"s0": 1.0, "s1": 1.0}),
        ("D0_to_D3", "v2", "slice"): pd.Series({"s0": 1.4, "s1": 0.6}),
        ("D0_to_D3", "v1", "mouse"): pd.Series({"m0": 1.0, "m1": 1.0}),
        ("D0_to_D3", "v2", "mouse"): pd.Series({"m0": 1.2, "m1": 0.8}),
    }

    time_frame, global_frame = m3_v2_benchmark.aggregate_outputs(shard_rows, dist_acc)

    assert "collapse_delta_v2_minus_v1" in time_frame.columns
    assert time_frame.loc[0, "entropy_delta_v2_minus_v1"] == pytest.approx(-0.3)
    assert global_frame.loc[0, "v2_retained_edges"] == 4


def test_output_root_rejects_protected_and_ssd_paths() -> None:
    m3_v2_benchmark.validate_output_root(Path("/home/zhutao/scratch/nichefate/m3_v2_benchmark_test"))
    with pytest.raises(ValueError):
        m3_v2_benchmark.validate_output_root(Path("/home/zhutao/scratch/nichefate/m3/full_by_shard"))
    with pytest.raises(ValueError):
        m3_v2_benchmark.validate_output_root(Path("/ssd/nichefate/m3_v2_benchmark"))
