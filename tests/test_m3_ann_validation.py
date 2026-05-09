import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nichefate.transition import CandidateNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m3_11_validate_ann_backend_on_sampled_shard.py"
SPEC = importlib.util.spec_from_file_location("m3_ann_validation", SCRIPT_PATH)
m3_ann_validation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(m3_ann_validation)


def toy_config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "use_ssd": False,
            "m3_output_dir": str(tmp_path / "m3"),
        },
        "full_m3": {
            "output_root": str(tmp_path / "m3" / "by_pair"),
        },
    }


def toy_args(tmp_path: Path, sample_size: int = 5000) -> Namespace:
    return Namespace(
        source_time="D21",
        target_time="D35",
        source_slice_id="082421_D21_m2_1_slice_2",
        sample_source_anchors=sample_size,
        candidate_k=30,
        exact_backend="sklearn_exact",
        ann_backend="pynndescent",
        output_dir=tmp_path / "ann_validation",
        allow_larger_sample=False,
        allow_non_default_shard=False,
    )


def test_deterministic_source_sampling() -> None:
    frame = pd.DataFrame({"value": range(100)})

    first = m3_ann_validation.deterministic_source_sample(frame, 10, 1)
    second = m3_ann_validation.deterministic_source_sample(frame, 10, 1)
    different = m3_ann_validation.deterministic_source_sample(frame, 10, 2)

    pd.testing.assert_frame_equal(first, second)
    assert first["value"].tolist() != different["value"].tolist()
    assert first["value"].is_monotonic_increasing


def test_validation_shard_selector_returns_exactly_one() -> None:
    shards = pd.DataFrame(
        [
            {
                "source_time": "D21",
                "target_time": "D35",
                "source_slice_id": "s0",
                "source_slice_file": "s0.m0.h5ad",
                "source_rows": 5,
                "target_time_rows": 10,
                "candidate_k": 30,
                "expected_edge_rows": 150,
            }
        ]
    )

    shard = m3_ann_validation.select_validation_shard(shards, "D21", "D35", "s0")

    assert shard["source_slice_id"] == "s0"


def test_validation_shard_selector_refuses_multiple_shards() -> None:
    shards = pd.DataFrame(
        [
            {
                "source_time": "D21",
                "target_time": "D35",
                "source_slice_id": "s0",
                "source_slice_file": "s0.m0.h5ad",
                "source_rows": 5,
                "target_time_rows": 10,
                "candidate_k": 30,
                "expected_edge_rows": 150,
            },
            {
                "source_time": "D21",
                "target_time": "D35",
                "source_slice_id": "s0",
                "source_slice_file": "s0.m0.h5ad",
                "source_rows": 5,
                "target_time_rows": 10,
                "candidate_k": 30,
                "expected_edge_rows": 150,
            },
        ]
    )

    with pytest.raises(ValueError, match="exactly one"):
        m3_ann_validation.select_validation_shard(shards, "D21", "D35", "s0")


def test_refuses_sample_size_over_5000_without_override(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="5000"):
        m3_ann_validation.validate_requested_scope(toy_args(tmp_path, 5001), toy_config(tmp_path))


def test_exact_and_ann_candidate_sets_compare_on_toy_data() -> None:
    exact = CandidateNeighbors(
        indices=np.array([[1, 2, 3], [4, 5, 6]]),
        distances=np.ones((2, 3)),
        backend="sklearn_exact",
        metric="euclidean",
    )
    ann = CandidateNeighbors(
        indices=np.array([[1, 3, 7], [4, 8, 9]]),
        distances=np.ones((2, 3)),
        backend="pynndescent",
        metric="euclidean",
    )

    overlap = m3_ann_validation.compare_candidate_sets(exact, ann)

    assert overlap["recall_at_k"].tolist() == [2 / 3, 1 / 3]
    assert overlap["top1_agreement"].tolist() == [True, True]
    assert overlap["jaccard_overlap"].tolist() == [0.5, 0.2]


def test_probability_drift_metrics_work_on_toy_edge_tables() -> None:
    exact = pd.DataFrame(
        {
            "source_anchor_id": ["s0", "s0"],
            "target_anchor_id": ["t0", "t1"],
            "raw_edge_weight": [1.0, 3.0],
            "mass_adjusted_weight": [1.0, 3.0],
            "row_normalized_transition_prob": [0.25, 0.75],
            "target_slice_id": ["a", "b"],
            "target_mouse_id": ["m0", "m1"],
        }
    )
    ann = pd.DataFrame(
        {
            "source_anchor_id": ["s0", "s0"],
            "target_anchor_id": ["t0", "t2"],
            "raw_edge_weight": [2.0, 2.0],
            "mass_adjusted_weight": [2.0, 2.0],
            "row_normalized_transition_prob": [0.5, 0.5],
            "target_slice_id": ["a", "c"],
            "target_mouse_id": ["m0", "m2"],
        }
    )

    drift = m3_ann_validation.drift_metrics(exact, ann)
    row_diag = m3_ann_validation.compare_row_diagnostics(exact, ann)

    assert drift["row_normalized_transition_prob_abs_drift"].sum() == pytest.approx(1.5)
    assert "row_entropy_delta" in row_diag
    assert row_diag["top1_probability_abs_delta"].iloc[0] == pytest.approx(0.25)


def test_figure_failure_is_warning_only(tmp_path: Path) -> None:
    overlap = pd.DataFrame({"recall_at_k": [1.0], "top1_agreement": [True]})
    drift = pd.DataFrame({"row_normalized_transition_prob_abs_drift": [0.0]})
    row_diag = pd.DataFrame({"row_entropy_delta": [0.0]})
    metrics = pd.DataFrame(
        [
            {
                "top1_agreement": 1.0,
                "sklearn_exact_runtime_seconds": 1.0,
                "pynndescent_runtime_seconds": 1.0,
                "sklearn_exact_max_rss_gib": 1.0,
                "pynndescent_max_rss_gib": 1.0,
                "target_slice_id_entropy_delta_mean": 0.0,
                "target_mouse_id_entropy_delta_mean": 0.0,
                "top_target_slice_id_fraction_delta_mean": 0.0,
                "top_target_mouse_id_fraction_delta_mean": 0.0,
            }
        ]
    )

    warnings = m3_ann_validation.generate_figures(
        tmp_path / "figures",
        overlap,
        drift,
        row_diag,
        metrics,
        force_failure=True,
    )

    assert warnings


def test_validation_outputs_refuse_production_m3_directories(tmp_path: Path) -> None:
    config = toy_config(tmp_path)

    with pytest.raises(ValueError, match="production M3"):
        m3_ann_validation.ensure_validation_output_dir(tmp_path / "m3" / "by_pair" / "x", config)


def test_validation_outputs_allow_project_name_containing_fate(tmp_path: Path) -> None:
    config = toy_config(tmp_path)
    output_dir = tmp_path / "nichefate" / "m3" / "ann_validation_D21_to_D35"

    m3_ann_validation.ensure_validation_output_dir(output_dir, config)


def test_no_global_or_downstream_output_paths_are_produced(tmp_path: Path) -> None:
    paths = m3_ann_validation.output_paths(tmp_path, "D21", "D35", "slice")
    text = "\n".join(path.name.lower() for path in paths.values())

    for token in ["markov", "gpcca", "fate", "branched", "nicheflow", "m5", "regulator"]:
        assert token not in text
