import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m3_12_validate_ann_backend_on_full_shard.py"
SPEC = importlib.util.spec_from_file_location("m3_ann_full_shard_validation", SCRIPT_PATH)
m3_full = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(m3_full)


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


def toy_edges(rows: list[tuple[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["source_anchor_id", "target_anchor_id"])
    frame["raw_edge_weight"] = 1.0
    frame["mass_adjusted_weight"] = 1.0
    frame["row_normalized_transition_prob"] = (
        frame.groupby("source_anchor_id")["target_anchor_id"].transform(lambda s: 1.0 / len(s))
    )
    frame["target_slice_id"] = frame["target_anchor_id"].str.slice(0, 2)
    frame["target_mouse_id"] = frame["target_anchor_id"].str.slice(0, 2)
    return frame


def test_exact_reference_path_is_required(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        m3_full.ensure_exact_reference(tmp_path / "missing.parquet")


def test_selected_shard_must_be_exactly_one() -> None:
    shards = pd.DataFrame(
        [
            {
                "source_time": "D21",
                "target_time": "D35",
                "source_slice_id": "s0",
                "source_slice_file": "s0.m0.h5ad",
                "source_rows": 1,
                "target_time_rows": 2,
                "candidate_k": 30,
                "expected_edge_rows": 30,
            },
            {
                "source_time": "D21",
                "target_time": "D35",
                "source_slice_id": "s0",
                "source_slice_file": "s0.m0.h5ad",
                "source_rows": 1,
                "target_time_rows": 2,
                "candidate_k": 30,
                "expected_edge_rows": 30,
            },
        ]
    )

    with pytest.raises(ValueError, match="exactly one"):
        m3_full.select_validation_shard(shards, "D21", "D35", "s0")


def test_production_output_directories_are_rejected(tmp_path: Path) -> None:
    config = toy_config(tmp_path)

    with pytest.raises(ValueError, match="production M3"):
        m3_full.ensure_validation_output_dir(tmp_path / "m3" / "by_pair" / "x", config)
    with pytest.raises(ValueError, match="production M3"):
        m3_full.ensure_validation_output_dir(tmp_path / "m3" / "timepair_pilot_D21_to_D35" / "x", config)


def test_full_source_slice_mode_does_not_downsample() -> None:
    frame = pd.DataFrame({"anchor_index": range(10)})

    full = m3_full.full_source_slice(frame)

    assert len(full) == 10
    pd.testing.assert_frame_equal(full, frame)


def test_candidate_overlap_metrics_work_on_toy_examples() -> None:
    exact = toy_edges(
        [
            ("s0", "t0"),
            ("s0", "t1"),
            ("s0", "t2"),
            ("s1", "t3"),
            ("s1", "t4"),
            ("s1", "t5"),
        ]
    )
    ann = toy_edges(
        [
            ("s0", "t0"),
            ("s0", "t2"),
            ("s0", "t9"),
            ("s1", "t3"),
            ("s1", "t8"),
            ("s1", "t7"),
        ]
    )

    overlap = m3_full.compare_candidate_edges(exact, ann, candidate_k=3)

    assert overlap["recall_at_k"].tolist() == [2 / 3, 1 / 3]
    assert overlap["jaccard_overlap"].tolist() == [0.5, 0.2]
    assert overlap["top1_agreement"].tolist() == [True, True]


def test_probability_drift_metrics_work_on_toy_examples(tmp_path: Path) -> None:
    exact = toy_edges([("s0", "t0"), ("s0", "t1")])
    ann = toy_edges([("s0", "t0"), ("s0", "t2")])
    ann.loc[ann["target_anchor_id"] == "t0", "row_normalized_transition_prob"] = 0.75
    ann.loc[ann["target_anchor_id"] == "t2", "row_normalized_transition_prob"] = 0.25
    overlap = m3_full.compare_candidate_edges(exact, ann, candidate_k=2)
    drift = m3_full.M3_11.drift_metrics(exact, ann)
    row_diag = m3_full.M3_11.compare_row_diagnostics(exact, ann)
    edge_path = tmp_path / "ann.parquet"
    edge_path.write_text("placeholder", encoding="utf-8")

    metrics = m3_full.build_full_shard_metrics(
        overlap,
        drift,
        row_diag,
        {"runtime_seconds": 1.0, "max_rss_gib": 2.0},
        {"exact_reference_runtime_seconds": 2.0, "exact_reference_max_rss_gib": 4.0},
        edge_path,
        {"source_time": "D21", "target_time": "D35", "source_rows": 1, "target_rows": 2, "candidate_k": 2},
    )

    assert metrics["row_normalized_transition_prob_abs_drift_mean"].iloc[0] > 0
    assert metrics["runtime_ratio_ann_over_exact"].iloc[0] == 0.5
    assert metrics["memory_ratio_ann_over_exact"].iloc[0] == 0.5


def test_missing_exact_candidates_are_reported() -> None:
    exact = toy_edges([("s0", "t0"), ("s0", "t1")])
    ann = toy_edges([("s0", "t0"), ("s0", "t1"), ("s_missing", "t2"), ("s_missing", "t3")])

    overlap = m3_full.compare_candidate_edges(exact, ann, candidate_k=2)

    assert int((~overlap["represented_in_exact"].astype(bool)).sum()) == 1
    assert int((~overlap["represented_in_ann"].astype(bool)).sum()) == 0


def test_figure_failure_is_warning_only(tmp_path: Path) -> None:
    overlap = pd.DataFrame(
        {
            "represented_in_both": [True],
            "recall_at_k": [1.0],
            "jaccard_overlap": [1.0],
        }
    )
    drift = pd.DataFrame({"row_normalized_transition_prob_abs_drift": [0.0]})
    row_diag = pd.DataFrame({"row_entropy_delta": [0.0]})
    metrics = pd.DataFrame(
        [
            {
                "exact_reference_runtime_seconds": 1.0,
                "ann_runtime_seconds": 1.0,
                "exact_reference_max_rss_gib": 1.0,
                "ann_max_rss_gib": 1.0,
                "target_slice_id_entropy_delta_mean": 0.0,
                "target_mouse_id_entropy_delta_mean": 0.0,
                "top_target_slice_id_fraction_delta_mean": 0.0,
                "top_target_mouse_id_fraction_delta_mean": 0.0,
            }
        ]
    )

    warnings = m3_full.generate_figures(
        tmp_path / "figures",
        overlap,
        drift,
        row_diag,
        metrics,
        force_failure=True,
    )

    assert warnings


def test_no_global_or_downstream_output_paths_are_produced(tmp_path: Path) -> None:
    paths = m3_full.output_paths(tmp_path, "D21", "D35", "slice")
    text = "\n".join(path.name.lower() for path in paths.values())

    for token in ["markov", "gpcca", "fate", "branched", "nicheflow", "m5", "regulator"]:
        assert token not in text


def test_no_dataset_specific_hard_coding_in_m3_core() -> None:
    paths = [
        PROJECT_ROOT / "src" / "nichefate" / "transition.py",
        PROJECT_ROOT / "scripts" / "m3_12_validate_ann_backend_on_full_shard.py",
        PROJECT_ROOT / "configs" / "m3_transition_kernel.yaml",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for token in ["Moffitt", "Cadinu", "DSS", "colon", "Day35", "Sample_type"]:
        assert token not in text
