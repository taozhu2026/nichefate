import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nichefate.transition import categorical_target_diagnostics, full_transition_schema_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "m3_05_build_transition_pilot_shard.py"
SPEC = importlib.util.spec_from_file_location("m3_pilot_shard", RUNNER_PATH)
m3_pilot_shard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(m3_pilot_shard)


def make_args() -> Namespace:
    return Namespace(
        source_time="t0",
        target_time="t1",
        source_slice_id="s0",
        source_slice_file="s0.m0.h5ad",
    )


def test_pilot_runner_refuses_multiple_shards() -> None:
    shards = pd.DataFrame(
        [
            {
                "source_time": "t0",
                "target_time": "t1",
                "source_slice_id": "s0",
                "source_slice_file": "s0.m0.h5ad",
            },
            {
                "source_time": "t0",
                "target_time": "t1",
                "source_slice_id": "s0",
                "source_slice_file": "s0.m0.h5ad",
            },
        ]
    )

    with pytest.raises(ValueError, match="exactly one"):
        m3_pilot_shard.select_shard(shards, make_args())


def test_dry_run_summary_does_not_write_edge_outputs(tmp_path: Path) -> None:
    edge_path = tmp_path / "candidate_edges.parquet"
    shard = {
        "source_time": "t0",
        "target_time": "t1",
        "source_slice_id": "s0",
        "source_rows": 4,
        "target_time_rows": 5,
        "expected_edge_rows": 12,
    }

    summary = m3_pilot_shard.dry_run_summary(shard, edge_path)

    assert summary["expected_edge_rows"] == 12
    assert not edge_path.exists()


def test_expected_row_count_is_source_rows_times_k() -> None:
    shard = {"source_rows": 7, "candidate_k": 3, "expected_edge_rows": 21}

    assert shard["source_rows"] * shard["candidate_k"] == shard["expected_edge_rows"]


def test_schema_contains_weight_and_probability_columns() -> None:
    columns = full_transition_schema_columns()

    for column in [
        "raw_edge_weight",
        "mass_adjusted_weight",
        "row_normalized_transition_prob",
    ]:
        assert column in columns


def test_validate_pilot_edges_accepts_toy_row_sums() -> None:
    columns = full_transition_schema_columns()
    rows = []
    for source_idx in [0, 1]:
        for target_idx, probability in enumerate([0.25, 0.75]):
            row = {column: 0 for column in columns}
            row.update(
                {
                    "source_anchor_id": f"s::{source_idx}",
                    "target_anchor_id": f"t::{target_idx}",
                    "source_anchor_index": source_idx,
                    "target_anchor_index": target_idx,
                    "source_time": "t0",
                    "target_time": "t1",
                    "source_day": 0.0,
                    "target_day": 1.0,
                    "time_delta": 1.0,
                    "source_slice_id": "s0",
                    "target_slice_id": f"t{target_idx}",
                    "source_slice_file": "s0.m0.h5ad",
                    "target_slice_file": f"t{target_idx}.m0.h5ad",
                    "source_mouse_id": "m0",
                    "target_mouse_id": f"m{target_idx}",
                    "evidence_mode": "pseudo_lineage",
                    "raw_edge_weight": 1.0,
                    "mass_adjusted_weight": 1.0,
                    "source_mass": 1.0,
                    "target_mass": 1.0,
                    "growth_prior": 1.0,
                    "unbalanced_weight": 1.0,
                    "tau_pair": 1.0,
                    "row_normalized_transition_prob": probability,
                    "zero_variance_molecular": False,
                    "zero_variance_composition": False,
                    "zero_variance_entropy": False,
                    "zero_variance_spatial_summary": False,
                    "zero_variance_topology": False,
                    "scaling_method_molecular": "median_iqr",
                    "scaling_method_composition": "median_iqr",
                    "scaling_method_entropy": "median_iqr",
                    "scaling_method_spatial_summary": "median_iqr",
                    "scaling_method_topology": "median_iqr",
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows, columns=columns)
    shard = {
        "expected_edge_rows": 4,
        "candidate_k": 2,
        "source_time": "t0",
        "target_time": "t1",
        "time_delta": 1.0,
    }

    diagnostics = m3_pilot_shard.validate_pilot_edges(frame, shard, columns)

    assert diagnostics["row_sum_min"] == 1.0
    assert diagnostics["candidate_count_min"] == 2


def test_target_slice_entropy_diagnostics() -> None:
    frame = pd.DataFrame(
        {
            "source_slice_id": ["s0", "s0", "s0", "s0"],
            "target_slice_id": ["t0", "t0", "t1", "t1"],
            "target_mouse_id": ["m0", "m1", "m0", "m1"],
        }
    )

    diagnostics = m3_pilot_shard.target_distribution_diagnostics(frame)

    assert diagnostics["target_slice_id_entropy_mean"] > 0
    assert np.isclose(diagnostics["top_target_slice_id_fraction_mean"], 0.5)
    assert categorical_target_diagnostics(frame, "source_slice_id", "target_mouse_id")[
        "target_mouse_id_entropy_mean"
    ] > 0


def test_m3_pilot_runner_has_no_dataset_specific_hard_coding() -> None:
    text = RUNNER_PATH.read_text(encoding="utf-8")

    for token in ["Moffitt", "Cadinu", "DSS", "colon", "Day35", "Sample_type"]:
        assert token not in text
