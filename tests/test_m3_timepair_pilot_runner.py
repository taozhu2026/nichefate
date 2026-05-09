import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nichefate.io import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "m3_06_run_transition_timepair_pilot.py"
SPEC = importlib.util.spec_from_file_location("m3_timepair_runner", RUNNER_PATH)
m3_timepair_runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(m3_timepair_runner)


def make_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_time": "t0",
                "target_time": "t1",
                "source_day": 0.0,
                "target_day": 1.0,
                "time_delta": 1.0,
                "source_slice_id": "s0",
                "source_slice_file": "s0.m0.h5ad",
                "source_rows": 2,
                "target_time_rows": 4,
                "target_slice_count": 2,
                "candidate_k": 2,
                "expected_edge_rows": 4,
            },
            {
                "source_time": "t1",
                "target_time": "t2",
                "source_day": 1.0,
                "target_day": 2.0,
                "time_delta": 1.0,
                "source_slice_id": "s1",
                "source_slice_file": "s1.m0.h5ad",
                "source_rows": 3,
                "target_time_rows": 5,
                "target_slice_count": 2,
                "candidate_k": 2,
                "expected_edge_rows": 6,
            },
        ]
    )


def make_edge_frame() -> pd.DataFrame:
    rows = []
    for source_idx in [0, 1]:
        for target_idx, probability in enumerate([0.25, 0.75]):
            rows.append(
                {
                    "source_anchor_id": f"s::{source_idx}",
                    "target_anchor_id": f"t::{target_idx}",
                    "source_time": "t0",
                    "target_time": "t1",
                    "source_day": 0.0,
                    "target_day": 1.0,
                    "time_delta": 1.0,
                    "source_slice_id": "s0",
                    "target_slice_id": "t0",
                    "source_slice_file": "s0.m0.h5ad",
                    "target_slice_file": "t0.m0.h5ad",
                    "source_mouse_id": "m0",
                    "target_mouse_id": f"m{target_idx}",
                    "combined_cost": float(target_idx),
                    "raw_edge_weight": 1.0,
                    "mass_adjusted_weight": 1.0,
                    "row_normalized_transition_prob": probability,
                    "tau_pair": 1.0,
                }
            )
    return pd.DataFrame(rows)


def make_args(tmp_path: Path, plan_csv: Path, **overrides: object) -> Namespace:
    values = {
        "config": "configs/m3_transition_kernel.yaml",
        "source_time": "t0",
        "target_time": "t1",
        "plan_csv": plan_csv,
        "output_dir": tmp_path,
        "dry_run": False,
        "resume": False,
        "overwrite": False,
        "max_shards": None,
        "stop_on_error": False,
        "backend": "sklearn_exact",
        "candidate_k": 2,
        "max_memory_gb_warning": 80.0,
        "blas_threads": 1,
    }
    values.update(overrides)
    return Namespace(**values)


def test_plan_filtering_selects_only_requested_time_pair() -> None:
    selected = m3_timepair_runner.filter_timepair_plan(make_plan(), "t0", "t1", candidate_k=2)

    assert len(selected) == 1
    assert set(selected["source_time"]) == {"t0"}
    assert set(selected["target_time"]) == {"t1"}
    assert int(selected["expected_edge_rows"].sum()) == 4


def test_dry_run_writes_no_edge_parquet(tmp_path: Path) -> None:
    plan_csv = tmp_path / "plan.csv"
    make_plan().to_csv(plan_csv, index=False)

    result = m3_timepair_runner.execute_timepair(
        make_args(tmp_path, plan_csv, dry_run=True),
    )

    assert result == 0
    assert list(tmp_path.glob("*.parquet")) == []


def test_resume_skip_logic_uses_existing_valid_outputs(tmp_path: Path) -> None:
    shard = make_plan().iloc[0].to_dict()
    paths = m3_timepair_runner.shard_paths(tmp_path, "t0", "t1", "s0")
    make_edge_frame().to_parquet(paths["edges"], index=False)
    paths["report"].write_text("existing report\n", encoding="utf-8")

    valid, qc, *_ = m3_timepair_runner.validate_existing_output(
        paths["edges"],
        paths["report"],
        shard,
        "sklearn_exact",
    )

    assert valid
    assert qc is not None
    assert qc["status"] == "SKIPPED_RESUME"
    assert qc["observed_edge_rows"] == 4


def test_overwrite_protection_blocks_existing_outputs(tmp_path: Path) -> None:
    plan_csv = tmp_path / "plan.csv"
    make_plan().head(1).to_csv(plan_csv, index=False)
    paths = m3_timepair_runner.shard_paths(tmp_path, "t0", "t1", "s0")
    paths["edges"].touch()

    with pytest.raises(FileExistsError):
        m3_timepair_runner.execute_timepair(make_args(tmp_path, plan_csv))


def test_resume_and_overwrite_combination_is_rejected(tmp_path: Path) -> None:
    plan_csv = tmp_path / "plan.csv"
    make_plan().head(1).to_csv(plan_csv, index=False)

    with pytest.raises(ValueError, match="either --resume or --overwrite"):
        m3_timepair_runner.execute_timepair(
            make_args(tmp_path, plan_csv, resume=True, overwrite=True),
        )


def test_no_global_markov_or_fate_output_paths_are_defined(tmp_path: Path) -> None:
    paths = m3_timepair_runner.timepair_paths(tmp_path, "t0", "t1")

    text = "\n".join(path.name for path in paths.values()).lower()
    assert "markov" not in text
    assert "fate" not in text
    assert "gpcca" not in text


def test_fixed_k_config_placeholders_remain_present() -> None:
    full = load_config("configs/m3_transition_kernel.yaml")["full_m3"]

    assert full["candidate_k_mode"] == "fixed"
    assert full["adaptive_k_options"]["fraction_of_target"] is None
    assert full["adaptive_k_options"]["min_k"] == 30
    assert full["adaptive_k_options"]["max_k"] == 100


def test_qc_aggregation_and_warning_only_collapse() -> None:
    shard = make_plan().iloc[0].to_dict()

    qc, anchor_qc, slice_flow, mouse_flow, warnings = m3_timepair_runner.compute_edge_qc(
        make_edge_frame(),
        shard,
        backend="sklearn_exact",
        runtime_seconds=1.0,
        max_rss_kb=1024,
        output_size_bytes=10,
    )
    passes, reason = m3_timepair_runner.required_qc_passes(qc)

    assert passes, reason
    assert any("target-slice" in warning for warning in warnings)
    assert len(anchor_qc) == 2
    assert np.isclose(slice_flow["edge_mass"].sum(), 2.0)
    assert not mouse_flow.empty


def test_missing_mouse_metadata_skips_mouse_flow_with_warning() -> None:
    frame = make_edge_frame()
    frame["target_mouse_id"] = np.nan

    _, mouse_flow, warnings = m3_timepair_runner.edge_flow_tables(frame)

    assert mouse_flow.empty
    assert any("Mouse metadata" in warning for warning in warnings)


def test_visualization_table_generation_works_on_synthetic_inputs(tmp_path: Path) -> None:
    shard = make_plan().iloc[0].to_dict()
    frame = make_edge_frame()
    qc, anchor_qc, slice_flow, mouse_flow, warnings = m3_timepair_runner.compute_edge_qc(
        frame,
        shard,
        backend="sklearn_exact",
        runtime_seconds=1.0,
        max_rss_kb=1024,
        output_size_bytes=10,
    )

    m3_timepair_runner.write_timepair_outputs(
        tmp_path,
        "t0",
        "t1",
        [
            m3_timepair_runner.build_manifest_row(
                shard,
                tmp_path / "edge.parquet",
                tmp_path / "report.md",
                "COMPLETED",
            )
        ],
        [qc],
        [anchor_qc],
        [slice_flow],
        [mouse_flow],
        {
            "target_retrieval_matrix_gib": 0.1,
            "target_rerank_matrix_gib": 0.1,
            "source_shard_matrix_gib": 0.1,
            "approx_per_worker_memory_gib": 0.3,
            "safe_single_node_concurrency": 1,
            "max_memory_gb_warning": 80,
        },
        warnings,
        started_at=0.0,
        dry_run=True,
    )

    assert (tmp_path / "plot_table_shard_qc_t0_to_t1.csv").exists()
    assert (tmp_path / "plot_table_slice_flow_t0_to_t1.csv").exists()
    assert (tmp_path / "plot_table_mouse_flow_t0_to_t1.csv").exists()


def test_no_forbidden_imports_are_introduced() -> None:
    text = RUNNER_PATH.read_text(encoding="utf-8")

    forbidden = [
        ("import", "squidpy"),
        ("import", "spatialdata"),
        ("import", "harmonypy"),
        ("from", "squidpy"),
        ("from", "spatialdata"),
        ("from", "harmonypy"),
    ]
    for prefix, package in forbidden:
        token = f"{prefix} {package}"
        assert token not in text
