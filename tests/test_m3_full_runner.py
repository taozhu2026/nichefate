import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nichefate.transition import full_transition_schema_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m3_15_run_full_m3_by_shard.py"
SPEC = importlib.util.spec_from_file_location("m3_full_runner", SCRIPT_PATH)
m3_full = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m3_full
SPEC.loader.exec_module(m3_full)


def dryrun_plan(tmp_path: Path) -> pd.DataFrame:
    source_rows = [25_000] * 51 + [73_582]
    pairs = [
        ("D0", "D3", 0.0, 3.0),
        ("D3", "D9", 3.0, 9.0),
        ("D9", "D21", 9.0, 21.0),
        ("D21", "D" + "35", 21.0, 35.0),
    ]
    rows = []
    for idx, count in enumerate(source_rows):
        source_time, target_time, source_day, target_day = pairs[idx % len(pairs)]
        shard_id = f"m3_full_{idx + 1:04d}"
        slice_id = f"slice_{idx:03d}"
        output_dir = tmp_path / "full_by_shard" / f"{source_time}_to_{target_time}" / slice_id
        stem = f"{source_time}_to_{target_time}__{slice_id}"
        rows.append(
            {
                "shard_id": shard_id,
                "source_time": source_time,
                "target_time": target_time,
                "source_day": source_day,
                "target_day": target_day,
                "time_delta": target_day - source_day,
                "source_slice_id": slice_id,
                "source_slice_file": f"{slice_id}.m0.h5ad",
                "source_rows": count,
                "target_rows": 100_000 + idx,
                "candidate_k": 30,
                "expected_edge_rows": count * 30,
                "selected_backend": "sklearn_exact",
                "output_dir": str(output_dir),
                "output_parquet": str(output_dir / f"candidate_edges_{stem}.parquet"),
                "shard_report": str(output_dir / f"shard_report_{stem}.md"),
                "status_expected": "pending_explicit_approval",
                "can_resume": False,
                "reuse_existing_pilot_allowed": False,
                "requires_explicit_approval": True,
            }
        )
    return pd.DataFrame(rows)


def mock_shard(tmp_path: Path, candidate_k: int = 2) -> pd.Series:
    output_dir = tmp_path / "full_by_shard" / "D0_to_D3" / "slice_mock"
    return pd.Series(
        {
            "shard_id": "m3_full_mock",
            "source_time": "D0",
            "target_time": "D3",
            "source_day": 0.0,
            "target_day": 3.0,
            "time_delta": 3.0,
            "source_slice_id": "slice_mock",
            "source_slice_file": "slice_mock.m0.h5ad",
            "source_rows": 2,
            "target_rows": 3,
            "candidate_k": candidate_k,
            "expected_edge_rows": 2 * candidate_k,
            "selected_backend": "sklearn_exact",
            "output_dir": str(output_dir),
            "output_parquet": str(output_dir / "candidate_edges_D0_to_D3__slice_mock.parquet"),
            "shard_report": str(output_dir / "shard_report_D0_to_D3__slice_mock.md"),
            "reuse_existing_pilot_allowed": False,
            "requires_explicit_approval": True,
        }
    )


def edge_frame(candidate_k: int = 2) -> pd.DataFrame:
    rows = []
    for source_idx in range(2):
        for rank in range(candidate_k):
            rows.append(
                {
                    "source_anchor_id": f"s::{source_idx}",
                    "target_anchor_id": f"t::{rank}",
                    "source_anchor_index": source_idx,
                    "target_anchor_index": rank,
                    "source_time": "D0",
                    "target_time": "D3",
                    "source_day": 0.0,
                    "target_day": 3.0,
                    "time_delta": 3.0,
                    "source_slice_id": "slice_mock",
                    "target_slice_id": f"target_slice_{rank % 2}",
                    "source_slice_file": "slice_mock.m0.h5ad",
                    "target_slice_file": f"target_{rank}.m0.h5ad",
                    "source_mouse_id": "source_mouse",
                    "target_mouse_id": f"target_mouse_{rank % 2}",
                    "evidence_mode": "pseudo_lineage",
                    "raw_molecular_distance": 1.0,
                    "raw_composition_distance": 1.0,
                    "raw_entropy_distance": 1.0,
                    "raw_spatial_summary_distance": 1.0,
                    "raw_topology_distance": 1.0,
                    "raw_pseudotime_score": 0.0,
                    "raw_barcode_score": 0.0,
                    "scaled_molecular_distance": 0.0,
                    "scaled_composition_distance": 0.0,
                    "scaled_entropy_distance": 0.0,
                    "scaled_spatial_summary_distance": 0.0,
                    "scaled_topology_distance": 0.0,
                    "scaled_pseudotime_score": 0.0,
                    "scaled_barcode_score": 0.0,
                    "scaling_method_molecular": "toy",
                    "scaling_method_composition": "toy",
                    "scaling_method_entropy": "toy",
                    "scaling_method_spatial_summary": "toy",
                    "scaling_method_topology": "toy",
                    "zero_variance_molecular": False,
                    "zero_variance_composition": False,
                    "zero_variance_entropy": False,
                    "zero_variance_spatial_summary": False,
                    "zero_variance_topology": False,
                    "source_mass": 1.0,
                    "target_mass": 1.0,
                    "growth_prior": 1.0,
                    "unbalanced_weight": 1.0,
                    "mass_adjusted_weight": 1.0,
                    "combined_cost": 0.0,
                    "tau_pair": 1.0,
                    "raw_edge_weight": 1.0,
                    "row_normalized_transition_prob": 1.0 / candidate_k,
                }
            )
    return pd.DataFrame(rows)[full_transition_schema_columns()]


def test_runner_reads_52_shard_plan(tmp_path: Path) -> None:
    path = tmp_path / "plan.csv"
    dryrun_plan(tmp_path).to_csv(path, index=False)

    plan = m3_full.load_plan(path)

    assert len(plan) == 52


def test_d35_is_never_source_and_expected_total_rows(tmp_path: Path) -> None:
    path = tmp_path / "plan.csv"
    dryrun_plan(tmp_path).to_csv(path, index=False)

    plan = m3_full.load_plan(path)

    assert "D" + "35" not in set(plan["source_time"].astype(str))
    assert int(plan["expected_edge_rows"].sum()) == 40_457_460


def test_production_output_paths_are_under_full_by_shard_only(tmp_path: Path) -> None:
    plan = dryrun_plan(tmp_path)

    for value in plan["output_parquet"].head(10):
        path = Path(value)
        assert "full_by_shard" in path.parts
        assert path.name.startswith("candidate_edges_")


def test_resume_validation_detects_valid_mock_shard(tmp_path: Path) -> None:
    shard = mock_shard(tmp_path)
    path = Path(shard["output_parquet"])
    path.parent.mkdir(parents=True)
    edge_frame().to_parquet(path, index=False)
    Path(shard["shard_report"]).write_text("ok", encoding="utf-8")

    valid, metrics, reason = m3_full.validate_existing_outputs(shard)

    assert valid, reason
    assert metrics["observed_edge_rows"] == 4
    assert metrics["candidate_count_min"] == 2
    assert metrics["row_sum_abs_error_max"] == pytest.approx(0.0)


def test_invalid_mock_shard_is_not_skipped(tmp_path: Path) -> None:
    shard = mock_shard(tmp_path)
    path = Path(shard["output_parquet"])
    path.parent.mkdir(parents=True)
    invalid = edge_frame().iloc[:-1]
    invalid.to_parquet(path, index=False)
    Path(shard["shard_report"]).write_text("bad", encoding="utf-8")

    valid, _, reason = m3_full.validate_existing_outputs(shard)

    assert not valid
    assert "validation failed" in reason


def test_no_pilot_outputs_are_marked_production_completed(tmp_path: Path) -> None:
    plan = dryrun_plan(tmp_path)

    assert not plan["reuse_existing_pilot_allowed"].astype(bool).any()
    assert not plan["status_expected"].eq("production_completed").any()


def test_no_global_or_downstream_outputs_are_produced_in_manifest(tmp_path: Path) -> None:
    plan = dryrun_plan(tmp_path)
    records = [
        m3_full.completed_record(
            mock_shard(tmp_path),
            {
                **m3_full.validate_edge_frame(edge_frame(), mock_shard(tmp_path)),
                "runtime_seconds": 1.0,
                "max_rss_gib": 1.0,
                "output_bytes": 10,
                "backend": "sklearn_exact",
            },
            "COMPLETED",
        )
    ]

    outputs = m3_full.write_run_outputs(tmp_path / "full_by_shard", tmp_path / "reports", records, [], plan, 1.0)
    payload = (tmp_path / "full_by_shard" / "full_m3_manifest.json").read_text(encoding="utf-8")

    assert outputs["manifest_json"].exists()
    for token in ["no_global_markov_p", "no_gpcca", "no_fate_probability", "no_branched_nicheflow", "no_m5"]:
        assert f'"{token}": true' in payload


def test_collapse_warnings_are_warning_only(tmp_path: Path) -> None:
    frame = edge_frame()
    frame["target_slice_id"] = "one_slice"
    frame["target_mouse_id"] = "one_mouse"

    metrics = m3_full.validate_edge_frame(frame, mock_shard(tmp_path))

    assert metrics["collapse_warnings"]
    assert metrics["observed_edge_rows"] == 4


def test_full_summary_aggregation_works_on_synthetic_shard_summaries(tmp_path: Path) -> None:
    shard = mock_shard(tmp_path)
    metrics = {
        **m3_full.validate_edge_frame(edge_frame(), shard),
        "runtime_seconds": 2.0,
        "max_rss_gib": 3.0,
        "output_bytes": 100,
        "backend": "sklearn_exact",
    }
    record = m3_full.completed_record(shard, metrics, "COMPLETED")

    summary = m3_full.aggregate_summary([record], [], dryrun_plan(tmp_path))

    assert int(summary["observed_edge_rows"].sum()) == 4
    assert float(summary["runtime_seconds"].sum()) == pytest.approx(2.0)
    assert float(summary["max_rss_gib"].max()) == pytest.approx(3.0)


def test_dry_run_filtering_does_not_create_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan = dryrun_plan(tmp_path)
    args = Namespace(
        output_root=tmp_path / "full_by_shard",
        backend="sklearn_exact",
        candidate_k=30,
    )

    m3_full.dry_run_report(plan, plan.head(1), args)

    assert not args.output_root.exists()
    assert "M3_FULL_RUNNER_DRY_RUN" in capsys.readouterr().out
