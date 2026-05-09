import importlib.util
from argparse import Namespace
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m3_07_review_timepair_pilot_and_plan_next.py"
SPEC = importlib.util.spec_from_file_location("m3_pilot_review", SCRIPT_PATH)
m3_pilot_review = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(m3_pilot_review)


def mock_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_time": "t0",
                "target_time": "t1",
                "source_slice_id": "s0",
                "source_slice_file": "s0.m0.h5ad",
                "source_rows": 2,
                "target_rows": 4,
                "candidate_k": 2,
                "expected_edge_rows": 4,
                "status": "COMPLETED",
                "edge_path": "edge.parquet",
                "report_path": "report.md",
            }
        ]
    )


def mock_qc() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "status": "PASS",
                "source_time": "t0",
                "target_time": "t1",
                "source_slice_id": "s0",
                "source_rows": 2,
                "target_rows": 4,
                "expected_edge_rows": 4,
                "observed_edge_rows": 4,
                "candidate_k": 2,
                "runtime_seconds": 1.0,
                "max_rss_gib": 0.5,
                "output_size_bytes": 100,
                "row_sum_abs_error_max": 0.0,
                "candidate_count_min": 2,
                "candidate_count_max": 2,
                "candidate_count_mean": 2.0,
                "n_nan": 0,
                "n_inf": 0,
                "probability_min": 0.25,
                "probability_max": 0.75,
                "row_entropy_mean": 0.8,
                "row_entropy_median": 0.8,
                "row_entropy_p05": 0.7,
                "row_entropy_p95": 0.9,
                "top1_probability_mean": 0.75,
                "top1_probability_median": 0.75,
                "top1_probability_p95": 0.8,
                "target_slice_entropy_mean": 0.0,
                "target_mouse_entropy_mean": 0.0,
                "top_target_slice_fraction_p95": 1.0,
                "top_target_mouse_fraction_p95": 1.0,
            }
        ]
    )


def mock_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_time": "t0",
                "target_time": "t1",
                "source_slice_id": "s0",
                "source_rows": 2,
                "target_time_rows": 4,
                "expected_edge_rows": 4,
            },
            {
                "source_time": "t1",
                "target_time": "t2",
                "source_slice_id": "s1",
                "source_rows": 3,
                "target_time_rows": 5,
                "expected_edge_rows": 6,
            },
        ]
    )


def write_mock_inputs(root: Path) -> tuple[Path, Path]:
    pilot = root / "pilot"
    reports = root / "reports"
    pilot.mkdir()
    reports.mkdir()
    stem = "D21_to_D35"
    (pilot / f"timepair_report_{stem}.md").write_text("report\n", encoding="utf-8")
    mock_manifest().to_csv(pilot / f"timepair_manifest_{stem}.csv", index=False)
    mock_qc().to_csv(pilot / f"timepair_qc_summary_{stem}.csv", index=False)
    mock_qc().to_csv(pilot / f"plot_table_shard_qc_{stem}.csv", index=False)
    pd.DataFrame(
        [{"source_slice_id": "s0", "target_slice_id": "t0", "edge_count": 4, "edge_mass": 2.0}]
    ).to_csv(pilot / f"plot_table_slice_flow_{stem}.csv", index=False)
    pd.DataFrame(
        [{"source_mouse_id": "m0", "target_mouse_id": "m1", "edge_count": 4, "edge_mass": 2.0}]
    ).to_csv(pilot / f"plot_table_mouse_flow_{stem}.csv", index=False)
    mock_plan().to_csv(reports / "plan.csv", index=False)
    return pilot, reports


def test_review_metrics_reads_mock_qc_summary() -> None:
    metrics = m3_pilot_review.review_metrics(mock_manifest(), mock_qc())

    assert int(metrics["completed_shards"].iloc[0]) == 1
    assert int(metrics["observed_edge_rows"].iloc[0]) == 4
    assert metrics["certainty_classification"].iloc[0] == "too_sharp"


def test_collapse_warning_logic_is_warning_only() -> None:
    warnings = m3_pilot_review.collapse_warnings(mock_qc())
    metrics = m3_pilot_review.review_metrics(mock_manifest(), mock_qc())

    assert bool(warnings["target_slice_collapse_warning"].iloc[0])
    assert bool(warnings["target_mouse_collapse_warning"].iloc[0])
    assert int(metrics["failed_shards"].iloc[0]) == 0


def test_runtime_projection_has_all_methods_without_optional_d0_report() -> None:
    projection = m3_pilot_review.runtime_projection(mock_plan(), mock_qc(), None)

    assert {
        "edge_throughput_projection_seconds",
        "knn_complexity_projection_seconds",
        "conservative_projection_seconds",
    } <= set(projection.columns)
    assert "ALL" in set(projection["source_time"])


def test_missing_optional_d0_report_does_not_fail() -> None:
    parsed = m3_pilot_review.parse_single_shard_report(Path("/missing/report.md"))

    assert parsed is None


def test_m4a_contract_contains_no_global_p_construction_command(tmp_path: Path) -> None:
    path = tmp_path / "m4a.md"
    m3_pilot_review.write_m4a_contract(path)
    text = path.read_text(encoding="utf-8").lower()

    assert "does not construct" in text
    assert "gpcca execution" not in text
    assert "fit(" not in text


def test_review_output_paths_do_not_target_downstream_outputs(tmp_path: Path) -> None:
    paths = m3_pilot_review.output_paths(tmp_path, "t0", "t1")
    names = "\n".join(path.name.lower() for path in paths.values())

    for token in ["gpcca", "fate", "branched", "nicheflow", "m5", "regulator"]:
        assert token not in names


def test_edge_parquet_count_guard_detects_accidental_edges(tmp_path: Path) -> None:
    root = tmp_path / "m3"
    root.mkdir()
    (root / "candidate_edges_existing.parquet").touch()

    assert m3_pilot_review.count_edge_parquets(root) == 1


def test_run_review_writes_reports_and_preserves_edge_count(tmp_path: Path) -> None:
    pilot, reports = write_mock_inputs(tmp_path)
    args = Namespace(
        pilot_dir=pilot,
        reports_dir=reports,
        plan_csv=reports / "plan.csv",
        d0_single_shard_report=tmp_path / "missing.md",
        source_time="D21",
        target_time="D35",
        skip_figure=True,
    )

    result = m3_pilot_review.run_review(args)

    assert result["before_edge_parquet_count"] == result["after_edge_parquet_count"]
    assert (reports / "m3_D21_to_D35_pilot_review_metrics.csv").exists()
    assert (reports / "m3_backend_runtime_projection.csv").exists()
