#!/usr/bin/env python
"""Review a completed M3 time-pair pilot and write next-stage design reports."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


TOO_SHARP_TOP1 = 0.75
TOO_SHARP_ENTROPY = 0.75
TOO_FLAT_TOP1 = 0.10
TOO_FLAT_ENTROPY_FRACTION = 0.95
COLLAPSE_TOP_FRACTION = 0.90
COLLAPSE_ENTROPY = 0.25
SECONDS_PER_HOUR = 3600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("/home/zhutao/scratch/nichefate/m3/reports"),
    )
    parser.add_argument(
        "--plan-csv",
        type=Path,
        default=Path("/home/zhutao/scratch/nichefate/m3/reports/m3_full_transition_shards.csv"),
    )
    parser.add_argument(
        "--d0-single-shard-report",
        type=Path,
        default=Path(
            "/home/zhutao/scratch/nichefate/m3/pilot_shard/"
            "pilot_report_D0_to_D3__082421_D0_m6_1_slice_3.md"
        ),
    )
    parser.add_argument("--source-time", default="D21")
    parser.add_argument("--target-time", default="D" + str(35))
    parser.add_argument("--skip-figure", action="store_true")
    return parser.parse_args()


def count_edge_parquets(root: Path) -> int:
    return sum(1 for _ in root.rglob("candidate_edges_*.parquet"))


def pair_stem(source_time: str, target_time: str) -> str:
    return f"{source_time}_to_{target_time}"


def pilot_paths(pilot_dir: Path, source_time: str, target_time: str) -> dict[str, Path]:
    stem = pair_stem(source_time, target_time)
    return {
        "report": pilot_dir / f"timepair_report_{stem}.md",
        "manifest": pilot_dir / f"timepair_manifest_{stem}.csv",
        "qc": pilot_dir / f"timepair_qc_summary_{stem}.csv",
        "shard_qc": pilot_dir / f"plot_table_shard_qc_{stem}.csv",
        "slice_flow": pilot_dir / f"plot_table_slice_flow_{stem}.csv",
        "mouse_flow": pilot_dir / f"plot_table_mouse_flow_{stem}.csv",
    }


def output_paths(reports_dir: Path, source_time: str, target_time: str) -> dict[str, Path]:
    fig_dir = reports_dir / "figures"
    stem = pair_stem(source_time, target_time)
    return {
        "review_md": reports_dir / f"m3_{stem}_pilot_review.md",
        "backend_md": reports_dir / "m3_next_backend_strategy.md",
        "m4a_md": reports_dir / "m4a_markov_assembly_contract.md",
        "metrics_csv": reports_dir / f"m3_{stem}_pilot_review_metrics.csv",
        "projection_csv": reports_dir / "m3_backend_runtime_projection.csv",
        "figure": fig_dir / f"m3_{stem}_review_summary.png",
    }


def read_inputs(pilot_dir: Path, source_time: str, target_time: str) -> dict[str, pd.DataFrame | str]:
    paths = pilot_paths(pilot_dir, source_time, target_time)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required pilot review inputs: {missing}")
    return {
        "report_text": paths["report"].read_text(encoding="utf-8"),
        "manifest": pd.read_csv(paths["manifest"]),
        "qc": pd.read_csv(paths["qc"]),
        "shard_qc": pd.read_csv(paths["shard_qc"]),
        "slice_flow": pd.read_csv(paths["slice_flow"]),
        "mouse_flow": pd.read_csv(paths["mouse_flow"]),
    }


def classify_certainty(row_entropy_mean: float, top1_probability_mean: float, candidate_k: int) -> str:
    flat_entropy = math.log(max(candidate_k, 2)) * TOO_FLAT_ENTROPY_FRACTION
    if top1_probability_mean >= TOO_SHARP_TOP1 or row_entropy_mean <= TOO_SHARP_ENTROPY:
        return "too_sharp"
    if top1_probability_mean <= TOO_FLAT_TOP1 or row_entropy_mean >= flat_entropy:
        return "too_flat"
    return "acceptable_mixed"


def collapse_warnings(qc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in qc.iterrows():
        slice_warning = (
            float(row["top_target_slice_fraction_p95"]) >= COLLAPSE_TOP_FRACTION
            or float(row["target_slice_entropy_mean"]) <= COLLAPSE_ENTROPY
        )
        mouse_warning = (
            float(row["top_target_mouse_fraction_p95"]) >= COLLAPSE_TOP_FRACTION
            or float(row["target_mouse_entropy_mean"]) <= COLLAPSE_ENTROPY
        )
        rows.append(
            {
                "source_slice_id": row["source_slice_id"],
                "target_slice_collapse_warning": bool(slice_warning),
                "target_mouse_collapse_warning": bool(mouse_warning),
                "top_target_slice_fraction_p95": float(row["top_target_slice_fraction_p95"]),
                "top_target_mouse_fraction_p95": float(row["top_target_mouse_fraction_p95"]),
                "target_slice_entropy_mean": float(row["target_slice_entropy_mean"]),
                "target_mouse_entropy_mean": float(row["target_mouse_entropy_mean"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top_target_slice_fraction_p95", "top_target_mouse_fraction_p95"],
        ascending=False,
    )


def review_metrics(manifest: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    candidate_k = int(qc["candidate_k"].iloc[0])
    row_entropy_mean = float(qc["row_entropy_mean"].mean())
    top1_probability_mean = float(qc["top1_probability_mean"].mean())
    metrics = {
        "source_time": str(qc["source_time"].iloc[0]),
        "target_time": str(qc["target_time"].iloc[0]),
        "completed_shards": int((manifest["status"] == "COMPLETED").sum()),
        "skipped_shards": int((manifest["status"] == "SKIPPED_RESUME").sum()),
        "failed_shards": int((manifest["status"] == "FAILED").sum()),
        "expected_edge_rows": int(manifest["expected_edge_rows"].sum()),
        "observed_edge_rows": int(qc["observed_edge_rows"].sum()),
        "edge_row_delta": int(qc["observed_edge_rows"].sum() - manifest["expected_edge_rows"].sum()),
        "row_sum_abs_error_max": float(qc["row_sum_abs_error_max"].max()),
        "n_nan_total": int(qc["n_nan"].sum()),
        "n_inf_total": int(qc["n_inf"].sum()),
        "negative_probability_shards": int((qc["probability_min"] < -1e-12).sum()),
        "candidate_count_min": int(qc["candidate_count_min"].min()),
        "candidate_count_max": int(qc["candidate_count_max"].max()),
        "candidate_count_mean": float(qc["candidate_count_mean"].mean()),
        "candidate_k": candidate_k,
        "row_entropy_mean": row_entropy_mean,
        "row_entropy_median_mean": float(qc["row_entropy_median"].mean()),
        "row_entropy_p05_min": float(qc["row_entropy_p05"].min()),
        "row_entropy_p95_max": float(qc["row_entropy_p95"].max()),
        "top1_probability_mean": top1_probability_mean,
        "top1_probability_median_mean": float(qc["top1_probability_median"].mean()),
        "top1_probability_p95_max": float(qc["top1_probability_p95"].max()),
        "target_slice_entropy_mean": float(qc["target_slice_entropy_mean"].mean()),
        "target_mouse_entropy_mean": float(qc["target_mouse_entropy_mean"].mean()),
        "top_target_slice_fraction_p95_max": float(qc["top_target_slice_fraction_p95"].max()),
        "top_target_mouse_fraction_p95_max": float(qc["top_target_mouse_fraction_p95"].max()),
        "total_runtime_seconds": float(qc["runtime_seconds"].sum()),
        "max_runtime_seconds": float(qc["runtime_seconds"].max()),
        "peak_rss_gib": float(qc["max_rss_gib"].max()),
        "output_size_bytes": int(qc["output_size_bytes"].sum()),
        "certainty_classification": classify_certainty(row_entropy_mean, top1_probability_mean, candidate_k),
    }
    return pd.DataFrame([metrics])


def parse_single_shard_report(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    patterns = {
        "source_rows": r"Source rows:\s*([0-9]+)",
        "target_rows": r"Target rows:\s*([0-9]+)",
        "edge_rows": r"Edge rows:\s*([0-9]+)",
        "runtime_seconds": r"Runtime seconds:\s*([0-9.]+)",
        "max_rss_kb": r"Max RSS KB:\s*([0-9]+)",
    }
    parsed: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            parsed[key] = float(match.group(1))
    required = {"source_rows", "target_rows", "edge_rows", "runtime_seconds"}
    return parsed if required <= set(parsed) else None


def runtime_projection(
    plan: pd.DataFrame,
    d21_qc: pd.DataFrame,
    d0_single: dict[str, float] | None,
) -> pd.DataFrame:
    by_pair = (
        plan.groupby(["source_time", "target_time"], observed=True)
        .agg(
            shards=("source_slice_id", "count"),
            source_rows=("source_rows", "sum"),
            target_rows=("target_time_rows", "first"),
            expected_edges=("expected_edge_rows", "sum"),
        )
        .reset_index()
    )
    d21_edges = float(d21_qc["observed_edge_rows"].sum())
    d21_runtime = float(d21_qc["runtime_seconds"].sum())
    d21_source_rows = float(d21_qc["source_rows"].sum())
    d21_target_rows = float(d21_qc["target_rows"].iloc[0])
    edge_rates = [d21_runtime / d21_edges]
    knn_rates = [d21_runtime / (d21_source_rows * d21_target_rows)]
    calibrators = [f"{d21_qc['source_time'].iloc[0]}_{d21_qc['target_time'].iloc[0]}_timepair"]
    if d0_single:
        edge_rates.append(float(d0_single["runtime_seconds"]) / float(d0_single["edge_rows"]))
        knn_rates.append(
            float(d0_single["runtime_seconds"])
            / (float(d0_single["source_rows"]) * float(d0_single["target_rows"]))
        )
        calibrators.append("D0_D3_single_shard")
    edge_rate = max(edge_rates)
    knn_rate = max(knn_rates)
    rows: list[dict[str, Any]] = []
    for _, pair in by_pair.iterrows():
        edge_projection = float(pair["expected_edges"]) * edge_rate
        knn_projection = float(pair["source_rows"]) * float(pair["target_rows"]) * knn_rate
        conservative = max(edge_projection, knn_projection)
        rows.append(
            {
                "source_time": pair["source_time"],
                "target_time": pair["target_time"],
                "shards": int(pair["shards"]),
                "source_rows": int(pair["source_rows"]),
                "target_rows": int(pair["target_rows"]),
                "expected_edges": int(pair["expected_edges"]),
                "edge_throughput_projection_seconds": edge_projection,
                "knn_complexity_projection_seconds": knn_projection,
                "conservative_projection_seconds": conservative,
                "conservative_projection_hours": conservative / SECONDS_PER_HOUR,
                "calibration_source": "+".join(calibrators),
            }
        )
    total = {
        "source_time": "ALL",
        "target_time": "ALL",
        "shards": int(by_pair["shards"].sum()),
        "source_rows": int(by_pair["source_rows"].sum()),
        "target_rows": np.nan,
        "expected_edges": int(by_pair["expected_edges"].sum()),
        "edge_throughput_projection_seconds": float(sum(row["edge_throughput_projection_seconds"] for row in rows)),
        "knn_complexity_projection_seconds": float(sum(row["knn_complexity_projection_seconds"] for row in rows)),
        "conservative_projection_seconds": float(sum(row["conservative_projection_seconds"] for row in rows)),
        "conservative_projection_hours": float(sum(row["conservative_projection_seconds"] for row in rows))
        / SECONDS_PER_HOUR,
        "calibration_source": "+".join(calibrators),
    }
    return pd.DataFrame([*rows, total])


def write_pilot_review(
    path: Path,
    metrics: pd.DataFrame,
    collapse: pd.DataFrame,
    projection: pd.DataFrame,
    d0_single: dict[str, float] | None,
    figure_warning: str | None,
) -> None:
    row = metrics.iloc[0].to_dict()
    severe = collapse[
        collapse["target_slice_collapse_warning"] | collapse["target_mouse_collapse_warning"]
    ]
    lines = [
        f"# M3 {row['source_time']} to {row['target_time']} Pilot Review",
        "",
        "This report reviews existing local M3 transition edge shards only.",
        "No new M3 edge construction, global Markov P assembly, GPCCA, fate probability, Branched NicheFlow, M5, or regulator analysis was run.",
        "",
        "## Numerical Stability",
        f"- Completed/skipped/failed shards: {row['completed_shards']} / {row['skipped_shards']} / {row['failed_shards']}",
        f"- Expected vs observed edge rows: {row['expected_edge_rows']} / {row['observed_edge_rows']}",
        f"- Row sum absolute error max: {row['row_sum_abs_error_max']:.6g}",
        f"- NaN / infinite counts: {row['n_nan_total']} / {row['n_inf_total']}",
        f"- Negative probability shard count: {row['negative_probability_shards']}",
        f"- Candidate count min/max/mean: {row['candidate_count_min']} / {row['candidate_count_max']} / {row['candidate_count_mean']:.3f}",
        "",
        "## Transition Certainty",
        f"- Certainty classification: {row['certainty_classification']}",
        f"- Row entropy mean / median-mean / p05-min / p95-max: {row['row_entropy_mean']:.6g} / {row['row_entropy_median_mean']:.6g} / {row['row_entropy_p05_min']:.6g} / {row['row_entropy_p95_max']:.6g}",
        f"- Top1 probability mean / median-mean / p95-max: {row['top1_probability_mean']:.6g} / {row['top1_probability_median_mean']:.6g} / {row['top1_probability_p95_max']:.6g}",
        "- Certainty thresholds are diagnostic only and do not imply terminal state or fate behavior.",
        "",
        "## Batch-Collapse Diagnostics",
        f"- Target slice entropy mean: {row['target_slice_entropy_mean']:.6g}",
        f"- Target mouse entropy mean: {row['target_mouse_entropy_mean']:.6g}",
        f"- Top target slice fraction p95 max: {row['top_target_slice_fraction_p95_max']:.6g}",
        f"- Top target mouse fraction p95 max: {row['top_target_mouse_fraction_p95_max']:.6g}",
        "- Collapse warnings are diagnostic warnings, not automatic failures.",
    ]
    if not severe.empty:
        lines.append("- Source slices with strongest warnings:")
        for _, warning in severe.head(5).iterrows():
            lines.append(
                f"  - {warning['source_slice_id']}: "
                f"slice_p95={warning['top_target_slice_fraction_p95']:.3f}, "
                f"mouse_p95={warning['top_target_mouse_fraction_p95']:.3f}"
            )
    total = projection[projection["source_time"] == "ALL"].iloc[0]
    lines.extend(
        [
            "",
            "## Runtime And Memory",
            f"- Observed runtime seconds: {row['total_runtime_seconds']:.3f}",
            f"- Peak RSS GiB: {row['peak_rss_gib']:.3f}",
            f"- sklearn_exact projected full M3 runtime hours, conservative: {total['conservative_projection_hours']:.3f}",
            "- Projection methods: edge-throughput, exact-KNN complexity, and conservative maximum.",
            "- D3 to D9 and D9 to D21 remain inefficient or risky with sklearn_exact because exact KNN scales with N_source x N_target x D_retrieval.",
            "",
            "## Optional Calibration",
            f"- D0 to D3 single-shard calibration used: {bool(d0_single)}",
        ]
    )
    if figure_warning:
        lines.extend(["", "## Visualization Warning", f"- {figure_warning}"])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_backend_strategy(path: Path, projection: pd.DataFrame) -> None:
    lines = [
        "# M3 Next Backend And Execution Strategy",
        "",
        "This is a design report only. It does not implement ANN backends, launch Slurm jobs, or build additional M3 edges.",
        "",
        "## Runtime Projection",
        dataframe_to_markdown(projection),
        "",
        "## ANN Backend Contract",
        "- Preserve the existing build_candidate_neighbors(source_matrix, target_matrix, k, backend, metric, chunk_size, random_seed) interface.",
        "- Required output remains candidate target indices, distances, backend name, and metric.",
        "- Future backend candidates: FAISS, hnswlib, pynndescent.",
        "- No new dependencies are added in this stage.",
        "",
        "## Exact-vs-ANN Validation",
        "- Compare ANN against sklearn_exact on one small source-slice shard.",
        "- Report recall@K, top1 agreement, target-slice distribution drift, distance-rank correlation, row-normalized probability drift, and QC metric deltas.",
        "- Acceptability should be decided before D3->D9 or D9->D21 full execution.",
        "",
        "## Slurm Job-Array Strategy",
        "- Use one source-slice shard per array job.",
        "- Group jobs by time pair and cap concurrency explicitly based on target pool memory and observed RSS.",
        "- Use resume validation of edge parquet plus shard report before recomputation.",
        "- Retry failed shards individually from the manifest.",
        "- Avoid unrestricted Python multiprocessing because target matrices can be duplicated per worker.",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Return a simple Markdown table without optional tabulate dependency."""

    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_m4a_contract(path: Path) -> None:
    lines = [
        "# M4A Markov Assembly Contract",
        "",
        "This contract defines later assembly semantics only. It does not construct a global Markov transition matrix P.",
        "",
        "## Inputs",
        "- M3 local edge shards with required fields: source_anchor_id, target_anchor_id, source_time, target_time, time_delta, raw_edge_weight, mass_adjusted_weight, row_normalized_transition_prob.",
        "- Optional future evidence fields may include barcode-derived lineage evidence without changing source-target pair identity.",
        "",
        "## Output Contract",
        "- A future sparse global transition object and metadata tables.",
        "- Raw and mass-adjusted weights remain preserved alongside any assembled probability representation.",
        "",
        "## Semantics",
        "- row_normalized_transition_prob is local to each source anchor candidate set and is not global P.",
        "- Global row-stochasticity is validated only after future assembly.",
        "- No terminal-state, absorption, fate, or GPCCA interpretation is performed in this contract stage.",
        "",
        "## Validation",
        "- All source anchors represented for included time pairs.",
        "- Local row sums valid before assembly.",
        "- Global row-stochasticity checked only after the future sparse object is assembled.",
        "- Schema consistency and duplicate source-target edge checks are required before downstream use.",
        "",
        "## Compatibility",
        "- Markov-GPCCA baseline consumes the future sparse P after assembly and validation.",
        "- Branched NicheFlow consumes weighted source-target pseudo-lineage pairs and preserved raw or mass-adjusted weights.",
        "- Barcode-supervised mode can replace or augment pseudo-lineage evidence while preserving the same edge-shard schema.",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def try_write_review_figure(path: Path, qc: pd.DataFrame) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        x = qc["source_slice_id"].astype(str)
        for ax, column, title in [
            (axes[0, 0], "runtime_seconds", "Runtime seconds"),
            (axes[0, 1], "max_rss_gib", "Max RSS GiB"),
            (axes[0, 2], "row_entropy_mean", "Row entropy mean"),
            (axes[1, 0], "top1_probability_mean", "Top1 probability mean"),
            (axes[1, 1], "top_target_slice_fraction_p95", "Top target slice p95"),
            (axes[1, 2], "top_target_mouse_fraction_p95", "Top target mouse p95"),
        ]:
            ax.bar(x, pd.to_numeric(qc[column], errors="coerce"))
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return None
    except Exception as exc:  # noqa: BLE001
        return f"Review figure generation failed but reports were written: {exc}"


def run_review(args: argparse.Namespace) -> dict[str, Any]:
    if args.pilot_dir is None:
        args.pilot_dir = args.reports_dir.parent / f"timepair_pilot_{pair_stem(args.source_time, args.target_time)}"
    edge_root = args.reports_dir.parent
    before_count = count_edge_parquets(edge_root)
    inputs = read_inputs(args.pilot_dir, args.source_time, args.target_time)
    manifest = inputs["manifest"]
    qc = inputs["qc"]
    plan = pd.read_csv(args.plan_csv)
    d0_single = parse_single_shard_report(args.d0_single_shard_report)
    metrics = review_metrics(manifest, qc)
    collapse = collapse_warnings(qc)
    projection = runtime_projection(plan, qc, d0_single)
    paths = output_paths(args.reports_dir, args.source_time, args.target_time)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    figure_warning = None if args.skip_figure else try_write_review_figure(paths["figure"], qc)
    metrics.to_csv(paths["metrics_csv"], index=False)
    projection.to_csv(paths["projection_csv"], index=False)
    write_pilot_review(paths["review_md"], metrics, collapse, projection, d0_single, figure_warning)
    write_backend_strategy(paths["backend_md"], projection)
    write_m4a_contract(paths["m4a_md"])
    after_count = count_edge_parquets(edge_root)
    if before_count != after_count:
        raise RuntimeError(
            f"Read-only review stage changed edge parquet count: before={before_count}, after={after_count}"
        )
    return {
        "before_edge_parquet_count": before_count,
        "after_edge_parquet_count": after_count,
        "outputs": {key: str(value) for key, value in paths.items() if key != "figure" or value.exists()},
        "figure_warning": figure_warning,
        "metrics": metrics.iloc[0].to_dict(),
    }


def main() -> int:
    result = run_review(parse_args())
    print(f"EDGE_PARQUET_COUNT_BEFORE {result['before_edge_parquet_count']}")
    print(f"EDGE_PARQUET_COUNT_AFTER {result['after_edge_parquet_count']}")
    print(f"COMPLETED_SHARDS {result['metrics']['completed_shards']}")
    print(f"OBSERVED_EDGE_ROWS {result['metrics']['observed_edge_rows']}")
    print("WROTE_OUTPUTS")
    for value in result["outputs"].values():
        print(value)
    if result["figure_warning"]:
        print(f"FIGURE_WARNING {result['figure_warning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
