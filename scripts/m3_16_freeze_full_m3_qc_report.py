#!/usr/bin/env python
"""Freeze final M3 full-shard QC artifacts without constructing new edges."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config
from nichefate.transition import full_transition_schema_columns


DEFAULT_PLAN_CSV = Path("/home/zhutao/scratch/nichefate/m3/reports/m3_full_m3_final_dryrun_shards.csv")
DEFAULT_OUTPUT_ROOT = Path("/home/zhutao/scratch/nichefate/m3/full_by_shard")
DEFAULT_REPORTS_DIR = Path("/home/zhutao/scratch/nichefate/m3/reports")
DEFAULT_CONFIG = Path("configs/m3_transition_kernel.yaml")
DEFAULT_EXISTING_FIGURES_DIR = DEFAULT_REPORTS_DIR / "figures" / "full_m3"
DEFAULT_FINAL_FIGURES_DIR = DEFAULT_REPORTS_DIR / "figures" / "full_m3_final"

FINAL_MANIFEST_CSV = "m3_full_m3_final_freeze_manifest.csv"
FINAL_MANIFEST_JSON = "m3_full_m3_final_freeze_manifest.json"
FINAL_SCHEMA_JSON = "m3_full_m3_final_schema.json"
FINAL_QC_SUMMARY_CSV = "m3_full_m3_final_qc_summary.csv"
FINAL_HANDOFF_MD = "m3_full_m3_final_handoff_to_m4a.md"
FINAL_FIGURE_INVENTORY_CSV = "m3_full_m3_final_figure_inventory.csv"
FINAL_FIGURE_INVENTORY_JSON = "m3_full_m3_final_figure_inventory.json"

EXPECTED_EXISTING_FIGURES = [
    "m3_full_runtime_memory_by_shard.png",
    "m3_full_edge_rows_by_time_pair.png",
    "m3_full_row_entropy_by_time_pair.png",
    "m3_full_top1_probability_by_time_pair.png",
    "m3_full_target_slice_collapse_by_time_pair.png",
    "m3_full_target_mouse_collapse_by_time_pair.png",
    "m3_full_source_slice_qc_heatmap.png",
    "m3_full_summary_dashboard.png",
]

FINAL_FIGURE_NAMES = [
    "m3_full_m3_final_rows_by_time_pair.png",
    "m3_full_m3_final_row_sum_qc_by_time_pair.png",
    "m3_full_m3_final_entropy_top1_by_time_pair.png",
    "m3_full_m3_final_runtime_memory_by_time_pair.png",
    "m3_full_m3_final_collapse_diagnostics_by_time_pair.png",
]

NO_DOWNSTREAM_FLAGS = {
    "no_global_markov_p": True,
    "no_gpcca": True,
    "no_fate_probability": True,
    "no_branched_nicheflow": True,
    "no_m5": True,
    "no_regulator_analysis": True,
}


def _load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M3_15 = _load_script_module(
    "m3_15_run_full_m3_by_shard",
    PROJECT_ROOT / "scripts" / "m3_15_run_full_m3_by_shard.py",
)


@dataclass(frozen=True)
class FreezeExpectations:
    shard_count: int
    total_edge_rows: int
    candidate_k: int
    expected_time_pairs: tuple[tuple[str, str], ...]
    row_sum_atol: float


DEFAULT_EXPECTATIONS = FreezeExpectations(
    shard_count=M3_15.EXPECTED_SHARD_COUNT,
    total_edge_rows=M3_15.EXPECTED_TOTAL_EDGE_ROWS,
    candidate_k=M3_15.DEFAULT_CANDIDATE_K,
    expected_time_pairs=(("D0", "D3"), ("D3", "D9"), ("D9", "D21"), ("D21", "D35")),
    row_sum_atol=M3_15.ROW_SUM_ATOL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--existing-figures-dir", type=Path, default=None)
    parser.add_argument("--final-figures-dir", type=Path, default=None)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def assert_no_ssd_path(path: Path, label: str) -> None:
    text = str(path.resolve())
    if text == "/ssd" or text.startswith("/ssd/"):
        raise ValueError(f"Refusing to use /ssd for {label}: {path}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(json_safe(payload), indent=2) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(tmp, dpi=140)
    os.replace(tmp, path)


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    stack = [path]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
    return int(total)


def load_plan_for_freeze(path: Path, expectations: FreezeExpectations) -> pd.DataFrame:
    if expectations == DEFAULT_EXPECTATIONS:
        plan = M3_15.load_plan(path)
    else:
        if not path.exists():
            raise FileNotFoundError(f"Missing shard plan: {path}")
        plan = pd.read_csv(path)
    if len(plan) != expectations.shard_count:
        raise ValueError(f"Expected {expectations.shard_count} shards, found {len(plan)}.")
    if int(plan["expected_edge_rows"].sum()) != expectations.total_edge_rows:
        raise ValueError("Shard plan expected edge rows do not match freeze expectations.")
    if bool((plan["source_time"].astype(str) == "D35").any()):
        raise ValueError("D35 must not be a source time for full M3.")
    if not bool((plan["candidate_k"].astype(int) == expectations.candidate_k).all()):
        raise ValueError(f"All shards must use candidate_k={expectations.candidate_k}.")
    observed_pairs = {
        (str(row["source_time"]), str(row["target_time"]))
        for _, row in plan[["source_time", "target_time"]].drop_duplicates().iterrows()
    }
    expected_pairs = set(expectations.expected_time_pairs)
    if observed_pairs != expected_pairs:
        raise ValueError(f"Time pairs {sorted(observed_pairs)} != expected {sorted(expected_pairs)}.")
    return plan.reset_index(drop=True)


def validate_plan_paths_under_output_root(plan: pd.DataFrame, output_root: Path) -> None:
    root = output_root.resolve()
    for column in ["output_parquet", "shard_report"]:
        for value in plan[column].astype(str):
            path = Path(value).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"Plan path is outside output root: {path}") from exc


def load_control_outputs(output_root: Path, reports_dir: Path, expectations: FreezeExpectations) -> dict[str, Any]:
    paths = {
        "completed_shards_csv": output_root / "completed_shards.csv",
        "failed_shards_txt": output_root / "failed_shards.txt",
        "full_m3_manifest_csv": output_root / "full_m3_manifest.csv",
        "full_m3_manifest_json": output_root / "full_m3_manifest.json",
        "m3_full_m3_run_summary_csv": reports_dir / "m3_full_m3_run_summary.csv",
        "m3_full_m3_run_summary_md": reports_dir / "m3_full_m3_run_summary.md",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required M3-15 control outputs: {missing}")

    failed_text = paths["failed_shards_txt"].read_text(encoding="utf-8")
    failed_lines = [line for line in failed_text.splitlines() if line.strip()]
    if failed_lines:
        raise RuntimeError(f"M3-15 failed_shards.txt is not empty: {failed_lines[:5]}")

    completed = pd.read_csv(paths["completed_shards_csv"])
    if len(completed) != expectations.shard_count:
        raise ValueError(f"completed_shards.csv has {len(completed)} rows, expected {expectations.shard_count}.")
    if int(completed["observed_edge_rows"].sum()) != expectations.total_edge_rows:
        raise ValueError("completed_shards.csv observed rows do not match freeze expectations.")

    summary = pd.read_csv(paths["m3_full_m3_run_summary_csv"])
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "completed": completed,
        "summary": summary,
        "failed_lines": failed_lines,
    }


def _completed_lookup(completed: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row["shard_id"]): row
        for row in completed.to_dict("records")
        if "shard_id" in row
    }


def validate_existing_shards_for_freeze(plan: pd.DataFrame, completed: pd.DataFrame) -> list[dict[str, Any]]:
    completed_by_id = _completed_lookup(completed)
    invalid: list[str] = []
    records: list[dict[str, Any]] = []
    for _, shard in plan.iterrows():
        valid, metrics, reason = M3_15.validate_existing_outputs(shard)
        shard_id = str(shard["shard_id"])
        if not valid:
            invalid.append(f"{shard_id}: {reason}")
            continue
        completed_row = completed_by_id.get(shard_id, {})
        record = {
            "shard_id": shard_id,
            "source_time": str(shard["source_time"]),
            "target_time": str(shard["target_time"]),
            "source_slice_id": str(shard["source_slice_id"]),
            "source_rows": int(shard["source_rows"]),
            "target_rows": int(shard["target_rows"]),
            "candidate_k": int(shard["candidate_k"]),
            "expected_edge_rows": int(shard["expected_edge_rows"]),
            "output_parquet": str(shard["output_parquet"]),
            "shard_report": str(shard["shard_report"]),
            "m3_15_status": str(completed_row.get("status", "unknown")),
            "m3_16_status": "FINAL_QC_VALIDATED",
            "runtime_seconds": float(completed_row.get("runtime_seconds", 0.0)),
            "max_rss_gib": float(completed_row.get("max_rss_gib", 0.0)),
            "backend": str(completed_row.get("backend", shard.get("selected_backend", "unknown"))),
            "tau_pair": float(completed_row.get("tau_pair", np.nan)),
            **metrics,
        }
        records.append(record)
    if invalid:
        raise RuntimeError("Invalid or incomplete full-M3 shards:\n" + "\n".join(invalid[:20]))
    return records


def records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if len(frame) and "collapse_warnings" in frame.columns:
        frame["collapse_warnings_text"] = frame["collapse_warnings"].apply(
            lambda values: "|".join(values) if isinstance(values, list) else str(values or "")
        )
    return frame


def summarize_time_pairs(records: list[dict[str, Any]], plan: pd.DataFrame) -> pd.DataFrame:
    frame = records_frame(records)
    if frame.empty:
        return pd.DataFrame()
    frame["has_collapse_warning"] = frame["collapse_warnings"].apply(lambda values: bool(values))
    frame["has_slice_collapse_warning"] = frame["collapse_warnings"].apply(
        lambda values: any("target-slice" in item for item in values) if isinstance(values, list) else False
    )
    frame["has_mouse_collapse_warning"] = frame["collapse_warnings"].apply(
        lambda values: any("target-mouse" in item for item in values) if isinstance(values, list) else False
    )
    summary = (
        frame.groupby(["source_time", "target_time"], observed=True)
        .agg(
            shards=("shard_id", "count"),
            observed_edge_rows=("observed_edge_rows", "sum"),
            expected_edge_rows=("expected_edge_rows", "sum"),
            source_anchors=("source_anchors_represented", "sum"),
            candidate_count_min=("candidate_count_min", "min"),
            candidate_count_max=("candidate_count_max", "max"),
            row_sum_min=("row_sum_min", "min"),
            row_sum_max=("row_sum_max", "max"),
            row_sum_abs_error_max=("row_sum_abs_error_max", "max"),
            row_entropy_mean=("row_entropy_mean", "mean"),
            row_entropy_median=("row_entropy_median", "median"),
            top1_probability_mean=("top1_probability_mean", "mean"),
            top1_probability_median=("top1_probability_median", "median"),
            target_slice_id_entropy_mean=("target_slice_id_entropy_mean", "mean"),
            target_mouse_id_entropy_mean=("target_mouse_id_entropy_mean", "mean"),
            top_target_slice_id_fraction_p95=("top_target_slice_id_fraction_p95", "max"),
            top_target_mouse_id_fraction_p95=("top_target_mouse_id_fraction_p95", "max"),
            collapse_warning_shards=("has_collapse_warning", "sum"),
            slice_collapse_warning_shards=("has_slice_collapse_warning", "sum"),
            mouse_collapse_warning_shards=("has_mouse_collapse_warning", "sum"),
            runtime_seconds=("runtime_seconds", "sum"),
            max_rss_gib=("max_rss_gib", "max"),
            output_bytes=("output_bytes", "sum"),
        )
        .reset_index()
    )
    summary["planned_shards_total"] = int(len(plan))
    summary["failed_shards"] = 0
    return summary


def validate_final_criteria(
    plan: pd.DataFrame,
    records: list[dict[str, Any]],
    control: dict[str, Any],
    expectations: FreezeExpectations,
) -> dict[str, Any]:
    frame = records_frame(records)
    criteria = {
        "completed_shards_eq_expected": int(len(records)) == expectations.shard_count,
        "failed_shards_eq_zero": len(control["failed_lines"]) == 0,
        "total_observed_edge_rows_eq_expected": int(frame["observed_edge_rows"].sum()) == expectations.total_edge_rows,
        "d35_source_shards_eq_zero": int((plan["source_time"].astype(str) == "D35").sum()) == 0,
        "all_expected_time_pairs_present": {
            (str(row["source_time"]), str(row["target_time"]))
            for _, row in plan[["source_time", "target_time"]].drop_duplicates().iterrows()
        }
        == set(expectations.expected_time_pairs),
        "candidate_k_eq_expected_for_every_source": bool(
            (frame["candidate_count_min"].astype(int) == expectations.candidate_k).all()
            and (frame["candidate_count_max"].astype(int) == expectations.candidate_k).all()
        ),
        "row_sums_approximately_one": bool(frame["row_sum_abs_error_max"].astype(float).max() <= expectations.row_sum_atol),
        "no_nan_or_inf_probabilities": int(frame["nonfinite_numeric_count"].sum()) == 0,
        "no_negative_probabilities": int(frame["negative_probability_count"].sum()) == 0,
        "raw_and_mass_adjusted_weights_present": bool(
            {"raw_edge_weight", "mass_adjusted_weight"} <= set(full_transition_schema_columns())
        ),
        "completed_shards_csv_exists": Path(control["paths"]["completed_shards_csv"]).exists(),
        "failed_shards_txt_exists_and_empty": Path(control["paths"]["failed_shards_txt"]).exists()
        and len(control["failed_lines"]) == 0,
        "full_m3_manifest_exists": Path(control["paths"]["full_m3_manifest_csv"]).exists()
        and Path(control["paths"]["full_m3_manifest_json"]).exists(),
        "m3_full_m3_run_summary_exists": Path(control["paths"]["m3_full_m3_run_summary_csv"]).exists()
        and Path(control["paths"]["m3_full_m3_run_summary_md"]).exists(),
    }
    failed = [name for name, ok in criteria.items() if not ok]
    if failed:
        raise RuntimeError(f"M3-16 final criteria failed: {failed}")
    return criteria


def inventory_figures(existing_dir: Path, final_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for figure_set, directory, names in [
        ("m3_15_existing", existing_dir, EXPECTED_EXISTING_FIGURES),
        ("m3_16_final", final_dir, FINAL_FIGURE_NAMES),
    ]:
        for name in names:
            path = directory / name
            rows.append(
                {
                    "figure_set": figure_set,
                    "figure_name": name,
                    "path": str(path),
                    "exists": path.exists(),
                    "bytes": int(path.stat().st_size) if path.exists() else 0,
                }
            )
    return pd.DataFrame(rows)


def _plot_bar(ax: Any, labels: pd.Series, values: pd.Series, title: str, ylabel: str) -> None:
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)


def generate_final_figures(final_dir: Path, records: list[dict[str, Any]], summary: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if not records or summary.empty:
        return warnings
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        final_dir.mkdir(parents=True, exist_ok=True)
        labels = summary["source_time"].astype(str) + "->" + summary["target_time"].astype(str)

        fig, ax = plt.subplots(figsize=(7, 4))
        _plot_bar(ax, labels, summary["observed_edge_rows"], "Final M3 edge rows", "rows")
        fig.tight_layout()
        atomic_save_figure(fig, final_dir / "m3_full_m3_final_rows_by_time_pair.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        _plot_bar(ax, labels, summary["row_sum_abs_error_max"], "Final M3 row-sum QC", "max abs error")
        fig.tight_layout()
        atomic_save_figure(fig, final_dir / "m3_full_m3_final_row_sum_qc_by_time_pair.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        _plot_bar(axes[0], labels, summary["row_entropy_mean"], "Row entropy mean", "entropy")
        _plot_bar(axes[1], labels, summary["top1_probability_mean"], "Top1 probability mean", "probability")
        fig.tight_layout()
        atomic_save_figure(fig, final_dir / "m3_full_m3_final_entropy_top1_by_time_pair.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        _plot_bar(axes[0], labels, summary["runtime_seconds"], "Runtime by time pair", "seconds")
        _plot_bar(axes[1], labels, summary["max_rss_gib"], "Peak RSS by time pair", "GiB")
        fig.tight_layout()
        atomic_save_figure(fig, final_dir / "m3_full_m3_final_runtime_memory_by_time_pair.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        _plot_bar(
            axes[0],
            labels,
            summary["slice_collapse_warning_shards"],
            "Target-slice collapse warnings",
            "shards",
        )
        _plot_bar(
            axes[1],
            labels,
            summary["mouse_collapse_warning_shards"],
            "Target-mouse collapse warnings",
            "shards",
        )
        fig.tight_layout()
        atomic_save_figure(fig, final_dir / "m3_full_m3_final_collapse_diagnostics_by_time_pair.png")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Final figure generation failed but M3-16 QC completed: {exc}")
    return warnings


def manifest_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = records_frame(records)
    columns = [
        "shard_id",
        "source_time",
        "target_time",
        "source_slice_id",
        "m3_15_status",
        "m3_16_status",
        "observed_edge_rows",
        "expected_edge_rows",
        "candidate_count_min",
        "candidate_count_max",
        "row_sum_abs_error_max",
        "row_entropy_mean",
        "top1_probability_mean",
        "top_target_slice_id_fraction_p95",
        "top_target_mouse_id_fraction_p95",
        "runtime_seconds",
        "max_rss_gib",
        "output_bytes",
        "output_parquet",
        "shard_report",
        "collapse_warnings_text",
    ]
    return frame[columns].sort_values("shard_id").reset_index(drop=True)


def schema_payload(criteria: dict[str, Any], figure_warnings: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "m3_full_m3_final_schema_v1",
        "edge_schema_version": "m3_full_transition_schema_v1",
        "edge_schema_columns": full_transition_schema_columns(),
        "final_freeze_criteria": criteria,
        "figure_warnings_are_warning_only": True,
        "figure_warnings": figure_warnings,
        **NO_DOWNSTREAM_FLAGS,
    }


def handoff_report(
    summary: pd.DataFrame,
    manifest: pd.DataFrame,
    figure_inventory: pd.DataFrame,
    criteria: dict[str, Any],
    disk_usage_bytes: int,
    figure_warnings: list[str],
    reports_dir: Path,
    final_figures_dir: Path,
) -> str:
    total_rows = int(summary["observed_edge_rows"].sum()) if len(summary) else 0
    total_runtime = float(summary["runtime_seconds"].sum()) if len(summary) else 0.0
    peak_rss = float(summary["max_rss_gib"].max()) if len(summary) else 0.0
    total_output_bytes = int(summary["output_bytes"].sum()) if len(summary) else 0
    collapse_shards = int((manifest["collapse_warnings_text"].astype(str) != "").sum()) if len(manifest) else 0
    missing_figures = figure_inventory[~figure_inventory["exists"]]

    lines = [
        "# M3-16 Full M3 Final QC Freeze And M4A Handoff",
        "",
        "M3-15 production full-M3 shard construction is complete and frozen for M4A planning.",
        "This stage reads existing local transition edge shards only; it does not build global transition objects.",
        "",
        "## Final Status",
        f"- completed shards: {len(manifest)}",
        "- failed shards: 0",
        f"- total observed edge rows: {total_rows}",
        f"- output disk usage bytes: {disk_usage_bytes}",
        f"- edge parquet bytes from shard records: {total_output_bytes}",
        f"- recorded runtime seconds: {total_runtime:.3f}",
        f"- peak max RSS GiB: {peak_rss:.4f}",
        f"- shards with warning-only collapse diagnostics: {collapse_shards}",
        "",
        "## Per-Time-Pair QC",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            "- "
            f"{row['source_time']}->{row['target_time']}: "
            f"shards={int(row['shards'])}, rows={int(row['observed_edge_rows'])}, "
            f"row_sum_abs_error_max={row['row_sum_abs_error_max']:.6g}, "
            f"entropy_mean={row['row_entropy_mean']:.6g}, "
            f"top1_mean={row['top1_probability_mean']:.6g}, "
            f"slice_collapse_warning_shards={int(row['slice_collapse_warning_shards'])}, "
            f"mouse_collapse_warning_shards={int(row['mouse_collapse_warning_shards'])}, "
            f"runtime_seconds={row['runtime_seconds']:.3f}, "
            f"max_rss_gib={row['max_rss_gib']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Freeze Artifacts",
            f"- final manifest CSV: {reports_dir / FINAL_MANIFEST_CSV}",
            f"- final manifest JSON: {reports_dir / FINAL_MANIFEST_JSON}",
            f"- final schema JSON: {reports_dir / FINAL_SCHEMA_JSON}",
            f"- final QC summary CSV: {reports_dir / FINAL_QC_SUMMARY_CSV}",
            f"- figure inventory CSV: {reports_dir / FINAL_FIGURE_INVENTORY_CSV}",
            f"- final figures directory: {final_figures_dir}",
            "",
            "## Downstream Boundary",
            "- no global Markov P was assembled",
            "- no GPCCA was run",
            "- no fate probability was computed",
            "- no Branched NicheFlow was run",
            "- no M5 was run",
            "- no regulator analysis was run",
            "",
            "## M4A Handoff",
            "M4A may plan global Markov P assembly from the frozen full-M3 local edge shards.",
            "M4A must treat `row_normalized_transition_prob` as local to each source candidate set until a separate global assembly stage explicitly constructs the global Markov object.",
            "",
            "## Criteria",
        ]
    )
    lines.extend([f"- {name}: {value}" for name, value in criteria.items()])
    if figure_warnings:
        lines.extend(["", "## Figure Warnings", *[f"- {warning}" for warning in figure_warnings]])
    if len(missing_figures):
        lines.extend(
            [
                "",
                "## Missing Figures",
                *[f"- {row['figure_set']}: {row['figure_name']}" for row in missing_figures.to_dict("records")],
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_freeze_outputs(
    plan: pd.DataFrame,
    output_root: Path,
    reports_dir: Path,
    existing_figures_dir: Path,
    final_figures_dir: Path,
    expectations: FreezeExpectations = DEFAULT_EXPECTATIONS,
    generate_figures: bool = True,
) -> dict[str, Any]:
    validate_plan_paths_under_output_root(plan, output_root)
    control = load_control_outputs(output_root, reports_dir, expectations)
    records = validate_existing_shards_for_freeze(plan, control["completed"])
    summary = summarize_time_pairs(records, plan)
    criteria = validate_final_criteria(plan, records, control, expectations)
    figure_warnings = generate_final_figures(final_figures_dir, records, summary) if generate_figures else []
    figure_inventory = inventory_figures(existing_figures_dir, final_figures_dir)
    disk_bytes = directory_size_bytes(output_root)
    manifest = manifest_frame(records)
    return {
        "generated_at_utc": utc_now_iso(),
        "records": records,
        "manifest": manifest,
        "summary": summary,
        "criteria": criteria,
        "control": control,
        "figure_inventory": figure_inventory,
        "figure_warnings": figure_warnings,
        "disk_usage_bytes": disk_bytes,
        "reports_dir": reports_dir,
        "final_figures_dir": final_figures_dir,
    }


def write_freeze_outputs(outputs: dict[str, Any]) -> dict[str, Path]:
    reports_dir = Path(outputs["reports_dir"])
    manifest = outputs["manifest"]
    summary = outputs["summary"]
    figure_inventory = outputs["figure_inventory"]
    criteria = outputs["criteria"]
    figure_warnings = outputs["figure_warnings"]
    disk_bytes = int(outputs["disk_usage_bytes"])

    paths = {
        "manifest_csv": reports_dir / FINAL_MANIFEST_CSV,
        "manifest_json": reports_dir / FINAL_MANIFEST_JSON,
        "schema_json": reports_dir / FINAL_SCHEMA_JSON,
        "summary_csv": reports_dir / FINAL_QC_SUMMARY_CSV,
        "handoff_md": reports_dir / FINAL_HANDOFF_MD,
        "figure_inventory_csv": reports_dir / FINAL_FIGURE_INVENTORY_CSV,
        "figure_inventory_json": reports_dir / FINAL_FIGURE_INVENTORY_JSON,
    }
    atomic_write_csv(paths["manifest_csv"], manifest)
    atomic_write_csv(paths["summary_csv"], summary)
    atomic_write_csv(paths["figure_inventory_csv"], figure_inventory)
    atomic_write_json(
        paths["manifest_json"],
        {
            "schema_version": "m3_full_m3_final_freeze_manifest_v1",
            "generated_at_utc": outputs["generated_at_utc"],
            "records": manifest.to_dict("records"),
            "disk_usage_bytes": disk_bytes,
            **NO_DOWNSTREAM_FLAGS,
        },
    )
    atomic_write_json(paths["schema_json"], schema_payload(criteria, figure_warnings))
    atomic_write_json(
        paths["figure_inventory_json"],
        {
            "schema_version": "m3_full_m3_final_figure_inventory_v1",
            "generated_at_utc": outputs["generated_at_utc"],
            "figures": figure_inventory.to_dict("records"),
            "figure_warnings": figure_warnings,
        },
    )
    atomic_write_text(
        paths["handoff_md"],
        handoff_report(
            summary,
            manifest,
            figure_inventory,
            criteria,
            disk_bytes,
            figure_warnings,
            reports_dir,
            Path(outputs["final_figures_dir"]),
        ),
    )
    return paths


def run(args: argparse.Namespace) -> int:
    start = time.monotonic()
    config = load_config(args.config)
    reports_dir = args.reports_dir or Path(config["paths"]["reports_dir"])
    existing_figures_dir = args.existing_figures_dir or reports_dir / "figures" / "full_m3"
    final_figures_dir = args.final_figures_dir or reports_dir / "figures" / "full_m3_final"

    for label, path in [
        ("output root", args.output_root),
        ("reports dir", reports_dir),
        ("existing figures dir", existing_figures_dir),
        ("final figures dir", final_figures_dir),
    ]:
        assert_no_ssd_path(Path(path), label)
    if bool(config["paths"].get("use_ssd", False)):
        raise RuntimeError("Refusing M3-16 while paths.use_ssd is true.")
    if config["full_m3"].get("write_global_kernel"):
        raise RuntimeError("Refusing M3-16 while full_m3.write_global_kernel is true.")

    plan = load_plan_for_freeze(args.plan_csv, DEFAULT_EXPECTATIONS)
    outputs = build_freeze_outputs(
        plan=plan,
        output_root=args.output_root,
        reports_dir=reports_dir,
        existing_figures_dir=existing_figures_dir,
        final_figures_dir=final_figures_dir,
        expectations=DEFAULT_EXPECTATIONS,
        generate_figures=True,
    )
    written = write_freeze_outputs(outputs)
    runtime = time.monotonic() - start

    print("M3_16_FULL_M3_FINAL_FREEZE_COMPLETE")
    print(f"VALIDATED_SHARDS {len(outputs['manifest'])}")
    print(f"FAILED_SHARDS 0")
    print(f"TOTAL_OBSERVED_EDGE_ROWS {int(outputs['summary']['observed_edge_rows'].sum())}")
    print(f"DISK_USAGE_BYTES {int(outputs['disk_usage_bytes'])}")
    print(f"TOTAL_RUNTIME_SECONDS {runtime:.3f}")
    print(f"HANDOFF_REPORT {written['handoff_md']}")
    print("NO_GLOBAL_MARKOV_P True")
    print("NO_GPCCA True")
    print("NO_FATE_PROBABILITY True")
    print("NO_BRANCHED_NICHEFLOW True")
    print("NO_M5 True")
    print("NO_REGULATOR_ANALYSIS True")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
