#!/usr/bin/env python
"""Run a bounded M3 transition pilot for one adjacent time pair."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any


def _requested_blas_threads(argv: list[str]) -> str:
    for idx, value in enumerate(argv):
        if value == "--blas-threads" and idx + 1 < len(argv):
            return str(argv[idx + 1])
        if value.startswith("--blas-threads="):
            return value.split("=", 1)[1]
    return "1"


for _thread_var in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[_thread_var] = _requested_blas_threads(sys.argv)

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from nichefate.io import load_config
from nichefate.transition import full_transition_schema_columns, matrix_memory_gb

_PILOT_SPEC = importlib.util.spec_from_file_location(
    "m3_pilot_shard",
    PROJECT_ROOT / "scripts" / "m3_05_build_transition_pilot_shard.py",
)
if _PILOT_SPEC is None or _PILOT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("Could not load m3_05_build_transition_pilot_shard.py")
_PILOT = importlib.util.module_from_spec(_PILOT_SPEC)
_PILOT_SPEC.loader.exec_module(_PILOT)

REQUIRED_PLAN_COLUMNS = {
    "source_time",
    "target_time",
    "source_day",
    "target_day",
    "time_delta",
    "source_slice_id",
    "source_slice_file",
    "source_rows",
    "target_time_rows",
    "target_slice_count",
    "candidate_k",
    "expected_edge_rows",
}

LIGHTWEIGHT_EDGE_COLUMNS = [
    "source_anchor_id",
    "target_anchor_id",
    "source_time",
    "target_time",
    "source_day",
    "target_day",
    "time_delta",
    "source_slice_id",
    "target_slice_id",
    "source_slice_file",
    "target_slice_file",
    "source_mouse_id",
    "target_mouse_id",
    "combined_cost",
    "raw_edge_weight",
    "mass_adjusted_weight",
    "row_normalized_transition_prob",
    "tau_pair",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_transition_kernel.yaml")
    parser.add_argument("--source-time", required=True)
    parser.add_argument("--target-time", required=True)
    parser.add_argument("--plan-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--backend", default="sklearn_exact")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--max-memory-gb-warning", type=float)
    parser.add_argument("--blas-threads", type=int, default=1)
    return parser.parse_args()


def token(value: object) -> str:
    return _PILOT._safe_token(value)


def pair_stem(source_time: str, target_time: str) -> str:
    return f"{token(source_time)}_to_{token(target_time)}"


def shard_paths(output_dir: Path, source_time: str, target_time: str, source_slice_id: str) -> dict[str, Path]:
    return _PILOT.output_paths(output_dir, source_time, target_time, source_slice_id)


def timepair_paths(output_dir: Path, source_time: str, target_time: str) -> dict[str, Path]:
    stem = pair_stem(source_time, target_time)
    return {
        "manifest_csv": output_dir / f"timepair_manifest_{stem}.csv",
        "manifest_json": output_dir / f"timepair_manifest_{stem}.json",
        "qc_csv": output_dir / f"timepair_qc_summary_{stem}.csv",
        "qc_json": output_dir / f"timepair_qc_summary_{stem}.json",
        "report": output_dir / f"timepair_report_{stem}.md",
        "shard_qc_table": output_dir / f"plot_table_shard_qc_{stem}.csv",
        "slice_flow_table": output_dir / f"plot_table_slice_flow_{stem}.csv",
        "mouse_flow_table": output_dir / f"plot_table_mouse_flow_{stem}.csv",
    }


def figure_paths(output_dir: Path, source_time: str, target_time: str) -> dict[str, Path]:
    stem = pair_stem(source_time, target_time)
    fig_dir = output_dir / "figures"
    return {
        "runtime_memory": fig_dir / f"m3_{stem}_shard_runtime_memory.png",
        "edge_qc": fig_dir / f"m3_{stem}_edge_qc_distributions.png",
        "collapse": fig_dir / f"m3_{stem}_batch_collapse_diagnostics.png",
        "source_qc": fig_dir / f"m3_{stem}_source_slice_qc_heatmap.png",
        "slice_flow": fig_dir / f"m3_{stem}_slice_flow_heatmap.png",
        "mouse_flow": fig_dir / f"m3_{stem}_mouse_flow_heatmap.png",
        "dashboard": fig_dir / f"m3_{stem}_summary_dashboard.png",
    }


def filter_timepair_plan(
    shards: pd.DataFrame,
    source_time: str,
    target_time: str,
    candidate_k: int,
    max_shards: int | None = None,
) -> pd.DataFrame:
    missing = sorted(REQUIRED_PLAN_COLUMNS - set(shards.columns))
    if missing:
        raise KeyError(f"Plan CSV is missing required columns: {missing}")
    selected = shards[
        (shards["source_time"].astype(str) == str(source_time))
        & (shards["target_time"].astype(str) == str(target_time))
    ].copy()
    if selected.empty:
        raise ValueError(f"No shards found for requested time pair {source_time}->{target_time}.")
    if set(selected["source_time"].astype(str)) != {str(source_time)}:
        raise ValueError("Filtered plan contains an unexpected source_time.")
    if set(selected["target_time"].astype(str)) != {str(target_time)}:
        raise ValueError("Filtered plan contains an unexpected target_time.")
    if not bool((selected["candidate_k"].astype(int) == int(candidate_k)).all()):
        raise ValueError("Requested candidate_k does not match the design table.")
    required_nonnull = ["source_slice_id", "source_slice_file", "source_rows", "expected_edge_rows"]
    if int(selected[required_nonnull].isna().sum().sum()):
        raise ValueError("Filtered plan has missing required shard fields.")
    expected = selected["source_rows"].astype(int) * int(candidate_k)
    if not bool((expected == selected["expected_edge_rows"].astype(int)).all()):
        raise ValueError("Plan expected_edge_rows does not equal source_rows x candidate_k.")
    selected = selected.sort_values("source_slice_id").reset_index(drop=True)
    if max_shards is not None:
        selected = selected.head(int(max_shards)).copy()
    return selected


def load_time_pair(config: dict[str, Any], source_time: str, target_time: str) -> dict[str, Any]:
    reports_dir = Path(config["paths"]["reports_dir"])
    pairs = json.loads((reports_dir / "m3_time_pairs.json").read_text(encoding="utf-8"))
    for pair in pairs:
        if str(pair["source_time"]) == str(source_time) and str(pair["target_time"]) == str(target_time):
            return pair
    raise ValueError(f"Missing time-pair metadata for {source_time}->{target_time}.")


def feature_columns(config: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    reports_dir = Path(config["paths"]["reports_dir"])
    feature_groups = json.loads((reports_dir / "m3_feature_groups.json").read_text(encoding="utf-8"))
    retrieval = [
        column
        for group in config["full_m3"]["retrieval_feature_groups"]
        for column in feature_groups["feature_groups"][group]
    ]
    rerank = [
        column
        for group in config["full_m3"]["rerank_feature_groups"]
        for column in feature_groups["feature_groups"][group]
    ]
    read_columns = list(dict.fromkeys(config["input"]["metadata_columns"] + retrieval + rerank))
    return feature_groups, retrieval, rerank, read_columns


def estimate_memory(
    plan: pd.DataFrame,
    retrieval_dimensions: int,
    rerank_dimensions: int,
    max_memory_gb_warning: float,
) -> dict[str, float]:
    max_source_rows = int(plan["source_rows"].max())
    target_rows = int(plan["target_time_rows"].iloc[0])
    target_retrieval = matrix_memory_gb(target_rows, retrieval_dimensions)
    target_rerank = matrix_memory_gb(target_rows, rerank_dimensions)
    source_shard = matrix_memory_gb(max_source_rows, retrieval_dimensions) + matrix_memory_gb(
        max_source_rows,
        rerank_dimensions,
    )
    per_worker = target_retrieval + target_rerank + source_shard
    safe_concurrency = max(1, int(max_memory_gb_warning // per_worker)) if per_worker else 1
    return {
        "target_retrieval_matrix_gib": target_retrieval,
        "target_rerank_matrix_gib": target_rerank,
        "source_shard_matrix_gib": source_shard,
        "approx_per_worker_memory_gib": per_worker,
        "safe_single_node_concurrency": float(safe_concurrency),
        "max_memory_gb_warning": float(max_memory_gb_warning),
    }


def finite_string_series(series: pd.Series) -> pd.Series:
    values = series.astype("string")
    lowered = values.str.lower()
    return values.notna() & ~lowered.isin(["", "nan", "none", "null", "<na>"])


def mouse_metadata_complete(frame: pd.DataFrame) -> bool:
    if "source_mouse_id" not in frame or "target_mouse_id" not in frame:
        return False
    return bool(finite_string_series(frame["source_mouse_id"]).all() and finite_string_series(frame["target_mouse_id"]).all())


def _entropy_and_top_fraction(grouped: pd.core.groupby.generic.SeriesGroupBy) -> tuple[pd.Series, pd.Series]:
    entropies: dict[str, float] = {}
    top_fractions: dict[str, float] = {}
    for key, values in grouped:
        probs = values.astype(str).value_counts(normalize=True).to_numpy(dtype=float)
        entropies[str(key)] = float(-(probs * np.log(np.clip(probs, 1e-300, None))).sum()) if len(probs) else 0.0
        top_fractions[str(key)] = float(probs.max()) if len(probs) else 0.0
    return pd.Series(entropies, dtype=float), pd.Series(top_fractions, dtype=float)


def source_anchor_qc(frame: pd.DataFrame) -> pd.DataFrame:
    prob = frame["row_normalized_transition_prob"].astype(float)
    work = frame[["source_anchor_id", "source_slice_id", "target_slice_id"]].copy()
    work["source_mouse_id"] = frame["source_mouse_id"].astype(str) if "source_mouse_id" in frame else ""
    work["target_mouse_id"] = frame["target_mouse_id"].astype(str) if "target_mouse_id" in frame else ""
    work["_entropy_term"] = -(prob.clip(lower=1e-300) * np.log(prob.clip(lower=1e-300)))
    grouped = work.groupby("source_anchor_id", observed=True)
    row_entropy = grouped["_entropy_term"].sum()
    top1 = frame.groupby("source_anchor_id", observed=True)["row_normalized_transition_prob"].max()
    target_slice_entropy, top_target_slice = _entropy_and_top_fraction(grouped["target_slice_id"])
    if mouse_metadata_complete(frame):
        target_mouse_entropy, top_target_mouse = _entropy_and_top_fraction(grouped["target_mouse_id"])
    else:
        target_mouse_entropy = pd.Series(np.nan, index=row_entropy.index, dtype=float)
        top_target_mouse = pd.Series(np.nan, index=row_entropy.index, dtype=float)
    metadata = grouped[["source_slice_id", "source_mouse_id"]].first()
    return pd.DataFrame(
        {
            "source_anchor_id": row_entropy.index,
            "source_slice_id": metadata["source_slice_id"].astype(str).to_numpy(),
            "source_mouse_id": metadata["source_mouse_id"].astype(str).to_numpy(),
            "row_entropy": row_entropy.to_numpy(dtype=float),
            "top1_probability": top1.reindex(row_entropy.index).to_numpy(dtype=float),
            "target_slice_entropy": target_slice_entropy.reindex(row_entropy.index).to_numpy(dtype=float),
            "top_target_slice_fraction": top_target_slice.reindex(row_entropy.index).to_numpy(dtype=float),
            "target_mouse_entropy": target_mouse_entropy.reindex(row_entropy.index).to_numpy(dtype=float),
            "top_target_mouse_fraction": top_target_mouse.reindex(row_entropy.index).to_numpy(dtype=float),
        }
    )


def _series_summary(values: pd.Series, prefix: str) -> dict[str, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_p95": np.nan,
        }
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_median": float(finite.median()),
        f"{prefix}_p05": float(finite.quantile(0.05)),
        f"{prefix}_p95": float(finite.quantile(0.95)),
    }


def edge_flow_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    slice_flow = (
        frame.groupby(["source_slice_id", "target_slice_id"], observed=True)
        .agg(
            edge_count=("row_normalized_transition_prob", "size"),
            edge_mass=("row_normalized_transition_prob", "sum"),
        )
        .reset_index()
    )
    if mouse_metadata_complete(frame):
        mouse_flow = (
            frame.groupby(["source_mouse_id", "target_mouse_id"], observed=True)
            .agg(
                edge_count=("row_normalized_transition_prob", "size"),
                edge_mass=("row_normalized_transition_prob", "sum"),
            )
            .reset_index()
        )
    else:
        mouse_flow = pd.DataFrame(columns=["source_mouse_id", "target_mouse_id", "edge_count", "edge_mass"])
        warnings.append("Mouse metadata is missing or incomplete; mouse-flow diagnostics were skipped for this shard.")
    return slice_flow, mouse_flow, warnings


def compute_edge_qc(
    frame: pd.DataFrame,
    shard: dict[str, Any],
    backend: str,
    runtime_seconds: float,
    max_rss_kb: int,
    output_size_bytes: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    numeric = frame.select_dtypes(include=[np.number, "bool"])
    probabilities = frame["row_normalized_transition_prob"].astype(float)
    counts = frame.groupby("source_anchor_id", observed=True).size()
    row_sums = frame.groupby("source_anchor_id", observed=True)["row_normalized_transition_prob"].sum()
    anchor_qc = source_anchor_qc(frame)
    slice_flow, mouse_flow, warnings = edge_flow_tables(frame)
    qc: dict[str, Any] = {
        "status": "PASS",
        "source_time": shard["source_time"],
        "target_time": shard["target_time"],
        "source_slice_id": shard["source_slice_id"],
        "source_slice_file": shard["source_slice_file"],
        "source_rows": int(shard["source_rows"]),
        "target_rows": int(shard["target_time_rows"]),
        "expected_edge_rows": int(shard["expected_edge_rows"]),
        "observed_edge_rows": int(len(frame)),
        "candidate_k": int(shard["candidate_k"]),
        "backend": backend,
        "runtime_seconds": float(runtime_seconds),
        "max_rss_gib": float(max_rss_kb / 1024 / 1024),
        "output_size_bytes": int(output_size_bytes),
        "tau_pair": float(pd.to_numeric(frame["tau_pair"], errors="coerce").dropna().median()),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "row_sum_abs_error_max": float(np.abs(row_sums.to_numpy(dtype=float) - 1.0).max()),
        "source_anchors_represented": int(counts.shape[0]),
        "candidate_count_min": int(counts.min()),
        "candidate_count_max": int(counts.max()),
        "candidate_count_mean": float(counts.mean()),
        "n_nan": int(numeric.isna().sum().sum()),
        "n_inf": int(np.isinf(numeric.to_numpy(dtype=float)).sum()),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
    }
    qc.update(_series_summary(anchor_qc["row_entropy"], "row_entropy"))
    qc.update(_series_summary(anchor_qc["top1_probability"], "top1_probability"))
    qc.update(_series_summary(anchor_qc["target_slice_entropy"], "target_slice_entropy"))
    qc.update(_series_summary(anchor_qc["top_target_slice_fraction"], "top_target_slice_fraction"))
    qc.update(_series_summary(anchor_qc["target_mouse_entropy"], "target_mouse_entropy"))
    qc.update(_series_summary(anchor_qc["top_target_mouse_fraction"], "top_target_mouse_fraction"))
    warnings.extend(collapse_warnings(qc))
    return qc, anchor_qc, slice_flow, mouse_flow, warnings


def collapse_warnings(qc: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if float(qc.get("top_target_slice_fraction_p95", 0.0) or 0.0) >= 0.9:
        warnings.append("High target-slice concentration warning.")
    if float(qc.get("target_slice_entropy_mean", 1.0) or 0.0) <= 0.1:
        warnings.append("Low target-slice entropy warning.")
    mouse_top = qc.get("top_target_mouse_fraction_p95")
    mouse_entropy = qc.get("target_mouse_entropy_mean")
    if pd.notna(mouse_top) and float(mouse_top) >= 0.9:
        warnings.append("High target-mouse concentration warning.")
    if pd.notna(mouse_entropy) and float(mouse_entropy) <= 0.1:
        warnings.append("Low target-mouse entropy warning.")
    return warnings


def required_qc_passes(qc: dict[str, Any]) -> tuple[bool, str]:
    if int(qc["observed_edge_rows"]) != int(qc["expected_edge_rows"]):
        return False, "observed_edge_rows does not match expected_edge_rows"
    if int(qc["candidate_count_min"]) != int(qc["candidate_k"]) or int(qc["candidate_count_max"]) != int(qc["candidate_k"]):
        return False, "candidate count is not equal to candidate_k"
    if float(qc["row_sum_abs_error_max"]) > 1e-6:
        return False, "row sums are outside tolerance"
    if int(qc["n_nan"]) or int(qc["n_inf"]):
        return False, "NaN or infinite numeric values detected"
    if float(qc["probability_min"]) < -1e-12:
        return False, "negative transition probabilities detected"
    return True, ""


def read_edge_required_columns(edge_path: Path) -> pd.DataFrame:
    return pd.read_parquet(edge_path, columns=LIGHTWEIGHT_EDGE_COLUMNS)


def validate_existing_output(
    edge_path: Path,
    report_path: Path,
    shard: dict[str, Any],
    backend: str,
) -> tuple[bool, dict[str, Any] | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, list[str], str]:
    if not edge_path.exists() or not report_path.exists():
        return False, None, None, None, None, [], "missing parquet or report"
    try:
        frame = read_edge_required_columns(edge_path)
        metadata_cols = [
            "source_time",
            "target_time",
            "time_delta",
            "source_slice_id",
            "target_slice_id",
            "source_slice_file",
            "target_slice_file",
            "source_mouse_id",
            "target_mouse_id",
        ]
        if int(frame[metadata_cols].isna().sum().sum()):
            return False, None, None, None, None, [], "missing required source/target metadata"
        if not bool((frame["source_time"].astype(str) == str(shard["source_time"])).all()):
            return False, None, None, None, None, [], "source_time mismatch"
        if not bool((frame["target_time"].astype(str) == str(shard["target_time"])).all()):
            return False, None, None, None, None, [], "target_time mismatch"
        if not bool(np.allclose(frame["time_delta"].to_numpy(dtype=float), float(shard["time_delta"]))):
            return False, None, None, None, None, [], "time_delta mismatch"
        qc, anchor_qc, slice_flow, mouse_flow, warnings = compute_edge_qc(
            frame,
            shard,
            backend=backend,
            runtime_seconds=0.0,
            max_rss_kb=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            output_size_bytes=edge_path.stat().st_size,
        )
        qc["status"] = "SKIPPED_RESUME"
        ok, reason = required_qc_passes(qc)
        return ok, qc, anchor_qc, slice_flow, mouse_flow, warnings, reason
    except Exception as exc:  # noqa: BLE001
        return False, None, None, None, None, [], str(exc)


def write_shard_report(path: Path, qc: dict[str, Any], warnings: list[str]) -> None:
    lines = [
        "# M3 Time-Pair Pilot Shard Report",
        "",
        "This report covers one source-slice edge shard inside a bounded time-pair pilot.",
        "`row_normalized_transition_prob` is local to each source niche candidate set and is not a global Markov transition probability.",
        "This shard does not build global Markov P, GPCCA, fate probabilities, Branched NicheFlow, M5, or regulator outputs.",
        "",
    ]
    for key in [
        "status",
        "source_time",
        "target_time",
        "source_slice_id",
        "source_slice_file",
        "source_rows",
        "target_rows",
        "expected_edge_rows",
        "observed_edge_rows",
        "candidate_k",
        "backend",
        "runtime_seconds",
        "max_rss_gib",
        "output_size_bytes",
        "tau_pair",
        "row_sum_min",
        "row_sum_max",
        "row_sum_abs_error_max",
        "row_entropy_mean",
        "top1_probability_mean",
        "target_slice_entropy_mean",
        "top_target_slice_fraction_p95",
        "target_mouse_entropy_mean",
        "top_target_mouse_fraction_p95",
    ]:
        lines.append(f"- {key}: {qc.get(key)}")
    if warnings:
        lines.extend(["", "## Warnings", *[f"- {warning}" for warning in warnings]])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def build_manifest_row(
    shard: dict[str, Any],
    edge_path: Path,
    report_path: Path,
    status: str,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "source_time": shard["source_time"],
        "target_time": shard["target_time"],
        "source_slice_id": shard["source_slice_id"],
        "source_slice_file": shard["source_slice_file"],
        "source_rows": int(shard["source_rows"]),
        "target_rows": int(shard["target_time_rows"]),
        "candidate_k": int(shard["candidate_k"]),
        "expected_edge_rows": int(shard["expected_edge_rows"]),
        "status": status,
        "reason": reason,
        "edge_path": str(edge_path),
        "report_path": str(report_path),
    }


def plot_heatmap(ax: Any, table: pd.DataFrame, x: str, y: str, value: str, title: str) -> None:
    pivot = table.pivot_table(index=y, columns=x, values=value, aggfunc="sum", fill_value=0.0)
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def generate_figures(
    output_dir: Path,
    source_time: str,
    target_time: str,
    qc_df: pd.DataFrame,
    anchor_qc_df: pd.DataFrame,
    slice_flow: pd.DataFrame,
    mouse_flow: pd.DataFrame,
) -> list[str]:
    warnings: list[str] = []
    paths = figure_paths(output_dir, source_time, target_time)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(qc_df["source_slice_id"].astype(str), qc_df["runtime_seconds"].astype(float))
        axes[0].set_title("Runtime seconds")
        axes[0].tick_params(axis="x", rotation=45)
        axes[1].bar(qc_df["source_slice_id"].astype(str), qc_df["max_rss_gib"].astype(float))
        axes[1].set_title("Max RSS GiB")
        axes[1].tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(paths["runtime_memory"], dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, column in zip(
            axes.ravel(),
            ["row_entropy", "top1_probability", "target_slice_entropy", "target_mouse_entropy"],
            strict=True,
        ):
            values = pd.to_numeric(anchor_qc_df[column], errors="coerce").dropna()
            ax.hist(values, bins=40)
            ax.set_title(column)
        fig.tight_layout()
        fig.savefig(paths["edge_qc"], dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, column in zip(
            axes,
            ["top_target_slice_fraction", "top_target_mouse_fraction"],
            strict=True,
        ):
            values = pd.to_numeric(anchor_qc_df[column], errors="coerce").dropna()
            ax.hist(values, bins=40)
            ax.set_title(column)
        fig.tight_layout()
        fig.savefig(paths["collapse"], dpi=140)
        plt.close(fig)

        heat_columns = [
            "observed_edge_rows",
            "runtime_seconds",
            "max_rss_gib",
            "row_entropy_mean",
            "top1_probability_mean",
            "top_target_slice_fraction_mean",
            "top_target_mouse_fraction_mean",
        ]
        heat = qc_df.set_index("source_slice_id")[heat_columns].astype(float)
        normalized = (heat - heat.min()) / (heat.max() - heat.min()).replace(0, 1)
        fig, ax = plt.subplots(figsize=(9, 4))
        image = ax.imshow(normalized.to_numpy(dtype=float), aspect="auto")
        ax.set_title("Source slice QC")
        ax.set_xticks(range(len(heat_columns)))
        ax.set_xticklabels(heat_columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(normalized.index)))
        ax.set_yticklabels(normalized.index, fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(paths["source_qc"], dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4))
        plot_heatmap(ax, slice_flow, "target_slice_id", "source_slice_id", "edge_mass", "Slice flow mass")
        fig.tight_layout()
        fig.savefig(paths["slice_flow"], dpi=140)
        plt.close(fig)

        if mouse_flow.empty:
            warnings.append("Mouse-flow figure skipped because mouse metadata is missing or incomplete.")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_heatmap(ax, mouse_flow, "target_mouse_id", "source_mouse_id", "edge_mass", "Mouse flow mass")
            fig.tight_layout()
            fig.savefig(paths["mouse_flow"], dpi=140)
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].bar(qc_df["source_slice_id"].astype(str), qc_df["runtime_seconds"].astype(float))
        axes[0, 0].set_title("Runtime")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 1].hist(pd.to_numeric(anchor_qc_df["row_entropy"], errors="coerce").dropna(), bins=40)
        axes[0, 1].set_title("Row entropy")
        axes[1, 0].hist(pd.to_numeric(anchor_qc_df["top1_probability"], errors="coerce").dropna(), bins=40)
        axes[1, 0].set_title("Top1 probability")
        axes[1, 1].hist(pd.to_numeric(anchor_qc_df["top_target_slice_fraction"], errors="coerce").dropna(), bins=40)
        axes[1, 1].set_title("Top target slice fraction")
        fig.tight_layout()
        fig.savefig(paths["dashboard"], dpi=140)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Visualization failed but QC passed: {exc}")
    return warnings


def write_timepair_outputs(
    output_dir: Path,
    source_time: str,
    target_time: str,
    manifest_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    anchor_qc_rows: list[pd.DataFrame],
    slice_flows: list[pd.DataFrame],
    mouse_flows: list[pd.DataFrame],
    memory: dict[str, float],
    warnings: list[str],
    started_at: float,
    dry_run: bool = False,
) -> list[str]:
    paths = timepair_paths(output_dir, source_time, target_time)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows)
    qc_df = pd.DataFrame(qc_rows)
    manifest.to_csv(paths["manifest_csv"], index=False)
    write_json(paths["manifest_json"], manifest.to_dict(orient="records"))
    if not qc_df.empty:
        qc_df.to_csv(paths["qc_csv"], index=False)
        qc_df.to_csv(paths["shard_qc_table"], index=False)
        write_json(paths["qc_json"], qc_df.to_dict(orient="records"))
    anchor_qc_df = pd.concat(anchor_qc_rows, ignore_index=True) if anchor_qc_rows else pd.DataFrame()
    slice_flow = pd.concat(slice_flows, ignore_index=True) if slice_flows else pd.DataFrame()
    mouse_flow = pd.concat(mouse_flows, ignore_index=True) if mouse_flows else pd.DataFrame()
    if not slice_flow.empty:
        slice_flow = (
            slice_flow.groupby(["source_slice_id", "target_slice_id"], observed=True)
            .agg(edge_count=("edge_count", "sum"), edge_mass=("edge_mass", "sum"))
            .reset_index()
        )
        slice_flow.to_csv(paths["slice_flow_table"], index=False)
    if not mouse_flow.empty:
        mouse_flow = (
            mouse_flow.groupby(["source_mouse_id", "target_mouse_id"], observed=True)
            .agg(edge_count=("edge_count", "sum"), edge_mass=("edge_mass", "sum"))
            .reset_index()
        )
        mouse_flow.to_csv(paths["mouse_flow_table"], index=False)
    elif not dry_run:
        warnings.append("Mouse-flow table skipped because mouse metadata is missing or incomplete.")
    if not dry_run and not qc_df.empty and not anchor_qc_df.empty and not slice_flow.empty:
        warnings.extend(generate_figures(output_dir, source_time, target_time, qc_df, anchor_qc_df, slice_flow, mouse_flow))
    report = timepair_report(source_time, target_time, manifest, qc_df, memory, warnings, time.monotonic() - started_at)
    paths["report"].write_text(report, encoding="utf-8")
    return warnings


def timepair_report(
    source_time: str,
    target_time: str,
    manifest: pd.DataFrame,
    qc_df: pd.DataFrame,
    memory: dict[str, float],
    warnings: list[str],
    runtime_seconds: float,
) -> str:
    expected_edges = int(manifest["expected_edge_rows"].sum()) if "expected_edge_rows" in manifest else 0
    observed_edges = int(qc_df["observed_edge_rows"].sum()) if not qc_df.empty else 0
    target_pool = int(manifest["target_rows"].iloc[0]) if not manifest.empty else 0
    candidate_k = int(manifest["candidate_k"].iloc[0]) if not manifest.empty else 0
    lines = [
        f"# M3 Time-Pair Pilot Report: {source_time} -> {target_time}",
        "",
        "This stage constructs local candidate edge shards for one time pair only.",
        "`row_normalized_transition_prob` is local to a source niche candidate set and is not a global Markov transition matrix P.",
        "Slice-flow and mouse-flow heatmaps are diagnostic summaries only; they do not assemble, normalize, or validate global P.",
        "No GPCCA, fate probability, Branched NicheFlow, M5, or regulator analysis was run.",
        "",
        "## Status",
        f"- Shards planned: {len(manifest)}",
        f"- Shards completed: {int((manifest['status'] == 'COMPLETED').sum()) if 'status' in manifest else 0}",
        f"- Shards skipped by resume: {int((manifest['status'] == 'SKIPPED_RESUME').sum()) if 'status' in manifest else 0}",
        f"- Shards failed: {int((manifest['status'] == 'FAILED').sum()) if 'status' in manifest else 0}",
        f"- Expected edge rows: {expected_edges}",
        f"- Observed edge rows: {observed_edges}",
        f"- Total runtime seconds: {runtime_seconds:.3f}",
        "",
        "## Fixed-K Diagnostics",
        f"- Candidate K: {candidate_k}",
        f"- Target pool size: {target_pool}",
        f"- K / target_pool_size: {(candidate_k / target_pool) if target_pool else 0:.8f}",
        f"- Expected candidate edge density: {(candidate_k / target_pool) if target_pool else 0:.8f}",
        "- Fixed K can induce density bias when target-time niche density varies.",
        "- KNN kth-distance distribution should be checked in a future pilot.",
        "- Target-slice entropy is a batch-effect diagnostic, not an automatic failure.",
        "",
        "## Compute And Memory Risk",
        "- Exact KNN complexity: O(N_source x N_target x D_retrieval).",
        "- sklearn_exact is acceptable for sampled preflight and small/medium pilot shards.",
        "- sklearn_exact may become a bottleneck for larger D3->D9 and D9->D21 execution.",
        "- FAISS, hnswlib, and pynndescent remain future ANN backend options.",
        "- No new dependencies were added in this stage.",
        "- Sampled-preflight extrapolation is approximate and may underestimate exact KNN runtime at full target-pool scale.",
        f"- Target retrieval matrix GiB: {memory['target_retrieval_matrix_gib']:.4f}",
        f"- Target rerank matrix GiB: {memory['target_rerank_matrix_gib']:.4f}",
        f"- Source shard matrix GiB: {memory['source_shard_matrix_gib']:.4f}",
        f"- Approx per-worker matrix memory GiB: {memory['approx_per_worker_memory_gib']:.4f}",
        f"- Safe single-node concurrency under warning threshold: {int(memory['safe_single_node_concurrency'])}",
        "- Current recommended execution mode remains sequential.",
        "- Python multiprocessing is not recommended initially because target matrices may be duplicated per worker.",
        "- Future full execution should prefer Slurm/job-array style execution with an explicit concurrency cap.",
        "- Future optimization may use memmap or shared target matrices.",
    ]
    if not qc_df.empty:
        lines.extend(
            [
                "",
                "## QC Summary",
                f"- Peak max RSS GiB: {float(qc_df['max_rss_gib'].max()):.4f}",
                f"- Row sum abs error max: {float(qc_df['row_sum_abs_error_max'].max()):.6g}",
                f"- Row entropy mean across shards: {float(qc_df['row_entropy_mean'].mean()):.6g}",
                f"- Top1 probability mean across shards: {float(qc_df['top1_probability_mean'].mean()):.6g}",
                f"- Top target slice fraction p95 max: {float(qc_df['top_target_slice_fraction_p95'].max()):.6g}",
            ]
        )
        if "top_target_mouse_fraction_p95" in qc_df:
            mouse = pd.to_numeric(qc_df["top_target_mouse_fraction_p95"], errors="coerce")
            if mouse.notna().any():
                lines.append(f"- Top target mouse fraction p95 max: {float(mouse.max()):.6g}")
    if warnings:
        lines.extend(["", "## Warnings", *[f"- {warning}" for warning in warnings]])
    return "\n".join(lines).rstrip() + "\n"


def print_dry_run(plan: pd.DataFrame, memory: dict[str, float], source_time: str, target_time: str) -> None:
    print("DRY_RUN True")
    print(f"TIME_PAIR {source_time}->{target_time}")
    print(f"PLANNED_SHARDS {len(plan)}")
    print(f"EXPECTED_EDGE_ROWS {int(plan['expected_edge_rows'].sum())}")
    print(f"SOURCE_ROWS {int(plan['source_rows'].sum())}")
    print(f"TARGET_ROWS {int(plan['target_time_rows'].iloc[0])}")
    print(f"CANDIDATE_K {int(plan['candidate_k'].iloc[0])}")
    print(f"APPROX_PER_WORKER_MATRIX_GIB {memory['approx_per_worker_memory_gib']:.4f}")
    print(f"SAFE_SINGLE_NODE_CONCURRENCY {int(memory['safe_single_node_concurrency'])}")


def execute_timepair(args: argparse.Namespace) -> int:
    started = time.monotonic()
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")
    config = load_config(args.config)
    if config["full_m3"]["enabled"]:
        raise RuntimeError("Refusing to run while full_m3.enabled is true.")
    if config["full_m3"].get("write_global_kernel"):
        raise RuntimeError("Refusing to build or configure a global Markov kernel.")
    config["full_m3"]["neighbor_backend"] = args.backend
    config["full_m3"]["candidate_k"] = int(args.candidate_k)
    config["full_m3"]["candidate_k_mode"] = "fixed"
    max_memory = args.max_memory_gb_warning
    if max_memory is None:
        max_memory = float(config["full_m3"].get("max_memory_gb_warning", 80))
    plan = filter_timepair_plan(
        pd.read_csv(args.plan_csv),
        args.source_time,
        args.target_time,
        args.candidate_k,
        args.max_shards,
    )
    feature_groups, retrieval, rerank, read_columns = feature_columns(config)
    memory = estimate_memory(plan, len(retrieval), len(rerank), float(max_memory))
    if args.dry_run:
        print_dry_run(plan, memory, args.source_time, args.target_time)
        return 0
    if memory["approx_per_worker_memory_gib"] > float(max_memory):
        raise MemoryError("Estimated per-worker matrix memory exceeds max_memory_gb_warning.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema_columns = full_transition_schema_columns()
    paths = _PILOT._paths(config)

    manifest_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    anchor_qc_rows: list[pd.DataFrame] = []
    slice_flows: list[pd.DataFrame] = []
    mouse_flows: list[pd.DataFrame] = []
    warnings: list[str] = []
    pending: list[dict[str, Any]] = []

    for shard in plan.to_dict(orient="records"):
        output = shard_paths(args.output_dir, args.source_time, args.target_time, str(shard["source_slice_id"]))
        if output["edges"].exists() or output["report"].exists():
            if args.overwrite:
                pending.append(shard)
                continue
            if args.resume:
                valid, qc, anchor_qc, slice_flow, mouse_flow, shard_warnings, reason = validate_existing_output(
                    output["edges"],
                    output["report"],
                    shard,
                    args.backend,
                )
                warnings.extend([f"{shard['source_slice_id']}: {warning}" for warning in shard_warnings])
                if valid and qc is not None and anchor_qc is not None and slice_flow is not None and mouse_flow is not None:
                    manifest_rows.append(build_manifest_row(shard, output["edges"], output["report"], "SKIPPED_RESUME"))
                    qc_rows.append(qc)
                    anchor_qc_rows.append(anchor_qc)
                    slice_flows.append(slice_flow)
                    if not mouse_flow.empty:
                        mouse_flows.append(mouse_flow)
                    continue
                manifest_rows.append(build_manifest_row(shard, output["edges"], output["report"], "FAILED", reason))
                if args.stop_on_error:
                    break
                continue
            raise FileExistsError(f"Output exists; use --resume or --overwrite: {output['edges']}")
        pending.append(shard)

    target = None
    if pending:
        pair = load_time_pair(config, args.source_time, args.target_time)
        target = _PILOT._load_target_time(paths["m2_by_slice_dir"], pair["target_slices"], read_columns)

    for shard in pending:
        output = shard_paths(args.output_dir, args.source_time, args.target_time, str(shard["source_slice_id"]))
        shard_start = time.monotonic()
        try:
            source = pd.read_parquet(
                _PILOT._slice_path(paths["m2_by_slice_dir"], str(shard["source_slice_id"])),
                columns=read_columns,
            )
            frame, metadata = _PILOT.build_pilot_edges(source, target, shard, config, feature_groups)
            _PILOT.validate_pilot_edges(frame, shard, schema_columns)
            frame.to_parquet(output["edges"], index=False)
            runtime = time.monotonic() - shard_start
            max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            qc, anchor_qc, slice_flow, mouse_flow, shard_warnings = compute_edge_qc(
                frame,
                shard,
                backend=metadata["backend"],
                runtime_seconds=runtime,
                max_rss_kb=max_rss,
                output_size_bytes=output["edges"].stat().st_size,
            )
            ok, reason = required_qc_passes(qc)
            if not ok:
                raise ValueError(reason)
            write_shard_report(output["report"], qc, shard_warnings)
            manifest_rows.append(build_manifest_row(shard, output["edges"], output["report"], "COMPLETED"))
            qc_rows.append(qc)
            anchor_qc_rows.append(anchor_qc)
            slice_flows.append(slice_flow)
            if not mouse_flow.empty:
                mouse_flows.append(mouse_flow)
            warnings.extend([f"{shard['source_slice_id']}: {warning}" for warning in shard_warnings])
            print(f"COMPLETED {shard['source_slice_id']} edge_rows={qc['observed_edge_rows']} runtime_seconds={runtime:.3f}")
        except Exception as exc:  # noqa: BLE001
            manifest_rows.append(build_manifest_row(shard, output["edges"], output["report"], "FAILED", str(exc)))
            print(f"FAILED {shard['source_slice_id']} reason={exc}", file=sys.stderr)
            if args.stop_on_error:
                break
        finally:
            if "frame" in locals():
                del frame
            if "source" in locals():
                del source
            gc.collect()

    write_timepair_outputs(
        args.output_dir,
        args.source_time,
        args.target_time,
        manifest_rows,
        qc_rows,
        anchor_qc_rows,
        slice_flows,
        mouse_flows,
        memory,
        warnings,
        started,
    )
    manifest = pd.DataFrame(manifest_rows)
    failed = int((manifest["status"] == "FAILED").sum()) if not manifest.empty else 0
    completed = int((manifest["status"] == "COMPLETED").sum()) if not manifest.empty else 0
    skipped = int((manifest["status"] == "SKIPPED_RESUME").sum()) if not manifest.empty else 0
    observed = int(pd.DataFrame(qc_rows)["observed_edge_rows"].sum()) if qc_rows else 0
    expected = int(plan["expected_edge_rows"].sum())
    print(f"TIME_PAIR {args.source_time}->{args.target_time}")
    print(f"COMPLETED_SHARDS {completed}")
    print(f"SKIPPED_SHARDS {skipped}")
    print(f"FAILED_SHARDS {failed}")
    print(f"OBSERVED_EDGE_ROWS {observed}")
    print(f"EXPECTED_EDGE_ROWS {expected}")
    print(f"TOTAL_RUNTIME_SECONDS {time.monotonic() - started:.3f}")
    return 1 if failed else 0


def main() -> int:
    return execute_timepair(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
