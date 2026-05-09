#!/usr/bin/env python
"""Validate pynndescent on one full M3 source-slice shard against exact output."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

for _thread_var in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_thread_var, "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config
from nichefate.transition import (
    CandidateNeighborBackendStatus,
    full_transition_schema_columns,
    inspect_candidate_neighbor_backend,
)


DEFAULT_SOURCE_TIME = "D21"
DEFAULT_TARGET_TIME = "D35"
DEFAULT_SOURCE_SLICE_ID = "082421_D21_m2_1_slice_2"
DEFAULT_CANDIDATE_K = 30
DEFAULT_PLAN_CSV = Path("/home/zhutao/scratch/nichefate/m3/reports/m3_full_transition_shards.csv")
DEFAULT_EXACT_REFERENCE = Path(
    "/home/zhutao/scratch/nichefate/m3/timepair_pilot_D21_to_D35/"
    "candidate_edges_D21_to_D35__082421_D21_m2_1_slice_2.parquet"
)
DEFAULT_OUTPUT_DIR = Path("/home/zhutao/scratch/nichefate/m3/ann_full_shard_validation_D21_to_D35")
OUTPUT_TOKENS = [
    "global_markov",
    "markov_p",
    "gpcca",
    "fate",
    "branched",
    "nicheflow",
    "m5",
    "regulator",
]


def _load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M3_05 = _load_script_module("m3_05_build_transition_pilot_shard", PROJECT_ROOT / "scripts" / "m3_05_build_transition_pilot_shard.py")
M3_11 = _load_script_module("m3_11_validate_ann_backend_on_sampled_shard", PROJECT_ROOT / "scripts" / "m3_11_validate_ann_backend_on_sampled_shard.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_transition_kernel.yaml")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--source-time", default=DEFAULT_SOURCE_TIME)
    parser.add_argument("--target-time", default=DEFAULT_TARGET_TIME)
    parser.add_argument("--source-slice-id", default=DEFAULT_SOURCE_SLICE_ID)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--ann-backend", default="pynndescent")
    parser.add_argument("--exact-reference", type=Path, default=DEFAULT_EXACT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--allow-non-default-shard", action="store_true")
    return parser.parse_args()


def _safe_token(value: object) -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return text.strip("_") or "value"


def output_paths(output_dir: Path, source_time: str, target_time: str, source_slice_id: str) -> dict[str, Path]:
    stem = f"{_safe_token(source_time)}_to_{_safe_token(target_time)}__{_safe_token(source_slice_id)}"
    return {
        "edges": output_dir / f"ann_full_shard_edges_{stem}.parquet",
        "report": output_dir / f"ann_full_shard_validation_report_{stem}.md",
        "metrics": output_dir / f"ann_full_shard_validation_metrics_{stem}.csv",
        "summary": output_dir / f"ann_full_shard_validation_summary_{stem}.json",
        "overlap": output_dir / f"ann_full_shard_candidate_overlap_{stem}.csv",
        "figures_dir": output_dir / "figures",
    }


def _assert_no_ssd(config: dict[str, Any]) -> None:
    if bool(config["paths"].get("use_ssd", False)):
        raise RuntimeError("Refusing full-shard ANN validation while paths.use_ssd is true.")
    for value in config.get("paths", {}).values():
        if isinstance(value, str) and value.startswith("/ssd"):
            raise RuntimeError(f"Refusing to use /ssd path in full-shard ANN validation: {value}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def ensure_validation_output_dir(output_dir: Path, config: dict[str, Any]) -> None:
    resolved = output_dir.resolve()
    production_roots = [
        Path(config["full_m3"]["output_root"]).resolve(),
        Path(config["paths"]["m3_output_dir"]).resolve() / "by_pair",
        Path(config["paths"]["m3_output_dir"]).resolve() / "timepair_pilot_D21_to_D35",
    ]
    for root in production_roots:
        if _is_relative_to(resolved, root):
            raise ValueError(f"Refusing to write ANN validation outputs under production M3 directory: {root}")
    lower = " ".join(part.lower() for part in resolved.parts if part.lower() != "nichefate")
    for token in OUTPUT_TOKENS:
        if token in lower:
            raise ValueError(f"Refusing downstream-looking ANN validation output path containing {token!r}.")


def ensure_exact_reference(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing exact reference edge parquet: {path}")
    if path.name.startswith("ann_"):
        raise ValueError("Exact reference must be an existing sklearn_exact shard, not ANN validation output.")


def validate_requested_scope(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if (
        not args.allow_non_default_shard
        and (
            str(args.source_time) != DEFAULT_SOURCE_TIME
            or str(args.target_time) != DEFAULT_TARGET_TIME
            or str(args.source_slice_id) != DEFAULT_SOURCE_SLICE_ID
        )
    ):
        raise ValueError(
            "This validation stage is scoped to D21->D35 "
            f"{DEFAULT_SOURCE_SLICE_ID}; pass --allow-non-default-shard to override."
        )
    if int(args.candidate_k) != DEFAULT_CANDIDATE_K:
        raise ValueError("This full-shard validation stage is scoped to candidate_k=30.")
    if args.ann_backend != "pynndescent":
        raise ValueError("This stage is scoped to pynndescent validation only.")
    _assert_no_ssd(config)
    ensure_validation_output_dir(args.output_dir, config)
    ensure_exact_reference(args.exact_reference)


def select_validation_shard(
    shards: pd.DataFrame,
    source_time: str,
    target_time: str,
    source_slice_id: str,
) -> dict[str, Any]:
    return M3_11.select_validation_shard(shards, source_time, target_time, source_slice_id)


def full_source_slice(frame: pd.DataFrame) -> pd.DataFrame:
    """Return all source anchors without downsampling."""

    return frame.reset_index(drop=True).copy()


def load_full_shard_data(
    config: dict[str, Any],
    shard: dict[str, Any],
    source_time: str,
    target_time: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feature_groups, _, _, read_columns = M3_11.feature_columns(config)
    root = Path(config["paths"]["m2_by_slice_dir"])
    source = pd.read_parquet(M3_11.slice_path(root, str(shard["source_slice_id"])), columns=read_columns)
    source = full_source_slice(source)
    if not bool((source["time"].astype(str) == str(source_time)).all()):
        raise ValueError("Selected source slice contains unexpected source_time values.")
    target_frames = [
        pd.read_parquet(M3_11.slice_path(root, slice_id), columns=read_columns)
        for slice_id in M3_11.target_slices_for_pair(config, source_time, target_time)
    ]
    target = pd.concat(target_frames, ignore_index=True)
    if not bool((target["time"].astype(str) == str(target_time)).all()):
        raise ValueError("Target pool contains unexpected target_time values.")
    return source, target, feature_groups


def max_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / float(1024**2)


def parse_exact_shard_report(exact_reference: Path) -> dict[str, float | None]:
    report = exact_reference.with_name(exact_reference.name.replace("candidate_edges_", "pilot_report_").replace(".parquet", ".md"))
    parsed: dict[str, float | None] = {
        "exact_reference_runtime_seconds": None,
        "exact_reference_max_rss_gib": None,
        "exact_reference_output_size_bytes": float(exact_reference.stat().st_size),
    }
    if not report.exists():
        return parsed
    text = report.read_text(encoding="utf-8")
    patterns = {
        "exact_reference_runtime_seconds": r"runtime_seconds:\s*([0-9.eE+-]+)",
        "exact_reference_max_rss_gib": r"max_rss_gib:\s*([0-9.eE+-]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            parsed[key] = float(match.group(1))
    return parsed


def add_candidate_rank(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame[["source_anchor_id", "target_anchor_id"]].copy()
    ranked["rank"] = ranked.groupby("source_anchor_id", observed=True).cumcount() + 1
    return ranked


def distance_rank_correlation(exact_ranks: dict[str, int], ann_ranks: dict[str, int]) -> float:
    common = sorted(set(exact_ranks) & set(ann_ranks))
    if len(common) < 2:
        return np.nan
    exact_values = np.array([exact_ranks[target] for target in common], dtype=float)
    ann_values = np.array([ann_ranks[target] for target in common], dtype=float)
    if np.std(exact_values) == 0.0 or np.std(ann_values) == 0.0:
        return np.nan
    return float(np.corrcoef(exact_values, ann_values)[0, 1])


def compare_candidate_edges(exact_edges: pd.DataFrame, ann_edges: pd.DataFrame, candidate_k: int) -> pd.DataFrame:
    exact_ranked = add_candidate_rank(exact_edges)
    ann_ranked = add_candidate_rank(ann_edges)
    exact_groups = {
        source: group.set_index("target_anchor_id")["rank"].astype(int).to_dict()
        for source, group in exact_ranked.groupby("source_anchor_id", observed=True)
    }
    ann_groups = {
        source: group.set_index("target_anchor_id")["rank"].astype(int).to_dict()
        for source, group in ann_ranked.groupby("source_anchor_id", observed=True)
    }
    source_ids = sorted(set(exact_groups) | set(ann_groups))
    rows: list[dict[str, Any]] = []
    for source_id in source_ids:
        exact_ranks = exact_groups.get(source_id, {})
        ann_ranks = ann_groups.get(source_id, {})
        exact_set = set(exact_ranks)
        ann_set = set(ann_ranks)
        intersection = exact_set & ann_set
        union = exact_set | ann_set
        represented_both = bool(exact_set and ann_set)
        rows.append(
            {
                "source_anchor_id": source_id,
                "represented_in_exact": bool(exact_set),
                "represented_in_ann": bool(ann_set),
                "represented_in_both": represented_both,
                "exact_candidate_count": len(exact_set),
                "ann_candidate_count": len(ann_set),
                "shared_candidate_count": len(intersection),
                "recall_at_k": len(intersection) / float(candidate_k) if represented_both else np.nan,
                "jaccard_overlap": len(intersection) / float(len(union)) if union and represented_both else np.nan,
                "top1_agreement": (
                    min(exact_ranks, key=exact_ranks.get) == min(ann_ranks, key=ann_ranks.get)
                    if represented_both
                    else np.nan
                ),
                "distance_rank_correlation": distance_rank_correlation(exact_ranks, ann_ranks)
                if represented_both
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _mean_bool(series: pd.Series) -> float:
    values = series.dropna()
    return float(values.astype(bool).mean()) if len(values) else np.nan


def build_full_shard_metrics(
    overlap: pd.DataFrame,
    drift: pd.DataFrame,
    row_diag: pd.DataFrame,
    ann_timing: dict[str, float],
    exact_report: dict[str, float | None],
    ann_edge_path: Path,
    context: dict[str, Any],
) -> pd.DataFrame:
    both = overlap[overlap["represented_in_both"].astype(bool)]
    metrics: dict[str, Any] = {
        **context,
        **exact_report,
        "ann_runtime_seconds": float(ann_timing["runtime_seconds"]),
        "ann_max_rss_gib": float(ann_timing["max_rss_gib"]),
        "ann_output_size_bytes": int(ann_edge_path.stat().st_size) if ann_edge_path.exists() else 0,
        "runtime_ratio_ann_over_exact": np.nan,
        "memory_ratio_ann_over_exact": np.nan,
        "source_anchors_represented_both": int(overlap["represented_in_both"].sum()),
        "source_anchors_missing_exact": int((~overlap["represented_in_exact"].astype(bool)).sum()),
        "source_anchors_missing_ann": int((~overlap["represented_in_ann"].astype(bool)).sum()),
        "recall_at_30_mean": float(both["recall_at_k"].mean()),
        "recall_at_30_median": float(both["recall_at_k"].median()),
        "recall_at_30_p05": float(both["recall_at_k"].quantile(0.05)),
        "recall_at_30_p95": float(both["recall_at_k"].quantile(0.95)),
        "top1_agreement": _mean_bool(both["top1_agreement"]),
        "jaccard_overlap_mean": float(both["jaccard_overlap"].mean()),
        "jaccard_overlap_median": float(both["jaccard_overlap"].median()),
        "distance_rank_correlation_mean": float(both["distance_rank_correlation"].mean(skipna=True)),
    }
    if exact_report.get("exact_reference_runtime_seconds"):
        metrics["runtime_ratio_ann_over_exact"] = float(
            metrics["ann_runtime_seconds"] / float(exact_report["exact_reference_runtime_seconds"])
        )
    if exact_report.get("exact_reference_max_rss_gib"):
        metrics["memory_ratio_ann_over_exact"] = float(
            metrics["ann_max_rss_gib"] / float(exact_report["exact_reference_max_rss_gib"])
        )
    for column in ["raw_edge_weight", "mass_adjusted_weight", "row_normalized_transition_prob"]:
        metrics.update(M3_11.summarize_abs(drift[f"{column}_abs_drift"], f"{column}_abs_drift"))
    for metric in [
        "row_entropy",
        "top1_probability",
        "target_slice_id_entropy",
        "target_mouse_id_entropy",
        "top_target_slice_id_fraction",
        "top_target_mouse_id_fraction",
    ]:
        metrics[f"{metric}_exact_mean"] = float(row_diag[f"{metric}_exact"].mean())
        metrics[f"{metric}_ann_mean"] = float(row_diag[f"{metric}_ann"].mean())
        metrics[f"{metric}_delta_mean"] = float(row_diag[f"{metric}_delta"].mean())
        metrics[f"{metric}_abs_delta_mean"] = float(row_diag[f"{metric}_abs_delta"].mean())
        metrics[f"{metric}_abs_delta_p95"] = float(row_diag[f"{metric}_abs_delta"].quantile(0.95))
    metrics["soft_recall_pass"] = bool(metrics["recall_at_30_mean"] >= 0.8)
    metrics["soft_top1_pass"] = bool(metrics["top1_agreement"] >= 0.8)
    metrics["soft_jaccard_pass"] = bool(metrics["jaccard_overlap_mean"] >= 0.7)
    metrics["soft_probability_drift_ok"] = bool(metrics["row_normalized_transition_prob_abs_drift_p95"] <= 0.05)
    metrics["soft_row_entropy_shift_ok"] = bool(metrics["row_entropy_abs_delta_mean"] <= 0.25)
    metrics["soft_top1_probability_shift_ok"] = bool(metrics["top1_probability_abs_delta_mean"] <= 0.10)
    metrics["soft_target_collapse_shift_ok"] = bool(
        metrics["top_target_slice_id_fraction_abs_delta_mean"] <= 0.10
        and metrics["top_target_mouse_id_fraction_abs_delta_mean"] <= 0.10
    )
    metrics["soft_validation_pass"] = bool(
        metrics["soft_recall_pass"]
        and metrics["soft_top1_pass"]
        and metrics["soft_jaccard_pass"]
        and metrics["soft_probability_drift_ok"]
        and metrics["soft_row_entropy_shift_ok"]
        and metrics["soft_top1_probability_shift_ok"]
        and metrics["soft_target_collapse_shift_ok"]
    )
    return pd.DataFrame([metrics])


def generate_figures(
    figures_dir: Path,
    overlap: pd.DataFrame,
    drift: pd.DataFrame,
    row_diag: pd.DataFrame,
    metrics: pd.DataFrame,
    force_failure: bool = False,
) -> list[str]:
    warnings: list[str] = []
    try:
        if force_failure:
            raise RuntimeError("forced figure failure")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir.mkdir(parents=True, exist_ok=True)
        row = metrics.iloc[0]
        both = overlap[overlap["represented_in_both"].astype(bool)]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(both["recall_at_k"].dropna(), bins=40)
        ax.axvline(0.8, color="red", linestyle="--")
        ax.set_title("Full-Shard Recall@30")
        ax.set_xlabel("recall@30")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_full_shard_recall_distribution.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(both["jaccard_overlap"].dropna(), bins=40)
        ax.axvline(0.7, color="red", linestyle="--")
        ax.set_title("Full-Shard Jaccard Overlap")
        ax.set_xlabel("Jaccard")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_full_shard_jaccard_distribution.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(drift["row_normalized_transition_prob_abs_drift"].dropna(), bins=50)
        ax.axvline(0.05, color="red", linestyle="--")
        ax.set_title("Full-Shard Probability Drift")
        ax.set_xlabel("absolute drift")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_full_shard_probability_drift.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(row_diag["row_entropy_delta"].dropna(), bins=50)
        ax.set_title("Full-Shard Row Entropy Delta")
        ax.set_xlabel("ANN - exact")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_full_shard_entropy_delta.png", dpi=140)
        plt.close(fig)

        labels = ["slice entropy", "mouse entropy", "top slice", "top mouse"]
        values = [
            row["target_slice_id_entropy_delta_mean"],
            row["target_mouse_id_entropy_delta_mean"],
            row["top_target_slice_id_fraction_delta_mean"],
            row["top_target_mouse_id_fraction_delta_mean"],
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=25)
        ax.set_title("Target Slice/Mouse Diagnostic Delta")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_full_shard_target_slice_mouse_delta.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].bar(["exact ref", "ANN"], [row["exact_reference_runtime_seconds"], row["ann_runtime_seconds"]])
        axes[0].set_title("Runtime seconds")
        axes[1].bar(["exact ref", "ANN"], [row["exact_reference_max_rss_gib"], row["ann_max_rss_gib"]])
        axes[1].set_title("Max RSS GiB")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_full_shard_runtime_memory.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        axes[0, 0].hist(both["recall_at_k"].dropna(), bins=40)
        axes[0, 0].set_title("Recall@30")
        axes[0, 1].hist(both["jaccard_overlap"].dropna(), bins=40)
        axes[0, 1].set_title("Jaccard")
        axes[1, 0].hist(drift["row_normalized_transition_prob_abs_drift"].dropna(), bins=40)
        axes[1, 0].set_title("Probability drift")
        axes[1, 1].hist(row_diag["row_entropy_delta"].dropna(), bins=40)
        axes[1, 1].set_title("Entropy delta")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_full_shard_summary_dashboard.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Figure generation failed but validation reports/tables were written: {exc}")
    return warnings


def write_report(path: Path, metrics: pd.DataFrame, figure_warnings: list[str], ann_metadata: dict[str, Any]) -> None:
    row = metrics.iloc[0].to_dict()
    lines = [
        "# M3 ANN Full-Shard Validation",
        "",
        "This is an ANN full-shard validation output. It is not production full M3, does not overwrite the exact time-pair pilot, and does not assemble global Markov P.",
        "`row_normalized_transition_prob` remains local to each source niche candidate set.",
        "No GPCCA, fate probability, Branched NicheFlow, M5, or regulator analysis was run.",
        "",
        "## Selected Shard",
        f"- source_time: {row['source_time']}",
        f"- target_time: {row['target_time']}",
        f"- source_slice_id: {row['source_slice_id']}",
        f"- source rows: {int(row['source_rows'])}",
        f"- target rows: {int(row['target_rows'])}",
        f"- candidate_k: {int(row['candidate_k'])}",
        "",
        "## Candidate Retrieval",
        f"- source anchors represented in both outputs: {int(row['source_anchors_represented_both'])}",
        f"- missing exact / missing ANN source anchors: {int(row['source_anchors_missing_exact'])} / {int(row['source_anchors_missing_ann'])}",
        f"- recall@30 mean/median/p05/p95: {row['recall_at_30_mean']:.6g} / {row['recall_at_30_median']:.6g} / {row['recall_at_30_p05']:.6g} / {row['recall_at_30_p95']:.6g}",
        f"- top1 agreement: {row['top1_agreement']:.6g}",
        f"- mean Jaccard overlap: {row['jaccard_overlap_mean']:.6g}",
        f"- mean rank correlation: {row['distance_rank_correlation_mean']:.6g}",
        "",
        "## Transition Drift",
        f"- row probability abs drift mean/median/p95: {row['row_normalized_transition_prob_abs_drift_mean']:.6g} / {row['row_normalized_transition_prob_abs_drift_median']:.6g} / {row['row_normalized_transition_prob_abs_drift_p95']:.6g}",
        f"- raw edge weight abs drift mean/p95: {row['raw_edge_weight_abs_drift_mean']:.6g} / {row['raw_edge_weight_abs_drift_p95']:.6g}",
        f"- mass-adjusted weight abs drift mean/p95: {row['mass_adjusted_weight_abs_drift_mean']:.6g} / {row['mass_adjusted_weight_abs_drift_p95']:.6g}",
        f"- row entropy exact/ANN/delta mean: {row['row_entropy_exact_mean']:.6g} / {row['row_entropy_ann_mean']:.6g} / {row['row_entropy_delta_mean']:.6g}",
        f"- top1 probability exact/ANN/delta mean: {row['top1_probability_exact_mean']:.6g} / {row['top1_probability_ann_mean']:.6g} / {row['top1_probability_delta_mean']:.6g}",
        "",
        "## Batch/Collapse Diagnostics",
        f"- target slice entropy exact/ANN/delta mean: {row['target_slice_id_entropy_exact_mean']:.6g} / {row['target_slice_id_entropy_ann_mean']:.6g} / {row['target_slice_id_entropy_delta_mean']:.6g}",
        f"- target mouse entropy exact/ANN/delta mean: {row['target_mouse_id_entropy_exact_mean']:.6g} / {row['target_mouse_id_entropy_ann_mean']:.6g} / {row['target_mouse_id_entropy_delta_mean']:.6g}",
        f"- top target slice fraction exact/ANN/delta mean: {row['top_target_slice_id_fraction_exact_mean']:.6g} / {row['top_target_slice_id_fraction_ann_mean']:.6g} / {row['top_target_slice_id_fraction_delta_mean']:.6g}",
        f"- top target mouse fraction exact/ANN/delta mean: {row['top_target_mouse_id_fraction_exact_mean']:.6g} / {row['top_target_mouse_id_fraction_ann_mean']:.6g} / {row['top_target_mouse_id_fraction_delta_mean']:.6g}",
        "",
        "## Runtime And Memory",
        f"- exact reference runtime seconds: {row['exact_reference_runtime_seconds']}",
        f"- exact reference max RSS GiB: {row['exact_reference_max_rss_gib']}",
        f"- ANN runtime seconds: {row['ann_runtime_seconds']:.3f}",
        f"- ANN max RSS GiB: {row['ann_max_rss_gib']:.4f}",
        f"- runtime ratio ANN/exact: {row['runtime_ratio_ann_over_exact']}",
        f"- memory ratio ANN/exact: {row['memory_ratio_ann_over_exact']}",
        f"- ANN output file size bytes: {int(row['ann_output_size_bytes'])}",
        "",
        "## Soft Validation Thresholds",
        f"- recall@30 mean >= 0.8: {bool(row['soft_recall_pass'])}",
        f"- top1 agreement >= 0.8: {bool(row['soft_top1_pass'])}",
        f"- mean Jaccard overlap >= 0.7: {bool(row['soft_jaccard_pass'])}",
        f"- probability drift p95 <= 0.05 preferred: {bool(row['soft_probability_drift_ok'])}",
        f"- entropy shift ok: {bool(row['soft_row_entropy_shift_ok'])}",
        f"- top1 probability shift ok: {bool(row['soft_top1_probability_shift_ok'])}",
        f"- target collapse shift ok: {bool(row['soft_target_collapse_shift_ok'])}",
        f"- overall soft validation pass: {bool(row['soft_validation_pass'])}",
        "",
        "## ANN Metadata",
        f"- backend: {ann_metadata.get('backend')}",
        f"- tau_pair: {ann_metadata.get('tau_pair')}",
    ]
    if figure_warnings:
        lines.extend(["", "## Figure Warnings", *[f"- {warning}" for warning in figure_warnings]])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_skip_outputs(paths: dict[str, Path], reason: str, status: CandidateNeighborBackendStatus) -> None:
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "SKIPPED",
        "reason": reason,
        "backend_status": asdict(status),
        "no_production_m3_edges": True,
        "no_full_m3": True,
        "no_global_markov_p": True,
        "no_gpcca_fate_branched_m5_regulator": True,
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    paths["report"].write_text(
        "# M3 ANN Full-Shard Validation Skipped\n\n"
        f"- Reason: {reason}\n"
        "- No package installation or conda modification was attempted.\n"
        "- No production M3 edges, full M3, global Markov P, GPCCA, fate probability, Branched NicheFlow, M5, or regulator analysis was run.\n",
        encoding="utf-8",
    )


def run_validation(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    validate_requested_scope(args, config)
    paths = output_paths(args.output_dir, args.source_time, args.target_time, args.source_slice_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ann_status = inspect_candidate_neighbor_backend(args.ann_backend, run_toy_check=True)
    if not ann_status.available:
        reason = f"{args.ann_backend} is not usable: {ann_status.reason}"
        write_skip_outputs(paths, reason, ann_status)
        print(f"ANN_FULL_SHARD_VALIDATION_SKIPPED {reason}")
        return 0

    shard = select_validation_shard(
        pd.read_csv(args.plan_csv),
        args.source_time,
        args.target_time,
        args.source_slice_id,
    )
    source, target, feature_groups = load_full_shard_data(config, shard, args.source_time, args.target_time)
    if len(source) != int(shard["source_rows"]):
        raise ValueError(f"Full source-slice mode expected {shard['source_rows']} rows, found {len(source)}.")
    exact_edges = pd.read_parquet(args.exact_reference)
    expected_rows = int(shard["source_rows"]) * int(args.candidate_k)
    if len(exact_edges) != expected_rows:
        raise ValueError(f"Exact reference rows {len(exact_edges)} != expected {expected_rows}.")

    ann_config = json.loads(json.dumps(config))
    ann_config["full_m3"]["neighbor_backend"] = args.ann_backend
    ann_config["full_m3"]["candidate_k"] = int(args.candidate_k)
    start = time.monotonic()
    ann_edges, ann_metadata = M3_05.build_pilot_edges(source, target, shard, ann_config, feature_groups)
    ann_timing = {"runtime_seconds": time.monotonic() - start, "max_rss_gib": max_rss_gib()}
    if list(ann_edges.columns) != full_transition_schema_columns():
        raise ValueError("ANN full-shard validation edge schema does not match M3 full transition schema.")
    ann_edges.to_parquet(paths["edges"], index=False)

    overlap = compare_candidate_edges(exact_edges, ann_edges, int(args.candidate_k))
    drift = M3_11.drift_metrics(exact_edges, ann_edges)
    row_diag = M3_11.compare_row_diagnostics(exact_edges, ann_edges)
    exact_report = parse_exact_shard_report(args.exact_reference)
    context = {
        "source_time": args.source_time,
        "target_time": args.target_time,
        "source_slice_id": args.source_slice_id,
        "source_slice_file": shard["source_slice_file"],
        "source_rows": len(source),
        "target_rows": len(target),
        "candidate_k": int(args.candidate_k),
        "expected_edge_rows": expected_rows,
        "ann_backend": args.ann_backend,
        "exact_reference": str(args.exact_reference),
    }
    metrics = build_full_shard_metrics(
        overlap,
        drift,
        row_diag,
        ann_timing,
        exact_report,
        paths["edges"],
        context,
    )
    metrics.to_csv(paths["metrics"], index=False)
    overlap.to_csv(paths["overlap"], index=False)
    figure_warnings = generate_figures(paths["figures_dir"], overlap, drift, row_diag, metrics)
    write_report(paths["report"], metrics, figure_warnings, ann_metadata)
    summary = {
        "status": "COMPLETED",
        "metrics": metrics.iloc[0].to_dict(),
        "ann_backend_status": asdict(ann_status),
        "ann_metadata": ann_metadata,
        "figure_warnings": figure_warnings,
        "outputs": {key: str(value) for key, value in paths.items()},
        "no_production_m3_edges": True,
        "no_full_m3": True,
        "no_global_markov_p": True,
        "no_gpcca": True,
        "no_fate_probability": True,
        "no_branched_nicheflow": True,
        "no_m5": True,
        "no_regulator_analysis": True,
    }
    paths["summary"].write_text(json.dumps(M3_11.json_safe(summary), indent=2) + "\n", encoding="utf-8")
    row = metrics.iloc[0]
    print(f"ANN_FULL_SHARD_VALIDATION_COMPLETED {args.source_time}->{args.target_time} {args.source_slice_id}")
    print(f"SOURCE_ROWS {len(source)}")
    print(f"TARGET_ROWS {len(target)}")
    print(f"CANDIDATE_K {int(args.candidate_k)}")
    print(f"RECALL_AT_30_MEAN {row['recall_at_30_mean']:.6g}")
    print(f"TOP1_AGREEMENT {row['top1_agreement']:.6g}")
    print(f"JACCARD_OVERLAP_MEAN {row['jaccard_overlap_mean']:.6g}")
    print(f"SOFT_VALIDATION_PASS {bool(row['soft_validation_pass'])}")
    return 0


def main() -> int:
    return run_validation(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
