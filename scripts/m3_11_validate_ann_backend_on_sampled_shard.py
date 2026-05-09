#!/usr/bin/env python
"""Validate an optional ANN backend against exact KNN on one sampled M3 shard."""

from __future__ import annotations

import argparse
import json
import os
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
    CandidateNeighbors,
    build_candidate_neighbors,
    combine_scaled_evidence,
    inspect_candidate_neighbor_backend,
    pair_adaptive_temperature,
    pairwise_row_distance,
    row_normalize_weights,
    safe_scale_vector,
    standardize_feature_matrices,
)


DEFAULT_SOURCE_TIME = "D21"
DEFAULT_TARGET_TIME = "D35"
DEFAULT_SOURCE_SLICE_ID = "082421_D21_m2_1_slice_2"
DEFAULT_SAMPLE_SIZE = 5000
DEFAULT_CANDIDATE_K = 30
DEFAULT_OUTPUT_DIR = Path("/home/zhutao/scratch/nichefate/m3/ann_validation_D21_to_D35")
DEFAULT_PLAN_CSV = Path("/home/zhutao/scratch/nichefate/m3/reports/m3_full_transition_shards.csv")

RETRIEVAL_GROUPS = ["molecular_state", "cell_type_composition", "entropy"]
RERANK_GROUPS = [
    "molecular_state",
    "cell_type_composition",
    "entropy",
    "spatial_summary",
    "topology",
]
GROUP_TO_EVIDENCE = {
    "molecular_state": "molecular",
    "cell_type_composition": "composition",
    "entropy": "entropy",
    "spatial_summary": "spatial_summary",
    "topology": "topology",
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_transition_kernel.yaml")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--source-time", default=DEFAULT_SOURCE_TIME)
    parser.add_argument("--target-time", default=DEFAULT_TARGET_TIME)
    parser.add_argument("--source-slice-id", default=DEFAULT_SOURCE_SLICE_ID)
    parser.add_argument("--sample-source-anchors", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--exact-backend", default="sklearn_exact")
    parser.add_argument("--ann-backend", default="pynndescent")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--allow-larger-sample", action="store_true")
    parser.add_argument("--allow-non-default-shard", action="store_true")
    parser.add_argument("--skip-candidate-tables", action="store_true")
    return parser.parse_args()


def _safe_token(value: object) -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return text.strip("_") or "value"


def output_paths(output_dir: Path, source_time: str, target_time: str, source_slice_id: str) -> dict[str, Path]:
    stem = f"{_safe_token(source_time)}_to_{_safe_token(target_time)}__{_safe_token(source_slice_id)}"
    return {
        "report": output_dir / f"ann_validation_report_{stem}.md",
        "metrics": output_dir / f"ann_validation_metrics_{stem}.csv",
        "summary": output_dir / f"ann_validation_summary_{stem}.json",
        "overlap": output_dir / f"candidate_overlap_summary_{stem}.csv",
        "exact_candidates": output_dir / f"exact_candidates_sample_{stem}.parquet",
        "ann_candidates": output_dir / f"pynndescent_candidates_sample_{stem}.parquet",
        "figures_dir": output_dir / "figures",
    }


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
    if int(args.sample_source_anchors) > DEFAULT_SAMPLE_SIZE and not args.allow_larger_sample:
        raise ValueError("Refusing sample size > 5000 without --allow-larger-sample.")
    if int(args.sample_source_anchors) <= 0:
        raise ValueError("--sample-source-anchors must be positive.")
    if int(args.candidate_k) <= 0:
        raise ValueError("--candidate-k must be positive.")
    if args.ann_backend != "pynndescent":
        raise ValueError("This stage is scoped to pynndescent validation only.")
    if args.exact_backend != "sklearn_exact":
        raise ValueError("This stage uses sklearn_exact as the exact validation backend.")
    _assert_no_ssd(config)
    ensure_validation_output_dir(args.output_dir, config)


def _assert_no_ssd(config: dict[str, Any]) -> None:
    if bool(config["paths"].get("use_ssd", False)):
        raise RuntimeError("Refusing ANN validation while paths.use_ssd is true.")
    for value in config.get("paths", {}).values():
        if isinstance(value, str) and value.startswith("/ssd"):
            raise RuntimeError(f"Refusing to use /ssd path in ANN validation: {value}")


def ensure_validation_output_dir(output_dir: Path, config: dict[str, Any]) -> None:
    resolved = output_dir.resolve()
    production_roots = [
        Path(config["full_m3"]["output_root"]).resolve(),
        Path(config["paths"]["m3_output_dir"]).resolve() / "by_pair",
    ]
    for root in production_roots:
        if _is_relative_to(resolved, root):
            raise ValueError(f"Refusing to write ANN validation outputs under production M3 directory: {root}")
    lower = " ".join(part.lower() for part in resolved.parts if part.lower() != "nichefate")
    for token in OUTPUT_TOKENS:
        if token in lower:
            raise ValueError(f"Refusing downstream-looking ANN validation output path containing {token!r}.")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def select_validation_shard(
    shards: pd.DataFrame,
    source_time: str,
    target_time: str,
    source_slice_id: str,
) -> dict[str, Any]:
    required = {
        "source_time",
        "target_time",
        "source_slice_id",
        "source_slice_file",
        "source_rows",
        "target_time_rows",
        "candidate_k",
        "expected_edge_rows",
    }
    missing = sorted(required - set(shards.columns))
    if missing:
        raise KeyError(f"Shard table is missing required columns: {missing}")
    selected = shards[
        (shards["source_time"].astype(str) == str(source_time))
        & (shards["target_time"].astype(str) == str(target_time))
        & (shards["source_slice_id"].astype(str) == str(source_slice_id))
    ]
    if len(selected) != 1:
        raise ValueError(f"Validation shard selector must match exactly one row, found {len(selected)}.")
    return selected.iloc[0].to_dict()


def deterministic_source_sample(frame: pd.DataFrame, sample_size: int, random_seed: int) -> pd.DataFrame:
    if sample_size >= len(frame):
        return frame.reset_index(drop=True).copy()
    rng = np.random.default_rng(int(random_seed))
    selected = np.sort(rng.choice(len(frame), size=int(sample_size), replace=False))
    return frame.iloc[selected].reset_index(drop=True).copy()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def slice_path(root: Path, slice_id: str) -> Path:
    return root / slice_id / f"m2_representation_{slice_id}.parquet"


def target_slices_for_pair(config: dict[str, Any], source_time: str, target_time: str) -> list[str]:
    pairs = load_json(Path(config["paths"]["reports_dir"]) / "m3_time_pairs.json")
    for pair in pairs:
        if str(pair["source_time"]) == str(source_time) and str(pair["target_time"]) == str(target_time):
            return [str(value) for value in pair["target_slices"]]
    raise ValueError(f"Missing time-pair metadata for {source_time}->{target_time}.")


def feature_columns(config: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    payload = load_json(Path(config["paths"]["reports_dir"]) / "m3_feature_groups.json")
    groups = payload["feature_groups"]
    retrieval = list(dict.fromkeys(column for group in RETRIEVAL_GROUPS for column in groups[group]))
    rerank = list(dict.fromkeys(column for group in RERANK_GROUPS for column in groups[group]))
    read_columns = list(dict.fromkeys(config["input"]["metadata_columns"] + retrieval + rerank))
    return payload, retrieval, rerank, read_columns


def load_validation_data(
    config: dict[str, Any],
    shard: dict[str, Any],
    source_time: str,
    target_time: str,
    sample_size: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str], list[str]]:
    feature_groups, retrieval, rerank, read_columns = feature_columns(config)
    root = Path(config["paths"]["m2_by_slice_dir"])
    source = pd.read_parquet(slice_path(root, str(shard["source_slice_id"])), columns=read_columns)
    if not bool((source["time"].astype(str) == str(source_time)).all()):
        raise ValueError("Selected source slice contains unexpected source_time values.")
    source_sample = deterministic_source_sample(source, sample_size, random_seed)
    target_frames = [
        pd.read_parquet(slice_path(root, slice_id), columns=read_columns)
        for slice_id in target_slices_for_pair(config, source_time, target_time)
    ]
    target = pd.concat(target_frames, ignore_index=True)
    if not bool((target["time"].astype(str) == str(target_time)).all()):
        raise ValueError("Target pool contains unexpected target_time values.")
    return source_sample, target, feature_groups, retrieval, rerank


def max_rss_gib() -> float:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss_kb) / float(1024**2)


def run_backend(
    source_retrieval: np.ndarray,
    target_retrieval: np.ndarray,
    backend: str,
    metric: str,
    candidate_k: int,
    chunk_size: int,
    random_seed: int,
) -> tuple[CandidateNeighbors | CandidateNeighborBackendStatus, dict[str, float]]:
    start = time.monotonic()
    result = build_candidate_neighbors(
        source_retrieval,
        target_retrieval,
        int(candidate_k),
        backend=backend,
        metric=metric,
        chunk_size=chunk_size,
        random_seed=random_seed,
    )
    return result, {"runtime_seconds": time.monotonic() - start, "max_rss_gib": max_rss_gib()}


def candidate_table(
    source: pd.DataFrame,
    target: pd.DataFrame,
    neighbors: CandidateNeighbors,
) -> pd.DataFrame:
    source_anchor_ids = (
        source["slice_id"].astype(str).to_numpy()
        + "::"
        + source["anchor_index"].astype(str).to_numpy()
    )
    target_anchor_ids = (
        target["slice_id"].astype(str).to_numpy()
        + "::"
        + target["anchor_index"].astype(str).to_numpy()
    )
    source_ids = np.repeat(source_anchor_ids, neighbors.indices.shape[1])
    ranks = np.tile(np.arange(1, neighbors.indices.shape[1] + 1), len(source))
    target_idx = neighbors.indices.reshape(-1)
    return pd.DataFrame(
        {
            "backend": neighbors.backend,
            "source_anchor_id": source_ids,
            "rank": ranks,
            "target_anchor_id": target_anchor_ids[target_idx],
            "target_local_index": target_idx,
            "candidate_distance": neighbors.distances.reshape(-1),
        }
    )


def compare_candidate_sets(
    exact: CandidateNeighbors,
    ann: CandidateNeighbors,
    source: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if exact.indices.shape != ann.indices.shape:
        raise ValueError("Exact and ANN candidate arrays must have the same shape.")
    k = exact.indices.shape[1]
    if source is None:
        source_anchor_ids = np.array([f"source::{idx}" for idx in range(exact.indices.shape[0])])
    else:
        source_anchor_ids = (
            source["slice_id"].astype(str).to_numpy()
            + "::"
            + source["anchor_index"].astype(str).to_numpy()
        )
    rows: list[dict[str, Any]] = []
    for idx, (exact_row, ann_row) in enumerate(zip(exact.indices, ann.indices, strict=True)):
        exact_set = set(int(value) for value in exact_row)
        ann_set = set(int(value) for value in ann_row)
        intersection = exact_set & ann_set
        union = exact_set | ann_set
        rows.append(
            {
                "source_anchor_id": source_anchor_ids[idx],
                "recall_at_k": len(intersection) / float(k),
                "jaccard_overlap": len(intersection) / float(len(union)) if union else 0.0,
                "top1_agreement": int(exact_row[0]) == int(ann_row[0]),
                "distance_rank_correlation": distance_rank_correlation(exact_row, ann_row),
                "exact_top1_target_local_index": int(exact_row[0]),
                "ann_top1_target_local_index": int(ann_row[0]),
            }
        )
    return pd.DataFrame(rows)


def distance_rank_correlation(exact_row: np.ndarray, ann_row: np.ndarray) -> float:
    exact_rank = {int(value): rank for rank, value in enumerate(exact_row, start=1)}
    ann_rank = {int(value): rank for rank, value in enumerate(ann_row, start=1)}
    common = sorted(set(exact_rank) & set(ann_rank))
    if len(common) < 2:
        return np.nan
    exact_values = np.array([exact_rank[value] for value in common], dtype=float)
    ann_values = np.array([ann_rank[value] for value in common], dtype=float)
    if np.std(exact_values) == 0.0 or np.std(ann_values) == 0.0:
        return np.nan
    return float(np.corrcoef(exact_values, ann_values)[0, 1])


def edge_metadata(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    shard: dict[str, Any],
) -> pd.DataFrame:
    src = source.iloc[source_idx]
    tgt = target.iloc[target_idx]
    return pd.DataFrame(
        {
            "source_anchor_id": src["slice_id"].astype(str).to_numpy()
            + "::"
            + src["anchor_index"].astype(str).to_numpy(),
            "target_anchor_id": tgt["slice_id"].astype(str).to_numpy()
            + "::"
            + tgt["anchor_index"].astype(str).to_numpy(),
            "source_anchor_index": src["anchor_index"].to_numpy(),
            "target_anchor_index": tgt["anchor_index"].to_numpy(),
            "source_time": src["time"].astype(str).to_numpy(),
            "target_time": tgt["time"].astype(str).to_numpy(),
            "source_day": src["time_day"].to_numpy(),
            "target_day": tgt["time_day"].to_numpy(),
            "time_delta": float(shard["time_delta"]),
            "source_slice_id": src["slice_id"].astype(str).to_numpy(),
            "target_slice_id": tgt["slice_id"].astype(str).to_numpy(),
            "source_slice_file": src["slice_file"].astype(str).to_numpy(),
            "target_slice_file": tgt["slice_file"].astype(str).to_numpy(),
            "source_mouse_id": src["mouse_id"].astype(str).to_numpy(),
            "target_mouse_id": tgt["mouse_id"].astype(str).to_numpy(),
            "evidence_mode": "ann_validation_pseudo_lineage",
        }
    )


def build_validation_edges(
    source: pd.DataFrame,
    target: pd.DataFrame,
    shard: dict[str, Any],
    config: dict[str, Any],
    feature_groups: dict[str, Any],
    neighbors: CandidateNeighbors,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    groups = feature_groups["feature_groups"]
    cost_cfg = config["cost"]
    source_idx = np.repeat(np.arange(len(source)), neighbors.indices.shape[1])
    target_idx = neighbors.indices.reshape(-1)
    frame = edge_metadata(source, target, source_idx, target_idx, shard)
    scaled_column_by_group = {}
    scaling_stats: dict[str, dict[str, Any]] = {}
    for group in RERANK_GROUPS:
        evidence = GROUP_TO_EVIDENCE[group]
        raw_col = f"raw_{evidence}_distance"
        scaled_col = f"scaled_{evidence}_distance"
        metric = "l1" if group == "cell_type_composition" else "euclidean"
        frame[raw_col] = pairwise_row_distance(source, target, source_idx, target_idx, groups[group], metric=metric)
        frame[scaled_col], stats = safe_scale_vector(frame[raw_col], float(cost_cfg["min_scale"]))
        scaled_column_by_group[group] = scaled_col
        scaling_stats[group] = stats
    frame["source_mass"] = 1.0
    frame["target_mass"] = 1.0
    frame["growth_prior"] = 1.0
    frame["unbalanced_weight"] = 1.0
    frame["combined_cost"] = combine_scaled_evidence(
        frame,
        cost_cfg["evidence_weights"],
        scaled_column_by_group,
    )
    tau = pair_adaptive_temperature(frame["combined_cost"], float(cost_cfg["min_temperature"]))
    exponent = np.clip(-frame["combined_cost"].to_numpy(dtype=float) / tau, -700, 700)
    frame["tau_pair"] = tau
    frame["raw_edge_weight"] = np.exp(exponent)
    frame["mass_adjusted_weight"] = frame["raw_edge_weight"]
    frame["row_normalized_transition_prob"] = row_normalize_weights(frame)
    return frame, {"backend": neighbors.backend, "tau_pair": tau, "scaling_stats": scaling_stats}


def entropy_from_probs(values: pd.Series) -> float:
    probs = values.astype(float).to_numpy()
    return float(-(probs * np.log(np.clip(probs, 1e-300, None))).sum())


def categorical_entropy_and_top_fraction(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for source_id, values in frame.groupby("source_anchor_id", observed=True)[column]:
        probs = values.astype(str).value_counts(normalize=True).to_numpy(dtype=float)
        rows.append(
            {
                "source_anchor_id": source_id,
                f"{column}_entropy": float(-(probs * np.log(np.clip(probs, 1e-300, None))).sum())
                if len(probs)
                else 0.0,
                f"top_{column}_fraction": float(probs.max()) if len(probs) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def row_diagnostics(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    prob = frame.groupby("source_anchor_id", observed=True)["row_normalized_transition_prob"]
    result = pd.DataFrame(
        {
            "source_anchor_id": prob.sum().index,
            f"row_entropy_{label}": prob.apply(entropy_from_probs).to_numpy(dtype=float),
            f"top1_probability_{label}": prob.max().to_numpy(dtype=float),
        }
    )
    for column in ["target_slice_id", "target_mouse_id"]:
        diag = categorical_entropy_and_top_fraction(frame, column)
        diag = diag.rename(
            columns={
                f"{column}_entropy": f"{column}_entropy_{label}",
                f"top_{column}_fraction": f"top_{column}_fraction_{label}",
            }
        )
        result = result.merge(diag, on="source_anchor_id", how="left")
    return result


def compare_row_diagnostics(exact_edges: pd.DataFrame, ann_edges: pd.DataFrame) -> pd.DataFrame:
    exact = row_diagnostics(exact_edges, "exact")
    ann = row_diagnostics(ann_edges, "ann")
    merged = exact.merge(ann, on="source_anchor_id", how="inner")
    for metric in [
        "row_entropy",
        "top1_probability",
        "target_slice_id_entropy",
        "target_mouse_id_entropy",
        "top_target_slice_id_fraction",
        "top_target_mouse_id_fraction",
    ]:
        merged[f"{metric}_delta"] = merged[f"{metric}_ann"] - merged[f"{metric}_exact"]
        merged[f"{metric}_abs_delta"] = merged[f"{metric}_delta"].abs()
    return merged


def drift_metrics(exact_edges: pd.DataFrame, ann_edges: pd.DataFrame) -> pd.DataFrame:
    columns = ["raw_edge_weight", "mass_adjusted_weight", "row_normalized_transition_prob"]
    exact = exact_edges[["source_anchor_id", "target_anchor_id", *columns]].rename(
        columns={column: f"{column}_exact" for column in columns}
    )
    ann = ann_edges[["source_anchor_id", "target_anchor_id", *columns]].rename(
        columns={column: f"{column}_ann" for column in columns}
    )
    merged = exact.merge(ann, on=["source_anchor_id", "target_anchor_id"], how="outer")
    for column in columns:
        merged[f"{column}_exact"] = merged[f"{column}_exact"].fillna(0.0)
        merged[f"{column}_ann"] = merged[f"{column}_ann"].fillna(0.0)
        merged[f"{column}_abs_drift"] = (merged[f"{column}_ann"] - merged[f"{column}_exact"]).abs()
    return merged


def summarize_abs(series: pd.Series, prefix: str) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {f"{prefix}_mean": np.nan, f"{prefix}_median": np.nan, f"{prefix}_p95": np.nan}
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_p95": float(values.quantile(0.95)),
    }


def build_metrics(
    overlap: pd.DataFrame,
    exact_neighbors: CandidateNeighbors,
    ann_neighbors: CandidateNeighbors,
    drift: pd.DataFrame,
    row_diag: pd.DataFrame,
    exact_timing: dict[str, float],
    ann_timing: dict[str, float],
    context: dict[str, Any],
) -> pd.DataFrame:
    metrics: dict[str, Any] = {
        **context,
        "sklearn_exact_runtime_seconds": float(exact_timing["runtime_seconds"]),
        "pynndescent_runtime_seconds": float(ann_timing["runtime_seconds"]),
        "runtime_ratio_ann_over_exact": float(ann_timing["runtime_seconds"] / exact_timing["runtime_seconds"])
        if exact_timing["runtime_seconds"]
        else np.nan,
        "sklearn_exact_max_rss_gib": float(exact_timing["max_rss_gib"]),
        "pynndescent_max_rss_gib": float(ann_timing["max_rss_gib"]),
        "memory_ratio_ann_over_exact": float(ann_timing["max_rss_gib"] / exact_timing["max_rss_gib"])
        if exact_timing["max_rss_gib"]
        else np.nan,
        "recall_at_30_mean": float(overlap["recall_at_k"].mean()),
        "recall_at_30_median": float(overlap["recall_at_k"].median()),
        "recall_at_30_p05": float(overlap["recall_at_k"].quantile(0.05)),
        "recall_at_30_p95": float(overlap["recall_at_k"].quantile(0.95)),
        "top1_agreement": float(overlap["top1_agreement"].mean()),
        "jaccard_overlap_mean": float(overlap["jaccard_overlap"].mean()),
        "jaccard_overlap_median": float(overlap["jaccard_overlap"].median()),
        "distance_rank_correlation_mean": float(overlap["distance_rank_correlation"].mean(skipna=True)),
        "exact_candidate_distance_mean": float(np.mean(exact_neighbors.distances)),
        "exact_candidate_distance_median": float(np.median(exact_neighbors.distances)),
        "exact_candidate_distance_p95": float(np.quantile(exact_neighbors.distances, 0.95)),
        "ann_candidate_distance_mean": float(np.mean(ann_neighbors.distances)),
        "ann_candidate_distance_median": float(np.median(ann_neighbors.distances)),
        "ann_candidate_distance_p95": float(np.quantile(ann_neighbors.distances, 0.95)),
    }
    for column in ["raw_edge_weight", "mass_adjusted_weight", "row_normalized_transition_prob"]:
        metrics.update(summarize_abs(drift[f"{column}_abs_drift"], f"{column}_abs_drift"))
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
    metrics["soft_row_entropy_shift_ok"] = bool(metrics["row_entropy_abs_delta_mean"] <= 0.25)
    metrics["soft_top1_probability_shift_ok"] = bool(metrics["top1_probability_abs_delta_mean"] <= 0.10)
    metrics["soft_target_collapse_shift_ok"] = bool(
        metrics["top_target_slice_id_fraction_abs_delta_mean"] <= 0.10
        and metrics["top_target_mouse_id_fraction_abs_delta_mean"] <= 0.10
    )
    metrics["soft_validation_pass"] = bool(
        metrics["soft_recall_pass"]
        and metrics["soft_top1_pass"]
        and metrics["soft_row_entropy_shift_ok"]
        and metrics["soft_top1_probability_shift_ok"]
        and metrics["soft_target_collapse_shift_ok"]
    )
    return pd.DataFrame([metrics])


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, np.ndarray)) else False:
        return None
    return value


def write_skip_outputs(paths: dict[str, Path], reason: str, status: CandidateNeighborBackendStatus) -> None:
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "SKIPPED",
        "reason": reason,
        "backend_status": asdict(status),
        "no_production_m3_edges": True,
        "no_global_markov_p": True,
        "no_gpcca_fate_branched_m5_regulator": True,
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    paths["report"].write_text(
        "# M3 ANN Validation Skipped\n\n"
        f"- Reason: {reason}\n"
        "- No package installation or conda modification was attempted.\n"
        "- No production M3 edges, full M3, global Markov P, GPCCA, fate probability, Branched NicheFlow, M5, or regulator analysis was run.\n",
        encoding="utf-8",
    )


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    figure_warnings: list[str],
    exact_metadata: dict[str, Any],
    ann_metadata: dict[str, Any],
) -> None:
    row = metrics.iloc[0].to_dict()
    lines = [
        "# M3 Exact-vs-pynndescent ANN Validation",
        "",
        "This is a bounded ANN validation output. It is not a production M3 edge shard and does not assemble global Markov P, run GPCCA, compute fate probabilities, run Branched NicheFlow, M5, or regulator analysis.",
        "",
        "## Selected Shard",
        f"- source_time: {row['source_time']}",
        f"- target_time: {row['target_time']}",
        f"- source_slice_id: {row['source_slice_id']}",
        f"- source sample size: {int(row['source_sample_size'])}",
        f"- target rows: {int(row['target_rows'])}",
        f"- candidate_k: {int(row['candidate_k'])}",
        "",
        "## Candidate Retrieval",
        f"- recall@30 mean: {row['recall_at_30_mean']:.6g}",
        f"- recall@30 median: {row['recall_at_30_median']:.6g}",
        f"- recall@30 p05/p95: {row['recall_at_30_p05']:.6g} / {row['recall_at_30_p95']:.6g}",
        f"- top1 agreement: {row['top1_agreement']:.6g}",
        f"- mean Jaccard overlap: {row['jaccard_overlap_mean']:.6g}",
        f"- mean distance-rank correlation: {row['distance_rank_correlation_mean']:.6g}",
        "",
        "## Downstream Local Transition Comparison",
        f"- row_normalized_transition_prob abs drift mean/p95: {row['row_normalized_transition_prob_abs_drift_mean']:.6g} / {row['row_normalized_transition_prob_abs_drift_p95']:.6g}",
        f"- row entropy exact/ANN/delta mean: {row['row_entropy_exact_mean']:.6g} / {row['row_entropy_ann_mean']:.6g} / {row['row_entropy_delta_mean']:.6g}",
        f"- top1 probability exact/ANN/delta mean: {row['top1_probability_exact_mean']:.6g} / {row['top1_probability_ann_mean']:.6g} / {row['top1_probability_delta_mean']:.6g}",
        f"- target slice entropy delta mean: {row['target_slice_id_entropy_delta_mean']:.6g}",
        f"- target mouse entropy delta mean: {row['target_mouse_id_entropy_delta_mean']:.6g}",
        f"- top target slice fraction delta mean: {row['top_target_slice_id_fraction_delta_mean']:.6g}",
        f"- top target mouse fraction delta mean: {row['top_target_mouse_id_fraction_delta_mean']:.6g}",
        "",
        "## Runtime And Memory",
        f"- sklearn_exact runtime seconds: {row['sklearn_exact_runtime_seconds']:.3f}",
        f"- pynndescent runtime seconds: {row['pynndescent_runtime_seconds']:.3f}",
        f"- runtime ratio ANN/exact: {row['runtime_ratio_ann_over_exact']:.6g}",
        f"- sklearn_exact max RSS GiB: {row['sklearn_exact_max_rss_gib']:.4f}",
        f"- pynndescent max RSS GiB: {row['pynndescent_max_rss_gib']:.4f}",
        f"- memory ratio ANN/exact: {row['memory_ratio_ann_over_exact']:.6g}",
        "",
        "## Soft Diagnostic Thresholds",
        f"- recall@30 >= 0.8: {bool(row['soft_recall_pass'])}",
        f"- top1 agreement >= 0.8: {bool(row['soft_top1_pass'])}",
        f"- row entropy shift ok: {bool(row['soft_row_entropy_shift_ok'])}",
        f"- top1 probability shift ok: {bool(row['soft_top1_probability_shift_ok'])}",
        f"- target collapse shift ok: {bool(row['soft_target_collapse_shift_ok'])}",
        f"- overall soft validation pass: {bool(row['soft_validation_pass'])}",
        "",
        "## Backend Metadata",
        f"- exact backend tau_pair: {exact_metadata.get('tau_pair')}",
        f"- ANN backend tau_pair: {ann_metadata.get('tau_pair')}",
    ]
    if figure_warnings:
        lines.extend(["", "## Figure Warnings", *[f"- {warning}" for warning in figure_warnings]])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(overlap["recall_at_k"], bins=30)
        ax.axvline(0.8, color="red", linestyle="--")
        ax.set_title("Recall@30 Distribution")
        ax.set_xlabel("recall@30")
        ax.set_ylabel("source anchors")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_recall_distribution.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.bar(["top1 agreement"], [float(row["top1_agreement"])])
        ax.axhline(0.8, color="red", linestyle="--")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_top1_agreement_summary.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(drift["row_normalized_transition_prob_abs_drift"], bins=40)
        ax.set_title("Probability Drift")
        ax.set_xlabel("absolute drift")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_probability_drift_distribution.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(row_diag["row_entropy_delta"], bins=40)
        ax.set_title("Row Entropy Delta")
        ax.set_xlabel("ANN - exact")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_entropy_delta_distribution.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].bar(["exact", "pynndescent"], [row["sklearn_exact_runtime_seconds"], row["pynndescent_runtime_seconds"]])
        axes[0].set_title("Runtime seconds")
        axes[1].bar(["exact", "pynndescent"], [row["sklearn_exact_max_rss_gib"], row["pynndescent_max_rss_gib"]])
        axes[1].set_title("Max RSS GiB")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_runtime_memory_comparison.png", dpi=140)
        plt.close(fig)

        labels = [
            "slice entropy",
            "mouse entropy",
            "top slice fraction",
            "top mouse fraction",
        ]
        values = [
            row["target_slice_id_entropy_delta_mean"],
            row["target_mouse_id_entropy_delta_mean"],
            row["top_target_slice_id_fraction_delta_mean"],
            row["top_target_mouse_id_fraction_delta_mean"],
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=25)
        ax.set_title("Target Diagnostic Delta")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_target_slice_mouse_diagnostic_delta.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Figure generation failed but validation tables were written: {exc}")
    return warnings


def run_validation(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    validate_requested_scope(args, config)
    paths = output_paths(args.output_dir, args.source_time, args.target_time, args.source_slice_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ann_status = inspect_candidate_neighbor_backend(args.ann_backend, run_toy_check=True)
    if not ann_status.available:
        reason = f"{args.ann_backend} is not usable: {ann_status.reason}"
        write_skip_outputs(paths, reason, ann_status)
        print(f"ANN_VALIDATION_SKIPPED {reason}")
        return 0

    shard = select_validation_shard(
        pd.read_csv(args.plan_csv),
        args.source_time,
        args.target_time,
        args.source_slice_id,
    )
    source, target, feature_groups, retrieval_columns, _ = load_validation_data(
        config,
        shard,
        args.source_time,
        args.target_time,
        int(args.sample_source_anchors),
        int(args.random_seed),
    )
    source_retrieval, target_retrieval, standardize_stats = standardize_feature_matrices(
        source[retrieval_columns].to_numpy(dtype=float),
        target[retrieval_columns].to_numpy(dtype=float),
        float(config["cost"]["min_scale"]),
    )
    metric = config["candidate_edges"].get("retrieval_metric", "euclidean")
    chunk_size = int(config["candidate_edges"].get("numpy_chunk_size", 512))
    exact_neighbors, exact_timing = run_backend(
        source_retrieval,
        target_retrieval,
        args.exact_backend,
        metric,
        int(args.candidate_k),
        chunk_size,
        int(args.random_seed),
    )
    if isinstance(exact_neighbors, CandidateNeighborBackendStatus):
        raise RuntimeError(f"Exact backend unavailable: {exact_neighbors.reason}")
    ann_neighbors, ann_timing = run_backend(
        source_retrieval,
        target_retrieval,
        args.ann_backend,
        metric,
        int(args.candidate_k),
        chunk_size,
        int(args.random_seed),
    )
    if isinstance(ann_neighbors, CandidateNeighborBackendStatus):
        write_skip_outputs(paths, ann_neighbors.reason, ann_neighbors)
        print(f"ANN_VALIDATION_SKIPPED {ann_neighbors.reason}")
        return 0

    validation_shard = dict(shard)
    validation_shard["source_rows"] = len(source)
    validation_shard["candidate_k"] = int(args.candidate_k)
    validation_shard["expected_edge_rows"] = len(source) * int(args.candidate_k)
    exact_edges, exact_metadata = build_validation_edges(
        source,
        target,
        validation_shard,
        config,
        feature_groups,
        exact_neighbors,
    )
    ann_edges, ann_metadata = build_validation_edges(
        source,
        target,
        validation_shard,
        config,
        feature_groups,
        ann_neighbors,
    )
    overlap = compare_candidate_sets(exact_neighbors, ann_neighbors, source)
    drift = drift_metrics(exact_edges, ann_edges)
    row_diag = compare_row_diagnostics(exact_edges, ann_edges)
    context = {
        "source_time": args.source_time,
        "target_time": args.target_time,
        "source_slice_id": args.source_slice_id,
        "source_slice_file": shard["source_slice_file"],
        "source_sample_size": len(source),
        "source_rows_full_shard": int(shard["source_rows"]),
        "target_rows": len(target),
        "candidate_k": int(args.candidate_k),
        "expected_validation_edge_rows": len(source) * int(args.candidate_k),
        "retrieval_feature_columns": len(retrieval_columns),
        "zero_variance_retrieval_columns": standardize_stats["zero_variance_columns"],
        "exact_backend": args.exact_backend,
        "ann_backend": args.ann_backend,
    }
    metrics = build_metrics(
        overlap,
        exact_neighbors,
        ann_neighbors,
        drift,
        row_diag,
        exact_timing,
        ann_timing,
        context,
    )

    metrics.to_csv(paths["metrics"], index=False)
    overlap.to_csv(paths["overlap"], index=False)
    if not args.skip_candidate_tables:
        candidate_table(source, target, exact_neighbors).to_parquet(paths["exact_candidates"], index=False)
        candidate_table(source, target, ann_neighbors).to_parquet(paths["ann_candidates"], index=False)
    figure_warnings = generate_figures(paths["figures_dir"], overlap, drift, row_diag, metrics)
    write_report(paths["report"], metrics, figure_warnings, exact_metadata, ann_metadata)
    summary = {
        "status": "COMPLETED",
        "metrics": metrics.iloc[0].to_dict(),
        "ann_backend_status": asdict(ann_status),
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
    paths["summary"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    row = metrics.iloc[0]
    print(f"ANN_VALIDATION_COMPLETED {args.source_time}->{args.target_time} {args.source_slice_id}")
    print(f"SOURCE_SAMPLE_SIZE {len(source)}")
    print(f"TARGET_ROWS {len(target)}")
    print(f"CANDIDATE_K {int(args.candidate_k)}")
    print(f"RECALL_AT_30_MEAN {row['recall_at_30_mean']:.6g}")
    print(f"TOP1_AGREEMENT {row['top1_agreement']:.6g}")
    print(f"SOFT_VALIDATION_PASS {bool(row['soft_validation_pass'])}")
    return 0


def main() -> int:
    return run_validation(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
