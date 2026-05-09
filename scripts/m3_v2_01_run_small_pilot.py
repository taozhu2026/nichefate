#!/usr/bin/env python
"""Run the bounded M3-v2 D9->D21 small pilot."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

for _thread_var in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_thread_var, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config
from nichefate.m3_v2_kernel import (
    exponential_gate,
    jensen_shannon_by_source,
    pairwise_l2_for_edges,
    robust_scale_fit,
    robust_scale_transform,
    row_normalize_weights,
    slice_mouse_gate,
    source_adaptive_tau,
    source_entropy_and_top1,
    validate_probabilities,
)


FORBIDDEN_OUTPUT_TOKENS = [
    "m4a",
    "m4c",
    "gpcca",
    "m4d",
    "m5",
    "regulator",
    "branchsbm",
    "branched_nicheflow",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_v2_pilot.yaml")
    return parser.parse_args()


def max_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / float(1024**2)


def validate_output_root(path: Path) -> None:
    resolved = path.resolve()
    forbidden_roots = [
        Path("/home/zhutao/scratch/nichefate/m3").resolve(),
        Path("/home/zhutao/scratch/nichefate/m4a").resolve(),
        Path("/home/zhutao/scratch/nichefate/m4b").resolve(),
        Path("/home/zhutao/scratch/nichefate/m4c").resolve(),
    ]
    for root in forbidden_roots:
        if resolved == root or root in resolved.parents:
            raise ValueError(f"Refusing to write M3-v2 pilot outputs under frozen v1 path: {resolved}")


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    validate_output_root(output_root)
    paths = {
        "root": output_root,
        "reports": output_root / "reports",
        "figures": output_root / "reports" / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_feature_groups(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text())
    return {name: list(columns) for name, columns in payload["feature_groups"].items()}


def selected_feature_columns(config: dict[str, Any], groups: dict[str, list[str]]) -> dict[str, list[str]]:
    features = config["features"]
    state = groups[features["state_feature_group"]][: int(features["state_feature_limit"])]
    composition = groups[features["composition_feature_group"]][
        : int(features["composition_feature_limit"])
    ]
    spatial_topology: list[str] = []
    for group in features["spatial_topology_feature_groups"]:
        spatial_topology.extend(groups[group])
    return {
        "state": state,
        "composition": composition,
        "spatial_topology": list(dict.fromkeys(spatial_topology)),
    }


def read_refined_mapping(path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(path)
    return mapping[
        [
            "raw_terminal_macrostate",
            "refined_endpoint_id",
            "refined_endpoint_label",
            "confidence_tier_after_refinement",
        ]
    ].rename(columns={"raw_terminal_macrostate": "dominant_fate"})


def load_m4e_source_metadata(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["paths"]["m4e_node_neighborhood"])
    columns = [
        "anchor_id",
        "slice_id",
        "mouse_id",
        "time_label",
        "leiden_neigh",
        "cell_type_l1",
        "cell_type_l3",
        "x",
        "y",
    ]
    meta = pd.read_parquet(path, columns=columns)
    source_time = str(config["pilot"]["source_time"])
    return meta[meta["time_label"].astype(str) == source_time].copy()


def stratified_source_sample(meta: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    meta = meta.reset_index(drop=True)
    if len(meta) <= cap:
        sample = meta.copy()
        sample["sampling_stratum"] = (
            sample["slice_id"].astype(str)
            + "|"
            + sample["mouse_id"].astype(str)
            + "|"
            + sample["leiden_neigh"].astype(str)
            + "|"
            + sample["cell_type_l1"].astype(str)
        )
        return sample
    work = meta.copy()
    work["sampling_stratum"] = (
        work["slice_id"].astype(str)
        + "|"
        + work["mouse_id"].astype(str)
        + "|"
        + work["leiden_neigh"].astype(str)
        + "|"
        + work["cell_type_l1"].astype(str)
    )
    groups = work.groupby("sampling_stratum", sort=True).indices
    sizes = pd.Series({key: len(indices) for key, indices in groups.items()}, dtype=float)
    ideal = sizes / sizes.sum() * int(cap)
    counts = np.floor(ideal).astype(int)
    if len(counts) <= cap:
        counts[counts == 0] = 1
    while int(counts.sum()) > cap:
        candidates = counts[counts > 0].sort_values(ascending=True)
        counts.loc[candidates.index[0]] -= 1
    residual = (ideal - np.floor(ideal)).sort_values(ascending=False)
    remaining = int(cap - counts.sum())
    for key in residual.index:
        if remaining <= 0:
            break
        if counts.loc[key] < sizes.loc[key]:
            counts.loc[key] += 1
            remaining -= 1
    rng = np.random.default_rng(int(seed))
    selected_indices: list[int] = []
    for key, indices in groups.items():
        n = int(counts.loc[key])
        if n <= 0:
            continue
        selected_indices.extend(rng.choice(indices, size=n, replace=False).tolist())
    sample = work.loc[selected_indices].sort_values("anchor_id").reset_index(drop=True)
    if len(sample) != cap:
        raise RuntimeError(f"Sampling produced {len(sample)} rows, expected {cap}.")
    return sample


def discover_edge_shards(root: Path) -> dict[str, Path]:
    shards = {}
    for path in sorted(root.glob("*/candidate_edges_D9_to_D21__*.parquet")):
        shards[path.parent.name] = path
    if not shards:
        raise FileNotFoundError(f"No frozen M3-v1 D9->D21 edge shards found under {root}")
    return shards


def load_selected_v1_edges(edge_root: Path, sampled_sources: pd.DataFrame) -> pd.DataFrame:
    shards = discover_edge_shards(edge_root)
    frames: list[pd.DataFrame] = []
    sampled_source_ids = set(sampled_sources["anchor_id"].astype(str))
    for slice_id, selected in sampled_sources.groupby("slice_id", sort=True):
        if slice_id not in shards:
            raise FileNotFoundError(f"Missing M3-v1 edge shard for sampled source slice {slice_id}")
        wanted = set(selected["anchor_id"].astype(str))
        frame = pd.read_parquet(shards[str(slice_id)])
        frame = frame[frame["source_anchor_id"].isin(wanted)].copy()
        frames.append(frame)
    edges = pd.concat(frames, ignore_index=True)
    found_source_ids = set(edges["source_anchor_id"].astype(str))
    missing_sources = sampled_source_ids - found_source_ids
    if missing_sources:
        first_missing = sorted(missing_sources)[0]
        raise RuntimeError(
            f"Missing candidate edges for {len(missing_sources)} sampled source anchors; "
            f"first missing source: {first_missing}"
        )
    per_source_counts = edges.groupby("source_anchor_id", sort=False).size()
    if per_source_counts.nunique() != 1:
        raise RuntimeError(
            "Selected M3-v1 candidate edge count is not constant per sampled source: "
            f"min={int(per_source_counts.min())}, max={int(per_source_counts.max())}"
        )
    expected = len(sampled_sources) * int(per_source_counts.iloc[0])
    if len(edges) != expected:
        raise RuntimeError(f"Expected {expected} selected candidate edges, found {len(edges)}.")
    return edges


def m2_slice_path(m2_root: Path, slice_id: str) -> Path:
    return m2_root / slice_id / f"m2_representation_{slice_id}.parquet"


def load_m2_features(
    m2_root: Path,
    slice_ids: list[str],
    anchor_ids: set[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    frames = []
    read_columns = ["slice_id", "anchor_index", *feature_columns]
    for slice_id in sorted(set(slice_ids)):
        path = m2_slice_path(m2_root, slice_id)
        frame = pd.read_parquet(path, columns=read_columns)
        frame["anchor_id"] = frame["slice_id"].astype(str) + "::" + frame["anchor_index"].astype(str)
        frame = frame[frame["anchor_id"].isin(anchor_ids)].copy()
        frames.append(frame[["anchor_id", *feature_columns]])
    out = pd.concat(frames, ignore_index=True)
    if out["anchor_id"].duplicated().any():
        raise ValueError("Duplicate M2 anchor_id rows after feature loading.")
    return out.set_index("anchor_id")


def aligned_feature_matrix(frame: pd.DataFrame, anchor_order: pd.Index, columns: list[str]) -> np.ndarray:
    missing = sorted(set(anchor_order) - set(frame.index))
    if missing:
        raise KeyError(f"Missing {len(missing)} M2 feature rows; first missing anchor: {missing[0]}")
    return frame.loc[anchor_order, columns].to_numpy(dtype=np.float32)


def compute_distance_block(
    source_features: pd.DataFrame,
    target_features: pd.DataFrame,
    source_order: pd.Index,
    target_order: pd.Index,
    source_pos: np.ndarray,
    target_pos: np.ndarray,
    columns: list[str],
    chunk_size: int,
    min_scale: float,
) -> tuple[np.ndarray, int]:
    if not columns:
        return np.zeros(len(source_pos), dtype=np.float32), 0
    src = aligned_feature_matrix(source_features, source_order, columns)
    tgt = aligned_feature_matrix(target_features, target_order, columns)
    stats = robust_scale_fit([src, tgt], min_scale=min_scale)
    src_scaled = robust_scale_transform(src, stats)
    tgt_scaled = robust_scale_transform(tgt, stats)
    distances = pairwise_l2_for_edges(src_scaled, tgt_scaled, source_pos, target_pos, chunk_size)
    return distances, stats.zero_scale_columns


def load_annotation_frames(config: dict[str, Any], anchor_ids: set[str]) -> pd.DataFrame:
    m4e_cols = [
        "anchor_id",
        "global_node_index",
        "slice_id",
        "mouse_id",
        "time_label",
        "leiden_neigh",
        "cell_type_l1",
        "cell_type_l3",
        "x",
        "y",
    ]
    m4c_cols = [
        "anchor_id",
        "dominant_fate",
        "dominant_fate_label",
        "dominant_fate_probability",
        "normalized_plasticity_entropy",
    ]
    m4e = pd.read_parquet(config["paths"]["m4e_node_neighborhood"], columns=m4e_cols)
    m4e = m4e[m4e["anchor_id"].isin(anchor_ids)].copy()
    m4c = pd.read_parquet(config["paths"]["m4c_node_summary"], columns=m4c_cols)
    m4c = m4c[m4c["anchor_id"].isin(anchor_ids)].copy()
    mapping = read_refined_mapping(Path(config["paths"]["refined_endpoint_mapping"]))
    m4c = m4c.merge(mapping, on="dominant_fate", how="left")
    out = m4e.merge(m4c, on="anchor_id", how="left")
    if out["anchor_id"].duplicated().any():
        raise ValueError("Duplicate anchor_id rows in annotation join.")
    return out.set_index("anchor_id")


def add_annotation_columns(edges: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "leiden_neigh",
        "cell_type_l1",
        "cell_type_l3",
        "x",
        "y",
        "dominant_fate",
        "dominant_fate_label",
        "dominant_fate_probability",
        "normalized_plasticity_entropy",
        "refined_endpoint_id",
        "refined_endpoint_label",
        "confidence_tier_after_refinement",
    ]
    for prefix, key in [("source", "source_anchor_id"), ("target", "target_anchor_id")]:
        joined = annotations.reindex(edges[key].astype(str))[columns].reset_index(drop=True)
        for col in columns:
            edges[f"{prefix}_{col}"] = joined[col].to_numpy()
    return edges


def weighted_category_distribution(edges: pd.DataFrame, probability_col: str, category_col: str) -> pd.Series:
    values = edges[[category_col, probability_col]].copy()
    values[category_col] = values[category_col].fillna("NA").astype(str)
    dist = values.groupby(category_col)[probability_col].sum()
    total = float(dist.sum())
    return dist / total if total > 0 else dist


def normalized_entropy(distribution: pd.Series) -> float:
    probs = distribution[distribution > 0].to_numpy(dtype=float)
    if len(probs) <= 1:
        return 0.0
    entropy = float(-(probs * np.log(probs)).sum())
    return entropy / float(np.log(len(probs)))


def top_targets(edges: pd.DataFrame, probability_col: str) -> pd.DataFrame:
    idx = edges.groupby("source_anchor_id", sort=False)[probability_col].idxmax()
    return edges.loc[idx].copy().reset_index(drop=True)


def agreement_rate(frame: pd.DataFrame, left: str, right: str) -> float:
    valid = frame[left].notna() & frame[right].notna()
    if not bool(valid.any()):
        return float("nan")
    return float((frame.loc[valid, left].astype(str) == frame.loc[valid, right].astype(str)).mean())


def weighted_numeric_mean(edges: pd.DataFrame, probability_col: str, value_col: str) -> float:
    values = pd.to_numeric(edges[value_col], errors="coerce")
    weights = pd.to_numeric(edges[probability_col], errors="coerce")
    valid = values.notna() & weights.notna() & (weights >= 0)
    if not bool(valid.any()):
        return float("nan")
    denominator = float(weights[valid].sum())
    if denominator <= 0:
        return float("nan")
    return float((values[valid] * weights[valid]).sum() / denominator)


def same_cell_type_neighborhood_fate_separation(top: pd.DataFrame) -> float:
    required = ["source_cell_type_l3", "source_leiden_neigh", "target_refined_endpoint_id"]
    frame = top.dropna(subset=required).copy()
    if frame.empty:
        return float("nan")
    dominant = (
        frame.groupby(["source_cell_type_l3", "source_leiden_neigh"], sort=False)[
            "target_refined_endpoint_id"
        ]
        .agg(lambda values: values.astype(str).value_counts().index[0])
        .reset_index()
    )
    by_cell = dominant.groupby("source_cell_type_l3", sort=False).agg(
        source_neighborhoods=("source_leiden_neigh", "nunique"),
        target_endpoints=("target_refined_endpoint_id", "nunique"),
    )
    eligible = by_cell[by_cell["source_neighborhoods"] > 1].copy()
    if eligible.empty:
        return float("nan")
    separation = (eligible["target_endpoints"] - 1) / (eligible["source_neighborhoods"] - 1)
    return float(separation.clip(lower=0.0, upper=1.0).mean())


def sorted_spatial_smoothness(top: pd.DataFrame, endpoint_col: str) -> float:
    values = []
    for _, group in top.dropna(subset=["source_x", "source_y"]).groupby("source_slice_id", sort=False):
        ordered = group.sort_values(["source_x", "source_y"])
        labels = ordered[endpoint_col].fillna("NA").astype(str).to_numpy()
        if len(labels) > 1:
            values.append(float((labels[1:] == labels[:-1]).mean()))
    return float(np.mean(values)) if values else float("nan")


def metric_summary(edges: pd.DataFrame, source_level: pd.DataFrame, runtime: dict[str, float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    v1_top = top_targets(edges, "row_normalized_transition_prob")
    v2_top = top_targets(edges, "v2_row_normalized_transition_prob")
    v1_slice = weighted_category_distribution(edges, "row_normalized_transition_prob", "target_slice_id")
    v2_slice = weighted_category_distribution(edges, "v2_row_normalized_transition_prob", "target_slice_id")
    v1_mouse = weighted_category_distribution(edges, "row_normalized_transition_prob", "target_mouse_id")
    v2_mouse = weighted_category_distribution(edges, "v2_row_normalized_transition_prob", "target_mouse_id")
    v1_leiden = weighted_category_distribution(edges, "row_normalized_transition_prob", "target_leiden_neigh")
    v2_leiden = weighted_category_distribution(edges, "v2_row_normalized_transition_prob", "target_leiden_neigh")

    v1_entropy = float(source_level["v1_transition_entropy"].mean())
    v2_entropy = float(source_level["v2_transition_entropy"].mean())
    v1_top1 = float(source_level["v1_top1_probability"].mean())
    v2_top1 = float(source_level["v2_top1_probability"].mean())
    v1_collapse = max(float(v1_slice.max()), float(v1_mouse.max()))
    v2_collapse = max(float(v2_slice.max()), float(v2_mouse.max()))

    rows = [
        {
            "metric_name": "top-target Leiden_neigh consistency",
            "v1_value": agreement_rate(v1_top, "source_leiden_neigh", "target_leiden_neigh"),
            "v2_value": agreement_rate(v2_top, "source_leiden_neigh", "target_leiden_neigh"),
            "preferred_direction": "higher",
            "status": "computed",
            "notes": "Top target label agreement between source and target Leiden neighborhood.",
        },
        {
            "metric_name": "top-target fine cell cluster consistency",
            "v1_value": agreement_rate(v1_top, "source_cell_type_l3", "target_cell_type_l3"),
            "v2_value": agreement_rate(v2_top, "source_cell_type_l3", "target_cell_type_l3"),
            "preferred_direction": "higher_or_explainable",
            "status": "computed",
            "notes": "Top target label agreement for fine cell-type cluster.",
        },
        {
            "metric_name": "source-target refined endpoint plausibility",
            "v1_value": agreement_rate(v1_top, "source_refined_endpoint_id", "target_refined_endpoint_id"),
            "v2_value": agreement_rate(v2_top, "source_refined_endpoint_id", "target_refined_endpoint_id"),
            "preferred_direction": "higher_or_explainable",
            "status": "computed",
            "notes": "Top target agreement with existing M4E refined endpoint IDs.",
        },
        {
            "metric_name": "transition entropy / top1 concentration",
            "v1_value": v1_entropy,
            "v2_value": v2_entropy,
            "preferred_direction": "moderate_non_degenerate",
            "status": "computed",
            "notes": f"Mean entropy shown; mean top1 v1={v1_top1:.4f}, v2={v2_top1:.4f}.",
        },
        {
            "metric_name": "spatial smoothness of predicted dominant endpoint",
            "v1_value": sorted_spatial_smoothness(v1_top, "target_refined_endpoint_id"),
            "v2_value": sorted_spatial_smoothness(v2_top, "target_refined_endpoint_id"),
            "preferred_direction": "higher_without_over_smoothing",
            "status": "computed",
            "notes": "Sorted-coordinate neighbor agreement proxy within each source slice.",
        },
        {
            "metric_name": "slice/mouse collapse diagnostics",
            "v1_value": v1_collapse,
            "v2_value": v2_collapse,
            "preferred_direction": "lower",
            "status": "computed",
            "notes": "Maximum of target slice and mouse probability-mass concentration.",
        },
        {
            "metric_name": "target-neighborhood diversity",
            "v1_value": normalized_entropy(v1_leiden),
            "v2_value": normalized_entropy(v2_leiden),
            "preferred_direction": "higher_bounded",
            "status": "computed",
            "notes": "Normalized entropy of target Leiden-neighborhood probability mass.",
        },
        {
            "metric_name": "stability across random seeds/subsamples",
            "v1_value": np.nan,
            "v2_value": np.nan,
            "preferred_direction": "higher_stability",
            "status": "not_computed_single_seed_pilot",
            "notes": "Requires repeat pilot runs with additional seeds/subsamples.",
        },
        {
            "metric_name": "endpoint-attraction agreement with M4C-v1",
            "v1_value": agreement_rate(v1_top, "source_refined_endpoint_id", "target_refined_endpoint_id"),
            "v2_value": agreement_rate(v2_top, "source_refined_endpoint_id", "target_refined_endpoint_id"),
            "preferred_direction": "preserve_or_improve",
            "status": "computed",
            "notes": "Uses existing M4C/M4E endpoint labels only; no M4C probabilities recomputed.",
        },
        {
            "metric_name": "plasticity enrichment in transition/ulcer/repair-like neighborhoods",
            "v1_value": weighted_numeric_mean(
                edges, "row_normalized_transition_prob", "target_normalized_plasticity_entropy"
            ),
            "v2_value": weighted_numeric_mean(
                edges, "v2_row_normalized_transition_prob", "target_normalized_plasticity_entropy"
            ),
            "preferred_direction": "higher_in_expected_neighborhoods",
            "status": "proxy_computed",
            "notes": "Weighted target plasticity mean proxy; curated transition/ulcer/repair-like neighborhood set was not introduced.",
        },
        {
            "metric_name": "same-anchor-cell-type, different-neighborhood fate separation",
            "v1_value": same_cell_type_neighborhood_fate_separation(v1_top),
            "v2_value": same_cell_type_neighborhood_fate_separation(v2_top),
            "preferred_direction": "higher_if_biologically_supported",
            "status": "proxy_computed",
            "notes": "Proxy over top-target refined endpoint diversity across source neighborhoods within each fine cell type.",
        },
        {
            "metric_name": "computational runtime/memory",
            "v1_value": np.nan,
            "v2_value": runtime["runtime_seconds"],
            "preferred_direction": "bounded",
            "status": "computed_v2_only",
            "notes": f"Runtime seconds shown in v2_value; peak RSS GiB={runtime['max_rss_gib']:.3f}.",
        },
    ]
    summary = pd.DataFrame(rows)
    summary["delta_v2_minus_v1"] = [
        v2 - v1 if np.isfinite(v1) and np.isfinite(v2) else np.nan
        for v1, v2 in zip(summary["v1_value"], summary["v2_value"])
    ]
    summary = summary[
        [
            "metric_name",
            "v1_value",
            "v2_value",
            "delta_v2_minus_v1",
            "preferred_direction",
            "status",
            "notes",
        ]
    ]
    details = {
        "v1_entropy_mean": v1_entropy,
        "v2_entropy_mean": v2_entropy,
        "v1_top1_probability_mean": v1_top1,
        "v2_top1_probability_mean": v2_top1,
        "v1_top_slice_fraction": float(v1_slice.max()),
        "v2_top_slice_fraction": float(v2_slice.max()),
        "v1_top_mouse_fraction": float(v1_mouse.max()),
        "v2_top_mouse_fraction": float(v2_mouse.max()),
        "v1_target_leiden_distribution": v1_leiden.to_dict(),
        "v2_target_leiden_distribution": v2_leiden.to_dict(),
        "mean_js_divergence": float(source_level["v1_v2_js_divergence"].mean()),
        "v1_weighted_target_plasticity_mean": weighted_numeric_mean(
            edges, "row_normalized_transition_prob", "target_normalized_plasticity_entropy"
        ),
        "v2_weighted_target_plasticity_mean": weighted_numeric_mean(
            edges, "v2_row_normalized_transition_prob", "target_normalized_plasticity_entropy"
        ),
    }
    return summary, details


def decision_preview(summary: pd.DataFrame, details: dict[str, Any], row_qc: dict[str, Any]) -> str:
    if not row_qc["row_sum_pass"]:
        return "revise_v2_and_repeat_pilot"
    collapse_delta = float(details["v2_top_mouse_fraction"] - details["v1_top_mouse_fraction"])
    entropy_delta = float(details["v2_entropy_mean"] - details["v1_entropy_mean"])
    plaus = summary[summary["metric_name"] == "source-target refined endpoint plausibility"].iloc[0]
    plaus_delta = float(plaus["delta_v2_minus_v1"])
    if collapse_delta < -0.02 and plaus_delta >= -0.03:
        return "keep_v1_and_v2_as_complementary"
    if collapse_delta > 0.05 or entropy_delta < -0.2 or plaus_delta < -0.05:
        return "keep_v1_as_main_baseline"
    return "revise_v2_and_repeat_pilot"


def write_figures(edges: pd.DataFrame, source_level: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cols, title, path_name in [
        (["v1_transition_entropy", "v2_transition_entropy"], "v1 vs v2 transition entropy", "v1_v2_transition_entropy_distribution.png"),
        (["v1_top1_probability", "v2_top1_probability"], "v1 vs v2 top1 probability", "v1_v2_top1_probability_distribution.png"),
        (["v1_v2_js_divergence"], "source-level v1-v2 divergence", "source_level_v1_v2_js_divergence.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for col in cols:
            ax.hist(source_level[col].dropna(), bins=40, alpha=0.55, label=col)
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / path_name, dpi=dpi)
        plt.close(fig)

    for category, path_name, title in [
        ("target_leiden_neigh", "target_leiden_neigh_composition_comparison.png", "Target Leiden neighborhood mass"),
        ("target_cell_type_l3", "target_fine_cluster_composition_comparison.png", "Target fine cluster mass"),
        ("target_slice_id", "target_slice_concentration_comparison.png", "Target slice mass"),
        ("target_mouse_id", "target_mouse_concentration_comparison.png", "Target mouse mass"),
    ]:
        v1 = weighted_category_distribution(edges, "row_normalized_transition_prob", category)
        v2 = weighted_category_distribution(edges, "v2_row_normalized_transition_prob", category)
        labels = sorted(set(v1.index).union(set(v2.index)))[:18]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(7, 0.38 * len(labels) + 4), 4.5))
        ax.bar(x - 0.2, [v1.get(label, 0.0) for label in labels], width=0.4, label="v1")
        ax.bar(x + 0.2, [v2.get(label, 0.0) for label in labels], width=0.4, label="v2")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("probability mass fraction")
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / path_name, dpi=dpi)
        plt.close(fig)

    source_slice = edges["source_slice_id"].value_counts().index[0]
    sub = edges[edges["source_slice_id"] == source_slice].copy()
    labels = sorted(set(sub["target_leiden_neigh"].fillna("NA").astype(str)))
    v1 = weighted_category_distribution(sub, "row_normalized_transition_prob", "target_leiden_neigh")
    v2 = weighted_category_distribution(sub, "v2_row_normalized_transition_prob", "target_leiden_neigh")
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.2, [v1.get(label, 0.0) for label in labels], width=0.4, label="v1")
    ax.bar(x + 0.2, [v2.get(label, 0.0) for label in labels], width=0.4, label="v2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("probability mass fraction")
    ax.set_title(f"Representative source slice transition summary: {source_slice}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "representative_source_slice_transition_summary.png", dpi=dpi)
    plt.close(fig)


def write_reports(
    paths: dict[str, Path],
    summary: pd.DataFrame,
    details: dict[str, Any],
    row_qc: dict[str, Any],
    decision: str,
    counts: dict[str, Any],
) -> None:
    metric_table = summary.copy()
    for col in ["v1_value", "v2_value", "delta_v2_minus_v1"]:
        metric_table[col] = metric_table[col].map(
            lambda value: f"{float(value):.4g}" if pd.notna(value) else "NA"
        )
    metric_table["status"] = metric_table["status"].fillna("")
    metric_table["notes"] = metric_table["notes"].fillna("")
    table_lines = [
        "| metric_name | v1_value | v2_value | delta_v2_minus_v1 | preferred_direction | status | notes |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in metric_table.itertuples(index=False):
        table_lines.append(
            f"| {row.metric_name} | {row.v1_value} | {row.v2_value} | "
            f"{row.delta_v2_minus_v1} | {row.preferred_direction} | {row.status} | {row.notes} |"
        )
    markdown_table = "\n".join(table_lines)
    report = f"""# M3-v2 Small Pilot Report

## Scope

- Transition pair: D9 -> D21
- Source anchors: {counts['source_anchor_count']}
- Candidate edges: {counts['candidate_edge_count']}
- Candidate source: frozen M3-v1 D9->D21 candidate edges
- Barcode mode: pseudo-only, `G_barcode = 1.0`

No M4A-v2 assembly, M4C-v2 propagation, pyGPCCA, M4D diagnostics, K_gpcca, M5/regulator, BranchSBM / Branched NicheFlow, or barcode preprocessing was run.

## Row-Sum QC

- finite: {row_qc['finite']}
- nonnegative: {row_qc['nonnegative']}
- row_sum_max_abs_error: {row_qc['row_sum_max_abs_error']:.6g}
- row_sum_pass: {row_qc['row_sum_pass']}

## Key Metrics

{markdown_table}
"""
    (paths["reports"] / "m3_v2_pilot_report.md").write_text(report)

    comparison = f"""# M3-v1 vs M3-v2 Pilot Comparison

M3-v2 reweights the same selected M3-v1 candidate edge set. Changes therefore reflect v2 kernel weighting rather than candidate retrieval changes.

- v1 mean entropy: {details['v1_entropy_mean']:.4f}
- v2 mean entropy: {details['v2_entropy_mean']:.4f}
- v1 mean top1 probability: {details['v1_top1_probability_mean']:.4f}
- v2 mean top1 probability: {details['v2_top1_probability_mean']:.4f}
- v1 top slice fraction: {details['v1_top_slice_fraction']:.4f}
- v2 top slice fraction: {details['v2_top_slice_fraction']:.4f}
- v1 top mouse fraction: {details['v1_top_mouse_fraction']:.4f}
- v2 top mouse fraction: {details['v2_top_mouse_fraction']:.4f}
- mean source-level JS divergence: {details['mean_js_divergence']:.4f}
"""
    (paths["reports"] / "m3_v1_vs_v2_pilot_comparison.md").write_text(comparison)

    decision_text = f"""# M3-v2 Pilot Decision Preview

Decision preview: `{decision}`

This is not a full adoption decision. M3-v2 must not replace M3-v1 merely because it is more sophisticated. The next step should be chosen after reviewing the pilot metrics and figures.
"""
    (paths["reports"] / "m3_v2_pilot_decision_preview.md").write_text(decision_text)


def run_pilot(config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    output_root = Path(config["paths"]["output_root"])
    paths = ensure_dirs(output_root)
    groups = load_feature_groups(Path(config["paths"]["m3_feature_groups"]))
    feature_cols = selected_feature_columns(config, groups)
    all_feature_cols = list(dict.fromkeys(feature_cols["state"] + feature_cols["composition"] + feature_cols["spatial_topology"]))

    source_meta = load_m4e_source_metadata(config)
    sampled_sources = stratified_source_sample(
        source_meta,
        int(config["pilot"]["source_anchor_cap"]),
        int(config["pilot"]["random_seed"]),
    )
    sampled_sources.to_csv(output_root / "pilot_source_anchor_sample.csv", index=False)

    edges = load_selected_v1_edges(Path(config["paths"]["m3_v1_d9_d21_edges_root"]), sampled_sources)
    edges.to_parquet(output_root / "pilot_candidate_edges_v1_reference.parquet", index=False)

    source_order = pd.Index(pd.factorize(edges["source_anchor_id"], sort=False)[1].astype(str))
    target_order = pd.Index(pd.factorize(edges["target_anchor_id"], sort=False)[1].astype(str))
    source_pos = pd.Series(np.arange(len(source_order), dtype=np.int32), index=source_order)
    target_pos = pd.Series(np.arange(len(target_order), dtype=np.int32), index=target_order)
    edge_source_pos = edges["source_anchor_id"].map(source_pos).to_numpy(dtype=np.int32)
    edge_target_pos = edges["target_anchor_id"].map(target_pos).to_numpy(dtype=np.int32)
    source_codes = edge_source_pos

    m2_root = Path(config["paths"]["m2_by_slice_dir"])
    source_features = load_m2_features(
        m2_root,
        sorted(edges["source_slice_id"].astype(str).unique()),
        set(source_order),
        all_feature_cols,
    )
    target_features = load_m2_features(
        m2_root,
        sorted(edges["target_slice_id"].astype(str).unique()),
        set(target_order),
        all_feature_cols,
    )

    kernel_config = config["kernel"]
    chunk_size = int(kernel_config["distance_chunk_size"])
    min_scale = float(kernel_config["min_scale"])
    d_state, zero_state = compute_distance_block(
        source_features,
        target_features,
        source_order,
        target_order,
        edge_source_pos,
        edge_target_pos,
        feature_cols["state"],
        chunk_size,
        min_scale,
    )
    tau_state = source_adaptive_tau(
        d_state,
        source_codes,
        quantile=float(kernel_config["tau_quantile"]),
        min_tau=float(kernel_config["min_tau"]),
    )
    g_state = exponential_gate(d_state, tau_state, strength=1.0)

    d_comp, zero_comp = compute_distance_block(
        source_features,
        target_features,
        source_order,
        target_order,
        edge_source_pos,
        edge_target_pos,
        feature_cols["composition"],
        chunk_size,
        min_scale,
    )
    tau_comp = source_adaptive_tau(d_comp, source_codes, min_tau=float(kernel_config["min_tau"]))
    g_comp = exponential_gate(d_comp, tau_comp, strength=float(kernel_config["composition_gate_strength"]))

    d_spatial, zero_spatial = compute_distance_block(
        source_features,
        target_features,
        source_order,
        target_order,
        edge_source_pos,
        edge_target_pos,
        feature_cols["spatial_topology"],
        chunk_size,
        min_scale,
    )
    tau_spatial = source_adaptive_tau(d_spatial, source_codes, min_tau=float(kernel_config["min_tau"]))
    g_spatial = exponential_gate(
        d_spatial,
        tau_spatial,
        strength=float(kernel_config["spatial_topology_gate_strength"]),
    )
    g_slice_mouse = slice_mouse_gate(
        edges["target_slice_id"],
        edges["target_mouse_id"],
        strength=float(kernel_config["slice_mouse_gate_strength"]),
        min_gate=float(kernel_config["slice_mouse_gate_min"]),
    )
    g_time = np.ones(len(edges), dtype=np.float32)
    g_barcode = np.full(len(edges), float(kernel_config["barcode_gate"]), dtype=np.float32)

    weights = g_state * g_time * g_comp * g_spatial * g_slice_mouse * g_barcode
    v2_prob = row_normalize_weights(weights, source_codes)
    row_qc = validate_probabilities(v2_prob, source_codes)

    all_anchor_ids = set(source_order).union(set(target_order))
    annotations = load_annotation_frames(config, all_anchor_ids)
    edges = add_annotation_columns(edges, annotations)
    edges["v2_d_state"] = d_state
    edges["v2_tau_state"] = tau_state
    edges["v2_g_time"] = g_time
    edges["v2_d_composition"] = d_comp
    edges["v2_g_composition"] = g_comp
    edges["v2_d_spatial_topology"] = d_spatial
    edges["v2_g_spatial_topology"] = g_spatial
    edges["v2_g_slice_mouse"] = g_slice_mouse
    edges["v2_g_barcode"] = g_barcode
    edges["v2_unnormalized_weight"] = weights
    edges["v2_row_normalized_transition_prob"] = v2_prob
    edges.to_parquet(output_root / "pilot_candidate_edges_v2_reweighted.parquet", index=False)

    v1_stats = source_entropy_and_top1(edges["row_normalized_transition_prob"].to_numpy(), source_codes)
    v2_stats = source_entropy_and_top1(v2_prob, source_codes)
    js = jensen_shannon_by_source(edges["row_normalized_transition_prob"].to_numpy(), v2_prob, source_codes)
    source_lookup = pd.DataFrame({"source_code": np.arange(len(source_order)), "source_anchor_id": source_order})
    source_level = (
        source_lookup.merge(v1_stats.rename(columns={"transition_entropy": "v1_transition_entropy", "top1_probability": "v1_top1_probability"}), on="source_code")
        .merge(v2_stats.rename(columns={"transition_entropy": "v2_transition_entropy", "top1_probability": "v2_top1_probability"}), on="source_code")
        .merge(js, on="source_code")
    )
    source_level = source_level.merge(
        sampled_sources[["anchor_id", "slice_id", "mouse_id", "leiden_neigh", "cell_type_l1", "cell_type_l3"]].rename(
            columns={
                "anchor_id": "source_anchor_id",
                "slice_id": "source_slice_id",
                "mouse_id": "source_mouse_id",
                "leiden_neigh": "source_leiden_neigh",
                "cell_type_l1": "source_cell_type_l1",
                "cell_type_l3": "source_cell_type_l3",
            }
        ),
        on="source_anchor_id",
        how="left",
    )
    source_level.to_parquet(output_root / "pilot_source_level_v1_v2_comparison.parquet", index=False)

    runtime = {
        "runtime_seconds": float(time.time() - started),
        "max_rss_gib": max_rss_gib(),
    }
    summary, details = metric_summary(edges, source_level, runtime)
    decision = decision_preview(summary, details, row_qc)
    summary.to_csv(output_root / "pilot_metric_summary.csv", index=False)
    qc_summary = pd.DataFrame(
        [
            {"metric": "source_anchor_count", "value": len(source_order)},
            {"metric": "candidate_edge_count", "value": len(edges)},
            {"metric": "v2_row_sum_max_abs_error", "value": row_qc["row_sum_max_abs_error"]},
            {"metric": "v2_row_sum_pass", "value": row_qc["row_sum_pass"]},
            {"metric": "zero_scale_state_columns", "value": zero_state},
            {"metric": "zero_scale_composition_columns", "value": zero_comp},
            {"metric": "zero_scale_spatial_topology_columns", "value": zero_spatial},
            {"metric": "decision_preview", "value": decision},
        ]
    )
    qc_summary.to_csv(paths["reports"] / "m3_v2_pilot_qc_summary.csv", index=False)

    payload = {
        "source_anchor_count": int(len(source_order)),
        "candidate_edge_count": int(len(edges)),
        "row_qc": row_qc,
        "details": details,
        "decision_preview": decision,
        "runtime": runtime,
        "feature_counts": {key: len(value) for key, value in feature_cols.items()},
    }
    (output_root / "pilot_metric_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (output_root / "pilot_config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=True))

    write_figures(edges, source_level, paths["figures"], int(config["figures"]["dpi"]))
    write_reports(
        paths,
        summary,
        details,
        row_qc,
        decision,
        {"source_anchor_count": len(source_order), "candidate_edge_count": len(edges)},
    )
    return payload


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    validate_output_root(output_root)
    forbidden = "\n".join(FORBIDDEN_OUTPUT_TOKENS)
    if any(token in str(output_root).lower() for token in forbidden.splitlines()):
        raise ValueError(f"Forbidden output token in output root: {output_root}")
    payload = run_pilot(config)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
