#!/usr/bin/env python
"""Run M3-v2-03 confirmatory constrained pilots."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.m3_v2_kernel import (  # noqa: E402
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


OUTPUT_ROOT = Path("/home/zhutao/scratch/nichefate/m3_v2_pilot_confirmatory")
M3_V2_01_ROOT = Path("/home/zhutao/scratch/nichefate/m3_v2_pilot")
M3_V2_02_ROOT = Path("/home/zhutao/scratch/nichefate/m3_v2_pilot_tuning")
M3_FULL_BY_SHARD = Path("/home/zhutao/scratch/nichefate/m3/full_by_shard")
M2_BY_SLICE = Path("/home/zhutao/scratch/nichefate/m2/by_slice")
M3_FEATURE_GROUPS = Path("/home/zhutao/scratch/nichefate/m3/reports/m3_feature_groups.json")
M4E_NODE_NEIGHBORHOOD = Path(
    "/home/zhutao/scratch/nichefate/m4e/neighborhood_annotation/node_neighborhood_annotation.parquet"
)
M4C_NODE_SUMMARY = Path("/home/zhutao/scratch/nichefate/m4c/fate_probabilities/fate_probability_node_summary.parquet")
REFINED_ENDPOINT_MAPPING = Path("/home/zhutao/scratch/nichefate/m4e/endpoint_refinement/refined_endpoint_mapping.csv")

SOURCE_CAP = 50_000
ROW_QC_ATOL = 1e-5
MIN_SCALE = 1e-6
MIN_TAU = 1e-6
TAU_QUANTILE = 0.5
DISTANCE_CHUNK_SIZE = 100_000
VARIANT_NAME = "v1prior_1.0_tau_0.5_top10"


@dataclass(frozen=True)
class PilotSpec:
    pilot_id: str
    source_time: str
    target_time: str
    seed: int
    source_cap: int = SOURCE_CAP
    optional: bool = False
    exclude_m3_v2_01_sources: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--skip-optional-c", action="store_true")
    return parser.parse_args()


def max_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / float(1024**2)


def validate_output_root(path: Path) -> None:
    resolved = path.resolve()
    protected_roots = [
        Path("/home/zhutao/scratch/nichefate/m3").resolve(),
        Path("/home/zhutao/scratch/nichefate/m4a").resolve(),
        Path("/home/zhutao/scratch/nichefate/m4b").resolve(),
        Path("/home/zhutao/scratch/nichefate/m4c").resolve(),
        M3_V2_01_ROOT.resolve(),
        M3_V2_02_ROOT.resolve(),
    ]
    for root in protected_roots:
        if resolved == root or root in resolved.parents:
            raise ValueError(f"Refusing to write confirmatory outputs under protected path: {resolved}")


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    validate_output_root(output_root)
    paths = {
        "root": output_root,
        "reports": output_root / "reports",
        "figures": output_root / "reports" / "figures",
        "pilots": output_root / "pilot_edges",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def pilot_specs(skip_optional_c: bool = False) -> list[PilotSpec]:
    specs = [
        PilotSpec("A_D9_D21_repeat", "D9", "D21", seed=271_829, exclude_m3_v2_01_sources=True),
        PilotSpec("B_D3_D9", "D3", "D9", seed=314_159),
    ]
    if not skip_optional_c:
        specs.append(PilotSpec("C_D21_D35_optional", "D21", "D35", seed=161_803, optional=True))
    return specs


def load_feature_groups() -> dict[str, list[str]]:
    payload = json.loads(M3_FEATURE_GROUPS.read_text())
    return {name: list(columns) for name, columns in payload["feature_groups"].items()}


def selected_feature_columns() -> dict[str, list[str]]:
    groups = load_feature_groups()
    spatial_topology = list(dict.fromkeys(groups["spatial_summary"] + groups["topology"]))
    return {
        "state": groups["molecular_state"][:150],
        "composition": groups["cell_type_composition"][:120],
        "spatial_topology": spatial_topology,
    }


def load_source_metadata(source_time: str) -> pd.DataFrame:
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
    meta = pd.read_parquet(M4E_NODE_NEIGHBORHOOD, columns=columns)
    return meta[meta["time_label"].astype(str) == str(source_time)].copy()


def stratified_source_sample(meta: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    meta = meta.reset_index(drop=True)
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
    if len(work) <= cap:
        return work.sort_values("anchor_id").reset_index(drop=True)
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


def m3_v2_01_source_ids() -> set[str]:
    path = M3_V2_01_ROOT / "pilot_source_anchor_sample.csv"
    if not path.is_file():
        return set()
    sample = pd.read_csv(path, usecols=["anchor_id"])
    return set(sample["anchor_id"].astype(str))


def edge_root(source_time: str, target_time: str) -> Path:
    return M3_FULL_BY_SHARD / f"{source_time}_to_{target_time}"


def discover_edge_shards(source_time: str, target_time: str) -> dict[str, Path]:
    root = edge_root(source_time, target_time)
    pattern = f"candidate_edges_{source_time}_to_{target_time}__*.parquet"
    shards = {path.parent.name: path for path in sorted(root.glob(f"*/{pattern}"))}
    if not shards:
        raise FileNotFoundError(f"No frozen M3-v1 edge shards found under {root}")
    return shards


def load_selected_v1_edges(spec: PilotSpec, sampled_sources: pd.DataFrame) -> pd.DataFrame:
    shards = discover_edge_shards(spec.source_time, spec.target_time)
    frames: list[pd.DataFrame] = []
    sampled_source_ids = set(sampled_sources["anchor_id"].astype(str))
    columns = [
        "source_anchor_id",
        "target_anchor_id",
        "source_slice_id",
        "target_slice_id",
        "source_mouse_id",
        "target_mouse_id",
        "row_normalized_transition_prob",
    ]
    for slice_id, selected in sampled_sources.groupby("slice_id", sort=True):
        if slice_id not in shards:
            raise FileNotFoundError(f"Missing M3-v1 edge shard for sampled source slice {slice_id}")
        wanted = set(selected["anchor_id"].astype(str))
        frame = pd.read_parquet(shards[str(slice_id)], columns=columns)
        frame = frame[frame["source_anchor_id"].isin(wanted)].copy()
        frames.append(frame)
    edges = pd.concat(frames, ignore_index=True)
    found_source_ids = set(edges["source_anchor_id"].astype(str))
    missing_sources = sampled_source_ids - found_source_ids
    if missing_sources:
        first_missing = sorted(missing_sources)[0]
        raise RuntimeError(
            f"Missing candidate edges for {len(missing_sources)} sampled sources; "
            f"first missing source: {first_missing}"
        )
    per_source_counts = edges.groupby("source_anchor_id", sort=False).size()
    if per_source_counts.nunique() != 1:
        raise RuntimeError(
            "Selected M3-v1 candidate edge count is not constant per source: "
            f"min={int(per_source_counts.min())}, max={int(per_source_counts.max())}"
        )
    return edges


def m2_slice_path(slice_id: str) -> Path:
    return M2_BY_SLICE / slice_id / f"m2_representation_{slice_id}.parquet"


def load_m2_features(slice_ids: list[str], anchor_ids: set[str], feature_columns: list[str]) -> pd.DataFrame:
    frames = []
    read_columns = ["slice_id", "anchor_index", *feature_columns]
    for slice_id in sorted(set(slice_ids)):
        frame = pd.read_parquet(m2_slice_path(slice_id), columns=read_columns)
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
) -> tuple[np.ndarray, int]:
    if not columns:
        return np.zeros(len(source_pos), dtype=np.float32), 0
    src = aligned_feature_matrix(source_features, source_order, columns)
    tgt = aligned_feature_matrix(target_features, target_order, columns)
    stats = robust_scale_fit([src, tgt], min_scale=MIN_SCALE)
    src_scaled = robust_scale_transform(src, stats)
    tgt_scaled = robust_scale_transform(tgt, stats)
    distances = pairwise_l2_for_edges(src_scaled, tgt_scaled, source_pos, target_pos, DISTANCE_CHUNK_SIZE)
    return distances, stats.zero_scale_columns


def read_refined_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(REFINED_ENDPOINT_MAPPING)
    return mapping[
        [
            "raw_terminal_macrostate",
            "refined_endpoint_id",
            "refined_endpoint_label",
            "confidence_tier_after_refinement",
        ]
    ].rename(columns={"raw_terminal_macrostate": "dominant_fate"})


def load_annotations(anchor_ids: set[str]) -> pd.DataFrame:
    m4e_cols = [
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
    m4c_cols = [
        "anchor_id",
        "dominant_fate",
        "dominant_fate_label",
        "dominant_fate_probability",
        "normalized_plasticity_entropy",
    ]
    m4e = pd.read_parquet(M4E_NODE_NEIGHBORHOOD, columns=m4e_cols)
    m4e = m4e[m4e["anchor_id"].isin(anchor_ids)].copy()
    m4c = pd.read_parquet(M4C_NODE_SUMMARY, columns=m4c_cols)
    m4c = m4c[m4c["anchor_id"].isin(anchor_ids)].copy()
    m4c = m4c.merge(read_refined_mapping(), on="dominant_fate", how="left")
    out = m4e.merge(m4c, on="anchor_id", how="left")
    if out["anchor_id"].duplicated().any():
        raise ValueError("Duplicate anchor_id rows in annotation join.")
    return out.set_index("anchor_id")


def add_annotations(edges: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
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


def top_k_weights(weights: np.ndarray, source_codes: np.ndarray, top_k: int) -> np.ndarray:
    work = pd.DataFrame({"source_code": source_codes, "weight": np.asarray(weights, dtype=np.float64)})
    ranks = work.groupby("source_code", sort=False)["weight"].rank(method="first", ascending=False)
    return np.where(ranks.to_numpy() <= int(top_k), work["weight"].to_numpy(), 0.0)


def apply_best_variant(edges: pd.DataFrame, source_codes: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.time()
    tau = np.clip(edges["v2_tau_state"].to_numpy(dtype=np.float64) * 0.5, 1e-12, None)
    weights = np.exp(-edges["v2_d_state"].to_numpy(dtype=np.float64) / tau)
    weights *= np.clip(edges["v2_g_composition"].to_numpy(dtype=np.float64), 0.0, None)
    weights *= np.clip(edges["v2_g_spatial_topology"].to_numpy(dtype=np.float64), 0.0, None)
    weights *= np.clip(edges["v2_g_slice_mouse"].to_numpy(dtype=np.float64), 0.0, None)
    weights *= np.clip(edges["row_normalized_transition_prob"].to_numpy(dtype=np.float64), 1e-300, None)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = top_k_weights(weights, source_codes, top_k=10)
    probabilities = row_normalize_weights(weights, source_codes)
    qc = validate_probabilities(probabilities, source_codes, atol=ROW_QC_ATOL)
    qc.update(
        {
            "weight_finite": bool(np.isfinite(weights).all()),
            "weight_nonnegative": bool((weights >= 0).all()),
            "nonzero_edge_fraction": float((weights > 0).mean()),
            "runtime_seconds": float(time.time() - started),
        }
    )
    return probabilities, qc


def compute_v2_components(edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    feature_cols = selected_feature_columns()
    all_feature_cols = list(
        dict.fromkeys(feature_cols["state"] + feature_cols["composition"] + feature_cols["spatial_topology"])
    )
    source_order = pd.Index(pd.factorize(edges["source_anchor_id"], sort=False)[1].astype(str))
    target_order = pd.Index(pd.factorize(edges["target_anchor_id"], sort=False)[1].astype(str))
    source_pos = pd.Series(np.arange(len(source_order), dtype=np.int32), index=source_order)
    target_pos = pd.Series(np.arange(len(target_order), dtype=np.int32), index=target_order)
    edge_source_pos = edges["source_anchor_id"].map(source_pos).to_numpy(dtype=np.int32)
    edge_target_pos = edges["target_anchor_id"].map(target_pos).to_numpy(dtype=np.int32)
    source_features = load_m2_features(
        sorted(edges["source_slice_id"].astype(str).unique()),
        set(source_order),
        all_feature_cols,
    )
    target_features = load_m2_features(
        sorted(edges["target_slice_id"].astype(str).unique()),
        set(target_order),
        all_feature_cols,
    )
    d_state, zero_state = compute_distance_block(
        source_features,
        target_features,
        source_order,
        target_order,
        edge_source_pos,
        edge_target_pos,
        feature_cols["state"],
    )
    tau_state = source_adaptive_tau(d_state, edge_source_pos, quantile=TAU_QUANTILE, min_tau=MIN_TAU)
    d_comp, zero_comp = compute_distance_block(
        source_features,
        target_features,
        source_order,
        target_order,
        edge_source_pos,
        edge_target_pos,
        feature_cols["composition"],
    )
    tau_comp = source_adaptive_tau(d_comp, edge_source_pos, quantile=TAU_QUANTILE, min_tau=MIN_TAU)
    d_spatial, zero_spatial = compute_distance_block(
        source_features,
        target_features,
        source_order,
        target_order,
        edge_source_pos,
        edge_target_pos,
        feature_cols["spatial_topology"],
    )
    tau_spatial = source_adaptive_tau(d_spatial, edge_source_pos, quantile=TAU_QUANTILE, min_tau=MIN_TAU)
    edges["v2_d_state"] = d_state
    edges["v2_tau_state"] = tau_state
    edges["v2_g_composition"] = exponential_gate(d_comp, tau_comp, strength=1.0)
    edges["v2_g_spatial_topology"] = exponential_gate(d_spatial, tau_spatial, strength=1.0)
    edges["v2_g_slice_mouse"] = slice_mouse_gate(
        edges["target_slice_id"],
        edges["target_mouse_id"],
        strength=0.25,
        min_gate=0.2,
    )
    return edges, {
        "zero_scale_state_columns": zero_state,
        "zero_scale_composition_columns": zero_comp,
        "zero_scale_spatial_topology_columns": zero_spatial,
        "unique_source_anchors": len(source_order),
        "unique_target_anchors": len(target_order),
    }


def top_targets(edges: pd.DataFrame, probabilities: np.ndarray, source_codes: np.ndarray) -> pd.DataFrame:
    work = pd.DataFrame({"source_code": source_codes, "probability": probabilities}, index=edges.index)
    idx = work.groupby("source_code", sort=False)["probability"].idxmax()
    return edges.loc[idx].copy().reset_index(drop=True)


def agreement_rate(frame: pd.DataFrame, left: str, right: str) -> float:
    valid = frame[left].notna() & frame[right].notna()
    if not bool(valid.any()):
        return float("nan")
    return float((frame.loc[valid, left].astype(str) == frame.loc[valid, right].astype(str)).mean())


def weighted_category_distribution(edges: pd.DataFrame, probabilities: np.ndarray, category_col: str) -> pd.Series:
    work = pd.DataFrame(
        {
            "category": edges[category_col].fillna("NA").astype(str).to_numpy(),
            "probability": probabilities,
        }
    )
    dist = work.groupby("category", sort=False)["probability"].sum()
    total = float(dist.sum())
    return dist / total if total > 0 else dist


def normalized_entropy(distribution: pd.Series) -> float:
    probs = distribution[distribution > 0].to_numpy(dtype=float)
    if len(probs) <= 1:
        return 0.0
    entropy = float(-(probs * np.log(probs)).sum())
    return entropy / float(np.log(len(probs)))


def method_metrics(
    edges: pd.DataFrame,
    probabilities: np.ndarray,
    source_codes: np.ndarray,
    pilot: PilotSpec,
    method: str,
    qc: dict[str, Any],
    runtime_seconds: float,
    peak_rss_gib: float,
) -> dict[str, Any]:
    source_stats = source_entropy_and_top1(probabilities, source_codes)
    top = top_targets(edges, probabilities, source_codes)
    leiden_dist = weighted_category_distribution(edges, probabilities, "target_leiden_neigh")
    slice_dist = weighted_category_distribution(edges, probabilities, "target_slice_id")
    mouse_dist = weighted_category_distribution(edges, probabilities, "target_mouse_id")
    return {
        "pilot_id": pilot.pilot_id,
        "source_time": pilot.source_time,
        "target_time": pilot.target_time,
        "method": method,
        "source_anchor_count": int(len(np.unique(source_codes))),
        "candidate_edge_count": int(len(edges)),
        "row_sum_pass": bool(qc["row_sum_pass"]),
        "row_sum_max_abs_error": float(qc["row_sum_max_abs_error"]),
        "weight_finite": bool(qc.get("weight_finite", np.isfinite(probabilities).all())),
        "weight_nonnegative": bool(qc.get("weight_nonnegative", (probabilities >= 0).all())),
        "nonzero_edge_fraction": float(qc.get("nonzero_edge_fraction", (probabilities > 0).mean())),
        "leiden_consistency": agreement_rate(top, "source_leiden_neigh", "target_leiden_neigh"),
        "fine_cell_cluster_consistency": agreement_rate(top, "source_cell_type_l3", "target_cell_type_l3"),
        "refined_endpoint_plausibility": agreement_rate(
            top,
            "source_refined_endpoint_id",
            "target_refined_endpoint_id",
        ),
        "transition_entropy_mean": float(source_stats["transition_entropy"].mean()),
        "top1_probability_mean": float(source_stats["top1_probability"].mean()),
        "target_neighborhood_diversity": normalized_entropy(leiden_dist),
        "target_slice_concentration": float(slice_dist.max()),
        "target_mouse_concentration": float(mouse_dist.max()),
        "slice_mouse_collapse": max(float(slice_dist.max()), float(mouse_dist.max())),
        "runtime_seconds": float(runtime_seconds),
        "peak_rss_gib": float(peak_rss_gib),
    }


def v1_qc(probabilities: np.ndarray, source_codes: np.ndarray) -> dict[str, Any]:
    qc = validate_probabilities(probabilities, source_codes, atol=ROW_QC_ATOL)
    qc.update(
        {
            "weight_finite": bool(np.isfinite(probabilities).all()),
            "weight_nonnegative": bool((probabilities >= 0).all()),
            "nonzero_edge_fraction": float((probabilities > 0).mean()),
        }
    )
    return qc


def add_pilot_acceptance(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    out["passes_acceptance"] = False
    out["endpoint_ok"] = False
    out["leiden_ok"] = False
    out["entropy_ok"] = False
    out["top1_ok"] = False
    out["collapse_ok"] = False
    out["diversity_ok"] = False
    out["runtime_memory_ok"] = False
    out["delta_endpoint_vs_v1"] = np.nan
    out["delta_leiden_vs_v1"] = np.nan
    out["delta_entropy_vs_v1"] = np.nan
    out["delta_top1_vs_v1"] = np.nan
    out["delta_collapse_vs_v1"] = np.nan
    out["delta_diversity_vs_v1"] = np.nan
    for pilot_id, group in out.groupby("pilot_id", sort=False):
        v1 = group[group["method"] == "v1_reference"].iloc[0]
        idx = (out["pilot_id"] == pilot_id) & (out["method"] == VARIANT_NAME)
        out.loc[idx, "delta_endpoint_vs_v1"] = (
            out.loc[idx, "refined_endpoint_plausibility"] - float(v1["refined_endpoint_plausibility"])
        )
        out.loc[idx, "delta_leiden_vs_v1"] = out.loc[idx, "leiden_consistency"] - float(v1["leiden_consistency"])
        out.loc[idx, "delta_entropy_vs_v1"] = (
            out.loc[idx, "transition_entropy_mean"] - float(v1["transition_entropy_mean"])
        )
        out.loc[idx, "delta_top1_vs_v1"] = out.loc[idx, "top1_probability_mean"] - float(v1["top1_probability_mean"])
        out.loc[idx, "delta_collapse_vs_v1"] = out.loc[idx, "slice_mouse_collapse"] - float(v1["slice_mouse_collapse"])
        out.loc[idx, "delta_diversity_vs_v1"] = (
            out.loc[idx, "target_neighborhood_diversity"] - float(v1["target_neighborhood_diversity"])
        )
        out.loc[idx, "endpoint_ok"] = out.loc[idx, "refined_endpoint_plausibility"] >= float(
            v1["refined_endpoint_plausibility"]
        ) - 0.02
        out.loc[idx, "leiden_ok"] = out.loc[idx, "leiden_consistency"] >= float(v1["leiden_consistency"]) - 0.03
        out.loc[idx, "entropy_ok"] = out.loc[idx, "transition_entropy_mean"] < 3.0
        out.loc[idx, "top1_ok"] = out.loc[idx, "top1_probability_mean"] >= 0.15
        out.loc[idx, "collapse_ok"] = out.loc[idx, "slice_mouse_collapse"] <= float(v1["slice_mouse_collapse"]) + 0.005
        out.loc[idx, "diversity_ok"] = out.loc[idx, "target_neighborhood_diversity"] >= float(
            v1["target_neighborhood_diversity"]
        ) - 0.03
        out.loc[idx, "runtime_memory_ok"] = (out.loc[idx, "runtime_seconds"] <= 300.0) & (
            out.loc[idx, "peak_rss_gib"] <= 8.0
        )
        criteria = [
            "row_sum_pass",
            "weight_finite",
            "weight_nonnegative",
            "endpoint_ok",
            "leiden_ok",
            "entropy_ok",
            "top1_ok",
            "collapse_ok",
            "diversity_ok",
            "runtime_memory_ok",
        ]
        out.loc[idx, "passes_acceptance"] = out.loc[idx, criteria].all(axis=1)
    return out


def run_one_pilot(spec: PilotSpec, paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    meta = load_source_metadata(spec.source_time)
    excluded_sources = set()
    if spec.exclude_m3_v2_01_sources:
        excluded_sources = m3_v2_01_source_ids()
        meta = meta[~meta["anchor_id"].astype(str).isin(excluded_sources)].copy()
    sampled_sources = stratified_source_sample(meta, spec.source_cap, spec.seed)
    sample_path = paths["root"] / f"confirmatory_pilot_{spec.pilot_id}_source_anchor_sample.csv"
    sampled_sources.to_csv(sample_path, index=False)
    edges = load_selected_v1_edges(spec, sampled_sources)
    edges, component_qc = compute_v2_components(edges)
    annotations = load_annotations(set(edges["source_anchor_id"].astype(str)).union(set(edges["target_anchor_id"].astype(str))))
    edges = add_annotations(edges, annotations)
    source_codes, _ = pd.factorize(edges["source_anchor_id"], sort=False)
    source_codes = source_codes.astype(np.int32)

    v1_prob = edges["row_normalized_transition_prob"].to_numpy(dtype=np.float64)
    variant_prob, variant_qc = apply_best_variant(edges, source_codes)
    v1_metrics = method_metrics(edges, v1_prob, source_codes, spec, "v1_reference", v1_qc(v1_prob, source_codes), 0.0, max_rss_gib())
    variant_metrics = method_metrics(
        edges,
        variant_prob,
        source_codes,
        spec,
        VARIANT_NAME,
        variant_qc,
        time.time() - started,
        max_rss_gib(),
    )
    metrics = add_pilot_acceptance(pd.DataFrame([v1_metrics, variant_metrics]))
    js = jensen_shannon_by_source(v1_prob, variant_prob, source_codes)
    js = js.rename(columns={"v1_v2_js_divergence": "best_variant_js_divergence_from_v1"})
    js["pilot_id"] = spec.pilot_id
    js.to_parquet(paths["pilots"] / f"{spec.pilot_id}_source_level_divergence.parquet", index=False)

    slim_edges = edges[
        [
            "source_anchor_id",
            "target_anchor_id",
            "source_slice_id",
            "target_slice_id",
            "source_mouse_id",
            "target_mouse_id",
            "row_normalized_transition_prob",
            "v2_d_state",
            "v2_tau_state",
            "v2_g_composition",
            "v2_g_spatial_topology",
            "v2_g_slice_mouse",
        ]
    ].copy()
    slim_edges[f"{VARIANT_NAME}_probability"] = variant_prob
    slim_edges.to_parquet(paths["pilots"] / f"{spec.pilot_id}_candidate_edges_best_variant.parquet", index=False)
    info = {
        "pilot_id": spec.pilot_id,
        "source_time": spec.source_time,
        "target_time": spec.target_time,
        "seed": spec.seed,
        "optional": spec.optional,
        "excluded_m3_v2_01_source_count": len(excluded_sources),
        "source_anchor_count": int(len(np.unique(source_codes))),
        "candidate_edge_count": int(len(edges)),
        "runtime_seconds": float(time.time() - started),
        "peak_rss_gib": max_rss_gib(),
        **component_qc,
    }
    return metrics, info


def decide(combined: pd.DataFrame) -> tuple[str, str]:
    required = combined[
        combined["pilot_id"].isin(["A_D9_D21_repeat", "B_D3_D9"]) & (combined["method"] == VARIANT_NAME)
    ].copy()
    if len(required) != 2:
        return "revise_v2_and_repeat_pilot", "Required confirmatory pilots are incomplete."
    pass_count = int(required["passes_acceptance"].sum())
    if pass_count == 2:
        improves_core = (
            (required["delta_endpoint_vs_v1"] >= 0).all()
            and (required["delta_leiden_vs_v1"] >= 0).all()
            and (required["delta_collapse_vs_v1"] <= 0).all()
        )
        if improves_core:
            return "adopt_v2_for_full_production", "Both required pilots pass and improve core plausibility/artifact metrics."
        return (
            "keep_v1_and_v2_as_complementary",
            "Both required pilots pass, but the constrained v2 remains a v1-prior sharpening rather than an independent replacement.",
        )
    if pass_count == 1:
        return "revise_v2_and_repeat_pilot", "Only one required pilot passed acceptance criteria."
    return "keep_v1_as_main_baseline", "Both required confirmatory pilots failed acceptance criteria."


def write_figures(combined: pd.DataFrame, output_root: Path) -> None:
    fig_dir = output_root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("transition_entropy_mean", "entropy_by_pilot_method.png", "Mean transition entropy"),
        ("top1_probability_mean", "mean_top1_by_pilot_method.png", "Mean top1 probability"),
        ("leiden_consistency", "leiden_consistency_by_pilot_method.png", "Leiden consistency"),
        ("refined_endpoint_plausibility", "endpoint_plausibility_by_pilot_method.png", "Endpoint plausibility"),
        ("slice_mouse_collapse", "slice_mouse_collapse_by_pilot_method.png", "Slice/mouse collapse"),
    ]
    for value_col, filename, title in plot_specs:
        pivot = combined.pivot(index="pilot_id", columns="method", values=value_col)
        ax = pivot.plot(kind="bar", figsize=(8, 4.8), width=0.72)
        ax.set_title(title)
        ax.set_ylabel(value_col)
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=8)
        ax.figure.tight_layout()
        ax.figure.savefig(fig_dir / filename, dpi=170)
        plt.close(ax.figure)

    delta_cols = [
        "delta_endpoint_vs_v1",
        "delta_leiden_vs_v1",
        "delta_entropy_vs_v1",
        "delta_top1_vs_v1",
        "delta_collapse_vs_v1",
        "delta_diversity_vs_v1",
    ]
    variant = combined[combined["method"] == VARIANT_NAME].set_index("pilot_id")
    matrix = variant[delta_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(delta_cols)))
    ax.set_xticklabels(delta_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(variant.index)))
    ax.set_yticklabels(variant.index.tolist())
    ax.set_title("Best variant delta vs v1")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "delta_metrics_heatmap.png", dpi=170)
    plt.close(fig)

    divergence_files = sorted((output_root / "pilot_edges").glob("*_source_level_divergence.parquet"))
    if divergence_files:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for path in divergence_files:
            frame = pd.read_parquet(path)
            ax.hist(
                frame["best_variant_js_divergence_from_v1"].dropna(),
                bins=40,
                alpha=0.45,
                label=frame["pilot_id"].iloc[0],
            )
        ax.set_xlabel("JS divergence from v1")
        ax.set_ylabel("source count")
        ax.set_title("Source-level divergence from v1")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "source_level_divergence_from_v1.png", dpi=170)
        plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    work = frame[columns].copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: f"{float(value):.4g}" if pd.notna(value) else "NA")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in work.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_reports(
    output_root: Path,
    combined: pd.DataFrame,
    decision_table: pd.DataFrame,
    decision: str,
    reason: str,
    skipped_optional: list[dict[str, Any]],
) -> None:
    reports = output_root / "reports"
    variant = combined[combined["method"] == VARIANT_NAME].copy()
    report = f"""# M3-v2-03 Confirmatory Pilot Report

## Scope

Only `{VARIANT_NAME}` was evaluated against the frozen M3-v1 reference on selected candidate edge sets. Candidate edges were not regenerated beyond reading existing M3-v1 shards for selected source anchors. No M4A-v2 assembly, M4C-v2 propagation, pyGPCCA, M4D diagnostics, K_gpcca, M5/regulator, BranchSBM / Branched NicheFlow, or barcode preprocessing was run.

## Variant Metrics

{markdown_table(combined, ['pilot_id', 'method', 'source_anchor_count', 'candidate_edge_count', 'leiden_consistency', 'refined_endpoint_plausibility', 'transition_entropy_mean', 'top1_probability_mean', 'slice_mouse_collapse'])}
"""
    (reports / "m3_v2_03_confirmatory_pilot_report.md").write_text(report)

    summary = f"""# M3-v2-03 V1 vs Best Variant Summary

{markdown_table(variant, ['pilot_id', 'passes_acceptance', 'delta_endpoint_vs_v1', 'delta_leiden_vs_v1', 'delta_entropy_vs_v1', 'delta_top1_vs_v1', 'delta_collapse_vs_v1', 'delta_diversity_vs_v1'])}
"""
    (reports / "m3_v2_03_v1_vs_best_variant_summary.md").write_text(summary)

    skipped_text = "\n".join(
        f"- {row['pilot_id']}: {row['reason']}" for row in skipped_optional
    ) or "- None"
    recommendation = f"""# M3-v2-03 Decision Recommendation

Decision category: `{decision}`

Reason: {reason}

Optional pilot status:

{skipped_text}

Recommended next engineering step: if the decision remains complementary/adoptable, run a narrowly scoped implementation review for whether this constrained v1-prior kernel should be packaged as an explicit M3-v2 mode. Do not start full M3-v2 -> M4A-v2 -> M4C-v2 production until that review is complete.
"""
    (reports / "m3_v2_03_decision_recommendation.md").write_text(recommendation)

    decision_table.to_csv(output_root / "confirmatory_pilot_decision_table.csv", index=False)


def write_inventory(output_root: Path) -> None:
    inventory_path = output_root / "reports" / "m3_v2_03_output_inventory.csv"
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != inventory_path:
            rows.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(output_root)),
                    "file_type": path.suffix.lstrip(".") or "text",
                    "size_bytes": path.stat().st_size,
                }
            )
    pd.DataFrame(rows).to_csv(inventory_path, index=False)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    return value


def run(output_root: Path, skip_optional_c: bool = False) -> dict[str, Any]:
    started = time.time()
    paths = ensure_dirs(output_root)
    specs = pilot_specs(skip_optional_c=skip_optional_c)
    config = {
        "output_root": str(output_root),
        "variant": VARIANT_NAME,
        "source_cap": SOURCE_CAP,
        "pilots": [spec.__dict__ for spec in specs],
        "acceptance": {
            "endpoint_delta_floor": -0.02,
            "leiden_delta_floor": -0.03,
            "entropy_max": 3.0,
            "top1_min": 0.15,
            "collapse_tolerance": 0.005,
            "diversity_delta_floor": -0.03,
            "runtime_seconds_max": 300.0,
            "peak_rss_gib_max": 8.0,
        },
    }
    (output_root / "confirmatory_pilot_config_resolved.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    metric_frames: list[pd.DataFrame] = []
    pilot_infos: list[dict[str, Any]] = []
    skipped_optional: list[dict[str, Any]] = []
    for spec in specs:
        try:
            metrics, info = run_one_pilot(spec, paths)
        except Exception as exc:
            if spec.optional:
                skipped_optional.append({"pilot_id": spec.pilot_id, "reason": str(exc)})
                continue
            raise
        metric_frames.append(metrics)
        pilot_infos.append(info)
        metric_file_stem = {
            "A_D9_D21_repeat": "A_D9_D21",
            "B_D3_D9": "B_D3_D9",
            "C_D21_D35_optional": "C_D21_D35",
        }[spec.pilot_id]
        metrics.to_csv(output_root / f"confirmatory_pilot_{metric_file_stem}_metrics.csv", index=False)

    combined = pd.concat(metric_frames, ignore_index=True)
    combined.to_csv(output_root / "confirmatory_pilot_combined_metric_summary.csv", index=False)
    decision, reason = decide(combined)
    decision_table = combined[combined["method"] == VARIANT_NAME].copy()
    decision_table["decision_category"] = decision
    decision_table["decision_reason"] = reason
    decision_table.to_csv(output_root / "confirmatory_pilot_decision_table.csv", index=False)
    payload = {
        "decision": decision,
        "decision_reason": reason,
        "variant": VARIANT_NAME,
        "pilot_infos": pilot_infos,
        "skipped_optional": skipped_optional,
        "source_anchor_count_total": int(sum(info["source_anchor_count"] for info in pilot_infos)),
        "candidate_edge_count_total": int(sum(info["candidate_edge_count"] for info in pilot_infos)),
        "runtime_seconds": float(time.time() - started),
        "peak_rss_gib": max_rss_gib(),
    }
    (output_root / "confirmatory_pilot_combined_metric_summary.json").write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True)
    )
    write_figures(combined, output_root)
    write_reports(output_root, combined, decision_table, decision, reason, skipped_optional)
    write_inventory(output_root)
    return payload


def main() -> None:
    args = parse_args()
    payload = run(Path(args.output_root), skip_optional_c=bool(args.skip_optional_c))
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
