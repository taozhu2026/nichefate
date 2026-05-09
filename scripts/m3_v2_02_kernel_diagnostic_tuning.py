#!/usr/bin/env python
"""Diagnose and tune the bounded M3-v2 D9->D21 pilot kernel."""

from __future__ import annotations

import argparse
import json
import math
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
    jensen_shannon_by_source,
    row_normalize_weights,
    source_entropy_and_top1,
    validate_probabilities,
)


INPUT_ROOT = Path("/home/zhutao/scratch/nichefate/m3_v2_pilot")
OUTPUT_ROOT = Path("/home/zhutao/scratch/nichefate/m3_v2_pilot_tuning")
ROW_QC_ATOL = 1e-5

EDGE_COLUMNS = [
    "source_anchor_id",
    "target_anchor_id",
    "target_slice_id",
    "target_mouse_id",
    "row_normalized_transition_prob",
    "source_leiden_neigh",
    "target_leiden_neigh",
    "source_cell_type_l3",
    "target_cell_type_l3",
    "source_refined_endpoint_id",
    "target_refined_endpoint_id",
    "v2_d_state",
    "v2_tau_state",
    "v2_g_composition",
    "v2_g_spatial_topology",
    "v2_g_slice_mouse",
    "v2_unnormalized_weight",
    "v2_row_normalized_transition_prob",
]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    tau_scale: float = 1.0
    v1_lambda: float = 0.0
    comp_power: float = 1.0
    topo_power: float = 1.0
    top_k: int | None = None
    reference: str = "variant"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(INPUT_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
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
        INPUT_ROOT.resolve(),
    ]
    for root in forbidden_roots:
        if resolved == root or root in resolved.parents:
            raise ValueError(f"Refusing to write M3-v2 tuning outputs under protected path: {resolved}")


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    validate_output_root(output_root)
    paths = {
        "root": output_root,
        "reports": output_root / "reports",
        "figures": output_root / "reports" / "figures",
        "component_figures": output_root / "reports" / "figures" / "component_distribution_plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec("v2_01_original_recomputed", reference="v2_01"),
        VariantSpec("tau_0.5", tau_scale=0.5),
        VariantSpec("tau_0.25", tau_scale=0.25),
        VariantSpec("v1prior_0.5", v1_lambda=0.5),
        VariantSpec("v1prior_1.0", v1_lambda=1.0),
        VariantSpec("v1prior_0.5_tau_0.5", tau_scale=0.5, v1_lambda=0.5),
        VariantSpec("v1prior_1.0_tau_0.5", tau_scale=0.5, v1_lambda=1.0),
        VariantSpec("v1prior_0.5_gatepow2", v1_lambda=0.5, comp_power=2.0, topo_power=2.0),
        VariantSpec("v1prior_1.0_gatepow2", v1_lambda=1.0, comp_power=2.0, topo_power=2.0),
        VariantSpec("v1prior_0.5_tau_0.5_top10", tau_scale=0.5, v1_lambda=0.5, top_k=10),
        VariantSpec("v1prior_1.0_tau_0.5_top10", tau_scale=0.5, v1_lambda=1.0, top_k=10),
        VariantSpec("v1_reference", reference="v1"),
    ]


def read_edges(input_root: Path) -> pd.DataFrame:
    path = input_root / "pilot_candidate_edges_v2_reweighted.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing M3-v2-01 edge table: {path}")
    edges = pd.read_parquet(path, columns=EDGE_COLUMNS)
    missing = sorted(set(EDGE_COLUMNS) - set(edges.columns))
    if missing:
        raise ValueError(f"Missing required M3-v2-01 edge columns: {missing}")
    return edges


def finite_stats(name: str, values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    row: dict[str, Any] = {
        "component": name,
        "count": int(len(array)),
        "finite_count": int(len(finite)),
        "finite_fraction": float(len(finite) / len(array)) if len(array) else float("nan"),
    }
    if len(finite) == 0:
        for key in ["min", "q01", "q05", "q10", "q25", "median", "q75", "q90", "q95", "q99", "max", "mean", "std"]:
            row[key] = float("nan")
        row["fraction_lt_0_01"] = float("nan")
        row["fraction_lt_0_05"] = float("nan")
        row["fraction_gt_0_95"] = float("nan")
        return row
    quantiles = np.nanpercentile(finite, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    row.update(
        {
            "min": float(np.nanmin(finite)),
            "q01": float(quantiles[0]),
            "q05": float(quantiles[1]),
            "q10": float(quantiles[2]),
            "q25": float(quantiles[3]),
            "median": float(quantiles[4]),
            "q75": float(quantiles[5]),
            "q90": float(quantiles[6]),
            "q95": float(quantiles[7]),
            "q99": float(quantiles[8]),
            "max": float(np.nanmax(finite)),
            "mean": float(np.nanmean(finite)),
            "std": float(np.nanstd(finite)),
            "fraction_lt_0_01": float((finite < 0.01).mean()),
            "fraction_lt_0_05": float((finite < 0.05).mean()),
            "fraction_gt_0_95": float((finite > 0.95).mean()),
        }
    )
    return row


def top_k_weights(weights: np.ndarray, source_codes: np.ndarray, top_k: int | None) -> np.ndarray:
    if top_k is None:
        return np.asarray(weights, dtype=np.float64)
    if top_k <= 0:
        raise ValueError("top_k must be positive when provided.")
    work = pd.DataFrame({"source_code": source_codes, "weight": np.asarray(weights, dtype=np.float64)})
    ranks = work.groupby("source_code", sort=False)["weight"].rank(method="first", ascending=False)
    return np.where(ranks.to_numpy() <= int(top_k), work["weight"].to_numpy(), 0.0)


def compute_variant_weights(edges: pd.DataFrame, spec: VariantSpec) -> np.ndarray:
    if spec.reference == "v1":
        return edges["row_normalized_transition_prob"].to_numpy(dtype=np.float64)
    tau = np.clip(
        edges["v2_tau_state"].to_numpy(dtype=np.float64) * float(spec.tau_scale),
        1e-12,
        None,
    )
    d_state = edges["v2_d_state"].to_numpy(dtype=np.float64)
    weights = np.exp(-d_state / tau)
    weights *= np.power(
        np.clip(edges["v2_g_composition"].to_numpy(dtype=np.float64), 0.0, None),
        float(spec.comp_power),
    )
    weights *= np.power(
        np.clip(edges["v2_g_spatial_topology"].to_numpy(dtype=np.float64), 0.0, None),
        float(spec.topo_power),
    )
    weights *= np.clip(edges["v2_g_slice_mouse"].to_numpy(dtype=np.float64), 0.0, None)
    if spec.v1_lambda > 0:
        weights *= np.power(
            np.clip(edges["row_normalized_transition_prob"].to_numpy(dtype=np.float64), 1e-300, None),
            float(spec.v1_lambda),
        )
    return np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)


def compute_variant_probabilities(
    edges: pd.DataFrame,
    source_codes: np.ndarray,
    spec: VariantSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.time()
    weights = compute_variant_weights(edges, spec)
    weights = top_k_weights(weights, source_codes, spec.top_k)
    probabilities = row_normalize_weights(weights, source_codes)
    qc = validate_probabilities(probabilities, source_codes, atol=ROW_QC_ATOL)
    qc.update(
        {
            "variant": spec.name,
            "runtime_seconds": float(time.time() - started),
            "weight_finite": bool(np.isfinite(weights).all()),
            "weight_nonnegative": bool((weights >= 0).all()),
            "nonzero_edge_fraction": float((weights > 0).mean()),
        }
    )
    return probabilities, qc


def spearman_by_source(
    left: np.ndarray,
    right: np.ndarray,
    source_codes: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "source_code": source_codes,
            "left": np.asarray(left, dtype=np.float64),
            "right": np.asarray(right, dtype=np.float64),
        }
    )
    frame["left_rank"] = frame.groupby("source_code", sort=False)["left"].rank(method="average")
    frame["right_rank"] = frame.groupby("source_code", sort=False)["right"].rank(method="average")
    frame["rank_product"] = frame["left_rank"] * frame["right_rank"]
    grouped = frame.groupby("source_code", sort=False)
    n = grouped.size().astype(float)
    sum_l = grouped["left_rank"].sum()
    sum_r = grouped["right_rank"].sum()
    sum_ll = grouped["left_rank"].apply(lambda values: float(np.square(values).sum()))
    sum_rr = grouped["right_rank"].apply(lambda values: float(np.square(values).sum()))
    sum_lr = grouped["rank_product"].sum()
    cov = sum_lr - (sum_l * sum_r / n)
    var_l = sum_ll - (sum_l * sum_l / n)
    var_r = sum_rr - (sum_r * sum_r / n)
    denom = np.sqrt(var_l * var_r)
    corr = np.where(denom > 0, cov / denom, np.nan)
    return pd.DataFrame({"source_code": n.index.to_numpy(), "spearman_rank_correlation": corr})


def source_entropy_values(probabilities: np.ndarray, source_codes: np.ndarray) -> pd.DataFrame:
    return source_entropy_and_top1(probabilities, source_codes)


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


def variant_metrics(
    edges: pd.DataFrame,
    probabilities: np.ndarray,
    source_codes: np.ndarray,
    spec: VariantSpec,
    qc: dict[str, Any],
) -> dict[str, Any]:
    source_stats = source_entropy_values(probabilities, source_codes)
    top = top_targets(edges, probabilities, source_codes)
    leiden_dist = weighted_category_distribution(edges, probabilities, "target_leiden_neigh")
    slice_dist = weighted_category_distribution(edges, probabilities, "target_slice_id")
    mouse_dist = weighted_category_distribution(edges, probabilities, "target_mouse_id")
    js = jensen_shannon_by_source(
        edges["row_normalized_transition_prob"].to_numpy(dtype=np.float64),
        probabilities,
        source_codes,
    )
    return {
        "variant": spec.name,
        "reference": spec.reference,
        "tau_scale": spec.tau_scale,
        "v1_lambda": spec.v1_lambda,
        "comp_power": spec.comp_power,
        "topo_power": spec.topo_power,
        "top_k": spec.top_k if spec.top_k is not None else "",
        "row_sum_pass": bool(qc["row_sum_pass"]),
        "row_sum_max_abs_error": float(qc["row_sum_max_abs_error"]),
        "weight_finite": bool(qc["weight_finite"]),
        "weight_nonnegative": bool(qc["weight_nonnegative"]),
        "nonzero_edge_fraction": float(qc["nonzero_edge_fraction"]),
        "runtime_seconds": float(qc["runtime_seconds"]),
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
        "mean_js_divergence_from_v1": float(js["v1_v2_js_divergence"].mean()),
    }


def add_acceptance_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    v1 = out[out["variant"] == "v1_reference"].iloc[0]
    out["endpoint_ok"] = out["refined_endpoint_plausibility"] >= float(v1["refined_endpoint_plausibility"]) - 0.02
    out["leiden_ok"] = out["leiden_consistency"] >= float(v1["leiden_consistency"]) - 0.03
    out["entropy_ok"] = out["transition_entropy_mean"] < 3.0
    out["top1_ok"] = out["top1_probability_mean"] >= 0.15
    out["collapse_ok"] = out["slice_mouse_collapse"] <= float(v1["slice_mouse_collapse"]) + 0.005
    out["diversity_ok"] = out["target_neighborhood_diversity"] >= float(v1["target_neighborhood_diversity"]) - 0.03
    out["hard_qc_pass"] = (
        out["row_sum_pass"]
        & out["weight_finite"]
        & out["weight_nonnegative"]
        & out[
            [
                "leiden_consistency",
                "fine_cell_cluster_consistency",
                "refined_endpoint_plausibility",
                "transition_entropy_mean",
                "top1_probability_mean",
                "target_neighborhood_diversity",
                "slice_mouse_collapse",
            ]
        ]
        .notna()
        .all(axis=1)
    )
    criteria = ["endpoint_ok", "leiden_ok", "entropy_ok", "top1_ok", "collapse_ok", "diversity_ok"]
    out["acceptance_criteria_passed"] = out[criteria].sum(axis=1).astype(int)
    out["passes_acceptance"] = out["hard_qc_pass"] & (out["acceptance_criteria_passed"] == len(criteria))
    out["delta_endpoint_vs_v1"] = out["refined_endpoint_plausibility"] - float(v1["refined_endpoint_plausibility"])
    out["delta_leiden_vs_v1"] = out["leiden_consistency"] - float(v1["leiden_consistency"])
    out["delta_collapse_vs_v1"] = out["slice_mouse_collapse"] - float(v1["slice_mouse_collapse"])
    out["delta_diversity_vs_v1"] = out["target_neighborhood_diversity"] - float(v1["target_neighborhood_diversity"])
    return out


def rank_variants(metrics: pd.DataFrame) -> pd.DataFrame:
    ranked = add_acceptance_columns(metrics)
    ranked["is_v1_reference"] = ranked["variant"] == "v1_reference"
    ranked["is_v2_01"] = ranked["variant"] == "v2_01_original_recomputed"
    ranked = ranked.sort_values(
        [
            "is_v1_reference",
            "passes_acceptance",
            "acceptance_criteria_passed",
            "delta_endpoint_vs_v1",
            "delta_leiden_vs_v1",
            "top1_probability_mean",
        ],
        ascending=[True, False, False, False, False, False],
    ).reset_index(drop=True)
    ranked["decision_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def choose_decision(ranked: pd.DataFrame) -> tuple[str, pd.Series]:
    candidates = ranked[ranked["variant"] != "v1_reference"].copy()
    best = candidates.iloc[0]
    if bool(best["passes_acceptance"]):
        if (
            float(best["delta_endpoint_vs_v1"]) >= 0
            and float(best["delta_leiden_vs_v1"]) >= 0
            and float(best["delta_collapse_vs_v1"]) < -0.005
        ):
            return "adopt_v2_for_full_production", best
        if float(best["delta_endpoint_vs_v1"]) >= -0.02 and float(best["delta_leiden_vs_v1"]) >= -0.03:
            return "revise_v2_and_repeat_pilot", best
        return "keep_v1_and_v2_as_complementary", best
    return "keep_v1_as_main_baseline", best


def write_component_figures(component_values: dict[str, np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in component_values.items():
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(finite, bins=60, color="#4267ac", alpha=0.85)
        ax.set_title(name)
        ax.set_ylabel("edge count")
        ax.set_xlabel("value")
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=160)
        plt.close(fig)


def write_summary_figures(metrics: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("transition_entropy_mean", "entropy_by_variant.png", "Mean transition entropy"),
        ("top1_probability_mean", "top1_probability_by_variant.png", "Mean top1 probability"),
        ("leiden_consistency", "leiden_consistency_by_variant.png", "Leiden consistency"),
        ("refined_endpoint_plausibility", "refined_endpoint_plausibility_by_variant.png", "Refined endpoint plausibility"),
        ("slice_mouse_collapse", "slice_mouse_collapse_by_variant.png", "Slice/mouse collapse"),
    ]
    labels = metrics["variant"].tolist()
    x = np.arange(len(labels))
    for col, filename, title in plot_specs:
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.65), 4.8))
        ax.bar(x, metrics[col].to_numpy(dtype=float), color="#4f7f71")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(metrics["transition_entropy_mean"], metrics["refined_endpoint_plausibility"], s=42)
    for row in metrics.itertuples(index=False):
        ax.annotate(row.variant, (row.transition_entropy_mean, row.refined_endpoint_plausibility), fontsize=7)
    ax.set_xlabel("mean transition entropy")
    ax.set_ylabel("refined endpoint plausibility")
    ax.set_title("Pareto: endpoint plausibility vs entropy")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_endpoint_plausibility_vs_entropy.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(metrics["slice_mouse_collapse"], metrics["leiden_consistency"], s=42)
    for row in metrics.itertuples(index=False):
        ax.annotate(row.variant, (row.slice_mouse_collapse, row.leiden_consistency), fontsize=7)
    ax.set_xlabel("slice/mouse collapse")
    ax.set_ylabel("Leiden consistency")
    ax.set_title("Pareto: Leiden consistency vs slice/mouse collapse")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_leiden_consistency_vs_slice_mouse_collapse.png", dpi=170)
    plt.close(fig)


def write_rank_correlation_figure(correlations: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(correlations["spearman_rank_correlation"].dropna(), bins=50, color="#8c5a6b", alpha=0.85)
    ax.set_xlabel("Spearman rank correlation with v1")
    ax.set_ylabel("source count")
    ax.set_title("v1 vs best-variant rank correlation")
    fig.tight_layout()
    fig.savefig(out_dir / "v1_vs_best_variant_rank_correlation_distribution.png", dpi=170)
    plt.close(fig)


def diagnostic_tables(edges: pd.DataFrame, source_codes: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    d_state = edges["v2_d_state"].to_numpy(dtype=np.float64)
    tau = edges["v2_tau_state"].to_numpy(dtype=np.float64)
    ratio = d_state / np.clip(tau, 1e-12, None)
    state_gate = np.exp(-ratio)
    state_only_prob = row_normalize_weights(state_gate, source_codes)
    state_stats = source_entropy_values(state_only_prob, source_codes)
    v1_prob = edges["row_normalized_transition_prob"].to_numpy(dtype=np.float64)
    v2_prob = edges["v2_row_normalized_transition_prob"].to_numpy(dtype=np.float64)
    v1_stats = source_entropy_values(v1_prob, source_codes)
    v2_stats = source_entropy_values(v2_prob, source_codes)
    rank_corr = spearman_by_source(v1_prob, v2_prob, source_codes)
    component_values = {
        "d_state": d_state,
        "tau_i": tau,
        "d_state_over_tau_i": ratio,
        "exp_neg_d_state_over_tau_i": state_gate,
        "G_composition": edges["v2_g_composition"].to_numpy(dtype=np.float64),
        "G_spatial_topology": edges["v2_g_spatial_topology"].to_numpy(dtype=np.float64),
        "G_slice_mouse": edges["v2_g_slice_mouse"].to_numpy(dtype=np.float64),
        "final_unnormalized_weight": edges["v2_unnormalized_weight"].to_numpy(dtype=np.float64),
        "v1_row_entropy": v1_stats["transition_entropy"].to_numpy(dtype=np.float64),
        "v2_row_entropy": v2_stats["transition_entropy"].to_numpy(dtype=np.float64),
        "state_only_row_entropy": state_stats["transition_entropy"].to_numpy(dtype=np.float64),
        "v1_top1_probability": v1_stats["top1_probability"].to_numpy(dtype=np.float64),
        "v2_top1_probability": v2_stats["top1_probability"].to_numpy(dtype=np.float64),
        "state_only_top1_probability": state_stats["top1_probability"].to_numpy(dtype=np.float64),
        "v1_vs_v2_rank_correlation": rank_corr["spearman_rank_correlation"].to_numpy(dtype=np.float64),
    }
    table = pd.DataFrame([finite_stats(name, values) for name, values in component_values.items()])
    row_size = pd.Series(source_codes).value_counts().median()
    uniform_entropy = float(math.log(float(row_size))) if row_size and row_size > 1 else float("nan")
    answers = {
        "median_candidates_per_source": float(row_size),
        "uniform_entropy_for_median_row": uniform_entropy,
        "v2_entropy_ratio_to_uniform": float(v2_stats["transition_entropy"].mean() / uniform_entropy),
        "state_only_entropy_ratio_to_uniform": float(state_stats["transition_entropy"].mean() / uniform_entropy),
        "v2_mean_top1_probability": float(v2_stats["top1_probability"].mean()),
        "state_only_mean_top1_probability": float(state_stats["top1_probability"].mean()),
        "median_v1_v2_rank_correlation": float(rank_corr["spearman_rank_correlation"].median()),
        "median_composition_gate": float(np.nanmedian(edges["v2_g_composition"])),
        "median_spatial_topology_gate": float(np.nanmedian(edges["v2_g_spatial_topology"])),
        "median_slice_mouse_gate": float(np.nanmedian(edges["v2_g_slice_mouse"])),
    }
    answers["is_effectively_uniform"] = bool(
        answers["v2_entropy_ratio_to_uniform"] > 0.95 and answers["v2_mean_top1_probability"] < 0.15
    )
    answers["tau_effectively_too_soft"] = bool(
        answers["state_only_entropy_ratio_to_uniform"] > 0.90
        or answers["state_only_mean_top1_probability"] < 0.15
    )
    answers["gates_mostly_near_one"] = bool(
        answers["median_composition_gate"] > 0.8
        and answers["median_spatial_topology_gate"] > 0.8
        and answers["median_slice_mouse_gate"] > 0.8
    )
    answers["d_state_weakly_discriminative"] = bool(answers["state_only_mean_top1_probability"] < 0.15)
    answers["erased_v1_ranking"] = bool(answers["median_v1_v2_rank_correlation"] < 0.30)
    return table, answers, component_values


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
    paths: dict[str, Path],
    component_table: pd.DataFrame,
    answers: dict[str, Any],
    metrics: pd.DataFrame,
    ranked: pd.DataFrame,
    decision: str,
    best: pd.Series,
) -> None:
    diagnostic = f"""# M3-v2-01 Diffuseness Diagnostic

## Answers

- Is tau_i too large? `{answers['tau_effectively_too_soft']}`. State-only entropy ratio to uniform was {answers['state_only_entropy_ratio_to_uniform']:.4f}, with mean state-only top1 probability {answers['state_only_mean_top1_probability']:.4f}.
- Are gates mostly near 1? `{answers['gates_mostly_near_one']}`. Median gates: composition {answers['median_composition_gate']:.4f}, spatial/topology {answers['median_spatial_topology_gate']:.4f}, slice/mouse {answers['median_slice_mouse_gate']:.4f}.
- Is d_state weakly discriminative among candidate targets? `{answers['d_state_weakly_discriminative']}`.
- Did v2 erase useful v1 edge ranking? `{answers['erased_v1_ranking']}`. Median v1-v2 Spearman rank correlation was {answers['median_v1_v2_rank_correlation']:.4f}.
- Is the v2 kernel effectively uniform over the candidate targets? `{answers['is_effectively_uniform']}`. V2 entropy/uniform entropy ratio was {answers['v2_entropy_ratio_to_uniform']:.4f}; mean v2 top1 probability was {answers['v2_mean_top1_probability']:.4f}.

## Component Distribution Summary

{markdown_table(component_table, ['component', 'median', 'q10', 'q90', 'mean', 'std', 'fraction_gt_0_95'])}
"""
    (paths["reports"] / "m3_v2_01_diffuseness_diagnostic.md").write_text(diagnostic)

    report = f"""# M3-v2-02 Tuning Report

## Scope

- Input: read-only M3-v2-01 pilot candidate edge table.
- Candidate edges were not regenerated.
- Source-anchor set was not changed.
- No M4A-v2 assembly, M4C-v2 propagation, pyGPCCA, M4D diagnostics, K_gpcca, custom GPCCA, M5/regulator, BranchSBM / Branched NicheFlow, or barcode preprocessing was run.

## Variant Metrics

{markdown_table(metrics, ['variant', 'leiden_consistency', 'fine_cell_cluster_consistency', 'refined_endpoint_plausibility', 'transition_entropy_mean', 'top1_probability_mean', 'slice_mouse_collapse', 'target_neighborhood_diversity'])}
"""
    (paths["reports"] / "m3_v2_02_tuning_report.md").write_text(report)

    comparison = f"""# M3-v2-02 V1 vs Variants Comparison

Best non-v1 variant by decision ranking: `{best['variant']}`

{markdown_table(ranked, ['decision_rank', 'variant', 'passes_acceptance', 'acceptance_criteria_passed', 'delta_endpoint_vs_v1', 'delta_leiden_vs_v1', 'delta_collapse_vs_v1'])}
"""
    (paths["reports"] / "m3_v2_02_v1_vs_variants_comparison.md").write_text(comparison)

    recommendation = f"""# M3-v2-02 Decision Recommendation

Decision category: `{decision}`

Best non-v1 variant: `{best['variant']}`

- passes_acceptance: {best['passes_acceptance']}
- refined endpoint plausibility: {best['refined_endpoint_plausibility']:.4f}
- Leiden consistency: {best['leiden_consistency']:.4f}
- transition entropy: {best['transition_entropy_mean']:.4f}
- mean top1 probability: {best['top1_probability_mean']:.4f}
- slice/mouse collapse: {best['slice_mouse_collapse']:.4f}

If no tuned variant meets acceptance criteria, keep M3-v1/M4C-v1 as the pseudo-only Plan A baseline and move next to DARLIN preprocessing / barcode input contract. K_gpcca should remain a separate later pilot.
"""
    (paths["reports"] / "m3_v2_02_decision_recommendation.md").write_text(recommendation)


def write_inventory(output_root: Path) -> None:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(output_root)),
                    "file_type": path.suffix.lstrip(".") or "text",
                    "size_bytes": path.stat().st_size,
                }
            )
    inventory_path = output_root / "reports" / "m3_v2_02_output_inventory.csv"
    rows.append(
        {
            "path": str(inventory_path),
            "relative_path": str(inventory_path.relative_to(output_root)),
            "file_type": "csv",
            "size_bytes": inventory_path.stat().st_size if inventory_path.exists() else 0,
        }
    )
    pd.DataFrame(rows).drop_duplicates("relative_path").to_csv(inventory_path, index=False)


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


def run(input_root: Path, output_root: Path) -> dict[str, Any]:
    started = time.time()
    paths = ensure_output_dirs(output_root)
    edges = read_edges(input_root)
    source_codes, _ = pd.factorize(edges["source_anchor_id"], sort=False)
    source_codes = source_codes.astype(np.int32)

    component_table, answers, component_values = diagnostic_tables(edges, source_codes)
    component_table.to_csv(paths["reports"] / "m3_v2_01_kernel_component_distributions.csv", index=False)
    write_component_figures(component_values, paths["component_figures"])

    metric_rows = []
    variant_probabilities: dict[str, np.ndarray] = {}
    qc_rows = []
    for spec in variant_specs():
        probabilities, qc = compute_variant_probabilities(edges, source_codes, spec)
        variant_probabilities[spec.name] = probabilities
        qc_rows.append(qc)
        metric_rows.append(variant_metrics(edges, probabilities, source_codes, spec, qc))
    metrics = pd.DataFrame(metric_rows)
    ranked = rank_variants(metrics)
    decision, best = choose_decision(ranked)

    metrics.to_csv(output_root / "variant_metric_summary.csv", index=False)
    ranked.to_csv(output_root / "variant_ranked_decision_table.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(paths["reports"] / "m3_v2_02_variant_qc.csv", index=False)

    payload = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "edge_count": int(len(edges)),
        "source_anchor_count": int(len(np.unique(source_codes))),
        "variant_count": int(len(metrics)),
        "decision": decision,
        "best_variant": best.to_dict(),
        "diffuseness_answers": answers,
        "runtime_seconds": float(time.time() - started),
        "max_rss_gib": max_rss_gib(),
    }
    (output_root / "variant_metric_summary.json").write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))

    write_summary_figures(metrics, paths["figures"])
    best_probs = variant_probabilities[str(best["variant"])]
    best_corr = spearman_by_source(
        edges["row_normalized_transition_prob"].to_numpy(dtype=np.float64),
        best_probs,
        source_codes,
    )
    best_corr.to_csv(paths["reports"] / "best_variant_rank_correlation.csv", index=False)
    write_rank_correlation_figure(best_corr, paths["figures"])
    write_reports(paths, component_table, answers, metrics, ranked, decision, best)
    write_inventory(output_root)
    return payload


def main() -> None:
    args = parse_args()
    payload = run(Path(args.input_root), Path(args.output_root))
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
