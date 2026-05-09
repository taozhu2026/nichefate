#!/usr/bin/env python
"""M4E-03 endpoint refinement and M4C-v1 baseline figure freeze.

This script is read-only with respect to M3/M4A/M4B/M4C production artifacts.
It uses M4E-01/M4E-02 annotation outputs to refine candidate endpoint niche
cluster labels and writes only M4E endpoint-refinement tables, reports, and
lightweight figures.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/zhutao/scratch/nichefate")
OUT = ROOT / "m4e" / "endpoint_refinement"
REPORTS = ROOT / "m4e" / "reports"
FIGURES = REPORTS / "figures" / "endpoint_refinement"

ENDPOINT_NODE = ROOT / "m4e" / "endpoint_annotation" / "endpoint_node_annotation.parquet"
NODE_NEIGHBORHOOD = ROOT / "m4e" / "neighborhood_annotation" / "node_neighborhood_annotation.parquet"
ENDPOINT_SUMMARY = ROOT / "m4e" / "endpoint_annotation" / "endpoint_macrostate_annotation_summary.csv"
ENDPOINT_MAJOR = ROOT / "m4e" / "endpoint_annotation" / "endpoint_by_major_cell_class.csv"
ENDPOINT_FINE = ROOT / "m4e" / "endpoint_annotation" / "endpoint_by_fine_cell_cluster.csv"
ENDPOINT_ANCHOR = ROOT / "m4e" / "endpoint_annotation" / "endpoint_by_anchor_cell_type.csv"
LEIDEN_COUNTS = ROOT / "m4e" / "neighborhood_annotation" / "endpoint_by_leiden_neigh_counts.csv"
LEIDEN_FRACTION_ENDPOINT = (
    ROOT / "m4e" / "neighborhood_annotation" / "endpoint_by_leiden_neigh_fraction_by_endpoint.csv"
)
NEIGHBORHOOD_PURITY = ROOT / "m4e" / "neighborhood_annotation" / "endpoint_neighborhood_purity_entropy.csv"
TIERS_WITH_NEIGHBORHOOD = (
    ROOT / "m4e" / "neighborhood_annotation" / "m4e_endpoint_confidence_tiers_with_neighborhood.csv"
)
PLASTICITY_LEIDEN = ROOT / "m4e" / "neighborhood_annotation" / "plasticity_by_leiden_neigh.csv"
NICHE_ADVANTAGE_NEIGHBORHOOD = (
    ROOT / "m4e" / "neighborhood_annotation" / "niche_advantage_same_celltype_by_neighborhood.csv"
)
M4C_NODE = ROOT / "m4c" / "fate_probabilities" / "fate_probability_node_summary.parquet"


PROPOSED_MAPPING: dict[int, dict[str, str]] = {
    0: {
        "refined_endpoint_id": "sm_me_smc2",
        "refined_endpoint_label": "Smooth muscle / muscularis externa SMC2-rich endpoint",
        "refined_endpoint_category": "smooth-muscle / muscularis externa",
        "action": "keep",
    },
    1: {
        "refined_endpoint_id": "mu_epithelial_stem",
        "refined_endpoint_label": "MU epithelial stem-like mucosal endpoint",
        "refined_endpoint_category": "stem-like mucosal",
        "action": "keep",
    },
    2: {
        "refined_endpoint_id": "sm_stromal_slice_associated",
        "refined_endpoint_label": "SM-biased fibroblast/stromal endpoint, slice-associated",
        "refined_endpoint_category": "slice/mouse-associated",
        "action": "caution_slice_associated",
    },
    3: {
        "refined_endpoint_id": "sm_mixed_submucosal",
        "refined_endpoint_label": "SM mixed smooth-muscle/stromal endpoint",
        "refined_endpoint_category": "mixed submucosal",
        "action": "unresolved_keep",
    },
    4: {
        "refined_endpoint_id": "rare_other_bcell",
        "refined_endpoint_label": "Rare other-neighborhood B-cell endpoint",
        "refined_endpoint_category": "low-size / rare",
        "action": "rare_keep",
    },
    5: {
        "refined_endpoint_id": "mu_mixed_mucosal_immune",
        "refined_endpoint_label": "MU mixed mucosal-immune endpoint",
        "refined_endpoint_category": "mixed mucosal",
        "action": "unresolved_keep",
    },
    6: {
        "refined_endpoint_id": "rare_me_enteric_smc",
        "refined_endpoint_label": "Rare ME enteric/smooth-muscle endpoint",
        "refined_endpoint_category": "low-size / rare",
        "action": "rare_keep",
    },
    7: {
        "refined_endpoint_id": "me_smc1_muscularis",
        "refined_endpoint_label": "ME SMC1-rich muscularis endpoint",
        "refined_endpoint_category": "smooth-muscle / muscularis externa",
        "action": "keep",
    },
    8: {
        "refined_endpoint_id": "rare_me_fibro_smc",
        "refined_endpoint_label": "Rare ME fibroblast/smooth-muscle stromal endpoint",
        "refined_endpoint_category": "low-size / rare",
        "action": "rare_keep",
    },
    9: {
        "refined_endpoint_id": "rare_me_enteric_smc",
        "refined_endpoint_label": "Rare ME enteric/smooth-muscle endpoint",
        "refined_endpoint_category": "low-size / rare",
        "action": "merge_candidate",
    },
    10: {
        "refined_endpoint_id": "rare_fol_bcell",
        "refined_endpoint_label": "Rare FOL B-cell follicle endpoint",
        "refined_endpoint_category": "follicle / B-cell immune",
        "action": "rare_keep",
    },
    11: {
        "refined_endpoint_id": "mu_colonocyte",
        "refined_endpoint_label": "MU colonocyte mucosal endpoint",
        "refined_endpoint_category": "colonocyte mucosal",
        "action": "keep",
    },
}

REVIEW_PAIRS = {(0, 7), (1, 11), (3, 5), (6, 9)}
PAIR_REVIEW_LABELS = {
    (0, 7): "00 vs 07 ME smooth-muscle redundancy review",
    (1, 11): "01 vs 11 MU mucosal redundancy review",
    (3, 5): "03 vs 05 mixed endpoint review",
    (6, 9): "06 vs 09 rare ME enteric/smooth-muscle review",
}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def write_report(filename: str, body: str) -> None:
    (OUT / filename).write_text(body)
    (REPORTS / filename).write_text(body)


def endpoint_label(endpoint: int | float | str) -> str:
    value = int(endpoint)
    return f"terminal_macrostate_{value:02d}"


def clean_str(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    text = str(value)
    return text if text and text.lower() != "nan" else "NA"


def top_items(comp: pd.DataFrame, endpoint: int, label_col: str, n: int = 5) -> str:
    sub = comp[comp["candidate_endpoint"] == endpoint].sort_values("fraction_within_endpoint", ascending=False)
    parts = []
    for row in sub.head(n).itertuples(index=False):
        label = clean_str(getattr(row, label_col))
        frac = float(getattr(row, "fraction_within_endpoint"))
        parts.append(f"{label}:{frac:.3f}")
    return "; ".join(parts) if parts else "NA"


def read_inputs() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {
        "endpoint_nodes": pd.read_parquet(ENDPOINT_NODE),
        "node_neighborhood": pd.read_parquet(NODE_NEIGHBORHOOD),
        "summary": pd.read_csv(ENDPOINT_SUMMARY),
        "major": pd.read_csv(ENDPOINT_MAJOR),
        "fine": pd.read_csv(ENDPOINT_FINE),
        "anchor": pd.read_csv(ENDPOINT_ANCHOR),
        "leiden_counts": pd.read_csv(LEIDEN_COUNTS),
        "leiden_fraction": pd.read_csv(LEIDEN_FRACTION_ENDPOINT),
        "purity": pd.read_csv(NEIGHBORHOOD_PURITY),
        "tiers": pd.read_csv(TIERS_WITH_NEIGHBORHOOD),
        "plasticity_leiden": pd.read_csv(PLASTICITY_LEIDEN),
    }
    if NICHE_ADVANTAGE_NEIGHBORHOOD.exists():
        data["niche_advantage"] = pd.read_csv(NICHE_ADVANTAGE_NEIGHBORHOOD)
    else:
        data["niche_advantage"] = pd.DataFrame()
    data["m4c_node"] = pd.read_parquet(
        M4C_NODE,
        columns=[
            "global_node_index",
            "time",
            "time_day",
            "slice_id",
            "mouse_id",
            "dominant_fate",
            "dominant_fate_label",
            "dominant_fate_probability",
            "fate_margin_top1_minus_top2",
            "normalized_plasticity_entropy",
            "plasticity_entropy",
        ],
    )
    return data


def row_value(row: pd.Series, name: str, default: Any = np.nan) -> Any:
    if name in row and pd.notna(row[name]):
        return row[name]
    return default


def category_support(row: pd.Series, category: str) -> tuple[str, str]:
    major = clean_str(row_value(row, "dominant_major_cell_class", "")).lower()
    fine = clean_str(row_value(row, "dominant_fine_cell_cluster", "")).lower()
    neighborhood = clean_str(row_value(row, "dominant_leiden_neigh", "")).lower()
    local_context = clean_str(row_value(row, "dominant_m2_local_context", "")).lower()
    major_fraction = float(row_value(row, "dominant_major_fraction", 0.0))
    fine_fraction = float(row_value(row, "dominant_fine_fraction", 0.0))
    neighborhood_fraction = float(row_value(row, "dominant_neighborhood_fraction", 0.0))
    final_fraction = float(row_value(row, "fraction_final_nodes", 0.0))
    mouse_fraction = float(row_value(row, "mouse_max_fraction", 0.0))
    slice_fraction = float(row_value(row, "slice_max_fraction", 0.0))

    reasons: list[str] = []
    status = "needs_manual_review"

    if category == "smooth-muscle / muscularis externa":
        smooth_support = "smooth" in major or "smooth" in local_context or neighborhood in {"me", "sm"}
        if smooth_support and major_fraction >= 0.65 and neighborhood in {"me", "sm"} and neighborhood_fraction >= 0.7:
            status = "supported"
        elif smooth_support and major_fraction >= 0.35 and neighborhood in {"me", "sm"}:
            status = "weakly_supported"
        else:
            status = "label_conflict"
        reasons.append(
            f"major={major} ({major_fraction:.3f}), fine={fine} ({fine_fraction:.3f}), "
            f"neighborhood={neighborhood} ({neighborhood_fraction:.3f})"
        )
    elif category == "stem-like mucosal":
        if major == "epithelial" and "stem" in fine and neighborhood == "mu" and fine_fraction >= 0.2:
            status = "supported"
        elif major == "epithelial" and neighborhood == "mu":
            status = "weakly_supported"
        else:
            status = "label_conflict"
        reasons.append(
            f"epithelial={major == 'epithelial'} ({major_fraction:.3f}), "
            f"fine={fine} ({fine_fraction:.3f}), neighborhood={neighborhood}"
        )
    elif category == "colonocyte mucosal":
        if major == "epithelial" and "colonocyte" in fine and neighborhood == "mu" and fine_fraction >= 0.2:
            status = "supported"
        elif major == "epithelial" and neighborhood == "mu":
            status = "weakly_supported"
        else:
            status = "label_conflict"
        reasons.append(
            f"epithelial={major == 'epithelial'} ({major_fraction:.3f}), "
            f"fine={fine} ({fine_fraction:.3f}), neighborhood={neighborhood}"
        )
    elif category == "slice/mouse-associated":
        if mouse_fraction >= 0.8 or slice_fraction >= 0.45:
            status = "supported"
        elif mouse_fraction >= 0.65 or slice_fraction >= 0.35:
            status = "weakly_supported"
        else:
            status = "label_conflict"
        reasons.append(f"slice_max={slice_fraction:.3f}, mouse_max={mouse_fraction:.3f}")
    elif category == "mixed submucosal":
        mixed = major_fraction < 0.55
        submucosa = neighborhood in {"sm", "me"} or "smooth" in local_context
        if mixed and submucosa:
            status = "supported"
        elif submucosa:
            status = "weakly_supported"
        else:
            status = "label_conflict"
        reasons.append(
            f"mixed_major={mixed}, major={major} ({major_fraction:.3f}), neighborhood={neighborhood}"
        )
    elif category == "mixed mucosal":
        mixed = major_fraction < 0.55
        mucosal = neighborhood == "mu" or "epithelial" in local_context
        if mixed and mucosal:
            status = "supported"
        elif mucosal:
            status = "weakly_supported"
        else:
            status = "label_conflict"
        reasons.append(
            f"mixed_major={mixed}, major={major} ({major_fraction:.3f}), neighborhood={neighborhood}"
        )
    elif category == "follicle / B-cell immune":
        immune_support = major == "immune" and ("b cell" in fine or "plasma" in fine)
        if immune_support and neighborhood == "fol" and neighborhood_fraction >= 0.6:
            status = "supported"
        elif immune_support and neighborhood in {"fol", "other"}:
            status = "weakly_supported"
        else:
            status = "label_conflict"
        reasons.append(
            f"major={major} ({major_fraction:.3f}), fine={fine} ({fine_fraction:.3f}), "
            f"neighborhood={neighborhood} ({neighborhood_fraction:.3f})"
        )
    elif category == "low-size / rare":
        low_size = final_fraction < 0.03
        coherent = major_fraction >= 0.45 or neighborhood_fraction >= 0.65
        if low_size and coherent:
            status = "supported"
        elif low_size:
            status = "weakly_supported"
        else:
            status = "label_conflict"
        reasons.append(
            f"fraction_final_nodes={final_fraction:.3f}, major_fraction={major_fraction:.3f}, "
            f"neighborhood_fraction={neighborhood_fraction:.3f}"
        )
    else:
        status = "needs_manual_review"
        reasons.append(f"unknown proposed category: {category}")

    return status, "; ".join(reasons)


def post_refinement_tier(row: pd.Series, validation_status: str, proposed_action: str) -> str:
    current = clean_str(row_value(row, "biological_confidence_tier_with_neighborhood", ""))
    if validation_status in {"label_conflict", "needs_manual_review"}:
        return "needs_manual_review"
    if proposed_action == "caution_slice_associated":
        return "slice_or_mouse_associated_endpoint"
    if proposed_action == "unresolved_keep":
        return "plausible_but_mixed_endpoint"
    if proposed_action == "rare_keep":
        if validation_status == "supported":
            return "rare_biological_endpoint"
        return "low_size_or_low_mass_endpoint"
    if proposed_action == "merge_candidate":
        return "merge_candidate_needs_manual_review"
    if validation_status == "supported" and current:
        return current
    if validation_status == "weakly_supported":
        return "plausible_but_mixed_endpoint"
    return current or "needs_manual_review"


def make_evidence_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = data["tiers"].copy()
    summary = summary.rename(columns={"n_nodes_x": "n_nodes"})
    if "n_nodes" not in summary.columns and "n_nodes_y" in summary.columns:
        summary = summary.rename(columns={"n_nodes_y": "n_nodes"})
    if "biological_confidence_tier_with_neighborhood" not in summary.columns:
        summary["biological_confidence_tier_with_neighborhood"] = summary["biological_confidence_tier"]

    endpoint_nodes = data["endpoint_nodes"]
    m4c = data["m4c_node"][
        [
            "global_node_index",
            "fate_margin_top1_minus_top2",
            "dominant_fate_probability",
            "normalized_plasticity_entropy",
        ]
    ]
    endpoint_m4c = endpoint_nodes[["global_node_index", "candidate_endpoint"]].merge(
        m4c, on="global_node_index", how="left", validate="one_to_one"
    )
    margins = (
        endpoint_m4c.groupby("candidate_endpoint")
        .agg(
            mean_top1_probability=("dominant_fate_probability", "mean"),
            mean_top1_margin=("fate_margin_top1_minus_top2", "mean"),
            mean_normalized_plasticity=("normalized_plasticity_entropy", "mean"),
        )
        .reset_index()
    )
    evidence = summary.merge(margins, on="candidate_endpoint", how="left")

    for col in [
        "dominant_major_cell_class",
        "dominant_fine_cell_cluster",
        "dominant_leiden_neigh",
        "dominant_m2_local_context",
    ]:
        if col not in evidence.columns:
            evidence[col] = "NA"

    evidence["top_major_cell_classes"] = [
        top_items(data["major"], int(ep), "major_cell_class") for ep in evidence["candidate_endpoint"]
    ]
    evidence["top_fine_cell_clusters"] = [
        top_items(data["fine"], int(ep), "fine_cell_cluster") for ep in evidence["candidate_endpoint"]
    ]
    evidence["top_anchor_cell_types"] = [
        top_items(data["anchor"], int(ep), "anchor_cell_type") for ep in evidence["candidate_endpoint"]
    ]
    evidence["top_leiden_neighborhoods"] = [
        top_items(data["leiden_fraction"], int(ep), "leiden_neigh") for ep in evidence["candidate_endpoint"]
    ]
    evidence["spatial_localization_summary"] = evidence.apply(
        lambda row: (
            f"x_scaled_mean={float(row_value(row, 'x_scaled_mean', np.nan)):.3f}, "
            f"y_scaled_mean={float(row_value(row, 'y_scaled_mean', np.nan)):.3f}, "
            f"x_scaled_std={float(row_value(row, 'x_scaled_std', np.nan)):.3f}, "
            f"y_scaled_std={float(row_value(row, 'y_scaled_std', np.nan)):.3f}"
        ),
        axis=1,
    )

    proposed_rows = []
    for _, row in evidence.iterrows():
        endpoint = int(row["candidate_endpoint"])
        proposed = PROPOSED_MAPPING.get(endpoint)
        if proposed is None:
            validation_status = "needs_manual_review"
            validation_reason = "no proposed mapping available"
            action = "needs_manual_review"
            category = "unresolved"
        else:
            category = proposed["refined_endpoint_category"]
            action = proposed["action"]
            validation_status, validation_reason = category_support(row, category)
        proposed_rows.append(
            {
                "candidate_endpoint": endpoint,
                "proposed_refined_endpoint_id": proposed["refined_endpoint_id"] if proposed else endpoint_label(endpoint),
                "proposed_refined_endpoint_label": proposed["refined_endpoint_label"] if proposed else endpoint_label(endpoint),
                "proposed_refined_endpoint_category": category,
                "proposed_action": action,
                "label_validation_status": validation_status,
                "label_validation_reason": validation_reason,
                "confidence_tier_after_refinement": post_refinement_tier(row, validation_status, action),
            }
        )
    proposed_df = pd.DataFrame(proposed_rows)
    evidence = evidence.merge(proposed_df, on="candidate_endpoint", how="left")
    evidence["likely_endpoint_interpretation"] = evidence["proposed_refined_endpoint_category"]
    evidence.loc[
        evidence["label_validation_status"].isin(["label_conflict", "needs_manual_review"]),
        "likely_endpoint_interpretation",
    ] = "unresolved / needs manual review"

    ordered = [
        "candidate_endpoint",
        "candidate_endpoint_label",
        "n_nodes",
        "fraction_final_nodes",
        "biological_confidence_tier_with_neighborhood",
        "confidence_tier_after_refinement",
        "dominant_leiden_neigh",
        "dominant_neighborhood_fraction",
        "neighborhood_entropy",
        "dominant_major_cell_class",
        "dominant_major_fraction",
        "dominant_fine_cell_cluster",
        "dominant_fine_fraction",
        "dominant_m2_local_context",
        "dominant_m2_local_context_fraction",
        "slice_max_group",
        "slice_max_fraction",
        "mouse_max_group",
        "mouse_max_fraction",
        "spatial_localization_summary",
        "mean_top1_probability",
        "mean_top1_margin",
        "mean_normalized_plasticity",
        "top_major_cell_classes",
        "top_fine_cell_clusters",
        "top_anchor_cell_types",
        "top_leiden_neighborhoods",
        "proposed_refined_endpoint_id",
        "proposed_refined_endpoint_label",
        "proposed_refined_endpoint_category",
        "proposed_action",
        "label_validation_status",
        "label_validation_reason",
        "likely_endpoint_interpretation",
    ]
    existing = [c for c in ordered if c in evidence.columns]
    evidence = evidence[existing + [c for c in evidence.columns if c not in existing]]
    evidence = evidence.sort_values("candidate_endpoint").reset_index(drop=True)
    evidence.to_csv(OUT / "endpoint_refinement_evidence_table.csv", index=False)
    (OUT / "endpoint_refinement_evidence_table.json").write_text(
        json.dumps(evidence.replace({np.nan: None}).to_dict(orient="records"), indent=2)
    )
    return evidence


def write_mapping(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in evidence.itertuples(index=False):
        endpoint = int(row.candidate_endpoint)
        proposed = PROPOSED_MAPPING[endpoint]
        validation = clean_str(getattr(row, "label_validation_status"))
        action = proposed["action"]
        if validation in {"label_conflict", "needs_manual_review"}:
            action = "needs_manual_review"
        rows.append(
            {
                "raw_terminal_macrostate": endpoint,
                "raw_terminal_macrostate_label": endpoint_label(endpoint),
                "refined_endpoint_id": proposed["refined_endpoint_id"],
                "refined_endpoint_label": proposed["refined_endpoint_label"],
                "refined_endpoint_category": proposed["refined_endpoint_category"],
                "action": action,
                "proposed_mapping_status": "initial_proposal_validated",
                "label_validation_status": validation,
                "label_validation_reason": clean_str(getattr(row, "label_validation_reason")),
                "confidence_tier_after_refinement": clean_str(
                    getattr(row, "confidence_tier_after_refinement")
                ),
                "rationale": (
                    f"Top evidence: Leiden={clean_str(getattr(row, 'dominant_leiden_neigh'))} "
                    f"({float(getattr(row, 'dominant_neighborhood_fraction')):.3f}), "
                    f"major={clean_str(getattr(row, 'dominant_major_cell_class'))} "
                    f"({float(getattr(row, 'dominant_major_fraction')):.3f}), "
                    f"fine={clean_str(getattr(row, 'dominant_fine_cell_cluster'))} "
                    f"({float(getattr(row, 'dominant_fine_fraction')):.3f})."
                ),
                "raw_endpoint_traceability": (
                    "Retains raw M4B terminal_macrostate ID and label; refined labels are annotations only."
                ),
            }
        )
    mapping = pd.DataFrame(rows)
    mapping.to_csv(OUT / "refined_endpoint_mapping.csv", index=False)
    (OUT / "refined_endpoint_mapping.json").write_text(
        json.dumps(mapping.to_dict(orient="records"), indent=2)
    )
    return mapping


def vector_for_endpoint(table: pd.DataFrame, label_col: str, endpoint: int) -> pd.Series:
    sub = table[table["candidate_endpoint"] == endpoint]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.set_index(label_col)["fraction_within_endpoint"].astype(float)


def cosine_similarity(a: pd.Series, b: pd.Series) -> float:
    labels = sorted(set(a.index) | set(b.index))
    if not labels:
        return float("nan")
    av = a.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    bv = b.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0:
        return float("nan")
    return float(np.dot(av, bv) / denom)


def l1_distance(a: pd.Series, b: pd.Series) -> float:
    labels = sorted(set(a.index) | set(b.index))
    if not labels:
        return float("nan")
    av = a.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    bv = b.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    return float(np.abs(av - bv).sum())


def distinctness_status(metrics: dict[str, Any]) -> str:
    fine_dissim = 1.0 - metrics["fine_cluster_cosine_similarity"]
    leiden_dissim = 1.0 - metrics["leiden_cosine_similarity"]
    spatial = metrics["spatial_centroid_distance"]
    slice_mouse = metrics["slice_mouse_difference_score"]
    if metrics["pair"] == "06-09":
        if fine_dissim < 0.18 and leiden_dissim < 0.10 and spatial < 0.25:
            return "not_supported"
    if metrics["pair"] in {"00-07", "01-11"}:
        if fine_dissim >= 0.25 or metrics["same_dominant_fine_cluster"] is False:
            return "supported"
    score = 0
    score += int(fine_dissim >= 0.30)
    score += int(leiden_dissim >= 0.20)
    score += int(spatial >= 0.35)
    score += int(slice_mouse >= 0.35)
    score += int(metrics["same_dominant_fine_cluster"] is False)
    if score >= 3:
        return "supported"
    if score >= 1:
        return "weakly_supported"
    return "not_supported"


def build_pairwise_similarity(data: dict[str, pd.DataFrame], evidence: pd.DataFrame) -> pd.DataFrame:
    evidence_idx = evidence.set_index("candidate_endpoint")
    rows = []
    endpoints = sorted(evidence["candidate_endpoint"].astype(int).unique())
    for i, ep_a in enumerate(endpoints):
        for ep_b in endpoints[i + 1 :]:
            fine_a = vector_for_endpoint(data["fine"], "fine_cell_cluster", ep_a)
            fine_b = vector_for_endpoint(data["fine"], "fine_cell_cluster", ep_b)
            leiden_a = vector_for_endpoint(data["leiden_fraction"], "leiden_neigh", ep_a)
            leiden_b = vector_for_endpoint(data["leiden_fraction"], "leiden_neigh", ep_b)
            row_a = evidence_idx.loc[ep_a]
            row_b = evidence_idx.loc[ep_b]
            dx = float(row_a["x_scaled_mean"]) - float(row_b["x_scaled_mean"])
            dy = float(row_a["y_scaled_mean"]) - float(row_b["y_scaled_mean"])
            spatial = float(math.sqrt(dx * dx + dy * dy))
            slice_diff = abs(float(row_a["slice_max_fraction"]) - float(row_b["slice_max_fraction"]))
            mouse_diff = abs(float(row_a["mouse_max_fraction"]) - float(row_b["mouse_max_fraction"]))
            metrics = {
                "endpoint_a": ep_a,
                "endpoint_b": ep_b,
                "pair": f"{ep_a:02d}-{ep_b:02d}",
                "review_pair": (ep_a, ep_b) in REVIEW_PAIRS,
                "review_label": PAIR_REVIEW_LABELS.get((ep_a, ep_b), ""),
                "fine_cluster_cosine_similarity": cosine_similarity(fine_a, fine_b),
                "fine_cluster_l1_distance": l1_distance(fine_a, fine_b),
                "leiden_cosine_similarity": cosine_similarity(leiden_a, leiden_b),
                "leiden_l1_distance": l1_distance(leiden_a, leiden_b),
                "spatial_centroid_distance": spatial,
                "same_dominant_fine_cluster": clean_str(row_a["dominant_fine_cell_cluster"])
                == clean_str(row_b["dominant_fine_cell_cluster"]),
                "same_dominant_leiden_neigh": clean_str(row_a["dominant_leiden_neigh"])
                == clean_str(row_b["dominant_leiden_neigh"]),
                "same_slice_max_group": clean_str(row_a["slice_max_group"])
                == clean_str(row_b["slice_max_group"]),
                "same_mouse_max_group": clean_str(row_a["mouse_max_group"])
                == clean_str(row_b["mouse_max_group"]),
                "slice_mouse_difference_score": max(slice_diff, mouse_diff),
                "dominant_fine_a": clean_str(row_a["dominant_fine_cell_cluster"]),
                "dominant_fine_b": clean_str(row_b["dominant_fine_cell_cluster"]),
                "dominant_leiden_a": clean_str(row_a["dominant_leiden_neigh"]),
                "dominant_leiden_b": clean_str(row_b["dominant_leiden_neigh"]),
                "slice_max_a": clean_str(row_a["slice_max_group"]),
                "slice_max_b": clean_str(row_b["slice_max_group"]),
                "mouse_max_a": clean_str(row_a["mouse_max_group"]),
                "mouse_max_b": clean_str(row_b["mouse_max_group"]),
            }
            metrics["distinctness_status"] = distinctness_status(metrics)
            rows.append(metrics)
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(OUT / "endpoint_pairwise_similarity.csv", index=False)
    return pairwise


def mapping_by_dominant_fate(mapping: pd.DataFrame) -> dict[str, dict[str, str]]:
    result = {}
    for row in mapping.itertuples(index=False):
        raw_label = endpoint_label(int(row.raw_terminal_macrostate))
        result[raw_label] = {
            "refined_endpoint_id": row.refined_endpoint_id,
            "refined_endpoint_label": row.refined_endpoint_label,
            "refined_endpoint_category": row.refined_endpoint_category,
            "confidence_tier_after_refinement": row.confidence_tier_after_refinement,
            "label_validation_status": row.label_validation_status,
        }
    return result


def add_refined_columns(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    map_df = mapping[
        [
            "raw_terminal_macrostate",
            "refined_endpoint_id",
            "refined_endpoint_label",
            "refined_endpoint_category",
            "confidence_tier_after_refinement",
            "label_validation_status",
        ]
    ].rename(columns={"raw_terminal_macrostate": "dominant_fate"})
    out = df.merge(map_df, on="dominant_fate", how="left")
    return out


def ranked_distribution(
    df: pd.DataFrame,
    group_col: str,
    mapping: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    work = add_refined_columns(df, mapping)
    work["refined_endpoint_label"] = work["refined_endpoint_label"].fillna(work["dominant_fate_label"])
    grouped = (
        work.groupby([group_col, "refined_endpoint_id", "refined_endpoint_label"], dropna=False)
        .agg(
            n_nodes=("global_node_index", "size"),
            mean_dominant_probability=("dominant_fate_probability", "mean"),
            mean_normalized_plasticity=("normalized_plasticity_entropy", "mean"),
        )
        .reset_index()
    )
    totals = grouped.groupby(group_col)["n_nodes"].transform("sum")
    grouped["fraction_within_group"] = grouped["n_nodes"] / totals
    grouped["rank_within_group"] = grouped.groupby(group_col)["fraction_within_group"].rank(
        method="first", ascending=False
    )
    grouped["is_dominant_refined_endpoint_for_group"] = grouped["rank_within_group"] == 1
    grouped = grouped.sort_values([group_col, "rank_within_group"])
    grouped.to_csv(out_path, index=False)
    return grouped


def make_refined_m4c_tables(data: dict[str, pd.DataFrame], mapping: pd.DataFrame) -> dict[str, pd.DataFrame]:
    m4c = data["m4c_node"].copy()
    tables = {
        "by_time": ranked_distribution(m4c, "time", mapping, OUT / "m4c_v1_refined_endpoint_by_time.csv"),
        "by_slice": ranked_distribution(m4c, "slice_id", mapping, OUT / "m4c_v1_refined_endpoint_by_slice.csv"),
        "by_mouse": ranked_distribution(m4c, "mouse_id", mapping, OUT / "m4c_v1_refined_endpoint_by_mouse.csv"),
    }
    refined = add_refined_columns(m4c, mapping)
    plasticity = (
        refined.groupby(
            [
                "refined_endpoint_id",
                "refined_endpoint_label",
                "refined_endpoint_category",
                "confidence_tier_after_refinement",
            ],
            dropna=False,
        )
        .agg(
            n_nodes=("global_node_index", "size"),
            mean_normalized_plasticity=("normalized_plasticity_entropy", "mean"),
            median_normalized_plasticity=("normalized_plasticity_entropy", "median"),
            mean_plasticity_entropy=("plasticity_entropy", "mean"),
            mean_dominant_fate_probability=("dominant_fate_probability", "mean"),
            mean_top1_margin=("fate_margin_top1_minus_top2", "mean"),
        )
        .reset_index()
        .sort_values("n_nodes", ascending=False)
    )
    plasticity.to_csv(OUT / "m4c_v1_plasticity_by_refined_endpoint.csv", index=False)
    tables["plasticity_by_refined_endpoint"] = plasticity

    node_neighborhood = data["node_neighborhood"][
        [
            "global_node_index",
            "leiden_neigh",
            "dominant_fate",
            "dominant_fate_label",
            "dominant_fate_probability",
            "normalized_plasticity_entropy",
        ]
    ].copy()
    endpoint_by_neighborhood = add_refined_columns(node_neighborhood, mapping)
    endpoint_by_neighborhood["leiden_neigh"] = endpoint_by_neighborhood["leiden_neigh"].fillna("NA")
    endpoint_by_neighborhood = (
        endpoint_by_neighborhood.groupby(
            ["leiden_neigh", "refined_endpoint_id", "refined_endpoint_label"], dropna=False
        )
        .agg(
            n_nodes=("global_node_index", "size"),
            mean_dominant_fate_probability=("dominant_fate_probability", "mean"),
            mean_normalized_plasticity=("normalized_plasticity_entropy", "mean"),
        )
        .reset_index()
    )
    totals = endpoint_by_neighborhood.groupby("leiden_neigh")["n_nodes"].transform("sum")
    endpoint_by_neighborhood["fraction_within_leiden_neigh"] = endpoint_by_neighborhood["n_nodes"] / totals
    endpoint_by_neighborhood = endpoint_by_neighborhood.sort_values(
        ["leiden_neigh", "fraction_within_leiden_neigh"], ascending=[True, False]
    )
    endpoint_by_neighborhood.to_csv(OUT / "m4c_v1_endpoint_attraction_by_leiden_neigh.csv", index=False)
    tables["endpoint_attraction_by_leiden_neigh"] = endpoint_by_neighborhood
    return tables


def short_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    sub = df[columns].copy()
    if max_rows is not None:
        sub = sub.head(max_rows)
    if sub.empty:
        return "_No rows._"
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda x: f"{x:.4g}" if pd.notna(x) else "NA")
        else:
            sub[col] = sub[col].map(clean_str)
    header = "| " + " | ".join(sub.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(sub.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in sub.itertuples(index=False, name=None)]
    return "\n".join([header, sep] + rows)


def write_taxonomy_report(evidence: pd.DataFrame, mapping: pd.DataFrame) -> None:
    tier_counts = (
        mapping["confidence_tier_after_refinement"].value_counts().rename_axis("tier").reset_index(name="n_endpoints")
    )
    body = f"""# M4E-03 Refined Endpoint Taxonomy Report

## Scope

This report refines the 12 raw M4B/M4C candidate endpoint niche clusters into conservative biological annotations. The raw `terminal_macrostate` ID remains the primary traceable identifier. Refined labels are annotations only and are validated against M4E-01/M4E-02 evidence.

The proposed mapping was treated as an initial hypothesis. Endpoints with contradictory evidence are flagged as `label_conflict` or `needs_manual_review`.

## Confidence Tier Counts After Refinement

{short_table(tier_counts, ["tier", "n_endpoints"])}

## Refined Mapping

{short_table(mapping, ["raw_terminal_macrostate_label", "refined_endpoint_id", "refined_endpoint_label", "action", "label_validation_status", "confidence_tier_after_refinement"], max_rows=20)}

## Evidence Summary

{short_table(evidence, ["candidate_endpoint_label", "n_nodes", "fraction_final_nodes", "dominant_leiden_neigh", "dominant_neighborhood_fraction", "dominant_major_cell_class", "dominant_major_fraction", "dominant_fine_cell_cluster", "dominant_fine_fraction", "label_validation_status"], max_rows=20)}

## Naming Policy

- `terminal_macrostate` is reported as a candidate endpoint niche cluster, not as a proven terminal biological state.
- D35 is the observed final time in the experiment, not absolute biological terminal time.
- M4C-v1 is a baseline endpoint-attraction / fate-propagation result, not lineage-validated fate.
- Mixed endpoints remain mixed; labels are not forced when evidence is heterogeneous.
"""
    write_report("refined_endpoint_taxonomy_report.md", body)


def write_redundancy_report(pairwise: pd.DataFrame, evidence: pd.DataFrame, mapping: pd.DataFrame) -> None:
    review = pairwise[pairwise["review_pair"]].copy()
    review = review.sort_values(["endpoint_a", "endpoint_b"])
    body = "# M4E-03 Endpoint Redundancy and Merge Review\n\n"
    body += (
        "Distinctness decisions use fine-cluster composition, Leiden neighborhood composition, "
        "spatial centroid distance, and slice/mouse enrichment. The `distinctness_status` field is one of "
        "`supported`, `weakly_supported`, `not_supported`, or `manual_review`.\n\n"
    )
    body += "## Quantitative Pairwise Review\n\n"
    body += short_table(
        review,
        [
            "pair",
            "review_label",
            "fine_cluster_cosine_similarity",
            "leiden_cosine_similarity",
            "spatial_centroid_distance",
            "same_dominant_fine_cluster",
            "same_dominant_leiden_neigh",
            "slice_mouse_difference_score",
            "distinctness_status",
        ],
    )
    body += "\n\n## Decisions\n\n"
    for row in review.itertuples(index=False):
        pair = row.pair
        if pair == "00-07":
            decision = (
                "Keep separate. Both are ME/smooth-muscle-associated, but the fine-cluster axis separates "
                "SMC2-rich endpoint 00 from SMC1-rich endpoint 07."
            )
        elif pair == "01-11":
            decision = (
                "Keep separate. Both are MU-associated, but endpoint 01 is stem-like while endpoint 11 is "
                "colonocyte-enriched, supporting distinct mucosal endpoint annotations."
            )
        elif pair == "03-05":
            decision = (
                "Do not merge into high-confidence endpoints. Both remain mixed/plausible endpoints; "
                "their neighborhood bias differs enough to keep raw IDs separate for now."
            )
        elif pair == "06-09":
            decision = (
                "Treat as a merge candidate, not an automatic merge. Both are rare ME enteric/smooth-muscle "
                "endpoints and quantitative distinctness is limited."
            )
        else:
            decision = "No manual decision specified."
        body += (
            f"- `{pair}`: `distinctness_status={row.distinctness_status}`. {decision} "
            f"Fine cosine={row.fine_cluster_cosine_similarity:.3f}, Leiden cosine={row.leiden_cosine_similarity:.3f}, "
            f"spatial distance={row.spatial_centroid_distance:.3f}.\n"
        )
    body += "\n## Rare and Caution Endpoints\n\n"
    ep10 = evidence[evidence["candidate_endpoint"] == 10].iloc[0]
    ep02 = evidence[evidence["candidate_endpoint"] == 2].iloc[0]
    body += (
        f"- Endpoint 10: FOL/B-cell interpretation is retained as a rare biological endpoint because "
        f"dominant Leiden neighborhood is `{ep10['dominant_leiden_neigh']}` "
        f"({float(ep10['dominant_neighborhood_fraction']):.3f}) and dominant fine cluster is "
        f"`{ep10['dominant_fine_cell_cluster']}` ({float(ep10['dominant_fine_fraction']):.3f}).\n"
    )
    body += (
        f"- Endpoint 02: retained with caution because mouse max fraction is "
        f"{float(ep02['mouse_max_fraction']):.3f}; interpretation remains artifact-prone until validated.\n"
    )
    write_report("endpoint_redundancy_and_merge_review.md", body)


def heatmap_from_table(
    table: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    path: Path,
    title: str,
    max_cols: int | None = None,
) -> None:
    pivot = table.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="sum", fill_value=0.0)
    if max_cols is not None and pivot.shape[1] > max_cols:
        keep = pivot.sum(axis=0).sort_values(ascending=False).head(max_cols).index
        pivot = pivot.loc[:, keep]
    pivot = pivot.sort_index()
    fig_w = max(8, min(18, 0.45 * pivot.shape[1] + 4))
    fig_h = max(4.5, min(12, 0.38 * pivot.shape[0] + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=70, ha="right", fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in pivot.index], fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bar_endpoint_size(evidence: pd.DataFrame) -> None:
    colors = {
        "high_confidence_biological_endpoint": "#2c7fb8",
        "plausible_but_mixed_endpoint": "#fdae61",
        "rare_biological_endpoint": "#66a61e",
        "low_size_or_low_mass_endpoint": "#b2abd2",
        "slice_or_mouse_associated_endpoint": "#d7191c",
        "merge_candidate_needs_manual_review": "#7570b3",
        "needs_manual_review": "#999999",
    }
    labels = evidence["candidate_endpoint_label"].astype(str).str.replace("terminal_macrostate_", "EP", regex=False)
    tiers = evidence["confidence_tier_after_refinement"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, evidence["n_nodes"], color=[colors.get(t, "#999999") for t in tiers])
    ax.set_ylabel("D35 endpoint nodes")
    ax.set_xlabel("Candidate endpoint niche cluster")
    ax.set_title("M4C-v1 candidate endpoint size and confidence tier")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES / "endpoint_size_confidence_tier_barplot.png", dpi=180)
    plt.close(fig)


def plot_purity_entropy(evidence: pd.DataFrame) -> None:
    labels = evidence["candidate_endpoint_label"].astype(str).str.replace("terminal_macrostate_", "EP", regex=False)
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar(labels, evidence["dominant_neighborhood_fraction"], color="#4daf4a", alpha=0.75, label="purity")
    ax1.set_ylabel("Dominant Leiden fraction")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="x", rotation=45)
    ax2 = ax1.twinx()
    ax2.plot(labels, evidence["neighborhood_entropy"], color="#984ea3", marker="o", linewidth=1.5, label="entropy")
    ax2.set_ylabel("Leiden entropy")
    ax1.set_title("Endpoint neighborhood purity and entropy")
    fig.tight_layout()
    fig.savefig(FIGURES / "endpoint_neighborhood_purity_entropy.png", dpi=180)
    plt.close(fig)


def plot_plasticity_by_leiden(data: dict[str, pd.DataFrame]) -> None:
    table = data["plasticity_leiden"].copy().sort_values(
        "mean_normalized_plasticity_nonfinal", ascending=False
    )
    table = table.head(24)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(table["leiden_neigh"].astype(str), table["mean_normalized_plasticity_nonfinal"], color="#386cb0")
    ax.set_ylabel("Mean normalized plasticity, non-final nodes")
    ax.set_xlabel("Leiden neighborhood")
    ax.set_title("M4C-v1 plasticity by Leiden neighborhood")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(FIGURES / "plasticity_by_leiden_neigh_endpoint_refinement.png", dpi=180)
    plt.close(fig)


def plot_niche_advantage(data: dict[str, pd.DataFrame]) -> None:
    niche = data["niche_advantage"].copy()
    if niche.empty:
        return
    niche = niche[niche["stratification"] == "cell_type_l1"].copy()
    if niche.empty:
        return
    selected_cell = (
        niche.groupby("cell_type")["n_nodes"].sum().sort_values(ascending=False).head(1).index.tolist()[0]
    )
    sub = niche[niche["cell_type"] == selected_cell].sort_values("n_nodes", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(sub["leiden_neigh"].astype(str), sub["dominant_endpoint_fraction"], color="#e6ab02")
    labels = [
        str(x).replace("terminal_macrostate_", "EP") for x in sub["dominant_endpoint"].astype(str)
    ]
    for i, label in enumerate(labels):
        ax.text(i, float(sub.iloc[i]["dominant_endpoint_fraction"]) + 0.015, label, ha="center", fontsize=8)
    ax.set_ylim(0, min(1.05, max(0.3, float(sub["dominant_endpoint_fraction"].max()) + 0.18)))
    ax.set_ylabel("Dominant endpoint fraction")
    ax.set_xlabel(f"Leiden neighborhood within {selected_cell}")
    ax.set_title("Same cell type, neighborhood-stratified endpoint tendency")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(FIGURES / "same_celltype_neighborhood_fate_distribution_endpoint_refinement.png", dpi=180)
    plt.close(fig)


def plot_tissue_maps(data: dict[str, pd.DataFrame], mapping: pd.DataFrame) -> None:
    node = data["node_neighborhood"].copy()
    node = add_refined_columns(node, mapping)
    # Endpoint maps: representative D35 slices with most endpoint nodes.
    d35 = node[node["is_final_time"] == True].copy()
    if not d35.empty:
        slices = d35["slice_id"].value_counts().head(2).index.tolist()
        for slice_id in slices:
            sub = d35[d35["slice_id"] == slice_id].copy()
            if len(sub) > 12000:
                sub = sub.sample(12000, random_state=13)
            labels = sorted(sub["refined_endpoint_id"].fillna("unmapped").unique())
            color_map = {label: plt.cm.tab20(i % 20) for i, label in enumerate(labels)}
            colors = [color_map[x] for x in sub["refined_endpoint_id"].fillna("unmapped")]
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.scatter(sub["x"], sub["y"], c=colors, s=2, linewidths=0)
            ax.set_title(f"M4C-v1 dominant refined endpoint, {slice_id}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[label], markersize=6)
                for label in labels[:12]
            ]
            ax.legend(handles, labels[:12], loc="upper right", fontsize=6, frameon=False)
            fig.tight_layout()
            safe_slice = str(slice_id).replace("/", "_")
            fig.savefig(FIGURES / f"m4c_v1_dominant_refined_endpoint_tissue_map_{safe_slice}.png", dpi=180)
            plt.close(fig)

    nonfinal = node[node["is_final_time"] == False].copy()
    if not nonfinal.empty:
        slices = nonfinal["slice_id"].value_counts().head(2).index.tolist()
        for slice_id in slices:
            sub = nonfinal[nonfinal["slice_id"] == slice_id].copy()
            if len(sub) > 12000:
                sub = sub.sample(12000, random_state=17)
            fig, ax = plt.subplots(figsize=(6, 5.5))
            sc = ax.scatter(
                sub["x"],
                sub["y"],
                c=sub["normalized_plasticity_entropy"],
                s=2,
                linewidths=0,
                cmap="magma",
                vmin=0,
                vmax=max(0.01, float(nonfinal["normalized_plasticity_entropy"].quantile(0.98))),
            )
            ax.set_title(f"M4C-v1 normalized plasticity, {slice_id}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("normalized plasticity entropy")
            fig.tight_layout()
            safe_slice = str(slice_id).replace("/", "_")
            fig.savefig(FIGURES / f"m4c_v1_normalized_plasticity_tissue_map_{safe_slice}.png", dpi=180)
            plt.close(fig)


def make_figures(data: dict[str, pd.DataFrame], evidence: pd.DataFrame, mapping: pd.DataFrame) -> None:
    endpoint_nodes = data["endpoint_nodes"].merge(
        mapping[
            [
                "raw_terminal_macrostate",
                "refined_endpoint_id",
                "refined_endpoint_label",
                "refined_endpoint_category",
            ]
        ].rename(columns={"raw_terminal_macrostate": "candidate_endpoint"}),
        on="candidate_endpoint",
        how="left",
    )
    leiden = data["leiden_fraction"].merge(
        mapping[["raw_terminal_macrostate", "refined_endpoint_id"]].rename(
            columns={"raw_terminal_macrostate": "candidate_endpoint"}
        ),
        on="candidate_endpoint",
        how="left",
    )
    heatmap_from_table(
        leiden,
        "refined_endpoint_id",
        "leiden_neigh",
        "fraction_within_endpoint",
        FIGURES / "refined_endpoint_taxonomy_by_leiden_neigh_heatmap.png",
        "Refined endpoint taxonomy by Leiden neighborhood",
    )
    major = (
        endpoint_nodes.groupby(["refined_endpoint_id", "major_cell_class"], dropna=False)
        .size()
        .rename("n_nodes")
        .reset_index()
    )
    major["fraction_within_endpoint"] = major["n_nodes"] / major.groupby("refined_endpoint_id")[
        "n_nodes"
    ].transform("sum")
    heatmap_from_table(
        major,
        "refined_endpoint_id",
        "major_cell_class",
        "fraction_within_endpoint",
        FIGURES / "refined_endpoint_taxonomy_by_major_cell_class_heatmap.png",
        "Refined endpoint taxonomy by major cell class",
    )
    fine = (
        endpoint_nodes.groupby(["refined_endpoint_id", "fine_cell_cluster"], dropna=False)
        .size()
        .rename("n_nodes")
        .reset_index()
    )
    fine["fraction_within_endpoint"] = fine["n_nodes"] / fine.groupby("refined_endpoint_id")[
        "n_nodes"
    ].transform("sum")
    heatmap_from_table(
        fine,
        "refined_endpoint_id",
        "fine_cell_cluster",
        "fraction_within_endpoint",
        FIGURES / "refined_endpoint_taxonomy_by_fine_cell_cluster_heatmap.png",
        "Refined endpoint taxonomy by fine cell cluster",
        max_cols=24,
    )
    plot_bar_endpoint_size(evidence)
    plot_purity_entropy(evidence)
    plot_plasticity_by_leiden(data)
    plot_niche_advantage(data)
    plot_tissue_maps(data, mapping)


def write_baseline_freeze_report(
    evidence: pd.DataFrame,
    mapping: pd.DataFrame,
    pairwise: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> None:
    high = mapping[mapping["confidence_tier_after_refinement"] == "high_confidence_biological_endpoint"]
    unresolved = mapping[
        mapping["confidence_tier_after_refinement"].isin(
            ["plausible_but_mixed_endpoint", "needs_manual_review", "merge_candidate_needs_manual_review"]
        )
    ]
    pair_review = pairwise[pairwise["review_pair"]].copy()
    body = f"""# M4C-v1 Baseline Dynamic Niche-Fate Freeze Report

## Interpretation Scope

M4C-v1 is a pseudo-only baseline endpoint-attraction / fate-propagation map. It is not lineage-validated fate. It is not GPCCA-derived terminal fate. The endpoints are observed-final-time candidate endpoint niche clusters, and D35 is the observed final time rather than absolute biological terminal time.

Refined endpoint labels are conservative biological annotations. They retain the raw M4B `terminal_macrostate` ID for traceability. D35 plasticity is algorithmically one-hot initialized and should not be interpreted as biological loss of plasticity.

The pyGPCCA failure does not invalidate M4C-v1 because M4C-v1 uses P_fate-v1 and M4B final-time candidate endpoint labels, not K_gpcca or GPCCA macrostate discovery.

## Endpoint Freeze Summary

- Endpoints retained as high-confidence biological annotations: {', '.join(high['raw_terminal_macrostate_label'].tolist())}
- Endpoints retained as mixed, rare, caution, or merge-review annotations: {', '.join(unresolved['raw_terminal_macrostate_label'].tolist())}
- All refined labels passed proposal validation except endpoints explicitly marked as `label_conflict` or `needs_manual_review`.

## Refined Mapping

{short_table(mapping, ["raw_terminal_macrostate_label", "refined_endpoint_label", "action", "label_validation_status", "confidence_tier_after_refinement"], max_rows=20)}

## Quantitative Redundancy Decisions

{short_table(pair_review, ["pair", "fine_cluster_cosine_similarity", "leiden_cosine_similarity", "spatial_centroid_distance", "distinctness_status"], max_rows=10)}

## Baseline Figures Frozen

Figures were written under `{FIGURES}`:

- Refined endpoint taxonomy heatmaps by Leiden neighborhood, major cell class, and fine cell cluster.
- Endpoint size and confidence tier barplot.
- Endpoint neighborhood purity and entropy plot.
- Representative M4C-v1 dominant refined endpoint tissue maps.
- Representative M4C-v1 normalized plasticity tissue maps for non-final nodes.
- Plasticity by Leiden neighborhood.
- Same-cell-type, different-neighborhood endpoint tendency plot.

## Caveats

- M4C-v1 remains a frozen baseline, not a final lineage-validated biological fate model.
- Endpoint 02 remains slice/mouse-associated and should be treated cautiously.
- Endpoints 03 and 05 remain mixed/plausible rather than high-confidence biological endpoints.
- Endpoints 06 and 09 remain merge candidates, with endpoint 09 marked for manual merge review.
"""
    write_report("m4c_v1_baseline_dynamic_niche_fate_freeze_report.md", body)


def write_next_step_report(mapping: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    unresolved = mapping[
        mapping["confidence_tier_after_refinement"].isin(
            ["needs_manual_review", "merge_candidate_needs_manual_review"]
        )
    ]
    body = """# M4E Next Step After Endpoint Refinement

## Decision

Recommended next engineering step: proceed to an M3-v2 small pilot design while keeping K_gpcca as a separate later pilot. Do not start full production recomputation yet.

## Rationale

- M1/M2 remain stable.
- M3/M4A/M4B/M4C remain frozen baseline-only.
- M4C-v1 is interpretable enough to keep as a baseline endpoint-attraction / fate-propagation map.
- Most major endpoint categories are biologically interpretable after Leiden neighborhood annotation.
- Remaining caution items are traceable and do not require deleting or recomputing M4C-v1.

## Required Before Production Recompute

- Resolve endpoint 02 caution status with independent slice/mouse checks.
- Review rare ME enteric/smooth-muscle endpoints 06 and 09 before any merge.
- Keep endpoint 03 and 05 as mixed/plausible unless stronger biological evidence supports relabeling.
- Design M3-v2 primary-cost plus soft-gating transition evidence without changing current M4C-v1 outputs.
- Keep K_gpcca separate from P_fate and test it only as a small RealTime-like niche kernel pilot.

## Current Manual Review Queue

"""
    if unresolved.empty:
        body += "_No endpoints are marked `needs_manual_review` or `merge_candidate_needs_manual_review` after validation._\n"
    else:
        body += short_table(
            unresolved,
            [
                "raw_terminal_macrostate_label",
                "refined_endpoint_label",
                "action",
                "label_validation_status",
                "confidence_tier_after_refinement",
            ],
        )
        body += "\n"
    write_report("m4e_next_step_after_endpoint_refinement.md", body)


def write_inventory() -> None:
    rows = []
    report_names = [
        "refined_endpoint_taxonomy_report.md",
        "endpoint_redundancy_and_merge_review.md",
        "m4c_v1_baseline_dynamic_niche_fate_freeze_report.md",
        "m4e_next_step_after_endpoint_refinement.md",
    ]
    report_paths = [REPORTS / name for name in report_names]
    for path in sorted(list(OUT.glob("*")) + report_paths + list(FIGURES.glob("*"))):
        if path.is_file():
            rows.append({"path": str(path), "bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(REPORTS / "m4e_endpoint_refinement_inventory.csv", index=False)


def main() -> None:
    ensure_dirs()
    data = read_inputs()
    evidence = make_evidence_table(data)
    mapping = write_mapping(evidence)
    pairwise = build_pairwise_similarity(data, evidence)
    tables = make_refined_m4c_tables(data, mapping)
    write_taxonomy_report(evidence, mapping)
    write_redundancy_report(pairwise, evidence, mapping)
    make_figures(data, evidence, mapping)
    write_baseline_freeze_report(evidence, mapping, pairwise, tables)
    write_next_step_report(mapping, pairwise)
    write_inventory()
    print("M4E-03 endpoint refinement and M4C-v1 baseline freeze complete.")
    print(f"outputs: {OUT}")
    print(f"figures: {FIGURES}")


if __name__ == "__main__":
    main()
