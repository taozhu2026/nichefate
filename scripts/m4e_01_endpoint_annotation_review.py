#!/usr/bin/env python
"""M4E endpoint biological annotation and M4C baseline review.

This script is read-only with respect to M0-M4C artifacts. It writes only M4E
tables, reports, and lightweight figures.
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
import pyarrow.parquet as pq


ROOT = Path("/home/zhutao/scratch/nichefate")
PROJECT = Path("/home/zhutao/projects/nichefate")
OUT = ROOT / "m4e" / "endpoint_annotation"
REPORTS = ROOT / "m4e" / "reports"
FIGURES = REPORTS / "figures"

M4B_TERMINAL = ROOT / "m4b" / "terminal_states" / "terminal_macrostate_assignments.parquet"
M4B_SUMMARY = ROOT / "m4b" / "terminal_states" / "terminal_macrostate_summary.csv"
M4C_NODE = ROOT / "m4c" / "fate_probabilities" / "fate_probability_node_summary.parquet"
M4C_TIERS = ROOT / "m4c" / "reports" / "m4c_terminal_macrostate_confidence_tiers.csv"
M4A_NODE = ROOT / "m4a" / "node_table" / "global_node_table.parquet"
COORDS = ROOT / "m4d" / "visualization_layer" / "node_coordinates.parquet"
M2_BY_SLICE = ROOT / "m2" / "by_slice"
M2_SCHEMA = ROOT / "m2" / "reports" / "m2_full_feature_schema.json"
M0_INSPECTION = ROOT / "m0" / "reports" / "raw_anndata_inspection.json"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def parquet_info(path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    return {"path": str(path), "rows": pf.metadata.num_rows, "columns": pf.schema.names}


def entropy_from_counts(counts: pd.Series) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def top_label(series: pd.Series) -> tuple[str, float, int]:
    counts = series.fillna("NA").astype(str).value_counts()
    if counts.empty:
        return "NA", 0.0, 0
    label = str(counts.index[0])
    count = int(counts.iloc[0])
    return label, float(count / counts.sum()), count


def clean_feature_label(col: str) -> str:
    return (
        col.split("__")[-1]
        .replace("_", " ")
        .replace("smooth muscle cells", "smooth muscle cells")
        .title()
    )


def endpoint_label_from_composition(major: str, fine: str, major_fraction: float) -> str:
    fine_l = str(fine).lower()
    major_l = str(major).lower()
    if major_l == "epithelial" and major_fraction >= 0.5:
        if "stem" in fine_l:
            return "epithelial-rich / stem-like niche"
        if "colonocyte" in fine_l:
            return "epithelial-rich / colonocyte niche"
        if "ta" == fine_l or fine_l.startswith("ta"):
            return "epithelial-rich / transit-amplifying niche"
        if "repair" in fine_l or "clu" in fine_l:
            return "epithelial-rich / repair-associated niche"
        return "epithelial-rich niche"
    if major_l == "fibroblast" and major_fraction >= 0.35:
        return "fibroblast-rich stromal niche"
    if major_l == "immune" and major_fraction >= 0.5:
        if "neutrophil" in fine_l:
            return "immune-rich / neutrophil-associated niche"
        if "b cell" in fine_l or "plasma" in fine_l:
            return "immune-rich / B-cell-associated niche"
        if "macrophage" in fine_l or "monocyte" in fine_l:
            return "immune-rich / myeloid-associated niche"
        return "immune-rich niche"
    if "smooth muscle" in major_l and major_fraction >= 0.4:
        return "smooth-muscle/submucosa-associated niche"
    return "mixed/unresolved niche"


def phenotype_class(major: str, fine: str, major_fraction: float) -> str:
    label = endpoint_label_from_composition(major, fine, major_fraction)
    if label.startswith("epithelial-rich"):
        return "epithelial-rich"
    if label.startswith("fibroblast-rich"):
        return "fibroblast-rich"
    if "neutrophil" in label:
        return "neutrophil/lumen-associated"
    if label.startswith("immune-rich"):
        return "immune-rich"
    if label.startswith("smooth-muscle"):
        return "smooth-muscle/submucosa-associated"
    return "mixed/unresolved"


def classify_confidence(row: pd.Series) -> tuple[str, str]:
    if row["fraction_final_nodes"] < 0.03 or row["dominant_fate_node_fraction"] < 0.018:
        return (
            "low_size_or_low_mass_endpoint",
            "small final-time endpoint cluster or low dominant endpoint-attraction mass",
        )
    if row["slice_max_fraction"] >= 0.45 or row["mouse_max_fraction"] >= 0.8:
        return (
            "slice_or_mouse_associated_endpoint",
            "endpoint is disproportionately associated with one slice or mouse",
        )
    if row["dominant_major_fraction"] >= 0.65 and row["dominant_fine_fraction"] >= 0.25:
        return (
            "high_confidence_biological_endpoint",
            "large enough endpoint with coherent major and fine cell identity",
        )
    if row["dominant_major_fraction"] >= 0.35 and row["cell_type_l1_entropy"] <= 1.55:
        return (
            "plausible_but_mixed_endpoint",
            "dominant biological axis exists but endpoint remains compositionally mixed",
        )
    return (
        "unresolved_or_mixed_endpoint",
        "no single biological axis is strong enough for confident endpoint naming",
    )


def composition_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    counts = (
        df.groupby(["candidate_endpoint", "candidate_endpoint_label", label_col], dropna=False)
        .size()
        .rename("n_nodes")
        .reset_index()
    )
    totals_ep = counts.groupby("candidate_endpoint")["n_nodes"].transform("sum")
    totals_label = counts.groupby(label_col)["n_nodes"].transform("sum")
    counts["fraction_within_endpoint"] = counts["n_nodes"] / totals_ep
    counts["fraction_within_label"] = counts["n_nodes"] / totals_label
    return counts.sort_values(["candidate_endpoint", "n_nodes"], ascending=[True, False])


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: f"{value:.6g}")
        else:
            formatted[col] = formatted[col].astype(str)
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "/") for value in row) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator] + rows)


def load_m2_context() -> tuple[pd.DataFrame, dict[str, Any]]:
    files = sorted(M2_BY_SLICE.glob("*/m2_representation_*.parquet"))
    if not files:
        return pd.DataFrame(), {"files": 0, "rows": 0, "join_columns": []}
    schema_cols = pq.ParquetFile(files[0]).schema.names
    base_cols = ["slice_id", "anchor_index", "cell_type_l1", "cell_type_l2", "cell_type_l3"]
    l1_cols = [c for c in schema_cols if c.startswith("radius_x4__ct_l1__")]
    extra_cols = [
        c
        for c in [
            "radius_x4__ct_l1_entropy",
            "radius_x4__n_neighbors",
            "radius_x4__mean_neighbor_distance",
            "radius_x4__pseudo_local_density",
        ]
        if c in schema_cols
    ]
    columns = [c for c in base_cols + l1_cols + extra_cols if c in schema_cols]
    frames = [pd.read_parquet(p, columns=columns) for p in files]
    m2 = pd.concat(frames, ignore_index=True)
    if l1_cols:
        values = m2[l1_cols].to_numpy(dtype=float)
        best = values.argmax(axis=1)
        max_fraction = values[np.arange(values.shape[0]), best]
        labels = np.array([clean_feature_label(c) for c in l1_cols], dtype=object)
        m2["m2_local_dominant_l1"] = labels[best]
        m2["m2_local_dominant_l1_fraction"] = max_fraction
        m2["m2_local_context"] = np.where(
            max_fraction >= 0.40,
            "local_" + pd.Series(labels[best]).str.lower().str.replace(" ", "_").to_numpy() + "_rich",
            "mixed_local_niche",
        )
    else:
        m2["m2_local_dominant_l1"] = "unavailable"
        m2["m2_local_dominant_l1_fraction"] = np.nan
        m2["m2_local_context"] = "unavailable"
    meta = {"files": len(files), "rows": int(len(m2)), "join_columns": ["slice_id", "anchor_index"]}
    keep = [
        "slice_id",
        "anchor_index",
        "m2_local_dominant_l1",
        "m2_local_dominant_l1_fraction",
        "m2_local_context",
    ] + extra_cols
    return m2[keep], meta


def add_spatial_region(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for axis, labels in [("x_scaled_by_slice", ["left", "center", "right"]), ("y_scaled_by_slice", ["low", "middle", "high"])]:
        rank = result.groupby("slice_id")[axis].rank(method="average", pct=True)
        result[f"{axis}_bin"] = pd.cut(rank, bins=[0, 1 / 3, 2 / 3, 1.0], labels=labels, include_lowest=True)
    result["spatial_region_proxy"] = (
        result["x_scaled_by_slice_bin"].astype(str) + "_" + result["y_scaled_by_slice_bin"].astype(str)
    )
    return result


def write_metadata_audit(m2_meta: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    artifacts = {
        "M4B terminal assignments": M4B_TERMINAL,
        "M4C node summary": M4C_NODE,
        "M4A global node table": M4A_NODE,
        "M4D-00 coordinate cache": COORDS,
    }
    desired = [
        "global_node_index",
        "anchor_id",
        "slice_id",
        "anchor_index",
        "anchor_cell_id",
        "time",
        "time_day",
        "mouse_id",
        "cell_type_l1",
        "cell_type_l2",
        "cell_type_l3",
        "x_raw",
        "y_raw",
        "x_scaled_by_slice",
        "y_scaled_by_slice",
    ]
    for name, path in artifacts.items():
        info = parquet_info(path)
        cols = set(info["columns"])
        for field in desired:
            rows.append(
                {
                    "source": name,
                    "path": str(path),
                    "rows": info["rows"],
                    "field": field,
                    "available": field in cols,
                    "joinable": field in cols and field in {"global_node_index", "slice_id", "anchor_index", "anchor_cell_id"},
                    "notes": "Parquet schema inspection only",
                }
            )
        neighborhood_cols = [
            c
            for c in info["columns"]
            if any(token in c.lower() for token in ["cadinu", "moffitt", "neighborhood", "leiden_neigh", "cellular"])
        ]
        rows.append(
            {
                "source": name,
                "path": str(path),
                "rows": info["rows"],
                "field": "Cadinu/Moffitt neighborhood label",
                "available": bool(neighborhood_cols),
                "joinable": bool(neighborhood_cols),
                "notes": "; ".join(neighborhood_cols) if neighborhood_cols else "No neighborhood-label column in this table",
            }
        )
    raw_note = "not inspected"
    raw_joinable = False
    raw_available = False
    if M0_INSPECTION.exists():
        raw = json.loads(M0_INSPECTION.read_text())
        required = raw.get("adata.h5ad", {}).get("required_fields_present", {})
        day35_required = raw.get("adata_day35.h5ad", {}).get("required_fields_present", {})
        raw_available = bool(required.get("Leiden_neigh") or day35_required.get("Leiden_neigh"))
        raw_note = "Leiden_neigh present in raw h5ad inspection; not propagated to lightweight M2/M4 tables"
    rows.append(
        {
            "source": "M0 raw inspection report",
            "path": str(M0_INSPECTION),
            "rows": "",
            "field": "Leiden_neigh",
            "available": raw_available,
            "joinable": raw_joinable,
            "notes": raw_note,
        }
    )
    rows.append(
        {
            "source": "M2 by-slice representation",
            "path": str(M2_BY_SLICE),
            "rows": m2_meta.get("rows", 0),
            "field": "M2 local composition context",
            "available": m2_meta.get("rows", 0) > 0,
            "joinable": True,
            "notes": f"Loaded selected columns from {m2_meta.get('files', 0)} M2 slice files; join by slice_id + anchor_index",
        }
    )
    audit = pd.DataFrame(rows)
    audit.to_csv(REPORTS / "metadata_availability_audit.csv", index=False)
    (REPORTS / "metadata_availability_audit.md").write_text(
        "# M4E Metadata Availability Audit\n\n"
        "This audit used lightweight Parquet/CSV/JSON schemas and did not load raw M0 h5ad files.\n\n"
        "## Join Summary\n\n"
        "- Primary join key available: `global_node_index` across M4B, M4C, M4A, and M4D-00 coordinates.\n"
        "- Secondary validation keys available: `slice_id`, `anchor_index`, and `anchor_cell_id`.\n"
        "- Major/fine cell labels available: `cell_type_l1`, `cell_type_l2`, `cell_type_l3`.\n"
        "- Coordinates available from M4D-00: `x_raw`, `y_raw`, `x_scaled_by_slice`, `y_scaled_by_slice`.\n"
        "- M2 local composition context is joinable by `slice_id + anchor_index`.\n"
        "- Cadinu/Moffitt neighborhood labels are not present in joinable M2/M4 tables. "
        "The M0 raw inspection reports `Leiden_neigh` in h5ad, but it needs a future lightweight extraction table before use here.\n\n"
        "See `metadata_availability_audit.csv` for per-source fields.\n"
    )


def build_endpoint_summary(endpoint: pd.DataFrame, all_nodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_final = len(endpoint)
    dominant_counts = all_nodes.groupby("dominant_fate_label").size()
    dominant_prob = all_nodes.groupby("dominant_fate_label")["dominant_fate_probability"].mean()
    dominant_plasticity = all_nodes.groupby("dominant_fate_label")["normalized_plasticity_entropy"].mean()
    for endpoint_id, group in endpoint.groupby("candidate_endpoint"):
        label = group["candidate_endpoint_label"].iloc[0]
        major, major_frac, _ = top_label(group["major_cell_class"])
        fine, fine_frac, _ = top_label(group["fine_cell_cluster"])
        anchor, anchor_frac, _ = top_label(group["anchor_cell_type"])
        local_context, local_context_frac, _ = top_label(group["m2_local_context"])
        slice_label, slice_frac, _ = top_label(group["slice_id"])
        mouse_label, mouse_frac, _ = top_label(group["mouse_id"])
        tier_input = {
            "fraction_final_nodes": len(group) / total_final,
            "dominant_fate_node_fraction": dominant_counts.get(label, 0) / len(all_nodes),
            "slice_max_fraction": slice_frac,
            "mouse_max_fraction": mouse_frac,
            "dominant_major_fraction": major_frac,
            "dominant_fine_fraction": fine_frac,
            "cell_type_l1_entropy": entropy_from_counts(group["major_cell_class"].value_counts()),
        }
        tier, reason = classify_confidence(pd.Series(tier_input))
        rows.append(
            {
                "candidate_endpoint": int(endpoint_id),
                "candidate_endpoint_label": label,
                "n_nodes": int(len(group)),
                "fraction_final_nodes": float(len(group) / total_final),
                "dominant_major_cell_class": major,
                "dominant_major_fraction": major_frac,
                "cell_type_l1_entropy": tier_input["cell_type_l1_entropy"],
                "dominant_fine_cell_cluster": fine,
                "dominant_fine_fraction": fine_frac,
                "anchor_cell_type_top": anchor,
                "anchor_cell_type_top_fraction": anchor_frac,
                "dominant_m2_local_context": local_context,
                "dominant_m2_local_context_fraction": local_context_frac,
                "slice_max_group": slice_label,
                "slice_max_fraction": slice_frac,
                "mouse_max_group": mouse_label,
                "mouse_max_fraction": mouse_frac,
                "x_scaled_mean": float(group["x_scaled_by_slice"].mean()),
                "y_scaled_mean": float(group["y_scaled_by_slice"].mean()),
                "x_scaled_std": float(group["x_scaled_by_slice"].std()),
                "y_scaled_std": float(group["y_scaled_by_slice"].std()),
                "mean_m4c_top1_probability_final_nodes": float(group["dominant_fate_probability"].mean()),
                "mean_normalized_plasticity_final_nodes": float(group["normalized_plasticity_entropy"].mean()),
                "dominant_fate_node_count": int(dominant_counts.get(label, 0)),
                "dominant_fate_node_fraction": float(dominant_counts.get(label, 0) / len(all_nodes)),
                "mean_m4c_top1_probability_dominant_nodes": float(dominant_prob.get(label, np.nan)),
                "mean_normalized_plasticity_dominant_nodes": float(dominant_plasticity.get(label, np.nan)),
                "endpoint_biological_label": endpoint_label_from_composition(major, fine, major_frac),
                "endpoint_phenotype_class": phenotype_class(major, fine, major_frac),
                "biological_confidence_tier": tier,
                "confidence_reason": reason,
            }
        )
    summary = pd.DataFrame(rows).sort_values("candidate_endpoint")
    return summary


def write_composition_outputs(endpoint: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs = {
        "major": composition_table(endpoint, "major_cell_class"),
        "fine": composition_table(endpoint, "fine_cell_cluster"),
        "anchor": composition_table(endpoint, "anchor_cell_type"),
    }
    outputs["major"].to_csv(OUT / "endpoint_by_major_cell_class.csv", index=False)
    outputs["fine"].to_csv(OUT / "endpoint_by_fine_cell_cluster.csv", index=False)
    outputs["anchor"].to_csv(OUT / "endpoint_by_anchor_cell_type.csv", index=False)
    return outputs


def write_cadinu_absence_report() -> None:
    checked = [
        str(M4A_NODE),
        str(M4B_TERMINAL),
        str(M4C_NODE),
        str(COORDS),
        str(M2_BY_SLICE),
        str(M0_INSPECTION),
    ]
    placeholder = pd.DataFrame(
        [
            {
                "status": "neighborhood_labels_not_joinable",
                "reason": "No Cadinu/Moffitt/Leiden neighborhood label exists in lightweight M2/M4 tables.",
                "raw_note": "M0 raw inspection reports Leiden_neigh in h5ad; extract it to a lightweight keyed table before using it.",
            }
        ]
    )
    placeholder.to_csv(OUT / "endpoint_by_cadinu_neighborhood.csv", index=False)
    placeholder.to_csv(OUT / "endpoint_by_cadinu_neighborhood_normalized.csv", index=False)
    (REPORTS / "endpoint_neighborhood_overlap_report.md").write_text(
        "# Endpoint By Cadinu/Moffitt Neighborhood Overlap\n\n"
        "Status: not computed.\n\n"
        "No Cadinu/Moffitt cellular-neighborhood label was available in the joinable lightweight M2/M4 tables. "
        "The M0 raw AnnData inspection reports `Leiden_neigh` in raw h5ad files, but this label was not propagated "
        "to the M1/M2/M4A/M4B/M4C/M4D-00 Parquet artifacts used by this read-only review.\n\n"
        "Checked sources:\n\n"
        + "\n".join(f"- `{p}`" for p in checked)
        + "\n\nTo add this later, extract `Leiden_neigh` or the lab-approved Cadinu/Moffitt neighborhood label "
        "into a lightweight table keyed by `slice_id + anchor_index` or `global_node_index`, then rerun M4E.\n"
    )


def plasticity_tables(all_nodes: pd.DataFrame) -> None:
    def agg(group_cols: list[str]) -> pd.DataFrame:
        high = all_nodes["normalized_plasticity_entropy"] >= all_nodes["normalized_plasticity_entropy"].quantile(0.90)
        tmp = all_nodes.copy()
        tmp["high_plasticity"] = high
        return (
            tmp.groupby(group_cols, dropna=False)
            .agg(
                n_nodes=("global_node_index", "size"),
                mean_plasticity_entropy=("plasticity_entropy", "mean"),
                mean_normalized_plasticity=("normalized_plasticity_entropy", "mean"),
                mean_top1_probability=("dominant_fate_probability", "mean"),
                high_plasticity_fraction=("high_plasticity", "mean"),
            )
            .reset_index()
            .sort_values("mean_normalized_plasticity", ascending=False)
        )

    agg(["time", "time_day"]).to_csv(OUT / "plasticity_by_time.csv", index=False)
    agg(["major_cell_class"]).to_csv(OUT / "plasticity_by_major_cell_class.csv", index=False)
    agg(["fine_cell_cluster"]).to_csv(OUT / "plasticity_by_fine_cell_cluster.csv", index=False)
    agg(["spatial_region_proxy"]).to_csv(OUT / "plasticity_by_spatial_region_proxy.csv", index=False)
    pd.DataFrame(
        [{"status": "not_computed", "reason": "Cadinu/Moffitt neighborhood labels not joinable in lightweight artifacts"}]
    ).to_csv(OUT / "plasticity_by_cadinu_neighborhood.csv", index=False)


def niche_advantage(all_nodes: pd.DataFrame) -> pd.DataFrame:
    nonfinal = all_nodes[~all_nodes["is_final_time"].astype(bool)].copy()
    enough = nonfinal["major_cell_class"].value_counts()
    major_types = enough[enough >= 10000].index.tolist()
    rows = []
    for major in major_types:
        major_df = nonfinal[nonfinal["major_cell_class"] == major]
        for context, ctx_df in major_df.groupby("m2_local_context", dropna=False):
            if len(ctx_df) < 1000:
                continue
            counts = ctx_df["dominant_fate_label"].value_counts()
            dominant = counts.index[0]
            rows.append(
                {
                    "anchor_major_cell_class": major,
                    "niche_context": context,
                    "n_nodes": int(len(ctx_df)),
                    "dominant_candidate_endpoint": dominant,
                    "dominant_endpoint_fraction": float(counts.iloc[0] / counts.sum()),
                    "endpoint_entropy": entropy_from_counts(counts),
                    "endpoint_distribution_top5": "; ".join(
                        f"{idx}:{val / counts.sum():.3f}" for idx, val in counts.head(5).items()
                    ),
                }
            )
    result = pd.DataFrame(rows).sort_values(["anchor_major_cell_class", "n_nodes"], ascending=[True, False])
    result.to_csv(OUT / "niche_advantage_same_anchor_celltype_analysis.csv", index=False)
    return result


def plot_outputs(endpoint: pd.DataFrame, comps: dict[str, pd.DataFrame], all_nodes: pd.DataFrame, niche: pd.DataFrame) -> None:
    major_pivot = comps["major"].pivot(index="candidate_endpoint_label", columns="major_cell_class", values="fraction_within_endpoint").fillna(0)
    major_pivot.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab20")
    plt.ylabel("fraction within candidate endpoint")
    plt.xlabel("candidate endpoint niche cluster")
    plt.tight_layout()
    plt.savefig(FIGURES / "endpoint_major_cell_class_stacked_barplot.png", dpi=180)
    plt.close()

    top_fine = comps["fine"].groupby("fine_cell_cluster")["n_nodes"].sum().sort_values(ascending=False).head(18).index
    fine_pivot = (
        comps["fine"][comps["fine"]["fine_cell_cluster"].isin(top_fine)]
        .pivot(index="candidate_endpoint_label", columns="fine_cell_cluster", values="fraction_within_endpoint")
        .fillna(0)
    )
    plt.figure(figsize=(14, 6))
    plt.imshow(fine_pivot.to_numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="fraction within endpoint")
    plt.yticks(range(len(fine_pivot.index)), fine_pivot.index)
    plt.xticks(range(len(fine_pivot.columns)), fine_pivot.columns, rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "endpoint_fine_cluster_top_label_heatmap.png", dpi=180)
    plt.close()

    top_slices = endpoint["slice_id"].value_counts().head(2).index.tolist()
    for slice_id in top_slices:
        s = endpoint[endpoint["slice_id"] == slice_id]
        if len(s) > 20000:
            s = s.sample(20000, random_state=1)
        categories = sorted(s["endpoint_phenotype_class"].unique())
        color_map = {cat: i for i, cat in enumerate(categories)}
        plt.figure(figsize=(7, 6))
        plt.scatter(
            s["x_scaled_by_slice"],
            s["y_scaled_by_slice"],
            c=s["endpoint_phenotype_class"].map(color_map),
            s=2,
            cmap="tab10",
            alpha=0.75,
        )
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", markersize=5, label=cat)
            for cat in categories
        ]
        plt.legend(handles=handles, loc="best", fontsize=7)
        plt.title(f"Candidate endpoint labels: {slice_id}")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(FIGURES / f"endpoint_spatial_map_{slice_id}.png", dpi=180)
        plt.close()

    plast = pd.read_csv(OUT / "plasticity_by_major_cell_class.csv").sort_values("mean_normalized_plasticity", ascending=False)
    plt.figure(figsize=(9, 4))
    plt.bar(plast["major_cell_class"], plast["mean_normalized_plasticity"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("mean normalized plasticity")
    plt.tight_layout()
    plt.savefig(FIGURES / "plasticity_by_major_cell_class.png", dpi=180)
    plt.close()

    if not niche.empty:
        top_major = niche.groupby("anchor_major_cell_class")["n_nodes"].sum().sort_values(ascending=False).head(4).index
        sub = niche[niche["anchor_major_cell_class"].isin(top_major)].copy()
        sub["context_short"] = sub["anchor_major_cell_class"] + " | " + sub["niche_context"].astype(str)
        plot_df = sub.pivot(index="context_short", columns="dominant_candidate_endpoint", values="dominant_endpoint_fraction").fillna(0)
        plot_df.plot(kind="bar", stacked=True, figsize=(12, 5), colormap="tab20")
        plt.ylabel("dominant endpoint fraction")
        plt.tight_layout()
        plt.savefig(FIGURES / "same_anchor_celltype_niche_context_fate_distribution.png", dpi=180)
        plt.close()


def write_reports(
    summary: pd.DataFrame,
    all_nodes: pd.DataFrame,
    niche: pd.DataFrame,
    inventory: pd.DataFrame,
) -> None:
    tier_counts = summary["biological_confidence_tier"].value_counts().rename_axis("biological_confidence_tier").reset_index(name="n_endpoints")
    tier_counts.to_csv(OUT / "m4e_endpoint_confidence_tiers.csv", index=False)
    tier_counts.to_csv(REPORTS / "m4e_endpoint_confidence_tiers.csv", index=False)
    interpretable = summary[summary["biological_confidence_tier"].isin(["high_confidence_biological_endpoint", "plausible_but_mixed_endpoint"])]
    unresolved = summary[summary["biological_confidence_tier"].eq("unresolved_or_mixed_endpoint")]
    low = summary[summary["biological_confidence_tier"].eq("low_size_or_low_mass_endpoint")]
    slice_assoc = summary[summary["biological_confidence_tier"].eq("slice_or_mouse_associated_endpoint")]

    endpoint_lines = []
    for _, row in summary.iterrows():
        endpoint_lines.append(
            f"- {row.candidate_endpoint_label}: {row.endpoint_biological_label}; "
            f"tier=`{row.biological_confidence_tier}`; dominant major={row.dominant_major_cell_class} "
            f"({row.dominant_major_fraction:.3f}); dominant fine={row.dominant_fine_cell_cluster} "
            f"({row.dominant_fine_fraction:.3f})."
        )
    (REPORTS / "m4e_endpoint_biological_annotation_report.md").write_text(
        "# M4E Endpoint Biological Annotation Report\n\n"
        "Terminology: M4B `terminal_macrostate` IDs are reported here as candidate endpoint niche clusters. "
        "D35 is the observed final time, not an absolute biological terminal time.\n\n"
        "## Summary\n\n"
        f"- candidate endpoint niche clusters reviewed: {len(summary)}\n"
        f"- high-confidence or plausible biological endpoints: {len(interpretable)}\n"
        f"- low-size/low-mass endpoints: {len(low)}\n"
        f"- slice/mouse-associated endpoints: {len(slice_assoc)}\n"
        f"- unresolved/mixed endpoints: {len(unresolved)}\n\n"
        "## Endpoint Calls\n\n"
        + "\n".join(endpoint_lines)
        + "\n\n## Naming Policy\n\n"
        "- These are candidate endpoint niche clusters, not proven terminal states.\n"
        "- M4C-v1 is baseline endpoint-attraction / fate propagation, not lineage-validated fate.\n"
        "- Endpoint names should remain composition-aware and conservative when mixed.\n"
    )

    plast_time = pd.read_csv(OUT / "plasticity_by_time.csv")
    plast_major = pd.read_csv(OUT / "plasticity_by_major_cell_class.csv").head(5)
    plast_region = pd.read_csv(OUT / "plasticity_by_spatial_region_proxy.csv").head(5)
    (REPORTS / "m4c_plasticity_biological_validation.md").write_text(
        "# M4C Plasticity Biological Validation\n\n"
        "This review uses existing M4C normalized plasticity and entropy. It does not recompute M4C.\n\n"
        "## Time Trend\n\n"
        + dataframe_to_markdown(plast_time)
        + "\n\n## Highest Mean Plasticity By Major Cell Class\n\n"
        + dataframe_to_markdown(plast_major)
        + "\n\n## Highest Mean Plasticity By Spatial Region Proxy\n\n"
        + dataframe_to_markdown(plast_region)
        + "\n\n## Interpretation\n\n"
        "Cadinu/Moffitt neighborhood labels are not currently joinable, so enrichment in transition-like, ulcer/inflammatory, "
        "or repair-associated neighborhoods cannot be directly tested in this pass. The available proxy suggests plasticity "
        "can be stratified by anchor cell class and within-slice spatial region, but anatomical boundary calls require a "
        "curated neighborhood or tissue-region label.\n"
    )

    examples = []
    for major, group in niche.groupby("anchor_major_cell_class"):
        if group["dominant_candidate_endpoint"].nunique() >= 2:
            top = group.sort_values("n_nodes", ascending=False).head(4)
            examples.append(f"- {major}: " + "; ".join(f"{r.niche_context}->{r.dominant_candidate_endpoint}" for r in top.itertuples()))
    (REPORTS / "niche_advantage_same_anchor_celltype_report.md").write_text(
        "# Same Anchor Cell Type Niche-Context Analysis\n\n"
        "CellRank/CoSpar-style interpretations often operate at cell-state or cell-type level. "
        "For NicheFate, a key interpretability claim is that spatial niche context explains additional endpoint-attraction "
        "variation beyond anchor cell identity.\n\n"
        "This first-pass analysis stratified non-final nodes by anchor major cell class and M2-derived local composition context. "
        "Within each stratum it summarized the dominant M4C-v1 candidate endpoint distribution.\n\n"
        "## Examples With Different Endpoint Tendencies Within The Same Anchor Cell Class\n\n"
        + ("\n".join(examples) if examples else "- No qualifying multi-context examples found under current thresholds.")
        + "\n\nThe companion CSV contains endpoint-distribution entropy and top endpoint fractions by anchor type and local context.\n"
    )

    merge_candidates = summary[
        (summary["biological_confidence_tier"].isin(["low_size_or_low_mass_endpoint", "unresolved_or_mixed_endpoint"]))
    ]
    (REPORTS / "m4e_m4c_interpretability_review.md").write_text(
        "# M4E M4C Baseline Interpretability Review\n\n"
        "M4C-v1 remains interpretable as a baseline endpoint-attraction / fate propagation result, conditional on "
        "`P_fate-v1` and M4B candidate endpoint labels. The pyGPCCA failure does not invalidate this baseline because "
        "M4C is a forward propagation analysis, while pyGPCCA was a separate macrostate-discovery attempt on `P_super`.\n\n"
        "## Endpoint Interpretability\n\n"
        f"- Interpretable or plausible candidate endpoints: {len(interpretable)} of {len(summary)}.\n"
        f"- Endpoints requiring low-size/low-mass caution: {len(low)}.\n"
        f"- Slice/mouse-associated endpoint candidates: {len(slice_assoc)}.\n"
        f"- Unresolved/mixed endpoint candidates: {len(unresolved)}.\n\n"
        "## Merge / Relabel Candidates\n\n"
        + (
            "\n".join(
                f"- {r.candidate_endpoint_label}: {r.biological_confidence_tier}; {r.endpoint_biological_label}"
                for r in merge_candidates.itertuples()
            )
            if not merge_candidates.empty
            else "- No endpoint is currently flagged for merge/relabel."
        )
        + "\n\n## Missing Evidence\n\n"
        "- Cadinu/Moffitt or `Leiden_neigh` neighborhood labels need a lightweight joinable table.\n"
        "- Processed DARLIN clone/barcode tables are still missing for lineage validation.\n"
        "- M3-v2/K_gpcca production work should wait until endpoint labels and neighborhood evidence are reviewed.\n"
    )

    (REPORTS / "m4e_next_step_recommendation.md").write_text(
        "# M4E Next Step Recommendation\n\n"
        "Recommended next implementation step: extract and validate a lightweight cellular-neighborhood annotation table "
        "from the existing raw `Leiden_neigh` / Cadinu-Moffitt source, keyed by `global_node_index` or by "
        "`slice_id + anchor_index`, then rerun M4E neighborhood overlap and plasticity enrichment.\n\n"
        "Do not start M3-v2, K_gpcca production work, BranchSBM / Branched NicheFlow, M5/regulator analysis, or barcode "
        "preprocessing until candidate endpoint labels have neighborhood support and unresolved endpoints have been reviewed.\n\n"
        "M4C-v1 should be kept as a baseline. Low-size/low-mass and unresolved/mixed endpoint candidates should be marked "
        "conservative in figures and reports rather than over-named.\n"
    )
    refreshed = []
    for path in sorted(list(OUT.glob("*")) + list(REPORTS.glob("*.md")) + list(REPORTS.glob("*.csv")) + list(FIGURES.glob("*.png"))):
        refreshed.append({"path": str(path), "bytes": path.stat().st_size})
    pd.DataFrame(refreshed).to_csv(REPORTS / "m4e_endpoint_annotation_inventory.csv", index=False)


def main() -> None:
    ensure_dirs()
    terminal = pd.read_parquet(M4B_TERMINAL)
    terminal = terminal.rename(
        columns={
            "terminal_macrostate_id": "candidate_endpoint",
            "terminal_macrostate_label": "candidate_endpoint_label",
            "cell_type_l1": "major_cell_class",
            "cell_type_l2": "anchor_cell_type",
            "cell_type_l3": "fine_cell_cluster",
        }
    )
    m4c_cols = [
        "global_node_index",
        "is_final_time",
        "dominant_fate",
        "dominant_fate_label",
        "dominant_fate_probability",
        "plasticity_entropy",
        "normalized_plasticity_entropy",
        "time",
        "time_day",
        "slice_id",
        "anchor_index",
        "mouse_id",
        "cell_type_l1",
        "cell_type_l2",
        "cell_type_l3",
    ]
    m4c = pd.read_parquet(M4C_NODE, columns=m4c_cols).rename(
        columns={
            "cell_type_l1": "major_cell_class",
            "cell_type_l2": "anchor_cell_type",
            "cell_type_l3": "fine_cell_cluster",
        }
    )
    coord_cols = ["global_node_index", "x_raw", "y_raw", "x_scaled_by_slice", "y_scaled_by_slice", "coordinate_join_status"]
    coords = pd.read_parquet(COORDS, columns=coord_cols)
    m2_context, m2_meta = load_m2_context()
    write_metadata_audit(m2_meta)

    all_nodes = m4c.merge(coords, on="global_node_index", how="left", validate="one_to_one")
    all_nodes = all_nodes.merge(m2_context, on=["slice_id", "anchor_index"], how="left", validate="many_to_one")
    all_nodes["m2_local_context"] = all_nodes["m2_local_context"].fillna("m2_context_unmatched")
    all_nodes = add_spatial_region(all_nodes)

    endpoint_metric_cols = [
        "global_node_index",
        "is_final_time",
        "dominant_fate",
        "dominant_fate_label",
        "dominant_fate_probability",
        "plasticity_entropy",
        "normalized_plasticity_entropy",
    ]
    endpoint = terminal.merge(m4c[endpoint_metric_cols], on="global_node_index", how="left", validate="one_to_one")
    endpoint = endpoint.merge(coords, on="global_node_index", how="left", validate="one_to_one")
    endpoint = endpoint.merge(m2_context, on=["slice_id", "anchor_index"], how="left", validate="many_to_one")
    endpoint["m2_local_context"] = endpoint["m2_local_context"].fillna("m2_context_unmatched")

    summary = build_endpoint_summary(endpoint, all_nodes)
    endpoint = endpoint.merge(
        summary[["candidate_endpoint", "endpoint_biological_label", "endpoint_phenotype_class", "biological_confidence_tier"]],
        on="candidate_endpoint",
        how="left",
    )
    endpoint.to_parquet(OUT / "endpoint_node_annotation.parquet", index=False)
    summary.to_csv(OUT / "endpoint_macrostate_annotation_summary.csv", index=False)
    (OUT / "endpoint_macrostate_annotation_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2)
    )

    comps = write_composition_outputs(endpoint)
    write_cadinu_absence_report()
    plasticity_tables(all_nodes)
    niche = niche_advantage(all_nodes)
    plot_outputs(endpoint, comps, all_nodes, niche)

    inventory_rows = []
    for path in sorted(list(OUT.glob("*")) + list(REPORTS.glob("*.md")) + list(REPORTS.glob("*.csv")) + list(FIGURES.glob("*.png"))):
        inventory_rows.append({"path": str(path), "bytes": path.stat().st_size})
    inventory = pd.DataFrame(inventory_rows)
    write_reports(summary, all_nodes, niche, inventory)
    print(json.dumps({"status": "ok", "outputs": len(inventory_rows), "tier_counts": summary["biological_confidence_tier"].value_counts().to_dict()}, indent=2))


if __name__ == "__main__":
    main()
