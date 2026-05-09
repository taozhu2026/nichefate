#!/usr/bin/env python
"""M4E-02 lightweight cellular-neighborhood extraction and endpoint review.

Reads only h5ad obs metadata in backed mode. It does not load expression X and
does not modify upstream M3/M4 production outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("/home/zhutao/scratch/nichefate")
DATA_ROOT = Path("/data/zhutao/nichefate")
PROJECT = Path("/home/zhutao/projects/nichefate")

OUT = ROOT / "m4e" / "neighborhood_annotation"
REPORTS = ROOT / "m4e" / "reports"
FIGURES = REPORTS / "figures" / "neighborhood_annotation"

BY_TIME_DIR = DATA_ROOT / "m0" / "intermediate" / "by_time"
BY_TIME_FILES = [BY_TIME_DIR / f"{time}.h5ad" for time in ["D0", "D3", "D9", "D21", "D35"]]
RAW_D35 = ROOT / "merfish_colitis_raw" / "adata_day35.h5ad"

M4A_NODE = ROOT / "m4a" / "node_table" / "global_node_table.parquet"
M4B_TERMINAL = ROOT / "m4b" / "terminal_states" / "terminal_macrostate_assignments.parquet"
M4C_NODE = ROOT / "m4c" / "fate_probabilities" / "fate_probability_node_summary.parquet"
COORDS = ROOT / "m4d" / "visualization_layer" / "node_coordinates.parquet"
M4E_ENDPOINT = ROOT / "m4e" / "endpoint_annotation" / "endpoint_node_annotation.parquet"
M4E_ENDPOINT_SUMMARY = ROOT / "m4e" / "endpoint_annotation" / "endpoint_macrostate_annotation_summary.csv"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def read_h5ad_obs(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    a = ad.read_h5ad(path, backed="r")
    try:
        obs = a.obs
        if columns is not None:
            selected = [c for c in columns if c in obs.columns]
            df = obs[selected].copy()
        else:
            df = obs.copy()
        df.insert(0, "raw_cell_id", obs.index.astype(str))
        return df.reset_index(drop=True)
    finally:
        a.file.close()


def inspect_h5ad(path: Path) -> dict[str, Any]:
    a = ad.read_h5ad(path, backed="r")
    try:
        cols = list(a.obs.columns)
        relevant = [
            c
            for c in cols
            if c
            in {
                "x",
                "y",
                "Mouse_ID",
                "mouse_id",
                "Sample_type",
                "Slice_ID",
                "slice_id",
                "Tier1",
                "Tier2",
                "Tier3",
                "cell_type_l1",
                "cell_type_l2",
                "cell_type_l3",
                "Leiden_neigh",
                "neighborhood_original",
                "time",
                "time_day",
            }
        ]
        neigh_cols = [
            c
            for c in cols
            if any(token in c.lower() for token in ["leiden", "neigh", "cadinu", "moffitt", "cellular"])
        ]
        return {
            "path": str(path),
            "n_obs": int(a.n_obs),
            "obs_columns": cols,
            "relevant_obs_columns": relevant,
            "neighborhood_columns": neigh_cols,
            "obs_name_sample": list(map(str, a.obs_names[:5])),
            "can_join_to_m4_nodes": "slice_id" in cols and bool(neigh_cols),
            "join_key_recommendation": "slice_id + raw_cell_id to M4 anchor_cell_id",
        }
    finally:
        a.file.close()


def write_source_audit() -> list[dict[str, Any]]:
    candidate_paths = [p for p in BY_TIME_FILES if p.exists()]
    if RAW_D35.exists():
        candidate_paths.append(RAW_D35)
    rows = []
    details = []
    for path in candidate_paths:
        info = inspect_h5ad(path)
        details.append(info)
        cols = set(info["obs_columns"])
        rows.append(
            {
                "file_path": str(path),
                "n_obs": info["n_obs"],
                "has_cell_id_obs_name": True,
                "has_slice_id": "slice_id" in cols or "Slice_ID" in cols,
                "has_time": "time" in cols or "Sample_type" in cols,
                "has_time_day": "time_day" in cols,
                "has_mouse_or_sample": "mouse_id" in cols or "Mouse_ID" in cols or "Sample_type" in cols,
                "has_cell_type_l1": "cell_type_l1" in cols or "Tier1" in cols,
                "has_cell_type_l2": "cell_type_l2" in cols or "Tier2" in cols,
                "has_cell_type_l3": "cell_type_l3" in cols or "Tier3" in cols,
                "has_leiden_neigh": "Leiden_neigh" in cols,
                "has_neighborhood_original": "neighborhood_original" in cols,
                "has_xy": "x" in cols and "y" in cols,
                "can_join_to_m4_nodes": info["can_join_to_m4_nodes"],
                "join_key": info["join_key_recommendation"],
                "obs_columns": ";".join(info["obs_columns"]),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(REPORTS / "neighborhood_metadata_source_audit.csv", index=False)
    (REPORTS / "neighborhood_metadata_source_audit.md").write_text(
        "# M4E-02 Neighborhood Metadata Source Audit\n\n"
        "H5AD files were inspected with `backed='r'`. Expression matrices and layers were not loaded.\n\n"
        "## Selected Source\n\n"
        "The selected source for extraction is the five time-split M0 h5ad files under "
        f"`{BY_TIME_DIR}`. They contain `Leiden_neigh`, `neighborhood_original`, `slice_id`, "
        "`time`, `time_day`, cell-type labels, coordinates, and obs names matching M4 `anchor_cell_id`.\n\n"
        "Primary join key: `slice_id + raw_cell_id` where `raw_cell_id` is h5ad `obs_names` and matches M4A `anchor_cell_id`.\n\n"
        "## Candidate Files\n\n"
        + "\n".join(
            f"- `{row['file_path']}`: n_obs={row['n_obs']}, Leiden={row['has_leiden_neigh']}, "
            f"neighborhood_original={row['has_neighborhood_original']}, joinable={row['can_join_to_m4_nodes']}"
            for row in rows
        )
        + "\n"
    )
    return details


def extract_raw_neighborhood_metadata() -> pd.DataFrame:
    columns = [
        "x",
        "y",
        "Mouse_ID",
        "mouse_id",
        "Sample_type",
        "Slice_ID",
        "slice_id",
        "Tier1",
        "Tier2",
        "Tier3",
        "cell_type_l1",
        "cell_type_l2",
        "cell_type_l3",
        "Leiden_neigh",
        "neighborhood_original",
        "time",
        "time_day",
    ]
    frames = []
    for path in BY_TIME_FILES:
        df = read_h5ad_obs(path, columns=columns)
        df["source_h5ad"] = str(path)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["slice_id"] = raw.get("slice_id", pd.Series(index=raw.index, dtype=object)).fillna(raw.get("Slice_ID"))
    raw["mouse_id"] = raw.get("mouse_id", pd.Series(index=raw.index, dtype=object)).fillna(raw.get("Mouse_ID"))
    raw["cell_type_l1"] = raw.get("cell_type_l1", pd.Series(index=raw.index, dtype=object)).fillna(raw.get("Tier1"))
    raw["cell_type_l2"] = raw.get("cell_type_l2", pd.Series(index=raw.index, dtype=object))
    if "Tier2" in raw.columns:
        raw["cell_type_l2"] = raw["cell_type_l2"].fillna(raw["Tier2"])
    raw["cell_type_l3"] = raw.get("cell_type_l3", pd.Series(index=raw.index, dtype=object)).fillna(raw.get("Tier3"))
    raw["leiden_neigh"] = raw["Leiden_neigh"].astype("string")
    raw["cadinu_neighborhood_label"] = raw.get("neighborhood_original", raw["Leiden_neigh"]).astype("string")
    raw["raw_time_label"] = raw.get("time", pd.Series(index=raw.index, dtype=object)).fillna(raw.get("Sample_type"))
    keep = [
        "raw_cell_id",
        "slice_id",
        "mouse_id",
        "raw_time_label",
        "time",
        "time_day",
        "cell_type_l1",
        "cell_type_l2",
        "cell_type_l3",
        "leiden_neigh",
        "cadinu_neighborhood_label",
        "x",
        "y",
        "source_h5ad",
    ]
    raw = raw[[c for c in keep if c in raw.columns]]
    raw.to_parquet(OUT / "raw_cell_neighborhood_metadata.parquet", index=False)
    schema = {
        "schema_version": "m4e_02_raw_cell_neighborhood_metadata_v1",
        "rows": int(len(raw)),
        "columns": list(raw.columns),
        "source_files": [str(p) for p in BY_TIME_FILES],
        "primary_join": "slice_id + raw_cell_id to M4 anchor_cell_id",
        "expression_matrix_loaded": False,
    }
    (OUT / "raw_cell_neighborhood_metadata_schema.json").write_text(json.dumps(schema, indent=2))
    return raw


def entropy(counts: pd.Series) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def build_node_join(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m4a_cols = [
        "global_node_index",
        "anchor_id",
        "slice_id",
        "anchor_index",
        "anchor_cell_id",
        "time",
        "time_day",
        "mouse_id",
        "cell_type_l1",
        "cell_type_l3",
        "is_final_time",
    ]
    m4a = pd.read_parquet(M4A_NODE, columns=m4a_cols)
    m4c = pd.read_parquet(
        M4C_NODE,
        columns=[
            "global_node_index",
            "dominant_fate",
            "dominant_fate_label",
            "dominant_fate_probability",
            "plasticity_entropy",
            "normalized_plasticity_entropy",
        ],
    )
    terminal = pd.read_parquet(
        M4B_TERMINAL,
        columns=["global_node_index", "terminal_macrostate_id", "terminal_macrostate_label"],
    ).rename(
        columns={
            "terminal_macrostate_id": "candidate_endpoint",
            "terminal_macrostate_label": "candidate_endpoint_label",
        }
    )
    endpoint_cols = ["global_node_index", "endpoint_biological_label", "endpoint_phenotype_class", "biological_confidence_tier"]
    endpoint_extra = pd.read_parquet(M4E_ENDPOINT, columns=endpoint_cols)

    raw_join = raw.rename(columns={"raw_cell_id": "anchor_cell_id"}).copy()
    dup_count = int(raw_join.duplicated(["slice_id", "anchor_cell_id"]).sum())
    raw_join = raw_join.drop_duplicates(["slice_id", "anchor_cell_id"], keep="first")
    joined = m4a.merge(
        raw_join,
        on=["slice_id", "anchor_cell_id"],
        how="left",
        suffixes=("_m4", "_raw"),
        validate="many_to_one",
    )
    joined["join_key_used"] = "slice_id+anchor_cell_id"
    joined["join_status"] = np.where(joined["leiden_neigh"].notna(), "matched", "missing_raw_neighborhood")
    joined = joined.merge(m4c, on="global_node_index", how="left", validate="one_to_one")
    joined = joined.merge(terminal, on="global_node_index", how="left", validate="one_to_one")
    joined = joined.merge(endpoint_extra, on="global_node_index", how="left", validate="one_to_one")
    if "time_m4" in joined.columns:
        joined = joined.rename(columns={"time_m4": "time_label"})
    elif "time" in joined.columns:
        joined = joined.rename(columns={"time": "time_label"})
    if "time_day_m4" not in joined.columns and "time_day" in joined.columns:
        joined = joined.rename(columns={"time_day": "time_day_m4"})
    if "mouse_id_m4" not in joined.columns and "mouse_id" in joined.columns:
        joined = joined.rename(columns={"mouse_id": "mouse_id_m4"})
    joined["cell_type_l1_mismatch"] = (
        joined["cell_type_l1_m4"].notna()
        & joined["cell_type_l1_raw"].notna()
        & (joined["cell_type_l1_m4"].astype(str) != joined["cell_type_l1_raw"].astype(str))
    )
    joined["cell_type_l3_mismatch"] = (
        joined["cell_type_l3_m4"].notna()
        & joined["cell_type_l3_raw"].notna()
        & (joined["cell_type_l3_m4"].astype(str) != joined["cell_type_l3_raw"].astype(str))
    )
    out_cols = [
        "global_node_index",
        "anchor_id",
        "slice_id",
        "anchor_index",
        "anchor_cell_id",
        "time_label",
        "time_day_m4",
        "mouse_id_m4",
        "is_final_time",
        "candidate_endpoint",
        "candidate_endpoint_label",
        "endpoint_biological_label",
        "endpoint_phenotype_class",
        "biological_confidence_tier",
        "dominant_fate",
        "dominant_fate_label",
        "dominant_fate_probability",
        "plasticity_entropy",
        "normalized_plasticity_entropy",
        "cell_type_l1_m4",
        "cell_type_l3_m4",
        "cell_type_l1_raw",
        "cell_type_l2",
        "cell_type_l3_raw",
        "leiden_neigh",
        "cadinu_neighborhood_label",
        "x",
        "y",
        "join_key_used",
        "join_status",
        "cell_type_l1_mismatch",
        "cell_type_l3_mismatch",
    ]
    joined = joined[[c for c in out_cols if c in joined.columns]]
    joined = joined.rename(
        columns={
            "time_day_m4": "time_day",
            "mouse_id_m4": "mouse_id",
            "cell_type_l1_m4": "cell_type_l1",
            "cell_type_l3_m4": "cell_type_l3",
            "cell_type_l1_raw": "raw_cell_type_l1",
            "cell_type_l3_raw": "raw_cell_type_l3",
        }
    )
    joined.to_parquet(OUT / "node_neighborhood_annotation.parquet", index=False)

    qc_rows = []
    total = len(joined)
    matched = int(joined["leiden_neigh"].notna().sum())
    d35 = joined[joined["is_final_time"].astype(bool)]
    for label, frame in [("all_nodes", joined), ("d35_endpoint_nodes", d35)]:
        qc_rows.append(
            {
                "scope": label,
                "expected_rows": 1439542 if label == "all_nodes" else 90960,
                "observed_rows": int(len(frame)),
                "matched_rows": int(frame["leiden_neigh"].notna().sum()),
                "missing_rows": int(frame["leiden_neigh"].isna().sum()),
                "missing_fraction": float(frame["leiden_neigh"].isna().mean()) if len(frame) else 0.0,
                "global_node_unique": bool(frame["global_node_index"].is_unique),
                "raw_duplicate_join_keys_before_dedup": dup_count,
                "cell_type_l1_mismatches": int(frame["cell_type_l1_mismatch"].sum()),
                "cell_type_l3_mismatches": int(frame["cell_type_l3_mismatch"].sum()),
            }
        )
    by_time = (
        joined.groupby(["time_label", "time_day"], dropna=False)
        .agg(
            observed_rows=("global_node_index", "size"),
            matched_rows=("leiden_neigh", lambda s: int(s.notna().sum())),
            missing_rows=("leiden_neigh", lambda s: int(s.isna().sum())),
            missing_fraction=("leiden_neigh", lambda s: float(s.isna().mean())),
        )
        .reset_index()
    )
    qc = pd.concat([pd.DataFrame(qc_rows), by_time.assign(scope="by_time")], ignore_index=True)
    qc.to_csv(OUT / "node_neighborhood_join_qc.csv", index=False)
    (OUT / "node_neighborhood_join_qc.md").write_text(
        "# Node Neighborhood Join QC\n\n"
        f"- join key used: `slice_id + anchor_cell_id`\n"
        f"- all nodes observed: {total}\n"
        f"- all nodes matched to Leiden/neighborhood label: {matched}\n"
        f"- all nodes missing label: {total - matched}\n"
        f"- D35 endpoint rows observed: {len(d35)}\n"
        f"- D35 endpoint rows matched: {int(d35['leiden_neigh'].notna().sum())}\n"
        f"- duplicate raw join keys before deduplication: {dup_count}\n"
        f"- global_node_index unique: {joined['global_node_index'].is_unique}\n"
        f"- cell_type_l1 mismatches: {int(joined['cell_type_l1_mismatch'].sum())}\n"
        f"- cell_type_l3 mismatches: {int(joined['cell_type_l3_mismatch'].sum())}\n\n"
        "See `node_neighborhood_join_qc.csv` for by-time missing rates.\n"
    )
    return joined, qc


def endpoint_overlap(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint = joined[joined["candidate_endpoint"].notna() & joined["leiden_neigh"].notna()].copy()
    endpoint["candidate_endpoint"] = endpoint["candidate_endpoint"].astype(int)
    counts = (
        endpoint.groupby(["candidate_endpoint", "candidate_endpoint_label", "leiden_neigh"], dropna=False)
        .size()
        .rename("n_nodes")
        .reset_index()
    )
    counts.to_csv(OUT / "endpoint_by_leiden_neigh_counts.csv", index=False)
    ep_totals = counts.groupby("candidate_endpoint")["n_nodes"].transform("sum")
    neigh_totals = counts.groupby("leiden_neigh")["n_nodes"].transform("sum")
    frac_ep = counts.copy()
    frac_ep["fraction_within_endpoint"] = frac_ep["n_nodes"] / ep_totals
    frac_ep.to_csv(OUT / "endpoint_by_leiden_neigh_fraction_by_endpoint.csv", index=False)
    frac_neigh = counts.copy()
    frac_neigh["fraction_within_neighborhood"] = frac_neigh["n_nodes"] / neigh_totals
    frac_neigh.to_csv(OUT / "endpoint_by_leiden_neigh_fraction_by_neighborhood.csv", index=False)

    total_nodes = len(endpoint)
    endpoint_total = counts.groupby("candidate_endpoint")["n_nodes"].sum()
    neighborhood_total = counts.groupby("leiden_neigh")["n_nodes"].sum()
    expected = []
    for row in counts.itertuples(index=False):
        exp = float(endpoint_total[row.candidate_endpoint] * neighborhood_total[row.leiden_neigh] / total_nodes)
        expected.append(exp)
    counts["expected_count_independence"] = expected
    counts["enrichment_observed_over_expected"] = counts["n_nodes"] / counts["expected_count_independence"].replace(0, np.nan)
    counts.to_csv(OUT / "endpoint_by_leiden_neigh_enrichment.csv", index=False)

    rows = []
    for endpoint_id, group in counts.groupby("candidate_endpoint"):
        label = group["candidate_endpoint_label"].iloc[0]
        top = group.sort_values("n_nodes", ascending=False).iloc[0]
        purity = float(top["n_nodes"] / group["n_nodes"].sum())
        rows.append(
            {
                "candidate_endpoint": int(endpoint_id),
                "candidate_endpoint_label": label,
                "dominant_leiden_neigh": top["leiden_neigh"],
                "dominant_neighborhood_fraction": purity,
                "neighborhood_entropy": entropy(group.set_index("leiden_neigh")["n_nodes"]),
                "n_neighborhoods_observed": int(group["leiden_neigh"].nunique()),
                "n_nodes": int(group["n_nodes"].sum()),
            }
        )
    purity = pd.DataFrame(rows).sort_values("candidate_endpoint")
    purity.to_csv(OUT / "endpoint_neighborhood_purity_entropy.csv", index=False)
    return counts, purity


def update_confidence_with_neighborhood(purity: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(M4E_ENDPOINT_SUMMARY)
    merged = base.merge(purity, on=["candidate_endpoint", "candidate_endpoint_label"], how="left")
    updated = []
    reasons = []
    for row in merged.itertuples(index=False):
        tier = row.biological_confidence_tier
        reason = row.confidence_reason
        purity_val = getattr(row, "dominant_neighborhood_fraction")
        if pd.notna(purity_val):
            if tier == "high_confidence_biological_endpoint" and purity_val >= 0.45:
                reason += f"; neighborhood support: dominant {row.dominant_leiden_neigh} ({purity_val:.3f})"
            elif tier == "unresolved_or_mixed_endpoint" and purity_val >= 0.65:
                tier = "plausible_but_mixed_endpoint"
                reason += f"; upgraded cautiously because one neighborhood dominates ({row.dominant_leiden_neigh}, {purity_val:.3f})"
            elif purity_val < 0.35 and tier == "high_confidence_biological_endpoint":
                tier = "plausible_but_mixed_endpoint"
                reason += f"; downgraded because neighborhood composition is diffuse ({purity_val:.3f})"
        updated.append(tier)
        reasons.append(reason)
    merged["biological_confidence_tier_m4e01"] = merged["biological_confidence_tier"]
    merged["biological_confidence_tier_with_neighborhood"] = updated
    merged["confidence_reason_with_neighborhood"] = reasons
    merged.to_csv(OUT / "m4e_endpoint_confidence_tiers_with_neighborhood.csv", index=False)
    merged.to_csv(REPORTS / "m4e_endpoint_confidence_tiers_with_neighborhood.csv", index=False)
    return merged


def plasticity_by_neighborhood(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = joined[joined["leiden_neigh"].notna()].copy()
    frame["is_high_plasticity_nonfinal"] = False
    nonfinal = frame[~frame["is_final_time"].astype(bool)]
    cutoff = nonfinal["normalized_plasticity_entropy"].quantile(0.90)
    frame.loc[~frame["is_final_time"].astype(bool), "is_high_plasticity_nonfinal"] = (
        frame.loc[~frame["is_final_time"].astype(bool), "normalized_plasticity_entropy"] >= cutoff
    )
    by_neigh = (
        frame.groupby("leiden_neigh", dropna=False)
        .agg(
            n_nodes=("global_node_index", "size"),
            nonfinal_nodes=("is_final_time", lambda s: int((~s.astype(bool)).sum())),
            mean_normalized_plasticity_all=("normalized_plasticity_entropy", "mean"),
            mean_normalized_plasticity_nonfinal=(
                "normalized_plasticity_entropy",
                lambda s: float(s[~frame.loc[s.index, "is_final_time"].astype(bool)].mean()),
            ),
            high_plasticity_fraction_nonfinal=("is_high_plasticity_nonfinal", "mean"),
            mean_top1_probability=("dominant_fate_probability", "mean"),
        )
        .reset_index()
        .sort_values("mean_normalized_plasticity_nonfinal", ascending=False)
    )
    by_time = (
        frame.groupby(["time_label", "time_day", "leiden_neigh"], dropna=False)
        .agg(
            n_nodes=("global_node_index", "size"),
            mean_normalized_plasticity=("normalized_plasticity_entropy", "mean"),
            mean_entropy=("plasticity_entropy", "mean"),
            mean_top1_probability=("dominant_fate_probability", "mean"),
        )
        .reset_index()
    )
    by_neigh.to_csv(OUT / "plasticity_by_leiden_neigh.csv", index=False)
    by_time.to_csv(OUT / "plasticity_by_time_and_leiden_neigh.csv", index=False)
    return by_neigh, by_time


def niche_advantage(joined: pd.DataFrame) -> pd.DataFrame:
    nonfinal = joined[(~joined["is_final_time"].astype(bool)) & joined["leiden_neigh"].notna()].copy()
    rows = []
    for strat_col, min_count in [("cell_type_l1", 10000), ("cell_type_l3", 5000)]:
        enough = nonfinal[strat_col].value_counts()
        for cell_type in enough[enough >= min_count].index:
            cell_df = nonfinal[nonfinal[strat_col] == cell_type]
            for neigh, group in cell_df.groupby("leiden_neigh"):
                if len(group) < 500:
                    continue
                dist = group["dominant_fate_label"].value_counts()
                rows.append(
                    {
                        "stratification": strat_col,
                        "cell_type": cell_type,
                        "leiden_neigh": neigh,
                        "n_nodes": int(len(group)),
                        "dominant_endpoint": dist.index[0],
                        "dominant_endpoint_fraction": float(dist.iloc[0] / dist.sum()),
                        "endpoint_entropy": entropy(dist),
                        "endpoint_distribution_top5": "; ".join(
                            f"{idx}:{val / dist.sum():.3f}" for idx, val in dist.head(5).items()
                        ),
                    }
                )
    result = pd.DataFrame(rows).sort_values(["stratification", "cell_type", "n_nodes"], ascending=[True, True, False])
    result.to_csv(OUT / "niche_advantage_same_celltype_by_neighborhood.csv", index=False)
    return result


def short_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    sub = df.head(max_rows).copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda value: f"{value:.4g}")
        else:
            sub[col] = sub[col].astype(str)
    header = "| " + " | ".join(sub.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(sub.columns)) + " |"
    rows = ["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in sub.itertuples(index=False, name=None)]
    return "\n".join([header, sep] + rows)


def write_reports(
    source_details: list[dict[str, Any]],
    qc: pd.DataFrame,
    counts: pd.DataFrame,
    purity: pd.DataFrame,
    tiers: pd.DataFrame,
    plasticity_neigh: pd.DataFrame,
    plasticity_time: pd.DataFrame,
    niche: pd.DataFrame,
) -> None:
    tier_counts = (
        tiers["biological_confidence_tier_with_neighborhood"]
        .value_counts()
        .rename_axis("tier")
        .reset_index(name="n_endpoints")
    )
    high = tiers[tiers["biological_confidence_tier_with_neighborhood"] == "high_confidence_biological_endpoint"]
    mixed = tiers[tiers["biological_confidence_tier_with_neighborhood"].isin(["unresolved_or_mixed_endpoint", "plausible_but_mixed_endpoint"])]
    low = tiers[tiers["biological_confidence_tier_with_neighborhood"] == "low_size_or_low_mass_endpoint"]
    source_rows = [
        {"path": d["path"], "n_obs": d["n_obs"], "neighborhood_columns": ",".join(d["neighborhood_columns"])}
        for d in source_details
    ]
    report_qc = qc[qc["scope"].isin(["all_nodes", "d35_endpoint_nodes"])]
    (REPORTS / "m4e_neighborhood_annotation_report.md").write_text(
        "# M4E-02 Neighborhood Annotation Report\n\n"
        "H5AD sources were opened with `backed='r'`; expression matrices were not loaded.\n\n"
        "## Metadata Source Used\n\n"
        + short_table(pd.DataFrame(source_rows), max_rows=8)
        + "\n\n## Join QC\n\n"
        + short_table(report_qc)
        + "\n\nPrimary join key used: `slice_id + anchor_cell_id`, where `anchor_cell_id` matches h5ad `obs_names`.\n"
    )
    endpoint_lines = []
    for row in tiers.sort_values("candidate_endpoint").itertuples(index=False):
        endpoint_lines.append(
            f"- {row.candidate_endpoint_label}: {row.endpoint_biological_label}; "
            f"M4E-01 tier=`{row.biological_confidence_tier_m4e01}`; "
            f"with-neighborhood tier=`{row.biological_confidence_tier_with_neighborhood}`; "
            f"dominant neighborhood={row.dominant_leiden_neigh} "
            f"({row.dominant_neighborhood_fraction:.3f})."
        )
    (REPORTS / "endpoint_neighborhood_overlap_report.md").write_text(
        "# Endpoint By Leiden/Cadinu-Moffitt Neighborhood Overlap\n\n"
        "Status: computed from M0 by-time h5ad `obs` metadata joined to M4 nodes by `slice_id + anchor_cell_id`.\n\n"
        "## Dominant Neighborhood And Purity\n\n"
        + short_table(purity.sort_values("candidate_endpoint"), max_rows=20)
        + "\n\n## Highest Endpoint-Neighborhood Enrichment\n\n"
        + short_table(
            counts.sort_values("enrichment_observed_over_expected", ascending=False)
            [
                [
                    "candidate_endpoint_label",
                    "leiden_neigh",
                    "n_nodes",
                    "expected_count_independence",
                    "enrichment_observed_over_expected",
                ]
            ],
            max_rows=20,
        )
        + "\n\nNeighborhood labels are used as source-provided categorical annotations. "
        "Endpoint names remain conservative where cell-type composition or endpoint size is mixed.\n"
    )
    (REPORTS / "m4e_endpoint_biological_annotation_with_neighborhood.md").write_text(
        "# Endpoint Biological Annotation With Neighborhood\n\n"
        "M4B terminal macrostate IDs are described as candidate endpoint niche clusters. "
        "D35 is the observed final time, not absolute biological terminal time.\n\n"
        "## Tier Counts\n\n"
        + short_table(tier_counts)
        + "\n\n## Endpoint Calls\n\n"
        + "\n".join(endpoint_lines)
        + "\n\nNo endpoint was force-named solely from neighborhood labels; mixed endpoints remain conservative.\n"
    )
    merge_candidates = tiers[
        tiers["biological_confidence_tier_with_neighborhood"].isin(
            ["low_size_or_low_mass_endpoint", "unresolved_or_mixed_endpoint", "plausible_but_mixed_endpoint"]
        )
    ]
    (REPORTS / "m4e_m4c_interpretability_review_with_neighborhood.md").write_text(
        "# M4C Interpretability Review With Neighborhood\n\n"
        "M4C-v1 remains interpretable as a frozen baseline endpoint-attraction / fate-propagation map. "
        "Neighborhood labels improve biological annotation but do not make M4C lineage-validated fate.\n\n"
        f"- high-confidence endpoints after neighborhood review: {len(high)}\n"
        f"- low-size/low-mass endpoints: {len(low)}\n"
        f"- mixed/plausible/unresolved endpoints needing caution: {len(mixed)}\n\n"
        "## Merge / Relabel Candidates\n\n"
        + "\n".join(
            f"- {r.candidate_endpoint_label}: {r.biological_confidence_tier_with_neighborhood}; "
            f"{r.endpoint_biological_label}; dominant neighborhood={r.dominant_leiden_neigh}"
            for r in merge_candidates.itertuples(index=False)
        )
        + "\n\nM4C-v1 should remain a baseline dynamic niche-fate map, with endpoint labels marked by confidence tier.\n"
    )
    high_plast = plasticity_neigh.sort_values("mean_normalized_plasticity_nonfinal", ascending=False).head(12)
    (REPORTS / "m4c_plasticity_by_neighborhood_report.md").write_text(
        "# M4C Plasticity By Neighborhood\n\n"
        "This report uses existing M4C normalized plasticity and entropy. It does not recompute M4C.\n\n"
        "D35 rows are endpoint-initialized one-hot rows, so D35 low plasticity is algorithmic and should not be interpreted "
        "as biological loss of plasticity.\n\n"
        "## Highest Non-final Mean Plasticity Neighborhoods\n\n"
        + short_table(high_plast)
        + "\n\nLabel names are Leiden/Cadinu-Moffitt neighborhood labels as available in M0 metadata. "
        "Ulcer, repair, or transition-like interpretation should be assigned only where the source label vocabulary supports it.\n"
    )
    examples = []
    for (strat, cell_type), group in niche.groupby(["stratification", "cell_type"]):
        if group["dominant_endpoint"].nunique() >= 2:
            top = group.sort_values("n_nodes", ascending=False).head(5)
            examples.append(
                f"- {strat}={cell_type}: "
                + "; ".join(f"{r.leiden_neigh}->{r.dominant_endpoint}" for r in top.itertuples(index=False))
            )
        if len(examples) >= 20:
            break
    (REPORTS / "niche_advantage_same_celltype_by_neighborhood_report.md").write_text(
        "# Same Cell-Type Neighborhood-Stratified Endpoint Analysis\n\n"
        "CellRank/CoSpar interpret fate primarily at cell-state/cell-type level. "
        "This analysis tests whether NicheFate adds information from spatial neighborhood context by comparing "
        "dominant endpoint distributions within the same cell type across `Leiden_neigh` labels.\n\n"
        "## Examples With Neighborhood-Dependent Endpoint Tendencies\n\n"
        + ("\n".join(examples) if examples else "- No qualifying examples found under current thresholds.")
        + "\n\nThe companion CSV contains endpoint entropy and top endpoint fractions by cell type and neighborhood.\n"
    )
    (REPORTS / "m4e_next_step_recommendation_after_neighborhood_annotation.md").write_text(
        "# M4E Next Step Recommendation After Neighborhood Annotation\n\n"
        "Recommended next engineering step: endpoint refinement, not M3-v2 or K_gpcca production yet. "
        "Review low-size/low-mass and mixed endpoint clusters against the neighborhood overlap, endpoint composition, "
        "and representative maps, then decide whether to merge or relabel conservative endpoints.\n\n"
        "M4C-v1 is sufficiently interpretable to keep as a baseline dynamic niche-fate map, but remains not lineage-validated. "
        "After endpoint refinement, the next algorithmic branch can be either M3-v2 design or a small K_gpcca pilot; "
        "production GPCCA should still wait.\n"
    )


def plot_figures(
    counts: pd.DataFrame,
    purity: pd.DataFrame,
    plasticity_neigh: pd.DataFrame,
    plasticity_time: pd.DataFrame,
    niche: pd.DataFrame,
    joined: pd.DataFrame,
) -> None:
    frac = pd.read_csv(OUT / "endpoint_by_leiden_neigh_fraction_by_endpoint.csv")
    top_neigh = counts.groupby("leiden_neigh")["n_nodes"].sum().sort_values(ascending=False).head(20).index
    heat = (
        frac[frac["leiden_neigh"].isin(top_neigh)]
        .pivot(index="candidate_endpoint_label", columns="leiden_neigh", values="fraction_within_endpoint")
        .fillna(0)
    )
    plt.figure(figsize=(12, 6))
    plt.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="fraction within endpoint")
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "endpoint_by_leiden_neigh_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(purity["candidate_endpoint_label"], purity["dominant_neighborhood_fraction"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("dominant neighborhood fraction")
    plt.tight_layout()
    plt.savefig(FIGURES / "endpoint_neighborhood_purity_barplot.png", dpi=180)
    plt.close()

    top_plast = plasticity_neigh.head(20)
    plt.figure(figsize=(12, 5))
    plt.bar(top_plast["leiden_neigh"], top_plast["mean_normalized_plasticity_nonfinal"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("mean normalized plasticity, non-final")
    plt.tight_layout()
    plt.savefig(FIGURES / "plasticity_by_leiden_neigh_barplot.png", dpi=180)
    plt.close()

    top_time_neigh = (
        plasticity_time.groupby("leiden_neigh")["n_nodes"].sum().sort_values(ascending=False).head(18).index
    )
    time_heat = (
        plasticity_time[plasticity_time["leiden_neigh"].isin(top_time_neigh)]
        .pivot_table(index="time_label", columns="leiden_neigh", values="mean_normalized_plasticity", aggfunc="mean")
        .reindex(["D0", "D3", "D9", "D21", "D35"])
        .fillna(0)
    )
    plt.figure(figsize=(12, 4))
    plt.imshow(time_heat.to_numpy(), aspect="auto", cmap="magma")
    plt.colorbar(label="mean normalized plasticity")
    plt.yticks(range(len(time_heat.index)), time_heat.index)
    plt.xticks(range(len(time_heat.columns)), time_heat.columns, rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "time_by_leiden_neigh_plasticity_heatmap.png", dpi=180)
    plt.close()

    if not niche.empty:
        selected = []
        for (strat, cell_type), group in niche.groupby(["stratification", "cell_type"]):
            if strat == "cell_type_l1" and group["dominant_endpoint"].nunique() >= 2:
                selected.append((strat, cell_type))
            if len(selected) >= 4:
                break
        sub = niche[niche.set_index(["stratification", "cell_type"]).index.isin(selected)]
        if not sub.empty:
            sub = sub.sort_values("n_nodes", ascending=False).head(24)
            labels = sub["cell_type"] + " | " + sub["leiden_neigh"].astype(str)
            plt.figure(figsize=(12, 5))
            plt.bar(labels, sub["dominant_endpoint_fraction"])
            plt.xticks(rotation=70, ha="right")
            plt.ylabel("dominant endpoint fraction")
            plt.tight_layout()
            plt.savefig(FIGURES / "same_celltype_neighborhood_endpoint_distribution.png", dpi=180)
            plt.close()

    d35 = joined[joined["is_final_time"].astype(bool) & joined["leiden_neigh"].notna()].copy()
    coords = pd.read_parquet(COORDS, columns=["global_node_index", "x_scaled_by_slice", "y_scaled_by_slice"])
    d35 = d35.merge(coords, on="global_node_index", how="left")
    for slice_id in d35["slice_id"].value_counts().head(2).index:
        s = d35[d35["slice_id"] == slice_id].copy()
        if len(s) > 15000:
            s = s.sample(15000, random_state=1)
        for col, name in [("candidate_endpoint_label", "endpoint"), ("leiden_neigh", "neighborhood")]:
            cats = sorted(s[col].astype(str).unique())
            cmap = {cat: i for i, cat in enumerate(cats)}
            plt.figure(figsize=(7, 6))
            plt.scatter(
                s["x_scaled_by_slice"],
                s["y_scaled_by_slice"],
                c=s[col].astype(str).map(cmap),
                s=2,
                alpha=0.75,
                cmap="tab20",
            )
            plt.title(f"{name}: {slice_id}")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(FIGURES / f"representative_tissue_map_{name}_{slice_id}.png", dpi=180)
            plt.close()


def write_inventory() -> None:
    paths = sorted(list(OUT.glob("*")) + list(REPORTS.glob("m4e_*neighborhood*.md")) + list(REPORTS.glob("*neighborhood*.csv")) + list(FIGURES.glob("*.png")))
    inventory = pd.DataFrame({"path": [str(p) for p in paths], "bytes": [p.stat().st_size for p in paths]})
    inventory.to_csv(REPORTS / "m4e_neighborhood_annotation_inventory.csv", index=False)


def main() -> None:
    ensure_dirs()
    source_details = write_source_audit()
    raw = extract_raw_neighborhood_metadata()
    joined, qc = build_node_join(raw)
    counts, purity = endpoint_overlap(joined)
    tiers = update_confidence_with_neighborhood(purity)
    plasticity_neigh, plasticity_time = plasticity_by_neighborhood(joined)
    niche = niche_advantage(joined)
    write_reports(source_details, qc, counts, purity, tiers, plasticity_neigh, plasticity_time, niche)
    plot_figures(counts, purity, plasticity_neigh, plasticity_time, niche, joined)
    write_inventory()
    d35_qc = qc[qc["scope"] == "d35_endpoint_nodes"].iloc[0].to_dict()
    print(
        json.dumps(
            {
                "status": "ok",
                "raw_rows": int(len(raw)),
                "node_rows": int(len(joined)),
                "d35_matched_rows": int(d35_qc["matched_rows"]),
                "d35_missing_rows": int(d35_qc["missing_rows"]),
                "tier_counts": tiers["biological_confidence_tier_with_neighborhood"].value_counts().to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
