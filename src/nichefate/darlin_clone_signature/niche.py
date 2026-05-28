from __future__ import annotations

import numpy as np
import pandas as pd

from .common import make_cell_key, entropy_from_counts, simpson_from_counts, summarize_top_items


def aggregate_clone_membership(
    mapping: pd.DataFrame,
    assignment: pd.DataFrame,
    membership: pd.DataFrame,
    unit_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mapping.empty:
        return pd.DataFrame(), pd.DataFrame()
    mapping = mapping.copy()
    mapping["cell_key"] = make_cell_key(mapping)
    assignment_small = assignment[["cell_key", "assignment_status"]].drop_duplicates("cell_key") if "cell_key" in assignment else pd.DataFrame()
    joined = mapping.merge(assignment_small, on="cell_key", how="left")
    joined["assignment_status"] = joined["assignment_status"].fillna("unassigned")
    base = joined.groupby(unit_cols, dropna=False).agg(n_cellbins=("cell_key", "nunique")).reset_index()
    status_counts = (
        joined.groupby([*unit_cols, "assignment_status"], dropna=False)
        .agg(n_status_cellbins=("cell_key", "nunique"))
        .reset_index()
    )
    status_pivot = status_counts.pivot_table(index=unit_cols, columns="assignment_status", values="n_status_cellbins", fill_value=0).reset_index()
    for status in ["assigned_single", "assigned_multi", "ambiguous", "unassigned", "filtered"]:
        if status not in status_pivot:
            status_pivot[status] = 0
    if membership.empty:
        comp = pd.DataFrame(columns=[*unit_cols, "clone_set", "clone_id", "clone_class", "clone_weight"])
        summary = base.merge(status_pivot, on=unit_cols, how="left")
        return _finish_summary(summary, comp, unit_cols)
    members = membership[["cell_key", "clone_set", "clone_id", "clone_class", "membership_weight"]].copy()
    comp_joined = mapping[unit_cols + ["cell_key"]].merge(members, on="cell_key", how="inner")
    comp = (
        comp_joined.groupby([*unit_cols, "clone_set", "clone_id", "clone_class"], dropna=False, as_index=False)
        .agg(clone_weight=("membership_weight", "sum"), n_member_cellbins=("cell_key", "nunique"))
        .sort_values([*unit_cols, "clone_weight"], ascending=[True] * len(unit_cols) + [False])
    )
    summary = base.merge(status_pivot, on=unit_cols, how="left")
    summary = _finish_summary(summary, comp, unit_cols)
    return comp, summary


def _finish_summary(summary: pd.DataFrame, comp: pd.DataFrame, unit_cols: list[str]) -> pd.DataFrame:
    for status in ["assigned_single", "assigned_multi", "ambiguous", "unassigned", "filtered"]:
        if status not in summary:
            summary[status] = 0
        summary[status] = summary[status].fillna(0).astype(int)
    summary["n_clone_assigned_cellbins"] = summary["assigned_single"] + summary["assigned_multi"]
    summary["fraction_clone_assigned"] = (summary["n_clone_assigned_cellbins"] / summary["n_cellbins"].replace(0, np.nan)).fillna(0.0)
    summary["ambiguous_cellbin_fraction"] = (summary["ambiguous"] / summary["n_cellbins"].replace(0, np.nan)).fillna(0.0)
    summary["unassigned_cellbin_fraction"] = (summary["unassigned"] / summary["n_cellbins"].replace(0, np.nan)).fillna(0.0)
    if comp.empty:
        summary["n_clones_detected"] = 0
        summary["n_class_A_clones"] = 0
        summary["n_class_B_clones"] = 0
        summary["n_class_C_clones"] = 0
        summary["dominant_clone_id"] = ""
        summary["dominant_clone_class"] = ""
        summary["dominant_clone_fraction"] = 0.0
        summary["clone_entropy"] = 0.0
        summary["simpson_clone_diversity"] = 0.0
        summary["clone_richness"] = 0
        summary["top_clones"] = ""
        return summary
    richness = comp.groupby(unit_cols, dropna=False).agg(n_clones_detected=("clone_id", "nunique")).reset_index()
    for clone_class, col in [
        ("cross_locus_clone", "n_class_A_clones"),
        ("single_locus_recurrent_clone", "n_class_B_clones"),
        ("multi_feature_single_locus_clone", "n_class_C_clones"),
    ]:
        class_counts = (
            comp.loc[comp["clone_class"].eq(clone_class)]
            .groupby(unit_cols, dropna=False)["clone_id"]
            .nunique()
            .rename(col)
            .reset_index()
        )
        richness = richness.merge(class_counts, on=unit_cols, how="left")
    dominant = comp.sort_values([*unit_cols, "clone_weight"], ascending=[True] * len(unit_cols) + [False]).drop_duplicates(unit_cols)
    dominant = dominant[[*unit_cols, "clone_id", "clone_class", "clone_weight"]].rename(
        columns={
            "clone_id": "dominant_clone_id",
            "clone_class": "dominant_clone_class",
            "clone_weight": "dominant_clone_weight",
        }
    )
    diversity = (
        comp.groupby(unit_cols, dropna=False)["clone_weight"]
        .agg(
            clone_entropy=lambda s: entropy_from_counts(s.tolist()),
            simpson_clone_diversity=lambda s: simpson_from_counts(s.tolist()),
        )
        .reset_index()
    )
    top = (
        comp.groupby(unit_cols, dropna=False)
        .apply(lambda g: summarize_top_items(g.set_index("clone_id")["clone_weight"].sort_values(ascending=False)), include_groups=False)
        .reset_index(name="top_clones")
    )
    summary = summary.merge(richness, on=unit_cols, how="left").merge(dominant, on=unit_cols, how="left").merge(diversity, on=unit_cols, how="left").merge(top, on=unit_cols, how="left")
    for col in ["n_clones_detected", "n_class_A_clones", "n_class_B_clones", "n_class_C_clones"]:
        summary[col] = summary[col].fillna(0).astype(int)
    summary["clone_richness"] = summary["n_clones_detected"]
    summary["dominant_clone_id"] = summary["dominant_clone_id"].fillna("")
    summary["dominant_clone_class"] = summary["dominant_clone_class"].fillna("")
    summary["dominant_clone_weight"] = summary["dominant_clone_weight"].fillna(0.0)
    summary["dominant_clone_fraction"] = (summary["dominant_clone_weight"] / summary["n_clone_assigned_cellbins"].replace(0, np.nan)).fillna(0.0)
    summary["clone_entropy"] = summary["clone_entropy"].fillna(0.0)
    summary["simpson_clone_diversity"] = summary["simpson_clone_diversity"].fillna(0.0)
    summary["top_clones"] = summary["top_clones"].fillna("")
    return summary
