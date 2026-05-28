from __future__ import annotations

import numpy as np
import pandas as pd

from .aggregation import aggregate_lineage_to_groups
from .input_contract import EXPECTED_ASSAYS, PRIMARY_JOIN_KEY


def _feature_scope(frame: pd.DataFrame) -> pd.Series:
    return frame["assay"].astype(str) + "::" + frame["feature_id"].astype(str)


def aggregate_group_lineage(
    lineage_evidence: pd.DataFrame,
    group_assignment: pd.DataFrame,
    assays: tuple[str, ...] = EXPECTED_ASSAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = ["sample_id", "slice_id", "section_order", "group_id", "group_type", "niche_id"]
    summary = aggregate_lineage_to_groups(lineage_evidence, group_assignment, group_cols=group_cols, assays=assays)
    summary = summary.rename(
        columns={
            "member_cellbin_count": "n_member_cellbins",
            "n_cellbins_with_lineage_evidence": "n_member_cellbins_with_lineage",
            "lineage_entropy": "feature_entropy",
        }
    )
    if "fraction_member_cellbins_with_lineage" not in summary:
        summary["fraction_member_cellbins_with_lineage"] = summary[
            "fraction_member_cellbins_with_lineage_evidence"
        ]
    summary["evidence_present"] = summary["total_lineage_count"].gt(0).map(bool).astype(object)
    summary["local_context_not_tissue_abundance"] = True

    join_cols = list(PRIMARY_JOIN_KEY)
    evidence = lineage_evidence.copy()
    evidence["count"] = pd.to_numeric(evidence["count"], errors="raise")
    overlap_non_key = [
        column for column in evidence.columns if column in group_assignment.columns and column not in join_cols
    ]
    evidence_for_join = evidence.drop(columns=overlap_non_key, errors="ignore")
    joined = group_assignment.merge(evidence_for_join, on=join_cols, how="inner")
    joined["assay_feature_id"] = _feature_scope(joined)
    observed_assay_summary = (
        joined.groupby(group_cols + ["assay"], as_index=False)
        .agg(
            assay_total_count=("count", "sum"),
            assay_detected_feature_count=("assay_feature_id", "nunique"),
            assay_positive_member_cellbins=("cellbin_id", "nunique"),
        )
        .sort_values(group_cols + ["assay"])
    )
    group_base = group_assignment[group_cols].drop_duplicates()
    assay_base = pd.DataFrame({"assay": list(assays)})
    assay_summary = group_base.merge(assay_base, how="cross").merge(
        observed_assay_summary,
        on=group_cols + ["assay"],
        how="left",
    )
    for column in ["assay_total_count", "assay_detected_feature_count", "assay_positive_member_cellbins"]:
        assay_summary[column] = assay_summary[column].fillna(0).astype(int)
    assay_features = (
        assay_summary.pivot_table(
            index=group_cols,
            columns="assay",
            values="assay_detected_feature_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    assay_features.columns.name = None
    for assay in assays:
        if assay not in assay_features:
            assay_features[assay] = 0
    assay_features = assay_features[group_cols + list(assays)].rename(
        columns={assay: f"{assay}_detected_feature_count" for assay in assays}
    )
    summary = summary.merge(assay_features, on=group_cols, how="left")
    for assay in assays:
        col = f"{assay}_detected_feature_count"
        if col not in summary:
            summary[col] = 0
        summary[col] = summary[col].fillna(0).astype(int)

    feature_counts = (
        joined.groupby(group_cols + ["assay", "feature_id", "clone_id"], as_index=False)["count"].sum()
    )
    totals = summary[group_cols + ["total_lineage_count"]]
    feature_counts = feature_counts.merge(totals, on=group_cols, how="left")
    feature_counts["feature_fraction_in_group"] = np.where(
        feature_counts["total_lineage_count"] > 0,
        feature_counts["count"] / feature_counts["total_lineage_count"],
        0.0,
    )
    feature_counts = feature_counts.sort_values(
        group_cols + ["count", "assay", "feature_id"],
        ascending=[True] * len(group_cols) + [False, True, True],
    )
    feature_counts["feature_rank"] = feature_counts.groupby(group_cols).cumcount() + 1
    top_features = feature_counts.loc[feature_counts["feature_rank"] <= 10].copy()
    top_features = top_features.rename(columns={"count": "feature_count"})
    return summary, assay_summary, top_features


def group_lineage_coverage_metrics(cellbin_summary: pd.DataFrame, group_summary: pd.DataFrame) -> dict[str, float | int]:
    sampled = cellbin_summary.drop_duplicates(["sample_id", "slice_id", "cellbin_id"])
    lineage_positive = sampled["evidence_present"].astype(bool)
    lineage_fraction_column = (
        "fraction_member_cellbins_with_lineage"
        if "fraction_member_cellbins_with_lineage" in group_summary
        else "fraction_member_cellbins_with_lineage_evidence"
    )
    return {
        "sampled_cellbins": int(len(sampled)),
        "sampled_lineage_positive_cellbins": int(lineage_positive.sum()),
        "fraction_sampled_cellbins_with_lineage_evidence": float(lineage_positive.mean()) if len(sampled) else 0.0,
        "number_of_groups": int(len(group_summary)),
        "groups_with_ge1_lineage_positive_member": int((group_summary["n_member_cellbins_with_lineage"] >= 1).sum()),
        "groups_with_ge3_lineage_positive_members": int((group_summary["n_member_cellbins_with_lineage"] >= 3).sum()),
        "median_fraction_member_cellbins_with_lineage": float(group_summary[lineage_fraction_column].median()),
        "median_total_lineage_count_per_group": float(group_summary["total_lineage_count"].median()),
        "median_detected_feature_count_per_group": float(group_summary["detected_feature_count"].median()),
    }
