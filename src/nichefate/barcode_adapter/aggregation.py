from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .input_contract import (
    CELLBIN_SUMMARY_REQUIRED_FIELDS,
    EXPECTED_ASSAYS,
    GROUP_ASSIGNMENT_OPTIONAL_COLUMNS,
    GROUP_ASSIGNMENT_REQUIRED_COLUMNS,
    PRIMARY_JOIN_KEY,
)


def _feature_scope(frame: pd.DataFrame) -> pd.Series:
    return frame["assay"].astype(str) + "::" + frame["feature_id"].astype(str)


def compute_lineage_diversity_metrics(counts: Iterable[float]) -> dict[str, float]:
    values = np.asarray([float(value) for value in counts if float(value) > 0.0], dtype=float)
    total = float(values.sum())
    if total <= 0.0:
        return {"feature_entropy": 0.0, "simpson_diversity": 0.0}
    probabilities = values / total
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    simpson = float(1.0 - np.square(probabilities).sum())
    return {"feature_entropy": entropy, "simpson_diversity": simpson}


def _assay_balance(counts: Iterable[float]) -> float:
    values = np.asarray([float(value) for value in counts if float(value) > 0.0], dtype=float)
    if values.size <= 1:
        return 0.0
    probabilities = values / float(values.sum())
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy / math.log(values.size)


def _dominant_assay(frame: pd.DataFrame, assays: tuple[str, ...]) -> pd.Series:
    columns = [f"{assay}_total_count" for assay in assays]
    totals = frame[columns].copy()
    totals.columns = list(assays)
    max_values = totals.max(axis=1)
    dominant = totals.idxmax(axis=1)
    dominant[max_values <= 0] = ""
    return dominant


def _diversity_by_key(
    feature_counts: pd.DataFrame,
    key_cols: list[str],
    entropy_name: str,
) -> pd.DataFrame:
    rows = []
    for key, group in feature_counts.groupby(key_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        metrics = compute_lineage_diversity_metrics(group["count"].to_numpy())
        row = dict(zip(key_cols, key, strict=False))
        row[entropy_name] = metrics["feature_entropy"]
        row["simpson_diversity"] = metrics["simpson_diversity"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_cellbin_lineage_evidence(
    lineage_evidence: pd.DataFrame,
    cellbin_index: pd.DataFrame | None = None,
    assays: tuple[str, ...] = EXPECTED_ASSAYS,
) -> pd.DataFrame:
    """Summarize primary feature/clone evidence without allele expansion."""

    key_cols = list(PRIMARY_JOIN_KEY)
    evidence = lineage_evidence.copy()
    evidence["count"] = pd.to_numeric(evidence["count"], errors="raise")
    evidence = evidence.loc[evidence["count"] > 0].copy()
    evidence["assay_feature_id"] = _feature_scope(evidence)

    if cellbin_index is not None:
        base_cols = ["sample_id", "slice_id", "section_order", "cellbin_id", "x", "y"]
        base = cellbin_index[base_cols].drop_duplicates(key_cols).copy()
    else:
        base = (
            evidence[["sample_id", "slice_id", "section_order", "cellbin_id", "x", "y"]]
            .drop_duplicates(key_cols)
            .copy()
        )

    total = evidence.groupby(key_cols, as_index=False)["count"].sum()
    total = total.rename(columns={"count": "total_lineage_count"})
    detected_feature = (
        evidence.groupby(key_cols)["assay_feature_id"]
        .nunique()
        .reset_index(name="detected_feature_count")
    )
    detected_assay = (
        evidence.groupby(key_cols)["assay"].nunique().reset_index(name="detected_assay_count")
    )

    assay_total = evidence.pivot_table(
        index=key_cols,
        columns="assay",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    assay_total.columns.name = None
    for assay in assays:
        if assay not in assay_total.columns:
            assay_total[assay] = 0
    assay_total = assay_total[key_cols + list(assays)].rename(
        columns={assay: f"{assay}_total_count" for assay in assays}
    )

    assay_features = (
        evidence.groupby(key_cols + ["assay"])["assay_feature_id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    assay_features.columns.name = None
    for assay in assays:
        if assay not in assay_features.columns:
            assay_features[assay] = 0
    assay_features = assay_features[key_cols + list(assays)].rename(
        columns={assay: f"{assay}_detected_feature_count" for assay in assays}
    )

    feature_counts = (
        evidence.groupby(key_cols + ["assay", "feature_id"], as_index=False)["count"].sum()
    )
    feature_counts["assay_order"] = feature_counts["assay"].map(
        {assay: idx for idx, assay in enumerate(assays)}
    ).fillna(len(assays))
    dominant = feature_counts.sort_values(
        key_cols + ["count", "assay_order", "feature_id"],
        ascending=[True, True, True, False, True, True],
    ).drop_duplicates(key_cols)
    dominant = dominant[key_cols + ["assay", "feature_id", "count"]].rename(
        columns={
            "assay": "dominant_assay_from_feature",
            "feature_id": "dominant_feature_id",
            "count": "dominant_feature_count",
        }
    )

    diversity = _diversity_by_key(feature_counts, key_cols, "feature_entropy")

    summary = base.merge(total, on=key_cols, how="left")
    for frame in [detected_feature, detected_assay, assay_total, assay_features, dominant, diversity]:
        summary = summary.merge(frame, on=key_cols, how="left")

    count_cols = [
        "total_lineage_count",
        "detected_feature_count",
        "detected_assay_count",
        "dominant_feature_count",
        "feature_entropy",
        "simpson_diversity",
        *[f"{assay}_total_count" for assay in assays],
        *[f"{assay}_detected_feature_count" for assay in assays],
    ]
    for column in count_cols:
        if column not in summary.columns:
            summary[column] = 0
        summary[column] = summary[column].fillna(0)
    summary["dominant_assay"] = _dominant_assay(summary, assays)
    empty_dominant = summary["dominant_assay"].eq("")
    summary.loc[~empty_dominant, "dominant_assay"] = summary.loc[
        ~empty_dominant,
        "dominant_assay_from_feature",
    ].fillna(summary.loc[~empty_dominant, "dominant_assay"])
    summary["dominant_feature_id"] = summary["dominant_feature_id"].fillna("")
    summary["dominant_feature_fraction"] = np.where(
        summary["total_lineage_count"] > 0,
        summary["dominant_feature_count"] / summary["total_lineage_count"],
        0.0,
    )
    summary["evidence_present"] = summary["total_lineage_count"].gt(0).map(bool).astype(object)
    if "dominant_assay_from_feature" in summary.columns:
        summary = summary.drop(columns=["dominant_assay_from_feature"])
    return summary[list(CELLBIN_SUMMARY_REQUIRED_FIELDS)]


def _top_feature_string(feature_counts: pd.DataFrame, limit: int = 5) -> str:
    if feature_counts.empty:
        return ""
    ordered = feature_counts.sort_values(
        ["count", "assay", "feature_id"],
        ascending=[False, True, True],
    ).head(limit)
    return ";".join(
        f"{row.assay}::{row.feature_id}:{float(row.count):g}"
        for row in ordered.itertuples(index=False)
    )


def aggregate_lineage_to_groups(
    lineage_evidence: pd.DataFrame,
    group_assignment: pd.DataFrame,
    group_cols: list[str] | None = None,
    assays: tuple[str, ...] = EXPECTED_ASSAYS,
) -> pd.DataFrame:
    """Aggregate cellbin lineage evidence to arbitrary group assignments."""

    missing = [column for column in GROUP_ASSIGNMENT_REQUIRED_COLUMNS if column not in group_assignment.columns]
    if missing:
        raise ValueError(f"group_assignment is missing required columns: {', '.join(missing)}")
    if group_cols is None:
        group_cols = ["sample_id", "slice_id", "group_id"]
    missing_group_cols = [column for column in group_cols if column not in group_assignment.columns]
    if missing_group_cols:
        raise ValueError(f"group_assignment is missing group columns: {', '.join(missing_group_cols)}")

    key_cols = list(PRIMARY_JOIN_KEY)
    join_group_cols = list(dict.fromkeys(key_cols + group_cols))
    mapping = group_assignment.copy()
    duplicate_rows = int(mapping.duplicated(key_cols + ["group_id"]).sum())
    if duplicate_rows:
        raise ValueError(f"group_assignment has duplicate cellbin-to-group rows: {duplicate_rows}")

    evidence = lineage_evidence.copy()
    evidence["count"] = pd.to_numeric(evidence["count"], errors="raise")
    evidence = evidence.loc[evidence["count"] > 0].copy()
    evidence["assay_feature_id"] = _feature_scope(evidence)

    group_base = (
        mapping.groupby(group_cols, dropna=False)["cellbin_id"]
        .nunique()
        .reset_index(name="member_cellbin_count")
    )
    optional = [column for column in GROUP_ASSIGNMENT_OPTIONAL_COLUMNS if column in mapping.columns and column not in group_cols]
    for column in optional:
        values = (
            mapping.groupby(group_cols, dropna=False)[column]
            .agg(lambda series: ";".join(sorted(series.dropna().astype(str).unique())[:5]))
            .reset_index()
        )
        group_base = group_base.merge(values, on=group_cols, how="left")

    evidence_cellbins = evidence[key_cols].drop_duplicates().assign(_has_evidence=True)
    member_evidence = mapping[join_group_cols].drop_duplicates().merge(
        evidence_cellbins,
        on=key_cols,
        how="left",
    )
    member_evidence["_has_evidence"] = (
        member_evidence["_has_evidence"].where(member_evidence["_has_evidence"].notna(), False).astype(bool)
    )
    with_evidence = (
        member_evidence.groupby(group_cols, dropna=False)["_has_evidence"]
        .sum()
        .reset_index(name="n_cellbins_with_lineage_evidence")
    )

    overlap_non_key = [
        column for column in evidence.columns if column in mapping.columns and column not in key_cols
    ]
    evidence_for_join = evidence.drop(columns=overlap_non_key, errors="ignore")
    joined = mapping[join_group_cols].merge(evidence_for_join, on=key_cols, how="inner")
    if joined.empty:
        result = group_base.merge(with_evidence, on=group_cols, how="left")
        result["n_cellbins_with_lineage_evidence"] = result[
            "n_cellbins_with_lineage_evidence"
        ].fillna(0)
        return _empty_group_metrics(result, group_cols, assays)

    total = joined.groupby(group_cols, dropna=False)["count"].sum().reset_index(
        name="total_lineage_count"
    )
    detected_feature = (
        joined.groupby(group_cols, dropna=False)["assay_feature_id"]
        .nunique()
        .reset_index(name="detected_feature_count")
    )
    detected_assay = (
        joined.groupby(group_cols, dropna=False)["assay"]
        .nunique()
        .reset_index(name="detected_assay_count")
    )
    assay_total = joined.pivot_table(
        index=group_cols,
        columns="assay",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    assay_total.columns.name = None
    for assay in assays:
        if assay not in assay_total.columns:
            assay_total[assay] = 0
    assay_total = assay_total[group_cols + list(assays)].rename(
        columns={assay: f"{assay}_total_count" for assay in assays}
    )

    feature_counts = (
        joined.groupby(group_cols + ["assay", "feature_id"], dropna=False, as_index=False)[
            "count"
        ].sum()
    )
    feature_counts["assay_order"] = feature_counts["assay"].map(
        {assay: idx for idx, assay in enumerate(assays)}
    ).fillna(len(assays))
    dominant = feature_counts.sort_values(
        group_cols + ["count", "assay_order", "feature_id"],
        ascending=[True] * len(group_cols) + [False, True, True],
    ).drop_duplicates(group_cols)
    dominant = dominant[group_cols + ["assay", "feature_id", "count"]].rename(
        columns={
            "assay": "dominant_assay",
            "feature_id": "dominant_feature_id",
            "count": "dominant_feature_count",
        }
    )
    diversity = _diversity_by_key(feature_counts, group_cols, "lineage_entropy")
    top_features = (
        feature_counts.groupby(group_cols, dropna=False)
        .apply(_top_feature_string, include_groups=False)
        .reset_index(name="top_features")
    )

    result = group_base.merge(with_evidence, on=group_cols, how="left")
    for frame in [total, detected_feature, detected_assay, assay_total, dominant, diversity, top_features]:
        result = result.merge(frame, on=group_cols, how="left")
    return _finalize_group_metrics(result, assays)


def _empty_group_metrics(
    result: pd.DataFrame,
    group_cols: list[str],
    assays: tuple[str, ...],
) -> pd.DataFrame:
    result = result.copy()
    for column in [
        "total_lineage_count",
        "detected_feature_count",
        "detected_assay_count",
        "dominant_feature_count",
        "dominant_feature_fraction",
        "lineage_entropy",
        "simpson_diversity",
        "assay_balance",
        *[f"{assay}_total_count" for assay in assays],
    ]:
        result[column] = 0
    result["dominant_assay"] = ""
    result["dominant_feature_id"] = ""
    result["top_features"] = ""
    result["fraction_member_cellbins_with_lineage_evidence"] = 0.0
    result["fraction_member_cellbins_with_lineage"] = 0.0
    return result


def _finalize_group_metrics(result: pd.DataFrame, assays: tuple[str, ...]) -> pd.DataFrame:
    result = result.copy()
    numeric_cols = [
        "n_cellbins_with_lineage_evidence",
        "total_lineage_count",
        "detected_feature_count",
        "detected_assay_count",
        "dominant_feature_count",
        "lineage_entropy",
        "simpson_diversity",
        *[f"{assay}_total_count" for assay in assays],
    ]
    for column in numeric_cols:
        if column not in result.columns:
            result[column] = 0
        result[column] = result[column].fillna(0)
    result["fraction_member_cellbins_with_lineage_evidence"] = np.where(
        result["member_cellbin_count"] > 0,
        result["n_cellbins_with_lineage_evidence"] / result["member_cellbin_count"],
        0.0,
    )
    result["fraction_member_cellbins_with_lineage"] = result[
        "fraction_member_cellbins_with_lineage_evidence"
    ]
    result["dominant_assay"] = result.get("dominant_assay", "").fillna("")
    result["dominant_feature_id"] = result.get("dominant_feature_id", "").fillna("")
    result["dominant_feature_fraction"] = np.where(
        result["total_lineage_count"] > 0,
        result["dominant_feature_count"] / result["total_lineage_count"],
        0.0,
    )
    result["top_features"] = result.get("top_features", "").fillna("")
    result["assay_balance"] = result[[f"{assay}_total_count" for assay in assays]].apply(
        lambda row: _assay_balance(row.to_numpy()),
        axis=1,
    )
    return result
