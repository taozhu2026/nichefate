from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import (
    CELL_COLUMNS,
    CloneSignatureParams,
    assay_scoped_feature,
    compact_distribution,
    entropy_from_counts,
    make_cell_key,
)


def _allele_metadata(allele: pd.DataFrame) -> pd.DataFrame:
    if allele.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "slice_id",
                "section_order",
                "assay",
                "feature_id",
                "allele",
                "allele_is_missing",
                "n_alleles_for_feature",
            ]
        )
    work = allele.copy()
    work["allele"] = work.get("allele", "").fillna("").astype(str)
    if "allele_is_missing" in work:
        missing = work["allele_is_missing"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        missing = work["allele"].eq("")
    work["allele_is_missing_bool"] = missing
    grouped = (
        work.groupby(["sample_id", "slice_id", "section_order", "assay", "feature_id"], as_index=False)
        .agg(
            allele=("allele", lambda s: ";".join(sorted({v for v in s.astype(str) if v and v.lower() != "nan"})[:3])),
            allele_is_missing=("allele_is_missing_bool", "all"),
            n_alleles_for_feature=("n_alleles_for_feature", "max"),
        )
    )
    grouped["allele"] = grouped["allele"].fillna("")
    grouped["allele_is_missing"] = grouped["allele_is_missing"].astype(bool)
    return grouped


def _feature_class(cellbin_fraction: float, params: CloneSignatureParams, allele_unusable: bool = False) -> str:
    if allele_unusable:
        return "missing_or_unusable"
    if cellbin_fraction <= params.rare_threshold:
        return "rare"
    if cellbin_fraction <= params.low_frequency_threshold:
        return "low_frequency"
    return "common_filtered"


def build_canonical_evidence(
    lineage: pd.DataFrame,
    allele: pd.DataFrame,
    full_cellbins: pd.DataFrame,
    params: CloneSignatureParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build complete cellbin-feature evidence, feature reference, and cellbin complexity."""

    required = {"sample_id", "slice_id", "section_order", "assay", "cellbin_id", "feature_id", "count"}
    missing = sorted(required - set(lineage.columns))
    if missing:
        raise ValueError(f"complete lineage evidence is missing required columns: {missing}")
    evidence = lineage.copy()
    evidence["count"] = pd.to_numeric(evidence["count"], errors="raise")
    evidence = evidence.loc[evidence["count"] > 0].copy()
    evidence["assay_scoped_feature_id"] = assay_scoped_feature(evidence)
    evidence["cell_key"] = make_cell_key(evidence)
    if "x" not in evidence:
        evidence["x"] = np.nan
    if "y" not in evidence:
        evidence["y"] = np.nan
    group_cols = [
        "sample_id",
        "slice_id",
        "section_order",
        "cellbin_id",
        "cell_key",
        "x",
        "y",
        "assay",
        "feature_id",
        "assay_scoped_feature_id",
    ]
    evidence = evidence.groupby(group_cols, dropna=False, as_index=False).agg(count=("count", "sum"))

    total_reference_cellbins = int(full_cellbins[CELL_COLUMNS].drop_duplicates().shape[0]) if not full_cellbins.empty else int(evidence["cell_key"].nunique())
    lineage_positive_cellbins = int(evidence["cell_key"].nunique())
    allele_meta = _allele_metadata(allele)
    evidence = evidence.merge(
        allele_meta,
        on=["sample_id", "slice_id", "section_order", "assay", "feature_id"],
        how="left",
    )
    evidence["allele"] = evidence["allele"].fillna("")
    evidence["allele_is_missing"] = evidence["allele_is_missing"].fillna(True).astype(bool)
    evidence["n_alleles_for_feature"] = evidence["n_alleles_for_feature"].fillna(0).astype(int)

    feature_ref = (
        evidence.groupby(["assay", "feature_id", "assay_scoped_feature_id"], as_index=False)
        .agg(
            n_cellbins_detected=("cell_key", "nunique"),
            total_count=("count", "sum"),
            allele=("allele", lambda s: ";".join(sorted({v for v in s.astype(str) if v})[:3])),
            allele_is_missing=("allele_is_missing", "all"),
        )
        .sort_values(["assay", "feature_id"])
        .reset_index(drop=True)
    )
    feature_ref["cellbin_fraction"] = feature_ref["n_cellbins_detected"] / max(total_reference_cellbins, 1)
    section_rows = []
    for feature, group in evidence.groupby("assay_scoped_feature_id", sort=False):
        section_rows.append(
            {
                "assay_scoped_feature_id": feature,
                "section_distribution": compact_distribution(group, "section_order", "cell_key"),
            }
        )
    feature_ref = feature_ref.merge(pd.DataFrame(section_rows), on="assay_scoped_feature_id", how="left")
    epsilon = 1.0 / max(total_reference_cellbins * 10.0, 1.0)
    feature_ref["empirical_rarity_weight"] = -np.log(feature_ref["cellbin_fraction"] + epsilon)
    feature_ref["feature_class"] = [
        _feature_class(float(value), params)
        for value in feature_ref["cellbin_fraction"].tolist()
    ]
    feature_ref["valid_for_signature"] = feature_ref["feature_class"].isin(["rare", "low_frequency"])
    evidence = evidence.merge(
        feature_ref[
            [
                "assay_scoped_feature_id",
                "n_cellbins_detected",
                "cellbin_fraction",
                "empirical_rarity_weight",
                "feature_class",
                "valid_for_signature",
            ]
        ],
        on="assay_scoped_feature_id",
        how="left",
    )

    valid = evidence.loc[evidence["valid_for_signature"]].copy()
    total_by_cell = evidence.groupby("cell_key")["count"].sum()
    max_by_cell = evidence.groupby("cell_key")["count"].max()
    complexity = (
        evidence.groupby(["sample_id", "slice_id", "section_order", "cellbin_id", "cell_key"], as_index=False)
        .agg(
            n_detected_features=("assay_scoped_feature_id", "nunique"),
            total_lineage_count=("count", "sum"),
        )
    )
    for assay in ["CA", "TA", "RA"]:
        assay_counts = (
            evidence.loc[evidence["assay"].eq(assay)]
            .groupby("cell_key")["assay_scoped_feature_id"]
            .nunique()
            .rename(f"n_detected_{assay}_features")
        )
        complexity = complexity.merge(assay_counts, on="cell_key", how="left")
    complexity[["n_detected_CA_features", "n_detected_TA_features", "n_detected_RA_features"]] = complexity[
        ["n_detected_CA_features", "n_detected_TA_features", "n_detected_RA_features"]
    ].fillna(0).astype(int)
    entropy_rows = (
        evidence.groupby("cell_key")["count"]
        .agg(barcode_feature_entropy=lambda s: entropy_from_counts(s.tolist()))
        .reset_index()
    )
    complexity = complexity.merge(entropy_rows, on="cell_key", how="left")
    complexity["max_feature_fraction"] = (
        complexity["cell_key"].map(max_by_cell) / complexity["cell_key"].map(total_by_cell).replace(0, np.nan)
    ).fillna(0.0)
    valid_counts = valid.groupby("cell_key")["assay_scoped_feature_id"].nunique().rename("n_valid_signature_features")
    valid_loci = valid.groupby("cell_key")["assay"].nunique().rename("n_valid_signature_loci")
    complexity = complexity.merge(valid_counts, on="cell_key", how="left").merge(valid_loci, on="cell_key", how="left")
    complexity[["n_valid_signature_features", "n_valid_signature_loci"]] = complexity[
        ["n_valid_signature_features", "n_valid_signature_loci"]
    ].fillna(0).astype(int)
    complexity["possible_bridge_score"] = (
        complexity["n_valid_signature_features"]
        * np.maximum(complexity["n_valid_signature_loci"], 1)
        * (1.0 - complexity["max_feature_fraction"].clip(0, 1))
    )
    for q, col in [(0.99, "bridge_flag_p99"), (0.995, "bridge_flag_p995")]:
        threshold = float(complexity["possible_bridge_score"].quantile(q)) if not complexity.empty else 0.0
        complexity[col] = complexity["possible_bridge_score"].gt(threshold) & complexity["n_valid_signature_features"].gt(1)
    complexity["bridge_candidate"] = complexity["bridge_flag_p99"] | complexity["bridge_flag_p995"]

    per_cell_valid = complexity["n_valid_signature_features"]
    pair_events = int(((per_cell_valid * (per_cell_valid - 1)) // 2).sum())
    qc = {
        "complete_primary_evidence_rows": int(len(evidence)),
        "lineage_positive_cellbins": lineage_positive_cellbins,
        "total_reference_cellbins": total_reference_cellbins,
        "n_assay_scoped_features": int(feature_ref["assay_scoped_feature_id"].nunique()),
        "n_valid_signature_features": int(feature_ref["valid_for_signature"].sum()),
        "valid_features_per_cellbin_quantiles": {
            str(q): float(per_cell_valid.quantile(q)) if len(per_cell_valid) else 0.0
            for q in [0, 0.5, 0.9, 0.95, 0.99, 0.995, 1.0]
        },
        "estimated_pair_event_count": pair_events,
        "resource_warning": pair_events > params.high_complexity_pair_warning,
        "top_feature_table_role": "qc_summary_only_not_primary_clone_evidence",
    }
    return evidence, feature_ref, complexity, qc
