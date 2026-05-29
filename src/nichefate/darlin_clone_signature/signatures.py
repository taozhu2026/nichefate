from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import CloneSignatureParams, compact_distribution, entropy_from_counts


def _support_for_features(
    evidence: pd.DataFrame,
    features: list[str],
    feature_groups: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if feature_groups is None:
        support = evidence.loc[evidence["assay_scoped_feature_id"].isin(features)].copy()
    else:
        parts = [feature_groups[feature] for feature in features if feature in feature_groups]
        support = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if support.empty:
        return pd.DataFrame()
    per_cell = (
        support.groupby(["cell_key", "sample_id", "slice_id", "section_order", "cellbin_id"], as_index=False)
        .agg(
            n_supporting_features=("assay_scoped_feature_id", "nunique"),
            n_supporting_loci=("assay", "nunique"),
            total_support_count=("count", "sum"),
        )
    )
    return per_cell


def _signature_row(
    *,
    clone_id: str,
    clone_class: str,
    features: list[str],
    feature_reference: pd.DataFrame,
    evidence: pd.DataFrame,
    validation_status: str,
    rejection_reason: str = "",
    feature_groups: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    ref = feature_reference.set_index("assay_scoped_feature_id").loc[features].reset_index()
    if feature_groups is None:
        raw_support = evidence.loc[evidence["assay_scoped_feature_id"].isin(features)].copy()
    else:
        parts = [feature_groups[feature] for feature in features if feature in feature_groups]
        raw_support = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    support = _support_for_features(evidence, features, feature_groups)
    if clone_class == "cross_locus_clone":
        support_used = support.loc[support["n_supporting_loci"].ge(2) & support["n_supporting_features"].ge(2)]
    elif clone_class == "multi_feature_single_locus_clone":
        support_used = support.loc[support["n_supporting_features"].ge(2)]
    else:
        support_used = support
    loci = sorted(ref["assay"].astype(str).unique())
    total_support = float(support_used["total_support_count"].sum()) if not support_used.empty else 0.0
    counts = raw_support.groupby("assay_scoped_feature_id")["count"].sum() if not raw_support.empty else pd.Series(dtype=float)
    entropy = entropy_from_counts(counts.tolist())
    max_frac = float(ref["cellbin_fraction"].max()) if not ref.empty else 0.0
    bridge_dependency = 1.0 / max(int(support_used["cell_key"].nunique()) if not support_used.empty else 0, 1)
    confidence = (
        np.log1p(max(int(support_used["cell_key"].nunique()) if not support_used.empty else 0, 0))
        + float(ref["empirical_rarity_weight"].median() if not ref.empty else 0.0)
        + len(loci) * 0.5
        - bridge_dependency
    )
    return {
        "clone_id": clone_id,
        "clone_class": clone_class,
        "clone_set_high_confidence": bool(clone_class in {"cross_locus_clone", "multi_feature_single_locus_clone"} and validation_status == "valid"),
        "clone_set_expanded": bool(validation_status == "valid"),
        "n_features": int(len(features)),
        "n_loci": int(len(loci)),
        "loci_present": ";".join(loci),
        "n_supporting_cellbins": int(support_used["cell_key"].nunique()) if not support_used.empty else 0,
        "total_support_count": total_support,
        "max_single_feature_fraction": max_frac,
        "signature_entropy": float(entropy),
        "bridge_dependency_score": float(bridge_dependency),
        "empirical_confidence_score": float(confidence),
        "validation_status": validation_status,
        "rejection_reason": rejection_reason,
        "section_distribution": compact_distribution(support_used, "section_order") if not support_used.empty else "",
        "feature_list": ";".join(features),
    }


def _id_for_class(clone_class: str, index: int) -> str:
    prefix = {
        "cross_locus_clone": "L126_clone_A",
        "single_locus_recurrent_clone": "L126_clone_B",
        "multi_feature_single_locus_clone": "L126_clone_C",
    }[clone_class]
    return f"{prefix}_{index:06d}"


def build_clone_signatures(
    evidence: pd.DataFrame,
    feature_reference: pd.DataFrame,
    edges: pd.DataFrame,
    candidate_components: pd.DataFrame,
    complexity: pd.DataFrame,
    params: CloneSignatureParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build class A/B/C validated clone signatures under the empirical contract."""

    valid_evidence = evidence.loc[evidence["valid_for_signature"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    filtered_rows: list[dict[str, Any]] = []
    counters = {
        "cross_locus_clone": 0,
        "single_locus_recurrent_clone": 0,
        "multi_feature_single_locus_clone": 0,
    }
    ref = feature_reference.set_index("assay_scoped_feature_id")
    feature_support_stats = []
    feature_cell_sets: dict[str, set[str]] = {}
    feature_groups: dict[str, pd.DataFrame] = {}
    keep_cols = ["cell_key", "sample_id", "slice_id", "section_order", "cellbin_id", "assay", "count", "assay_scoped_feature_id"]
    for feature, group in valid_evidence.groupby("assay_scoped_feature_id", sort=False):
        group = group[keep_cols].copy()
        feature_groups[str(feature)] = group
        cells = set(group["cell_key"].astype(str))
        feature_cell_sets[str(feature)] = cells
        feature_support_stats.append(
            {
                "assay_scoped_feature_id": str(feature),
                "feature_support_cellbins": int(len(cells)),
                "feature_total_support_count": float(group["count"].sum()),
                "feature_section_distribution": compact_distribution(group.drop_duplicates("cell_key"), "section_order"),
            }
        )
    feature_stats = pd.DataFrame(feature_support_stats).set_index("assay_scoped_feature_id") if feature_support_stats else pd.DataFrame()
    if not candidate_components.empty:
        for component in candidate_components.to_dict(orient="records"):
            features = [item for item in str(component["features"]).split(";") if item]
            loci = sorted({feature.split("::", 1)[0] for feature in features})
            if len(features) > params.max_signature_component_features:
                filtered_rows.append(
                    {
                        "candidate_id": component["component_id"],
                        "candidate_class": "filtered",
                        "n_features": int(len(features)),
                        "n_loci": int(len(loci)),
                        "loci_present": ";".join(loci),
                        "n_supporting_cellbins": 0,
                        "bridge_dependency_score": 1.0,
                        "validation_status": "filtered",
                        "rejection_reason": "overmerged_component_too_many_features",
                        "feature_list": ";".join(features[:50]),
                    }
                )
                continue
            if len(loci) >= 2:
                clone_class = "cross_locus_clone"
                support = _support_for_features(valid_evidence, features, feature_groups)
                support_count = int(support.loc[support["n_supporting_loci"].ge(2), "cell_key"].nunique()) if not support.empty else 0
                valid = support_count >= params.min_cross_locus_support_cellbins
            else:
                clone_class = "multi_feature_single_locus_clone"
                support = _support_for_features(valid_evidence, features, feature_groups)
                support_count = int(support.loc[support["n_supporting_features"].ge(2), "cell_key"].nunique()) if not support.empty else 0
                valid = len(features) >= 2 and support_count >= params.min_multifeature_support_cellbins
            bridge_dependency = 1.0 / max(support_count, 1)
            common = bool(ref.loc[features, "feature_class"].eq("common_filtered").any()) if features else True
            if common:
                valid = False
                reason = "contains_common_filtered_feature"
            elif bridge_dependency > params.max_bridge_dependency_score:
                valid = False
                reason = "bridge_driven_or_single_cell_support"
            elif not valid:
                reason = "insufficient_repeated_compatible_support"
            else:
                reason = ""
            if valid:
                counters[clone_class] += 1
                clone_id = _id_for_class(clone_class, counters[clone_class])
                rows.append(
                    _signature_row(
                        clone_id=clone_id,
                        clone_class=clone_class,
                        features=features,
                        feature_reference=feature_reference,
                        evidence=valid_evidence,
                        validation_status="valid",
                        feature_groups=feature_groups,
                    )
                )
            else:
                filtered_rows.append(
                    {
                        "candidate_id": component["component_id"],
                        "candidate_class": clone_class,
                        "n_features": int(len(features)),
                        "n_loci": int(len(loci)),
                        "loci_present": ";".join(loci),
                        "n_supporting_cellbins": int(support_count),
                        "bridge_dependency_score": float(bridge_dependency),
                        "validation_status": "filtered",
                        "rejection_reason": reason,
                        "feature_list": ";".join(features),
                    }
                )

    bridge_keys = set(complexity.loc[complexity["bridge_candidate"], "cell_key"].astype(str)) if not complexity.empty else set()
    single_ref = feature_reference.loc[
        feature_reference["valid_for_signature"]
        & feature_reference["feature_class"].isin(["rare", "low_frequency"])
        & feature_reference["n_cellbins_detected"].ge(params.min_single_feature_cellbins)
    ].copy()
    single_ref = single_ref.sort_values(["n_cellbins_detected", "empirical_rarity_weight", "assay_scoped_feature_id"], ascending=[False, False, True])
    for feature_row in single_ref.to_dict(orient="records"):
        feature = str(feature_row["assay_scoped_feature_id"])
        feature_cells = feature_cell_sets.get(feature, set())
        if not feature_cells:
            continue
        bridge_fraction = len(feature_cells & bridge_keys) / max(len(feature_cells), 1)
        if bridge_fraction >= 0.5:
            filtered_rows.append(
                {
                    "candidate_id": feature,
                    "candidate_class": "single_locus_recurrent_clone",
                    "n_features": 1,
                    "n_loci": 1,
                    "loci_present": str(feature_row["assay"]),
                    "n_supporting_cellbins": int(len(feature_cells)),
                    "bridge_dependency_score": float(bridge_fraction),
                    "validation_status": "filtered",
                    "rejection_reason": "bridge_driven_single_feature",
                    "feature_list": feature,
                }
            )
            continue
        counters["single_locus_recurrent_clone"] += 1
        clone_id = _id_for_class("single_locus_recurrent_clone", counters["single_locus_recurrent_clone"])
        stats = feature_stats.loc[feature] if not feature_stats.empty and feature in feature_stats.index else None
        support_n = int(stats["feature_support_cellbins"]) if stats is not None else int(feature_row["n_cellbins_detected"])
        total_support = float(stats["feature_total_support_count"]) if stats is not None else float(feature_row["total_count"])
        bridge_dependency = 1.0 / max(support_n, 1)
        confidence = np.log1p(support_n) + float(feature_row["empirical_rarity_weight"]) + 0.5 - bridge_dependency
        rows.append(
            {
                "clone_id": clone_id,
                "clone_class": "single_locus_recurrent_clone",
                "clone_set_high_confidence": False,
                "clone_set_expanded": True,
                "n_features": 1,
                "n_loci": 1,
                "loci_present": str(feature_row["assay"]),
                "n_supporting_cellbins": support_n,
                "total_support_count": total_support,
                "max_single_feature_fraction": float(feature_row["cellbin_fraction"]),
                "signature_entropy": 0.0,
                "bridge_dependency_score": float(bridge_dependency),
                "empirical_confidence_score": float(confidence),
                "validation_status": "valid",
                "rejection_reason": "",
                "section_distribution": str(stats["feature_section_distribution"]) if stats is not None else "",
                "feature_list": feature,
            }
        )

    signatures = pd.DataFrame(rows)
    if signatures.empty:
        signatures = pd.DataFrame(
            columns=[
                "clone_id",
                "clone_class",
                "clone_set_high_confidence",
                "clone_set_expanded",
                "n_features",
                "n_loci",
                "loci_present",
                "n_supporting_cellbins",
                "total_support_count",
                "max_single_feature_fraction",
                "signature_entropy",
                "bridge_dependency_score",
                "empirical_confidence_score",
                "validation_status",
                "rejection_reason",
                "section_distribution",
                "feature_list",
            ]
        )
    else:
        signatures = signatures.sort_values(
            ["clone_class", "n_supporting_cellbins", "empirical_confidence_score", "clone_id"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
    membership_rows = []
    for signature in signatures.to_dict(orient="records"):
        for feature in str(signature["feature_list"]).split(";"):
            if not feature:
                continue
            assay = feature.split("::", 1)[0]
            membership_rows.append(
                {
                    "clone_id": signature["clone_id"],
                    "clone_class": signature["clone_class"],
                    "assay_scoped_feature_id": feature,
                    "assay": assay,
                    "feature_role": "signature_feature",
                    "clone_set_high_confidence": bool(signature["clone_set_high_confidence"]),
                    "clone_set_expanded": bool(signature["clone_set_expanded"]),
                }
            )
    membership = pd.DataFrame(membership_rows)
    filtered = pd.DataFrame(filtered_rows)
    payload = {
        "n_validated_clones": int(len(signatures)),
        "n_high_confidence_clones": int(signatures["clone_set_high_confidence"].sum()) if not signatures.empty else 0,
        "n_expanded_clones": int(signatures["clone_set_expanded"].sum()) if not signatures.empty else 0,
        "clone_class_counts": signatures["clone_class"].value_counts().to_dict() if not signatures.empty else {},
        "n_filtered_candidate_signatures": int(len(filtered)),
        "single_feature_clone_class_kept_separate": True,
    }
    return signatures, membership, filtered, payload
