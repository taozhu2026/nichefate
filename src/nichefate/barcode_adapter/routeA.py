from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from .aggregation import compute_lineage_diversity_metrics
from .input_contract import EXPECTED_ASSAYS, PRIMARY_JOIN_KEY


UNIT_LABEL = "L126_BOUNDED_PLANA_STYLE_UNITS_NOT_FULL_M1_M2"
REPRESENTATION_LABEL = "L126_BOUNDED_EXPRESSION_SPATIAL_REPRESENTATION_NOT_FULL_M0"


def normalize_log1p_sparse(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    csr = matrix.tocsr().astype(np.float32)
    cell_sums = np.asarray(csr.sum(axis=1)).ravel()
    scale = np.divide(1e4, cell_sums, out=np.zeros_like(cell_sums, dtype=np.float32), where=cell_sums > 0)
    csr = csr.multiply(scale[:, None]).tocsr()
    csr.data = np.log1p(csr.data)
    return csr


def sparse_total_and_detected(matrix: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    csr = matrix.tocsr()
    total = np.asarray(csr.sum(axis=1)).ravel()
    detected = np.diff(csr.indptr)
    return total, detected


def select_hvgs_by_variance(matrix: sparse.spmatrix, n_hvgs: int) -> np.ndarray:
    csr = matrix.tocsr()
    means = np.asarray(csr.mean(axis=0)).ravel()
    variances = np.asarray(csr.power(2).mean(axis=0)).ravel() - means**2
    n_select = min(int(n_hvgs), csr.shape[1])
    return np.argsort(variances)[-n_select:]


def compute_joint_svd_representation(
    matrices: list[sparse.spmatrix],
    metadata: list[pd.DataFrame],
    *,
    n_hvgs: int,
    n_pcs: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = sparse.vstack([matrix.tocsr() for matrix in matrices], format="csr")
    total_counts, detected_genes = sparse_total_and_detected(raw)
    norm = normalize_log1p_sparse(raw)
    hvg_idx = select_hvgs_by_variance(norm, n_hvgs)
    n_components = min(int(n_pcs), max(1, len(hvg_idx) - 1), max(1, norm.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    embedding = svd.fit_transform(norm[:, hvg_idx])
    frame = pd.concat(metadata, ignore_index=True).copy()
    frame["total_counts"] = total_counts
    frame["detected_genes"] = detected_genes
    frame["hvg_input_flag"] = True
    frame["sampled_flag"] = True
    for idx in range(embedding.shape[1]):
        frame[f"pca_{idx}"] = embedding[:, idx]
    payload = {
        "representation_label": REPRESENTATION_LABEL,
        "n_cellbins": int(frame.shape[0]),
        "n_genes": int(raw.shape[1]),
        "n_hvgs": int(len(hvg_idx)),
        "n_pcs": int(embedding.shape[1]),
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "finite_pca": bool(np.isfinite(embedding).all()),
    }
    return frame, payload


def build_group_representation(
    group_assignment: pd.DataFrame,
    representation: pd.DataFrame,
) -> pd.DataFrame:
    pca_cols = [column for column in representation.columns if column.startswith("pca_")]
    key_cols = list(PRIMARY_JOIN_KEY)
    member_features = group_assignment.merge(
        representation[key_cols + ["x", "y", *pca_cols]],
        on=key_cols,
        how="left",
        suffixes=("_assignment", ""),
    )
    if member_features[pca_cols].isna().any().any():
        raise ValueError("Group assignment contains member cellbins missing from representation")
    agg_spec: dict[str, tuple[str, str]] = {
        "sample_id": ("sample_id", "first"),
        "slice_id": ("slice_id", "first"),
        "section_order": ("section_order", "first"),
        "niche_id": ("niche_id", "first"),
        "group_type": ("group_type", "first"),
        "anchor_cellbin_id": ("anchor_cellbin_id", "first"),
        "centroid_x": ("x", "mean"),
        "centroid_y": ("y", "mean"),
        "anchor_x": ("anchor_x", "first"),
        "anchor_y": ("anchor_y", "first"),
        "n_member_cellbins": ("cellbin_id", "nunique"),
    }
    for column in pca_cols:
        agg_spec[f"pca_mean_{column.split('_', 1)[1]}"] = (column, "mean")
    grouped = (
        member_features.groupby("group_id", as_index=False, dropna=False)
        .agg(**agg_spec)
        .sort_values(["sample_id", "group_id"])
        .reset_index(drop=True)
    )
    grouped.insert(0, "unit_label", UNIT_LABEL)
    return grouped


def _section_entropy(counts: np.ndarray) -> float:
    values = counts.astype(float)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def assign_metaniches(
    group_representation: pd.DataFrame,
    *,
    n_metaniches: int,
    seed: int,
    section_purity_threshold: float = 0.9,
    tiny_group_threshold: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feature_cols = [column for column in group_representation.columns if column.startswith("pca_mean_")]
    feature_cols = feature_cols + ["centroid_x", "centroid_y"]
    matrix = group_representation[feature_cols].to_numpy(dtype=float)
    matrix = StandardScaler().fit_transform(matrix)
    n_clusters = min(int(n_metaniches), matrix.shape[0])
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = model.fit_predict(matrix)
    assignment = group_representation[
        ["sample_id", "slice_id", "section_order", "group_id", "niche_id"]
    ].copy()
    assignment["metaniche_id"] = [f"L126_routeA_metaniche_{label:03d}" for label in labels]
    enriched = group_representation.merge(assignment[["group_id", "metaniche_id"]], on="group_id", how="left")

    pca_mean_cols = [column for column in group_representation.columns if column.startswith("pca_mean_")]
    summary = (
        enriched.groupby("metaniche_id", as_index=False)
        .agg(
            n_groups=("group_id", "nunique"),
            centroid_x_mean=("centroid_x", "mean"),
            centroid_y_mean=("centroid_y", "mean"),
            section_order_min=("section_order", "min"),
            section_order_max=("section_order", "max"),
            **{column: (column, "mean") for column in pca_mean_cols},
        )
        .sort_values("metaniche_id")
        .reset_index(drop=True)
    )
    distributions = []
    for metaniche_id, group in enriched.groupby("metaniche_id", sort=True):
        counts = group["sample_id"].astype(str).value_counts().sort_index()
        total = int(counts.sum())
        purity = float(counts.max() / total) if total else 0.0
        distributions.append(
            {
                "metaniche_id": metaniche_id,
                "section_distribution": ";".join(f"{key}:{int(value)}" for key, value in counts.items()),
                "section_purity": purity,
                "section_entropy": _section_entropy(counts.to_numpy()),
                "section_dominated": bool(purity > section_purity_threshold),
            }
        )
    section_qc = pd.DataFrame(distributions)
    summary = summary.merge(section_qc, on="metaniche_id", how="left")
    summary["tiny_metaniche"] = summary["n_groups"] < int(tiny_group_threshold)
    sizes = summary["n_groups"].to_numpy(dtype=float)
    payload = {
        "unit_label": UNIT_LABEL,
        "n_groups": int(group_representation["group_id"].nunique()),
        "n_metaniches_requested": int(n_metaniches),
        "n_metaniches_observed": int(summary["metaniche_id"].nunique()),
        "section_purity_threshold": float(section_purity_threshold),
        "section_dominated_metaniches": int(summary["section_dominated"].sum()),
        "section_dominated_fraction": float(summary["section_dominated"].mean()) if len(summary) else 0.0,
        "tiny_group_threshold": int(tiny_group_threshold),
        "tiny_metaniches": int(summary["tiny_metaniche"].sum()),
        "tiny_metaniche_fraction": float(summary["tiny_metaniche"].mean()) if len(summary) else 0.0,
        "empty_metaniches": int(int(n_metaniches) - summary["metaniche_id"].nunique()),
        "size_min": float(np.min(sizes)) if sizes.size else 0.0,
        "size_p5": float(np.quantile(sizes, 0.05)) if sizes.size else 0.0,
        "size_median": float(np.median(sizes)) if sizes.size else 0.0,
        "size_p95": float(np.quantile(sizes, 0.95)) if sizes.size else 0.0,
        "size_max": float(np.max(sizes)) if sizes.size else 0.0,
    }
    return assignment, summary, payload


def _assay_balance(values: list[float]) -> float:
    counts = np.asarray([float(value) for value in values if float(value) > 0], dtype=float)
    if counts.size <= 1:
        return 0.0
    probabilities = counts / float(counts.sum())
    return float(-(probabilities * np.log(probabilities)).sum() / math.log(counts.size))


def aggregate_lineage_for_unit_mapping(
    lineage_evidence: pd.DataFrame,
    unit_mapping: pd.DataFrame,
    *,
    unit_col: str,
    assays: tuple[str, ...] = EXPECTED_ASSAYS,
    local_context: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key_cols = list(PRIMARY_JOIN_KEY)
    required = [unit_col, *key_cols]
    missing = [column for column in required if column not in unit_mapping.columns]
    if missing:
        raise ValueError(f"unit mapping missing columns: {missing}")
    mapping = unit_mapping[required].copy()
    if not local_context:
        mapping = mapping.drop_duplicates([unit_col, *key_cols])
    evidence = lineage_evidence.copy()
    evidence["count"] = pd.to_numeric(evidence["count"], errors="raise")
    evidence = evidence.loc[evidence["count"] > 0].copy()
    evidence["assay_feature_id"] = evidence["assay"].astype(str) + "::" + evidence["feature_id"].astype(str)

    units = mapping[[unit_col]].drop_duplicates()
    member = (
        mapping.groupby(unit_col, as_index=False)
        .agg(
            n_member_cellbin_records=("cellbin_id", "size"),
            n_unique_member_cellbins=("cellbin_id", "nunique"),
        )
    )
    evidence_keys = evidence[key_cols].drop_duplicates().assign(_has_evidence=True)
    member_evidence = mapping.merge(evidence_keys, on=key_cols, how="left")
    member_evidence["_has_evidence"] = member_evidence["_has_evidence"].fillna(False).astype(bool)
    coverage = (
        member_evidence.groupby(unit_col, as_index=False)
        .agg(
            n_lineage_positive_cellbin_records=("_has_evidence", "sum"),
        )
    )
    positive_unique = (
        member_evidence.loc[member_evidence["_has_evidence"], [unit_col, "cellbin_id"]]
        .drop_duplicates()
        .groupby(unit_col)
        .size()
        .reset_index(name="n_unique_lineage_positive_cellbins")
    )
    coverage = coverage.merge(positive_unique, on=unit_col, how="left")
    coverage["n_unique_lineage_positive_cellbins"] = coverage["n_unique_lineage_positive_cellbins"].fillna(0).astype(int)

    joined = mapping.merge(evidence, on=key_cols, how="inner")
    summary = units.merge(member, on=unit_col, how="left").merge(coverage, on=unit_col, how="left")
    if joined.empty:
        for column in [
            "total_lineage_count",
            "detected_feature_count",
            "detected_assay_count",
            "dominant_feature_count",
            "dominant_feature_fraction",
            "feature_entropy",
            "simpson_diversity",
            "assay_balance",
            *[f"{assay}_total_count" for assay in assays],
            *[f"{assay}_detected_feature_count" for assay in assays],
        ]:
            summary[column] = 0
        summary["dominant_assay"] = ""
        summary["dominant_feature_id"] = ""
        summary["fraction_lineage_positive"] = 0.0
        summary["evidence_present"] = pd.Series([False] * len(summary), dtype=object)
        summary["local_context_not_tissue_partition"] = bool(local_context)
        empty_assay = pd.DataFrame(columns=[unit_col, "assay", "assay_total_count", "assay_detected_feature_count"])
        empty_top = pd.DataFrame(columns=[unit_col, "assay", "feature_id", "clone_id", "feature_count", "feature_rank"])
        return summary, empty_assay, empty_top

    total = joined.groupby(unit_col, as_index=False)["count"].sum().rename(columns={"count": "total_lineage_count"})
    detected_feature = joined.groupby(unit_col)["assay_feature_id"].nunique().reset_index(name="detected_feature_count")
    detected_assay = joined.groupby(unit_col)["assay"].nunique().reset_index(name="detected_assay_count")
    assay_total = joined.pivot_table(index=unit_col, columns="assay", values="count", aggfunc="sum", fill_value=0).reset_index()
    assay_total.columns.name = None
    for assay in assays:
        if assay not in assay_total:
            assay_total[assay] = 0
    assay_total = assay_total[[unit_col, *assays]].rename(columns={assay: f"{assay}_total_count" for assay in assays})
    assay_features = (
        joined.groupby([unit_col, "assay"])["assay_feature_id"].nunique().unstack(fill_value=0).reset_index()
    )
    assay_features.columns.name = None
    for assay in assays:
        if assay not in assay_features:
            assay_features[assay] = 0
    assay_features = assay_features[[unit_col, *assays]].rename(columns={assay: f"{assay}_detected_feature_count" for assay in assays})
    feature_counts = joined.groupby([unit_col, "assay", "feature_id", "clone_id"], as_index=False)["count"].sum()
    feature_counts["assay_order"] = feature_counts["assay"].map({assay: idx for idx, assay in enumerate(assays)}).fillna(len(assays))
    dominant = (
        feature_counts.sort_values([unit_col, "count", "assay_order", "feature_id"], ascending=[True, False, True, True])
        .drop_duplicates(unit_col)
        [[unit_col, "assay", "feature_id", "count"]]
        .rename(columns={"assay": "dominant_assay", "feature_id": "dominant_feature_id", "count": "dominant_feature_count"})
    )
    diversity_rows = []
    for unit, group in feature_counts.groupby(unit_col, sort=False):
        metrics = compute_lineage_diversity_metrics(group["count"].to_numpy())
        diversity_rows.append({unit_col: unit, **metrics})
    diversity = pd.DataFrame(diversity_rows)
    summary = summary.merge(total, on=unit_col, how="left")
    for frame in [detected_feature, detected_assay, assay_total, assay_features, dominant, diversity]:
        summary = summary.merge(frame, on=unit_col, how="left")
    top = feature_counts.sort_values([unit_col, "count", "assay", "feature_id"], ascending=[True, False, True, True]).copy()
    top["feature_rank"] = top.groupby(unit_col).cumcount() + 1
    top = top.loc[top["feature_rank"] <= 10].rename(columns={"count": "feature_count"})

    numeric_cols = [
        "n_lineage_positive_cellbin_records",
        "n_unique_lineage_positive_cellbins",
        "total_lineage_count",
        "detected_feature_count",
        "detected_assay_count",
        "dominant_feature_count",
        "feature_entropy",
        "simpson_diversity",
        *[f"{assay}_total_count" for assay in assays],
        *[f"{assay}_detected_feature_count" for assay in assays],
    ]
    for column in numeric_cols:
        if column not in summary:
            summary[column] = 0
        summary[column] = summary[column].fillna(0)
    denominator = "n_member_cellbin_records" if local_context else "n_unique_member_cellbins"
    summary["fraction_lineage_positive"] = np.where(
        summary[denominator] > 0,
        summary["n_lineage_positive_cellbin_records" if local_context else "n_unique_lineage_positive_cellbins"] / summary[denominator],
        0.0,
    )
    summary["dominant_assay"] = summary.get("dominant_assay", "").fillna("")
    summary["dominant_feature_id"] = summary.get("dominant_feature_id", "").fillna("")
    summary["dominant_feature_fraction"] = np.where(
        summary["total_lineage_count"] > 0,
        summary["dominant_feature_count"] / summary["total_lineage_count"],
        0.0,
    )
    summary["assay_balance"] = summary[[f"{assay}_total_count" for assay in assays]].apply(
        lambda row: _assay_balance(row.to_list()), axis=1
    )
    summary["evidence_present"] = summary["total_lineage_count"].gt(0).map(bool).astype(object)
    summary["local_context_not_tissue_partition"] = bool(local_context)
    assay_summary = (
        joined.groupby([unit_col, "assay"], as_index=False)
        .agg(
            assay_total_count=("count", "sum"),
            assay_detected_feature_count=("assay_feature_id", "nunique"),
        )
        .sort_values([unit_col, "assay"])
    )
    return summary.sort_values(unit_col), assay_summary, top.drop(columns=["assay_order"]).sort_values([unit_col, "feature_rank"])


def compare_barcode_views(local: pd.DataFrame, unique: pd.DataFrame, unit_col: str) -> pd.DataFrame:
    keep = [
        unit_col,
        "total_lineage_count",
        "detected_feature_count",
        "fraction_lineage_positive",
        "feature_entropy",
        "dominant_feature_fraction",
        "RA_total_count",
        "TA_total_count",
        "CA_total_count",
    ]
    merged = local[keep].merge(unique[keep], on=unit_col, suffixes=("_local_context", "_unique_cellbin"))
    merged["local_to_unique_total_count_ratio"] = np.where(
        merged["total_lineage_count_unique_cellbin"] > 0,
        merged["total_lineage_count_local_context"] / merged["total_lineage_count_unique_cellbin"],
        0.0,
    )
    return merged


def build_state_matrix(
    metaniche_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        column
        for column in metaniche_summary.columns
        if column.startswith("pca_mean_") or column in {"centroid_x_mean", "centroid_y_mean"}
    ]
    metadata_cols = [
        "metaniche_id",
        "n_groups",
        "section_distribution",
        "section_purity",
        "section_entropy",
        "section_dominated",
        "tiny_metaniche",
    ]
    matrix = metaniche_summary[["metaniche_id", *feature_cols]].copy()
    metadata = metaniche_summary[metadata_cols].copy()
    return matrix, metadata


def gpcca_dryrun_checks(
    state_matrix: pd.DataFrame,
    state_metadata: pd.DataFrame,
    barcode_annotation: pd.DataFrame,
    *,
    unit_col: str = "metaniche_id",
) -> dict[str, Any]:
    feature = state_matrix.drop(columns=[unit_col]).to_numpy(dtype=float)
    join = state_metadata[[unit_col]].merge(barcode_annotation[[unit_col]], on=unit_col, how="left", indicator=True)
    checks = {
        "state_matrix_rows_equal_metadata_rows": int(len(state_matrix)) == int(len(state_metadata)),
        "finite_values_only": bool(np.isfinite(feature).all()),
        "nonzero_feature_variance": bool((np.nanvar(feature, axis=0) > 0).any()),
        "metaniche_size_threshold_passed": bool((~state_metadata["tiny_metaniche"].astype(bool)).all()),
        "barcode_annotation_join_success": bool(join["_merge"].eq("both").all()),
        "section_distribution_exists": bool(state_metadata["section_distribution"].astype(str).ne("").all()),
        "kernel_constructed": False,
        "gpcca_run": False,
        "fate_probability_computed": False,
    }
    label = (
        "L126_BOUNDED_GPCCA_INPUT_DRYRUN_READY"
        if all(value for key, value in checks.items() if key not in {"kernel_constructed", "gpcca_run", "fate_probability_computed"})
        else "L126_BOUNDED_GPCCA_INPUT_READY_WITH_WARNINGS"
    )
    return {"readiness_label": label, "checks": checks}


def forbidden_claim_hits(text: str) -> list[str]:
    forbidden = [
        "proves fate",
        "validated temporal fate",
        "true terminal state",
        "lineage-validated transition",
        "clonal expansion proven",
        "biological fate discovered",
        "terminal biological fate",
    ]
    lowered = text.lower()
    return [phrase for phrase in forbidden if phrase in lowered]
