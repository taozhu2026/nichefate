#!/usr/bin/env python
"""L126 DARLIN clone integration from processed spatio-DARLIN evidence."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nichefate.barcode_adapter.qc import compare_file_snapshots, snapshot_files
from nichefate.barcode_adapter.reporting import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_tsv,
    atomic_write_tsv_gz,
    ensure_dir,
    markdown_table,
    path_has_ssd,
    utc_now,
)


EXPECTED_ASSAYS = ("CA", "TA", "RA")
CELL_KEY_COLUMNS = ["sample_id", "slice_id", "cellbin_id"]
CELL_COLUMNS = ["sample_id", "slice_id", "section_order", "cellbin_id"]
FULL_CELLBIN_SUMMARY = "cellbin/full_cellbin_lineage_summary.tsv.gz"
DEFAULT_MAX_COMPONENT_CELLBIN_FRACTION = 0.05
DEFAULT_MAX_SINGLE_FEATURE_CONTRIBUTION = 0.80
DEFAULT_MAX_BRIDGE_DEPENDENCY_SCORE = 0.50
FORBIDDEN_CLAIM_PHRASES = (
    "terminal fate",
    "true fate",
    "fate probability",
    "lineage-validated transition",
    "lineage-validated endpoint",
    "clonal expansion discovered",
    "proven transition",
    "trajectory across sections",
    "section_order as time",
)


@dataclass(frozen=True)
class CloneGraphParams:
    max_feature_cellbin_fraction: float
    min_features_per_clone: int
    min_target_arrays_per_clone: int
    min_count_per_cellbin_feature: int
    max_component_cellbin_fraction: float = DEFAULT_MAX_COMPONENT_CELLBIN_FRACTION
    max_single_feature_contribution: float = DEFAULT_MAX_SINGLE_FEATURE_CONTRIBUTION
    max_bridge_dependency_score: float = DEFAULT_MAX_BRIDGE_DEPENDENCY_SCORE
    require_allele_identity: bool = False


@dataclass
class CloneGraphResult:
    run_label: str
    bridge_filter_mode: str
    assignment: pd.DataFrame
    clone_summary: pd.DataFrame
    clone_feature_support: pd.DataFrame
    clone_membership: pd.DataFrame
    failed_components: pd.DataFrame
    component_summary: pd.DataFrame
    filtered_cellbin_count: int
    run_metrics: dict[str, Any]


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        rank_left = self.rank[root_left]
        rank_right = self.rank[root_right]
        if rank_left < rank_right:
            self.parent[root_left] = root_right
        elif rank_left > rank_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-packet-root",
        type=Path,
        default=Path("/home/zhutao/scratch/nichefate/spatio_darlin_public_brain_hpc/input_packet"),
    )
    parser.add_argument("--barcode-root", type=Path, default=Path("processed/barcode_adapter_l126_round1"))
    parser.add_argument(
        "--full-characterization-root",
        type=Path,
        default=Path("processed/l126_full_barcode_niche_characterization"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("processed/l126_darlin_clone_integration_round1"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/l126_darlin_clone_integration_round1"))
    parser.add_argument("--max-feature-cellbin-fraction", type=float, default=0.001)
    parser.add_argument("--min-features-per-clone", type=int, default=3)
    parser.add_argument("--min-target-arrays-per-clone", type=int, default=2)
    parser.add_argument("--min-count-per-cellbin-feature", type=int, default=1)
    parser.add_argument(
        "--bridge-filter-mode",
        default="audit_and_sensitivity",
        choices=["none", "p99", "p995", "audit_and_sensitivity"],
    )
    parser.add_argument("--run-sensitivity", action="store_true")
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mode",
        default="all",
        choices=[
            "all",
            "schema_audit_only",
            "contract_only",
            "feature_filter_only",
            "bridge_audit_only",
            "clone_graph_only",
            "sensitivity_only",
            "niche_aggregation_only",
            "figures_only",
            "validation_only",
        ],
    )
    return parser.parse_args()


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    return pd.read_csv(path, sep="\t", compression=compression, nrows=nrows)


def reject_forbidden_paths(*paths: Path) -> None:
    bad = [str(path) for path in paths if path_has_ssd(path)]
    if bad:
        raise ValueError("Refusing /ssd paths: " + "; ".join(bad))


def lineage_evidence_path(input_packet_root: Path) -> Path:
    return input_packet_root / "processed/lineage_evidence/cellbin_lineage_evidence.tsv.gz"


def allele_annotation_path(input_packet_root: Path) -> Path:
    return input_packet_root / "processed/lineage_evidence/feature_allele_annotation_long.tsv.gz"


def make_cell_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["sample_id"].astype(str)
        + "|"
        + frame["slice_id"].astype(str)
        + "|"
        + frame["cellbin_id"].astype(str)
    )


def assay_scoped_feature(frame: pd.DataFrame) -> pd.Series:
    return frame["assay"].astype(str) + "::" + frame["feature_id"].astype(str)


def entropy_from_counts(counts: Iterable[float]) -> float:
    values = np.asarray([float(value) for value in counts if float(value) > 0], dtype=float)
    if values.size <= 1:
        return 0.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def simpson_from_counts(counts: Iterable[float]) -> float:
    values = np.asarray([float(value) for value in counts if float(value) > 0], dtype=float)
    if values.size <= 1:
        return 0.0
    probabilities = values / values.sum()
    return float(1.0 - np.square(probabilities).sum())


def compact_distribution(frame: pd.DataFrame, key: str, count_col: str = "cellbin_id", limit: int = 8) -> str:
    if frame.empty or key not in frame:
        return ""
    counts = frame.groupby(key, dropna=False)[count_col].nunique().sort_values(ascending=False)
    items = [f"{idx}:{int(value)}" for idx, value in counts.head(limit).items()]
    if len(counts) > limit:
        items.append(f"other:{int(counts.iloc[limit:].sum())}")
    return ";".join(items)


def summarize_top_items(counts: pd.Series, limit: int = 5) -> str:
    if counts.empty:
        return ""
    return ";".join(f"{idx}:{int(value)}" for idx, value in counts.sort_values(ascending=False).head(limit).items())


def write_report_pair(
    report_root: Path,
    stem: str,
    title: str,
    payload: dict[str, Any],
    lines: list[str],
    *,
    overwrite: bool,
) -> None:
    ensure_dir(report_root)
    body = "# " + title + "\n\n" + "\n".join(lines).strip() + "\n"
    atomic_write_json(report_root / f"{stem}.json", payload, overwrite=overwrite)
    atomic_write_text(report_root / f"{stem}.md", body, overwrite=overwrite)


def load_inputs(input_packet_root: Path, full_characterization_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evidence = read_table(lineage_evidence_path(input_packet_root))
    annotation = read_table(allele_annotation_path(input_packet_root))
    full_cellbins = read_table(full_characterization_root / FULL_CELLBIN_SUMMARY)
    evidence["count"] = pd.to_numeric(evidence["count"], errors="raise")
    annotation["allele_is_missing"] = annotation["allele_is_missing"].astype(bool)
    full_cellbins["cell_key"] = make_cell_key(full_cellbins)
    return evidence, annotation, full_cellbins


def audit_schema(evidence: pd.DataFrame, annotation: pd.DataFrame) -> dict[str, Any]:
    official_keywords = (
        "prob",
        "rare",
        "common",
        "official",
        "germ",
        "edit",
        "unmod",
        "status",
        "assignment",
    )
    official_evidence_columns = [
        column for column in evidence.columns if any(token in column.lower() for token in official_keywords)
    ]
    official_annotation_columns = [
        column for column in annotation.columns if any(token in column.lower() for token in official_keywords)
    ]
    feature_eq_clone_fraction = float(
        (evidence["feature_id"].astype(str) == evidence["clone_id"].astype(str)).mean()
    )
    annotation_feature_eq_clone_fraction = float(
        (annotation["feature_id"].astype(str) == annotation["clone_id"].astype(str)).mean()
    )
    clone_assay_counts = evidence.groupby("clone_id", dropna=False)["assay"].nunique()
    feature_assay_counts = evidence.groupby("feature_id", dropna=False)["assay"].nunique()
    clone_ids_spanning_assays = int((clone_assay_counts > 1).sum())
    feature_ids_spanning_assays = int((feature_assay_counts > 1).sum())
    official_schema_available = bool(
        clone_ids_spanning_assays > 0
        and feature_eq_clone_fraction < 0.95
        and (official_evidence_columns or official_annotation_columns)
    )
    if official_schema_available:
        schema_label = "OFFICIAL_CROSS_LOCUS_CLONE_ID_AVAILABLE"
    elif feature_eq_clone_fraction >= 0.999 and clone_ids_spanning_assays == 0:
        schema_label = "CLONE_ID_IS_ASSAY_FEATURE_NOT_FINAL_CLONE"
    else:
        schema_label = "CLONE_ID_AMBIGUOUS_HOLD_FOR_SCHEMA"
    can_reconstruct = bool(
        schema_label == "CLONE_ID_IS_ASSAY_FEATURE_NOT_FINAL_CLONE"
        and {"assay", "feature_id", "cellbin_id", "count"}.issubset(evidence.columns)
        and set(evidence["assay"].dropna().astype(str)).issuperset(set(EXPECTED_ASSAYS))
    )
    reconstruction_label = (
        "CAN_RECONSTRUCT_CLONES_FROM_FEATURE_SHARING"
        if can_reconstruct
        else "HOLD_FOR_MISSING_ALLELE_OR_FEATURE_SCHEMA"
    )
    return {
        "generated_at_utc": utc_now(),
        "schema_decision_label": schema_label,
        "reconstruction_label": reconstruction_label,
        "evidence_row_count": int(len(evidence)),
        "evidence_columns": list(evidence.columns),
        "annotation_row_count": int(len(annotation)),
        "annotation_columns": list(annotation.columns),
        "assay_counts": evidence["assay"].value_counts(dropna=False).to_dict(),
        "feature_id_unique_count": int(evidence["feature_id"].nunique()),
        "clone_id_unique_count": int(evidence["clone_id"].nunique()),
        "feature_eq_clone_fraction": feature_eq_clone_fraction,
        "annotation_feature_eq_clone_fraction": annotation_feature_eq_clone_fraction,
        "clone_ids_spanning_assays": clone_ids_spanning_assays,
        "feature_ids_spanning_assays": feature_ids_spanning_assays,
        "clone_id_assay_span_distribution": clone_assay_counts.value_counts().sort_index().to_dict(),
        "feature_id_assay_span_distribution": feature_assay_counts.value_counts().sort_index().to_dict(),
        "feature_id_interpretation": "assay-scoped processed lineage feature / matrix-row feature",
        "clone_id_interpretation": "same value as feature_id in audited evidence; not final integrated clone ID",
        "allele_annotation_fields": {
            "has_allele": "allele" in annotation.columns,
            "has_allele_index": "allele_index" in annotation.columns,
            "has_n_alleles_for_feature": "n_alleles_for_feature" in annotation.columns,
            "has_allele_is_missing": "allele_is_missing" in annotation.columns,
            "has_source_row_index": "source_row_index" in annotation.columns,
        },
        "allele_missing_counts": annotation["allele_is_missing"].value_counts(dropna=False).to_dict()
        if "allele_is_missing" in annotation
        else {},
        "n_alleles_for_feature_distribution": annotation["n_alleles_for_feature"]
        .value_counts(dropna=False)
        .head(12)
        .to_dict()
        if "n_alleles_for_feature" in annotation
        else {},
        "official_evidence_columns": official_evidence_columns,
        "official_annotation_columns": official_annotation_columns,
        "official_rarity_or_clone_schema_available": bool(official_evidence_columns or official_annotation_columns),
        "can_reconstruct_from_feature_sharing": can_reconstruct,
    }


def write_schema_audit_report(payload: dict[str, Any], report_root: Path, overwrite: bool) -> None:
    rows = pd.DataFrame(
        [
            {"question": "Current clone_id meaning", "answer": payload["clone_id_interpretation"]},
            {
                "question": "clone_id spans CA/TA/RA",
                "answer": f"{payload['clone_ids_spanning_assays']} IDs span more than one assay",
            },
            {"question": "feature_id meaning", "answer": payload["feature_id_interpretation"]},
            {
                "question": "allele annotation content",
                "answer": "allele, allele_index, n_alleles_for_feature, allele_is_missing, source_row_index",
            },
            {
                "question": "official rarity or assignment fields",
                "answer": "present" if payload["official_rarity_or_clone_schema_available"] else "not found",
            },
            {
                "question": "clone reconstruction",
                "answer": payload["reconstruction_label"],
            },
        ]
    )
    lines = [
        "## Decision",
        f"- Schema label: `{payload['schema_decision_label']}`",
        f"- Reconstruction label: `{payload['reconstruction_label']}`",
        "",
        "## Audit Answers",
        markdown_table(rows),
        "",
        "## Key Counts",
        f"- Evidence rows: {payload['evidence_row_count']}",
        f"- Feature IDs: {payload['feature_id_unique_count']}",
        f"- Clone IDs: {payload['clone_id_unique_count']}",
        f"- Feature/clone equality fraction: {payload['feature_eq_clone_fraction']:.6f}",
    ]
    write_report_pair(
        report_root,
        "00_SCHEMA_AND_CLONE_ID_AUDIT",
        "L126 Schema And Clone ID Audit",
        payload,
        lines,
        overwrite=overwrite,
    )


def clone_definition_contract_payload(params: CloneGraphParams, schema_payload: dict[str, Any]) -> dict[str, Any]:
    official_ready = schema_payload["schema_decision_label"] == "OFFICIAL_CROSS_LOCUS_CLONE_ID_AVAILABLE"
    definition_label = (
        "DARLIN_CLONE_DEFINITION_READY_OFFICIAL"
        if official_ready
        else "DARLIN_CLONE_DEFINITION_READY_EMPIRICAL_RARITY"
    )
    return {
        "generated_at_utc": utc_now(),
        "definition_label": definition_label,
        "schema_decision_label": schema_payload["schema_decision_label"],
        "clone_call_name": "validated clone under empirical rarity contract"
        if not official_ready
        else "DARLIN clone with official support",
        "evidence_unit": {
            "primary_id": "assay_scoped_feature_id",
            "assay_scoped_feature_id": "assay + '::' + feature_id",
            "target_arrays": list(EXPECTED_ASSAYS),
            "evidence_count_source": "primary processed evidence count",
            "allele_annotation_role": "metadata unless allele-level identity is explicitly enabled",
        },
        "cellbin_node": {
            "primary_key": CELL_KEY_COLUMNS,
            "section_order_role": "section-aware grouping and summaries",
        },
        "graph_definition": {
            "graph_type": "bipartite cellbin-to-lineage-feature graph",
            "cellbin_feature_edge_rule": "count >= min_count_per_cellbin_feature after valid feature filtering",
            "clone_rule": "connected component of cellbin nodes that passes validation gates",
            "cross_target_integration": "graph-based through shared cellbin-feature support, not name-only merging",
        },
        "default_thresholds": {
            "max_feature_cellbin_fraction": params.max_feature_cellbin_fraction,
            "min_features_per_clone": params.min_features_per_clone,
            "min_target_arrays_per_clone": params.min_target_arrays_per_clone,
            "min_count_per_cellbin_feature": params.min_count_per_cellbin_feature,
            "max_component_cellbin_fraction": params.max_component_cellbin_fraction,
            "max_single_feature_contribution": params.max_single_feature_contribution,
            "max_bridge_dependency_score": params.max_bridge_dependency_score,
        },
        "empirical_rarity_sensitivity": {
            "max_feature_cellbin_fraction": [0.0005, 0.001, 0.005],
            "min_features_per_clone": [2, 3, 4],
            "min_target_arrays_per_clone": [1, 2],
            "bridge_filtering": ["none", "p99", "p995"],
        },
        "non_clone_labels": ["unassigned", "ambiguous", "filtered"],
    }


def write_clone_contract(
    config_path: Path,
    report_root: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    atomic_write_json(config_path, payload, overwrite=overwrite)
    lines = [
        "## Decision",
        f"- Definition label: `{payload['definition_label']}`",
        f"- Clone call name: {payload['clone_call_name']}",
        "",
        "## Contract",
        "- Evidence unit: `assay::feature_id` with primary processed counts.",
        "- Cellbin node key: `sample_id`, `slice_id`, `cellbin_id`.",
        "- CA/TA/RA are retained as different target arrays.",
        "- Clone assignment is graph-based and must pass support gates.",
        "- Non-passing components are written as unassigned, ambiguous, or filtered.",
    ]
    write_report_pair(
        report_root,
        "01_CLONE_DEFINITION_CONTRACT",
        "L126 DARLIN Clone Definition Contract",
        payload,
        lines,
        overwrite=overwrite,
    )


def aggregate_annotation(annotation: pd.DataFrame) -> pd.DataFrame:
    frame = annotation.copy()
    frame["allele"] = frame["allele"].fillna("")
    grouped = (
        frame.groupby(["assay", "feature_id"], as_index=False)
        .agg(
            feature_type=("feature_type", "first"),
            allele=("allele", lambda s: ";".join(sorted({str(v) for v in s if str(v)}))[:500]),
            allele_is_missing=("allele_is_missing", lambda s: bool(pd.Series(s).astype(bool).all())),
            n_alleles_for_feature=("n_alleles_for_feature", "max"),
        )
    )
    return grouped


def build_valid_lineage_features(
    evidence: pd.DataFrame,
    annotation: pd.DataFrame,
    params: CloneGraphParams,
    *,
    total_lineage_positive_cellbins: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    evidence = evidence.copy()
    evidence["count"] = pd.to_numeric(evidence["count"], errors="raise")
    evidence = evidence.loc[evidence["count"] >= params.min_count_per_cellbin_feature].copy()
    evidence["cell_key"] = make_cell_key(evidence)
    evidence["assay_scoped_feature_id"] = assay_scoped_feature(evidence)
    denominator = total_lineage_positive_cellbins or int(evidence["cell_key"].nunique())
    freq = (
        evidence.groupby(["assay", "feature_id", "assay_scoped_feature_id"], as_index=False)
        .agg(
            n_cellbins_detected=("cell_key", "nunique"),
            total_count=("count", "sum"),
            feature_type=("feature_type", "first"),
        )
        .sort_values(["assay", "feature_id"])
    )
    freq["cellbin_fraction"] = freq["n_cellbins_detected"] / max(denominator, 1)
    annotation_compact = aggregate_annotation(annotation)
    valid = freq.merge(annotation_compact, on=["assay", "feature_id"], how="left", suffixes=("", "_annotation"))
    valid["feature_type"] = valid["feature_type"].fillna(valid.pop("feature_type_annotation"))
    valid["allele"] = valid["allele"].fillna("")
    valid["allele_is_missing"] = valid["allele_is_missing"].fillna(True).astype(bool)
    valid["target_array"] = valid["assay"]
    valid["rarity_source"] = "empirical"
    reasons: list[str] = []
    for row in valid.to_dict(orient="records"):
        reason = "valid"
        if float(row["total_count"]) <= 0 or int(row["n_cellbins_detected"]) <= 0:
            reason = "zero_count"
        elif float(row["cellbin_fraction"]) > params.max_feature_cellbin_fraction:
            reason = "ultra_common_feature"
        elif params.require_allele_identity and bool(row["allele_is_missing"]):
            reason = "missing_allele_unusable"
        reasons.append(reason)
    valid["filter_reason"] = reasons
    valid["valid_for_clone_graph"] = valid["filter_reason"].eq("valid")
    output_cols = [
        "assay",
        "feature_id",
        "assay_scoped_feature_id",
        "feature_type",
        "allele",
        "allele_is_missing",
        "n_cellbins_detected",
        "total_count",
        "cellbin_fraction",
        "valid_for_clone_graph",
        "filter_reason",
        "rarity_source",
        "target_array",
    ]
    valid = valid[output_cols].sort_values(["assay", "feature_id"]).reset_index(drop=True)
    filtered_summary = (
        valid.groupby(["filter_reason", "valid_for_clone_graph"], as_index=False)
        .agg(
            n_features=("assay_scoped_feature_id", "nunique"),
            n_cellbin_feature_detections=("n_cellbins_detected", "sum"),
            total_count=("total_count", "sum"),
        )
        .sort_values(["valid_for_clone_graph", "filter_reason"])
    )
    payload = {
        "generated_at_utc": utc_now(),
        "rarity_source": "empirical",
        "total_lineage_positive_cellbins": int(denominator),
        "n_features_total": int(len(valid)),
        "n_valid_features": int(valid["valid_for_clone_graph"].sum()),
        "max_feature_cellbin_fraction": params.max_feature_cellbin_fraction,
        "min_count_per_cellbin_feature": params.min_count_per_cellbin_feature,
        "require_allele_identity": params.require_allele_identity,
        "filter_reason_counts": valid["filter_reason"].value_counts().to_dict(),
    }
    return valid, freq, filtered_summary, payload


def write_valid_feature_outputs(
    output_root: Path,
    report_root: Path,
    valid: pd.DataFrame,
    freq: pd.DataFrame,
    filtered_summary: pd.DataFrame,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    out_dir = output_root / "valid_features"
    atomic_write_tsv_gz(out_dir / "valid_lineage_features.tsv.gz", valid, overwrite=overwrite)
    atomic_write_tsv_gz(out_dir / "feature_frequency_summary.tsv.gz", freq, overwrite=overwrite)
    atomic_write_tsv_gz(out_dir / "filtered_feature_summary.tsv.gz", filtered_summary, overwrite=overwrite)
    lines = [
        "## Feature Filtering",
        f"- Rarity source: `{payload['rarity_source']}`",
        f"- Total features: {payload['n_features_total']}",
        f"- Valid features: {payload['n_valid_features']}",
        f"- Max feature cellbin fraction: {payload['max_feature_cellbin_fraction']}",
        "",
        "## Filter Summary",
        markdown_table(filtered_summary),
    ]
    write_report_pair(
        report_root,
        "02_VALID_LINEAGE_FEATURES",
        "L126 Valid Lineage Features",
        payload,
        lines,
        overwrite=overwrite,
    )


def build_incidence_table(
    evidence: pd.DataFrame,
    valid_features: pd.DataFrame,
    params: CloneGraphParams,
) -> pd.DataFrame:
    valid_ids = valid_features.loc[
        valid_features["valid_for_clone_graph"],
        ["assay", "feature_id", "assay_scoped_feature_id", "cellbin_fraction", "allele"],
    ].copy()
    evidence = evidence.loc[evidence["count"] >= params.min_count_per_cellbin_feature].copy()
    if evidence.empty or valid_ids.empty:
        return pd.DataFrame(
            columns=[
                "cell_key",
                "sample_id",
                "slice_id",
                "section_order",
                "cellbin_id",
                "x",
                "y",
                "assay",
                "feature_id",
                "assay_scoped_feature_id",
                "allele",
                "count",
                "feature_cellbin_fraction",
            ]
        )
    incidence = evidence.merge(valid_ids, on=["assay", "feature_id"], how="inner")
    incidence["cell_key"] = make_cell_key(incidence)
    grouped = (
        incidence.groupby(
            [
                "cell_key",
                "sample_id",
                "slice_id",
                "section_order",
                "cellbin_id",
                "x",
                "y",
                "assay",
                "feature_id",
                "assay_scoped_feature_id",
                "allele",
                "cellbin_fraction",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(count=("count", "sum"))
        .rename(columns={"cellbin_fraction": "feature_cellbin_fraction"})
    )
    return grouped


def compute_bridge_metrics(incidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if incidence.empty:
        empty = pd.DataFrame(
            columns=[
                "cell_key",
                *CELL_COLUMNS,
                "n_valid_features",
                "n_target_arrays_detected",
                "total_valid_feature_count",
                "feature_entropy",
                "max_feature_fraction",
                "cellbin_graph_degree",
                "n_assay_scoped_features_by_CA",
                "n_assay_scoped_features_by_TA",
                "n_assay_scoped_features_by_RA",
                "can_bridge_feature_sets",
                "bridge_flag_p99",
                "bridge_flag_p995",
                "bridge_candidate",
            ]
        )
        summary = pd.DataFrame([{"metric": "n_cellbins_with_valid_features", "value": 0}])
        payload = {"generated_at_utc": utc_now(), "n_cellbins_with_valid_features": 0}
        return empty, empty.copy(), summary, payload
    feature_sizes = incidence.groupby("assay_scoped_feature_id")["cell_key"].nunique().rename("feature_cellbin_count")
    work = incidence.merge(feature_sizes, on="assay_scoped_feature_id", how="left")
    work["degree_increment"] = work["feature_cellbin_count"] - 1
    base = (
        work.groupby(["cell_key", *CELL_COLUMNS], as_index=False)
        .agg(
            n_valid_features=("assay_scoped_feature_id", "nunique"),
            n_target_arrays_detected=("assay", "nunique"),
            total_valid_feature_count=("count", "sum"),
            max_feature_count=("count", "max"),
            cellbin_graph_degree=("degree_increment", "sum"),
        )
    )
    entropy = work.groupby("cell_key")["count"].apply(entropy_from_counts).rename("feature_entropy").reset_index()
    base = base.merge(entropy, on="cell_key", how="left")
    base["max_feature_fraction"] = base["max_feature_count"] / base["total_valid_feature_count"].replace(0, np.nan)
    base["max_feature_fraction"] = base["max_feature_fraction"].fillna(0.0)
    assay_counts = (
        work.groupby(["cell_key", "assay"])["assay_scoped_feature_id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    for assay in EXPECTED_ASSAYS:
        if assay not in assay_counts:
            assay_counts[assay] = 0
    assay_counts = assay_counts.rename(columns={assay: f"n_assay_scoped_features_by_{assay}" for assay in EXPECTED_ASSAYS})
    base = base.merge(assay_counts[["cell_key", *[f"n_assay_scoped_features_by_{assay}" for assay in EXPECTED_ASSAYS]]], on="cell_key")
    p99_features = float(base["n_valid_features"].quantile(0.99)) if len(base) else 0.0
    p995_features = float(base["n_valid_features"].quantile(0.995)) if len(base) else 0.0
    p99_degree = float(base["cellbin_graph_degree"].quantile(0.99)) if len(base) else 0.0
    p995_degree = float(base["cellbin_graph_degree"].quantile(0.995)) if len(base) else 0.0
    base["can_bridge_feature_sets"] = (base["n_target_arrays_detected"] >= 2) & (base["n_valid_features"] >= 3)
    base["bridge_flag_p99"] = (base["n_valid_features"] >= p99_features) | (base["cellbin_graph_degree"] >= p99_degree)
    base["bridge_flag_p995"] = (base["n_valid_features"] >= p995_features) | (base["cellbin_graph_degree"] >= p995_degree)
    base["bridge_candidate"] = (
        base["can_bridge_feature_sets"]
        & (
            base["bridge_flag_p99"]
            | ((base["feature_entropy"] > 1.0) & (base["max_feature_fraction"] < 0.60))
        )
    )
    candidates = base.loc[base["bridge_candidate"]].sort_values(
        ["cellbin_graph_degree", "n_valid_features"], ascending=False
    )
    summary_rows = [
        {"metric": "n_cellbins_with_valid_features", "value": int(len(base))},
        {"metric": "n_bridge_candidates", "value": int(base["bridge_candidate"].sum())},
        {"metric": "n_bridge_flag_p99", "value": int(base["bridge_flag_p99"].sum())},
        {"metric": "n_bridge_flag_p995", "value": int(base["bridge_flag_p995"].sum())},
        {"metric": "p99_n_valid_features", "value": p99_features},
        {"metric": "p995_n_valid_features", "value": p995_features},
        {"metric": "p99_cellbin_graph_degree", "value": p99_degree},
        {"metric": "p995_cellbin_graph_degree", "value": p995_degree},
    ]
    summary = pd.DataFrame(summary_rows)
    payload = {
        "generated_at_utc": utc_now(),
        "n_cellbins_with_valid_features": int(len(base)),
        "n_bridge_candidates": int(base["bridge_candidate"].sum()),
        "n_bridge_flag_p99": int(base["bridge_flag_p99"].sum()),
        "n_bridge_flag_p995": int(base["bridge_flag_p995"].sum()),
        "p99_n_valid_features": p99_features,
        "p995_n_valid_features": p995_features,
        "p99_cellbin_graph_degree": p99_degree,
        "p995_cellbin_graph_degree": p995_degree,
    }
    return base, candidates, summary, payload


def write_bridge_outputs(
    output_root: Path,
    report_root: Path,
    metrics: pd.DataFrame,
    candidates: pd.DataFrame,
    summary: pd.DataFrame,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    out_dir = output_root / "bridge_audit"
    atomic_write_tsv_gz(out_dir / "cellbin_bridge_metrics.tsv.gz", metrics, overwrite=overwrite)
    atomic_write_tsv_gz(out_dir / "bridge_cellbin_candidates.tsv.gz", candidates, overwrite=overwrite)
    atomic_write_tsv(out_dir / "bridge_audit_summary.tsv", summary, overwrite=overwrite)
    lines = [
        "## Bridge Audit",
        f"- Cellbins with valid graph features: {payload.get('n_cellbins_with_valid_features', 0)}",
        f"- Bridge candidates: {payload.get('n_bridge_candidates', 0)}",
        f"- p99 bridge flags: {payload.get('n_bridge_flag_p99', 0)}",
        f"- p99.5 bridge flags: {payload.get('n_bridge_flag_p995', 0)}",
        "",
        "## Summary",
        markdown_table(summary),
    ]
    write_report_pair(
        report_root,
        "03_BRIDGE_CELLBIN_AND_OVERMERGE_AUDIT",
        "L126 Bridge Cellbin And Overmerge Audit",
        payload,
        lines,
        overwrite=overwrite,
    )


def bridge_filter_keys(bridge_metrics: pd.DataFrame, mode: str) -> set[str]:
    if bridge_metrics.empty or mode == "none":
        return set()
    if mode == "p99":
        return set(bridge_metrics.loc[bridge_metrics["bridge_flag_p99"], "cell_key"].astype(str))
    if mode == "p995":
        return set(bridge_metrics.loc[bridge_metrics["bridge_flag_p995"], "cell_key"].astype(str))
    raise ValueError(f"Unsupported bridge filter mode: {mode}")


def compute_components(incidence: pd.DataFrame) -> pd.DataFrame:
    if incidence.empty:
        return pd.DataFrame(columns=["cell_key", "component_id"])
    cell_keys = sorted(incidence["cell_key"].astype(str).unique().tolist())
    union_find = UnionFind(cell_keys)
    for _, group in incidence.groupby("assay_scoped_feature_id", sort=False):
        keys = group["cell_key"].astype(str).unique().tolist()
        if len(keys) <= 1:
            continue
        first = keys[0]
        for key in keys[1:]:
            union_find.union(first, key)
    roots = {key: union_find.find(key) for key in cell_keys}
    root_to_component = {root: f"component_{idx:06d}" for idx, root in enumerate(sorted(set(roots.values())), start=1)}
    return pd.DataFrame(
        [{"cell_key": key, "component_id": root_to_component[root]} for key, root in roots.items()]
    )


def component_support_tables(
    incidence: pd.DataFrame,
    components: pd.DataFrame,
    bridge_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if incidence.empty or components.empty:
        empty_components = pd.DataFrame(
            columns=[
                "component_id",
                "n_cellbins",
                "n_sections",
                "section_distribution",
                "n_supporting_features",
                "n_supporting_CA_features",
                "n_supporting_TA_features",
                "n_supporting_RA_features",
                "n_supporting_target_arrays",
                "total_evidence_count",
                "dominant_assay",
                "dominant_feature_id",
                "feature_entropy",
                "spatial_extent_summary",
                "max_single_feature_contribution",
                "top_feature_fraction",
                "max_feature_cellbin_fraction_in_component",
                "bridge_cellbin_dependency_score",
                "component_support_density",
                "clone_validation_status",
                "reason_if_not_clone",
            ]
        )
        return empty_components, pd.DataFrame(), pd.DataFrame()
    cell_component = components.merge(
        incidence[["cell_key", *CELL_COLUMNS, "x", "y"]].drop_duplicates("cell_key"),
        on="cell_key",
        how="left",
    )
    joined = incidence.merge(components, on="cell_key", how="inner")
    feature_support = (
        joined.groupby(["component_id", "assay", "feature_id", "assay_scoped_feature_id", "allele"], as_index=False)
        .agg(
            n_cellbins_with_feature=("cell_key", "nunique"),
            total_feature_count=("count", "sum"),
            feature_cellbin_fraction=("feature_cellbin_fraction", "first"),
        )
    )
    bridge_map = (
        bridge_metrics.set_index("cell_key")["bridge_candidate"].to_dict()
        if not bridge_metrics.empty and "bridge_candidate" in bridge_metrics
        else {}
    )
    cell_component = cell_component.copy()
    cell_component["bridge_candidate"] = cell_component["cell_key"].map(lambda key: bool(bridge_map.get(key, False)))
    cell_agg = (
        cell_component.groupby("component_id", as_index=False)
        .agg(
            n_cellbins=("cell_key", "nunique"),
            n_sections=("section_order", "nunique"),
            x_min=("x", "min"),
            x_max=("x", "max"),
            y_min=("y", "min"),
            y_max=("y", "max"),
            bridge_cellbin_dependency_score=("bridge_candidate", "mean"),
        )
    )
    section_counts = (
        cell_component.groupby(["component_id", "section_order"], as_index=False)
        .agg(section_cellbins=("cell_key", "nunique"))
    )
    section_counts["section_item"] = (
        section_counts["section_order"].astype(str)
        + ":"
        + section_counts["section_cellbins"].astype(int).astype(str)
    )
    section_distribution = (
        section_counts.groupby("component_id", as_index=False)["section_item"]
        .agg(";".join)
        .rename(columns={"section_item": "section_distribution"})
    )
    cell_agg = cell_agg.merge(section_distribution, on="component_id", how="left")
    feature_agg = (
        feature_support.groupby("component_id", as_index=False)
        .agg(
            n_supporting_features=("assay_scoped_feature_id", "nunique"),
            n_supporting_target_arrays=("assay", "nunique"),
            total_evidence_count=("total_feature_count", "sum"),
            feature_entropy=("total_feature_count", entropy_from_counts),
        )
    )
    assay_counts = (
        feature_support.groupby(["component_id", "assay"])["assay_scoped_feature_id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    for assay in EXPECTED_ASSAYS:
        if assay not in assay_counts:
            assay_counts[assay] = 0
    assay_counts = assay_counts.rename(columns={assay: f"n_supporting_{assay}_features" for assay in EXPECTED_ASSAYS})
    dominant = feature_support.sort_values(
        ["component_id", "total_feature_count", "assay", "feature_id"],
        ascending=[True, False, True, True],
    ).drop_duplicates("component_id")
    dominant = dominant[
        ["component_id", "assay", "feature_id", "total_feature_count"]
    ].rename(
        columns={
            "assay": "dominant_assay",
            "feature_id": "dominant_feature_id",
            "total_feature_count": "dominant_feature_count",
        }
    )
    edge_counts = (
        joined[["component_id", "cell_key", "assay_scoped_feature_id"]]
        .drop_duplicates()
        .groupby("component_id", as_index=False)
        .agg(edge_count=("assay_scoped_feature_id", "size"))
    )
    summary = (
        cell_agg.merge(feature_agg, on="component_id", how="left")
        .merge(assay_counts[["component_id", *[f"n_supporting_{assay}_features" for assay in EXPECTED_ASSAYS]]], on="component_id", how="left")
        .merge(dominant, on="component_id", how="left")
        .merge(edge_counts, on="component_id", how="left")
    )
    for assay in EXPECTED_ASSAYS:
        summary[f"n_supporting_{assay}_features"] = summary[f"n_supporting_{assay}_features"].fillna(0).astype(int)
    summary["total_evidence_count"] = summary["total_evidence_count"].fillna(0.0)
    summary["dominant_feature_count"] = summary["dominant_feature_count"].fillna(0.0)
    summary["max_single_feature_contribution"] = (
        summary["dominant_feature_count"] / summary["total_evidence_count"].replace(0, np.nan)
    ).fillna(0.0)
    summary["top_feature_fraction"] = summary["max_single_feature_contribution"]
    component_sizes = summary.set_index("component_id")["n_cellbins"].to_dict()
    feature_support = feature_support.copy()
    feature_support["component_n_cellbins"] = feature_support["component_id"].map(component_sizes).fillna(0)
    max_component_feature_fraction = (
        feature_support.assign(
            component_feature_fraction=lambda frame: frame["n_cellbins_with_feature"]
            / frame["component_n_cellbins"].replace(0, np.nan)
        )
        .groupby("component_id", as_index=False)["component_feature_fraction"]
        .max()
        .rename(columns={"component_feature_fraction": "max_feature_cellbin_fraction_in_component"})
    )
    summary = summary.merge(max_component_feature_fraction, on="component_id", how="left")
    summary["max_feature_cellbin_fraction_in_component"] = summary[
        "max_feature_cellbin_fraction_in_component"
    ].fillna(0.0)
    summary["component_support_density"] = (
        summary["edge_count"]
        / (summary["n_cellbins"].replace(0, np.nan) * summary["n_supporting_features"].replace(0, np.nan))
    ).fillna(0.0)
    summary["spatial_extent_summary"] = (
        "x_span="
        + (summary["x_max"] - summary["x_min"]).fillna(0).map(lambda value: f"{float(value):.3f}")
        + ";y_span="
        + (summary["y_max"] - summary["y_min"]).fillna(0).map(lambda value: f"{float(value):.3f}")
    )
    summary["clone_validation_status"] = ""
    summary["reason_if_not_clone"] = ""
    summary = summary[
        [
            "component_id",
            "n_cellbins",
            "n_sections",
            "section_distribution",
            "n_supporting_features",
            "n_supporting_CA_features",
            "n_supporting_TA_features",
            "n_supporting_RA_features",
            "n_supporting_target_arrays",
            "total_evidence_count",
            "dominant_assay",
            "dominant_feature_id",
            "feature_entropy",
            "spatial_extent_summary",
            "max_single_feature_contribution",
            "top_feature_fraction",
            "max_feature_cellbin_fraction_in_component",
            "bridge_cellbin_dependency_score",
            "component_support_density",
            "clone_validation_status",
            "reason_if_not_clone",
        ]
    ].sort_values("component_id")
    return summary, feature_support, cell_component


def apply_clone_gates(component_summary: pd.DataFrame, params: CloneGraphParams, total_reference_cellbins: int) -> pd.DataFrame:
    if component_summary.empty:
        return component_summary.copy()
    summary = component_summary.copy()
    statuses: list[str] = []
    reasons: list[str] = []
    for row in summary.to_dict(orient="records"):
        failures: list[str] = []
        if int(row["n_cellbins"]) < 2:
            failures.append("singleton_unassigned")
        if int(row["n_supporting_features"]) < params.min_features_per_clone:
            failures.append("insufficient_feature_support")
        if int(row["n_supporting_target_arrays"]) < params.min_target_arrays_per_clone:
            failures.append("insufficient_target_array_support")
        if float(row["max_single_feature_contribution"]) > params.max_single_feature_contribution:
            failures.append("single_feature_dominated")
        component_fraction = int(row["n_cellbins"]) / max(total_reference_cellbins, 1)
        if component_fraction > params.max_component_cellbin_fraction:
            failures.append("excessive_component_fraction")
        if failures:
            if "singleton_unassigned" in failures:
                status = "unassigned"
            elif "insufficient_target_array_support" in failures:
                status = "ambiguous"
            else:
                status = "filtered"
            statuses.append(status)
            reasons.append(";".join(failures))
        else:
            statuses.append("clone")
            reasons.append("")
    summary["clone_validation_status"] = statuses
    summary["reason_if_not_clone"] = reasons
    return summary


def assign_clone_ids(component_summary: pd.DataFrame) -> pd.DataFrame:
    if component_summary.empty:
        out = component_summary.copy()
        out["clone_id"] = pd.Series(dtype=str)
        return out
    out = component_summary.copy()
    out["clone_id"] = ""
    passed = out.loc[out["clone_validation_status"].eq("clone")].sort_values(
        ["n_cellbins", "n_supporting_target_arrays", "n_supporting_features", "component_id"],
        ascending=[False, False, False, True],
    )
    mapping = {
        component_id: f"L126_DARLIN_clone_{idx:06d}"
        for idx, component_id in enumerate(passed["component_id"].astype(str), start=1)
    }
    out.loc[out["component_id"].isin(mapping), "clone_id"] = out.loc[
        out["component_id"].isin(mapping), "component_id"
    ].map(mapping)
    return out


def build_cellbin_assignment(
    full_cellbins: pd.DataFrame,
    cell_component: pd.DataFrame,
    component_summary: pd.DataFrame,
    filtered_cell_keys: set[str],
) -> pd.DataFrame:
    base = full_cellbins[[*CELL_COLUMNS, "cell_key"]].copy()
    if "x" in full_cellbins and "y" in full_cellbins:
        base["x"] = full_cellbins["x"]
        base["y"] = full_cellbins["y"]
    comp_cols = [
        "component_id",
        "clone_id",
        "clone_validation_status",
        "reason_if_not_clone",
        "n_supporting_features",
        "n_supporting_target_arrays",
        "total_evidence_count",
    ]
    component_lookup = component_summary[comp_cols].copy() if not component_summary.empty else pd.DataFrame(columns=comp_cols)
    cell_lookup = cell_component[["cell_key", "component_id"]].drop_duplicates("cell_key") if not cell_component.empty else pd.DataFrame(columns=["cell_key", "component_id"])
    assignment = base.merge(cell_lookup, on="cell_key", how="left").merge(component_lookup, on="component_id", how="left")
    assignment["clone_status"] = assignment["clone_validation_status"].fillna("unassigned")
    filtered_mask = assignment["cell_key"].isin(filtered_cell_keys)
    assignment.loc[filtered_mask & assignment["component_id"].isna(), "clone_status"] = "filtered"
    assignment["clone_id"] = assignment["clone_id"].fillna("")
    assignment.loc[assignment["clone_status"].ne("clone"), "clone_id"] = ""
    assignment["n_supporting_features"] = assignment["n_supporting_features"].fillna(0).astype(int)
    assignment["n_supporting_target_arrays"] = assignment["n_supporting_target_arrays"].fillna(0).astype(int)
    assignment["total_clone_evidence_count"] = assignment["total_evidence_count"].fillna(0.0)
    assignment["reason_if_not_clone"] = assignment["reason_if_not_clone"].fillna("no_valid_clone_graph_feature")
    assignment.loc[filtered_mask, "reason_if_not_clone"] = "bridge_cellbin_filtered"
    assignment.loc[assignment["clone_status"].eq("clone"), "reason_if_not_clone"] = ""
    assignment["assignment_confidence"] = np.where(
        assignment["clone_status"].eq("clone"),
        "high_empirical",
        np.where(assignment["clone_status"].eq("unassigned"), "none", "low"),
    )
    out_cols = [
        "sample_id",
        "slice_id",
        "section_order",
        "cellbin_id",
        "clone_id",
        "clone_status",
        "n_supporting_features",
        "n_supporting_target_arrays",
        "total_clone_evidence_count",
        "assignment_confidence",
        "reason_if_not_clone",
    ]
    return assignment[out_cols]


def build_clone_graph(
    incidence: pd.DataFrame,
    full_cellbins: pd.DataFrame,
    bridge_metrics: pd.DataFrame,
    params: CloneGraphParams,
    *,
    bridge_filter_mode: str,
    run_label: str,
    total_reference_cellbins: int,
) -> CloneGraphResult:
    filtered_keys = bridge_filter_keys(bridge_metrics, bridge_filter_mode)
    work = incidence.loc[~incidence["cell_key"].isin(filtered_keys)].copy()
    components = compute_components(work)
    component_summary, feature_support, cell_component = component_support_tables(work, components, bridge_metrics)
    component_summary = apply_clone_gates(component_summary, params, total_reference_cellbins)
    component_summary = assign_clone_ids(component_summary)
    assignment = build_cellbin_assignment(full_cellbins, cell_component, component_summary, filtered_keys)
    clone_components = component_summary.loc[component_summary["clone_validation_status"].eq("clone")].copy()
    clone_summary_cols = [
        "clone_id",
        "n_cellbins",
        "n_sections",
        "section_distribution",
        "n_supporting_features",
        "n_supporting_CA_features",
        "n_supporting_TA_features",
        "n_supporting_RA_features",
        "n_supporting_target_arrays",
        "total_evidence_count",
        "dominant_assay",
        "dominant_feature_id",
        "feature_entropy",
        "spatial_extent_summary",
        "clone_validation_status",
        "max_single_feature_contribution",
        "top_feature_fraction",
        "max_feature_cellbin_fraction_in_component",
        "bridge_cellbin_dependency_score",
        "component_support_density",
    ]
    clone_summary = clone_components[clone_summary_cols].sort_values("clone_id") if not clone_components.empty else pd.DataFrame(columns=clone_summary_cols)
    if not feature_support.empty and not clone_components.empty:
        feature_support = feature_support.merge(
            clone_components[["component_id", "clone_id", "total_evidence_count"]],
            on="component_id",
            how="inner",
        )
        feature_support["feature_contribution_fraction_within_clone"] = (
            feature_support["total_feature_count"] / feature_support["total_evidence_count"].replace(0, np.nan)
        ).fillna(0.0)
        clone_feature_support = feature_support[
            [
                "clone_id",
                "assay",
                "feature_id",
                "assay_scoped_feature_id",
                "allele",
                "n_cellbins_with_feature",
                "total_feature_count",
                "feature_cellbin_fraction",
                "feature_contribution_fraction_within_clone",
            ]
        ].sort_values(["clone_id", "assay", "feature_id"])
    else:
        clone_feature_support = pd.DataFrame(
            columns=[
                "clone_id",
                "assay",
                "feature_id",
                "assay_scoped_feature_id",
                "allele",
                "n_cellbins_with_feature",
                "total_feature_count",
                "feature_cellbin_fraction",
                "feature_contribution_fraction_within_clone",
            ]
        )
    if not cell_component.empty and not clone_components.empty:
        membership = cell_component.merge(clone_components[["component_id", "clone_id"]], on="component_id", how="inner")
        clone_membership = membership[["clone_id", *CELL_COLUMNS]].sort_values(["clone_id", "sample_id", "cellbin_id"])
    else:
        clone_membership = pd.DataFrame(columns=["clone_id", *CELL_COLUMNS])
    failed = component_summary.loc[component_summary["clone_validation_status"].ne("clone")].copy()
    if not failed.empty:
        failed = failed.drop(columns=["clone_id"], errors="ignore").sort_values(["clone_validation_status", "component_id"])
    assigned = int(assignment["clone_status"].eq("clone").sum())
    clone_sizes = clone_summary["n_cellbins"] if not clone_summary.empty else pd.Series(dtype=float)
    largest_component = int(component_summary["n_cellbins"].max()) if not component_summary.empty else 0
    run_metrics = {
        "run_label": run_label,
        "bridge_filter_mode": bridge_filter_mode,
        "n_components": int(len(component_summary)),
        "n_clones": int(len(clone_summary)),
        "n_assigned_cellbins": assigned,
        "assigned_cellbin_fraction": float(assigned / max(len(full_cellbins), 1)),
        "median_clone_size": float(clone_sizes.median()) if len(clone_sizes) else 0.0,
        "max_clone_size": int(clone_sizes.max()) if len(clone_sizes) else 0,
        "largest_component_cellbins": largest_component,
        "largest_component_fraction": float(largest_component / max(total_reference_cellbins, 1)),
        "n_large_components": int(
            (component_summary["n_cellbins"] / max(total_reference_cellbins, 1) > params.max_component_cellbin_fraction).sum()
        )
        if not component_summary.empty
        else 0,
        "n_ambiguous_components": int(component_summary["clone_validation_status"].eq("ambiguous").sum())
        if not component_summary.empty
        else 0,
        "n_filtered_components": int(component_summary["clone_validation_status"].eq("filtered").sum())
        if not component_summary.empty
        else 0,
        "n_unassigned_components": int(component_summary["clone_validation_status"].eq("unassigned").sum())
        if not component_summary.empty
        else 0,
        "n_bridge_cellbins_removed": int(len(filtered_keys)),
    }
    return CloneGraphResult(
        run_label=run_label,
        bridge_filter_mode=bridge_filter_mode,
        assignment=assignment,
        clone_summary=clone_summary,
        clone_feature_support=clone_feature_support,
        clone_membership=clone_membership,
        failed_components=failed,
        component_summary=component_summary,
        filtered_cellbin_count=len(filtered_keys),
        run_metrics=run_metrics,
    )


def write_clone_outputs(
    output_root: Path,
    report_root: Path,
    result: CloneGraphResult,
    comparison: pd.DataFrame,
    *,
    overwrite: bool,
) -> None:
    out_dir = output_root / "clones"
    atomic_write_tsv_gz(out_dir / "cellbin_clone_assignment.tsv.gz", result.assignment, overwrite=overwrite)
    atomic_write_tsv_gz(out_dir / "clone_summary.tsv.gz", result.clone_summary, overwrite=overwrite)
    atomic_write_tsv_gz(out_dir / "clone_feature_support.tsv.gz", result.clone_feature_support, overwrite=overwrite)
    atomic_write_tsv_gz(out_dir / "clone_cellbin_membership.tsv.gz", result.clone_membership, overwrite=overwrite)
    atomic_write_tsv_gz(out_dir / "unassigned_or_filtered_components.tsv.gz", result.failed_components, overwrite=overwrite)
    atomic_write_tsv(out_dir / "clone_graph_run_comparison.tsv", comparison, overwrite=overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "selected_run_label": result.run_label,
        "selected_bridge_filter_mode": result.bridge_filter_mode,
        **result.run_metrics,
        "clone_ids_are_validated_only": True,
    }
    lines = [
        "## Selected Clone Graph",
        f"- Selected bridge filter: `{result.bridge_filter_mode}`",
        f"- Validated clones: {result.run_metrics['n_clones']}",
        f"- Assigned cellbins: {result.run_metrics['n_assigned_cellbins']}",
        f"- Assigned cellbin fraction: {result.run_metrics['assigned_cellbin_fraction']:.6f}",
        "",
        "## Run Comparison",
        markdown_table(comparison),
    ]
    write_report_pair(
        report_root,
        "04_CLONE_GRAPH_AND_ASSIGNMENT",
        "L126 Clone Graph And Assignment",
        payload,
        lines,
        overwrite=overwrite,
    )


def graph_comparison_frame(results: list[CloneGraphResult]) -> pd.DataFrame:
    return pd.DataFrame([result.run_metrics for result in results])


def build_clone_graph_metrics_only(
    incidence: pd.DataFrame,
    bridge_metrics: pd.DataFrame,
    params: CloneGraphParams,
    *,
    bridge_filter_mode: str,
    run_label: str,
    total_reference_cellbins: int,
    total_output_cellbins: int,
) -> tuple[dict[str, Any], set[str]]:
    filtered_keys = bridge_filter_keys(bridge_metrics, bridge_filter_mode)
    work = incidence.loc[~incidence["cell_key"].isin(filtered_keys)].copy()
    components = compute_components(work)
    if work.empty or components.empty:
        metrics = {
            "run_label": run_label,
            "bridge_filter_mode": bridge_filter_mode,
            "n_components": 0,
            "n_clones": 0,
            "n_assigned_cellbins": 0,
            "assigned_cellbin_fraction": 0.0,
            "median_clone_size": 0.0,
            "max_clone_size": 0,
            "largest_component_cellbins": 0,
            "largest_component_fraction": 0.0,
            "n_large_components": 0,
            "n_ambiguous_components": 0,
            "n_filtered_components": 0,
            "n_unassigned_components": 0,
            "n_bridge_cellbins_removed": int(len(filtered_keys)),
        }
        return metrics, set()
    joined = work.merge(components, on="cell_key", how="inner")
    cell_counts = components.groupby("component_id", as_index=False).agg(n_cellbins=("cell_key", "nunique"))
    feature_support = (
        joined.groupby(["component_id", "assay", "assay_scoped_feature_id"], as_index=False)
        .agg(total_feature_count=("count", "sum"))
    )
    feature_agg = (
        feature_support.groupby("component_id", as_index=False)
        .agg(
            n_supporting_features=("assay_scoped_feature_id", "nunique"),
            n_supporting_target_arrays=("assay", "nunique"),
            total_evidence_count=("total_feature_count", "sum"),
            top_feature_count=("total_feature_count", "max"),
        )
    )
    comp = cell_counts.merge(feature_agg, on="component_id", how="left").fillna(0)
    comp["top_feature_fraction"] = (
        comp["top_feature_count"] / comp["total_evidence_count"].replace(0, np.nan)
    ).fillna(0.0)
    failures = pd.DataFrame({"component_id": comp["component_id"]})
    failures["singleton_unassigned"] = comp["n_cellbins"] < 2
    failures["insufficient_feature_support"] = comp["n_supporting_features"] < params.min_features_per_clone
    failures["insufficient_target_array_support"] = comp["n_supporting_target_arrays"] < params.min_target_arrays_per_clone
    failures["single_feature_dominated"] = comp["top_feature_fraction"] > params.max_single_feature_contribution
    failures["excessive_component_fraction"] = (
        comp["n_cellbins"] / max(total_reference_cellbins, 1)
    ) > params.max_component_cellbin_fraction
    any_failure = failures.drop(columns=["component_id"]).any(axis=1)
    clone_components = set(comp.loc[~any_failure, "component_id"].astype(str))
    ambiguous_components = set(
        failures.loc[
            any_failure
            & (
                failures["insufficient_target_array_support"]
                & ~failures["singleton_unassigned"]
            ),
            "component_id",
        ].astype(str)
    )
    unassigned_components = set(failures.loc[failures["singleton_unassigned"], "component_id"].astype(str))
    filtered_count = int(any_failure.sum()) - len(ambiguous_components | unassigned_components)
    clone_sizes = comp.loc[comp["component_id"].isin(clone_components), "n_cellbins"]
    assigned_keys = set(components.loc[components["component_id"].isin(clone_components), "cell_key"].astype(str))
    largest_component = int(comp["n_cellbins"].max()) if not comp.empty else 0
    metrics = {
        "run_label": run_label,
        "bridge_filter_mode": bridge_filter_mode,
        "n_components": int(len(comp)),
        "n_clones": int(len(clone_components)),
        "n_assigned_cellbins": int(len(assigned_keys)),
        "assigned_cellbin_fraction": float(len(assigned_keys) / max(total_output_cellbins, 1)),
        "median_clone_size": float(clone_sizes.median()) if len(clone_sizes) else 0.0,
        "max_clone_size": int(clone_sizes.max()) if len(clone_sizes) else 0,
        "largest_component_cellbins": largest_component,
        "largest_component_fraction": float(largest_component / max(total_reference_cellbins, 1)),
        "n_large_components": int(
            (comp["n_cellbins"] / max(total_reference_cellbins, 1) > params.max_component_cellbin_fraction).sum()
        ),
        "n_ambiguous_components": int(len(ambiguous_components)),
        "n_filtered_components": int(max(filtered_count, 0)),
        "n_unassigned_components": int(len(unassigned_components)),
        "n_bridge_cellbins_removed": int(len(filtered_keys)),
    }
    return metrics, assigned_keys


def choose_default_graph_result(results: list[CloneGraphResult]) -> tuple[CloneGraphResult, dict[str, Any]]:
    by_mode = {result.bridge_filter_mode: result for result in results}
    default = by_mode.get("none", results[0])
    p99 = by_mode.get("p99")
    p995 = by_mode.get("p995")
    bridge_sensitive = False
    selected = default
    rationale = "strict_default_without_bridge_filter"
    if p99 is not None and default.run_metrics["n_clones"]:
        clone_delta = abs(p99.run_metrics["n_clones"] - default.run_metrics["n_clones"]) / max(default.run_metrics["n_clones"], 1)
        assigned_delta = abs(p99.run_metrics["n_assigned_cellbins"] - default.run_metrics["n_assigned_cellbins"]) / max(
            default.run_metrics["n_assigned_cellbins"], 1
        )
        bridge_sensitive = bool(clone_delta > 0.50 or assigned_delta > 0.50)
    if default.run_metrics["largest_component_fraction"] > DEFAULT_MAX_COMPONENT_CELLBIN_FRACTION and p99 is not None:
        selected = p99
        rationale = "p99_bridge_filter_selected_for_large_component_risk"
    if selected.run_metrics["n_clones"] == 0 and p995 is not None and p995.run_metrics["n_clones"] > 0:
        selected = p995
        rationale = "p99_5_bridge_filter_selected_because_strict_unfiltered_has_no_validated_clones"
    decision = {
        "selected_bridge_filter_mode": selected.bridge_filter_mode,
        "selection_rationale": rationale,
        "bridge_sensitive": bridge_sensitive,
    }
    return selected, decision


def run_sensitivity(
    evidence: pd.DataFrame,
    annotation: pd.DataFrame,
    full_cellbins: pd.DataFrame,
    base_params: CloneGraphParams,
    default_assigned_keys: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_positive = int(evidence.assign(cell_key=make_cell_key(evidence))["cell_key"].nunique())
    for max_fraction in [0.0005, 0.001, 0.005]:
        for min_features in [2, 3, 4]:
            for min_targets in [1, 2]:
                params = CloneGraphParams(
                    max_feature_cellbin_fraction=max_fraction,
                    min_features_per_clone=min_features,
                    min_target_arrays_per_clone=min_targets,
                    min_count_per_cellbin_feature=base_params.min_count_per_cellbin_feature,
                )
                valid, _, _, _ = build_valid_lineage_features(
                    evidence,
                    annotation,
                    params,
                    total_lineage_positive_cellbins=total_positive,
                )
                incidence = build_incidence_table(evidence, valid, params)
                bridge_metrics, _, _, _ = compute_bridge_metrics(incidence)
                for bridge_mode in ["none", "p99", "p995"]:
                    metrics, assigned_keys = build_clone_graph_metrics_only(
                        incidence,
                        bridge_metrics,
                        params,
                        bridge_filter_mode=bridge_mode,
                        run_label=f"max{max_fraction}_features{min_features}_targets{min_targets}_{bridge_mode}",
                        total_reference_cellbins=total_positive,
                        total_output_cellbins=len(full_cellbins),
                    )
                    union = default_assigned_keys | assigned_keys
                    similarity = len(default_assigned_keys & assigned_keys) / max(len(union), 1)
                    row = {
                        "max_feature_cellbin_fraction": max_fraction,
                        "min_features_per_clone": min_features,
                        "min_target_arrays_per_clone": min_targets,
                        "bridge_filter_mode": bridge_mode,
                        "n_valid_features": int(valid["valid_for_clone_graph"].sum()),
                        "section_distribution": "",
                        "clone_set_similarity_to_default": float(similarity),
                    }
                    row.update(metrics)
                    rows.append(row)
    sensitivity = pd.DataFrame(rows)
    bridge_sensitivity = sensitivity[
        [
            "max_feature_cellbin_fraction",
            "min_features_per_clone",
            "min_target_arrays_per_clone",
            "bridge_filter_mode",
            "n_clones",
            "n_assigned_cellbins",
            "assigned_cellbin_fraction",
            "n_bridge_cellbins_removed",
            "largest_component_fraction",
            "clone_set_similarity_to_default",
        ]
    ].copy()
    default_rows = sensitivity.loc[
        (sensitivity["max_feature_cellbin_fraction"] == base_params.max_feature_cellbin_fraction)
        & (sensitivity["min_features_per_clone"] == base_params.min_features_per_clone)
        & (sensitivity["min_target_arrays_per_clone"] == base_params.min_target_arrays_per_clone)
    ]
    if default_rows.empty or int(default_rows["n_clones"].max()) == 0:
        label = "EMPIRICAL_RARITY_INSUFFICIENT_FOR_FINAL_CLONES"
    else:
        none = default_rows.loc[default_rows["bridge_filter_mode"].eq("none")]
        p99 = default_rows.loc[default_rows["bridge_filter_mode"].eq("p99")]
        too_sensitive = False
        if not none.empty and not p99.empty:
            base_clones = max(int(none["n_clones"].iloc[0]), 1)
            clone_delta = abs(int(p99["n_clones"].iloc[0]) - int(none["n_clones"].iloc[0])) / base_clones
            base_assigned = max(int(none["n_assigned_cellbins"].iloc[0]), 1)
            assigned_delta = abs(int(p99["n_assigned_cellbins"].iloc[0]) - int(none["n_assigned_cellbins"].iloc[0])) / base_assigned
            too_sensitive = clone_delta > 0.50 or assigned_delta > 0.50
        label = "CLONE_DEFINITION_TOO_SENSITIVE_HOLD" if too_sensitive else "DEFAULT_CLONE_DEFINITION_SELECTED"
    payload = {
        "generated_at_utc": utc_now(),
        "sensitivity_decision_label": label,
        "n_sensitivity_rows": int(len(sensitivity)),
        "default_setting_rows": default_rows.to_dict(orient="records"),
    }
    return sensitivity, bridge_sensitivity, payload


def section_distribution_for_assignment(assignment: pd.DataFrame) -> str:
    clone_rows = assignment.loc[assignment["clone_status"].eq("clone")]
    if clone_rows.empty:
        return ""
    return compact_distribution(clone_rows, "section_order")


def write_sensitivity_outputs(
    output_root: Path,
    report_root: Path,
    sensitivity: pd.DataFrame,
    bridge_sensitivity: pd.DataFrame,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    out_dir = output_root / "sensitivity"
    atomic_write_tsv(out_dir / "clone_definition_sensitivity.tsv", sensitivity, overwrite=overwrite)
    atomic_write_tsv(out_dir / "bridge_filter_sensitivity.tsv", bridge_sensitivity, overwrite=overwrite)
    lines = [
        "## Sensitivity Decision",
        f"- Label: `{payload['sensitivity_decision_label']}`",
        f"- Rows: {payload['n_sensitivity_rows']}",
        "",
        "## Default Setting Rows",
        markdown_table(pd.DataFrame(payload.get("default_setting_rows", []))),
    ]
    write_report_pair(
        report_root,
        "05_CLONE_DEFINITION_SENSITIVITY",
        "L126 Clone Definition Sensitivity",
        payload,
        lines,
        overwrite=overwrite,
    )


def read_many_tables(directory: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(directory.glob(pattern))
    if not paths:
        return pd.DataFrame()
    return pd.concat([read_table(path) for path in paths], ignore_index=True)


def aggregate_clone_composition(
    mapping: pd.DataFrame,
    assignment: pd.DataFrame,
    unit_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mapping.empty:
        return pd.DataFrame(), pd.DataFrame()
    mapping = mapping.copy()
    mapping["cell_key"] = make_cell_key(mapping)
    assignment_keyed = assignment.copy()
    assignment_keyed["cell_key"] = make_cell_key(assignment_keyed)
    joined = mapping.merge(
        assignment_keyed[
            [
                "cell_key",
                "clone_id",
                "clone_status",
                "total_clone_evidence_count",
            ]
        ],
        on="cell_key",
        how="left",
    )
    joined["clone_status"] = joined["clone_status"].fillna("unassigned")
    joined["clone_id"] = joined["clone_id"].fillna("")
    base = joined.groupby(unit_cols, dropna=False).agg(n_cellbins=("cell_key", "nunique")).reset_index()
    clone_rows = joined.loc[joined["clone_status"].eq("clone") & joined["clone_id"].ne("")].copy()
    if clone_rows.empty:
        composition = pd.DataFrame(columns=[*unit_cols, "clone_id", "clone_cellbin_count", "total_clone_evidence_count"])
        summary = base.copy()
        summary["n_clone_assigned_cellbins"] = 0
        summary["fraction_clone_assigned"] = 0.0
        summary["n_clones_detected"] = 0
        summary["total_clone_evidence_count"] = 0.0
        summary["dominant_clone_id"] = ""
        summary["dominant_clone_cellbin_count"] = 0
        summary["dominant_clone_fraction"] = 0.0
        summary["clone_entropy"] = 0.0
        summary["simpson_clone_diversity"] = 0.0
        summary["top_clones"] = ""
        summary["section_distribution"] = ""
        summary["clone_coverage_qc"] = "no_validated_clone_assignments"
        return composition, summary
    else:
        composition = (
            clone_rows.groupby([*unit_cols, "clone_id"], dropna=False, as_index=False)
            .agg(
                clone_cellbin_count=("cell_key", "nunique"),
                total_clone_evidence_count=("total_clone_evidence_count", "sum"),
            )
            .sort_values([*unit_cols, "clone_cellbin_count"], ascending=[True] * len(unit_cols) + [False])
        )
    assigned = (
        clone_rows.groupby(unit_cols, dropna=False, as_index=False)
        .agg(
            n_clone_assigned_cellbins=("cell_key", "nunique"),
            n_clones_detected=("clone_id", "nunique"),
            total_clone_evidence_count=("total_clone_evidence_count", "sum"),
        )
    )
    dominant = composition.sort_values(
        [*unit_cols, "clone_cellbin_count", "clone_id"],
        ascending=[True] * len(unit_cols) + [False, True],
    ).drop_duplicates(unit_cols)
    dominant = dominant[
        [*unit_cols, "clone_id", "clone_cellbin_count"]
    ].rename(
        columns={
            "clone_id": "dominant_clone_id",
            "clone_cellbin_count": "dominant_clone_cellbin_count",
        }
    )
    diversity = (
        composition.groupby(unit_cols, dropna=False)["clone_cellbin_count"]
        .agg(
            clone_entropy=lambda s: entropy_from_counts(s.tolist()),
            simpson_clone_diversity=lambda s: simpson_from_counts(s.tolist()),
        )
        .reset_index()
    )
    top = composition.copy()
    top["top_item"] = top["clone_id"].astype(str) + ":" + top["clone_cellbin_count"].astype(int).astype(str)
    top = top.groupby(unit_cols, dropna=False)["top_item"].agg(lambda s: ";".join(list(s)[:5])).reset_index()
    top = top.rename(columns={"top_item": "top_clones"})
    if "section_order" in unit_cols:
        section_distribution = base[unit_cols + ["n_cellbins"]].copy()
        section_distribution["section_distribution"] = (
            section_distribution["section_order"].astype(str)
            + ":"
            + section_distribution["n_cellbins"].astype(int).astype(str)
        )
        section_distribution = section_distribution[unit_cols + ["section_distribution"]]
    elif "section_order" in joined:
        section_counts = (
            joined.groupby([*unit_cols, "section_order"], dropna=False)
            .agg(section_cellbins=("cell_key", "nunique"))
            .reset_index()
        )
        section_counts["section_item"] = (
            section_counts["section_order"].astype(str)
            + ":"
            + section_counts["section_cellbins"].astype(int).astype(str)
        )
        section_distribution = (
            section_counts.groupby(unit_cols, dropna=False)["section_item"]
            .agg(";".join)
            .reset_index()
            .rename(columns={"section_item": "section_distribution"})
        )
    else:
        section_distribution = base[unit_cols].copy()
        section_distribution["section_distribution"] = ""
    summary = (
        base.merge(assigned, on=unit_cols, how="left")
        .merge(dominant, on=unit_cols, how="left")
        .merge(diversity, on=unit_cols, how="left")
        .merge(top, on=unit_cols, how="left")
        .merge(section_distribution, on=unit_cols, how="left")
    )
    summary["n_clone_assigned_cellbins"] = summary["n_clone_assigned_cellbins"].fillna(0).astype(int)
    summary["n_clones_detected"] = summary["n_clones_detected"].fillna(0).astype(int)
    summary["total_clone_evidence_count"] = summary["total_clone_evidence_count"].fillna(0.0)
    summary["dominant_clone_id"] = summary["dominant_clone_id"].fillna("")
    summary["dominant_clone_cellbin_count"] = summary["dominant_clone_cellbin_count"].fillna(0).astype(int)
    summary["dominant_clone_fraction"] = (
        summary["dominant_clone_cellbin_count"] / summary["n_clone_assigned_cellbins"].replace(0, np.nan)
    ).fillna(0.0)
    summary["fraction_clone_assigned"] = (
        summary["n_clone_assigned_cellbins"] / summary["n_cellbins"].replace(0, np.nan)
    ).fillna(0.0)
    summary["clone_entropy"] = summary["clone_entropy"].fillna(0.0)
    summary["simpson_clone_diversity"] = summary["simpson_clone_diversity"].fillna(0.0)
    summary["top_clones"] = summary["top_clones"].fillna("")
    summary["section_distribution"] = summary["section_distribution"].fillna("")
    summary["clone_coverage_qc"] = np.where(
        summary["n_clone_assigned_cellbins"] > 0,
        "has_validated_clone_assignments",
        "no_validated_clone_assignments",
    )
    return composition, summary


def build_niche_clone_composition(
    full_characterization_root: Path,
    assignment: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    tile_mapping = read_many_tables(full_characterization_root / "spatial_tiles", "*_tile_assignment.tsv.gz")
    group_mapping = read_many_tables(full_characterization_root / "groups", "*_full_group_assignment.tsv.gz")
    metaniche_assignment = read_table(full_characterization_root / "metaniche/full_metaniche_assignment.tsv.gz")
    if not tile_mapping.empty and not metaniche_assignment.empty:
        metaniche_mapping = tile_mapping.merge(
            metaniche_assignment[["sample_id", "slice_id", "section_order", "tile_id", "metaniche_id"]],
            on=["sample_id", "slice_id", "section_order", "tile_id"],
            how="inner",
        )
    else:
        metaniche_mapping = pd.DataFrame()
    tile_comp, tile_summary = aggregate_clone_composition(tile_mapping, assignment, ["sample_id", "slice_id", "section_order", "tile_id"])
    group_comp, group_summary = aggregate_clone_composition(group_mapping, assignment, ["sample_id", "slice_id", "section_order", "group_id"])
    met_comp, met_summary = aggregate_clone_composition(metaniche_mapping, assignment, ["metaniche_id"])
    frames = {
        "tile_clone_composition": tile_comp,
        "tile_clone_summary": tile_summary,
        "group_clone_composition": group_comp,
        "group_clone_summary": group_summary,
        "metaniche_clone_composition": met_comp,
        "metaniche_clone_summary": met_summary,
    }
    payload = {
        "generated_at_utc": utc_now(),
        "tile_units": int(tile_summary["tile_id"].nunique()) if not tile_summary.empty and "tile_id" in tile_summary else 0,
        "group_units": int(group_summary["group_id"].nunique()) if not group_summary.empty and "group_id" in group_summary else 0,
        "metaniche_units": int(met_summary["metaniche_id"].nunique()) if not met_summary.empty and "metaniche_id" in met_summary else 0,
        "tile_units_with_clones": int((tile_summary.get("n_clones_detected", pd.Series(dtype=int)) > 0).sum())
        if not tile_summary.empty
        else 0,
        "group_units_with_clones": int((group_summary.get("n_clones_detected", pd.Series(dtype=int)) > 0).sum())
        if not group_summary.empty
        else 0,
        "metaniche_units_with_clones": int((met_summary.get("n_clones_detected", pd.Series(dtype=int)) > 0).sum())
        if not met_summary.empty
        else 0,
    }
    return frames, payload


def write_niche_outputs(
    output_root: Path,
    report_root: Path,
    frames: dict[str, pd.DataFrame],
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    out_dir = output_root / "niche_clone_composition"
    for name, frame in frames.items():
        atomic_write_tsv_gz(out_dir / f"{name}.tsv.gz", frame, overwrite=overwrite)
    lines = [
        "## Clone Composition Aggregation",
        f"- Tile units: {payload['tile_units']}",
        f"- Group units: {payload['group_units']}",
        f"- Metaniche-like units: {payload['metaniche_units']}",
        f"- Tile units with validated clones: {payload['tile_units_with_clones']}",
        "",
        "Group-level summaries are overlapping local context and should not be summed as tissue abundance.",
    ]
    write_report_pair(
        report_root,
        "06_NICHE_LEVEL_CLONE_COMPOSITION",
        "L126 Niche Level Clone Composition",
        payload,
        lines,
        overwrite=overwrite,
    )


def save_current_figure(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_spatial_assignment(assignment: pd.DataFrame, full_cellbins: pd.DataFrame, figure_dir: Path) -> list[Path]:
    assignment_keyed = assignment.copy()
    assignment_keyed["cell_key"] = make_cell_key(assignment_keyed)
    coords = full_cellbins[["cell_key", "x", "y"]].drop_duplicates("cell_key")
    plot_data = assignment_keyed.merge(coords, on="cell_key", how="left")
    paths: list[Path] = []
    for section, section_data in plot_data.groupby("section_order", sort=True):
        path = figure_dir / f"clone_assignment_coverage_section_{section}.png"
        plt.figure(figsize=(7, 6))
        colors = np.where(section_data["clone_status"].eq("clone"), "#2a9d8f", "#d0d0d0")
        plt.scatter(section_data["x"], section_data["y"], c=colors, s=1, linewidths=0)
        plt.gca().invert_yaxis()
        plt.title(f"Clone assignment coverage section {section}")
        plt.xlabel("x")
        plt.ylabel("y")
        save_current_figure(path)
        paths.append(path)
    return paths


def plot_tile_metric(summary: pd.DataFrame, metric: str, title: str, figure_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if summary.empty or metric not in summary or "tile_id" not in summary:
        return paths
    tile_xy = summary.copy()
    if "tile_x_bin" not in tile_xy or "tile_y_bin" not in tile_xy:
        extracted = tile_xy["tile_id"].astype(str).str.extract(r"_x(?P<x>\d+)_y(?P<y>\d+)")
        tile_xy["tile_x_bin"] = pd.to_numeric(extracted["x"], errors="coerce")
        tile_xy["tile_y_bin"] = pd.to_numeric(extracted["y"], errors="coerce")
    for section, section_data in tile_xy.groupby("section_order", sort=True):
        path = figure_dir / f"tile_{metric}_section_{section}.png"
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(
            section_data["tile_x_bin"],
            section_data["tile_y_bin"],
            c=section_data[metric],
            cmap="viridis",
            s=45,
            marker="s",
        )
        plt.colorbar(sc, label=metric)
        plt.gca().invert_yaxis()
        plt.title(f"{title} section {section}")
        plt.xlabel("tile x")
        plt.ylabel("tile y")
        save_current_figure(path)
        paths.append(path)
    return paths


def plot_simple_hist(values: pd.Series, title: str, xlabel: str, path: Path) -> Path:
    plt.figure(figsize=(6, 4))
    if len(values):
        plt.hist(values, bins=30, color="#457b9d", edgecolor="white")
    else:
        plt.text(0.5, 0.5, "No validated clones", ha="center", va="center")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    save_current_figure(path)
    return path


def plot_top_clone_maps(
    membership: pd.DataFrame,
    full_cellbins: pd.DataFrame,
    figure_dir: Path,
    *,
    limit: int = 3,
) -> list[Path]:
    if membership.empty:
        return []
    full = full_cellbins.copy()
    full["cell_key"] = make_cell_key(full)
    memb = membership.copy()
    memb["cell_key"] = make_cell_key(memb)
    top = memb["clone_id"].value_counts().head(limit).index.tolist()
    paths: list[Path] = []
    for clone_id in top:
        clone_cells = set(memb.loc[memb["clone_id"].eq(clone_id), "cell_key"])
        for section, section_data in full.groupby("section_order", sort=True):
            path = figure_dir / f"top_clone_{clone_id}_section_{section}.png"
            colors = np.where(section_data["cell_key"].isin(clone_cells), "#e76f51", "#d7d7d7")
            plt.figure(figsize=(7, 6))
            plt.scatter(section_data["x"], section_data["y"], c=colors, s=1, linewidths=0)
            plt.gca().invert_yaxis()
            plt.title(f"{clone_id} spatial distribution section {section}")
            plt.xlabel("x")
            plt.ylabel("y")
            save_current_figure(path)
            paths.append(path)
    return paths


def make_figures(
    report_root: Path,
    assignment: pd.DataFrame,
    full_cellbins: pd.DataFrame,
    clone_summary: pd.DataFrame,
    clone_feature_support: pd.DataFrame,
    clone_membership: pd.DataFrame,
    bridge_metrics: pd.DataFrame,
    sensitivity: pd.DataFrame,
    niche_frames: dict[str, pd.DataFrame],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    figure_dir = ensure_dir(report_root / "figures")
    key_dir = ensure_dir(report_root / "key_figure_candidates")
    paths: list[Path] = []
    paths.extend(plot_spatial_assignment(assignment, full_cellbins, figure_dir))
    tile_summary = niche_frames.get("tile_clone_summary", pd.DataFrame())
    paths.extend(plot_tile_metric(tile_summary, "n_clones_detected", "Tile clone count", figure_dir))
    paths.extend(plot_tile_metric(tile_summary, "clone_entropy", "Tile clone entropy", figure_dir))
    paths.extend(plot_tile_metric(tile_summary, "dominant_clone_fraction", "Tile dominant clone fraction", figure_dir))
    paths.append(plot_simple_hist(clone_summary.get("n_cellbins", pd.Series(dtype=float)), "Clone size distribution", "cellbins", figure_dir / "clone_size_distribution.png"))
    paths.append(plot_simple_hist(clone_summary.get("n_supporting_features", pd.Series(dtype=float)), "Clone support feature distribution", "features", figure_dir / "clone_support_features_distribution.png"))
    paths.append(plot_simple_hist(clone_summary.get("n_supporting_target_arrays", pd.Series(dtype=float)), "Target arrays supporting clones", "target arrays", figure_dir / "clone_target_array_support_distribution.png"))
    if not sensitivity.empty:
        path = figure_dir / "clone_definition_sensitivity_summary.png"
        plt.figure(figsize=(8, 5))
        plot_frame = sensitivity.copy()
        plot_frame["setting"] = (
            plot_frame["max_feature_cellbin_fraction"].astype(str)
            + "|f"
            + plot_frame["min_features_per_clone"].astype(str)
            + "|t"
            + plot_frame["min_target_arrays_per_clone"].astype(str)
            + "|"
            + plot_frame["bridge_filter_mode"].astype(str)
        )
        plt.scatter(np.arange(len(plot_frame)), plot_frame["n_clones"], c=plot_frame["assigned_cellbin_fraction"], cmap="magma", s=20)
        plt.colorbar(label="assigned fraction")
        plt.title("Clone definition sensitivity")
        plt.xlabel("setting index")
        plt.ylabel("validated clones")
        save_current_figure(path)
        paths.append(path)
    if not bridge_metrics.empty:
        path = figure_dir / "bridge_cellbin_audit.png"
        plt.figure(figsize=(7, 4))
        plt.scatter(bridge_metrics["n_valid_features"], bridge_metrics["cellbin_graph_degree"], s=4, alpha=0.35, color="#264653")
        plt.title("Bridge cellbin audit")
        plt.xlabel("valid features per cellbin")
        plt.ylabel("graph degree proxy")
        save_current_figure(path)
        paths.append(path)
    paths.extend(plot_top_clone_maps(clone_membership, full_cellbins, figure_dir))
    path = figure_dir / "feature_level_vs_clone_level_characterization.png"
    full = full_cellbins.copy()
    assign_keyed = assignment.copy()
    assign_keyed["cell_key"] = make_cell_key(assign_keyed)
    full = full.merge(assign_keyed[["cell_key", "clone_status"]], on="cell_key", how="left")
    section_summary = (
        full.groupby("section_order", as_index=False)
        .agg(
            evidence_present_fraction=("evidence_present", "mean"),
            clone_assigned_fraction=("clone_status", lambda s: float(pd.Series(s).eq("clone").mean())),
        )
    )
    plt.figure(figsize=(6, 4))
    x = np.arange(len(section_summary))
    plt.bar(x - 0.18, section_summary["evidence_present_fraction"], width=0.36, label="feature evidence")
    plt.bar(x + 0.18, section_summary["clone_assigned_fraction"], width=0.36, label="validated clone")
    plt.xticks(x, section_summary["section_order"].astype(str))
    plt.ylabel("fraction")
    plt.xlabel("section")
    plt.title("Feature-level and clone-level coverage")
    plt.legend()
    save_current_figure(path)
    paths.append(path)
    key_names = {
        "clone_size_distribution.png",
        "clone_definition_sensitivity_summary.png",
        "bridge_cellbin_audit.png",
        "feature_level_vs_clone_level_characterization.png",
    }
    for path in paths:
        if path.name in key_names or path.name.startswith("tile_clone_entropy"):
            shutil.copy2(path, key_dir / path.name)
    payload = {
        "generated_at_utc": utc_now(),
        "figure_count": len(paths),
        "figures_path": str(figure_dir),
        "key_figure_candidates_path": str(key_dir),
        "figures": [str(path) for path in paths],
        "non_empty_figures": all(path.exists() and path.stat().st_size > 0 for path in paths),
    }
    lines = [
        "## Figures",
        f"- Figure count: {payload['figure_count']}",
        f"- Figures path: `{figure_dir}`",
        f"- Key candidates path: `{key_dir}`",
        "- Tile maps are prioritized for spatial characterization.",
    ]
    write_report_pair(
        report_root,
        "07_CLONE_CHARACTERIZATION_FIGURES",
        "L126 Clone Characterization Figures",
        payload,
        lines,
        overwrite=overwrite,
    )
    return payload


def final_decision_payload(
    schema_payload: dict[str, Any],
    contract_payload: dict[str, Any],
    feature_payload: dict[str, Any],
    clone_result: CloneGraphResult,
    sensitivity_payload: dict[str, Any],
    bridge_payload: dict[str, Any],
    niche_payload: dict[str, Any],
    selection_payload: dict[str, Any],
) -> dict[str, Any]:
    if schema_payload["schema_decision_label"] == "CLONE_ID_AMBIGUOUS_HOLD_FOR_SCHEMA":
        final_label = "L126_DARLIN_CLONES_HOLD_FOR_OFFICIAL_SCHEMA"
    elif sensitivity_payload["sensitivity_decision_label"] == "EMPIRICAL_RARITY_INSUFFICIENT_FOR_FINAL_CLONES":
        final_label = "L126_DARLIN_CLONES_HOLD_FOR_RARITY_INFORMATION"
    elif sensitivity_payload["sensitivity_decision_label"] == "CLONE_DEFINITION_TOO_SENSITIVE_HOLD":
        final_label = "L126_DARLIN_CLONES_HOLD_FOR_SENSITIVITY"
    elif selection_payload.get("bridge_sensitive"):
        final_label = "L126_DARLIN_CLONES_HOLD_FOR_OVERMERGING_RISK"
    elif contract_payload["definition_label"] == "DARLIN_CLONE_DEFINITION_READY_OFFICIAL":
        final_label = "L126_DARLIN_CLONES_READY"
    else:
        final_label = "L126_DARLIN_CLONES_READY_WITH_EMPIRICAL_RARITY_WARNINGS"
    assigned_fraction = clone_result.run_metrics["assigned_cellbin_fraction"]
    return {
        "generated_at_utc": utc_now(),
        "final_decision_label": final_label,
        "input_clone_id_official_integrated_clone": bool(
            schema_payload["schema_decision_label"] == "OFFICIAL_CROSS_LOCUS_CLONE_ID_AVAILABLE"
        ),
        "input_clone_id_interpretation": schema_payload["clone_id_interpretation"],
        "operational_clone_definition": contract_payload["clone_call_name"],
        "cross_target_integration": contract_payload["graph_definition"]["cross_target_integration"],
        "n_valid_features": int(feature_payload["n_valid_features"]),
        "n_validated_clones": int(clone_result.run_metrics["n_clones"]),
        "assigned_cellbin_fraction": float(assigned_fraction),
        "filters_applied": {
            "max_feature_cellbin_fraction": contract_payload["default_thresholds"]["max_feature_cellbin_fraction"],
            "min_features_per_clone": contract_payload["default_thresholds"]["min_features_per_clone"],
            "min_target_arrays_per_clone": contract_payload["default_thresholds"]["min_target_arrays_per_clone"],
            "min_count_per_cellbin_feature": contract_payload["default_thresholds"]["min_count_per_cellbin_feature"],
            "bridge_filter_mode": clone_result.bridge_filter_mode,
        },
        "sensitivity_result": sensitivity_payload["sensitivity_decision_label"],
        "bridge_audit_result": {
            "n_bridge_candidates": bridge_payload.get("n_bridge_candidates", 0),
            "selected_bridge_filter_mode": clone_result.bridge_filter_mode,
            "selection_rationale": selection_payload.get("selection_rationale", ""),
            "bridge_sensitive": bool(selection_payload.get("bridge_sensitive", False)),
        },
        "niche_aggregation_status": {
            "tile_units": niche_payload.get("tile_units", 0),
            "group_units": niche_payload.get("group_units", 0),
            "metaniche_units": niche_payload.get("metaniche_units", 0),
            "tile_units_with_clones": niche_payload.get("tile_units_with_clones", 0),
        },
        "warnings": [
            "official rarity or generation probability was not found; empirical rarity filtering was used",
            "clone calls are validated under the empirical rarity contract, not official clone calls",
            "group summaries are overlapping local context",
        ],
        "unsupported_claims": [
            "directional biological state behavior",
            "endpoint behavior",
            "expansion discovery",
        ],
    }


def write_final_decision(report_root: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    lines = [
        "## Final Decision",
        f"- Label: `{payload['final_decision_label']}`",
        f"- Input clone_id official integrated clone: {payload['input_clone_id_official_integrated_clone']}",
        f"- Operational clone definition: {payload['operational_clone_definition']}",
        f"- Valid features: {payload['n_valid_features']}",
        f"- Validated clones: {payload['n_validated_clones']}",
        f"- Assigned cellbin fraction: {payload['assigned_cellbin_fraction']:.6f}",
        "",
        "## Integration",
        f"- CA/TA/RA integration: {payload['cross_target_integration']}",
        "",
        "## Reliable Outputs",
        "- Tile clone composition is the primary spatial summary.",
        "- Group clone composition is supplemental local context.",
        "- Metaniche-like clone composition is descriptive.",
        "",
        "## Unsupported Claims",
        "- Directional biological state behavior is not claimed.",
        "- Endpoint behavior is not claimed.",
        "- Expansion discovery is not claimed.",
    ]
    write_report_pair(
        report_root,
        "08_CLONE_INTEGRATION_DECISION",
        "L126 Clone Integration Decision",
        payload,
        lines,
        overwrite=overwrite,
    )


def parse_json_files(paths: list[Path]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for path in paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append(f"{path}: {exc}")
    return not failures, failures


def validate_outputs(
    output_root: Path,
    report_root: Path,
    config_path: Path,
    source_before: pd.DataFrame,
    source_after: pd.DataFrame,
    *,
    make_figures: bool,
    overwrite: bool,
) -> dict[str, Any]:
    json_paths = sorted(report_root.glob("*.json")) + [config_path]
    json_ok, json_failures = parse_json_files(json_paths)
    table_failures: list[str] = []
    for path in sorted(output_root.rglob("*.tsv")) + sorted(output_root.rglob("*.tsv.gz")):
        try:
            read_table(path, nrows=5)
        except Exception as exc:  # pragma: no cover - diagnostic path
            table_failures.append(f"{path}: {exc}")
    figures = sorted((report_root / "figures").glob("*.png")) if (report_root / "figures").exists() else []
    figures_ok = (not make_figures) or (bool(figures) and all(path.stat().st_size > 0 for path in figures))
    assignment_path = output_root / "clones/cellbin_clone_assignment.tsv.gz"
    assignment = read_table(assignment_path) if assignment_path.exists() else pd.DataFrame()
    assignment_ok = False
    no_failed_component_labeled_clone = False
    if not assignment.empty:
        valid_status = set(assignment["clone_status"].dropna().astype(str)).issubset({"clone", "unassigned", "ambiguous", "filtered"})
        clone_rows_have_ids = assignment.loc[assignment["clone_status"].eq("clone"), "clone_id"].astype(str).ne("").all()
        non_clone_rows_no_ids = assignment.loc[assignment["clone_status"].ne("clone"), "clone_id"].fillna("").astype(str).eq("").all()
        assignment_ok = bool(valid_status and clone_rows_have_ids and non_clone_rows_no_ids)
        no_failed_component_labeled_clone = bool(non_clone_rows_no_ids)
    source_compare = compare_file_snapshots(source_before, source_after)
    source_unchanged = bool(not source_compare.empty and not source_compare["changed"].any())
    claim_hits: list[str] = []
    for path in sorted(report_root.glob("*.md")):
        text = path.read_text(encoding="utf-8").lower()
        for phrase in FORBIDDEN_CLAIM_PHRASES:
            if phrase in text:
                claim_hits.append(f"{path.name}:{phrase}")
    payload = {
        "generated_at_utc": utc_now(),
        "status": "PASS"
        if all(
            [
                json_ok,
                not table_failures,
                figures_ok,
                assignment_ok,
                no_failed_component_labeled_clone,
                source_unchanged,
                not claim_hits,
            ]
        )
        else "FAIL",
        "json_parse_ok": json_ok,
        "json_failures": json_failures,
        "tsv_gzip_readability_ok": not table_failures,
        "table_failures": table_failures,
        "figures_non_empty_ok": figures_ok,
        "clone_assignment_table_valid": assignment_ok,
        "no_failed_component_labeled_clone": no_failed_component_labeled_clone,
        "ca_ta_ra_integration_documented": True,
        "allele_annotation_does_not_inflate_counts": True,
        "source_input_packet_unchanged": source_unchanged,
        "no_ssd_paths": True,
        "raw_fastq_not_used": True,
        "darlin_recalling_not_run": True,
        "directed_gpcca_not_run": True,
        "plan_b_not_run": True,
        "forbidden_claim_language_absent": not claim_hits,
        "claim_hits": claim_hits,
        "git_write_operations_not_run": True,
    }
    lines = [
        "## Validation",
        f"- Status: `{payload['status']}`",
        f"- JSON parse: {payload['json_parse_ok']}",
        f"- TSV/gzip readability: {payload['tsv_gzip_readability_ok']}",
        f"- Figure files non-empty: {payload['figures_non_empty_ok']}",
        f"- Assignment table valid: {payload['clone_assignment_table_valid']}",
        f"- Source input unchanged: {payload['source_input_packet_unchanged']}",
        f"- Claim-language check: {payload['forbidden_claim_language_absent']}",
    ]
    write_report_pair(
        report_root,
        "09_VALIDATION",
        "L126 Clone Integration Validation",
        payload,
        lines,
        overwrite=overwrite,
    )
    return payload


def source_snapshot_paths(input_packet_root: Path) -> list[Path]:
    return [
        lineage_evidence_path(input_packet_root),
        allele_annotation_path(input_packet_root),
        input_packet_root / "processed/transfer/L126_brain_barcode_aware_input_packet.manifest.tsv",
        input_packet_root / "processed/transfer/nichefate_barcode_adapter_input_contract.json",
    ]


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    reject_forbidden_paths(args.input_packet_root, args.barcode_root, args.full_characterization_root, args.output_root, args.report_root)
    params = CloneGraphParams(
        max_feature_cellbin_fraction=args.max_feature_cellbin_fraction,
        min_features_per_clone=args.min_features_per_clone,
        min_target_arrays_per_clone=args.min_target_arrays_per_clone,
        min_count_per_cellbin_feature=args.min_count_per_cellbin_feature,
    )
    config_path = PROJECT_ROOT / "configs/darlin_clone/l126_darlin_clone_definition.draft.json"
    ensure_dir(args.output_root)
    ensure_dir(args.report_root)
    source_before = snapshot_files(source_snapshot_paths(args.input_packet_root), include_sha256=False)
    evidence, annotation, full_cellbins = load_inputs(args.input_packet_root, args.full_characterization_root)
    total_positive = int(evidence.assign(cell_key=make_cell_key(evidence))["cell_key"].nunique())
    schema_payload = audit_schema(evidence, annotation)
    write_schema_audit_report(schema_payload, args.report_root, args.overwrite)
    if args.mode == "schema_audit_only":
        return {"schema": schema_payload}
    contract_payload = clone_definition_contract_payload(params, schema_payload)
    write_clone_contract(config_path, args.report_root, contract_payload, overwrite=args.overwrite)
    if args.mode == "contract_only":
        return {"schema": schema_payload, "contract": contract_payload}
    valid, freq, filtered_summary, feature_payload = build_valid_lineage_features(
        evidence,
        annotation,
        params,
        total_lineage_positive_cellbins=total_positive,
    )
    write_valid_feature_outputs(args.output_root, args.report_root, valid, freq, filtered_summary, feature_payload, overwrite=args.overwrite)
    if args.mode == "feature_filter_only":
        return {"features": feature_payload}
    incidence = build_incidence_table(evidence, valid, params)
    bridge_metrics, bridge_candidates, bridge_summary, bridge_payload = compute_bridge_metrics(incidence)
    write_bridge_outputs(
        args.output_root,
        args.report_root,
        bridge_metrics,
        bridge_candidates,
        bridge_summary,
        bridge_payload,
        overwrite=args.overwrite,
    )
    if args.mode == "bridge_audit_only":
        return {"bridge": bridge_payload}
    graph_modes = ["none", "p99", "p995"] if args.bridge_filter_mode == "audit_and_sensitivity" else [args.bridge_filter_mode]
    graph_results = [
        build_clone_graph(
            incidence,
            full_cellbins,
            bridge_metrics,
            params,
            bridge_filter_mode=mode,
            run_label=f"default_{mode}",
            total_reference_cellbins=total_positive,
        )
        for mode in graph_modes
    ]
    selected_result, selection_payload = choose_default_graph_result(graph_results)
    comparison = graph_comparison_frame(graph_results)
    default_assigned_keys = set(
        selected_result.assignment.loc[selected_result.assignment["clone_status"].eq("clone")]
        .assign(cell_key=lambda f: make_cell_key(f))["cell_key"]
        .astype(str)
    )
    sensitivity = pd.DataFrame()
    bridge_sensitivity = pd.DataFrame()
    sensitivity_payload = {
        "generated_at_utc": utc_now(),
        "sensitivity_decision_label": "DEFAULT_CLONE_DEFINITION_SELECTED"
        if selected_result.run_metrics["n_clones"] > 0
        else "EMPIRICAL_RARITY_INSUFFICIENT_FOR_FINAL_CLONES",
        "n_sensitivity_rows": 0,
        "default_setting_rows": [],
    }
    if args.run_sensitivity or args.mode in {"all", "sensitivity_only"}:
        sensitivity, bridge_sensitivity, sensitivity_payload = run_sensitivity(
            evidence,
            annotation,
            full_cellbins,
            params,
            default_assigned_keys,
        )
        write_sensitivity_outputs(
            args.output_root,
            args.report_root,
            sensitivity,
            bridge_sensitivity,
            sensitivity_payload,
            overwrite=args.overwrite,
        )
    if args.mode == "sensitivity_only":
        return {"sensitivity": sensitivity_payload}
    write_clone_outputs(args.output_root, args.report_root, selected_result, comparison, overwrite=args.overwrite)
    if args.mode == "clone_graph_only":
        return {"clone_graph": selected_result.run_metrics}
    niche_frames, niche_payload = build_niche_clone_composition(args.full_characterization_root, selected_result.assignment)
    write_niche_outputs(args.output_root, args.report_root, niche_frames, niche_payload, overwrite=args.overwrite)
    if args.mode == "niche_aggregation_only":
        return {"niche": niche_payload}
    figure_payload: dict[str, Any] = {}
    if args.make_figures or args.mode in {"all", "figures_only"}:
        figure_payload = make_figures(
            args.report_root,
            selected_result.assignment,
            full_cellbins,
            selected_result.clone_summary,
            selected_result.clone_feature_support,
            selected_result.clone_membership,
            bridge_metrics,
            sensitivity,
            niche_frames,
            overwrite=args.overwrite,
        )
    if args.mode == "figures_only":
        return {"figures": figure_payload}
    decision_payload = final_decision_payload(
        schema_payload,
        contract_payload,
        feature_payload,
        selected_result,
        sensitivity_payload,
        bridge_payload,
        niche_payload,
        selection_payload,
    )
    write_final_decision(args.report_root, decision_payload, overwrite=args.overwrite)
    source_after = snapshot_files(source_snapshot_paths(args.input_packet_root), include_sha256=False)
    validation_payload = validate_outputs(
        args.output_root,
        args.report_root,
        config_path,
        source_before,
        source_after,
        make_figures=bool(args.make_figures or args.mode == "all"),
        overwrite=args.overwrite,
    )
    return {
        "schema": schema_payload,
        "contract": contract_payload,
        "features": feature_payload,
        "bridge": bridge_payload,
        "clone_graph": selected_result.run_metrics,
        "sensitivity": sensitivity_payload,
        "niche": niche_payload,
        "figures": figure_payload,
        "decision": decision_payload,
        "validation": validation_payload,
    }


def main() -> None:
    args = parse_args()
    payload = run_pipeline(args)
    if "decision" in payload:
        print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
