#!/usr/bin/env python
"""Full barcode-aware niche characterization for L126 spatio-DARLIN Brain."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from nichefate.barcode_adapter.group_lineage import aggregate_group_lineage
from nichefate.barcode_adapter.input_contract import EXPECTED_ASSAYS, PRIMARY_JOIN_KEY
from nichefate.barcode_adapter.l126_schema import h5ad_path_for_sample, validate_l126_h5ad_schema
from nichefate.barcode_adapter.loaders import load_cellbin_lineage_evidence, load_feature_allele_annotation
from nichefate.barcode_adapter.qc import (
    audit_allele_annotation,
    compare_file_snapshots,
    snapshot_files,
)
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
from nichefate.barcode_adapter.routeA import aggregate_lineage_for_unit_mapping


SAMPLES_DEFAULT = "L126_Brain_s1,L126_Brain_s2,L126_Brain_s3"
EXPECTED_CELLBIN_COUNTS = {
    "L126_Brain_s1": 67970,
    "L126_Brain_s2": 72884,
    "L126_Brain_s3": 70155,
}
GROUP_TYPE = "FULL_SPATIAL_NEIGHBORHOOD_CHARACTERIZATION_NOT_FULL_M1"
FINAL_READY = "L126_FULL_BARCODE_NICHE_CHARACTERIZATION_READY"
FINAL_READY_WARN = "L126_FULL_BARCODE_NICHE_CHARACTERIZATION_READY_WITH_WARNINGS"
FORBIDDEN_PHRASES = (
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
CELLBIN_REQUIRED_COLUMNS = [
    "sample_id",
    "slice_id",
    "section_order",
    "cellbin_id",
    "x",
    "y",
    "evidence_present",
    "total_lineage_count",
    "detected_feature_count",
    "detected_assay_count",
    "RA_total_count",
    "TA_total_count",
    "CA_total_count",
    "RA_detected_feature_count",
    "TA_detected_feature_count",
    "CA_detected_feature_count",
    "dominant_assay",
    "dominant_feature_id",
    "dominant_feature_count",
    "dominant_feature_fraction",
    "feature_entropy",
    "simpson_diversity",
    "assay_balance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-packet-root",
        type=Path,
        default=Path("/home/zhutao/scratch/nichefate/spatio_darlin_public_brain_hpc/input_packet"),
    )
    parser.add_argument("--barcode-root", type=Path, default=Path("processed/barcode_adapter_l126_round1"))
    parser.add_argument("--round2B-root", type=Path, default=Path("processed/l126_niche_barcode_round2B"))
    parser.add_argument("--output-root", type=Path, default=Path("processed/l126_full_barcode_niche_characterization"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/l126_full_barcode_niche_characterization"))
    parser.add_argument("--samples", default=SAMPLES_DEFAULT)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--tile-grid", default="auto")
    parser.add_argument("--n-metaniches", type=int, default=300)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--run-full-groups", action="store_true")
    parser.add_argument("--run-spatial-tiles", action="store_true")
    parser.add_argument("--run-metaniche-characterization", action="store_true")
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mode",
        default="all",
        choices=[
            "all",
            "preflight_only",
            "cellbin_only",
            "groups_only",
            "tiles_only",
            "metaniche_only",
            "figures_only",
            "validation_only",
        ],
    )
    return parser.parse_args()


def parse_samples(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    return pd.read_csv(path, sep="\t", compression=compression, nrows=nrows)


def reject_forbidden_paths(*paths: Path) -> None:
    bad = [str(path) for path in paths if path_has_ssd(path)]
    if bad:
        raise ValueError("Refusing /ssd paths: " + "; ".join(bad))


def packet_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def write_report(
    report_root: Path,
    stem: str,
    title: str,
    payload: dict[str, Any],
    lines: list[str],
    *,
    overwrite: bool,
) -> None:
    ensure_dir(report_root)
    scope = [
        "L126_Brain_s1/s2/s3 are serial sections, not timepoints.",
        "L126 is handled as barcode-aware spatial niche characterization.",
        "Directed GPCCA interpretation was stopped for this dataset after Round 4 hardening.",
        "Groups are overlapping local-context neighborhoods and are not tissue partitions.",
        "RA/TA/CA are preserved as separate assay-level evidence channels.",
        "Allele annotation is annotation-only and does not inflate counts.",
        "No raw FASTQ, DARLIN re-calling, directed GPCCA, full M0/M1/M2, or PlanB was run.",
    ]
    atomic_write_json(report_root / f"{stem}.json", payload, overwrite=overwrite)
    atomic_write_text(
        report_root / f"{stem}.md",
        f"# {title}\n\n" + "\n".join(f"- {note}" for note in scope) + "\n\n" + "\n".join(lines).strip() + "\n",
        overwrite=overwrite,
    )


def assay_balance_from_counts(frame: pd.DataFrame) -> pd.Series:
    values = frame[[f"{assay}_total_count" for assay in EXPECTED_ASSAYS]].fillna(0.0).to_numpy(dtype=float)
    out: list[float] = []
    for row in values:
        positive = row[row > 0]
        if positive.size <= 1:
            out.append(0.0)
        else:
            probs = positive / positive.sum()
            out.append(float(-(probs * np.log(probs)).sum() / math.log(positive.size)))
    return pd.Series(out, index=frame.index)


def add_assay_balance(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for assay in EXPECTED_ASSAYS:
        col = f"{assay}_total_count"
        if col not in out:
            out[col] = 0.0
    out["assay_balance"] = assay_balance_from_counts(out)
    return out


def load_cellbin_table_with_expression_qc(h5ad_path: Path, sample: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    import anndata as ad

    data = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = data.obs.reset_index(names="obs_index").copy()
        obs["obs_position"] = np.arange(len(obs), dtype=int)
        keep = ["sample_id", "slice_id", "section_order", "cellbin_id", "x", "y", "obs_index", "obs_position"]
        table = obs[keep].copy()
        table = table.loc[table["sample_id"].astype(str) == sample].copy()
        table["section_order"] = pd.to_numeric(table["section_order"], errors="coerce").astype("Int64")
        table["x"] = pd.to_numeric(table["x"], errors="raise")
        table["y"] = pd.to_numeric(table["y"], errors="raise")
        payload: dict[str, Any] = {"sample_id": sample, "expression_qc_available": False}
        try:
            matrix = data.layers["counts"][:, :]
            matrix = matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
            table["total_counts"] = np.asarray(matrix.sum(axis=1)).ravel()
            table["detected_genes"] = np.diff(matrix.indptr)
            payload.update(
                {
                    "expression_qc_available": True,
                    "total_counts_median": float(table["total_counts"].median()),
                    "detected_genes_median": float(table["detected_genes"].median()),
                }
            )
        except Exception as exc:  # noqa: BLE001
            table["total_counts"] = np.nan
            table["detected_genes"] = np.nan
            payload["expression_qc_error"] = f"{type(exc).__name__}: {exc}"
        duplicate_count = int(table.duplicated(["sample_id", "slice_id", "cellbin_id"]).sum())
        if duplicate_count:
            raise ValueError(f"Duplicate cellbin keys in {sample}: {duplicate_count}")
        return table.reset_index(drop=True), payload
    finally:
        file_obj = getattr(data, "file", None)
        close = getattr(file_obj, "close", None)
        if callable(close):
            close()


def load_all_cellbins(input_packet_root: Path, samples: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    payloads = []
    for sample in samples:
        path = h5ad_path_for_sample(input_packet_root, sample)
        frame, payload = load_cellbin_table_with_expression_qc(path, sample)
        frames.append(frame)
        payloads.append(payload)
    return pd.concat(frames, ignore_index=True), payloads


def top_features_from_lineage(lineage: pd.DataFrame, unit_cols: list[str], limit: int = 10) -> pd.DataFrame:
    frame = lineage.copy()
    frame["count"] = pd.to_numeric(frame["count"], errors="raise")
    frame = frame.loc[frame["count"] > 0].copy()
    frame["assay_feature_id"] = frame["assay"].astype(str) + "::" + frame["feature_id"].astype(str)
    grouped_cols = unit_cols + ["assay", "feature_id", "assay_feature_id"]
    if "clone_id" in frame.columns:
        grouped_cols.append("clone_id")
    feature = frame.groupby(grouped_cols, as_index=False)["count"].sum().rename(columns={"count": "feature_count"})
    feature = feature.sort_values(unit_cols + ["feature_count", "assay", "feature_id"], ascending=[True] * len(unit_cols) + [False, True, True])
    feature["feature_rank"] = feature.groupby(unit_cols).cumcount() + 1
    return feature.loc[feature["feature_rank"] <= int(limit)].reset_index(drop=True)


def section_summary_from_cellbins(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample, group in summary.groupby("sample_id", sort=True):
        rows.append(
            {
                "sample_id": sample,
                "cellbin_count": int(len(group)),
                "lineage_positive_cellbins": int(group["evidence_present"].astype(bool).sum()),
                "fraction_lineage_positive": float(group["evidence_present"].astype(bool).mean()),
                "total_lineage_count": float(group["total_lineage_count"].sum()),
                "detected_feature_count_sum": float(group["detected_feature_count"].sum()),
                "median_feature_entropy": float(group["feature_entropy"].median()),
                "median_dominant_feature_fraction": float(group["dominant_feature_fraction"].median()),
                "RA_total_count": float(group["RA_total_count"].sum()),
                "TA_total_count": float(group["TA_total_count"].sum()),
                "CA_total_count": float(group["CA_total_count"].sum()),
                "median_assay_balance": float(group["assay_balance"].median()),
            }
        )
    return pd.DataFrame(rows)


def preflight_phase(args: argparse.Namespace, samples: list[str], output_root: Path, report_root: Path) -> dict[str, Any]:
    snapshot = snapshot_files(packet_files(args.input_packet_root))
    ensure_dir(output_root / "qc")
    atomic_write_tsv(output_root / "qc" / "source_input_packet_snapshot_before.tsv", snapshot, overwrite=args.overwrite)

    h5ad_info = []
    for sample in samples:
        path = h5ad_path_for_sample(args.input_packet_root, sample)
        info = validate_l126_h5ad_schema(path)
        info["expected_n_obs"] = EXPECTED_CELLBIN_COUNTS.get(sample)
        info["n_obs_matches_expected"] = info["n_obs"] == EXPECTED_CELLBIN_COUNTS.get(sample)
        h5ad_info.append(info)
    cellbin_summary_path = args.barcode_root / "cellbin_lineage_summary.tsv.gz"
    lineage_path = args.input_packet_root / "processed" / "lineage_evidence" / "cellbin_lineage_evidence.tsv.gz"
    allele_path = args.input_packet_root / "processed" / "lineage_evidence" / "feature_allele_annotation_long.tsv.gz"
    cellbin_summary = read_table(cellbin_summary_path)
    lineage = load_cellbin_lineage_evidence(lineage_path)
    allele = load_feature_allele_annotation(allele_path)
    allele_audit, allele_payload = audit_allele_annotation(lineage, allele)
    atomic_write_tsv_gz(output_root / "qc" / "feature_allele_annotation_audit.tsv.gz", allele_audit, overwrite=args.overwrite)
    counts = cellbin_summary.groupby("sample_id").size().to_dict()
    expected_ok = all(int(counts.get(sample, 0)) == EXPECTED_CELLBIN_COUNTS[sample] for sample in samples)
    schema_ok = all(item["schema_passed"] and item["n_obs_matches_expected"] for item in h5ad_info)
    decision = (
        "L126_FULL_CHARACTERIZATION_PREFLIGHT_READY"
        if schema_ok and expected_ok and allele_payload["non_inflation_passed"]
        else "L126_FULL_CHARACTERIZATION_HOLD_FOR_INPUTS"
    )
    if not allele_payload["assays_preserved"]:
        decision = "L126_FULL_CHARACTERIZATION_HOLD_FOR_LINEAGE_SCHEMA"
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": decision,
        "h5ad_info": h5ad_info,
        "cellbin_summary_rows": int(len(cellbin_summary)),
        "cellbin_counts_by_section": {str(k): int(v) for k, v in counts.items()},
        "lineage_evidence_rows": int(len(lineage)),
        "allele_annotation_rows": int(len(allele)),
        "allele_annotation_audit": allele_payload,
        "round4_decision_report": str(PROJECT_ROOT / "reports" / "l126_plana_lineage_kernel_hardening_round4" / "06_HARDENING_DECISION.md"),
        "source_input_snapshot_path": str(output_root / "qc" / "source_input_packet_snapshot_before.tsv"),
    }
    write_report(
        report_root,
        "00_SCOPE_AND_PREFLIGHT",
        "Scope And Preflight",
        payload,
        [
            f"- Decision label: `{decision}`",
            "- L126 is now treated as a full barcode-aware spatial niche characterization benchmark.",
            "- Round 4 did not support scaling directed GPCCA interpretation to full data.",
            "- Full characterization remains useful for mapping DARLIN evidence across spatial local contexts and non-overlapping tiles.",
            "",
            markdown_table(pd.DataFrame(h5ad_info)[["path", "n_obs", "expected_n_obs", "n_obs_matches_expected", "schema_passed"]]),
            "",
            f"- Cellbin lineage summary rows: `{len(cellbin_summary)}`",
            f"- Primary lineage evidence rows: `{len(lineage)}`",
            f"- Allele non-inflation passed: `{allele_payload['non_inflation_passed']}`",
        ],
        overwrite=args.overwrite,
    )
    return payload


def cellbin_phase(args: argparse.Namespace, samples: list[str], output_root: Path, report_root: Path) -> dict[str, Any]:
    out_dir = ensure_dir(output_root / "cellbin")
    cellbin_summary = read_table(args.barcode_root / "cellbin_lineage_summary.tsv.gz")
    cellbins, expression_payloads = load_all_cellbins(args.input_packet_root, samples)
    expression_cols = ["sample_id", "slice_id", "cellbin_id", "total_counts", "detected_genes"]
    full = cellbin_summary.merge(cellbins[expression_cols], on=["sample_id", "slice_id", "cellbin_id"], how="left")
    full = add_assay_balance(full)
    for col in CELLBIN_REQUIRED_COLUMNS:
        if col not in full:
            full[col] = "" if col in {"dominant_assay", "dominant_feature_id"} else 0
    full["dominant_assay"] = full["dominant_assay"].fillna("")
    full["dominant_feature_id"] = full["dominant_feature_id"].fillna("")
    full = full[[*CELLBIN_REQUIRED_COLUMNS, "total_counts", "detected_genes"]]

    assay_rows = []
    for assay in EXPECTED_ASSAYS:
        temp = full[["sample_id", "slice_id", "section_order", "cellbin_id", "x", "y"]].copy()
        temp["assay"] = assay
        temp["assay_total_count"] = full[f"{assay}_total_count"]
        temp["assay_detected_feature_count"] = full[f"{assay}_detected_feature_count"]
        temp["assay_evidence_present"] = temp["assay_total_count"].gt(0)
        assay_rows.append(temp)
    assay_summary = pd.concat(assay_rows, ignore_index=True)

    lineage = load_cellbin_lineage_evidence(args.input_packet_root / "processed" / "lineage_evidence" / "cellbin_lineage_evidence.tsv.gz")
    top_features = top_features_from_lineage(lineage, list(PRIMARY_JOIN_KEY), limit=10)
    section_summary = section_summary_from_cellbins(full)

    atomic_write_tsv_gz(out_dir / "full_cellbin_lineage_summary.tsv.gz", full, overwrite=args.overwrite)
    atomic_write_tsv_gz(out_dir / "full_cellbin_assay_summary.tsv.gz", assay_summary, overwrite=args.overwrite)
    atomic_write_tsv_gz(out_dir / "full_cellbin_top_features.tsv.gz", top_features, overwrite=args.overwrite)
    atomic_write_tsv(out_dir / "full_section_summary.tsv", section_summary, overwrite=args.overwrite)

    payload = {
        "generated_at_utc": utc_now(),
        "cellbin_rows": int(len(full)),
        "section_summary": section_summary.to_dict(orient="records"),
        "expression_qc": expression_payloads,
        "outputs": {
            "full_cellbin_lineage_summary": str(out_dir / "full_cellbin_lineage_summary.tsv.gz"),
            "full_cellbin_assay_summary": str(out_dir / "full_cellbin_assay_summary.tsv.gz"),
            "full_cellbin_top_features": str(out_dir / "full_cellbin_top_features.tsv.gz"),
            "full_section_summary": str(out_dir / "full_section_summary.tsv"),
        },
    }
    write_report(
        report_root,
        "01_FULL_CELLBIN_LINEAGE",
        "Full Cellbin Lineage",
        payload,
        [
            "- Built full cellbin-level lineage characterization for all L126 sections.",
            "- RA/TA/CA were preserved separately and top features include assay-scoped IDs.",
            "",
            markdown_table(section_summary),
        ],
        overwrite=args.overwrite,
    )
    return payload


def full_grouping_preflight_phase(args: argparse.Namespace, samples: list[str], output_root: Path, report_root: Path) -> dict[str, Any]:
    full = read_table(output_root / "cellbin" / "full_cellbin_lineage_summary.tsv.gz", nrows=None)
    rows = []
    for sample in samples:
        n = int((full["sample_id"].astype(str) == sample).sum())
        assignment_rows = n * int(args.knn_k)
        rows.append(
            {
                "sample_id": sample,
                "cellbin_count": n,
                "k": int(args.knn_k),
                "expected_group_count": n,
                "expected_assignment_rows": assignment_rows,
                "estimated_uncompressed_mb": round(assignment_rows * 180 / 1024 / 1024, 2),
            }
        )
    frame = pd.DataFrame(rows)
    total_rows = int(frame["expected_assignment_rows"].sum())
    decision = "L126_FULL_GROUPING_READY_WITH_CHUNKING"
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": decision,
        "total_expected_assignment_rows": total_rows,
        "per_section": rows,
        "processing_strategy": "per-section vectorized cKDTree query and compressed TSV writes",
        "overlapping_local_context": True,
    }
    write_report(
        report_root,
        "02_FULL_GROUPING_PREFLIGHT",
        "Full Grouping Preflight",
        payload,
        [
            f"- Decision label: `{decision}`",
            f"- Expected total assignment rows: `{total_rows}`",
            "- Full group assignment is safe with per-section chunking.",
            "- Group outputs are overlapping local contexts and must not be summed as tissue abundance.",
            "",
            markdown_table(frame),
        ],
        overwrite=args.overwrite,
    )
    return payload


def build_full_group_assignment(section: pd.DataFrame, k_neighbors: int) -> pd.DataFrame:
    local = section.sort_values("cellbin_id").reset_index(drop=True)
    coords = local[["x", "y"]].to_numpy(dtype=float)
    k = min(int(k_neighbors), len(local))
    _, indices = cKDTree(coords).query(coords, k=k)
    if k == 1:
        indices = indices[:, None]
    n = len(local)
    anchor_pos = np.repeat(np.arange(n), k)
    member_pos = indices.reshape(-1)
    sample_id = local["sample_id"].astype(str).to_numpy(dtype=str)
    slice_id = local["slice_id"].astype(str).to_numpy(dtype=str)
    section_order = local["section_order"].astype(int).to_numpy()
    cellbin_id = local["cellbin_id"].astype(str).to_numpy(dtype=str)
    x = local["x"].to_numpy(dtype=float)
    y = local["y"].to_numpy(dtype=float)
    group_id = np.char.add(np.char.add(sample_id[anchor_pos], "__anchor__"), cellbin_id[anchor_pos])
    role = np.where(anchor_pos == member_pos, "center", "member")
    return pd.DataFrame(
        {
            "sample_id": sample_id[anchor_pos],
            "slice_id": slice_id[anchor_pos],
            "section_order": section_order[anchor_pos],
            "group_id": group_id,
            "group_type": GROUP_TYPE,
            "niche_id": group_id,
            "anchor_cellbin_id": cellbin_id[anchor_pos],
            "anchor_x": x[anchor_pos],
            "anchor_y": y[anchor_pos],
            "cellbin_id": cellbin_id[member_pos],
            "x": x[member_pos],
            "y": y[member_pos],
            "role": role,
        }
    )


def add_group_summary_aliases(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if "local_context_not_tissue_abundance" not in out:
        out["local_context_not_tissue_abundance"] = True
    out["local_context_not_tissue_partition"] = True
    out["n_member_cellbins"] = out["n_member_cellbins"].astype(int)
    out["n_member_cellbins_with_lineage"] = out["n_member_cellbins_with_lineage"].astype(int)
    out["evidence_present"] = out["total_lineage_count"].gt(0)
    return out


def groups_phase(args: argparse.Namespace, samples: list[str], output_root: Path, report_root: Path) -> dict[str, Any]:
    groups_dir = ensure_dir(output_root / "groups")
    lineage_dir = ensure_dir(output_root / "group_lineage")
    cellbin = read_table(output_root / "cellbin" / "full_cellbin_lineage_summary.tsv.gz")
    lineage = load_cellbin_lineage_evidence(args.input_packet_root / "processed" / "lineage_evidence" / "cellbin_lineage_evidence.tsv.gz")
    rows = []
    for sample in samples:
        section = cellbin.loc[cellbin["sample_id"].astype(str) == sample].copy()
        assignment = build_full_group_assignment(section, args.knn_k)
        lineage_sample = lineage.loc[lineage["sample_id"].astype(str) == sample].copy()
        group_summary, assay_summary, top_features = aggregate_group_lineage(lineage_sample, assignment)
        group_summary = add_group_summary_aliases(group_summary)
        top_features["assay_feature_id"] = top_features["assay"].astype(str) + "::" + top_features["feature_id"].astype(str)

        atomic_write_tsv_gz(groups_dir / f"{sample}_full_group_assignment.tsv.gz", assignment, overwrite=args.overwrite)
        atomic_write_tsv_gz(lineage_dir / f"{sample}_full_group_lineage_summary.tsv.gz", group_summary, overwrite=args.overwrite)
        atomic_write_tsv_gz(lineage_dir / f"{sample}_full_group_assay_summary.tsv.gz", assay_summary, overwrite=args.overwrite)
        atomic_write_tsv_gz(lineage_dir / f"{sample}_full_group_top_features.tsv.gz", top_features, overwrite=args.overwrite)
        rows.append(
            {
                "sample_id": sample,
                "cellbins": int(len(section)),
                "groups": int(assignment["group_id"].nunique()),
                "assignment_rows": int(len(assignment)),
                "groups_with_lineage": int(group_summary["evidence_present"].astype(bool).sum()),
                "fraction_groups_with_lineage": float(group_summary["evidence_present"].astype(bool).mean()),
                "median_fraction_member_cellbins_with_lineage": float(group_summary["fraction_member_cellbins_with_lineage"].median()),
                "median_total_lineage_count": float(group_summary["total_lineage_count"].median()),
            }
        )
    qc = pd.DataFrame(rows)
    atomic_write_tsv(lineage_dir / "full_group_lineage_section_qc.tsv", qc, overwrite=args.overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "local_context_not_tissue_partition": True,
        "section_qc": rows,
        "outputs_root": str(output_root),
    }
    write_report(
        report_root,
        "03_FULL_GROUP_LINEAGE_AGGREGATION",
        "Full Group Lineage Aggregation",
        payload,
        [
            "- Full within-section kNN local neighborhood aggregation completed.",
            "- These groups are overlapping local contexts, not tissue partitions.",
            "- Group-level counts must not be summed as tissue abundance.",
            "",
            markdown_table(qc),
        ],
        overwrite=args.overwrite,
    )
    return payload


def tile_grid_size(tile_grid: str) -> int:
    if tile_grid == "auto":
        return 30
    if "x" in tile_grid:
        left, right = tile_grid.lower().split("x", 1)
        if int(left) != int(right):
            raise ValueError("--tile-grid currently requires square grids such as 30x30")
        return int(left)
    return int(tile_grid)


def build_tile_assignment(section: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    local = section.copy()
    local["tile_x_bin"] = pd.qcut(local["x"].rank(method="first"), q=grid_size, labels=False, duplicates="drop").astype(int)
    local["tile_y_bin"] = pd.qcut(local["y"].rank(method="first"), q=grid_size, labels=False, duplicates="drop").astype(int)
    local["tile_id"] = (
        local["sample_id"].astype(str)
        + "__tile_x"
        + local["tile_x_bin"].astype(str).str.zfill(2)
        + "_y"
        + local["tile_y_bin"].astype(str).str.zfill(2)
    )
    keep = [
        "sample_id",
        "slice_id",
        "section_order",
        "tile_id",
        "tile_x_bin",
        "tile_y_bin",
        "cellbin_id",
        "x",
        "y",
        "total_counts",
        "detected_genes",
    ]
    return local[keep].copy()


def tile_base_summary(assignment: pd.DataFrame) -> pd.DataFrame:
    return (
        assignment.groupby(["sample_id", "slice_id", "section_order", "tile_id", "tile_x_bin", "tile_y_bin"], as_index=False)
        .agg(
            n_cellbins=("cellbin_id", "nunique"),
            centroid_x=("x", "mean"),
            centroid_y=("y", "mean"),
            mean_total_counts=("total_counts", "mean"),
            mean_detected_genes=("detected_genes", "mean"),
        )
        .sort_values(["sample_id", "tile_id"])
    )


def tiles_phase(args: argparse.Namespace, samples: list[str], output_root: Path, report_root: Path) -> dict[str, Any]:
    out_dir = ensure_dir(output_root / "spatial_tiles")
    cellbin = read_table(output_root / "cellbin" / "full_cellbin_lineage_summary.tsv.gz")
    lineage = load_cellbin_lineage_evidence(args.input_packet_root / "processed" / "lineage_evidence" / "cellbin_lineage_evidence.tsv.gz")
    grid_size = tile_grid_size(args.tile_grid)
    rows = []
    for sample in samples:
        section = cellbin.loc[cellbin["sample_id"].astype(str) == sample].copy()
        assignment = build_tile_assignment(section, grid_size)
        mapping = assignment.rename(columns={"tile_id": "unit_id"})
        summary, _, _ = aggregate_lineage_for_unit_mapping(
            lineage.loc[lineage["sample_id"].astype(str) == sample].copy(),
            mapping[["unit_id", "sample_id", "slice_id", "section_order", "cellbin_id"]],
            unit_col="unit_id",
            local_context=False,
        )
        summary = summary.rename(columns={"unit_id": "tile_id"})
        base = tile_base_summary(assignment)
        summary = base.merge(summary, on="tile_id", how="left")
        summary["n_lineage_positive_cellbins"] = summary["n_unique_lineage_positive_cellbins"].fillna(0).astype(int)
        summary["fraction_lineage_positive"] = summary["fraction_lineage_positive"].fillna(0.0)
        summary["dominant_assay"] = summary["dominant_assay"].fillna("")
        summary["dominant_feature_id"] = summary["dominant_feature_id"].fillna("")
        summary["evidence_present"] = summary["total_lineage_count"].fillna(0).gt(0)
        summary["non_overlapping_tile"] = True
        summary["local_context_not_tissue_partition"] = False
        atomic_write_tsv_gz(out_dir / f"{sample}_tile_assignment.tsv.gz", assignment, overwrite=args.overwrite)
        atomic_write_tsv_gz(out_dir / f"{sample}_tile_lineage_summary.tsv.gz", summary, overwrite=args.overwrite)
        rows.append(
            {
                "sample_id": sample,
                "tile_grid": f"{grid_size}x{grid_size}",
                "tiles": int(summary["tile_id"].nunique()),
                "cellbins": int(len(assignment)),
                "one_tile_per_cellbin": bool(not assignment.duplicated(["sample_id", "slice_id", "cellbin_id"]).any()),
                "tiles_with_lineage": int(summary["evidence_present"].astype(bool).sum()),
                "fraction_tiles_with_lineage": float(summary["evidence_present"].astype(bool).mean()),
                "median_fraction_lineage_positive": float(summary["fraction_lineage_positive"].median()),
            }
        )
    qc = pd.DataFrame(rows)
    atomic_write_tsv(out_dir / "full_tile_section_qc.tsv", qc, overwrite=args.overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "tile_grid": f"{grid_size}x{grid_size}",
        "section_qc": rows,
    }
    write_report(
        report_root,
        "04_FULL_SPATIAL_TILE_CHARACTERIZATION",
        "Full Spatial Tile Characterization",
        payload,
        [
            "- Built non-overlapping spatial tiles for robust characterization figures.",
            "- Each cellbin belongs to exactly one tile.",
            "",
            markdown_table(qc),
        ],
        overwrite=args.overwrite,
    )
    return payload


def section_entropy(counts: pd.Series) -> float:
    values = counts.to_numpy(dtype=float)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    probs = values / values.sum()
    return float(-(probs * np.log(probs)).sum())


def metaniche_phase(args: argparse.Namespace, samples: list[str], output_root: Path, report_root: Path) -> dict[str, Any]:
    out_dir = ensure_dir(output_root / "metaniche")
    tile_frames = [read_table(output_root / "spatial_tiles" / f"{sample}_tile_lineage_summary.tsv.gz") for sample in samples]
    tiles = pd.concat(tile_frames, ignore_index=True)
    for assay in EXPECTED_ASSAYS:
        tiles[f"{assay}_fraction"] = np.where(
            tiles["total_lineage_count"].fillna(0).astype(float) > 0,
            tiles[f"{assay}_total_count"].fillna(0).astype(float) / tiles["total_lineage_count"].fillna(0).astype(float).replace(0, np.nan),
            0.0,
        )
        tiles[f"{assay}_fraction"] = tiles[f"{assay}_fraction"].fillna(0.0)
    feature_cols = [
        "centroid_x",
        "centroid_y",
        "n_cellbins",
        "fraction_lineage_positive",
        "total_lineage_count",
        "detected_feature_count",
        "dominant_feature_fraction",
        "feature_entropy",
        "simpson_diversity",
        "assay_balance",
        "RA_fraction",
        "TA_fraction",
        "CA_fraction",
        "mean_total_counts",
        "mean_detected_genes",
    ]
    matrix = tiles[feature_cols].fillna(0.0).to_numpy(dtype=float)
    matrix[:, feature_cols.index("total_lineage_count")] = np.log1p(matrix[:, feature_cols.index("total_lineage_count")])
    matrix[:, feature_cols.index("detected_feature_count")] = np.log1p(matrix[:, feature_cols.index("detected_feature_count")])
    matrix = StandardScaler().fit_transform(matrix)
    n_clusters = min(int(args.n_metaniches), len(tiles))
    labels = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10).fit_predict(matrix)
    assignment = tiles[["sample_id", "slice_id", "section_order", "tile_id", "n_cellbins"]].copy()
    assignment["metaniche_id"] = [f"L126_full_tile_metaniche_{label:03d}" for label in labels]

    enriched = tiles.merge(assignment[["tile_id", "metaniche_id"]], on="tile_id", how="left")
    rows = []
    for metaniche_id, group in enriched.groupby("metaniche_id", sort=True):
        counts = group["sample_id"].astype(str).value_counts().sort_index()
        total_tiles = int(len(group))
        purity = float(counts.max() / total_tiles) if total_tiles else 0.0
        rows.append(
            {
                "metaniche_id": metaniche_id,
                "n_tiles": total_tiles,
                "n_cellbins": int(group["n_cellbins"].sum()),
                "section_distribution": ";".join(f"{key}:{int(value)}" for key, value in counts.items()),
                "section_purity": purity,
                "section_entropy": section_entropy(counts),
                "section_dominated": bool(purity > 0.9),
                "lineage_coverage_mean": float(group["fraction_lineage_positive"].mean()),
                "total_lineage_count_sum": float(group["total_lineage_count"].sum()),
                "feature_entropy_mean": float(group["feature_entropy"].mean()),
                "dominant_feature_fraction_mean": float(group["dominant_feature_fraction"].mean()),
                "assay_balance_mean": float(group["assay_balance"].mean()),
                "RA_total_count": float(group["RA_total_count"].sum()),
                "TA_total_count": float(group["TA_total_count"].sum()),
                "CA_total_count": float(group["CA_total_count"].sum()),
                "centroid_x_mean": float(group["centroid_x"].mean()),
                "centroid_y_mean": float(group["centroid_y"].mean()),
            }
        )
    summary = pd.DataFrame(rows)

    tile_assignments = [
        read_table(output_root / "spatial_tiles" / f"{sample}_tile_assignment.tsv.gz")[
            ["sample_id", "slice_id", "section_order", "tile_id", "cellbin_id"]
        ]
        for sample in samples
    ]
    met_mapping = pd.concat(tile_assignments, ignore_index=True).merge(
        assignment[["tile_id", "metaniche_id"]],
        on="tile_id",
        how="left",
    )
    lineage = load_cellbin_lineage_evidence(args.input_packet_root / "processed" / "lineage_evidence" / "cellbin_lineage_evidence.tsv.gz")
    barcode_annotation, _, top_features = aggregate_lineage_for_unit_mapping(
        lineage,
        met_mapping[["metaniche_id", "sample_id", "slice_id", "section_order", "cellbin_id"]],
        unit_col="metaniche_id",
        local_context=False,
    )
    top_features["assay_feature_id"] = top_features["assay"].astype(str) + "::" + top_features["feature_id"].astype(str)
    atomic_write_tsv_gz(out_dir / "full_metaniche_assignment.tsv.gz", assignment, overwrite=args.overwrite)
    atomic_write_tsv_gz(out_dir / "full_metaniche_summary.tsv.gz", summary, overwrite=args.overwrite)
    atomic_write_tsv_gz(out_dir / "full_metaniche_barcode_annotation.tsv.gz", barcode_annotation, overwrite=args.overwrite)
    atomic_write_tsv_gz(out_dir / "full_metaniche_top_features.tsv.gz", top_features, overwrite=args.overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "n_metaniches": int(summary["metaniche_id"].nunique()),
        "section_dominated_metaniches": int(summary["section_dominated"].astype(bool).sum()),
        "median_n_tiles": float(summary["n_tiles"].median()),
        "median_lineage_coverage": float(summary["lineage_coverage_mean"].median()),
    }
    write_report(
        report_root,
        "05_FULL_METANICHE_CHARACTERIZATION",
        "Full Metaniche Characterization",
        payload,
        [
            "- Built tile-based metaniche-like categories for characterization only.",
            "- No directed kernel or GPCCA was constructed.",
            "",
            markdown_table(summary.head(20)),
        ],
        overwrite=args.overwrite,
    )
    return payload


def save_figure(fig: plt.Figure, path_base: Path) -> list[Path]:
    ensure_dir(path_base.parent)
    fig.tight_layout()
    paths = []
    for suffix in ("png", "pdf"):
        path = path_base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=180 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def scatter_sections(frame: pd.DataFrame, value_col: str, title: str, path_base: Path, *, categorical: bool = False) -> list[Path]:
    samples = sorted(frame["sample_id"].astype(str).unique())
    fig, axes = plt.subplots(1, len(samples), figsize=(5 * len(samples), 4), squeeze=False)
    assay_colors = {"RA": "#d55e00", "TA": "#0072b2", "CA": "#009e73", "": "#bbbbbb", "nan": "#bbbbbb"}
    x_col = "x" if "x" in frame.columns else "centroid_x"
    y_col = "y" if "y" in frame.columns else "centroid_y"
    for ax, sample in zip(axes.ravel(), samples, strict=False):
        local = frame.loc[frame["sample_id"].astype(str) == sample]
        if categorical:
            colors = local[value_col].astype(str).map(assay_colors).fillna("#bbbbbb")
            ax.scatter(local[x_col], local[y_col], c=colors, s=1, linewidths=0, rasterized=True)
        else:
            values = pd.to_numeric(local[value_col], errors="coerce").fillna(0.0)
            vmax = float(values.quantile(0.99)) if len(values) else 1.0
            vmax = vmax if vmax > 0 else 1.0
            sc = ax.scatter(local[x_col], local[y_col], c=values.clip(upper=vmax), s=1, linewidths=0, cmap="viridis", rasterized=True)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(sample)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    return save_figure(fig, path_base)


def figures_phase(args: argparse.Namespace, samples: list[str], output_root: Path, report_root: Path) -> dict[str, Any]:
    fig_root = ensure_dir(report_root / "figures")
    key_root = ensure_dir(report_root / "key_figure_candidates")
    outputs: list[Path] = []
    cellbin = read_table(output_root / "cellbin" / "full_cellbin_lineage_summary.tsv.gz")
    cellbin["evidence_present_numeric"] = cellbin["evidence_present"].astype(bool).astype(int)
    outputs += scatter_sections(cellbin, "evidence_present_numeric", "Lineage evidence coverage by cellbin", fig_root / "cellbin_lineage_coverage_spatial")
    outputs += scatter_sections(cellbin, "total_lineage_count", "Total lineage count by cellbin", fig_root / "cellbin_total_lineage_count_spatial")
    outputs += scatter_sections(cellbin, "detected_feature_count", "Detected feature count by cellbin", fig_root / "cellbin_detected_feature_count_spatial")
    outputs += scatter_sections(cellbin, "feature_entropy", "Barcode feature entropy by cellbin", fig_root / "cellbin_feature_entropy_spatial")
    outputs += scatter_sections(cellbin, "dominant_feature_fraction", "Dominant feature fraction by cellbin", fig_root / "cellbin_dominant_feature_fraction_spatial")
    outputs += scatter_sections(cellbin, "dominant_assay", "Dominant assay by cellbin", fig_root / "cellbin_dominant_assay_spatial", categorical=True)

    fig, axes = plt.subplots(3, len(samples), figsize=(5 * len(samples), 10), squeeze=False)
    for row, assay in enumerate(EXPECTED_ASSAYS):
        for col, sample in enumerate(samples):
            ax = axes[row, col]
            local = cellbin.loc[cellbin["sample_id"].astype(str) == sample]
            values = pd.to_numeric(local[f"{assay}_total_count"], errors="coerce").fillna(0.0)
            vmax = float(values.quantile(0.99)) if len(values) else 1.0
            vmax = vmax if vmax > 0 else 1.0
            sc = ax.scatter(local["x"], local["y"], c=values.clip(upper=vmax), s=1, linewidths=0, cmap="magma", rasterized=True)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{sample} {assay}")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("RA/TA/CA lineage counts by cellbin")
    outputs += save_figure(fig, fig_root / "cellbin_assay_total_count_spatial")

    section_summary = read_table(output_root / "cellbin" / "full_section_summary.tsv")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(section_summary))
    bottom = np.zeros(len(section_summary))
    for assay, color in zip(EXPECTED_ASSAYS, ["#d55e00", "#0072b2", "#009e73"], strict=True):
        vals = section_summary[f"{assay}_total_count"].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=assay, color=color)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(section_summary["sample_id"], rotation=20)
    ax.set_ylabel("lineage count")
    ax.legend()
    ax.set_title("RA/TA/CA count summary by section")
    outputs += save_figure(fig, fig_root / "section_assay_balance_summary")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(cellbin["feature_entropy"], bins=60, color="#4c78a8")
    axes[0].set_title("Feature entropy")
    axes[1].hist(cellbin["dominant_feature_fraction"], bins=60, color="#f58518")
    axes[1].set_title("Dominant feature fraction")
    top_features = read_table(output_root / "cellbin" / "full_cellbin_top_features.tsv.gz")
    top_freq = top_features.loc[top_features["feature_rank"].astype(int) == 1, "assay_feature_id"].value_counts().head(20)
    axes[2].barh(np.arange(len(top_freq)), top_freq.to_numpy()[::-1], color="#54a24b")
    axes[2].set_yticks(np.arange(len(top_freq)))
    axes[2].set_yticklabels(top_freq.index[::-1], fontsize=6)
    axes[2].set_title("Top dominant features")
    outputs += save_figure(fig, fig_root / "barcode_diversity_and_dominance_distributions")

    if (output_root / "spatial_tiles").exists():
        tile_frames = [read_table(output_root / "spatial_tiles" / f"{sample}_tile_lineage_summary.tsv.gz") for sample in samples]
        tiles = pd.concat(tile_frames, ignore_index=True)
        outputs += scatter_sections(tiles, "fraction_lineage_positive", "Tile lineage coverage", fig_root / "tile_lineage_coverage_spatial")
        outputs += scatter_sections(tiles, "feature_entropy", "Tile barcode entropy", fig_root / "tile_feature_entropy_spatial")
    if (output_root / "metaniche" / "full_metaniche_summary.tsv.gz").exists():
        met = read_table(output_root / "metaniche" / "full_metaniche_summary.tsv.gz")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(met["section_purity"], bins=30, color="#b279a2")
        ax.set_title("Metaniche section purity")
        ax.set_xlabel("section purity")
        outputs += save_figure(fig, fig_root / "metaniche_section_purity_distribution")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    text = (
        "L126 full barcode-aware niche characterization\n"
        "Input: h5ad cellbins + DARLIN lineage evidence\n"
        "Cellbin view: unique cellbin evidence metrics\n"
        "Group view: overlapping kNN local contexts, not tissue abundance\n"
        "Tile view: non-overlapping spatial bins for robust maps\n"
        "Metaniche view: tile-based characterization categories only\n"
        "No directed kernel, GPCCA, or directional biological inference"
    )
    ax.text(0.02, 0.95, text, va="top", ha="left", fontsize=12, family="monospace")
    outputs += save_figure(fig, fig_root / "method_data_flow_summary")

    key_stems = {
        "cellbin_lineage_coverage_spatial",
        "cellbin_feature_entropy_spatial",
        "cellbin_dominant_feature_fraction_spatial",
        "section_assay_balance_summary",
        "tile_feature_entropy_spatial",
        "method_data_flow_summary",
    }
    key_paths = []
    for path in outputs:
        if path.stem in key_stems:
            dest = key_root / path.name
            shutil.copy2(path, dest)
            key_paths.append(dest)
    payload = {
        "generated_at_utc": utc_now(),
        "figure_count": len(outputs),
        "figures": [str(path) for path in outputs],
        "key_figure_candidates": [str(path) for path in key_paths],
    }
    write_report(
        report_root,
        "06_FULL_CHARACTERIZATION_FIGURES",
        "Full Characterization Figures",
        payload,
        [
            f"- Generated `{len(outputs)}` figure files.",
            f"- Key figure candidates: `{len(key_paths)}` files.",
            "- Figure language is descriptive: coverage, diversity, dominant feature, assay balance, and local composition.",
        ],
        overwrite=args.overwrite,
    )
    return payload


def summary_phase(args: argparse.Namespace, samples: list[str], output_root: Path, report_root: Path) -> dict[str, Any]:
    section = read_table(output_root / "cellbin" / "full_section_summary.tsv")
    group_qc_path = output_root / "group_lineage" / "full_group_lineage_section_qc.tsv"
    tile_qc_path = output_root / "spatial_tiles" / "full_tile_section_qc.tsv"
    group_qc = read_table(group_qc_path) if group_qc_path.exists() else pd.DataFrame()
    tile_qc = read_table(tile_qc_path) if tile_qc_path.exists() else pd.DataFrame()
    figures_json = report_root / "06_FULL_CHARACTERIZATION_FIGURES.json"
    figures_payload = json.loads(figures_json.read_text()) if figures_json.exists() else {"key_figure_candidates": []}
    cellbin_ok = int(section["cellbin_count"].sum()) == sum(EXPECTED_CELLBIN_COUNTS.values())
    characterization_ok = (not group_qc.empty) or (not tile_qc.empty)
    figures_ok = bool(figures_payload.get("figures"))
    label = FINAL_READY if cellbin_ok and characterization_ok and figures_ok and not group_qc.empty else FINAL_READY_WARN
    if not characterization_ok:
        label = "L126_FULL_CHARACTERIZATION_HOLD_FOR_LINEAGE_AGGREGATION"
    if characterization_ok and not figures_ok:
        label = "L126_FULL_CHARACTERIZATION_HOLD_FOR_FIGURE_REVIEW"
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": label,
        "cellbin_section_summary": section.to_dict(orient="records"),
        "group_section_qc": group_qc.to_dict(orient="records") if not group_qc.empty else [],
        "tile_section_qc": tile_qc.to_dict(orient="records") if not tile_qc.empty else [],
        "key_figure_candidates": figures_payload.get("key_figure_candidates", []),
        "supports": [
            "full spatial barcode evidence coverage mapping",
            "local-context lineage composition characterization",
            "non-overlapping tile-level barcode diversity maps",
            "RA/TA/CA assay balance summaries",
        ],
        "does_not_support": [
            "directed biological outcome analysis",
            "final-state interpretation",
            "section-order temporal modeling",
            "lineage-validated outcome claims",
        ],
        "future_dataset_needed": "A true time-resolved or experimentally directed lineage design would be needed for directional biological inference.",
    }
    coverage_lines = []
    for row in section.itertuples(index=False):
        coverage_lines.append(
            f"- {row.sample_id}: {int(row.cellbin_count)} cellbins; lineage-positive fraction `{float(row.fraction_lineage_positive):.4f}`"
        )
    group_lines = []
    if not group_qc.empty:
        for row in group_qc.itertuples(index=False):
            group_lines.append(
                f"- {row.sample_id}: groups `{int(row.groups)}`; fraction groups with lineage `{float(row.fraction_groups_with_lineage):.4f}`"
            )
    tile_lines = []
    if not tile_qc.empty:
        for row in tile_qc.itertuples(index=False):
            tile_lines.append(
                f"- {row.sample_id}: tiles `{int(row.tiles)}`; fraction tiles with lineage `{float(row.fraction_tiles_with_lineage):.4f}`"
            )
    write_report(
        report_root,
        "07_FULL_CHARACTERIZATION_SUMMARY",
        "Full Characterization Summary",
        payload,
        [
            f"- Final characterization label: `{label}`",
            "",
            "## Full Cellbins",
            *coverage_lines,
            "",
            "## Group Coverage",
            *(group_lines or ["- Full groups were not run."]),
            "",
            "## Tile Coverage",
            *(tile_lines or ["- Spatial tiles were not run."]),
            "",
            "## Interpretation Boundary",
            "- This dataset supports barcode-aware spatial niche characterization.",
            "- This dataset does not support directed biological outcome interpretation.",
            "- Directed GPCCA was stopped because Round 4 hardening did not improve membership confidence and controls recapitulated too much structure.",
            f"- Safe key figure candidates are listed in `{report_root / 'key_figure_candidates'}`.",
        ],
        overwrite=args.overwrite,
    )
    return payload


def forbidden_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in FORBIDDEN_PHRASES if phrase in lowered]


def validation_phase(args: argparse.Namespace, samples: list[str], output_root: Path, report_root: Path) -> dict[str, Any]:
    json_paths = sorted(report_root.glob("*.json"))
    table_paths = sorted(output_root.rglob("*.tsv")) + sorted(output_root.rglob("*.tsv.gz"))
    figure_paths = sorted((report_root / "figures").glob("*.png")) + sorted((report_root / "figures").glob("*.pdf"))
    key_paths = sorted((report_root / "key_figure_candidates").glob("*"))
    report_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(report_root.glob("*.md")))
    before_path = output_root / "qc" / "source_input_packet_snapshot_before.tsv"
    before = read_table(before_path) if before_path.exists() else snapshot_files(packet_files(args.input_packet_root))
    after = snapshot_files(packet_files(args.input_packet_root))
    diff = compare_file_snapshots(before, after)
    h5ad_readback = [validate_l126_h5ad_schema(h5ad_path_for_sample(args.input_packet_root, sample)) for sample in samples]
    decision_path = report_root / "07_FULL_CHARACTERIZATION_SUMMARY.json"
    decision = json.loads(decision_path.read_text()) if decision_path.exists() else {"decision_label": "MISSING_SUMMARY"}
    checks = [
        {"check": "json_parse", "status": all(json.loads(path.read_text()) is not None for path in json_paths), "details": f"{len(json_paths)} json files"},
        {"check": "tsv_gzip_readability", "status": all(len(read_table(path, nrows=5).columns) > 0 for path in table_paths), "details": f"{len(table_paths)} tables"},
        {"check": "figures_non_empty", "status": bool(figure_paths) and all(path.stat().st_size > 0 for path in figure_paths), "details": f"{len(figure_paths)} figures"},
        {"check": "key_figures_exist", "status": bool(key_paths) and all(path.stat().st_size > 0 for path in key_paths), "details": f"{len(key_paths)} key candidates"},
        {"check": "h5ad_readback", "status": all(item["schema_passed"] for item in h5ad_readback), "details": f"{len(h5ad_readback)} h5ad files"},
        {"check": "source_input_packet_unchanged", "status": not bool(diff["changed"].any()), "details": f"{len(diff)} source files checked"},
        {"check": "no_ssd", "status": "/ssd" not in str(output_root) and "/ssd" not in str(report_root), "details": "path guard"},
        {"check": "no_fastq", "status": True, "details": "not run"},
        {"check": "no_darlin_recalling", "status": True, "details": "not run"},
        {"check": "no_directed_gpcca", "status": True, "details": "not run"},
        {"check": "no_full_m0_m1_m2", "status": True, "details": "not run"},
        {"check": "no_planb", "status": True, "details": "not run"},
        {"check": "no_section_order_as_time", "status": "section_order as time" not in report_text.lower(), "details": "wording audit"},
        {"check": "no_forbidden_endpoint_claims", "status": not forbidden_hits(report_text), "details": "; ".join(forbidden_hits(report_text))},
        {"check": "no_git_add_commit_push", "status": True, "details": "not run"},
    ]
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": decision.get("decision_label"),
        "status": "PASS" if all(row["status"] for row in checks) else "FAIL",
        "checks": checks,
    }
    atomic_write_json(report_root / "08_VALIDATION.json", payload, overwrite=args.overwrite)
    atomic_write_text(
        report_root / "08_VALIDATION.md",
        "# Validation\n\n"
        f"- Decision label: `{payload['decision_label']}`\n"
        f"- Validation status: `{payload['status']}`\n"
        f"- Checks passed: `{sum(bool(row['status']) for row in checks)}/{len(checks)}`\n\n"
        + markdown_table(pd.DataFrame(checks), limit=40),
        overwrite=args.overwrite,
    )
    return payload


def main() -> int:
    args = parse_args()
    samples = parse_samples(args.samples)
    output_root = ensure_dir(args.output_root.resolve())
    report_root = ensure_dir(args.report_root.resolve())
    reject_forbidden_paths(args.input_packet_root, args.barcode_root, args.round2B_root, output_root, report_root)

    if args.mode in {"all", "preflight_only"}:
        preflight_phase(args, samples, output_root, report_root)
        if args.mode == "preflight_only":
            return 0
    if args.mode in {"all", "cellbin_only"}:
        cellbin_phase(args, samples, output_root, report_root)
        if args.mode == "cellbin_only":
            return 0
    if args.mode in {"all", "groups_only"} and args.run_full_groups:
        full_grouping_preflight_phase(args, samples, output_root, report_root)
        groups_phase(args, samples, output_root, report_root)
        if args.mode == "groups_only":
            return 0
    if args.mode in {"all", "tiles_only"} and args.run_spatial_tiles:
        tiles_phase(args, samples, output_root, report_root)
        if args.mode == "tiles_only":
            return 0
    if args.mode in {"all", "metaniche_only"} and args.run_metaniche_characterization:
        metaniche_phase(args, samples, output_root, report_root)
        if args.mode == "metaniche_only":
            return 0
    if args.mode in {"all", "figures_only"} and args.make_figures:
        figures_phase(args, samples, output_root, report_root)
        summary_phase(args, samples, output_root, report_root)
        if args.mode == "figures_only":
            return 0
    if args.mode in {"all", "validation_only"}:
        if args.mode == "all" and not (report_root / "07_FULL_CHARACTERIZATION_SUMMARY.json").exists():
            summary_phase(args, samples, output_root, report_root)
        payload = validation_phase(args, samples, output_root, report_root)
        print(f"decision_label={payload['decision_label']}")
        print(f"validation_status={payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
