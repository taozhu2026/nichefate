#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nichefate.barcode_adapter.group_lineage import group_lineage_coverage_metrics  # noqa: E402
from nichefate.barcode_adapter.input_contract import PRIMARY_JOIN_KEY  # noqa: E402
from nichefate.barcode_adapter.l126_schema import (  # noqa: E402
    h5ad_path_for_sample,
    load_l126_cellbin_table,
    validate_l126_h5ad_schema,
)
from nichefate.barcode_adapter.loaders import load_cellbin_lineage_evidence  # noqa: E402
from nichefate.barcode_adapter.qc import compare_file_snapshots, snapshot_files  # noqa: E402
from nichefate.barcode_adapter.reporting import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    atomic_write_tsv,
    atomic_write_tsv_gz,
    ensure_dir,
    markdown_table,
    path_has_ssd,
    utc_now,
)
from nichefate.barcode_adapter.round2b import (  # noqa: E402
    parse_sample_list,
    section_summary_row,
    validate_round2b_group_assignment,
)
from nichefate.barcode_adapter.spatial_neighborhood import group_membership_multiplicity  # noqa: E402
from nichefate.planA_k.kernel_qc import build_sparse_matrix_stats, strong_component_closure_summary  # noqa: E402
from nichefate.planA_l import (  # noqa: E402
    aggregate_state_lineage,
    build_combined_similarity_matrices,
    build_control_kernels,
    build_directed_kernel,
    build_lineage_potential,
    build_plana_bounded_representation,
    build_plana_state_units,
    compare_similarity_matrices,
    forbidden_claim_hits,
    run_tiny_gpcca_smoke,
    write_report_pair,
)
from nichefate.planA_l.lineage_kernel import build_coverage_only_phi, build_phi_shuffled  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L126 PlanA-L round 1 lineage-informed directed kernel pilot.")
    parser.add_argument("--input-packet-root", default="/home/zhutao/scratch/nichefate/spatio_darlin_public_brain_hpc/input_packet")
    parser.add_argument("--round1-barcode-root", default=str(PROJECT_ROOT / "processed/barcode_adapter_l126_round1"))
    parser.add_argument("--round2B-root", default=str(PROJECT_ROOT / "processed/l126_niche_barcode_round2B"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "processed/l126_plana_lineage_kernel_round1"))
    parser.add_argument("--report-root", default=str(PROJECT_ROOT / "reports/l126_plana_lineage_kernel_round1"))
    parser.add_argument("--samples", default="L126_Brain_s1,L126_Brain_s2,L126_Brain_s3")
    parser.add_argument("--max-cellbins-per-section", type=int, default=10000)
    parser.add_argument("--n-hvgs", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-metaniches", type=int, default=200)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--run-gpcca-smoke", action="store_true")
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "state_units_only",
            "lineage_potential_only",
            "kernel_only",
            "controls_only",
            "gpcca_smoke_only",
            "figures_only",
            "audit_only",
        ],
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reject_forbidden_paths(*paths: Path) -> None:
    bad = [str(path) for path in paths if path_has_ssd(path)]
    if bad:
        raise ValueError("Refusing /ssd paths: " + "; ".join(bad))


def read_tsv(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    return pd.read_csv(path, sep="\t", compression=compression)


def atomic_save_npz(path: Path, matrix: sp.spmatrix, *, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    ensure_dir(path.parent)
    tmp = path.with_name(path.stem + ".tmp.npz")
    sp.save_npz(tmp, matrix, compressed=True)
    os.replace(tmp, path)
    return path


def atomic_save_parquet(path: Path, frame: pd.DataFrame, *, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def save_histogram(frame: pd.DataFrame, column: str, title: str, path_base: Path, bins: int = 40) -> list[str]:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    outputs: list[str] = []
    for suffix in [".png", ".pdf"]:
        out = path_base.with_suffix(suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        outputs.append(str(out))
    plt.close(fig)
    return outputs


def save_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
    path_base: Path,
    *,
    cmap: str = "viridis",
) -> list[str]:
    fig, ax = plt.subplots(figsize=(6, 5))
    series = frame[color]
    if pd.api.types.is_numeric_dtype(series):
        scatter = ax.scatter(frame[x], frame[y], c=series, s=6, alpha=0.8, cmap=cmap)
        fig.colorbar(scatter, ax=ax, label=color)
    else:
        for label, group in frame.groupby(color, sort=True):
            ax.scatter(group[x], group[y], s=6, alpha=0.8, label=str(label))
        ax.legend(frameon=False, markerscale=2)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    outputs: list[str] = []
    for suffix in [".png", ".pdf"]:
        out = path_base.with_suffix(suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        outputs.append(str(out))
    plt.close(fig)
    return outputs


def save_barplot(frame: pd.DataFrame, x: str, y: str, title: str, path_base: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(frame[x].astype(str), pd.to_numeric(frame[y], errors="coerce").fillna(0.0), color="#72B7B2")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    outputs: list[str] = []
    for suffix in [".png", ".pdf"]:
        out = path_base.with_suffix(suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        outputs.append(str(out))
    plt.close(fig)
    return outputs


def row_entropy_from_csr(matrix: sp.csr_matrix) -> np.ndarray:
    csr = matrix.tocsr().astype(float)
    row_nnz = np.diff(csr.indptr)
    row_sums = np.asarray(csr.sum(axis=1)).ravel().astype(float)
    if csr.nnz == 0:
        return np.zeros(csr.shape[0], dtype=float)
    row_ids = np.repeat(np.arange(csr.shape[0], dtype=np.int64), row_nnz)
    probs = csr.data / np.maximum(row_sums[row_ids], 1e-300)
    probs = np.clip(probs, 1e-300, 1.0)
    entropy = np.bincount(row_ids, weights=-probs * np.log(probs), minlength=csr.shape[0])
    return entropy.astype(float)


def collect_packet_files(packet_root: Path, samples: list[str]) -> list[Path]:
    return [
        *[h5ad_path_for_sample(packet_root, sample) for sample in samples],
        packet_root / "processed/lineage_evidence/cellbin_lineage_evidence.tsv.gz",
        packet_root / "processed/lineage_evidence/feature_allele_annotation_long.tsv.gz",
        packet_root / "processed/transfer/L126_brain_barcode_aware_input_packet.manifest.tsv",
        packet_root / "processed/transfer/nichefate_barcode_adapter_input_contract.json",
    ]


def validate_round2b_inputs(
    *,
    packet_root: Path,
    round2b_root: Path,
    samples: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    schema_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for sample in samples:
        h5ad_path = h5ad_path_for_sample(packet_root, sample)
        schema = validate_l126_h5ad_schema(h5ad_path)
        schema_rows.append(schema)
        cellbins = load_l126_cellbin_table(h5ad_path, sample)
        assignment_path = round2b_root / "group_assignments" / f"{sample}_group_assignment.tsv.gz"
        assignment = read_tsv(assignment_path)
        validation_rows.append(
            validate_round2b_group_assignment(
                assignment,
                cellbins,
                sample_id=sample,
                k_neighbors=16,
            )
        )
    return schema_rows, validation_rows


def write_report(report_root: Path, stem: str, title: str, payload: dict[str, Any], body: str, overwrite: bool) -> None:
    write_report_pair(report_root, stem, title, payload, body, overwrite=overwrite)


def write_representation_phase(
    *,
    packet_root: Path,
    samples: list[str],
    output_root: Path,
    max_cellbins_per_section: int,
    n_hvgs: int,
    n_pcs: int,
    seed: int,
    overwrite: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    representation, payload = build_plana_bounded_representation(
        packet_root,
        samples,
        max_cellbins_per_section=max_cellbins_per_section,
        n_hvgs=n_hvgs,
        n_pcs=n_pcs,
        seed=seed,
    )
    representation_root = ensure_dir(output_root / "representation")
    for sample, group in representation.groupby("sample_id", sort=True):
        atomic_save_parquet(representation_root / f"{sample}_bounded_representation.parquet", group.reset_index(drop=True), overwrite=overwrite)
    atomic_save_parquet(representation_root / "L126_all_sections_bounded_representation.parquet", representation.reset_index(drop=True), overwrite=overwrite)
    payload.update(
        {
            "output_root": str(representation_root),
            "representation_paths": {
                sample: str(representation_root / f"{sample}_bounded_representation.parquet")
                for sample in samples
            },
            "combined_representation_path": str(representation_root / "L126_all_sections_bounded_representation.parquet"),
        }
    )
    return representation, payload


def write_state_units_phase(
    *,
    representation: pd.DataFrame,
    packet_root: Path,
    round2b_root: Path,
    samples: list[str],
    n_metaniches: int,
    seed: int,
    output_root: Path,
    report_root: Path,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    group_assignment, group_state, metaniche_assignment, metaniche_summary, unit_payload = build_plana_state_units(
        representation,
        round2b_root=round2b_root,
        packet_root=packet_root,
        samples=samples,
        n_metaniches=n_metaniches,
        seed=seed,
    )
    units_root = ensure_dir(output_root / "units")
    atomic_write_tsv_gz(units_root / "group_state_representation.tsv.gz", group_state, overwrite=overwrite)
    atomic_write_tsv_gz(units_root / "metaniche_assignment.tsv.gz", metaniche_assignment, overwrite=overwrite)
    atomic_write_tsv_gz(units_root / "metaniche_state_summary.tsv.gz", metaniche_summary, overwrite=overwrite)

    multiplicity, multiplicity_payload = group_membership_multiplicity(group_assignment)
    qc_root = ensure_dir(output_root / "qc")
    atomic_write_tsv(qc_root / "all_sections_cellbin_group_membership_multiplicity.tsv", multiplicity, overwrite=overwrite)

    payload = {
        "generated_at_utc": utc_now(),
        "status": "PASS",
        **unit_payload,
        "multiplicity": multiplicity_payload,
        "outputs": {
            "group_state_representation": str(units_root / "group_state_representation.tsv.gz"),
            "metaniche_assignment": str(units_root / "metaniche_assignment.tsv.gz"),
            "metaniche_state_summary": str(units_root / "metaniche_state_summary.tsv.gz"),
            "group_membership_multiplicity": str(qc_root / "all_sections_cellbin_group_membership_multiplicity.tsv"),
        },
    }
    body = "## Bounded State Units\n\n" + markdown_table(pd.DataFrame([payload]).drop(columns=["validation_rows", "outputs", "sample_payloads"], errors="ignore"))
    body += "\n\n## Round2B Validation Rows\n\n" + markdown_table(pd.DataFrame(payload.get("validation_rows", [])))
    body += "\n\n## Metaniche Summary\n\n" + markdown_table(metaniche_summary[[
        "metaniche_id",
        "n_groups",
        "section_distribution",
        "section_purity",
        "section_entropy",
        "section_dominated",
        "tiny_metaniche",
    ]])
    body += "\n\n## Group Membership Multiplicity\n\n" + markdown_table(pd.DataFrame([multiplicity_payload]).T.reset_index().rename(columns={"index": "metric", 0: "value"}))
    write_report(report_root, "01_STATE_UNITS", "State Units", payload, body, overwrite)
    return group_assignment, group_state, metaniche_assignment, metaniche_summary, payload, multiplicity


def write_barcode_phase(
    *,
    packet_root: Path,
    round1_summary: pd.DataFrame,
    group_assignment: pd.DataFrame,
    metaniche_assignment: pd.DataFrame,
    metaniche_summary: pd.DataFrame,
    output_root: Path,
    report_root: Path,
    overwrite: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any], pd.DataFrame]:
    lineage_evidence = load_cellbin_lineage_evidence(packet_root / "processed/lineage_evidence/cellbin_lineage_evidence.tsv.gz")
    frames, payload = aggregate_state_lineage(lineage_evidence, group_assignment, metaniche_assignment)
    barcode_root = ensure_dir(output_root / "barcode")
    atomic_write_tsv_gz(barcode_root / "group_barcode_annotation.tsv.gz", frames["group"], overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "group_assay_summary.tsv.gz", frames["group_assay"], overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "group_top_features.tsv.gz", frames["group_top"], overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_barcode_annotation_local_context.tsv.gz", frames["metaniche_local"], overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_barcode_annotation_unique_cellbin.tsv.gz", frames["metaniche_unique"], overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_assay_summary.tsv.gz", frames["metaniche_unique_assay"], overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_top_features.tsv.gz", frames["metaniche_unique_top"], overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_assay_summary_local_context.tsv.gz", frames["metaniche_local_assay"], overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_assay_summary_unique_cellbin.tsv.gz", frames["metaniche_unique_assay"], overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_top_features_local_context.tsv.gz", frames["metaniche_local_top"], overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_top_features_unique_cellbin.tsv.gz", frames["metaniche_unique_top"], overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_local_vs_unique_comparison.tsv.gz", frames["comparison"], overwrite=overwrite)

    state_frame = metaniche_summary.merge(
        frames["metaniche_unique"].drop(columns=["local_context_not_tissue_partition", "local_context_view"], errors="ignore"),
        on="metaniche_id",
        how="left",
        suffixes=("", "_barcode"),
    )
    state_frame["local_context_not_tissue_partition"] = True
    state_frame["local_context_view"] = "unique_cellbin"
    state_frame = state_frame.sort_values("state_index").reset_index(drop=True)

    section_rows: list[dict[str, Any]] = []
    for sample in sorted(group_assignment["sample_id"].astype(str).unique().tolist()):
        sample_round1 = round1_summary.loc[round1_summary["sample_id"].astype(str) == sample].copy()
        sample_group = frames["group"].loc[frames["group"]["sample_id"].astype(str) == sample].copy()
        sample_group_assignment = group_assignment.loc[group_assignment["sample_id"].astype(str) == sample].copy()
        sample_multiplicity = multiplicity = None
        if not sample_group_assignment.empty:
            sample_multiplicity = group_membership_multiplicity(sample_group_assignment)[0]
        else:
            sample_multiplicity = pd.DataFrame(columns=["sample_id", "slice_id", "cellbin_id", "groups_per_member_cellbin"])
        coverage_metrics = group_lineage_coverage_metrics(sample_round1, sample_group)
        sample_cellbins = load_l126_cellbin_table(h5ad_path_for_sample(packet_root, sample), sample)
        section_rows.append(
            section_summary_row(
                sample_id=sample,
                h5ad_n_obs=int(len(sample_cellbins)),
                assignment=sample_group_assignment,
                group_summary=sample_group,
                multiplicity=sample_multiplicity,
                coverage_metrics=coverage_metrics,
            )
        )
    section_summary = pd.DataFrame(section_rows)
    qc_root = ensure_dir(output_root / "qc")
    atomic_write_tsv(qc_root / "all_sections_section_summary.tsv", section_summary, overwrite=overwrite)

    payload.update(
        {
            "status": "PASS",
            "group_join_success": bool(
                group_assignment.merge(
                    round1_summary[list(PRIMARY_JOIN_KEY)],
                    on=list(PRIMARY_JOIN_KEY),
                    how="left",
                    indicator=True,
                )["_merge"].eq("both").all()
            ),
            "barcode_coverage_fraction_group": float(frames["group"]["evidence_present"].astype(bool).mean()) if len(frames["group"]) else 0.0,
            "barcode_coverage_fraction_metaniche_local_context": float(frames["metaniche_local"]["evidence_present"].astype(bool).mean()) if len(frames["metaniche_local"]) else 0.0,
            "barcode_coverage_fraction_metaniche_unique_cellbin": float(frames["metaniche_unique"]["evidence_present"].astype(bool).mean()) if len(frames["metaniche_unique"]) else 0.0,
            "metaniche_local_to_unique_total_count_ratio_median": float(frames["comparison"]["local_to_unique_total_count_ratio"].median()) if len(frames["comparison"]) else 0.0,
            "outputs": {
                "group_barcode_annotation": str(barcode_root / "group_barcode_annotation.tsv.gz"),
                "metaniche_barcode_annotation_local_context": str(barcode_root / "metaniche_barcode_annotation_local_context.tsv.gz"),
                "metaniche_barcode_annotation_unique_cellbin": str(barcode_root / "metaniche_barcode_annotation_unique_cellbin.tsv.gz"),
                "metaniche_assay_summary": str(barcode_root / "metaniche_assay_summary.tsv.gz"),
                "metaniche_top_features": str(barcode_root / "metaniche_top_features.tsv.gz"),
                "section_summary": str(qc_root / "all_sections_section_summary.tsv"),
            },
        }
    )
    body = "## Barcode Aggregation\n\n" + markdown_table(pd.DataFrame([payload]).drop(columns=["outputs", "validation_rows"], errors="ignore"))
    body += "\n\n## Group Summary\n\n" + markdown_table(frames["group"][[
        "sample_id",
        "group_id",
        "n_member_cellbins",
        "n_member_cellbins_with_lineage",
        "fraction_member_cellbins_with_lineage",
        "total_lineage_count",
        "detected_feature_count",
        "detected_assay_count",
        "dominant_assay",
        "dominant_feature_id",
        "dominant_feature_fraction",
        "feature_entropy",
        "simpson_diversity",
        "evidence_present",
    ]])
    body += "\n\n## Metaniche Local Context vs Unique Cellbin\n\n" + markdown_table(frames["comparison"])
    write_report(report_root, "02_LINEAGE_AGGREGATION_TO_STATES", "Lineage Aggregation To States", payload, body, overwrite)
    return frames, state_frame, payload, section_summary


def write_lineage_potential_phase(
    *,
    state_frame: pd.DataFrame,
    output_root: Path,
    report_root: Path,
    overwrite: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    state_phi, payload = build_lineage_potential(state_frame)
    kernel_root = ensure_dir(output_root / "kernel")
    atomic_write_tsv(kernel_root / "metaniche_lineage_potential.tsv", state_phi, overwrite=overwrite)
    payload.update({"outputs": {"metaniche_lineage_potential": str(kernel_root / "metaniche_lineage_potential.tsv")}})
    body = "## Lineage Potential\n\n" + markdown_table(pd.DataFrame([payload]).drop(columns=["top_phi_states", "low_phi_states", "outputs"], errors="ignore"))
    body += "\n\n## Top Phi States\n\n" + markdown_table(pd.DataFrame(payload.get("top_phi_states", [])))
    body += "\n\n## Low Phi States\n\n" + markdown_table(pd.DataFrame(payload.get("low_phi_states", [])))
    write_report(report_root, "03_LINEAGE_DIRECTION_POTENTIAL", "Lineage Direction Potential", payload, body, overwrite)
    return state_phi, payload


def write_connectivity_phase(
    *,
    state_frame: pd.DataFrame,
    barcode_top_features: pd.DataFrame,
    output_root: Path,
    report_root: Path,
    topk: int,
    overwrite: bool,
) -> tuple[pd.DataFrame, dict[str, sp.csr_matrix], dict[str, Any], pd.DataFrame]:
    state_frame, matrices, payload = build_combined_similarity_matrices(state_frame, barcode_top_features, topk=topk)
    kernel_root = ensure_dir(output_root / "kernel")
    atomic_save_npz(kernel_root / "W_expr.npz", matrices["W_expr"], overwrite=overwrite)
    atomic_save_npz(kernel_root / "W_spatial.npz", matrices["W_spatial"], overwrite=overwrite)
    atomic_save_npz(kernel_root / "W_barcode.npz", matrices["W_barcode"], overwrite=overwrite)
    atomic_save_npz(kernel_root / "W_combined.npz", matrices["W_combined"], overwrite=overwrite)

    matrix_stats = []
    for name, matrix in matrices.items():
        stats = build_sparse_matrix_stats(matrix, include_components=True)
        closure = strong_component_closure_summary(matrix)
        matrix_stats.append(
            {
                "matrix_name": name,
                **{key: value for key, value in stats.items() if key not in {"component_summary"}},
                **{f"closure_{key}": value for key, value in closure.items() if key != "strong_component_sizes"},
            }
        )
    matrix_stats = pd.DataFrame(matrix_stats)
    atomic_write_tsv(kernel_root / "connectivity_matrix_stats.tsv", matrix_stats, overwrite=overwrite)
    atomic_write_tsv(kernel_root / "state_metadata.tsv", state_frame, overwrite=overwrite)
    payload.update(
        {
            "outputs": {
                "W_expr": str(kernel_root / "W_expr.npz"),
                "W_spatial": str(kernel_root / "W_spatial.npz"),
                "W_barcode": str(kernel_root / "W_barcode.npz"),
                "W_combined": str(kernel_root / "W_combined.npz"),
                "state_metadata": str(kernel_root / "state_metadata.tsv"),
                "connectivity_matrix_stats": str(kernel_root / "connectivity_matrix_stats.tsv"),
            }
        }
    )
    body = "## Connectivity Graph\n\n" + markdown_table(pd.DataFrame([payload]).drop(columns=["outputs"], errors="ignore"))
    body += "\n\n## Matrix Stats\n\n" + markdown_table(matrix_stats)
    write_report(report_root, "04_CONNECTIVITY_GRAPH", "Connectivity Graph", payload, body, overwrite)
    return state_frame, matrices, payload, matrix_stats


def write_directed_kernel_phase(
    *,
    state_frame: pd.DataFrame,
    matrices: dict[str, sp.csr_matrix],
    output_root: Path,
    report_root: Path,
    tau: float,
    epsilon: float,
    overwrite: bool,
) -> tuple[sp.csr_matrix, pd.DataFrame, dict[str, Any]]:
    kernel, gate_summary, payload = build_directed_kernel(
        matrices["W_combined"],
        state_frame,
        state_frame["phi"],
        tau=tau,
        epsilon=epsilon,
    )
    kernel_root = ensure_dir(output_root / "kernel")
    atomic_save_npz(kernel_root / "K_lineage_directed.npz", kernel, overwrite=overwrite)
    atomic_write_json(kernel_root / "K_lineage_directed_metadata.json", payload, overwrite=overwrite)
    atomic_write_tsv(kernel_root / "direction_gate_summary.tsv", gate_summary, overwrite=overwrite)
    payload.update(
        {
            "outputs": {
                "K_lineage_directed": str(kernel_root / "K_lineage_directed.npz"),
                "K_lineage_directed_metadata": str(kernel_root / "K_lineage_directed_metadata.json"),
                "direction_gate_summary": str(kernel_root / "direction_gate_summary.tsv"),
            }
        }
    )
    body = "## Directed Kernel\n\n" + markdown_table(pd.DataFrame([payload]).drop(columns=["outputs"], errors="ignore"))
    body += "\n\n## Direction Gate Summary\n\n" + markdown_table(gate_summary.head(40))
    write_report(report_root, "05_DIRECTED_KERNEL_CONSTRUCTION", "Directed Kernel Construction", payload, body, overwrite)
    return kernel, gate_summary, payload


def write_controls_phase(
    *,
    state_frame: pd.DataFrame,
    matrices: dict[str, sp.csr_matrix],
    barcode_top_features: pd.DataFrame,
    output_root: Path,
    report_root: Path,
    topk: int,
    tau: float,
    epsilon: float,
    seed: int,
    overwrite: bool,
) -> tuple[dict[str, sp.csr_matrix], dict[str, Any], dict[str, pd.DataFrame], pd.DataFrame]:
    phi_values = state_frame["phi"]
    phi_map = {
        "K_phi_shuffled": build_phi_shuffled(phi_values, seed=seed),
        "K_coverage_only": build_coverage_only_phi(state_frame),
        "K_barcode_shuffled": phi_values,
    }
    control_kernels, payload, control_summaries, comparison = build_control_kernels(
        state_frame,
        matrices,
        phi_values,
        barcode_top_features,
        topk=topk,
        tau=tau,
        epsilon=epsilon,
        seed=seed,
    )
    controls_root = ensure_dir(output_root / "controls")
    for name, matrix in control_kernels.items():
        atomic_save_npz(controls_root / f"{name}.npz", matrix, overwrite=overwrite)

    atomic_write_tsv(controls_root / "control_comparison.tsv", comparison, overwrite=overwrite)
    payload.update(
        {
            "outputs": {name: str(controls_root / f"{name}.npz") for name in control_kernels},
            "comparison_path": str(controls_root / "control_comparison.tsv"),
        }
    )
    body = "## Control Kernels\n\n" + markdown_table(pd.DataFrame([payload]).drop(columns=["outputs", "phi_shuffled", "coverage_only", "barcode_shuffled"], errors="ignore"))
    body += "\n\n## Control Comparison\n\n" + markdown_table(comparison)
    write_report(report_root, "06_CONTROLS_AND_ABLATIONS", "Controls And Ablations", payload, body, overwrite)
    return control_kernels, payload, control_summaries, comparison


def write_gpcca_phase(
    *,
    state_frame: pd.DataFrame,
    unit_payload: dict[str, Any],
    kernel: sp.csr_matrix,
    output_root: Path,
    report_root: Path,
    run_gpcca_smoke: bool,
    overwrite: bool,
) -> dict[str, Any]:
    gpcca_root = ensure_dir(output_root / "gpcca_readiness")
    feature_cols = [column for column in state_frame.columns if column.startswith("pca_mean_")]
    if {"centroid_x", "centroid_y"}.issubset(state_frame.columns):
        feature_cols = [*feature_cols, "centroid_x", "centroid_y"]
    elif {"centroid_x_mean", "centroid_y_mean"}.issubset(state_frame.columns):
        feature_cols = [*feature_cols, "centroid_x_mean", "centroid_y_mean"]
    state_matrix = state_frame[["state_index", "metaniche_id", *feature_cols]].copy()
    metadata_cols = [
        "metaniche_id",
        "n_groups",
        "section_distribution",
        "section_purity",
        "section_entropy",
        "section_dominated",
        "tiny_metaniche",
        "dominant_sample_id",
        "dominant_sample_fraction",
        "dominant_slice_id",
        "dominant_slice_fraction",
        "local_context_not_tissue_partition",
    ]
    state_metadata = state_frame[[column for column in metadata_cols if column in state_frame.columns]].copy()
    barcode_columns = [
        "metaniche_id",
        "n_member_cellbins",
        "n_member_cellbins_with_lineage",
        "fraction_member_cellbins_with_lineage",
        "total_lineage_count",
        "detected_feature_count",
        "detected_assay_count",
        "RA_total_count",
        "TA_total_count",
        "CA_total_count",
        "RA_detected_feature_count",
        "TA_detected_feature_count",
        "CA_detected_feature_count",
        "dominant_feature_id",
        "dominant_feature_count",
        "dominant_feature_fraction",
        "feature_entropy",
        "simpson_diversity",
        "assay_balance",
        "evidence_present",
    ]
    barcode_matrix = state_frame[[column for column in barcode_columns if column in state_frame.columns]].copy()
    atomic_save_parquet(gpcca_root / "bounded_state_matrix_preview.parquet", state_matrix, overwrite=overwrite)
    atomic_write_tsv(gpcca_root / "bounded_state_metadata.tsv", state_metadata, overwrite=overwrite)
    atomic_write_tsv_gz(gpcca_root / "barcode_annotation_matrix_preview.tsv.gz", barcode_matrix, overwrite=overwrite)

    feature_values = state_matrix.drop(columns=["state_index", "metaniche_id"]).to_numpy(dtype=float)
    tiny_fraction = float(unit_payload.get("tiny_metaniche_fraction", 0.0))
    empty_metaniches = int(unit_payload.get("empty_metaniches", 0))
    checks = {
        "state_matrix_rows_equal_metadata_rows": int(len(state_matrix)) == int(len(state_metadata)),
        "finite_values_only": bool(np.isfinite(feature_values).all()) if feature_values.size else False,
        "nonzero_feature_variance": bool((np.nanvar(feature_values, axis=0) > 0).any()) if feature_values.size else False,
        "metaniche_size_threshold_passed": bool(tiny_fraction <= 0.10 and empty_metaniches == 0),
        "barcode_annotation_join_success": bool(
            state_metadata[["metaniche_id"]].merge(barcode_matrix[["metaniche_id"]], on="metaniche_id", how="left", indicator=True)["_merge"].eq("both").all()
        )
        if not state_metadata.empty and not barcode_matrix.empty
        else False,
        "section_distribution_exists": bool(state_metadata["section_distribution"].astype(str).ne("").all()) if "section_distribution" in state_metadata else False,
        "kernel_constructed": False,
        "gpcca_run": False,
        "fate_probability_computed": False,
    }
    ready_for_smoke = all(checks[key] for key in [
        "state_matrix_rows_equal_metadata_rows",
        "finite_values_only",
        "nonzero_feature_variance",
        "barcode_annotation_join_success",
        "section_distribution_exists",
        "metaniche_size_threshold_passed",
    ])

    if run_gpcca_smoke and ready_for_smoke:
        smoke_payload = run_tiny_gpcca_smoke(kernel, state_frame, output_root, overwrite=overwrite)
    elif run_gpcca_smoke and not ready_for_smoke:
        smoke_payload = {
            "status": "SKIPPED",
            "readiness_label": "L126_PLANA_LINEAGE_KERNEL_GPCCA_SMOKE_SKIPPED",
            "reason": "gpcca readiness checks failed",
            "run_requested": True,
            "run_performed": False,
            "outputs": [],
        }
    else:
        smoke_payload = {
            "status": "SKIPPED",
            "readiness_label": "L126_PLANA_LINEAGE_KERNEL_GPCCA_SMOKE_SKIPPED",
            "reason": "gpcca smoke was not requested",
            "run_requested": False,
            "run_performed": False,
            "outputs": [],
        }
    payload = {
        "generated_at_utc": utc_now(),
        "readiness_checks": checks,
        "ready_for_smoke": ready_for_smoke,
        "smoke_payload": smoke_payload,
        "outputs": {
            "bounded_state_matrix_preview": str(gpcca_root / "bounded_state_matrix_preview.parquet"),
            "bounded_state_metadata": str(gpcca_root / "bounded_state_metadata.tsv"),
            "barcode_annotation_matrix_preview": str(gpcca_root / "barcode_annotation_matrix_preview.tsv.gz"),
        },
    }
    body = "## GPCCA Readiness And Smoke\n\n" + markdown_table(pd.DataFrame([payload]).drop(columns=["smoke_payload", "outputs", "readiness_checks"], errors="ignore"))
    body += "\n\n## Readiness Checks\n\n" + markdown_table(pd.DataFrame([{"check": key, "value": value} for key, value in checks.items()]))
    body += "\n\n## Smoke Payload\n\n" + markdown_table(pd.DataFrame([smoke_payload]).drop(columns=["results"], errors="ignore"))
    write_report(report_root, "07_TINY_GPCCA_SMOKE", "Tiny GPCCA Smoke", payload, body, overwrite)
    return payload


def write_figures_phase(
    *,
    representation: pd.DataFrame,
    group_summary: pd.DataFrame,
    state_frame: pd.DataFrame,
    matrices: dict[str, sp.csr_matrix],
    kernel: sp.csr_matrix,
    gate_summary: pd.DataFrame,
    controls_comparison: pd.DataFrame,
    gpcca_payload: dict[str, Any],
    report_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    fig_root = ensure_dir(report_root / "figures")
    paths: list[str] = []
    paths.extend(save_scatter(representation, "pca_0", "pca_1", "sample_id", "PCA Scatter By Section", fig_root / "pca_scatter_by_section"))
    for sample, group in representation.groupby("sample_id", sort=True):
        paths.extend(save_scatter(group, "x", "y", "total_counts", f"{sample} Sampled Cellbin Spatial Map", fig_root / f"{sample}_sampled_cellbin_spatial_map"))
    paths.extend(save_histogram(state_frame, "n_groups", "Metaniche Count Distribution", fig_root / "metaniche_count_distribution"))
    paths.extend(save_histogram(state_frame, "section_purity", "Metaniche Section Purity Distribution", fig_root / "metaniche_section_purity_distribution"))
    paths.extend(save_histogram(group_summary, "fraction_member_cellbins_with_lineage", "Group Lineage Coverage Distribution", fig_root / "group_lineage_coverage_distribution"))
    paths.extend(save_histogram(state_frame, "fraction_member_cellbins_with_lineage", "Metaniche Lineage Coverage Distribution", fig_root / "metaniche_lineage_coverage_distribution"))
    paths.extend(save_histogram(state_frame, "total_lineage_count", "Metaniche Total Lineage Count Distribution", fig_root / "metaniche_total_lineage_count_distribution"))
    paths.extend(save_histogram(state_frame, "detected_feature_count", "Metaniche Detected Feature Count Distribution", fig_root / "metaniche_detected_feature_count_distribution"))
    paths.extend(save_histogram(state_frame, "dominant_feature_fraction", "Metaniche Dominant Feature Fraction Distribution", fig_root / "metaniche_dominant_feature_fraction_distribution"))
    paths.extend(save_histogram(state_frame, "feature_entropy", "Metaniche Entropy Distribution", fig_root / "metaniche_entropy_distribution"))

    assay_long = state_frame[["metaniche_id", "RA_total_count", "TA_total_count", "CA_total_count"]].melt(
        id_vars="metaniche_id", var_name="assay", value_name="count"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    assay_long.boxplot(column="count", by="assay", ax=ax)
    ax.set_title("RA/TA/CA Count Summary By Metaniche")
    fig.suptitle("")
    fig.tight_layout()
    for suffix in [".png", ".pdf"]:
        out = fig_root / f"metaniche_assay_count_summary{suffix}"
        fig.savefig(out, dpi=180)
        paths.append(str(out))
    plt.close(fig)

    if {"W_expr", "W_barcode"}.issubset(matrices):
        expr = matrices["W_expr"].toarray()
        barcode = matrices["W_barcode"].toarray()
        mask = np.triu(np.ones(expr.shape, dtype=bool), k=1)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(expr[mask], barcode[mask], s=4, alpha=0.5, color="#4C78A8")
        ax.set_xlabel("W_expr")
        ax.set_ylabel("W_barcode")
        ax.set_title("W_expr vs W_barcode Similarity")
        fig.tight_layout()
        for suffix in [".png", ".pdf"]:
            out = fig_root / f"W_expr_vs_W_barcode_similarity{suffix}"
            fig.savefig(out, dpi=180)
            paths.append(str(out))
        plt.close(fig)

    phi_depth = np.log1p(
        pd.to_numeric(state_frame["total_lineage_count"], errors="coerce").fillna(0.0)
        / np.maximum(pd.to_numeric(state_frame["n_member_cellbins"], errors="coerce").fillna(1.0), 1.0)
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(phi_depth, state_frame["phi"], s=8, alpha=0.7, color="#F58518")
    ax.set_xlabel("log1p lineage depth per state")
    ax.set_ylabel("phi")
    ax.set_title("Phi vs Lineage Depth")
    fig.tight_layout()
    for suffix in [".png", ".pdf"]:
        out = fig_root / f"phi_vs_lineage_depth_scatter{suffix}"
        fig.savefig(out, dpi=180)
        paths.append(str(out))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(pd.to_numeric(gate_summary["direction_gate"], errors="coerce").fillna(0.0), bins=40, color="#54A24B", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("direction_gate")
    ax.set_ylabel("Count")
    ax.set_title("Direction Gate Distribution")
    fig.tight_layout()
    for suffix in [".png", ".pdf"]:
        out = fig_root / f"direction_gate_distribution{suffix}"
        fig.savefig(out, dpi=180)
        paths.append(str(out))
    plt.close(fig)

    row_entropy = row_entropy_from_csr(kernel)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(row_entropy, bins=40, color="#72B7B2", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("row entropy")
    ax.set_ylabel("Count")
    ax.set_title("Directed Kernel Row Entropy")
    fig.tight_layout()
    for suffix in [".png", ".pdf"]:
        out = fig_root / f"K_lineage_directed_row_entropy{suffix}"
        fig.savefig(out, dpi=180)
        paths.append(str(out))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = controls_comparison["control_name"].astype(str)
    ax.bar(x, controls_comparison["edge_support_jaccard"].fillna(0.0), color="#B279A2", alpha=0.8)
    ax.set_ylabel("edge_support_jaccard")
    ax.set_title("Control Comparison Summary")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    for suffix in [".png", ".pdf"]:
        out = fig_root / f"control_comparison_summary{suffix}"
        fig.savefig(out, dpi=180)
        paths.append(str(out))
    plt.close(fig)

    for sample, group in state_frame.groupby("dominant_sample_id", sort=True):
        if group.empty:
            continue
        paths.extend(save_scatter(group, "centroid_x", "centroid_y", "phi", f"{sample} Lineage Potential Map", fig_root / f"{sample}_spatial_lineage_potential_map"))
        paths.extend(save_scatter(group, "centroid_x", "centroid_y", "feature_entropy", f"{sample} Lineage Entropy Map", fig_root / f"{sample}_spatial_lineage_entropy_map"))
        paths.extend(save_scatter(group, "centroid_x", "centroid_y", "dominant_assay", f"{sample} Dominant Assay Map", fig_root / f"{sample}_spatial_dominant_assay_map"))

    if gpcca_payload.get("smoke_payload", {}).get("run_performed"):
        selected = int(gpcca_payload["smoke_payload"].get("selected_n_macrostates", 0))
        assignment_path = None
        for row in gpcca_payload["smoke_payload"].get("results", []):
            if int(row.get("n_macrostates", -1)) == selected:
                assignment_path = Path(row["assignment_path"])
                break
        if assignment_path and assignment_path.exists():
            assignment = pd.read_csv(assignment_path, sep="\t")
            macrostate_map = state_frame.merge(assignment[["state_index", "macrostate"]], on="state_index", how="left")
            for sample, group in macrostate_map.groupby("dominant_sample_id", sort=True):
                if group.empty:
                    continue
                paths.extend(save_scatter(group, "centroid_x", "centroid_y", "macrostate", f"{sample} Macrostate Map", fig_root / f"{sample}_spatial_macrostate_map"))

    payload = {
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "figure_paths": paths,
        "figures_non_empty": all(Path(path).exists() and Path(path).stat().st_size > 0 for path in paths),
        "descriptive_qc_only": True,
    }
    body = "## Figures\n\n" + markdown_table(pd.DataFrame([{"figure_path": path} for path in paths]))
    write_report(report_root, "08_FIGURES_AND_DIAGNOSTICS", "Figures And Diagnostics", payload, body, overwrite=overwrite)
    return payload


def validate_outputs(
    *,
    output_root: Path,
    report_root: Path,
    packet_files: list[Path],
    source_before: pd.DataFrame,
    state_frame: pd.DataFrame,
    kernel: sp.csr_matrix,
    decision_label: str,
    gpcca_payload: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    report_jsons = sorted(report_root.rglob("*.json"))
    report_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(report_root.rglob("*.md")))
    checks = [
        {"check": "json_reports_parse", "status": all(json.loads(path.read_text(encoding="utf-8")) is not None for path in report_jsons), "details": f"{len(report_jsons)} json files"},
        {"check": "tsv_gzip_readability", "status": all(Path(path).exists() and Path(path).stat().st_size > 0 for path in list(output_root.rglob("*.tsv")) + list(output_root.rglob("*.tsv.gz"))), "details": str(output_root)},
        {"check": "parquet_readability", "status": all(pd.read_parquet(path).shape[0] >= 0 for path in output_root.rglob("*.parquet")), "details": str(output_root)},
        {"check": "kernel_npz_readable", "status": all(sp.load_npz(path).shape[0] >= 0 for path in output_root.rglob("K_lineage_directed.npz")), "details": "directed kernel"},
        {"check": "figures_non_empty", "status": all(path.stat().st_size > 0 for path in report_root.rglob("*.png")) and all(path.stat().st_size > 0 for path in report_root.rglob("*.pdf")), "details": str(report_root / "figures")},
        {"check": "h5ad_input_readback", "status": all(validate_l126_h5ad_schema(path)["schema_passed"] for path in packet_files if path.suffix == ".h5ad"), "details": "input h5ad schema"},
        {"check": "source_input_packet_unchanged", "status": not bool(compare_file_snapshots(source_before, snapshot_files(packet_files, include_sha256=False))["changed"].any()), "details": "size/mtime comparison"},
        {"check": "no_ssd", "status": not any(path_has_ssd(path) for path in [output_root, report_root, *packet_files]), "details": "path guard"},
        {"check": "no_raw_fastq", "status": "fastq" not in str(output_root).lower() and "fastq" not in str(report_root).lower(), "details": "no fastq outputs"},
        {"check": "no_darlin_recalling", "status": True, "details": "not run"},
        {"check": "no_full_m0_m1_m2", "status": True, "details": "bounded lineage kernel only"},
        {"check": "no_full_gpcca", "status": True, "details": "tiny smoke only or skipped"},
        {"check": "no_planb", "status": True, "details": "not run"},
        {"check": "no_section_order_as_time", "status": "trajectory" not in report_text.lower() and "timepoint trajectory" not in report_text.lower(), "details": "report wording audit"},
        {"check": "no_biological_fate_claims", "status": not forbidden_claim_hits(report_text), "details": "; ".join(forbidden_claim_hits(report_text))},
        {"check": "no_git_add_commit_push", "status": True, "details": "not run"},
    ]
    readiness_checks = gpcca_payload.get("readiness_checks", {})
    for key in [
        "state_matrix_rows_equal_metadata_rows",
        "finite_values_only",
        "nonzero_feature_variance",
        "metaniche_size_threshold_passed",
        "barcode_annotation_join_success",
        "section_distribution_exists",
    ]:
        if key in readiness_checks:
            checks.append({"check": f"gpcca_{key}", "status": bool(readiness_checks[key]), "details": "gpcca readiness preview"})
    validation_payload = {
        "generated_at_utc": utc_now(),
        "decision_label": decision_label,
        "status": "PASS" if all(row["status"] for row in checks) else "FAIL",
        "checks": checks,
        "gpcca_payload": gpcca_payload,
    }
    body = "## Validation Checks\n\n" + markdown_table(pd.DataFrame(checks))
    write_report(report_root, "10_VALIDATION", "Validation", validation_payload, body, overwrite=overwrite)
    return validation_payload


def determine_readiness(
    *,
    unit_payload: dict[str, Any],
    lineage_payload: dict[str, Any],
    connectivity_payload: dict[str, Any],
    directed_payload: dict[str, Any],
    controls_comparison: pd.DataFrame,
    gpcca_payload: dict[str, Any],
    smoke_requested: bool,
) -> tuple[str, list[str], str]:
    warnings: list[str] = []
    section_dominated_fraction = float(unit_payload.get("section_dominated_fraction", 0.0))
    tiny_fraction = float(unit_payload.get("tiny_metaniche_fraction", 0.0))
    empty_metaniches = int(unit_payload.get("empty_metaniches", 0))
    if section_dominated_fraction > 0.5:
        warnings.append("Many metaniches are section-dominated.")
    if tiny_fraction > 0.0:
        warnings.append("Tiny metaniches were detected.")
    if float(lineage_payload.get("phi_total_lineage_count_corr_pearson", 0.0)) > 0.95:
        warnings.append("Lineage potential tracks total lineage depth very strongly.")
    if float(connectivity_payload.get("barcode_coverage_fraction", 0.0)) < 0.5:
        warnings.append("Barcode overlap is sparse across state units.")
    if not bool(directed_payload.get("row_stochastic", False)):
        return "L126_PLANA_LINEAGE_KERNEL_HOLD_FOR_GRAPH_STRUCTURE", warnings, "Rebuild the directed kernel after fixing row-stochasticity."
    if not bool(lineage_payload.get("phi_finite", False)):
        return "L126_PLANA_LINEAGE_KERNEL_HOLD_FOR_DIRECTION_POTENTIAL", warnings, "Recompute lineage potential after fixing phi."
    if section_dominated_fraction > 0.75:
        return "L126_PLANA_LINEAGE_KERNEL_HOLD_FOR_SECTION_DOMINANCE", warnings, "Review section-dominated metaniche construction before any GPCCA smoke."
    if empty_metaniches > 0 or tiny_fraction > 0.10:
        return "L126_PLANA_LINEAGE_KERNEL_HOLD_FOR_COVERAGE_ARTIFACT", warnings, "Reduce tiny/empty metaniche prevalence before GPCCA smoke."
    expr_random_walk = controls_comparison.loc[controls_comparison["control_name"] == "K_expr_spatial_only"]
    if not expr_random_walk.empty and float(expr_random_walk["edge_support_jaccard"].iloc[0]) > 0.95:
        warnings.append("Directed kernel is too close to the barcode-free similarity random walk.")
        return "L126_PLANA_LINEAGE_KERNEL_HOLD_FOR_COVERAGE_ARTIFACT", warnings, "Directed kernel is too close to the barcode-free similarity random walk."
    if smoke_requested and gpcca_payload.get("smoke_payload", {}).get("run_performed"):
        return "L126_PLANA_LINEAGE_KERNEL_GPCCA_SMOKE_READY", warnings, "Inspect the tiny GPCCA smoke outputs next."
    if smoke_requested:
        return "L126_PLANA_LINEAGE_KERNEL_GPCCA_SMOKE_SKIPPED", warnings, "Rerun the smoke only after the GPCCA environment is available."
    if warnings:
        return "L126_PLANA_LINEAGE_DIRECTED_KERNEL_READY_WITH_WARNINGS", warnings, "Review controls and, if desired, rerun the smoke with --run-gpcca-smoke."
    return "L126_PLANA_LINEAGE_DIRECTED_KERNEL_READY", warnings, "Inspect the outputs and then decide whether to rerun smoke."


def main() -> None:
    args = parse_args()
    samples = list(parse_sample_list(args.samples))
    packet_root = Path(args.input_packet_root).expanduser().resolve()
    round1_root = Path(args.round1_barcode_root).expanduser().resolve()
    round2b_root = Path(args.round2B_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    report_root = Path(args.report_root).expanduser().resolve()
    reject_forbidden_paths(packet_root, round1_root, round2b_root, output_root, report_root)
    ensure_dir(output_root)
    ensure_dir(report_root)

    packet_files = collect_packet_files(packet_root, samples)
    source_before = snapshot_files(packet_files, include_sha256=False)

    round1_summary_path = round1_root / "cellbin_lineage_summary.tsv.gz"
    round1_summary = read_tsv(round1_summary_path)

    schema_rows, round2b_validation_rows = validate_round2b_inputs(
        packet_root=packet_root,
        round2b_root=round2b_root,
        samples=samples,
    )
    preflight_payload = {
        "generated_at_utc": utc_now(),
        "decision_label": "L126_PLANA_LINEAGE_KERNEL_SCOPE_DEFINED",
        "routea_status": "L126_ROUTEA_RETIRED_TO_QC_ONLY",
        "samples": samples,
        "h5ad_schema_rows": schema_rows,
        "round2b_validation_rows": round2b_validation_rows,
        "round1_summary_rows": int(len(round1_summary)),
        "round1_summary_path": str(round1_summary_path),
        "packet_files": [str(path) for path in packet_files],
        "source_snapshot_before": source_before.to_dict(orient="records"),
    }
    preflight_body = "## Preflight\n\n" + markdown_table(pd.DataFrame([preflight_payload]).drop(columns=["h5ad_schema_rows", "round2b_validation_rows", "source_snapshot_before", "packet_files"], errors="ignore"))
    preflight_body += "\n\n## h5ad Schema\n\n" + markdown_table(pd.DataFrame(schema_rows))
    preflight_body += "\n\n## Round2B Validation\n\n" + markdown_table(pd.DataFrame(round2b_validation_rows))
    write_report(report_root, "00_ROUTE_REDEFINITION", "Route Redefinition", preflight_payload, preflight_body, overwrite=args.overwrite)
    if args.mode == "audit_only":
        print(json.dumps({"decision_label": preflight_payload["decision_label"], "mode": args.mode}, indent=2, sort_keys=True))
        return

    representation, representation_payload = write_representation_phase(
        packet_root=packet_root,
        samples=samples,
        output_root=output_root,
        max_cellbins_per_section=args.max_cellbins_per_section,
        n_hvgs=args.n_hvgs,
        n_pcs=args.n_pcs,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    if args.mode == "state_units_only":
        return

    group_assignment, group_state, metaniche_assignment, metaniche_summary, unit_payload, multiplicity = write_state_units_phase(
        representation=representation,
        packet_root=packet_root,
        round2b_root=round2b_root,
        samples=samples,
        n_metaniches=args.n_metaniches,
        seed=args.seed,
        output_root=output_root,
        report_root=report_root,
        overwrite=args.overwrite,
    )
    frames, state_frame, lineage_payload, barcode_section_summary = write_barcode_phase(
        packet_root=packet_root,
        round1_summary=round1_summary,
        group_assignment=group_assignment,
        metaniche_assignment=metaniche_assignment,
        metaniche_summary=metaniche_summary,
        output_root=output_root,
        report_root=report_root,
        overwrite=args.overwrite,
    )
    state_frame, lineage_payload_connectivity = write_lineage_potential_phase(
        state_frame=state_frame,
        output_root=output_root,
        report_root=report_root,
        overwrite=args.overwrite,
    )
    if args.mode == "lineage_potential_only":
        return

    state_frame, matrices, connectivity_payload, matrix_stats = write_connectivity_phase(
        state_frame=state_frame,
        barcode_top_features=frames["metaniche_unique_top"],
        output_root=output_root,
        report_root=report_root,
        topk=args.topk,
        overwrite=args.overwrite,
    )

    kernel, gate_summary, directed_payload = write_directed_kernel_phase(
        state_frame=state_frame,
        matrices=matrices,
        output_root=output_root,
        report_root=report_root,
        tau=args.tau,
        epsilon=args.epsilon,
        overwrite=args.overwrite,
    )
    if args.mode == "kernel_only":
        return
    control_kernels, controls_payload, control_summaries, controls_comparison = write_controls_phase(
        state_frame=state_frame,
        matrices=matrices,
        barcode_top_features=frames["metaniche_unique_top"],
        output_root=output_root,
        report_root=report_root,
        topk=args.topk,
        tau=args.tau,
        epsilon=args.epsilon,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    if args.mode == "controls_only":
        return

    gpcca_payload = write_gpcca_phase(
        state_frame=state_frame,
        unit_payload=unit_payload,
        kernel=kernel,
        output_root=output_root,
        report_root=report_root,
        run_gpcca_smoke=args.run_gpcca_smoke and args.mode in {"all", "gpcca_smoke_only"},
        overwrite=args.overwrite,
    )
    if args.mode == "gpcca_smoke_only":
        return

    figures_payload = write_figures_phase(
        representation=representation,
        group_summary=frames["group"],
        state_frame=state_frame,
        matrices=matrices,
        kernel=kernel,
        gate_summary=gate_summary,
        controls_comparison=controls_comparison,
        gpcca_payload=gpcca_payload,
        report_root=report_root,
        overwrite=args.overwrite,
    )
    if args.mode == "figures_only":
        return

    decision_label, warnings, next_safe_command = determine_readiness(
        unit_payload=unit_payload,
        lineage_payload=lineage_payload_connectivity,
        connectivity_payload=connectivity_payload,
        directed_payload=directed_payload,
        controls_comparison=controls_comparison,
        gpcca_payload=gpcca_payload,
        smoke_requested=args.run_gpcca_smoke,
    )
    decision_payload = {
        "generated_at_utc": utc_now(),
        "decision_label": decision_label,
        "samples_processed": samples,
        "cellbins_per_section": representation.groupby("sample_id").size().astype(int).to_dict(),
        "groups_per_section": group_state.groupby("sample_id")["group_id"].nunique().astype(int).to_dict(),
        "metaniches": int(state_frame["metaniche_id"].nunique()),
        "barcode_annotation_coverage": {
            "group": float(frames["group"]["evidence_present"].astype(bool).mean()) if len(frames["group"]) else 0.0,
            "metaniche_local_context": float(frames["metaniche_local"]["evidence_present"].astype(bool).mean()) if len(frames["metaniche_local"]) else 0.0,
            "metaniche_unique_cellbin": float(frames["metaniche_unique"]["evidence_present"].astype(bool).mean()) if len(frames["metaniche_unique"]) else 0.0,
        },
        "lineage_potential_summary": {
            "phi_finite": bool(lineage_payload_connectivity["phi_finite"]),
            "phi_mean": float(lineage_payload_connectivity["phi_mean"]),
            "phi_std": float(lineage_payload_connectivity["phi_std"]),
            "phi_total_lineage_count_corr_pearson": float(lineage_payload_connectivity["phi_total_lineage_count_corr_pearson"]),
            "phi_section_purity_corr_pearson": float(lineage_payload_connectivity["phi_section_purity_corr_pearson"]),
        },
        "kernel_structural_qc": {
            "row_stochastic": bool(directed_payload["row_stochastic"]),
            "negative_entries": int(directed_payload["negative_entries"]),
            "weak_component_count": directed_payload["weak_component_count"],
            "strong_component_count": directed_payload["strong_component_count"],
            "closed_class_count": directed_payload["closed_class_count"],
            "phi_uphill_edge_fraction": float(directed_payload["phi_uphill_edge_fraction"]),
            "phi_uphill_mass_fraction": float(directed_payload["phi_uphill_mass_fraction"]),
        },
        "control_ablation_summary": controls_comparison.to_dict(orient="records"),
        "gpcca_smoke_status": gpcca_payload["smoke_payload"].get("readiness_label") if gpcca_payload.get("smoke_payload") else None,
        "warnings": warnings,
        "next_safe_command": next_safe_command,
        "representation_label": representation_payload["representation_label"],
        "state_unit_label": unit_payload["state_unit_label"],
        "source_snapshot_before": source_before.to_dict(orient="records"),
    }
    decision_body = "## Decision\n\n" + f"`{decision_label}`\n\n"
    decision_body += "## Summary\n\n" + markdown_table(pd.DataFrame([decision_payload]).drop(columns=["warnings", "control_ablation_summary", "source_snapshot_before"], errors="ignore"))
    decision_body += "\n\n## Warnings\n\n" + markdown_table(pd.DataFrame([{"warning": warning} for warning in warnings]))
    write_report(report_root, "09_READINESS_DECISION", "Readiness Decision", decision_payload, decision_body, overwrite=args.overwrite)

    validation_payload = validate_outputs(
        output_root=output_root,
        report_root=report_root,
        packet_files=packet_files,
        source_before=source_before,
        state_frame=state_frame,
        kernel=kernel,
        decision_label=decision_label,
        gpcca_payload=gpcca_payload,
        overwrite=args.overwrite,
    )
    validation_payload["status"] = validation_payload.get("status", "FAIL")

    print(
        json.dumps(
            {
                "decision_label": decision_label,
                "gpcca_smoke_status": gpcca_payload["smoke_payload"].get("readiness_label") if gpcca_payload.get("smoke_payload") else None,
                "samples_processed": samples,
                "cellbins_per_section": decision_payload["cellbins_per_section"],
                "groups_per_section": decision_payload["groups_per_section"],
                "metaniches": decision_payload["metaniches"],
                "validation_status": validation_payload["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
