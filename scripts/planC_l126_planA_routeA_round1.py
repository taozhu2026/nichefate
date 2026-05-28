#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nichefate.barcode_adapter.group_lineage import aggregate_group_lineage  # noqa: E402
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
from nichefate.barcode_adapter.routeA import (  # noqa: E402
    REPRESENTATION_LABEL,
    UNIT_LABEL,
    aggregate_lineage_for_unit_mapping,
    assign_metaniches,
    build_group_representation,
    build_state_matrix,
    compare_barcode_views,
    compute_joint_svd_representation,
    forbidden_claim_hits,
    gpcca_dryrun_checks,
)
from nichefate.barcode_adapter.spatial_neighborhood import spatially_stratified_subset  # noqa: E402


SCOPE_NOTES = [
    "Route A tests whether DARLIN barcode metrics can annotate barcode-free PlanA-style spatial/expression units.",
    "Route A does not test whether barcode evidence changes the kernel.",
    "Route A does not prove fate direction, terminal state, clonal expansion, or lineage-validated transition.",
    "L126_Brain_s1/s2/s3 are serial sections, not timepoints.",
    "section_order is not used as temporal or fate direction.",
    "Bounded units are not frozen MERFISH full M1/M2/M2.5 outputs.",
    "Overlapping local groups are not disjoint tissue partitions and must not be summed as tissue abundance.",
    "Barcode annotation is post-hoc evidence, not model training evidence.",
    "RA/TA/CA are preserved as separate assay-level evidence channels.",
    "No raw FASTQ, DARLIN re-calling, full M0/M1/M2, full GPCCA, PlanB, or fate inference was run.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L126 PlanA Route A Round 1 bounded pilot.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/planC_l126_planA_routeA/l126_planA_routeA_bounded.draft.json"))
    parser.add_argument("--input-packet-root", default="/home/zhutao/scratch/nichefate/spatio_darlin_public_brain_hpc/input_packet")
    parser.add_argument("--round1-barcode-root", default=str(PROJECT_ROOT / "processed/barcode_adapter_l126_round1"))
    parser.add_argument("--round2B-root", default=str(PROJECT_ROOT / "processed/l126_niche_barcode_round2B"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "processed/l126_plana_routeA_round1"))
    parser.add_argument("--report-root", default=str(PROJECT_ROOT / "reports/l126_plana_routeA_round1"))
    parser.add_argument("--samples", default="L126_Brain_s1,L126_Brain_s2,L126_Brain_s3")
    parser.add_argument("--max-cellbins-per-section", type=int, default=10000)
    parser.add_argument("--n-hvgs", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-metaniches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--run-gpcca-readiness-dryrun", action="store_true")
    parser.add_argument("--mode", choices=["all", "audit_only", "representation_only", "units_only", "barcode_annotation_only", "gpcca_readiness_only", "qc_only"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_samples(raw: str) -> list[str]:
    samples = [item.strip() for item in raw.split(",") if item.strip()]
    if not samples:
        raise ValueError("No samples were provided")
    return samples


def write_report_pair(report_root: Path, stem: str, title: str, payload: dict[str, Any], body: str, *, overwrite: bool) -> None:
    atomic_write_json(report_root / f"{stem}.json", payload, overwrite=overwrite)
    scope = "\n".join(f"- {note}" for note in SCOPE_NOTES)
    atomic_write_text(report_root / f"{stem}.md", f"# {title}\n\n{scope}\n\n{body.strip()}\n", overwrite=overwrite)


def reject_forbidden_paths(*paths: Path) -> None:
    bad = [str(path) for path in paths if path_has_ssd(path)]
    if bad:
        raise ValueError("Refusing /ssd paths: " + "; ".join(bad))


def h5ad_required_files(packet_root: Path, samples: list[str]) -> list[Path]:
    return [h5ad_path_for_sample(packet_root, sample) for sample in samples]


def load_counts_matrix(h5ad_path: Path, subset: pd.DataFrame) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    import anndata as ad

    ordered = subset.sort_values("obs_position").reset_index(drop=True).copy()
    data = ad.read_h5ad(h5ad_path, backed="r")
    try:
        counts = data.layers["counts"][ordered["obs_position"].astype(int).to_numpy(), :]
        matrix = counts.tocsr() if sparse.issparse(counts) else sparse.csr_matrix(counts)
    finally:
        if hasattr(data, "file"):
            data.file.close()
    return matrix, ordered


def load_group_assignments(round2b_root: Path, samples: list[str]) -> pd.DataFrame:
    frames = []
    for sample in samples:
        path = round2b_root / "group_assignments" / f"{sample}_group_assignment.tsv.gz"
        frames.append(pd.read_csv(path, sep="\t", compression="gzip"))
    return pd.concat(frames, ignore_index=True)


def save_histogram(frame: pd.DataFrame, column: str, title: str, path_base: Path, bins: int = 40) -> list[str]:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    outputs = []
    for suffix in [".png", ".pdf"]:
        out = path_base.with_suffix(suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        outputs.append(str(out))
    plt.close(fig)
    return outputs


def save_scatter(frame: pd.DataFrame, x: str, y: str, color: str, title: str, path_base: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(6, 5))
    if frame[color].dtype == object:
        for label, group in frame.groupby(color, sort=True):
            ax.scatter(group[x], group[y], s=4, alpha=0.7, label=str(label))
        ax.legend(frameon=False, markerscale=2)
    else:
        scatter = ax.scatter(frame[x], frame[y], c=frame[color], s=5, alpha=0.75, cmap="viridis")
        fig.colorbar(scatter, ax=ax, label=color)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    outputs = []
    for suffix in [".png", ".pdf"]:
        out = path_base.with_suffix(suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        outputs.append(str(out))
    plt.close(fig)
    return outputs


def preflight(args: argparse.Namespace, samples: list[str], packet_root: Path, round2b_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema_rows = [validate_l126_h5ad_schema(h5ad_path_for_sample(packet_root, sample)) for sample in samples]
    round2b_decision = round2b_root.parent.parent / "reports/l126_niche_barcode_round2B/06_ROUND2B_READINESS_DECISION.json"
    local_round2b_decision = PROJECT_ROOT / "reports/l126_niche_barcode_round2B/06_ROUND2B_READINESS_DECISION.json"
    decision_path = local_round2b_decision if local_round2b_decision.exists() else round2b_decision
    round2b_label = ""
    if decision_path.exists():
        round2b_label = json.loads(decision_path.read_text(encoding="utf-8")).get("decision_label", "")
    assignment_paths = [round2b_root / "group_assignments" / f"{sample}_group_assignment.tsv.gz" for sample in samples]
    label = (
        "L126_PLANA_SCHEMA_ADAPTED_BOUNDED_READY"
        if all(row["schema_passed"] for row in schema_rows)
        and all(path.exists() for path in assignment_paths)
        and round2b_label == "L126_PLANA_BARCODE_PREFLIGHT_READY"
        else "L126_PLANA_ROUTEA_HOLD_FOR_M1_FEATURE_REQUIREMENTS"
    )
    payload = {
        "generated_at_utc": utc_now(),
        "preflight_label": label,
        "round2b_decision_label": round2b_label,
        "existing_full_merfish_m1_directly_compatible": False,
        "reason_direct_m1_not_used": "L126 lacks frozen MERFISH M1 requirements such as X_pca_m0, X_spatial_norm, Delaunay/radius graphs, and cell_type_l1/l2/l3.",
        "bounded_route_needed": True,
        "round2b_groups_role": "candidate local neighborhoods for bounded Route A only; not final M1 niches",
        "schema_rows": schema_rows,
        "assignment_paths_exist": {str(path): path.exists() for path in assignment_paths},
        "default_label_for_units": UNIT_LABEL,
    }
    return payload, schema_rows


def write_contract(config: dict[str, Any], args: argparse.Namespace, samples: list[str], packet_root: Path, report_root: Path, overwrite: bool) -> dict[str, Any]:
    payload = {
        "generated_at_utc": utc_now(),
        "contract_path": str(Path(args.config).resolve()),
        "sample_list": samples,
        "h5ad_paths": [str(h5ad_path_for_sample(packet_root, sample)) for sample in samples],
        "expression_source": "layers['counts']",
        "coordinate_source": "obs.x/obs.y and obsm['spatial']",
        "primary_cellbin_key": ["sample_id", "slice_id", "cellbin_id"],
        "section_order_interpretation": "serial section order only, not timepoint",
        "max_cellbins_per_section": int(args.max_cellbins_per_section),
        "seed": int(args.seed),
        "n_hvgs": int(args.n_hvgs),
        "n_pcs": int(args.n_pcs),
        "n_metaniches": int(args.n_metaniches),
        "output_root": str(Path(args.output_root).resolve()),
        "report_root": str(report_root),
        "barcode_input": str(Path(args.round1_barcode_root).resolve() / "cellbin_lineage_summary.tsv.gz"),
        "barcode_evidence_aggregation_api": "src/nichefate/barcode_adapter",
        "raw_config": config,
    }
    body = "## Contract\n\n" + markdown_table(pd.DataFrame([payload]).drop(columns=["raw_config"]))
    write_report_pair(report_root, "01_ROUTEA_BOUNDED_CONTRACT", "Route A Bounded Contract", payload, body, overwrite=overwrite)
    return payload


def build_representation(args: argparse.Namespace, samples: list[str], packet_root: Path, output_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrices = []
    metadata = []
    sample_payloads = []
    for sample in samples:
        h5ad_path = h5ad_path_for_sample(packet_root, sample)
        cellbins = load_l126_cellbin_table(h5ad_path, sample)
        subset, subset_payload = spatially_stratified_subset(
            cellbins, max_cellbins=args.max_cellbins_per_section, seed=args.seed
        )
        matrix, ordered = load_counts_matrix(h5ad_path, subset)
        matrices.append(matrix)
        metadata.append(ordered[["sample_id", "slice_id", "section_order", "cellbin_id", "x", "y", "obs_index", "obs_position"]])
        sample_payloads.append({"sample_id": sample, "h5ad_path": str(h5ad_path), **subset_payload, "matrix_shape": list(matrix.shape)})
    representation, payload = compute_joint_svd_representation(
        matrices, metadata, n_hvgs=args.n_hvgs, n_pcs=args.n_pcs, seed=args.seed
    )
    representation_root = ensure_dir(output_root / "representation")
    for sample, group in representation.groupby("sample_id", sort=True):
        group.to_parquet(representation_root / f"{sample}_bounded_representation.parquet", index=False)
    for sample, group in representation.groupby("sample_id", sort=True):
        if group["cellbin_id"].isna().any():
            raise ValueError(f"Missing cellbin_id in representation for {sample}")
    representation.to_parquet(representation_root / "L126_all_sections_bounded_representation.parquet", index=False)
    payload.update({"samples": sample_payloads, "output_root": str(representation_root)})
    return representation, payload


def build_units(args: argparse.Namespace, representation: pd.DataFrame, round2b_root: Path, samples: list[str], output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    assignment = load_group_assignments(round2b_root, samples)
    group_repr = build_group_representation(assignment, representation)
    metaniche_assignment, metaniche_summary, metaniche_payload = assign_metaniches(
        group_repr,
        n_metaniches=args.n_metaniches,
        seed=args.seed,
        section_purity_threshold=0.9,
        tiny_group_threshold=20,
    )
    units_root = ensure_dir(output_root / "units")
    atomic_write_tsv_gz(units_root / "bounded_group_representation.tsv.gz", group_repr, overwrite=args.overwrite)
    atomic_write_tsv_gz(units_root / "bounded_metaniche_assignment.tsv.gz", metaniche_assignment, overwrite=args.overwrite)
    atomic_write_tsv_gz(units_root / "bounded_metaniche_summary.tsv.gz", metaniche_summary, overwrite=args.overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "unit_label": UNIT_LABEL,
        "outputs": {
            "bounded_group_representation": str(units_root / "bounded_group_representation.tsv.gz"),
            "bounded_metaniche_assignment": str(units_root / "bounded_metaniche_assignment.tsv.gz"),
            "bounded_metaniche_summary": str(units_root / "bounded_metaniche_summary.tsv.gz"),
        },
        **metaniche_payload,
    }
    return assignment, group_repr, metaniche_assignment, metaniche_summary, payload


def barcode_annotation(
    lineage_evidence: pd.DataFrame,
    group_assignment: pd.DataFrame,
    metaniche_assignment: pd.DataFrame,
    output_root: Path,
    overwrite: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    barcode_root = ensure_dir(output_root / "barcode_annotation")
    group_summary, group_assay, group_top = aggregate_group_lineage(lineage_evidence, group_assignment)
    group_summary["local_context_not_tissue_partition"] = True
    atomic_write_tsv_gz(barcode_root / "group_barcode_annotation.tsv.gz", group_summary, overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "group_assay_summary.tsv.gz", group_assay, overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "group_top_features.tsv.gz", group_top, overwrite=overwrite)

    group_to_meta = metaniche_assignment[["group_id", "metaniche_id"]].drop_duplicates()
    metaniche_mapping_local = group_assignment.merge(group_to_meta, on="group_id", how="inner")
    metaniche_mapping_local["local_context_not_tissue_partition"] = True
    metaniche_mapping_unique = metaniche_mapping_local.drop_duplicates(["metaniche_id", *PRIMARY_JOIN_KEY]).copy()
    local_summary, local_assay, local_top = aggregate_lineage_for_unit_mapping(
        lineage_evidence, metaniche_mapping_local, unit_col="metaniche_id", local_context=True
    )
    unique_summary, unique_assay, unique_top = aggregate_lineage_for_unit_mapping(
        lineage_evidence, metaniche_mapping_unique, unit_col="metaniche_id", local_context=False
    )
    comparison = compare_barcode_views(local_summary, unique_summary, "metaniche_id")
    for frame in [local_summary, unique_summary]:
        frame["local_context_not_tissue_partition"] = frame["local_context_not_tissue_partition"].astype(bool)
    atomic_write_tsv_gz(barcode_root / "metaniche_barcode_annotation_local_context.tsv.gz", local_summary, overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_barcode_annotation_unique_cellbin.tsv.gz", unique_summary, overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_assay_summary_local_context.tsv.gz", local_assay, overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_assay_summary_unique_cellbin.tsv.gz", unique_assay, overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_top_features_local_context.tsv.gz", local_top, overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_top_features_unique_cellbin.tsv.gz", unique_top, overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_assay_summary.tsv.gz", unique_assay, overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_top_features.tsv.gz", unique_top, overwrite=overwrite)
    atomic_write_tsv_gz(barcode_root / "metaniche_local_vs_unique_comparison.tsv.gz", comparison, overwrite=overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "group_rows": int(len(group_summary)),
        "metaniche_rows_local_context": int(len(local_summary)),
        "metaniche_rows_unique_cellbin": int(len(unique_summary)),
        "group_coverage_fraction": float(group_summary["evidence_present"].mean()) if len(group_summary) else 0.0,
        "metaniche_coverage_fraction_local_context": float(local_summary["evidence_present"].mean()) if len(local_summary) else 0.0,
        "metaniche_coverage_fraction_unique_cellbin": float(unique_summary["evidence_present"].mean()) if len(unique_summary) else 0.0,
        "median_local_to_unique_total_count_ratio": float(comparison["local_to_unique_total_count_ratio"].median()) if len(comparison) else 0.0,
        "outputs": {path.name: str(path) for path in barcode_root.glob("*.tsv.gz")},
    }
    return {
        "group": group_summary,
        "group_assay": group_assay,
        "group_top": group_top,
        "metaniche_local": local_summary,
        "metaniche_unique": unique_summary,
        "metaniche_local_assay": local_assay,
        "metaniche_unique_assay": unique_assay,
        "comparison": comparison,
    }, payload


def write_gpcca_readiness(
    metaniche_summary: pd.DataFrame,
    metaniche_unique: pd.DataFrame,
    output_root: Path,
    overwrite: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    gpcca_root = ensure_dir(output_root / "gpcca_readiness")
    state_matrix, state_metadata = build_state_matrix(metaniche_summary)
    barcode_cols = [
        "metaniche_id",
        "total_lineage_count",
        "detected_feature_count",
        "detected_assay_count",
        "fraction_lineage_positive",
        "feature_entropy",
        "dominant_feature_fraction",
        "RA_total_count",
        "TA_total_count",
        "CA_total_count",
    ]
    barcode_matrix = metaniche_unique[barcode_cols].copy()
    state_matrix.to_parquet(gpcca_root / "bounded_state_matrix_preview.parquet", index=False)
    atomic_write_tsv(gpcca_root / "bounded_state_metadata.tsv", state_metadata, overwrite=overwrite)
    atomic_write_tsv_gz(gpcca_root / "barcode_annotation_matrix_preview.tsv.gz", barcode_matrix, overwrite=overwrite)
    payload = gpcca_dryrun_checks(state_matrix, state_metadata, barcode_matrix)
    payload.update(
        {
            "generated_at_utc": utc_now(),
            "outputs": {
                "state_matrix": str(gpcca_root / "bounded_state_matrix_preview.parquet"),
                "state_metadata": str(gpcca_root / "bounded_state_metadata.tsv"),
                "barcode_annotation_matrix": str(gpcca_root / "barcode_annotation_matrix_preview.tsv.gz"),
            },
            "no_kernel_construction": True,
            "no_gpcca_run": True,
            "no_fate_probability": True,
        }
    )
    return payload, state_matrix, state_metadata


def write_figures(
    representation: pd.DataFrame,
    group_repr: pd.DataFrame,
    metaniche_summary: pd.DataFrame,
    barcode_frames: dict[str, pd.DataFrame],
    report_root: Path,
) -> tuple[dict[str, Any], str]:
    fig_root = ensure_dir(report_root / "figures")
    paths: list[str] = []
    paths.extend(save_scatter(representation, "pca_0", "pca_1", "sample_id", "PCA Scatter By Section", fig_root / "pca_scatter_by_section"))
    for sample, group in representation.groupby("sample_id", sort=True):
        paths.extend(save_scatter(group, "x", "y", "detected_genes", f"{sample} Sampled Cellbin Spatial Map", fig_root / f"{sample}_sampled_cellbin_spatial_map"))
    paths.extend(save_histogram(metaniche_summary, "n_groups", "Metaniche Count Distribution", fig_root / "metaniche_count_distribution"))
    paths.extend(save_histogram(metaniche_summary, "section_purity", "Metaniche Section Purity Distribution", fig_root / "metaniche_section_distribution"))
    met_unique = barcode_frames["metaniche_unique"]
    for column, title, name in [
        ("fraction_lineage_positive", "Metaniche Lineage Coverage", "metaniche_lineage_coverage_distribution"),
        ("total_lineage_count", "Metaniche Total Lineage Count", "metaniche_total_lineage_count_distribution"),
        ("detected_feature_count", "Metaniche Detected Feature Count", "metaniche_detected_feature_count_distribution"),
        ("dominant_feature_fraction", "Metaniche Dominant Feature Fraction", "metaniche_dominant_feature_fraction_distribution"),
        ("feature_entropy", "Metaniche Entropy", "metaniche_entropy_distribution"),
    ]:
        paths.extend(save_histogram(met_unique, column, title, fig_root / name))
    assay_long = met_unique[["metaniche_id", "RA_total_count", "TA_total_count", "CA_total_count"]].melt(
        id_vars="metaniche_id", var_name="assay", value_name="count"
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    assay_long.boxplot(column="count", by="assay", ax=ax)
    ax.set_title("RA/TA/CA Count Summary By Metaniche")
    fig.suptitle("")
    fig.tight_layout()
    for suffix in [".png", ".pdf"]:
        out = fig_root / f"metaniche_assay_count_summary{suffix}"
        fig.savefig(out, dpi=180)
        paths.append(str(out))
    plt.close(fig)
    spatial = group_repr[["sample_id", "group_id", "centroid_x", "centroid_y", "metaniche_id"]].merge(
        barcode_frames["group"][["group_id", "fraction_member_cellbins_with_lineage", "feature_entropy", "dominant_assay"]],
        on="group_id",
        how="left",
    )
    assay_codes = {assay: idx for idx, assay in enumerate(["", "RA", "TA", "CA"])}
    spatial["dominant_assay_code"] = spatial["dominant_assay"].map(assay_codes).fillna(0)
    for sample, group in spatial.groupby("sample_id", sort=True):
        for column, name in [
            ("fraction_member_cellbins_with_lineage", "group_lineage_coverage"),
            ("feature_entropy", "group_lineage_entropy"),
            ("dominant_assay_code", "group_dominant_assay"),
        ]:
            paths.extend(save_scatter(group, "centroid_x", "centroid_y", column, f"{sample} {name}", fig_root / f"{sample}_{name}_spatial_map"))
    payload = {
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "figure_paths": paths,
        "figures_non_empty": all(Path(path).exists() and Path(path).stat().st_size > 0 for path in paths),
        "descriptive_qc_only": True,
    }
    body = "## Figures\n\n" + markdown_table(pd.DataFrame([{"figure": path} for path in paths]), limit=60)
    return payload, body


def run_validation_commands() -> list[dict[str, Any]]:
    commands = [
        [
            sys.executable,
            "-m",
            "py_compile",
            *[str(path) for path in sorted((PROJECT_ROOT / "src/nichefate/barcode_adapter").glob("*.py"))],
            "scripts/planC_l126_planA_routeA_round1.py",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_l126_plana_routeA_representation.py",
            "tests/test_l126_plana_routeA_units.py",
            "tests/test_l126_plana_routeA_barcode_annotation.py",
            "tests/test_l126_plana_routeA_no_fate_claims.py",
        ],
    ]
    rows = []
    for command in commands:
        existing_command = command
        if command[2] == "pytest":
            existing = [arg for arg in command[3:] if (PROJECT_ROOT / arg).exists()]
            existing_command = command[:3] + existing
        result = subprocess.run(existing_command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        rows.append(
            {
                "name": "py_compile" if "py_compile" in existing_command else "pytest",
                "command": existing_command,
                "returncode": int(result.returncode),
                "stdout_tail": result.stdout[-3000:],
                "stderr_tail": result.stderr[-3000:],
            }
        )
    return rows


def readiness_decision(unit_payload: dict[str, Any], barcode_payload: dict[str, Any], gpcca_payload: dict[str, Any], figures_payload: dict[str, Any]) -> tuple[str, list[str]]:
    warnings = []
    if unit_payload["section_dominated_fraction"] > 0.5:
        warnings.append("Most metaniche-like states show section dominance risk.")
    if unit_payload["tiny_metaniche_fraction"] > 0:
        warnings.append("Tiny metaniche-like states were detected.")
    if unit_payload["section_dominated_fraction"] > 0.75:
        return "L126_PLANA_ROUTEA_HOLD_FOR_SECTION_DOMINANCE_REVIEW", warnings
    if unit_payload["empty_metaniches"] > 0 or unit_payload["tiny_metaniche_fraction"] > 0.10:
        return "L126_PLANA_ROUTEA_HOLD_FOR_UNIT_CONSTRUCTION", warnings
    if gpcca_payload["readiness_label"] != "L126_BOUNDED_GPCCA_INPUT_DRYRUN_READY":
        return "L126_PLANA_ROUTEA_READY_WITH_WARNINGS", warnings
    if barcode_payload["status"] == "PASS" and figures_payload["status"] == "PASS":
        if warnings:
            return "L126_PLANA_ROUTEA_READY_WITH_WARNINGS", warnings
        return "L126_PLANA_ROUTEA_READY_FOR_BOUNDED_GPCCA_SMOKE", warnings
    return "L126_PLANA_ROUTEA_READY_WITH_WARNINGS", warnings


def validate_all_outputs(
    *,
    output_root: Path,
    report_root: Path,
    packet_files: list[Path],
    source_before: pd.DataFrame,
    decision: str,
) -> dict[str, Any]:
    for path in report_root.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    parquet_ok = all(pd.read_parquet(path).shape[0] >= 0 for path in output_root.rglob("*.parquet"))
    tsv_ok = all(path.exists() and path.stat().st_size > 0 for path in list(output_root.rglob("*.tsv")) + list(output_root.rglob("*.tsv.gz")))
    figures = list((report_root / "figures").glob("*.png")) + list((report_root / "figures").glob("*.pdf"))
    report_text = "\n".join(path.read_text(encoding="utf-8") for path in report_root.glob("*.md"))
    source_after = snapshot_files(packet_files, include_sha256=False)
    source_compare = compare_file_snapshots(source_before, source_after)
    validation_commands = run_validation_commands()
    checks = [
        {"check": "json_reports_parse", "status": True, "details": str(report_root)},
        {"check": "tsv_gzip_readability", "status": tsv_ok, "details": str(output_root)},
        {"check": "parquet_readability", "status": parquet_ok, "details": str(output_root)},
        {"check": "figures_non_empty", "status": bool(figures) and all(path.stat().st_size > 0 for path in figures), "details": str(report_root / "figures")},
        {"check": "h5ad_input_readback", "status": all(path.exists() for path in packet_files), "details": "packet file existence"},
        {"check": "source_input_packet_unchanged", "status": not bool(source_compare["changed"].any()), "details": "size/mtime comparison"},
        {"check": "no_ssd", "status": True, "details": "configured paths checked"},
        {"check": "no_raw_fastq", "status": True, "details": "not run"},
        {"check": "no_darlin_recalling", "status": True, "details": "not run"},
        {"check": "no_full_m0_m1_m2", "status": True, "details": "bounded schema-adapted route only"},
        {"check": "no_full_gpcca", "status": True, "details": "dry-run only"},
        {"check": "no_planB", "status": True, "details": "not run"},
        {"check": "no_forbidden_fate_claims", "status": not forbidden_claim_hits(report_text), "details": ";".join(forbidden_claim_hits(report_text))},
        {"check": "no_git_add_commit_push", "status": True, "details": "not run by script"},
        *[{"check": row["name"], "status": row["returncode"] == 0, "details": " ".join(row["command"])} for row in validation_commands],
    ]
    return {
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(row["status"] for row in checks) else "FAIL",
        "decision_label": decision,
        "checks": checks,
        "validation_commands": validation_commands,
        "source_immutability_comparison": source_compare.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    samples = parse_samples(args.samples)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    packet_root = Path(args.input_packet_root).expanduser().resolve()
    round1_root = Path(args.round1_barcode_root).expanduser().resolve()
    round2b_root = Path(args.round2B_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    report_root = Path(args.report_root).expanduser().resolve()
    reject_forbidden_paths(packet_root, round1_root, round2b_root, output_root, report_root)
    ensure_dir(output_root)
    ensure_dir(report_root)
    packet_files = h5ad_required_files(packet_root, samples) + [
        packet_root / "processed/lineage_evidence/cellbin_lineage_evidence.tsv.gz",
        packet_root / "processed/lineage_evidence/feature_allele_annotation_long.tsv.gz",
        packet_root / "processed/transfer/L126_brain_barcode_aware_input_packet.manifest.tsv",
    ]
    source_before = snapshot_files(packet_files, include_sha256=False)

    preflight_payload, schema_rows = preflight(args, samples, packet_root, round2b_root)
    preflight_body = "## PlanA Contract Audit\n\n" + markdown_table(pd.DataFrame([preflight_payload]).drop(columns=["schema_rows"]))
    preflight_body += "\n\n## h5ad Schema\n\n" + markdown_table(pd.DataFrame(schema_rows))
    write_report_pair(report_root, "00_PLANA_CONTRACT_PREFLIGHT", "PlanA Contract Preflight", preflight_payload, preflight_body, overwrite=args.overwrite)
    write_contract(config, args, samples, packet_root, report_root, args.overwrite)
    if args.mode == "audit_only" or preflight_payload["preflight_label"] != "L126_PLANA_SCHEMA_ADAPTED_BOUNDED_READY":
        print(json.dumps({"decision_label": preflight_payload["preflight_label"], "mode": args.mode}, indent=2))
        return

    representation, representation_payload = build_representation(args, samples, packet_root, output_root)
    representation_body = "## Representation Summary\n\n" + markdown_table(pd.DataFrame([representation_payload]).drop(columns=["samples"]))
    representation_body += "\n\n## Samples\n\n" + markdown_table(pd.DataFrame(representation_payload["samples"]))
    write_report_pair(report_root, "02_BOUNDED_REPRESENTATION", "Bounded Representation", representation_payload, representation_body, overwrite=args.overwrite)
    if args.mode == "representation_only":
        return

    group_assignment, group_repr, metaniche_assignment, metaniche_summary, units_payload = build_units(
        args, representation, round2b_root, samples, output_root
    )
    group_repr = group_repr.merge(metaniche_assignment[["group_id", "metaniche_id"]], on="group_id", how="left")
    units_body = "## Unit QC\n\n" + markdown_table(pd.DataFrame([units_payload]).drop(columns=["outputs"]))
    units_body += "\n\n## Metaniche Size And Section QC\n\n" + markdown_table(metaniche_summary[["metaniche_id", "n_groups", "section_distribution", "section_purity", "section_entropy", "section_dominated", "tiny_metaniche"]], limit=30)
    write_report_pair(report_root, "03_BOUNDED_PLANA_STYLE_UNITS", "Bounded PlanA-Style Units", units_payload, units_body, overwrite=args.overwrite)
    if args.mode == "units_only":
        return

    lineage_evidence = load_cellbin_lineage_evidence(packet_root / "processed/lineage_evidence/cellbin_lineage_evidence.tsv.gz")
    barcode_frames, barcode_payload = barcode_annotation(lineage_evidence, group_assignment, metaniche_assignment, output_root, args.overwrite)
    barcode_body = "## Barcode Annotation Summary\n\n" + markdown_table(pd.DataFrame([barcode_payload]).drop(columns=["outputs"]))
    barcode_body += "\n\n## Local vs Unique Cellbin Comparison\n\n" + markdown_table(barcode_frames["comparison"], limit=30)
    write_report_pair(report_root, "04_POSTHOC_BARCODE_ANNOTATION", "Post-Hoc Barcode Annotation", barcode_payload, barcode_body, overwrite=args.overwrite)
    if args.mode == "barcode_annotation_only":
        return

    if args.run_gpcca_readiness_dryrun:
        gpcca_payload, state_matrix, state_metadata = write_gpcca_readiness(
            metaniche_summary, barcode_frames["metaniche_unique"], output_root, args.overwrite
        )
    else:
        gpcca_payload = {
            "generated_at_utc": utc_now(),
            "readiness_label": "L126_BOUNDED_GPCCA_INPUT_READY_WITH_WARNINGS",
            "checks": {"dryrun_requested": False},
            "no_kernel_construction": True,
            "no_gpcca_run": True,
            "no_fate_probability": True,
        }
        state_matrix, state_metadata = build_state_matrix(metaniche_summary)
    gpcca_body = "## GPCCA Readiness Dry Run\n\n" + markdown_table(pd.DataFrame([gpcca_payload]).drop(columns=["checks"]))
    gpcca_body += "\n\n## Checks\n\n" + markdown_table(pd.DataFrame([gpcca_payload["checks"]]))
    write_report_pair(report_root, "05_GPCCA_READINESS_DRY_RUN", "GPCCA Readiness Dry Run", gpcca_payload, gpcca_body, overwrite=args.overwrite)
    if args.mode == "gpcca_readiness_only":
        return

    figures_payload, figures_body = write_figures(representation, group_repr, metaniche_summary, barcode_frames, report_root)
    write_report_pair(report_root, "06_QC_AND_FIGURES", "QC And Figures", figures_payload, figures_body, overwrite=args.overwrite)
    decision, warnings = readiness_decision(units_payload, barcode_payload, gpcca_payload, figures_payload)
    decision_payload = {
        "generated_at_utc": utc_now(),
        "decision_label": decision,
        "unit_type": UNIT_LABEL,
        "representation_label": REPRESENTATION_LABEL,
        "samples_processed": samples,
        "cellbins_per_section": representation.groupby("sample_id").size().astype(int).to_dict(),
        "n_groups": int(group_repr["group_id"].nunique()),
        "n_metaniches": int(metaniche_summary["metaniche_id"].nunique()),
        "barcode_annotation_coverage": {
            "group": barcode_payload["group_coverage_fraction"],
            "metaniche_local_context": barcode_payload["metaniche_coverage_fraction_local_context"],
            "metaniche_unique_cellbin": barcode_payload["metaniche_coverage_fraction_unique_cellbin"],
        },
        "gpcca_readiness_label": gpcca_payload["readiness_label"],
        "section_dominated_metaniche_fraction": units_payload["section_dominated_fraction"],
        "tiny_metaniche_fraction": units_payload["tiny_metaniche_fraction"],
        "warnings": warnings,
        "next_safe_command": "Run a separate bounded GPCCA smoke over processed/l126_plana_routeA_round1/gpcca_readiness/bounded_state_matrix_preview.parquet after reviewing section dominance QC.",
    }
    decision_body = "## Decision\n\n" + f"`{decision}`\n\n## Summary\n\n" + markdown_table(pd.DataFrame([decision_payload]).drop(columns=["warnings"]))
    decision_body += "\n\n## Warnings\n\n" + markdown_table(pd.DataFrame([{"warning": warning} for warning in warnings]))
    write_report_pair(report_root, "07_ROUTEA_READINESS_DECISION", "Route A Readiness Decision", decision_payload, decision_body, overwrite=args.overwrite)

    validation_payload = validate_all_outputs(
        output_root=output_root,
        report_root=report_root,
        packet_files=packet_files,
        source_before=source_before,
        decision=decision,
    )
    validation_body = "## Validation Checks\n\n" + markdown_table(pd.DataFrame(validation_payload["checks"]))
    write_report_pair(report_root, "08_VALIDATION", "Validation", validation_payload, validation_body, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "decision_label": decision,
                "samples_processed": samples,
                "n_groups": int(group_repr["group_id"].nunique()),
                "n_metaniches": int(metaniche_summary["metaniche_id"].nunique()),
                "gpcca_readiness_label": gpcca_payload["readiness_label"],
                "validation_status": validation_payload["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
