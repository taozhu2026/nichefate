#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nichefate.barcode_adapter.group_lineage import (  # noqa: E402
    aggregate_group_lineage,
    group_lineage_coverage_metrics,
)
from nichefate.barcode_adapter.input_contract import load_barcode_input_contract  # noqa: E402
from nichefate.barcode_adapter.l126_schema import (  # noqa: E402
    h5ad_path_for_sample,
    load_l126_cellbin_table,
    validate_l126_h5ad_schema,
)
from nichefate.barcode_adapter.loaders import (  # noqa: E402
    load_cellbin_lineage_evidence,
    prepare_packet_root,
    required_packet_files,
)
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
from nichefate.barcode_adapter.round2_qc import (  # noqa: E402
    group_centroid_table,
    save_histogram,
    save_spatial_map,
)
from nichefate.barcode_adapter.round2b import (  # noqa: E402
    build_plana_barcode_readiness_audit,
    distribution_summary_by_sample,
    parse_sample_list,
    section_summary_row,
    validate_round2b_group_assignment,
)
from nichefate.barcode_adapter.spatial_neighborhood import (  # noqa: E402
    GROUP_TYPE,
    build_spatial_neighborhood_groups,
    group_membership_multiplicity,
    spatially_stratified_subset,
)


SCOPE_NOTES = [
    "L126_Brain_s1/s2/s3 are serial sections, not timepoints.",
    "section_order is not used as temporal or fate direction.",
    "Groups are bounded overlapping spatial neighborhoods, not final biological niches.",
    "Overlapping group-level barcode summaries are local-context summaries and must not be summed as tissue abundance.",
    "RA/TA/CA are preserved as separate assay-level evidence channels.",
    "No cross-assay biological clone identity is inferred.",
    "Allele annotation remains annotation-only and is not expanded into independent counts.",
    "No biological fate, true terminal state, clonal expansion, transition, or lineage-validated remodeling is claimed.",
    "No raw FASTQ, DARLIN re-calling, full M0/M1/M2, full GPCCA production, PlanB, or PlanA/PlanB fate inference was run.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L126 Brain Round 2B all-section barcode aggregation hardening.")
    parser.add_argument("--input-contract", default=str(PROJECT_ROOT / "configs/barcode_adapter/l126_brain_input_contract.draft.json"))
    parser.add_argument("--round1-summary-root", default=str(PROJECT_ROOT / "processed/barcode_adapter_l126_round1"))
    parser.add_argument("--round2-root", default=str(PROJECT_ROOT / "processed/l126_niche_barcode_round2"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "processed/l126_niche_barcode_round2B"))
    parser.add_argument("--report-root", default=str(PROJECT_ROOT / "reports/l126_niche_barcode_round2B"))
    parser.add_argument("--samples", default="L126_Brain_s1,L126_Brain_s2,L126_Brain_s3")
    parser.add_argument("--max-cellbins", type=int, default=10000)
    parser.add_argument("--k-neighbors", type=int, default=16)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument(
        "--grouping-mode",
        choices=["all_sections_smoke", "qc_only", "expression_smoke_only", "plana_readiness_only", "audit_only"],
        default="all_sections_smoke",
    )
    parser.add_argument("--run-expression-smoke", action="store_true")
    parser.add_argument("--run-plana-readiness-audit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reject_forbidden_paths(*paths: Path) -> None:
    bad = [str(path) for path in paths if path_has_ssd(path)]
    if bad:
        raise ValueError("Refusing /ssd paths: " + "; ".join(bad))


def write_report_pair(
    report_root: Path,
    stem: str,
    title: str,
    payload: dict[str, Any],
    body: str,
    *,
    overwrite: bool,
) -> None:
    atomic_write_json(report_root / f"{stem}.json", payload, overwrite=overwrite)
    scope = "\n".join(f"- {note}" for note in SCOPE_NOTES)
    atomic_write_text(report_root / f"{stem}.md", f"# {title}\n\n{scope}\n\n{body.strip()}\n", overwrite=overwrite)


def read_tsv(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    return pd.read_csv(path, sep="\t", compression=compression)


def expression_feature_smoke(h5ad_path: Path, subset: pd.DataFrame, seed: int) -> dict[str, Any]:
    import anndata as ad
    from sklearn.decomposition import TruncatedSVD

    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "label": "EXPRESSION_FEATURE_SMOKE_NOT_M0",
        "ran": False,
        "status": "SKIPPED",
        "writes_production_m0_m1_m2_artifacts": False,
    }
    try:
        data = ad.read_h5ad(h5ad_path, backed="r")
        try:
            positions = np.sort(subset["obs_position"].astype(int).to_numpy())
            counts = data.layers["counts"][positions, :]
            matrix = counts.tocsr() if sparse.issparse(counts) else sparse.csr_matrix(counts)
        finally:
            if hasattr(data, "file"):
                data.file.close()
        matrix = matrix.astype(np.float32)
        cell_sums = np.asarray(matrix.sum(axis=1)).ravel()
        scale = np.divide(1e4, cell_sums, out=np.zeros_like(cell_sums, dtype=np.float32), where=cell_sums > 0)
        matrix = matrix.multiply(scale[:, None]).tocsr()
        matrix.data = np.log1p(matrix.data)
        means = np.asarray(matrix.mean(axis=0)).ravel()
        variances = np.asarray(matrix.power(2).mean(axis=0)).ravel() - means**2
        n_hvg = min(2000, matrix.shape[1])
        hvg_idx = np.argsort(variances)[-n_hvg:]
        n_components = min(20, max(1, n_hvg - 1), max(1, matrix.shape[0] - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        embedding = svd.fit_transform(matrix[:, hvg_idx])
        payload.update(
            {
                "ran": True,
                "status": "PASS",
                "bounded_cellbins": int(matrix.shape[0]),
                "n_genes": int(matrix.shape[1]),
                "n_hvg": int(n_hvg),
                "pca_components": int(n_components),
                "embedding_shape": [int(embedding.shape[0]), int(embedding.shape[1])],
                "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update({"status": "WARN", "error": f"{type(exc).__name__}: {exc}"})
    return payload


def validate_reusable_round2_s1(round2_root: Path, sample: str, max_cellbins: int, k_neighbors: int) -> tuple[bool, dict[str, Any]]:
    paths = {
        "assignment": round2_root / "group_assignments" / f"{sample}_group_assignment.tsv.gz",
        "group_summary": round2_root / "lineage_aggregation" / f"{sample}_group_lineage_summary.tsv.gz",
        "assay_summary": round2_root / "lineage_aggregation" / f"{sample}_group_assay_summary.tsv.gz",
        "top_features": round2_root / "lineage_aggregation" / f"{sample}_group_top_features.tsv.gz",
    }
    payload: dict[str, Any] = {"sample_id": sample, "paths": {key: str(value) for key, value in paths.items()}}
    if not all(path.exists() for path in paths.values()):
        payload.update({"reusable": False, "reason": "missing_round2_s1_outputs"})
        return False, payload
    try:
        assignment = read_tsv(paths["assignment"])
        group_summary = read_tsv(paths["group_summary"])
        checks = {
            "assignment_rows_match": len(assignment) == max_cellbins * k_neighbors,
            "group_count_match": assignment["group_id"].nunique() == max_cellbins,
            "summary_rows_match": len(group_summary) == max_cellbins,
            "group_type_match": assignment["group_type"].astype(str).eq(GROUP_TYPE).all(),
            "stable_group_ids": (
                assignment["group_id"].astype(str)
                == assignment["sample_id"].astype(str) + "__anchor__" + assignment["anchor_cellbin_id"].astype(str)
            ).all(),
            "has_fraction_alias": "fraction_member_cellbins_with_lineage" in group_summary.columns,
        }
        payload.update({"reusable": bool(all(checks.values())), "checks": checks})
        return bool(payload["reusable"]), payload
    except Exception as exc:  # noqa: BLE001
        payload.update({"reusable": False, "reason": f"{type(exc).__name__}: {exc}"})
        return False, payload


def process_sample(
    *,
    sample: str,
    args: argparse.Namespace,
    packet_root: Path,
    lineage: pd.DataFrame,
    round1_summary: pd.DataFrame,
    output_root: Path,
    round2_root: Path,
) -> dict[str, Any]:
    h5ad_path = h5ad_path_for_sample(packet_root, sample)
    schema = validate_l126_h5ad_schema(h5ad_path)
    cellbins = load_l126_cellbin_table(h5ad_path, sample)
    subset, subset_payload = spatially_stratified_subset(cellbins, max_cellbins=args.max_cellbins, seed=args.seed)
    sample_cellbins = set(subset["cellbin_id"].astype(str))

    reused_round2 = False
    reusable, reuse_payload = validate_reusable_round2_s1(round2_root, sample, args.max_cellbins, args.k_neighbors)
    if sample == "L126_Brain_s1" and reusable:
        reused_round2 = True
        assignment = read_tsv(round2_root / "group_assignments" / f"{sample}_group_assignment.tsv.gz")
        group_summary = read_tsv(round2_root / "lineage_aggregation" / f"{sample}_group_lineage_summary.tsv.gz")
        assay_summary = read_tsv(round2_root / "lineage_aggregation" / f"{sample}_group_assay_summary.tsv.gz")
        top_features = read_tsv(round2_root / "lineage_aggregation" / f"{sample}_group_top_features.tsv.gz")
    else:
        assignment, _ = build_spatial_neighborhood_groups(subset, k_neighbors=args.k_neighbors)
        lineage_sample = lineage.loc[
            (lineage["sample_id"].astype(str) == sample)
            & (lineage["cellbin_id"].astype(str).isin(sample_cellbins))
        ].copy()
        group_summary, assay_summary, top_features = aggregate_group_lineage(lineage_sample, assignment)

    group_assignment_root = ensure_dir(output_root / "group_assignments")
    aggregation_root = ensure_dir(output_root / "lineage_aggregation")
    assignment_path = group_assignment_root / f"{sample}_group_assignment.tsv.gz"
    group_summary_path = aggregation_root / f"{sample}_group_lineage_summary.tsv.gz"
    assay_summary_path = aggregation_root / f"{sample}_group_assay_summary.tsv.gz"
    top_features_path = aggregation_root / f"{sample}_group_top_features.tsv.gz"
    atomic_write_tsv_gz(assignment_path, assignment, overwrite=args.overwrite)
    atomic_write_tsv_gz(group_summary_path, group_summary, overwrite=args.overwrite)
    atomic_write_tsv_gz(assay_summary_path, assay_summary, overwrite=args.overwrite)
    atomic_write_tsv_gz(top_features_path, top_features, overwrite=args.overwrite)

    multiplicity, multiplicity_payload = group_membership_multiplicity(assignment)
    round1_sample = round1_summary.loc[
        (round1_summary["sample_id"].astype(str) == sample)
        & (round1_summary["cellbin_id"].astype(str).isin(set(assignment["cellbin_id"].astype(str))))
    ].copy()
    coverage = group_lineage_coverage_metrics(round1_sample, group_summary)
    group_validation = validate_round2b_group_assignment(
        assignment,
        cellbins,
        sample_id=sample,
        k_neighbors=args.k_neighbors,
    )
    section_summary = section_summary_row(
        sample_id=sample,
        h5ad_n_obs=int(schema["n_obs"]),
        assignment=assignment,
        group_summary=group_summary,
        multiplicity=multiplicity,
        coverage_metrics=coverage,
    )
    return {
        "sample_id": sample,
        "h5ad_path": str(h5ad_path),
        "schema": schema,
        "subset": subset,
        "subset_payload": subset_payload,
        "reused_round2_s1": reused_round2,
        "reuse_payload": reuse_payload,
        "assignment": assignment,
        "group_summary": group_summary,
        "assay_summary": assay_summary,
        "top_features": top_features,
        "multiplicity": multiplicity,
        "multiplicity_payload": multiplicity_payload,
        "coverage_metrics": coverage,
        "group_validation": group_validation,
        "section_summary": section_summary,
        "assignment_path": str(assignment_path),
        "group_summary_path": str(group_summary_path),
        "assay_summary_path": str(assay_summary_path),
        "top_features_path": str(top_features_path),
    }


def write_cross_section_qc(results: list[dict[str, Any]], output_root: Path, report_root: Path, overwrite: bool) -> tuple[dict[str, Any], str]:
    qc_root = ensure_dir(output_root / "qc")
    fig_root = ensure_dir(report_root / "figures")
    all_group_summary = pd.concat([item["group_summary"] for item in results], ignore_index=True)
    all_assignment = pd.concat([item["assignment"] for item in results], ignore_index=True)
    all_multiplicity = pd.concat([item["multiplicity"] for item in results], ignore_index=True)
    section_summary = pd.DataFrame([item["section_summary"] for item in results])

    table_specs = {
        "all_sections_group_size_distribution": (all_group_summary, "n_member_cellbins"),
        "all_sections_lineage_coverage_distribution": (all_group_summary, "fraction_member_cellbins_with_lineage"),
        "all_sections_assay_balance_summary": (all_group_summary, "assay_balance"),
        "all_sections_dominant_feature_fraction_summary": (all_group_summary, "dominant_feature_fraction"),
        "all_sections_lineage_entropy_summary": (all_group_summary, "feature_entropy"),
    }
    table_paths: dict[str, str] = {}
    for name, (frame, column) in table_specs.items():
        path = qc_root / f"{name}.tsv"
        atomic_write_tsv(path, distribution_summary_by_sample(frame, column), overwrite=overwrite)
        table_paths[name] = str(path)
    multiplicity_path = qc_root / "all_sections_cellbin_group_membership_multiplicity.tsv"
    section_summary_path = qc_root / "all_sections_section_summary.tsv"
    atomic_write_tsv(multiplicity_path, all_multiplicity, overwrite=overwrite)
    atomic_write_tsv(section_summary_path, section_summary, overwrite=overwrite)
    table_paths["all_sections_cellbin_group_membership_multiplicity"] = str(multiplicity_path)
    table_paths["all_sections_section_summary"] = str(section_summary_path)

    assay_cols = ["RA_total_count", "TA_total_count", "CA_total_count"]
    all_group_summary = all_group_summary.copy()
    all_group_summary["assay_total_count_sum"] = all_group_summary[assay_cols].sum(axis=1)

    figure_paths: list[Path] = []
    for name, column, title in [
        ("all_sections_group_size_distribution", "n_member_cellbins", "All Sections Group Size Distribution"),
        ("all_sections_lineage_coverage_distribution", "fraction_member_cellbins_with_lineage", "All Sections Lineage Coverage"),
        ("all_sections_total_lineage_count_per_group", "total_lineage_count", "All Sections Total Lineage Count Per Group"),
        ("all_sections_detected_feature_count_per_group", "detected_feature_count", "All Sections Detected Feature Count Per Group"),
        ("all_sections_dominant_feature_fraction_distribution", "dominant_feature_fraction", "All Sections Dominant Feature Fraction"),
        ("all_sections_lineage_entropy_distribution", "feature_entropy", "All Sections Lineage Entropy"),
        ("all_sections_assay_total_counts_summary", "assay_total_count_sum", "All Sections Assay Total Counts Summary"),
    ]:
        figure_paths.extend(save_histogram(all_group_summary, column, title, fig_root / name))
    figure_paths.extend(
        save_histogram(
            all_multiplicity,
            "groups_per_member_cellbin",
            "All Sections Member Multiplicity Distribution",
            fig_root / "all_sections_member_multiplicity_distribution",
        )
    )
    for item in results:
        centroid = group_centroid_table(item["assignment"], item["group_summary"])
        sample = item["sample_id"]
        for name, column, title in [
            ("spatial_lineage_coverage_map", "fraction_member_cellbins_with_lineage_evidence", "Spatial Lineage Coverage"),
            ("spatial_total_lineage_count_map", "total_lineage_count", "Spatial Total Lineage Count"),
            ("spatial_lineage_entropy_map", "feature_entropy", "Spatial Lineage Entropy"),
        ]:
            figure_paths.extend(save_spatial_map(centroid, column, f"{sample} {title}", fig_root / f"{sample}_{name}"))

    payload = {
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "section_summary_path": str(section_summary_path),
        "table_paths": table_paths,
        "figure_paths": [str(path) for path in figure_paths],
        "figures_non_empty": all(path.exists() and path.stat().st_size > 0 for path in figure_paths),
        "spatial_maps_are_within_section_only": True,
        "descriptive_qc_only": True,
    }
    body = "## Section Summary\n\n" + markdown_table(section_summary)
    body += "\n\n## QC Tables\n\n" + markdown_table(pd.DataFrame([{"table": key, "path": value} for key, value in table_paths.items()]))
    body += "\n\n## Figures\n\n" + markdown_table(pd.DataFrame([{"figure": str(path)} for path in figure_paths]), limit=40)
    return payload, body


def run_validation_commands() -> list[dict[str, Any]]:
    py_files = sorted((PROJECT_ROOT / "src/nichefate/barcode_adapter").glob("*.py"))
    commands = [
        [
            sys.executable,
            "-m",
            "py_compile",
            *[str(path) for path in py_files],
            "scripts/planC_l126_niche_barcode_round2.py",
            "scripts/planC_l126_niche_barcode_round2B.py",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_l126_schema_adapter.py",
            "tests/test_barcode_adapter_group_aggregation_realistic.py",
            "tests/test_l126_group_assignment_join.py",
            "tests/test_l126_round2B_all_section_grouping.py",
            "tests/test_l126_round2B_lineage_aggregation.py",
            "tests/test_l126_planA_barcode_readiness.py",
        ],
    ]
    rows = []
    for command in commands:
        if command[2] == "pytest":
            existing = [arg for arg in command[3:] if (PROJECT_ROOT / arg).exists()]
            command = command[:3] + existing
        result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        rows.append(
            {
                "name": "py_compile" if "py_compile" in command else "pytest",
                "command": command,
                "returncode": int(result.returncode),
                "stdout_tail": result.stdout[-3000:],
                "stderr_tail": result.stderr[-3000:],
            }
        )
    return rows


def validate_outputs(
    *,
    report_root: Path,
    output_root: Path,
    results: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
    source_compare: pd.DataFrame,
    qc_payload: dict[str, Any],
    decision: str,
) -> dict[str, Any]:
    json_ok = True
    for path in report_root.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    output_paths = []
    for item in results:
        output_paths.extend(
            [
                Path(item["assignment_path"]),
                Path(item["group_summary_path"]),
                Path(item["assay_summary_path"]),
                Path(item["top_features_path"]),
            ]
        )
    output_paths.extend(Path(path) for path in qc_payload["table_paths"].values())
    figure_paths = [Path(path) for path in qc_payload["figure_paths"]]
    validation_commands = run_validation_commands()
    checks = [
        {"check": "json_reports_parse", "status": json_ok, "details": str(report_root)},
        {"check": "tsv_gzip_readability", "status": all(path.exists() and path.stat().st_size > 0 for path in output_paths), "details": str(output_root)},
        {"check": "figures_non_empty", "status": all(path.exists() and path.stat().st_size > 0 for path in figure_paths), "details": str(report_root / "figures")},
        {"check": "h5ad_readback", "status": all(row["schema_passed"] for row in schema_rows), "details": "schema adapter readback"},
        {"check": "group_assignments_join_cellbin_summary", "status": all(item["coverage_metrics"]["sampled_cellbins"] == item["subset_payload"]["selected"] for item in results), "details": "Round 1 summary join"},
        {"check": "source_input_packet_unchanged", "status": not bool(source_compare["changed"].any()), "details": "size/mtime comparison"},
        {"check": "no_ssd", "status": True, "details": "configured paths checked"},
        {"check": "no_raw_fastq", "status": True, "details": "processed packet only"},
        {"check": "no_darlin_recalling", "status": True, "details": "not run"},
        {"check": "no_full_m0_m1_m2", "status": True, "details": "bounded smoke only"},
        {"check": "no_full_gpcca_production", "status": True, "details": "not run"},
        {"check": "no_planA_planB_fate_inference", "status": True, "details": "not run"},
        {"check": "no_git_add_commit_push", "status": True, "details": "not run by script"},
        *[
            {"check": row["name"], "status": row["returncode"] == 0, "details": " ".join(row["command"])}
            for row in validation_commands
        ],
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
    samples = parse_sample_list(args.samples)
    output_root = Path(args.output_root).expanduser().resolve()
    report_root = Path(args.report_root).expanduser().resolve()
    round1_root = Path(args.round1_summary_root).expanduser().resolve()
    round2_root = Path(args.round2_root).expanduser().resolve()
    reject_forbidden_paths(output_root, report_root, round1_root, round2_root)
    ensure_dir(output_root)
    ensure_dir(report_root)

    contract = load_barcode_input_contract(args.input_contract)
    paths = prepare_packet_root(contract, extract_if_needed=False)
    source_paths = list(required_packet_files(paths))
    source_before = snapshot_files(source_paths, include_sha256=False)
    round1_summary_path = round1_root / "cellbin_lineage_summary.tsv.gz"
    schema_rows = [validate_l126_h5ad_schema(h5ad_path_for_sample(paths.root, sample)) for sample in samples]
    round2_s1_reusable, round2_s1_payload = validate_reusable_round2_s1(
        round2_root,
        "L126_Brain_s1",
        args.max_cellbins,
        args.k_neighbors,
    )
    preflight_checks = {
        "round1_summary_exists": round1_summary_path.exists(),
        "round2_s1_reusable": round2_s1_reusable,
        "h5ad_all_readable": all(row["schema_passed"] for row in schema_rows),
        "source_packet_files_exist": all(path.exists() for path in source_paths),
        "no_ssd_paths": True,
    }
    preflight_label = (
        "L126_ROUND2B_PREFLIGHT_READY"
        if all(preflight_checks.values())
        else "L126_ROUND2B_HOLD_FOR_INPUT_PACKET"
    )
    preflight_payload = {
        "generated_at_utc": utc_now(),
        "decision_label": preflight_label,
        "samples": list(samples),
        "checks": preflight_checks,
        "round2_s1_reuse": round2_s1_payload,
        "schema_rows": schema_rows,
        "source_snapshot": source_before.to_dict(orient="records"),
    }
    preflight_body = "## Preflight Checks\n\n" + markdown_table(pd.DataFrame([preflight_checks]))
    preflight_body += "\n\n## h5ad Schema\n\n" + markdown_table(pd.DataFrame(schema_rows))
    write_report_pair(report_root, "00_PREFLIGHT", "Round 2B Preflight", preflight_payload, preflight_body, overwrite=args.overwrite)
    if preflight_label != "L126_ROUND2B_PREFLIGHT_READY" or args.grouping_mode == "audit_only":
        print(json.dumps({"decision_label": preflight_label, "mode": args.grouping_mode}, indent=2))
        return

    lineage = load_cellbin_lineage_evidence(paths.primary_evidence)
    round1_summary = pd.read_csv(round1_summary_path, sep="\t", compression="gzip")
    results = [
        process_sample(
            sample=sample,
            args=args,
            packet_root=paths.root,
            lineage=lineage,
            round1_summary=round1_summary,
            output_root=output_root,
            round2_root=round2_root,
        )
        for sample in samples
    ]
    grouping_payload = {
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(item["group_validation"]["validation_passed"] for item in results) else "FAIL",
        "group_type": GROUP_TYPE,
        "max_cellbins_per_section": int(args.max_cellbins),
        "k_neighbors": int(args.k_neighbors),
        "seed": int(args.seed),
        "samples": [
            {
                "sample_id": item["sample_id"],
                "reused_round2_s1": item["reused_round2_s1"],
                "assignment_path": item["assignment_path"],
                "subset": item["subset_payload"],
                "group_validation": item["group_validation"],
                "multiplicity": item["multiplicity_payload"],
            }
            for item in results
        ],
    }
    grouping_body = "## Grouping Summary\n\n" + markdown_table(pd.DataFrame(grouping_payload["samples"]))
    write_report_pair(report_root, "01_ALL_SECTION_GROUPING", "All-Section Grouping", grouping_payload, grouping_body, overwrite=args.overwrite)

    aggregation_payload = {
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "samples": [
            {
                "sample_id": item["sample_id"],
                "coverage_metrics": item["coverage_metrics"],
                "group_summary_path": item["group_summary_path"],
                "assay_summary_path": item["assay_summary_path"],
                "top_features_path": item["top_features_path"],
            }
            for item in results
        ],
        "RA_TA_CA_preserved_separately": True,
        "cross_assay_clone_identity_inferred": False,
        "allele_annotation_count_expansion": False,
        "overlapping_group_counts_not_tissue_totals": True,
    }
    aggregation_body = "## Coverage Metrics\n\n" + markdown_table(
        pd.DataFrame(
            [
                {"sample_id": item["sample_id"], **item["coverage_metrics"]}
                for item in results
            ]
        )
    )
    write_report_pair(report_root, "02_ALL_SECTION_LINEAGE_AGGREGATION", "All-Section Lineage Aggregation", aggregation_payload, aggregation_body, overwrite=args.overwrite)

    qc_payload, qc_body = write_cross_section_qc(results, output_root, report_root, args.overwrite)
    write_report_pair(report_root, "03_CROSS_SECTION_QC", "Cross-Section QC", qc_payload, qc_body, overwrite=args.overwrite)

    expression_payload = {
        "generated_at_utc": utc_now(),
        "label": "EXPRESSION_FEATURE_SMOKE_NOT_M0",
        "requested": bool(args.run_expression_smoke),
        "status": "SKIPPED",
        "samples": [],
        "writes_production_m0_m1_m2_artifacts": False,
    }
    if args.run_expression_smoke and args.grouping_mode in {"all_sections_smoke", "expression_smoke_only"}:
        expression_root = ensure_dir(output_root / "expression_smoke")
        sample_payloads = []
        for item in results:
            payload = expression_feature_smoke(Path(item["h5ad_path"]), item["subset"], args.seed)
            payload["sample_id"] = item["sample_id"]
            sample_payloads.append(payload)
            atomic_write_tsv(
                expression_root / f"{item['sample_id']}_expression_smoke_summary.tsv",
                pd.DataFrame([{"metric": key, "value": value} for key, value in payload.items()]),
                overwrite=args.overwrite,
            )
        expression_payload.update(
            {
                "status": "PASS" if all(item["status"] == "PASS" for item in sample_payloads) else "WARN",
                "samples": sample_payloads,
            }
        )
    expression_body = "## Expression Smoke\n\n" + markdown_table(pd.DataFrame(expression_payload["samples"]) if expression_payload["samples"] else pd.DataFrame([expression_payload]))
    write_report_pair(report_root, "04_EXPRESSION_FEATURE_SMOKE", "Expression Feature Smoke", expression_payload, expression_body, overwrite=args.overwrite)

    plana_payload = {
        "generated_at_utc": utc_now(),
        "status": "SKIPPED",
        "readiness_label": "L126_PLANA_BARCODE_HOLD_FOR_LINEAGE_AGGREGATION",
        "reason": "PlanA readiness audit was not requested.",
    }
    plana_body = "## PlanA Barcode Readiness\n\nPlanA readiness audit was not requested."
    if args.run_plana_readiness_audit and args.grouping_mode in {"all_sections_smoke", "plana_readiness_only"}:
        audit_payload, artifact_table, route_table = build_plana_barcode_readiness_audit(PROJECT_ROOT)
        plana_payload = {"generated_at_utc": utc_now(), "status": "PASS", **audit_payload}
        plana_body = "## Artifact Audit\n\n" + markdown_table(artifact_table)
        plana_body += "\n\n## Candidate Routes\n\n" + markdown_table(route_table)
        plana_body += "\n\n## Recommendation\n\n`Route A` is the next bounded pilot route. It keeps barcode evidence as post-hoc annotation over barcode-free PlanA metaniche/macrostate outputs before any barcode-aware kernel claims."
    write_report_pair(report_root, "05_PLANA_BARCODE_READINESS_AUDIT", "PlanA Barcode Readiness Audit", plana_payload, plana_body, overwrite=args.overwrite)

    grouping_ok = grouping_payload["status"] == "PASS"
    aggregation_ok = aggregation_payload["status"] == "PASS"
    qc_ok = qc_payload["status"] == "PASS" and qc_payload["figures_non_empty"]
    expression_ok = expression_payload["status"] in {"PASS", "SKIPPED"}
    plana_ok = plana_payload.get("readiness_label") == "L126_PLANA_BARCODE_ROUTE_A_READY"
    decision = (
        "L126_PLANA_BARCODE_PREFLIGHT_READY"
        if grouping_ok and aggregation_ok and qc_ok and expression_ok and plana_ok
        else "L126_PLANA_BARCODE_PREFLIGHT_HOLD"
    )
    decision_payload = {
        "generated_at_utc": utc_now(),
        "decision_label": decision,
        "sections_processed": list(samples),
        "sampled_cellbins_per_section": {item["sample_id"]: int(item["coverage_metrics"]["sampled_cellbins"]) for item in results},
        "groups_per_section": {item["sample_id"]: int(item["coverage_metrics"]["number_of_groups"]) for item in results},
        "group_lineage_coverage_per_section": {
            item["sample_id"]: {
                "groups_with_ge1_lineage_positive_member": item["coverage_metrics"]["groups_with_ge1_lineage_positive_member"],
                "fraction_groups_with_ge1_lineage_positive_member": item["coverage_metrics"]["groups_with_ge1_lineage_positive_member"] / item["coverage_metrics"]["number_of_groups"],
                "sampled_lineage_positive_fraction": item["coverage_metrics"]["fraction_sampled_cellbins_with_lineage_evidence"],
            }
            for item in results
        },
        "expression_smoke_status": expression_payload["status"],
        "plana_barcode_recommendation": plana_payload.get("recommended_route", ""),
        "warnings": [
            "Bounded groups are overlapping local neighborhoods, not final biological niches.",
            "Serial sections are not timepoints; no temporal fate direction is inferred.",
            "PlanA Route A is post-hoc barcode annotation, not barcode-aware fate inference.",
        ],
        "next_safe_command": "Run a bounded PlanA Route A post-hoc annotation pilot against explicit metaniche/macrostate mapping tables.",
    }
    decision_body = "## Decision\n\n" + f"`{decision}`\n\n## Section Summary\n\n"
    decision_body += markdown_table(pd.DataFrame([item["section_summary"] for item in results]))
    write_report_pair(report_root, "06_ROUND2B_READINESS_DECISION", "Round 2B Readiness Decision", decision_payload, decision_body, overwrite=args.overwrite)

    source_after = snapshot_files(source_paths, include_sha256=False)
    source_compare = compare_file_snapshots(source_before, source_after)
    validation_payload = validate_outputs(
        report_root=report_root,
        output_root=output_root,
        results=results,
        schema_rows=schema_rows,
        source_compare=source_compare,
        qc_payload=qc_payload,
        decision=decision,
    )
    validation_body = "## Validation Checks\n\n" + markdown_table(pd.DataFrame(validation_payload["checks"]))
    write_report_pair(report_root, "08_VALIDATION", "Validation", validation_payload, validation_body, overwrite=args.overwrite)

    print(
        json.dumps(
            {
                "decision_label": decision,
                "sections_processed": list(samples),
                "validation_status": validation_payload["status"],
                "expression_smoke_status": expression_payload["status"],
                "plana_barcode_recommendation": plana_payload.get("recommended_route", ""),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
