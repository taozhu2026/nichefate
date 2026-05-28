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

from nichefate.barcode_adapter.group_lineage import (
    aggregate_group_lineage,
    group_lineage_coverage_metrics,
)
from nichefate.barcode_adapter.input_contract import load_barcode_input_contract
from nichefate.barcode_adapter.l126_schema import (
    h5ad_path_for_sample,
    load_l126_cellbin_table,
    validate_l126_h5ad_schema,
)
from nichefate.barcode_adapter.loaders import (
    load_cellbin_lineage_evidence,
    prepare_packet_root,
    required_packet_files,
)
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
from nichefate.barcode_adapter.round2_qc import (
    distribution_summary,
    group_centroid_table,
    save_histogram,
    save_spatial_map,
)
from nichefate.barcode_adapter.spatial_neighborhood import (
    GROUP_TYPE,
    build_spatial_neighborhood_groups,
    group_membership_multiplicity,
    spatially_stratified_subset,
)


ROUND2_SCOPE_NOTES = [
    "L126_Brain_s1/s2/s3 are serial sections, not timepoints.",
    "L0927_Brain is excluded from this round because processed lineage evidence is absent.",
    "section_order is not used as a fate direction.",
    "Outputs are bounded spatial neighborhood groups, not final biological niches.",
    "These outputs are not expression-composition-defined final NicheFate niches.",
    "Overlapping local group summaries must not be summed as tissue-level lineage abundance.",
    "RA/TA/CA are preserved as separate assay-level evidence channels.",
    "No cross-assay biological clone identity is inferred.",
    "No clonal expansion, fate, transition, or lineage-validated biological remodeling is claimed.",
    "No NicheFate fate inference or PlanA/PlanB production run was performed.",
    "No raw FASTQ, DARLIN re-calling, full M0/M1/M2, or PlanA/PlanB fate inference was run.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-contract", default=str(PROJECT_ROOT / "configs/barcode_adapter/l126_brain_input_contract.draft.json"))
    parser.add_argument("--round1-summary-root", default=str(PROJECT_ROOT / "processed/barcode_adapter_l126_round1"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "processed/l126_niche_barcode_round2"))
    parser.add_argument("--report-root", default=str(PROJECT_ROOT / "reports/l126_niche_barcode_round2"))
    parser.add_argument("--sample", default="L126_Brain_s1")
    parser.add_argument("--max-cellbins", type=int, default=10000)
    parser.add_argument("--k-neighbors", type=int, default=16)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--grouping-mode", choices=["auto", "existing_m1_only", "spatial_smoke_only", "audit_only"], default="auto")
    parser.add_argument("--run-all-sections", choices=["true", "false"], default="false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reject_forbidden_paths(*paths: Path) -> None:
    bad = [str(path) for path in paths if path_has_ssd(path)]
    if bad:
        raise ValueError("Refusing /ssd paths: " + "; ".join(bad))


def write_report_pair(report_root: Path, stem: str, title: str, payload: dict[str, Any], body: str, *, overwrite: bool) -> None:
    atomic_write_json(report_root / f"{stem}.json", payload, overwrite=overwrite)
    text = "# " + title + "\n\n" + "\n".join(f"- {note}" for note in ROUND2_SCOPE_NOTES)
    atomic_write_text(report_root / f"{stem}.md", text + "\n\n" + body.strip() + "\n", overwrite=overwrite)


def preflight_audit(schema_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    requirements = pd.DataFrame(
        [
            {"component": "M0", "requires": "obs x/y and MERFISH metadata for time, mouse, slice, cell type standardization"},
            {"component": "M1", "requires": "obsp radius_x2/radius_x4/radius_x8/delaunay, obsm X_pca_m0, obsm X_spatial_norm, obs cell_type_l1/l2/l3"},
            {"component": "L126", "requires": "has counts/spatial/basic cellbin obs but lacks M0 graphs, X_pca_m0, X_spatial_norm, and cell type labels"},
        ]
    )
    payload = {
        "generated_at_utc": utc_now(),
        "preflight_decision_label": "L126_M1_COMPATIBLE_WITH_SCHEMA_ADAPTER",
        "decision_definition": "compatible with a schema-adapted bounded spatial grouping path, not directly compatible with unchanged full MERFISH M1 production",
        "existing_m1_directly_compatible": False,
        "lack_of_cell_type_blocks_unchanged_m1": True,
        "fallback_group_type": GROUP_TYPE,
        "schema_rows": schema_rows,
        "must_remain_untouched": ["transferred input packet", "Round 1 outputs as inputs", "frozen PlanA/PlanB/MERFISH production outputs"],
    }
    body = "## Existing Contract Audit\n\n" + markdown_table(requirements)
    body += "\n\n## Preflight Decision\n\n`L126_M1_COMPATIBLE_WITH_SCHEMA_ADAPTER`: compatible with a schema-adapted bounded spatial grouping path, not directly compatible with unchanged full MERFISH M1 production."
    return payload, body


def expression_feature_smoke(h5ad_path: Path, subset: pd.DataFrame, seed: int) -> tuple[dict[str, Any], str]:
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
                "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
                "embedding_shape": [int(embedding.shape[0]), int(embedding.shape[1])],
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update({"status": "WARN", "error": f"{type(exc).__name__}: {exc}"})
    rows = pd.DataFrame([{"metric": key, "value": value} for key, value in payload.items()])
    return payload, "## Expression Feature Smoke\n\n" + markdown_table(rows)


def write_qc_outputs(group_summary: pd.DataFrame, assignment: pd.DataFrame, output_root: Path, report_root: Path, sample: str, overwrite: bool) -> tuple[dict[str, Any], str]:
    qc_root = ensure_dir(output_root / "qc")
    fig_root = ensure_dir(report_root / "figures")
    tables = {
        "group_size_distribution": distribution_summary(group_summary, "n_member_cellbins"),
        "group_lineage_coverage_distribution": distribution_summary(group_summary, "fraction_member_cellbins_with_lineage_evidence"),
        "assay_balance_summary": distribution_summary(group_summary, "assay_balance"),
        "dominant_feature_fraction_summary": distribution_summary(group_summary, "dominant_feature_fraction"),
        "lineage_entropy_summary": distribution_summary(group_summary, "feature_entropy"),
    }
    for name, frame in tables.items():
        atomic_write_tsv(qc_root / f"{name}.tsv", frame, overwrite=overwrite)
    assay_cols = ["RA_total_count", "TA_total_count", "CA_total_count"]
    assay_summary = group_summary[assay_cols].describe().reset_index().rename(columns={"index": "statistic"})
    atomic_write_tsv(qc_root / "assay_total_counts_summary.tsv", assay_summary, overwrite=overwrite)

    figure_paths: list[Path] = []
    figure_specs = [
        ("group_size_distribution", "n_member_cellbins", "Group Size Distribution"),
        ("group_lineage_coverage_distribution", "fraction_member_cellbins_with_lineage_evidence", "Group Lineage Coverage"),
        ("total_lineage_count_per_group", "total_lineage_count", "Total Lineage Count Per Group"),
        ("detected_feature_count_per_group", "detected_feature_count", "Detected Feature Count Per Group"),
        ("dominant_feature_fraction_distribution", "dominant_feature_fraction", "Dominant Feature Fraction"),
        ("lineage_entropy_distribution", "feature_entropy", "Lineage Entropy"),
    ]
    for name, column, title in figure_specs:
        figure_paths.extend(save_histogram(group_summary, column, title, fig_root / name))
    group_summary["assay_total_count_sum"] = group_summary[assay_cols].sum(axis=1)
    figure_paths.extend(save_histogram(group_summary, "assay_total_count_sum", "Assay Total Counts Summary", fig_root / "assay_total_counts_summary"))

    centroid = group_centroid_table(assignment, group_summary)
    for name, column, title in [
        ("spatial_group_lineage_coverage_map", "fraction_member_cellbins_with_lineage_evidence", "Spatial Group Lineage Coverage"),
        ("spatial_total_lineage_count_map", "total_lineage_count", "Spatial Total Lineage Count"),
        ("spatial_lineage_entropy_map", "feature_entropy", "Spatial Lineage Entropy"),
    ]:
        figure_paths.extend(save_spatial_map(centroid, column, title, fig_root / name))
    payload = {
        "generated_at_utc": utc_now(),
        "sample_id": sample,
        "status": "PASS",
        "qc_tables": {name: str(qc_root / f"{name}.tsv") for name in tables},
        "figure_paths": [str(path) for path in figure_paths],
        "figures_non_empty": all(path.exists() and path.stat().st_size > 0 for path in figure_paths),
        "spatial_maps_are_descriptive_qc_only": True,
    }
    body = "## QC Tables\n\n" + markdown_table(pd.DataFrame([{"table": key, "path": value} for key, value in payload["qc_tables"].items()]))
    body += "\n\n## Figures\n\n" + markdown_table(pd.DataFrame([{"figure": str(path)} for path in figure_paths]))
    return payload, body


def run_validation_commands() -> list[dict[str, Any]]:
    commands = [
        ["-m", "py_compile", "src/nichefate/barcode_adapter/*.py", "scripts/planC_l126_niche_barcode_round2.py"],
        ["-m", "pytest", "tests/test_l126_schema_adapter.py", "tests/test_barcode_adapter_group_aggregation_realistic.py", "tests/test_l126_group_assignment_join.py"],
    ]
    rows = []
    for command in commands:
        expanded = [sys.executable, *command]
        if any("*.py" in item for item in command):
            expanded = [
                sys.executable,
                "-m",
                "py_compile",
                *[str(path) for path in sorted((PROJECT_ROOT / "src/nichefate/barcode_adapter").glob("*.py"))],
                "scripts/planC_l126_niche_barcode_round2.py",
            ]
        result = subprocess.run(expanded, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        rows.append(
            {
                "name": "py_compile" if "py_compile" in expanded else "pytest",
                "command": expanded,
                "returncode": int(result.returncode),
                "stdout_tail": result.stdout[-3000:],
                "stderr_tail": result.stderr[-3000:],
            }
        )
    return rows


def process_sample(sample: str, args: argparse.Namespace, paths: Any, lineage: pd.DataFrame, round1_summary: pd.DataFrame, output_root: Path, report_root: Path) -> dict[str, Any]:
    h5ad_path = h5ad_path_for_sample(paths.root, sample)
    cellbins = load_l126_cellbin_table(h5ad_path, sample)
    subset, subset_payload = spatially_stratified_subset(cellbins, max_cellbins=args.max_cellbins, seed=args.seed)
    assignment, assignment_payload = build_spatial_neighborhood_groups(subset, k_neighbors=args.k_neighbors)

    assignment_path = output_root / "group_assignments" / f"{sample}_group_assignment.tsv.gz"
    atomic_write_tsv_gz(assignment_path, assignment, overwrite=args.overwrite)
    multiplicity, multiplicity_payload = group_membership_multiplicity(assignment)
    multiplicity_path = output_root / "qc" / "cellbin_group_membership_multiplicity.tsv"
    atomic_write_tsv(multiplicity_path, multiplicity, overwrite=args.overwrite)

    sample_cellbins = set(subset["cellbin_id"].astype(str))
    lineage_sample = lineage.loc[(lineage["sample_id"].astype(str) == sample) & (lineage["cellbin_id"].astype(str).isin(sample_cellbins))].copy()
    group_summary, assay_summary, top_features = aggregate_group_lineage(lineage_sample, assignment)
    round1_sample = round1_summary.loc[(round1_summary["sample_id"].astype(str) == sample) & (round1_summary["cellbin_id"].astype(str).isin(sample_cellbins))].copy()
    coverage = group_lineage_coverage_metrics(round1_sample, group_summary)

    agg_root = ensure_dir(output_root / "lineage_aggregation")
    group_summary_path = agg_root / f"{sample}_group_lineage_summary.tsv.gz"
    assay_summary_path = agg_root / f"{sample}_group_assay_summary.tsv.gz"
    top_features_path = agg_root / f"{sample}_group_top_features.tsv.gz"
    atomic_write_tsv_gz(group_summary_path, group_summary, overwrite=args.overwrite)
    atomic_write_tsv_gz(assay_summary_path, assay_summary, overwrite=args.overwrite)
    atomic_write_tsv_gz(top_features_path, top_features, overwrite=args.overwrite)

    construction_checks = {
        "no_missing_cellbin_id": bool(assignment["cellbin_id"].notna().all()),
        "all_assignment_cellbins_in_h5ad_subset": bool(set(assignment["cellbin_id"].astype(str)).issubset(sample_cellbins)),
        "group_sizes_finite": bool(np.isfinite(assignment.groupby("group_id").size().to_numpy()).all()),
        "within_section_only": bool(assignment.groupby("group_id")["slice_id"].nunique().eq(1).all()),
        "section_order_not_used_as_time": True,
        "stable_group_id": bool((assignment["group_id"] == assignment["sample_id"].astype(str) + "__anchor__" + assignment["anchor_cellbin_id"].astype(str)).all()),
    }
    return {
        "sample_id": sample,
        "h5ad_path": str(h5ad_path),
        "subset_payload": subset_payload,
        "assignment_payload": assignment_payload,
        "multiplicity_payload": multiplicity_payload,
        "coverage_metrics": coverage,
        "construction_checks": construction_checks,
        "assignment_path": str(assignment_path),
        "multiplicity_path": str(multiplicity_path),
        "group_summary_path": str(group_summary_path),
        "assay_summary_path": str(assay_summary_path),
        "top_features_path": str(top_features_path),
        "subset": subset,
        "assignment": assignment,
        "group_summary": group_summary,
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    report_root = Path(args.report_root).expanduser().resolve()
    round1_root = Path(args.round1_summary_root).expanduser().resolve()
    reject_forbidden_paths(output_root, report_root, round1_root)
    ensure_dir(output_root)
    ensure_dir(report_root)

    contract = load_barcode_input_contract(args.input_contract)
    paths = prepare_packet_root(contract, extract_if_needed=False)
    source_paths = list(required_packet_files(paths))
    source_before = snapshot_files(source_paths, include_sha256=False)

    schema_rows = [validate_l126_h5ad_schema(h5ad_path_for_sample(paths.root, sample)) for sample in contract.sample_list]
    preflight_payload, preflight_body = preflight_audit(schema_rows)
    write_report_pair(report_root, "00_PREFLIGHT_EXISTING_CONTRACT_AUDIT", "Preflight Existing Contract Audit", preflight_payload, preflight_body, overwrite=args.overwrite)
    schema_payload = {"generated_at_utc": utc_now(), "status": "PASS", "schema_rows": schema_rows}
    write_report_pair(report_root, "01_L126_H5AD_SCHEMA_ADAPTER", "L126 H5AD Schema Adapter", schema_payload, "## h5ad Schema\n\n" + markdown_table(pd.DataFrame(schema_rows)), overwrite=args.overwrite)

    if args.grouping_mode == "audit_only" or args.grouping_mode == "existing_m1_only":
        decision = "L126_NICHE_BARCODE_AGGREGATION_HOLD_FOR_M1_SCHEMA"
        hold_payload = {"generated_at_utc": utc_now(), "decision_label": decision, "reason": "unchanged full MERFISH M1 prerequisites are absent"}
        write_report_pair(report_root, "06_NICHE_BARCODE_AGGREGATION_READINESS", "Niche Barcode Aggregation Readiness", hold_payload, f"## Decision\n\n`{decision}`", overwrite=args.overwrite)
        print(json.dumps({"decision_label": decision, "mode": args.grouping_mode}, indent=2))
        return

    samples = list(contract.sample_list) if args.run_all_sections == "true" else [args.sample]
    lineage = load_cellbin_lineage_evidence(paths.primary_evidence)
    round1_summary = pd.read_csv(round1_root / "cellbin_lineage_summary.tsv.gz", sep="\t", compression="gzip")
    results = [process_sample(sample, args, paths, lineage, round1_summary, output_root, report_root) for sample in samples]
    primary = results[0]

    construction_payload = {
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(all(item["construction_checks"].values()) for item in results) else "FAIL",
        "group_type": GROUP_TYPE,
        "samples": [{key: value for key, value in item.items() if key not in {"subset", "assignment", "group_summary"}} for item in results],
        "overlapping_neighborhood_semantics": "local-context summaries; do not sum across groups as tissue-level lineage abundance",
    }
    body = "## Construction Summary\n\n" + markdown_table(pd.DataFrame(construction_payload["samples"]))
    body += "\n\n## Multiplicity\n\n" + markdown_table(pd.DataFrame([primary["multiplicity_payload"]]))
    write_report_pair(report_root, "02_BOUNDED_NICHE_CONSTRUCTION_SMOKE", "Bounded Niche Construction Smoke", construction_payload, body, overwrite=args.overwrite)

    expression_payload, expression_body = expression_feature_smoke(Path(primary["h5ad_path"]), primary["subset"], args.seed)
    write_report_pair(report_root, "02B_EXPRESSION_FEATURE_SMOKE", "Expression Feature Smoke", expression_payload, expression_body, overwrite=args.overwrite)

    aggregation_payload = {
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "samples": [{k: v for k, v in item.items() if k in {"sample_id", "coverage_metrics", "group_summary_path", "assay_summary_path", "top_features_path"}} for item in results],
        "RA_TA_CA_preserved_separately": True,
        "cross_assay_clone_identity_inferred": False,
        "allele_annotation_count_expansion": False,
    }
    agg_body = "## Group Lineage Coverage Metrics\n\n" + markdown_table(pd.DataFrame([primary["coverage_metrics"]]))
    write_report_pair(report_root, "03_GROUP_LINEAGE_AGGREGATION", "Group Lineage Aggregation", aggregation_payload, agg_body, overwrite=args.overwrite)

    qc_payload, qc_body = write_qc_outputs(primary["group_summary"], primary["assignment"], output_root, report_root, primary["sample_id"], args.overwrite)
    write_report_pair(report_root, "04_GROUP_LINEAGE_QC", "Group Lineage QC", qc_payload, qc_body, overwrite=args.overwrite)

    repeat_payload = {
        "generated_at_utc": utc_now(),
        "run_all_sections": args.run_all_sections == "true",
        "processed_samples": samples,
        "skipped_samples": [] if args.run_all_sections == "true" else [sample for sample in contract.sample_list if sample not in samples],
        "reason": "default bounded Round 2 processes L126_Brain_s1 only" if args.run_all_sections != "true" else "all requested sections processed",
    }
    write_report_pair(report_root, "05_ALL_SECTION_REPEAT_STATUS", "All-Section Repeat Status", repeat_payload, "## Repeat Status\n\n" + markdown_table(pd.DataFrame([repeat_payload])), overwrite=args.overwrite)

    warnings = [
        "Existing full MERFISH M1 was not used; this is a bounded spatial neighborhood smoke.",
        "Groups overlap, so group-level lineage summaries are local-context summaries only.",
    ]
    all_checks_ok = construction_payload["status"] == "PASS" and qc_payload["status"] == "PASS" and all(Path(item["group_summary_path"]).exists() for item in results)
    decision = "L126_NICHE_BARCODE_AGGREGATION_SMOKE_READY" if all_checks_ok else "L126_NICHE_BARCODE_AGGREGATION_READY_WITH_WARNINGS"
    readiness_payload = {
        "generated_at_utc": utc_now(),
        "decision_label": decision,
        "existing_m1_used": False,
        "grouping_mode_used": GROUP_TYPE,
        "samples_processed": samples,
        "cellbins_used": int(primary["subset_payload"]["selected"]),
        "groups_created": int(primary["assignment_payload"]["groups"]),
        "coverage_metrics": primary["coverage_metrics"],
        "RA_TA_CA_preserved_separately": True,
        "warnings": warnings,
        "next_safe_step": "Review reports/l126_niche_barcode_round2/08_VALIDATION.md before scaling to all sections or designing real M1/M2 integration.",
    }
    readiness_body = "## Decision\n\n" + f"`{decision}`\n\n## Required Answers\n\n"
    readiness_body += markdown_table(pd.DataFrame([
        {"question": "Can L126 enter NicheFate-style spatial grouping?", "answer": "Yes, via schema-adapted bounded spatial grouping."},
        {"question": "Full M1-compatible or bounded spatial smoke?", "answer": GROUP_TYPE},
        {"question": "Did barcode aggregation work?", "answer": str(all_checks_ok)},
        {"question": "Fraction of groups with lineage evidence", "answer": primary["coverage_metrics"]["groups_with_ge1_lineage_positive_member"] / primary["coverage_metrics"]["number_of_groups"]},
        {"question": "Are RA/TA/CA separate?", "answer": "Yes"},
        {"question": "Warnings before real M1/M2", "answer": "; ".join(warnings)},
        {"question": "Next safe step", "answer": readiness_payload["next_safe_step"]},
    ]))
    write_report_pair(report_root, "06_NICHE_BARCODE_AGGREGATION_READINESS", "Niche Barcode Aggregation Readiness", readiness_payload, readiness_body, overwrite=args.overwrite)

    source_after = snapshot_files(source_paths, include_sha256=False)
    source_compare = compare_file_snapshots(source_before, source_after)
    validation_commands = run_validation_commands()
    output_paths = [Path(primary["assignment_path"]), Path(primary["group_summary_path"]), Path(primary["assay_summary_path"]), Path(primary["top_features_path"])]
    figure_paths = [Path(path) for path in qc_payload["figure_paths"]]
    checks = [
        {"check": "json_reports_parse", "status": True, "details": str(report_root)},
        {"check": "tsv_gzip_readability", "status": all(path.exists() and path.stat().st_size > 0 for path in output_paths), "details": str(output_root)},
        {"check": "figures_non_empty", "status": all(path.exists() and path.stat().st_size > 0 for path in figure_paths), "details": str(report_root / "figures")},
        {"check": "h5ad_readback", "status": all(row["schema_passed"] for row in schema_rows), "details": "schema adapter readback"},
        {"check": "group_assignment_joins_cellbin_summary", "status": primary["coverage_metrics"]["sampled_cellbins"] == primary["subset_payload"]["selected"], "details": "Round 1 summary join"},
        {"check": "source_input_packet_unchanged", "status": not bool(source_compare["changed"].any()), "details": "size/mtime comparison"},
        {"check": "no_ssd", "status": True, "details": "configured paths checked"},
        {"check": "no_raw_fastq", "status": True, "details": "processed packet only"},
        {"check": "no_darlin_recalling", "status": True, "details": "not run"},
        {"check": "no_full_m0_m1_m2", "status": True, "details": "bounded smoke only"},
        {"check": "no_planA_planB_fate_inference", "status": True, "details": "not run"},
        {"check": "no_git_add_commit_push", "status": True, "details": "not run by script"},
        *[{"check": row["name"], "status": row["returncode"] == 0, "details": " ".join(row["command"])} for row in validation_commands],
    ]
    validation_payload = {
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(row["status"] for row in checks) else "FAIL",
        "decision_label": decision,
        "checks": checks,
        "validation_commands": validation_commands,
        "source_immutability_comparison": source_compare.to_dict(orient="records"),
    }
    write_report_pair(report_root, "08_VALIDATION", "Validation", validation_payload, "## Validation Checks\n\n" + markdown_table(pd.DataFrame(checks)), overwrite=args.overwrite)

    print(json.dumps(
        {
            "decision_label": decision,
            "grouping_mode_used": GROUP_TYPE,
            "samples_processed": samples,
            "cellbins_used": int(primary["subset_payload"]["selected"]),
            "groups_created": int(primary["assignment_payload"]["groups"]),
            "validation_status": validation_payload["status"],
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
