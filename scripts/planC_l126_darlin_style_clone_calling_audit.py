#!/usr/bin/env python
"""L126 DARLIN/MosaicLineage-style cellbin clone-calling feasibility audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nichefate.darlin_clone_signature.common import path_has_forbidden_ssd
from nichefate.darlin_clone_signature.reporting import (
    atomic_write_json,
    atomic_write_tsv,
    atomic_write_tsv_gz,
    ensure_dir,
    markdown_table,
    read_table,
    utc_now,
    write_report_pair,
)
from nichefate.darlin_style_clone_calling_audit import (
    assign_joint_clones,
    build_cellbin_allele_table,
    classify_alleles_for_policy,
    compare_reference_policies,
    inspect_mosaiclineage,
    load_reference_banks,
    select_default_joint_policy,
    validate_audit_outputs,
)
from nichefate.darlin_style_clone_calling_audit.core import (
    BANK_POLICIES,
    DEFAULT_THRESHOLD,
    DE_NOVO_POLICIES,
    ThresholdSpec,
    collapse_for_mosaiclineage,
    compare_to_empirical_models,
    decide_final_label,
    file_sha256,
    matrix_preview,
    summarize_filtering,
    summarize_joint_assignment,
)


DEFAULT_INPUT_PACKET_ROOT = Path("/home/zhutao/scratch/nichefate/spatio_darlin_public_brain_hpc/input_packet")
DEFAULT_MOSAIC_ROOT = Path("/home/zhutao/projects/darlin_cell_repro/code/MosaicLineage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-packet-root", type=Path, default=DEFAULT_INPUT_PACKET_ROOT)
    parser.add_argument("--mosaiclineage-root", type=Path, default=DEFAULT_MOSAIC_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_MOSAIC_ROOT / "reference")
    parser.add_argument("--round1-root", type=Path, default=Path("processed/l126_darlin_clone_integration_round1"))
    parser.add_argument("--round2-root", type=Path, default=Path("processed/l126_darlin_clone_signature_round2"))
    parser.add_argument("--round2-1-root", type=Path, default=Path("processed/l126_darlin_clone_signature_round2_1"))
    parser.add_argument("--output-root", type=Path, default=Path("processed/l126_darlin_style_clone_calling_audit"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/l126_darlin_style_clone_calling_audit"))
    parser.add_argument("--prob-cutoff", type=float, default=0.1)
    parser.add_argument("--sample-count-cutoff", type=int, default=2)
    parser.add_argument("--joint-allele-n-cutoff", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def lineage_path(input_packet_root: Path) -> Path:
    return input_packet_root / "processed/lineage_evidence/cellbin_lineage_evidence.tsv.gz"


def annotation_path(input_packet_root: Path) -> Path:
    return input_packet_root / "processed/lineage_evidence/feature_allele_annotation_long.tsv.gz"


def reject_forbidden_paths(*paths: Path) -> None:
    bad = [str(path) for path in paths if path_has_forbidden_ssd(path)]
    if bad:
        raise ValueError("Refusing /ssd paths: " + "; ".join(bad))


def input_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): file_sha256(path) for path in paths if path.exists()}


def threshold_grid(reference_banks: dict[str, pd.DataFrame]) -> list[ThresholdSpec]:
    max_sample_count = 3
    if reference_banks:
        max_sample_count = int(max(frame["sample_count"].replace(float("inf"), 0).max() for frame in reference_banks.values()))
        max_sample_count = max(max_sample_count, 3)
    rows: list[ThresholdSpec] = []
    for prob_label, prob in [("conservative", 0.01), ("tutorial_like", 0.1), ("permissive", 0.5)]:
        for sample_cutoff in [1, 2, 3, max_sample_count + 1]:
            for min_cellbins in [1, 2]:
                rows.append(ThresholdSpec(f"{prob_label}_sample{sample_cutoff}_mincell{min_cellbins}", prob, sample_cutoff, min_cellbins))
    return rows


def h5ad_inventory(input_packet_root: Path) -> dict[str, Any]:
    paths = sorted((input_packet_root / "processed/h5ad").glob("L126_Brain_s*.mRNA_processed.h5ad"))
    return {"n_l126_mrna_h5ad_files": len(paths), "h5ad_paths": [str(path) for path in paths], "used_for_clone_calling": False}


def existing_joint_clone_inventory(input_packet_root: Path) -> dict[str, Any]:
    files = [path for path in input_packet_root.glob("processed/**/*") if path.is_file() and "joint" in path.name.lower() and "clone" in path.name.lower()]
    line_cols = read_table(lineage_path(input_packet_root), nrows=1).columns.tolist() if lineage_path(input_packet_root).exists() else []
    has_joint_col = any("joint_clone" in col.lower() for col in line_cols)
    return {
        "spatio_darlin_joint_clone_table_found": bool(files or has_joint_col),
        "joint_clone_like_files": [str(path) for path in files],
        "lineage_evidence_has_joint_clone_column": bool(has_joint_col),
    }


def write_phase0(report_root: Path, payload: dict[str, Any], banks: dict[str, pd.DataFrame], *, overwrite: bool) -> None:
    bank_schema = []
    for policy, frame in banks.items():
        bank_schema.append(
            {
                "reference_bank_policy": policy,
                "columns": ",".join(frame.columns),
                "n_rows": len(frame),
                "n_unique_normalized_alleles": frame[["locus", "allele_normalized"]].drop_duplicates().shape[0],
                "invalid_alleles_column_present_or_imputed": "invalid_alleles" in frame.columns,
            }
        )
    payload = {**payload, "reference_bank_schema": bank_schema}
    write_report_pair(
        report_root,
        "00_MOSAICLINEAGE_AVAILABILITY",
        "MosaicLineage Availability",
        payload,
        [
            "## Availability",
            f"- Decision label: `{payload['decision_label']}`",
            f"- Local source function found: {payload['source_function_found']}",
            f"- Direct importable: {payload['import_status']['direct_darlin_importable']}",
            f"- Direct import error: `{payload['import_status']['direct_import_error']}`",
            f"- Bio needed by inspected function body: {payload['bio_needed_by_function_body']}",
            "",
            "## Function Logic",
            "- Required input columns: `RNA_id`, `locus`, `allele`, `normalized_count`, `sample_count`.",
            "- `joint_clone_id_tmp` joins CA/RA/TA allele values and fills missing loci as `nan`.",
            "- `joint_prob` is the product of locus allele probabilities with missing loci treated as 1.",
            "- `joint_allele_num` is the number of unique alleles in the connected component.",
            "- Ambiguous high-coupling alleles are prevented from creating strong links by the `joint_allele_N_cutoff` rule.",
            "",
            "## Reference Bank Schema",
            markdown_table(pd.DataFrame(bank_schema)),
        ],
        overwrite=overwrite,
    )


def write_phase1(output_root: Path, report_root: Path, allele_table: pd.DataFrame, inventory: dict[str, Any], joint_inventory: dict[str, Any], *, overwrite: bool) -> dict[str, Any]:
    atomic_write_tsv_gz(output_root / "cellbin_allele_table.tsv.gz", allele_table, overwrite=overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "n_rows": int(len(allele_table)),
        "n_cellbins": int(allele_table["RNA_id"].nunique()),
        "n_nonmissing_allele_rows": int((~allele_table["allele_is_missing"]).sum()),
        "n_assay_scoped_features": int(allele_table["assay_scoped_feature_id"].nunique()),
        "locus_distribution": allele_table["locus"].value_counts().to_dict(),
        "h5ad_inventory": inventory,
        "existing_joint_clone_inventory": joint_inventory,
    }
    write_report_pair(
        report_root,
        "01_L126_TO_DARLIN_SCHEMA_CONVERSION",
        "L126 To DARLIN Schema Conversion",
        payload,
        [
            "## Conversion",
            f"- Rows: {payload['n_rows']}",
            f"- Cellbins: {payload['n_cellbins']}",
            f"- Nonmissing allele rows: {payload['n_nonmissing_allele_rows']}",
            f"- spatio_DARLIN joint clone table already present: {joint_inventory['spatio_darlin_joint_clone_table_found']}",
            "- mRNA h5ad files were checked only for identity/metadata availability, not clone calling.",
            "- Input `clone_id` was not used as final clone.",
        ],
        overwrite=overwrite,
    )
    return payload


def write_phase2(output_root: Path, report_root: Path, mapping_summary: pd.DataFrame, full_mapped: pd.DataFrame, *, overwrite: bool) -> dict[str, Any]:
    atomic_write_tsv(output_root / "allele_bank_mapping_summary.tsv", mapping_summary, overwrite=overwrite)
    atomic_write_tsv_gz(output_root / "cellbin_allele_table_with_reference.tsv.gz", full_mapped, overwrite=overwrite)
    best = mapping_summary.sort_values("row_mapping_fraction", ascending=False).head(1).to_dict(orient="records")
    max_fraction = float(mapping_summary["row_mapping_fraction"].max()) if not mapping_summary.empty else 0.0
    label = "ALLELE_BANK_MAPPING_READY" if max_fraction >= 0.50 else ("ALLELE_BANK_MAPPING_PARTIAL_WITH_WARNINGS" if max_fraction > 0 else "HOLD_FOR_ALLELE_SCHEMA_MISMATCH")
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": label,
        "max_row_mapping_fraction": max_fraction,
        "best_mapping_row": best[0] if best else {},
        "bank_policies_audited": list(BANK_POLICIES),
        "mapping_modes_audited": ["raw_exact", "normalized", "normalized_locus_formatted"],
    }
    write_report_pair(
        report_root,
        "02_ALLELE_BANK_MAPPING_AUDIT",
        "Allele Bank Mapping Audit",
        payload,
        [
            "## Mapping Result",
            f"- Decision label: `{label}`",
            f"- Maximum row mapping fraction: {max_fraction:.6f}",
            "- Raw exact, normalized, and normalized locus-formatted mapping modes were audited.",
            "- Plain, `_Gr`, and union reference bank policies were compared.",
            "",
            "## Top Mapping Rows",
            markdown_table(mapping_summary.sort_values("row_mapping_fraction", ascending=False).head(12)),
        ],
        overwrite=overwrite,
    )
    return payload


def run_filtering_grid(mapped_tables: dict[str, pd.DataFrame], banks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for policy, mapped in mapped_tables.items():
        for threshold in threshold_grid(banks):
            for de_novo_policy in DE_NOVO_POLICIES:
                classified = classify_alleles_for_policy(mapped, threshold, de_novo_policy)
                rows.append(summarize_filtering(classified, policy, de_novo_policy, threshold))
    return pd.DataFrame(rows)


def run_joint_policy_grid(
    mapped_tables: dict[str, pd.DataFrame],
    threshold: ThresholdSpec,
    joint_allele_n_cutoff: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    rows = []
    classified_tables: dict[str, pd.DataFrame] = {}
    collapsed_tables: dict[str, pd.DataFrame] = {}
    assignment_tables: dict[str, pd.DataFrame] = {}
    clone_summary_tables: dict[str, pd.DataFrame] = {}
    allele_tables: dict[str, pd.DataFrame] = {}
    for policy, mapped in mapped_tables.items():
        for de_novo_policy in DE_NOVO_POLICIES:
            key = f"{policy}__{de_novo_policy}"
            classified = classify_alleles_for_policy(mapped, threshold, de_novo_policy)
            collapsed = collapse_for_mosaiclineage(classified)
            assigned_clones, assigned_rows, allele_table, status = assign_joint_clones(
                collapsed,
                prob_cutoff=threshold.prob_cutoff,
                sample_count_cutoff=threshold.sample_count_cutoff,
                joint_allele_N_cutoff=joint_allele_n_cutoff,
            )
            per_cell, clone_summary, payload = summarize_joint_assignment(
                assigned_clones,
                assigned_rows,
                classified,
                bank_policy=policy,
                de_novo_policy=de_novo_policy,
                threshold=threshold,
            )
            payload.update(status)
            rows.append(payload)
            classified_tables[key] = classified
            collapsed_tables[key] = collapsed
            assignment_tables[key] = per_cell
            clone_summary_tables[key] = clone_summary
            allele_tables[key] = allele_table
    return pd.DataFrame(rows), classified_tables, collapsed_tables, assignment_tables, clone_summary_tables, allele_tables


def write_phase3(output_root: Path, report_root: Path, filtering: pd.DataFrame, selected_valid: pd.DataFrame, *, overwrite: bool) -> dict[str, Any]:
    atomic_write_tsv(output_root / "rare_allele_filtering_summary.tsv", filtering, overwrite=overwrite)
    atomic_write_tsv_gz(output_root / "valid_cellbin_allele_table.tsv.gz", selected_valid, overwrite=overwrite)
    best = filtering.sort_values(["fraction_cellbins_with_valid_alleles_ge2_loci", "valid_allele_supported_cellbin_fraction"], ascending=False).head(1)
    payload = {
        "generated_at_utc": utc_now(),
        "n_filtering_rows": int(len(filtering)),
        "best_filtering_row": best.to_dict(orient="records")[0] if not best.empty else {},
        "selected_valid_rows": int(len(selected_valid)),
        "selected_valid_cellbins": int(selected_valid["RNA_id"].nunique()) if not selected_valid.empty else 0,
    }
    write_report_pair(
        report_root,
        "03_RARE_ALLELE_FILTERING",
        "Rare Allele Filtering",
        payload,
        [
            "## Filtering Grid",
            f"- Grid rows: {payload['n_filtering_rows']}",
            "- De novo policies A/B/C were evaluated.",
            f"- Selected valid allele table rows: {payload['selected_valid_rows']}",
            f"- Selected valid allele-supported cellbins: {payload['selected_valid_cellbins']}",
            "",
            "## Top Recovery Rows",
            markdown_table(filtering.sort_values(["fraction_cellbins_with_valid_alleles_ge2_loci", "valid_allele_supported_cellbin_fraction"], ascending=False).head(12)),
        ],
        overwrite=overwrite,
    )
    return payload


def write_phase4(
    output_root: Path,
    report_root: Path,
    policy_summary: pd.DataFrame,
    selected: dict[str, Any],
    selected_assignment: pd.DataFrame,
    selected_clone_summary: pd.DataFrame,
    selected_joint_alleles: pd.DataFrame,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    atomic_write_tsv(output_root / "joint_clone_policy_summary.tsv", policy_summary, overwrite=overwrite)
    atomic_write_tsv_gz(output_root / "cellbin_joint_clone_assignment.tsv.gz", selected_assignment, overwrite=overwrite)
    atomic_write_tsv_gz(output_root / "joint_clone_summary.tsv.gz", selected_clone_summary, overwrite=overwrite)
    atomic_write_tsv_gz(output_root / "joint_allele_table.tsv.gz", selected_joint_alleles, overwrite=overwrite)
    assigned_fraction = float(selected.get("joint_clone_assigned_cellbin_fraction", 0.0) or 0.0)
    if assigned_fraction <= 0:
        label = "HOLD_FOR_LOW_JOINT_CLONE_RECOVERY"
    elif selected.get("de_novo_policy") != "mapped_rare_only":
        label = "L126_JOINT_CLONE_CALLING_READY_WITH_SPATIAL_WARNINGS"
    else:
        label = "L126_JOINT_CLONE_CALLING_READY"
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": label,
        "selected_policy": selected,
        "n_policy_rows": int(len(policy_summary)),
        "selected_assignment_rows": int(len(selected_assignment)),
        "selected_joint_clone_rows": int(len(selected_clone_summary)),
        "source_adapter_used": True,
        "schema_mismatch": False,
    }
    write_report_pair(
        report_root,
        "04_MOSAICLINEAGE_JOINT_CLONE_CALLING",
        "MosaicLineage Joint Clone Calling",
        payload,
        [
            "## Joint Clone Calling",
            f"- Decision label: `{label}`",
            "- Direct package import was not required for execution; the inspected function logic was reproduced locally for the needed v0 algorithm.",
            f"- Selected bank policy: `{selected.get('reference_bank_policy', '')}`",
            f"- Selected de novo policy: `{selected.get('de_novo_policy', '')}`",
            f"- Joint clones: {selected.get('n_joint_clones', 0)}",
            f"- Joint clone-assigned cellbin fraction: {assigned_fraction:.6f}",
            f"- Recurrent joint clone cellbin fraction: {float(selected.get('recurrent_joint_clone_cellbin_fraction', 0.0) or 0.0):.6f}",
            "",
            "## Policy Summary",
            markdown_table(policy_summary.sort_values("selection_score" if "selection_score" in policy_summary else "joint_clone_assigned_cellbin_fraction", ascending=False).head(12)),
        ],
        overwrite=overwrite,
    )
    return payload


def write_phase5(output_root: Path, report_root: Path, comparison: pd.DataFrame, payload: dict[str, Any], *, overwrite: bool) -> None:
    atomic_write_tsv(output_root / "comparison_to_empirical_models.tsv", comparison, overwrite=overwrite)
    write_report_pair(
        report_root,
        "05_COMPARISON_TO_EMPIRICAL_MODELS",
        "Comparison To Empirical Models",
        payload,
        [
            "## Comparison",
            markdown_table(comparison),
            "- Round 2.1 membership remains the main comparator because it preserves partial and multi-clone support.",
        ],
        overwrite=overwrite,
    )


def write_phase6(output_root: Path, report_root: Path, preview: pd.DataFrame, recommendation: str, *, overwrite: bool) -> dict[str, Any]:
    atomic_write_tsv(output_root / "cellbin_joint_clone_matrix_preview.tsv", preview, overwrite=overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "preview_rows": int(len(preview)),
        "recommendation": recommendation,
        "full_niche_aggregation_run": False,
    }
    write_report_pair(
        report_root,
        "06_CELLBIN_CLONE_TO_NICHE_ADAPTER_DESIGN",
        "Cellbin Clone To Niche Adapter Design",
        payload,
        [
            "## Adapter Design",
            "- `cellbin_joint_clone_assignment.tsv.gz` can be converted to a sparse `cellbin x joint_clone_id` matrix.",
            "- Any tile or niche use should remain a downstream adapter step and was not run in this audit.",
            f"- Recommendation: {recommendation}",
            f"- Preview rows: {payload['preview_rows']}",
        ],
        overwrite=overwrite,
    )
    return payload


def write_final_report(
    report_root: Path,
    final_payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    write_report_pair(
        report_root,
        "07_FINAL_DECISION_AND_VALIDATION",
        "Final Decision And Validation",
        final_payload,
        [
            "## Final Decision",
            f"- Label: `{final_payload['final_decision_label']}`",
            "",
            "## Required Answers",
            f"- spatio_DARLIN output already had joint clones: {final_payload['spatio_darlin_joint_clone_table_found']}",
            f"- MosaicLineage reference banks found: {final_payload['mosaiclineage_reference_banks_found']}",
            f"- Best allele mapping rate: {final_payload['allele_mapping_rate']:.6f}",
            f"- Valid allele-supported cellbin fraction: {final_payload['valid_allele_supported_cellbin_fraction']:.6f}",
            f"- Joint clone count: {final_payload['joint_clone_count']}",
            f"- Joint clone-assigned cellbin fraction: {final_payload['joint_clone_assigned_cellbin_fraction']:.6f}",
            f"- Round 2.1 membership fraction: {final_payload['round2_1_membership_fraction']:.6f}",
            f"- Primary clone layer recommendation: {final_payload['primary_clone_layer_recommendation']}",
            f"- Validation status: `{final_payload['validation']['validation_status']}`",
            "",
            "## Warnings",
            markdown_table(pd.DataFrame({"warning": final_payload.get("warnings", [])})) if final_payload.get("warnings") else "- None.",
        ],
        overwrite=overwrite,
    )


def round21_fraction(round21_root: Path) -> float:
    decision = Path("reports/l126_darlin_clone_signature_round2_1/07_MEMBERSHIP_RESCUE_DECISION.json")
    if decision.exists():
        payload = json.loads(decision.read_text(encoding="utf-8"))
        return float(payload.get("membership_supported_cellbin_fraction", 0.0))
    summary = round21_root / "membership/cellbin_clone_membership_summary.tsv.gz"
    if summary.exists():
        frame = read_table(summary)
        return float(frame["assignment_mode"].isin(["single_clone_dominant", "multi_clone_supported", "ambiguous"]).mean())
    return 0.0


def main() -> None:
    args = parse_args()
    reject_forbidden_paths(args.input_packet_root, args.mosaiclineage_root, args.reference_root, args.output_root, args.report_root)
    ensure_dir(args.output_root)
    ensure_dir(args.report_root)

    line_path = lineage_path(args.input_packet_root)
    anno_path = annotation_path(args.input_packet_root)
    before = input_hashes([line_path, anno_path])

    banks = load_reference_banks(args.reference_root)
    mosaic_payload = inspect_mosaiclineage(args.mosaiclineage_root, args.reference_root)
    write_phase0(args.report_root, {**mosaic_payload, "generated_at_utc": utc_now()}, banks, overwrite=args.overwrite)

    allele_table = build_cellbin_allele_table(line_path, anno_path)
    h5ad_payload = h5ad_inventory(args.input_packet_root)
    joint_inventory = existing_joint_clone_inventory(args.input_packet_root)
    phase1_payload = write_phase1(args.output_root, args.report_root, allele_table, h5ad_payload, joint_inventory, overwrite=args.overwrite)

    mapping_summary, mapped_tables, full_mapped = compare_reference_policies(allele_table, banks)
    phase2_payload = write_phase2(args.output_root, args.report_root, mapping_summary, full_mapped, overwrite=args.overwrite)

    filtering = run_filtering_grid(mapped_tables, banks)
    joint_threshold = ThresholdSpec("tutorial_like", args.prob_cutoff, args.sample_count_cutoff, 1)
    policy_summary, classified_tables, collapsed_tables, assignment_tables, clone_summary_tables, joint_allele_tables = run_joint_policy_grid(
        mapped_tables,
        joint_threshold,
        args.joint_allele_n_cutoff,
    )
    selected = select_default_joint_policy(policy_summary)
    policy_summary = policy_summary.merge(
        pd.DataFrame([selected])[
            ["reference_bank_policy", "de_novo_policy", "selection_score"]
        ],
        on=["reference_bank_policy", "de_novo_policy"],
        how="left",
        suffixes=("", "_selected"),
    )
    selected_key = f"{selected.get('reference_bank_policy')}__{selected.get('de_novo_policy')}"
    selected_valid = classified_tables.get(selected_key, pd.DataFrame())
    selected_valid = selected_valid.loc[selected_valid.get("valid_for_joint_calling", pd.Series(dtype=bool)).astype(bool)].copy() if not selected_valid.empty else selected_valid
    phase3_payload = write_phase3(args.output_root, args.report_root, filtering, selected_valid, overwrite=args.overwrite)

    selected_assignment = assignment_tables.get(selected_key, pd.DataFrame())
    selected_clone_summary = clone_summary_tables.get(selected_key, pd.DataFrame())
    selected_joint_alleles = joint_allele_tables.get(selected_key, pd.DataFrame())
    phase4_payload = write_phase4(
        args.output_root,
        args.report_root,
        policy_summary,
        selected,
        selected_assignment,
        selected_clone_summary,
        selected_joint_alleles,
        overwrite=args.overwrite,
    )

    comparison, comparison_payload = compare_to_empirical_models(
        selected_assignment,
        {"round1": args.round1_root, "round2": args.round2_root, "round21": args.round2_1_root},
    )
    write_phase5(args.output_root, args.report_root, comparison, comparison_payload, overwrite=args.overwrite)

    r21_fraction = round21_fraction(args.round2_1_root)
    recommendation = "keep Round 2.1 membership as primary; use DARLIN-style joint_clone_id as secondary QC"
    if float(selected.get("joint_clone_assigned_cellbin_fraction", 0.0) or 0.0) > r21_fraction and selected.get("de_novo_policy") == "mapped_rare_only":
        recommendation = "eligible to replace Round 2.1 membership after biological review"
    elif float(selected.get("joint_clone_assigned_cellbin_fraction", 0.0) or 0.0) > r21_fraction:
        recommendation = "supplement Round 2.1 membership; de novo dependence prevents replacement without review"
    phase6_payload = write_phase6(args.output_root, args.report_root, matrix_preview(selected_assignment), recommendation, overwrite=args.overwrite)

    final_label, warnings, decision_stats = decide_final_label(selected, mapping_summary, r21_fraction)
    after = input_hashes([line_path, anno_path])
    validation_payload = validate_audit_outputs(args.output_root, args.report_root, before, after)
    best_filter = phase3_payload.get("best_filtering_row", {})
    final_payload = {
        "generated_at_utc": utc_now(),
        "final_decision_label": final_label,
        "warnings": warnings,
        "spatio_darlin_joint_clone_table_found": bool(joint_inventory["spatio_darlin_joint_clone_table_found"]),
        "mosaiclineage_reference_banks_found": not bool(mosaic_payload["missing_reference_files"]),
        "allele_mapping_rate": float(phase2_payload["max_row_mapping_fraction"]),
        "valid_allele_supported_cellbin_fraction": float(best_filter.get("valid_allele_supported_cellbin_fraction", 0.0) or 0.0),
        "joint_clone_count": int(selected.get("n_joint_clones", 0) or 0),
        "joint_clone_assigned_cellbin_fraction": float(selected.get("joint_clone_assigned_cellbin_fraction", 0.0) or 0.0),
        "joint_clone_assigned_cellbins": int(selected.get("n_joint_clone_assigned_cellbins", 0) or 0),
        "round2_1_membership_fraction": r21_fraction,
        "selected_reference_bank_policy": selected.get("reference_bank_policy", ""),
        "selected_de_novo_policy": selected.get("de_novo_policy", ""),
        "primary_clone_layer_recommendation": recommendation,
        "decision_stats": decision_stats,
        "phase_payloads": {
            "phase1": phase1_payload,
            "phase2": phase2_payload,
            "phase3": phase3_payload,
            "phase4": phase4_payload,
            "phase5": comparison_payload,
            "phase6": phase6_payload,
        },
        "validation": validation_payload,
        "next_safe_command": "sed -n '1,220p' reports/l126_darlin_style_clone_calling_audit/07_FINAL_DECISION_AND_VALIDATION.md",
    }
    write_final_report(args.report_root, final_payload, overwrite=args.overwrite)
    atomic_write_json(args.report_root / "07_FINAL_DECISION_AND_VALIDATION.json", final_payload, overwrite=True)


if __name__ == "__main__":
    main()
