#!/usr/bin/env python
"""Build L126 unified DARLIN-style joint clone niche layer v1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nichefate.darlin_clone_signature.reporting import (
    atomic_write_json,
    atomic_write_tsv,
    atomic_write_tsv_gz,
    ensure_dir,
    read_table,
)
from nichefate.darlin_joint_clone_niche_v1 import (
    MIN_CELLBINS_PER_ALLELE,
    NORMALIZED_COUNT_CUTOFF,
    REQUIRED_CLONE_QC_COLUMNS,
    SAMPLE_COUNT_CUTOFF,
    SELECTED_ALLELE_POLICY,
    SELECTED_REFERENCE_BANK_POLICY,
    SELECTED_THRESHOLD_LABEL,
    build_cellbin_assignment,
    build_cellbin_matrix,
    build_validated_clone_summary,
    comparison_table,
    final_label,
    input_packet_hashes,
    load_selected_audit_tables,
    load_tile_map,
    make_figures,
    qc_distribution_table,
    validate_outputs,
    write_aggregations,
    write_text_reports,
    write_validation_report,
)


DEFAULT_INPUT_PACKET_ROOT = Path("/home/zhutao/scratch/nichefate/spatio_darlin_public_brain_hpc/input_packet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path("processed/l126_darlin_style_clone_calling_audit"))
    parser.add_argument("--full-characterization-root", type=Path, default=Path("processed/l126_full_barcode_niche_characterization"))
    parser.add_argument("--round1-root", type=Path, default=Path("processed/l126_darlin_clone_integration_round1"))
    parser.add_argument("--round2-root", type=Path, default=Path("processed/l126_darlin_clone_signature_round2"))
    parser.add_argument("--round2-1-root", type=Path, default=Path("processed/l126_darlin_clone_signature_round2_1"))
    parser.add_argument("--input-packet-root", type=Path, default=DEFAULT_INPUT_PACKET_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("processed/l126_darlin_joint_clone_niche_v1"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/l126_darlin_joint_clone_niche_v1"))
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    ensure_dir(args.report_root)
    before_hashes = input_packet_hashes(args.input_packet_root)

    tables = load_selected_audit_tables(args.audit_root)
    tile_map = load_tile_map(args.full_characterization_root)
    total_st_cellbins = int(tile_map["cell_key"].nunique())
    total_lineage_cellbins = int(read_table(args.audit_root / "cellbin_allele_table.tsv.gz")["RNA_id"].nunique())

    clone_qc = build_validated_clone_summary(tables["clone_summary"], tables["valid_alleles"])
    clone_root = ensure_dir(args.output_root / "clones")
    atomic_write_tsv_gz(clone_root / "validated_joint_clone_summary.tsv.gz", clone_qc, overwrite=args.overwrite)

    assignment = build_cellbin_assignment(tables["assignment"], clone_qc)
    atomic_write_tsv_gz(clone_root / "cellbin_joint_clone_assignment.tsv.gz", assignment, overwrite=args.overwrite)

    cell_summary, clone_index, matrix_payload = build_cellbin_matrix(
        tile_map,
        assignment,
        clone_qc,
        ensure_dir(args.output_root / "matrix"),
        overwrite=args.overwrite,
    )
    niche_payload = write_aggregations(
        args.output_root,
        args.full_characterization_root,
        cell_summary,
        overwrite=args.overwrite,
    )

    comparison = comparison_table(
        args.audit_root,
        args.round1_root,
        args.round2_root,
        args.round2_1_root,
        args.output_root,
        total_lineage_cellbins,
        total_st_cellbins,
    )
    comparison_root = ensure_dir(args.output_root / "comparison")
    atomic_write_tsv(comparison_root / "reference_vs_unified_recovery.tsv", comparison, overwrite=args.overwrite)

    figure_payload = {"n_figures": 0, "n_key_figures": 0, "figures": []}
    if args.make_figures:
        figure_payload = make_figures(
            args.output_root,
            args.report_root,
            args.full_characterization_root,
            clone_qc,
            cell_summary,
            overwrite=args.overwrite,
        )

    valid_clones = clone_qc.loc[clone_qc["qc_status"].isin(["pass", "warning"])]
    assigned_cellbins = int(assignment.loc[assignment["qc_status"].isin(["pass", "warning"]), "cell_key"].nunique())
    assigned_fraction_lineage = assigned_cellbins / max(total_lineage_cellbins, 1)
    assigned_fraction_all_st = assigned_cellbins / max(total_st_cellbins, 1)
    largest_clone = valid_clones.sort_values("n_cellbins", ascending=False).head(1)
    largest_clone_cellbins = int(largest_clone["n_cellbins"].iloc[0]) if not largest_clone.empty else 0
    largest_clone_fraction = float(largest_clone["clone_size_fraction"].iloc[0]) if not largest_clone.empty else 0.0
    final_decision_label = final_label(clone_qc, assigned_fraction_lineage)

    policy_payload = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "selected_reference_bank_policy": SELECTED_REFERENCE_BANK_POLICY,
        "selected_allele_policy": SELECTED_ALLELE_POLICY,
        "selected_threshold_label": SELECTED_THRESHOLD_LABEL,
        "normalized_count_cutoff": NORMALIZED_COUNT_CUTOFF,
        "sample_count_cutoff": SAMPLE_COUNT_CUTOFF,
        "min_cellbins_per_allele": MIN_CELLBINS_PER_ALLELE,
        "reference_bank_status_role": "allele_level_qc_metadata",
        "primary_clone_unit": "validated_darlin_style_joint_clone",
        "reference_only_role": "conservative_qc_benchmark_and_sensitivity",
    }
    clone_payload = {
        "n_validated_joint_clones": int(len(valid_clones)),
        "n_total_joint_clone_rows": int(len(clone_qc)),
        "n_clone_assigned_cellbins": assigned_cellbins,
        "total_lineage_positive_cellbins": total_lineage_cellbins,
        "total_st_cellbins": total_st_cellbins,
        "assigned_fraction_lineage_positive": float(assigned_fraction_lineage),
        "assigned_fraction_all_st": float(assigned_fraction_all_st),
        "required_qc_columns_present": bool(set(REQUIRED_CLONE_QC_COLUMNS).issubset(clone_qc.columns)),
    }
    qc_payload = {
        "n_pass_clones": int(clone_qc["qc_status"].eq("pass").sum()),
        "n_warning_clones": int(clone_qc["qc_status"].eq("warning").sum()),
        "n_filtered_clones": int(clone_qc["qc_status"].eq("filtered").sum()),
        "largest_clone_cellbins": largest_clone_cellbins,
        "largest_clone_fraction": float(largest_clone_fraction),
        "n_giant_clone_flags": int(clone_qc["giant_clone_flag"].sum()),
        "n_overmerge_risk_flags": int(clone_qc["overmerge_risk_flag"].sum()),
        "n_homoplasy_risk_flags": int(clone_qc["homoplasy_risk_flag"].sum()),
        "distribution_table": qc_distribution_table(clone_qc),
    }
    comparison_payload = {
        "comparison_rows": int(len(comparison)),
        "reference_only_assigned_fraction": float(
            comparison.loc[comparison["model"].eq("reference_only_conservative_benchmark"), "assigned_fraction_lineage_positive"].iloc[0]
        ),
        "unified_assigned_fraction": float(
            comparison.loc[comparison["model"].eq("unified_darlin_style_joint_clones"), "assigned_fraction_lineage_positive"].iloc[0]
        ),
        "round2_1_assigned_fraction_all_st": float(
            comparison.loc[comparison["model"].eq("round2_1_clone_membership"), "assigned_fraction_all_st"].iloc[0]
        )
        if comparison["model"].eq("round2_1_clone_membership").any()
        else 0.0,
    }
    dynamics_payload = {
        "design_only": True,
        "objects": ["C_cellbin_clone", "C_tile_clone", "C_niche_clone"],
        "direction_requirement": "time_or_perturbation_or_biological_prior",
        "l126_limitation": "serial_sections_are_not_timepoints",
    }
    final_payload = {
        "final_decision_label": final_decision_label,
        "selected_unified_clone_policy": f"{SELECTED_REFERENCE_BANK_POLICY} + {SELECTED_ALLELE_POLICY}",
        "joint_clone_count": int(len(valid_clones)),
        "assigned_cellbins": assigned_cellbins,
        "assigned_fraction_lineage_positive": float(assigned_fraction_lineage),
        "assigned_fraction_all_st": float(assigned_fraction_all_st),
        "largest_clone_cellbins": largest_clone_cellbins,
        "largest_clone_fraction": float(largest_clone_fraction),
        "de_novo_status_role": "qc_annotation_not_clone_class",
        "tile_clone_coverage_fraction": float(niche_payload["tile_coverage_fraction"]),
        "metaniche_clone_coverage_fraction": float(niche_payload["metaniche_coverage_fraction"]),
        "key_figures_path": str(args.report_root / "key_figure_candidates"),
    }
    payloads = {
        "policy": policy_payload,
        "clones": clone_payload,
        "qc": qc_payload,
        "matrix": matrix_payload,
        "niche": niche_payload,
        "comparison": comparison_payload,
        "figures": figure_payload,
        "dynamics": dynamics_payload,
        "final": final_payload,
    }
    write_text_reports(
        args.output_root,
        args.report_root,
        payloads,
        clone_qc,
        comparison,
        overwrite=args.overwrite,
    )
    atomic_write_json(args.report_root / "RUN_PAYLOAD.json", payloads, overwrite=args.overwrite)
    after_hashes = input_packet_hashes(args.input_packet_root)
    validation = validate_outputs(args.output_root, args.report_root, before_hashes, after_hashes)
    write_validation_report(args.report_root, validation, overwrite=args.overwrite)
    final_payload["validation_status"] = validation["validation_status"]
    atomic_write_json(args.report_root / "08_FINAL_DECISION.json", final_payload, overwrite=True)


if __name__ == "__main__":
    main()
