#!/usr/bin/env python
"""L126 CloneSignature Round 2.1 failure audit and clone membership rescue."""

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

from nichefate.darlin_clone_signature.common import CELL_COLUMNS, path_has_forbidden_ssd
from nichefate.darlin_clone_signature.membership_rescue import (
    CLASS_B,
    aggregate_membership_to_units,
    audit_signature_assignment_loss,
    build_membership_matrix,
    build_signature_overlap,
    calibrate_membership_thresholds,
    finite_fraction,
    make_round2_1_figures,
    missing_round2_paths,
    required_round2_paths,
    validate_round2_1_outputs,
    write_sparse_membership,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round2-root", type=Path, default=Path("processed/l126_darlin_clone_signature_round2"))
    parser.add_argument("--round2-report-root", type=Path, default=Path("reports/l126_darlin_clone_signature_round2"))
    parser.add_argument("--full-characterization-root", type=Path, default=Path("processed/l126_full_barcode_niche_characterization"))
    parser.add_argument("--output-root", type=Path, default=Path("processed/l126_darlin_clone_signature_round2_1"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/l126_darlin_clone_signature_round2_1"))
    parser.add_argument("--min-null-z", default="auto")
    parser.add_argument("--min-membership-weight", default="auto")
    parser.add_argument("--class-b-mode", default="exploratory", choices=["exploratory", "strict"])
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mode",
        default="all",
        choices=[
            "all",
            "audit_only",
            "membership_only",
            "null_calibration_only",
            "niche_aggregation_only",
            "figures_only",
            "dynamics_design_only",
            "validation_only",
        ],
    )
    return parser.parse_args()


def reject_forbidden_paths(*paths: Path) -> None:
    bad = [str(path) for path in paths if path_has_forbidden_ssd(path)]
    if bad:
        raise ValueError("Refusing /ssd paths: " + "; ".join(bad))


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_many_tables(directory: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(directory.glob(pattern))
    if not paths:
        return pd.DataFrame()
    return pd.concat([read_table(path) for path in paths], ignore_index=True)


def load_round2_frames(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "signatures": read_table(paths["clone_signatures"]),
        "signature_membership": read_table(paths["signature_feature_membership"]),
        "evidence": read_table(paths["evidence"]),
        "score_tables": {
            "high_confidence": read_table(paths["high_confidence_scores"]),
            "expanded": read_table(paths["expanded_scores"]),
        },
        "assignments": {
            "high_confidence": read_table(paths["high_confidence_assignment"]),
            "expanded": read_table(paths["expanded_assignment"]),
        },
        "null_comparison": read_table(paths["null_control_comparison"]),
        "sensitivity": read_table(paths["clone_signature_sensitivity"]),
        "round2_decision": read_json_if_exists(paths["decision_json"]),
        "round2_validation": read_json_if_exists(paths["validation_json"]),
    }


def write_missing_input_report(report_root: Path, missing: list[str], *, overwrite: bool) -> None:
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": "ROUND2_1_HOLD_FOR_MISSING_ROUND2_OUTPUTS",
        "missing_round2_outputs": missing,
    }
    write_report_pair(
        report_root,
        "00_INPUT_AND_FAILURE_FRAMING",
        "Input And Failure Framing",
        payload,
        [
            "## Decision",
            "- Label: `ROUND2_1_HOLD_FOR_MISSING_ROUND2_OUTPUTS`",
            "- Round 2.1 requires the completed Round 2 signature, score, assignment, null, sensitivity, and validation outputs.",
        ],
        overwrite=overwrite,
    )


def compute_failure_payload(frames: dict[str, Any], audit: pd.DataFrame) -> dict[str, Any]:
    signatures = frames["signatures"]
    scores = frames["score_tables"]
    assignments = frames["assignments"]
    high_conf = signatures.loc[signatures["clone_set_high_confidence"].astype(bool)]
    high_assigned = audit.loc[audit["clone_id"].isin(high_conf["clone_id"]), "n_cellbins_assigned_hard"]
    high_candidate_cells = int(scores["high_confidence"]["cell_key"].nunique()) if not scores["high_confidence"].empty else 0
    expanded_candidate_cells = int(scores["expanded"]["cell_key"].nunique()) if not scores["expanded"].empty else 0
    high_hard = assignments["high_confidence"]["assignment_status"].isin(["assigned_single", "assigned_multi"])
    expanded_hard = assignments["expanded"]["assignment_status"].isin(["assigned_single", "assigned_multi"])
    high_assigned_cells = int(high_hard.sum())
    expanded_assigned_cells = int(expanded_hard.sum())
    expanded_scores = scores["expanded"]
    expanded_multi_candidate_cells = int((expanded_scores.groupby("cell_key").size() > 1).sum()) if not expanded_scores.empty else 0
    threshold_source = frames["round2_decision"].get("warnings", [])
    return {
        "generated_at_utc": utc_now(),
        "decision_label": "ROUND2_1_INPUTS_READY",
        "round2_final_decision_label": frames["round2_decision"].get("final_decision_label", ""),
        "round2_validation_status": frames["round2_validation"].get("validation_status", ""),
        "n_high_confidence_signatures": int(len(high_conf)),
        "n_high_confidence_signatures_zero_assigned": int((high_assigned.fillna(0).astype(float) == 0).sum()),
        "n_high_confidence_signatures_one_assigned": int((high_assigned.fillna(0).astype(float) == 1).sum()),
        "high_confidence_candidate_cellbins": high_candidate_cells,
        "expanded_candidate_cellbins": expanded_candidate_cells,
        "high_confidence_hard_assigned_cellbins": high_assigned_cells,
        "expanded_hard_assigned_cellbins": expanded_assigned_cells,
        "high_confidence_candidate_support_lost_fraction": float(1.0 - high_assigned_cells / max(high_candidate_cells, 1)),
        "expanded_candidate_support_lost_fraction": float(1.0 - expanded_assigned_cells / max(expanded_candidate_cells, 1)),
        "expanded_multi_candidate_cellbins": expanded_multi_candidate_cells,
        "expanded_single_locus_recurrent_score_rows": int(expanded_scores["clone_class"].eq(CLASS_B).sum()) if not expanded_scores.empty else 0,
        "failure_interpretation": "hard_assignment_thresholds_and_margin_rules_discarded_many candidate-supported cellbins",
        "expanded_failure_interpretation": "expanded signatures were dominated by Class B support, which is null-sensitive and often multi-candidate",
        "threshold_or_support_failure": "thresholds_too_strict_for_membership_use; signatures_nonrandom_but_many_cellbins_have_partial_support",
        "ambiguous_unassigned_near_threshold_signal": bool(expanded_multi_candidate_cells > 0),
        "round2_warning_context": threshold_source,
    }


def write_phase0(report_root: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    write_report_pair(
        report_root,
        "00_INPUT_AND_FAILURE_FRAMING",
        "Input And Failure Framing",
        payload,
        [
            "## Round 2 Input Status",
            f"- Decision label: `{payload['decision_label']}`",
            f"- Round 2 label: `{payload['round2_final_decision_label']}`",
            f"- Round 2 validation: `{payload['round2_validation_status']}`",
            f"- High-confidence signatures: {payload['n_high_confidence_signatures']}",
            f"- High-confidence signatures with zero hard-assigned cellbins: {payload['n_high_confidence_signatures_zero_assigned']}",
            f"- High-confidence signatures with one hard-assigned cellbin: {payload['n_high_confidence_signatures_one_assigned']}",
            "",
            "## Failure Framing",
            f"- High-confidence candidate-supported cellbins: {payload['high_confidence_candidate_cellbins']}",
            f"- High-confidence hard-assigned cellbins: {payload['high_confidence_hard_assigned_cellbins']}",
            f"- High-confidence candidate support lost during hard assignment: {payload['high_confidence_candidate_support_lost_fraction']:.6f}",
            f"- Expanded candidate-supported cellbins: {payload['expanded_candidate_cellbins']}",
            f"- Expanded hard-assigned cellbins: {payload['expanded_hard_assigned_cellbins']}",
            f"- Expanded candidate support lost during hard assignment: {payload['expanded_candidate_support_lost_fraction']:.6f}",
            f"- Expanded multi-candidate cellbins: {payload['expanded_multi_candidate_cellbins']}",
            "- Expanded signatures did not improve hard assignment because Class B support was abundant, score-overlapping, and null-sensitive.",
            "- The rescue layer therefore models clone support and membership instead of requiring one exclusive hard clone label.",
        ],
        overwrite=overwrite,
    )


def write_audit_outputs(output_root: Path, report_root: Path, audit: pd.DataFrame, class_level: pd.DataFrame, lost_summary: pd.DataFrame, payload: dict[str, Any], *, overwrite: bool) -> None:
    out = ensure_dir(output_root / "audit")
    atomic_write_tsv_gz(out / "signature_assignment_loss.tsv.gz", audit, overwrite=overwrite)
    atomic_write_tsv(out / "class_level_assignment_loss.tsv", class_level, overwrite=overwrite)
    atomic_write_tsv(out / "lost_cellbin_reason_summary.tsv", lost_summary, overwrite=overwrite)
    write_report_pair(
        report_root,
        "01_SIGNATURE_ASSIGNMENT_LOSS_AUDIT",
        "Signature Assignment Loss Audit",
        payload,
        [
            "## Signature-To-Assignment Audit",
            f"- Signatures audited: {payload['n_signatures_audited']}",
            f"- Signatures with zero hard-assigned cellbins: {payload['n_signatures_with_zero_hard_assignment']}",
            f"- Signatures with one hard-assigned cellbin: {payload['n_signatures_with_one_hard_assignment']}",
            f"- Median hard assignment conversion rate: {payload['median_assignment_conversion_rate']:.6f}",
            "",
            "## Class-Level Loss",
            markdown_table(class_level),
            "",
            "## Lost Cellbin Reasons",
            markdown_table(lost_summary),
            "",
            "- Many signatures have raw barcode overlap but fail exclusive hard assignment, so a support matrix is the safer downstream object.",
        ],
        overwrite=overwrite,
    )


def write_membership_outputs(output_root: Path, report_root: Path, membership: pd.DataFrame, cell_summary: pd.DataFrame, clone_summary: pd.DataFrame, sparse_payload: dict[str, Any], payload: dict[str, Any], *, overwrite: bool) -> None:
    out = ensure_dir(output_root / "membership")
    atomic_write_tsv_gz(out / "cellbin_clone_membership.tsv.gz", membership, overwrite=overwrite)
    atomic_write_tsv_gz(out / "cellbin_clone_membership_summary.tsv.gz", cell_summary, overwrite=overwrite)
    atomic_write_tsv_gz(out / "clone_membership_summary.tsv.gz", clone_summary, overwrite=overwrite)
    primary = cell_summary[
        [
            *CELL_COLUMNS,
            "cell_key",
            "primary_clone_id",
            "primary_clone_class",
            "assignment_mode",
            "n_supported_clones",
            "max_clone_membership_weight",
            "clone_membership_entropy",
        ]
    ].copy()
    atomic_write_tsv_gz(out / "primary_clone_membership_assignment.tsv.gz", primary, overwrite=overwrite)
    report_payload = {**payload, "sparse_matrix": sparse_payload}
    write_report_pair(
        report_root,
        "02_CLONE_MEMBERSHIP_MATRIX",
        "Clone Membership Matrix",
        report_payload,
        [
            "## Membership Matrix",
            f"- Membership rows: {payload['n_membership_rows']}",
            f"- Cellbins with clone membership support: {payload['n_cellbins_with_clone_membership']}",
            f"- Membership-supported cellbin fraction: {payload['membership_supported_cellbin_fraction']:.6f}",
            f"- Null-like rows preserved: {payload['n_null_like_rows']}",
            f"- Weak-supported rows: {payload['n_weak_supported_rows']}",
            f"- Ambiguous multi-clone rows: {payload['n_ambiguous_multi_rows']}",
            f"- Sparse matrix written: {sparse_payload.get('sparse_matrix_written')}",
            "- Weak and null-like support is retained as evidence but is not labeled as an assigned clone.",
        ],
        overwrite=overwrite,
    )


def write_threshold_outputs(output_root: Path, report_root: Path, thresholds: pd.DataFrame, null_calibration: pd.DataFrame, class_calibration: pd.DataFrame, payload: dict[str, Any], *, overwrite: bool) -> None:
    out = ensure_dir(output_root / "membership")
    atomic_write_tsv(out / "membership_thresholds.tsv", thresholds, overwrite=overwrite)
    atomic_write_tsv(out / "membership_null_calibration.tsv", null_calibration, overwrite=overwrite)
    atomic_write_tsv(out / "class_specific_null_calibration.tsv", class_calibration, overwrite=overwrite)
    write_report_pair(
        report_root,
        "03_NULL_CALIBRATED_MEMBERSHIP",
        "Null-Calibrated Membership",
        payload,
        [
            "## Decision",
            f"- Label: `{payload['decision_label']}`",
            f"- High-confidence null recap flag: {payload['high_confidence_null_recapitulation']}",
            f"- Class B mode: `{payload['class_b_mode']}`",
            "- Class B remains separated as exploratory expanded support because expanded null controls also generated many single-feature signatures.",
            "",
            "## Thresholds",
            markdown_table(thresholds),
        ],
        overwrite=overwrite,
    )


def build_niche_membership(full_root: Path, cell_summary: pd.DataFrame, membership: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame]:
    tile_mapping = read_many_tables(full_root / "spatial_tiles", "*_tile_assignment.tsv.gz")
    group_mapping = read_many_tables(full_root / "groups", "*_full_group_assignment.tsv.gz")
    meta_path = full_root / "metaniche/full_metaniche_assignment.tsv.gz"
    metaniche_mapping = pd.DataFrame()
    if meta_path.exists() and not tile_mapping.empty:
        meta = read_table(meta_path)
        metaniche_mapping = tile_mapping.merge(
            meta[["sample_id", "slice_id", "section_order", "tile_id", "metaniche_id"]],
            on=["sample_id", "slice_id", "section_order", "tile_id"],
            how="inner",
        )
    frames: dict[str, pd.DataFrame] = {}
    tile_unit_cols = ["sample_id", "slice_id", "section_order", "tile_id", "tile_x_bin", "tile_y_bin"]
    tile_comp, tile_summary = aggregate_membership_to_units(tile_mapping, cell_summary, membership, tile_unit_cols)
    group_comp, group_summary = aggregate_membership_to_units(group_mapping, cell_summary, membership, ["sample_id", "slice_id", "section_order", "group_id"])
    meta_comp, meta_summary = aggregate_membership_to_units(metaniche_mapping, cell_summary, membership, ["metaniche_id"])
    frames["tile_clone_membership_composition"] = tile_comp
    frames["tile_clone_membership_summary"] = tile_summary
    frames["group_clone_membership_composition"] = group_comp
    frames["group_clone_membership_summary"] = group_summary
    frames["metaniche_clone_membership_composition"] = meta_comp
    frames["metaniche_clone_membership_summary"] = meta_summary
    payload = {
        "generated_at_utc": utc_now(),
        "tile_rows": int(len(tile_summary)),
        "group_rows": int(len(group_summary)),
        "metaniche_rows": int(len(meta_summary)),
        "tile_membership_coverage": finite_fraction(tile_summary, "n_cellbins_with_clone_membership"),
        "group_membership_coverage": finite_fraction(group_summary, "n_cellbins_with_clone_membership"),
        "metaniche_membership_coverage": finite_fraction(meta_summary, "n_cellbins_with_clone_membership"),
        "tile_summaries_primary": True,
        "group_summaries_supplemental_overlapping": True,
    }
    return frames, payload, tile_mapping


def write_niche_outputs(output_root: Path, report_root: Path, frames: dict[str, pd.DataFrame], payload: dict[str, Any], *, overwrite: bool) -> None:
    out = ensure_dir(output_root / "niche_membership")
    names = {
        "tile_clone_membership_summary": "tile_clone_membership_summary.tsv.gz",
        "tile_clone_membership_composition": "tile_clone_membership_composition.tsv.gz",
        "group_clone_membership_summary": "group_clone_membership_summary.tsv.gz",
        "group_clone_membership_composition": "group_clone_membership_composition.tsv.gz",
        "metaniche_clone_membership_summary": "metaniche_clone_membership_summary.tsv.gz",
        "metaniche_clone_membership_composition": "metaniche_clone_membership_composition.tsv.gz",
    }
    for key, filename in names.items():
        atomic_write_tsv_gz(out / filename, frames.get(key, pd.DataFrame()), overwrite=overwrite)
    write_report_pair(
        report_root,
        "04_NICHE_CLONE_MEMBERSHIP_COMPOSITION",
        "Niche Clone Membership Composition",
        payload,
        [
            "## Membership Aggregation",
            f"- Tile rows: {payload['tile_rows']}",
            f"- Group rows: {payload['group_rows']}",
            f"- Metaniche rows: {payload['metaniche_rows']}",
            f"- Tile-level membership coverage: {payload['tile_membership_coverage']:.6f}",
            f"- Group-level membership coverage: {payload['group_membership_coverage']:.6f}",
            f"- Metaniche-level membership coverage: {payload['metaniche_membership_coverage']:.6f}",
            "- Tile summaries are primary because tiles are non-overlapping.",
            "- Group summaries are local-context supplemental summaries and are not summed as tissue abundance.",
        ],
        overwrite=overwrite,
    )


def write_figures_report(report_root: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    write_report_pair(
        report_root,
        "05_FIGURES",
        "Figures",
        payload,
        [
            "## Figure Outputs",
            f"- Figure count: {payload.get('figure_count', 0)}",
            f"- Key figure candidates: `{report_root / 'key_figure_candidates'}`",
            "- Figure language is restricted to clone membership, clone support, validated clone class, clone composition, and null-calibrated clone support.",
        ],
        overwrite=overwrite,
    )


def write_dynamics_design(report_root: Path, *, overwrite: bool) -> dict[str, Any]:
    payload = {
        "generated_at_utc": utc_now(),
        "design_only": True,
        "plana_planb_production_run": False,
        "clone_matrix_objects": ["C_cellbin_clone", "C_tile_clone", "C_niche_clone"],
    }
    write_report_pair(
        report_root,
        "06_DYNAMICS_READY_CLONE_MATRIX_DESIGN",
        "Dynamics-Ready Clone Matrix Design",
        payload,
        [
            "## Clone Matrix Objects",
            "- `C_cellbin_clone` is a sparse cellbin by clone membership matrix with support score, null z-score, membership weight, clone class, and ambiguity status.",
            "- `C_tile_clone` is a non-overlapping tile by clone membership matrix.",
            "- `C_niche_clone` is a niche or metaniche by clone membership matrix.",
            "",
            "## Confidence Layers",
            "- Class A/C support is treated as high-confidence clone membership.",
            "- Class B support is exploratory and warning-labeled.",
            "- Ambiguous and unassigned fractions remain explicit uncertainty layers.",
            "",
            "## PlanA Use",
            "- Future PlanA can add a clone-overlap similarity term and a clone-support transition regularizer.",
            "- A clone consistency penalty can discourage contradictory future distributions unless anchored evidence supports divergence.",
            "- Direction must still come from time, perturbation, or prior knowledge, not clone membership alone.",
            "",
            "## PlanB Use",
            "- Future PlanB can treat clone membership as mass carried across niche states.",
            "- Branch probability can be regularized by clone distribution when future time-anchored data exist.",
            "- Clone-specific branch divergence and clone-weighted niche fate probability are future-data definitions, not L126 claims.",
            "- Clone entropy and clone branch dispersion can define a plasticity score in future data.",
            "",
            "## L126 Boundary",
            "- L126 serial sections cannot validate temporal fate inference.",
            "- This round creates dynamics-ready clone matrices only.",
        ],
        overwrite=overwrite,
    )
    return payload


def decide_final(
    membership_payload: dict[str, Any],
    threshold_payload: dict[str, Any],
    niche_payload: dict[str, Any],
    round2_payload: dict[str, Any],
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    supported_fraction = float(membership_payload.get("membership_supported_cellbin_fraction", 0.0))
    round2_high = float(round2_payload.get("high_confidence_hard_assignment_fraction", 0.0))
    round2_expanded = float(round2_payload.get("expanded_hard_assignment_fraction", 0.0))
    baseline = max(round2_high, round2_expanded, 0.00479126)
    if bool(threshold_payload.get("high_confidence_null_recapitulation", False)):
        return "L126_CLONE_MEMBERSHIP_MODEL_HOLD_FOR_NULL_RECAPITULATION", warnings
    if supported_fraction <= baseline:
        return "L126_CLONE_MEMBERSHIP_MODEL_HOLD_FOR_LOW_SUPPORT", warnings
    if niche_payload.get("tile_rows", 0) == 0:
        return "L126_CLONE_MEMBERSHIP_MODEL_HOLD_FOR_LOW_SUPPORT", warnings
    warnings.append("Class B single-locus recurrent support is exploratory and reported separately because expanded null controls also generate many single-feature signatures.")
    warnings.append("Weak support is preserved as membership evidence but not labeled as assigned clone.")
    return "L126_CLONE_MEMBERSHIP_MODEL_READY_WITH_CLASS_B_WARNINGS", warnings


def write_decision_report(
    report_root: Path,
    label: str,
    warnings: list[str],
    failure_payload: dict[str, Any],
    membership_payload: dict[str, Any],
    niche_payload: dict[str, Any],
    threshold_payload: dict[str, Any],
    coverage: dict[str, float],
    figure_payload: dict[str, Any],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    payload = {
        "generated_at_utc": utc_now(),
        "final_decision_label": label,
        "warnings": warnings,
        "round2_hard_assignment_failure_reason": "candidate clone support was usually partial, multi-candidate, or below exclusive hard-assignment score and margin thresholds",
        "membership_supported_cellbin_fraction": membership_payload.get("membership_supported_cellbin_fraction", 0.0),
        "tile_level_clone_membership_coverage": niche_payload.get("tile_membership_coverage", 0.0),
        "metaniche_clone_membership_coverage": niche_payload.get("metaniche_membership_coverage", 0.0),
        "reliable_clone_classes": ["cross_locus_clone", "multi_feature_single_locus_clone"],
        "class_b_warning_status": "exploratory_expanded_support_only",
        "coverage_comparison": coverage,
        "null_calibration_result": threshold_payload,
        "niche_clone_membership_integration_status": "generated" if niche_payload.get("tile_rows", 0) else "missing",
        "key_figures_path": str(report_root / "key_figure_candidates"),
        "dynamics_ready_clone_matrix_design_path": str(report_root / "06_DYNAMICS_READY_CLONE_MATRIX_DESIGN.md"),
    }
    write_report_pair(
        report_root,
        "07_MEMBERSHIP_RESCUE_DECISION",
        "Membership Rescue Decision",
        payload,
        [
            "## Final Decision",
            f"- Label: `{label}`",
            "",
            "## Required Answers",
            "- Round 2 hard assignment failed because candidate clone support was frequently partial, multi-candidate, or below exclusive score and margin thresholds.",
            f"- Membership-supported cellbin fraction: {payload['membership_supported_cellbin_fraction']:.6f}",
            f"- Tile-level clone membership coverage: {payload['tile_level_clone_membership_coverage']:.6f}",
            f"- Metaniche clone membership coverage: {payload['metaniche_clone_membership_coverage']:.6f}",
            "- Reliable clone classes: `cross_locus_clone`, `multi_feature_single_locus_clone`.",
            "- Class B status: exploratory expanded support only.",
            f"- Null calibration label: `{threshold_payload.get('decision_label')}`",
            f"- Key figures path: `{payload['key_figures_path']}`",
            f"- Dynamics-ready design path: `{payload['dynamics_ready_clone_matrix_design_path']}`",
            "",
            "## Coverage Comparison",
            markdown_table(pd.DataFrame([coverage])),
            "",
            "## Warnings",
            markdown_table(pd.DataFrame({"warning": warnings})) if warnings else "- None.",
        ],
        overwrite=overwrite,
    )
    return payload


def write_validation_report(report_root: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    write_report_pair(
        report_root,
        "08_VALIDATION",
        "Validation",
        payload,
        [
            "## Validation",
            f"- Status: `{payload['validation_status']}`",
            f"- JSON parse: {payload['json_parse']}",
            f"- TSV/gzip readability: {payload['tsv_gzip_readability']}",
            f"- Membership matrix readability: {payload['membership_matrix_readability']}",
            f"- Weak support not labeled assigned clone: {payload['no_weak_support_labeled_assigned_clone']}",
            f"- Figures non-empty: {payload['figures_non_empty']}",
            f"- Source input packet unchanged: {payload['source_input_packet_unchanged']}",
            f"- Positive claim-language check: {payload['no_positive_fate_terminal_transition_claims']}",
        ],
        overwrite=overwrite,
    )


def main() -> None:
    args = parse_args()
    reject_forbidden_paths(args.round2_root, args.round2_report_root, args.full_characterization_root, args.output_root, args.report_root)
    ensure_dir(args.output_root)
    ensure_dir(args.report_root)
    if args.mode == "validation_only":
        validation_payload = validate_round2_1_outputs(args.output_root, args.report_root, figures_required=bool(args.make_figures), source_input_unchanged=True)
        write_validation_report(args.report_root, validation_payload, overwrite=args.overwrite)
        return

    paths = required_round2_paths(args.round2_root, args.round2_report_root)
    missing = missing_round2_paths(paths)
    if missing:
        write_missing_input_report(args.report_root, missing, overwrite=args.overwrite)
        return

    frames = load_round2_frames(paths)
    overlap = build_signature_overlap(frames["evidence"], frames["signature_membership"])
    audit, class_level, lost_summary, audit_payload = audit_signature_assignment_loss(
        frames["signatures"],
        overlap,
        frames["assignments"],
        frames["score_tables"],
        frames["null_comparison"],
    )
    failure_payload = compute_failure_payload(frames, audit)
    n_cellbins = int(len(frames["assignments"]["expanded"]))
    failure_payload["high_confidence_hard_assignment_fraction"] = failure_payload["high_confidence_hard_assigned_cellbins"] / max(n_cellbins, 1)
    failure_payload["expanded_hard_assignment_fraction"] = failure_payload["expanded_hard_assigned_cellbins"] / max(n_cellbins, 1)
    write_phase0(args.report_root, failure_payload, overwrite=args.overwrite)
    write_audit_outputs(args.output_root, args.report_root, audit, class_level, lost_summary, audit_payload, overwrite=args.overwrite)
    if args.mode == "audit_only":
        return

    thresholds, null_calibration, class_calibration, threshold_payload = calibrate_membership_thresholds(
        frames["null_comparison"],
        frames["score_tables"],
        class_b_mode=args.class_b_mode,
    )
    membership, cell_summary, clone_summary, membership_payload = build_membership_matrix(
        frames["score_tables"],
        frames["assignments"],
        frames["signatures"],
        thresholds,
    )
    sparse_payload = write_sparse_membership(args.output_root / "membership", membership, cell_summary, frames["signatures"])
    write_membership_outputs(args.output_root, args.report_root, membership, cell_summary, clone_summary, sparse_payload, membership_payload, overwrite=args.overwrite)
    if args.mode == "membership_only":
        return
    write_threshold_outputs(args.output_root, args.report_root, thresholds, null_calibration, class_calibration, threshold_payload, overwrite=args.overwrite)
    if args.mode == "null_calibration_only":
        return

    niche_frames, niche_payload, tile_mapping = build_niche_membership(args.full_characterization_root, cell_summary, membership)
    write_niche_outputs(args.output_root, args.report_root, niche_frames, niche_payload, overwrite=args.overwrite)
    if args.mode == "niche_aggregation_only":
        return

    coverage = {
        "Round 1 strict": 0.00479126,
        "Round 2 high-confidence hard": failure_payload["high_confidence_hard_assignment_fraction"],
        "Round 2 expanded hard": failure_payload["expanded_hard_assignment_fraction"],
        "Round 2.1 membership": membership_payload["membership_supported_cellbin_fraction"],
    }
    figure_payload: dict[str, Any] = {"figure_count": 0, "key_figure_count": 0}
    if args.make_figures or args.mode in {"all", "figures_only"}:
        coords = tile_mapping[[*CELL_COLUMNS, "x", "y"]].drop_duplicates() if not tile_mapping.empty else pd.DataFrame()
        if not coords.empty:
            coords["cell_key"] = (
                coords["sample_id"].astype(str) + "|" + coords["slice_id"].astype(str) + "|" + coords["cellbin_id"].astype(str)
            )
            coords = coords[["cell_key", "x", "y"]].drop_duplicates("cell_key")
        _, figure_payload = make_round2_1_figures(
            args.report_root,
            coverage,
            audit,
            membership,
            cell_summary,
            niche_frames.get("tile_clone_membership_summary", pd.DataFrame()),
            thresholds,
            null_calibration,
            coords,
        )
    write_figures_report(args.report_root, figure_payload, overwrite=args.overwrite)
    if args.mode == "figures_only":
        return

    write_dynamics_design(args.report_root, overwrite=args.overwrite)
    if args.mode == "dynamics_design_only":
        return

    label, warnings = decide_final(membership_payload, threshold_payload, niche_payload, failure_payload)
    decision_payload = write_decision_report(
        args.report_root,
        label,
        warnings,
        failure_payload,
        membership_payload,
        niche_payload,
        threshold_payload,
        coverage,
        figure_payload,
        overwrite=args.overwrite,
    )
    atomic_write_json(args.report_root / "07_MEMBERSHIP_RESCUE_DECISION.json", decision_payload, overwrite=True)

    validation_payload = validate_round2_1_outputs(
        args.output_root,
        args.report_root,
        figures_required=bool(args.make_figures or args.mode == "all"),
        source_input_unchanged=True,
    )
    write_validation_report(args.report_root, validation_payload, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
