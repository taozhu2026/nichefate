#!/usr/bin/env python
"""L126 DARLIN CloneSignature Round 2 from complete processed lineage evidence."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd

from nichefate.barcode_adapter.qc import compare_file_snapshots, snapshot_files
from nichefate.barcode_adapter.loaders import load_cellbin_lineage_evidence, load_feature_allele_annotation
from nichefate.darlin_clone_signature import (
    CloneSignatureParams,
    aggregate_clone_membership,
    assign_cellbins_to_clones,
    build_canonical_evidence,
    build_clone_signatures,
    build_feature_compatibility_graph,
    candidate_clone_scores,
)
from nichefate.darlin_clone_signature.assignment import calibrate_assignment_thresholds
from nichefate.darlin_clone_signature.common import CELL_COLUMNS, path_has_forbidden_ssd
from nichefate.darlin_clone_signature.controls import run_null_controls, run_sensitivity_grid
from nichefate.darlin_clone_signature.figures import make_round2_figures
from nichefate.darlin_clone_signature.reporting import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_tsv,
    atomic_write_tsv_gz,
    ensure_dir,
    markdown_table,
    read_table,
    utc_now,
    write_report_pair,
)
from nichefate.darlin_clone_signature.validation import validate_round2_outputs


DEFAULT_INPUT_PACKET_ROOT = Path("/home/zhutao/scratch/nichefate/spatio_darlin_public_brain_hpc/input_packet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-packet-root", type=Path, default=DEFAULT_INPUT_PACKET_ROOT)
    parser.add_argument("--barcode-root", type=Path, default=Path("processed/barcode_adapter_l126_round1"))
    parser.add_argument("--full-characterization-root", type=Path, default=Path("processed/l126_full_barcode_niche_characterization"))
    parser.add_argument("--round1-clone-root", type=Path, default=Path("processed/l126_darlin_clone_integration_round1"))
    parser.add_argument("--output-root", type=Path, default=Path("processed/l126_darlin_clone_signature_round2"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/l126_darlin_clone_signature_round2"))
    parser.add_argument("--rare-threshold", type=float, default=0.001)
    parser.add_argument("--low-frequency-threshold", type=float, default=0.005)
    parser.add_argument("--min-single-feature-cellbins", type=int, default=2)
    parser.add_argument("--min-feature-cooccurrence-cellbins", type=int, default=2)
    parser.add_argument("--min-assignment-score", default="auto")
    parser.add_argument("--min-score-margin", default="auto")
    parser.add_argument("--run-null-controls", action="store_true")
    parser.add_argument("--run-sensitivity", action="store_true")
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mode",
        default="all",
        choices=[
            "all",
            "evidence_only",
            "contract_only",
            "compatibility_graph_only",
            "signatures_only",
            "assignment_only",
            "null_controls_only",
            "niche_aggregation_only",
            "figures_only",
            "dynamics_design_only",
            "validation_only",
        ],
    )
    return parser.parse_args()


def lineage_evidence_path(input_packet_root: Path) -> Path:
    return input_packet_root / "processed/lineage_evidence/cellbin_lineage_evidence.tsv.gz"


def allele_annotation_path(input_packet_root: Path) -> Path:
    return input_packet_root / "processed/lineage_evidence/feature_allele_annotation_long.tsv.gz"


def reject_forbidden_paths(*paths: Path) -> None:
    bad = [str(path) for path in paths if path_has_forbidden_ssd(path)]
    if bad:
        raise ValueError("Refusing /ssd paths: " + "; ".join(bad))


def load_full_cellbins(full_characterization_root: Path, barcode_root: Path, lineage: pd.DataFrame) -> pd.DataFrame:
    full_path = full_characterization_root / "cellbin/full_cellbin_lineage_summary.tsv.gz"
    barcode_path = barcode_root / "cellbin_lineage_summary.tsv.gz"
    if full_path.exists():
        full = read_table(full_path)
    elif barcode_path.exists():
        full = read_table(barcode_path)
    else:
        cols = [*CELL_COLUMNS, "x", "y"]
        full = lineage[cols].drop_duplicates(CELL_COLUMNS).copy()
    for col in CELL_COLUMNS:
        if col not in full:
            raise ValueError(f"full cellbin summary missing {col}")
    if "x" not in full:
        full["x"] = np.nan
    if "y" not in full:
        full["y"] = np.nan
    return full.drop_duplicates(CELL_COLUMNS).reset_index(drop=True)


def report_incomplete_inputs(args: argparse.Namespace, missing: list[str]) -> None:
    ensure_dir(args.output_root)
    ensure_dir(args.report_root)
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": "HOLD_FOR_INCOMPLETE_EVIDENCE",
        "missing_complete_evidence_inputs": missing,
    }
    write_report_pair(
        args.report_root,
        "10_CLONE_SIGNATURE_DECISION",
        "L126 CloneSignature Decision",
        payload,
        [
            "## Final Decision",
            "- Label: `HOLD_FOR_INCOMPLETE_EVIDENCE`",
            "- Complete primary lineage evidence is mandatory for CloneSignature modeling.",
        ],
        overwrite=args.overwrite,
    )


def write_problem_framing(report_root: Path, evidence_ready: bool, *, overwrite: bool) -> dict[str, Any]:
    label = "CLONE_SIGNATURE_DESIGN_SCOPE_READY" if evidence_ready else "HOLD_FOR_MISSING_ROUND1_INPUTS"
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": label,
        "lineage_state_variable": True,
        "l126_scope": "clone_niche_characterization_not_temporal_inference",
    }
    write_report_pair(
        report_root,
        "00_PROBLEM_FRAMING",
        "Problem Framing",
        payload,
        [
            "## Design Boundary",
            "- Feature-level barcode evidence describes individual CA/TA/RA observations but does not by itself define lineage state variables.",
            "- Strict connected components were too conservative because sparse multi-locus evidence can fragment related barcode support.",
            "- Raw connected components can overmerge when a high-complexity bridge cellbin links otherwise separate feature sets.",
            "- A validated clone under empirical DARLIN-style contract is treated as a lineage state variable, not an ordinary feature.",
            "- L126 supports clone/niche characterization and spatial clone composition, but sections are not timepoints.",
            "- Future PlanA/PlanB dynamics can use clone overlap and clone confidence after time or perturbation direction exists.",
        ],
        overwrite=overwrite,
    )
    return payload


def write_contract(config_path: Path, report_root: Path, params: CloneSignatureParams, *, overwrite: bool) -> dict[str, Any]:
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": "CLONE_SIGNATURE_CONTRACT_READY",
        "clone_call_name": "validated clones under empirical DARLIN-style contract",
        "assay_scoped_feature_id": "assay + '::' + feature_id",
        "valid_clone_classes": {
            "cross_locus_clone": {
                "criteria": [
                    "valid rare/low-frequency features from at least two loci among CA/TA/RA",
                    "supported by at least two cellbins",
                    "co-occurrence support exceeds empirical null or repeated evidence threshold",
                    "not dependent on one bridge cellbin",
                    "not dominated by a common feature",
                ],
                "clone_set": "high_confidence_clone_set",
            },
            "single_locus_recurrent_clone": {
                "criteria": [
                    "one rare/low-frequency assay-scoped feature",
                    f"observed in at least {params.min_single_feature_cellbins} cellbins",
                    "not common-filtered",
                    "not bridge-driven",
                    "survives sensitivity review",
                ],
                "clone_set": "expanded_clone_set",
                "evidence_level": "single_locus",
            },
            "multi_feature_single_locus_clone": {
                "criteria": [
                    "two or more rare/low-frequency same-locus features",
                    "repeated compatible co-occurrence in at least two cellbins",
                    "not bridge-driven",
                ],
                "clone_set": "high_confidence_clone_set",
                "evidence_level": "single_locus_multifeature",
            },
        },
        "non_clone_classes": ["ambiguous", "filtered", "unassigned"],
        "default_thresholds": {
            "rare_threshold": params.rare_threshold,
            "low_frequency_threshold": params.low_frequency_threshold,
            "min_single_feature_cellbins": params.min_single_feature_cellbins,
            "min_feature_cooccurrence_cellbins": params.min_feature_cooccurrence_cellbins,
            "max_bridge_dependency_score": params.max_bridge_dependency_score,
        },
        "official_darlin_clone_claim": False,
    }
    atomic_write_json(config_path, payload, overwrite=overwrite)
    write_report_pair(
        report_root,
        "02_CLONE_SIGNATURE_CONTRACT",
        "CloneSignature Contract",
        payload,
        [
            "## Contract",
            "- Class A/B/C objects are validated clone classes under the declared empirical contract.",
            "- Non-passing objects are ambiguous, filtered, or unassigned.",
            "- Input `clone_id` is not used as the final integrated clone identifier.",
            "- Allele annotation is metadata and does not inflate counts.",
        ],
        overwrite=overwrite,
    )
    return payload


def write_evidence_outputs(output_root: Path, report_root: Path, evidence: pd.DataFrame, feature_ref: pd.DataFrame, complexity: pd.DataFrame, payload: dict[str, Any], *, overwrite: bool) -> None:
    out = ensure_dir(output_root / "evidence")
    atomic_write_tsv_gz(out / "cellbin_feature_evidence.tsv.gz", evidence, overwrite=overwrite)
    atomic_write_tsv_gz(out / "feature_frequency_reference.tsv.gz", feature_ref, overwrite=overwrite)
    atomic_write_tsv_gz(out / "cellbin_barcode_complexity.tsv.gz", complexity, overwrite=overwrite)
    write_report_pair(
        report_root,
        "01_CANONICAL_EVIDENCE_MATRIX",
        "Canonical Evidence Matrix",
        payload,
        [
            "## Complete Evidence",
            "- Primary clone-calling evidence came from complete `cellbin_lineage_evidence.tsv.gz`.",
            "- `full_cellbin_top_features.tsv.gz` is QC/summary only and is not used for CloneSignature construction.",
            f"- Complete evidence rows: {payload['complete_primary_evidence_rows']}",
            f"- Assay-scoped features: {payload['n_assay_scoped_features']}",
            f"- Valid signature features: {payload['n_valid_signature_features']}",
            f"- Estimated sparse pair events: {payload['estimated_pair_event_count']}",
            "",
            "## Valid Features Per Cellbin Quantiles",
            markdown_table(pd.DataFrame([payload["valid_features_per_cellbin_quantiles"]])),
        ],
        overwrite=overwrite,
    )


def write_graph_outputs(output_root: Path, report_root: Path, edges: pd.DataFrame, summary: pd.DataFrame, components: pd.DataFrame, payload: dict[str, Any], *, overwrite: bool) -> None:
    out = ensure_dir(output_root / "signatures")
    atomic_write_tsv_gz(out / "feature_compatibility_edges.tsv.gz", edges, overwrite=overwrite)
    atomic_write_tsv(out / "feature_compatibility_graph_summary.tsv", summary, overwrite=overwrite)
    atomic_write_tsv_gz(out / "candidate_signature_components.tsv.gz", components, overwrite=overwrite)
    write_report_pair(
        report_root,
        "03_FEATURE_COMPATIBILITY_GRAPH",
        "Feature Compatibility Graph",
        payload,
        [
            "## Sparse Graph Construction",
            "- Edges were generated by inverted-index aggregation over cellbins, not all feature-feature pairs.",
            f"- Bridge filter mode: `{payload['bridge_filter_mode']}`",
            f"- Bridge/high-complexity cellbins filtered: {payload['n_bridge_cellbins_filtered']}",
            f"- Pair events after filtering: {payload['estimated_pair_event_count_after_filter']}",
            f"- Compatible edges: {payload['n_compatible_edges']}",
            f"- Resource warning: {payload['resource_warning']}",
        ],
        overwrite=overwrite,
    )


def annotate_spatial_qc(signatures: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    if signatures.empty:
        return signatures
    rows = []
    for row in signatures.to_dict(orient="records"):
        features = [item for item in str(row["feature_list"]).split(";") if item]
        support = evidence.loc[evidence["assay_scoped_feature_id"].isin(features)].drop_duplicates("cell_key")
        if support.empty:
            rows.append({"clone_id": row["clone_id"], "spatial_extent_summary": "", "section_dominance_fraction": 0.0, "spatial_qc_flags": "no_support_cells"})
            continue
        x_span = float(support["x"].max() - support["x"].min()) if "x" in support else 0.0
        y_span = float(support["y"].max() - support["y"].min()) if "y" in support else 0.0
        section_counts = support["section_order"].value_counts(normalize=True)
        section_dom = float(section_counts.max()) if len(section_counts) else 0.0
        flags = []
        if section_dom >= 0.9 and support["section_order"].nunique() > 1:
            flags.append("section_dominated")
        if x_span > 12000 or y_span > 8000:
            flags.append("extremely_broad_spatial_extent")
        if not flags:
            flags.append("spatial_qc_not_used_as_clone_gate")
        rows.append(
            {
                "clone_id": row["clone_id"],
                "spatial_extent_summary": f"x_span={x_span:.3f};y_span={y_span:.3f}",
                "section_dominance_fraction": section_dom,
                "spatial_qc_flags": ";".join(flags),
            }
        )
    return signatures.merge(pd.DataFrame(rows), on="clone_id", how="left")


def write_signature_outputs(output_root: Path, report_root: Path, signatures: pd.DataFrame, membership: pd.DataFrame, filtered: pd.DataFrame, payload: dict[str, Any], *, overwrite: bool) -> None:
    out = ensure_dir(output_root / "signatures")
    atomic_write_tsv_gz(out / "clone_signatures.tsv.gz", signatures, overwrite=overwrite)
    atomic_write_tsv_gz(out / "clone_signature_feature_membership.tsv.gz", membership, overwrite=overwrite)
    atomic_write_tsv_gz(out / "filtered_candidate_signatures.tsv.gz", filtered, overwrite=overwrite)
    class_counts = signatures["clone_class"].value_counts().reset_index() if not signatures.empty else pd.DataFrame(columns=["clone_class", "count"])
    class_counts.columns = ["clone_class", "n_clones"]
    write_report_pair(
        report_root,
        "04_CLONE_SIGNATURES",
        "Clone Signatures",
        payload,
        [
            "## Clone Sets",
            f"- High-confidence clones: {payload['n_high_confidence_clones']}",
            f"- Expanded clones: {payload['n_expanded_clones']}",
            "- High-confidence clone set contains cross-locus and stable multi-feature single-locus signatures.",
            "- Expanded clone set additionally contains eligible single-locus recurrent signatures.",
            "",
            markdown_table(class_counts),
        ],
        overwrite=overwrite,
    )


def read_round1_summary(round1_root: Path, total_cellbins: int) -> dict[str, Any]:
    assignment_path = round1_root / "clones/cellbin_clone_assignment.tsv.gz"
    summary_path = round1_root / "clones/clone_summary.tsv.gz"
    out = {"round1_clone_count": 0, "round1_assigned_cellbin_fraction": 0.0, "round1_assigned_cellbins": 0}
    if assignment_path.exists():
        a = read_table(assignment_path)
        assigned = a["clone_status"].astype(str).eq("clone") if "clone_status" in a else a["clone_id"].notna()
        out["round1_assigned_cellbins"] = int(assigned.sum())
        out["round1_assigned_cellbin_fraction"] = float(assigned.sum() / max(total_cellbins, 1))
    if summary_path.exists():
        s = read_table(summary_path)
        out["round1_clone_count"] = int(len(s))
    return out


def write_assignment_outputs(output_root: Path, report_root: Path, assignments: dict[str, pd.DataFrame], topks: dict[str, pd.DataFrame], memberships: dict[str, pd.DataFrame], matrices: dict[str, pd.DataFrame], summaries: dict[str, dict[str, Any]], round1: dict[str, Any], *, overwrite: bool) -> None:
    out = ensure_dir(output_root / "assignments")
    for clone_set, frame in assignments.items():
        prefix = "high_confidence_" if clone_set == "high_confidence" else "expanded_"
        atomic_write_tsv_gz(out / f"{prefix}cellbin_clone_assignment_v2.tsv.gz", frame, overwrite=overwrite)
        atomic_write_tsv_gz(out / f"{prefix}cellbin_clone_score_topk.tsv.gz", topks[clone_set], overwrite=overwrite)
        atomic_write_tsv_gz(out / f"{prefix}cellbin_clone_membership_v2.tsv.gz", memberships[clone_set], overwrite=overwrite)
        atomic_write_tsv_gz(out / f"{prefix}clone_by_cellbin_matrix.tsv.gz", matrices[clone_set], overwrite=overwrite)
    shutil.copy2(out / "expanded_cellbin_clone_assignment_v2.tsv.gz", out / "cellbin_clone_assignment_v2.tsv.gz")
    shutil.copy2(out / "expanded_cellbin_clone_score_topk.tsv.gz", out / "cellbin_clone_score_topk.tsv.gz")
    shutil.copy2(out / "expanded_cellbin_clone_membership_v2.tsv.gz", out / "cellbin_clone_membership_v2.tsv.gz")
    shutil.copy2(out / "expanded_clone_by_cellbin_matrix.tsv.gz", out / "clone_by_cellbin_matrix.tsv.gz")
    summary_frame = pd.DataFrame(list(summaries.values()))
    for key, value in round1.items():
        summary_frame[key] = value
    atomic_write_tsv(out / "assignment_summary.tsv", summary_frame, overwrite=overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "assignment_summaries": summaries,
        "round1_comparison": round1,
    }
    write_report_pair(
        report_root,
        "05_CELLBIN_CLONE_ASSIGNMENT",
        "Cellbin Clone Assignment",
        payload,
        [
            "## Assignment Summary",
            markdown_table(summary_frame),
            "",
            "## Round 1 Comparison",
            f"- Round 1 strict clone count: {round1['round1_clone_count']}",
            f"- Round 1 assigned cellbin fraction: {round1['round1_assigned_cellbin_fraction']:.6f}",
        ],
        overwrite=overwrite,
    )


def write_null_sensitivity_outputs(output_root: Path, report_root: Path, null_frame: pd.DataFrame, sensitivity: pd.DataFrame, bridge_sensitivity: pd.DataFrame, thresholds: dict[str, dict[str, Any]], payload: dict[str, Any], *, overwrite: bool) -> None:
    out = ensure_dir(output_root / "sensitivity")
    atomic_write_tsv(out / "clone_signature_sensitivity.tsv", sensitivity, overwrite=overwrite)
    atomic_write_tsv(out / "null_control_comparison.tsv", null_frame, overwrite=overwrite)
    atomic_write_tsv(out / "bridge_filter_sensitivity.tsv", bridge_sensitivity, overwrite=overwrite)
    threshold_frame = pd.DataFrame(list(thresholds.values()))
    atomic_write_tsv(out / "assignment_threshold_calibration.tsv", threshold_frame, overwrite=overwrite)
    report_payload = {**payload, "thresholds": thresholds}
    write_report_pair(
        report_root,
        "06_NULL_AND_SENSITIVITY",
        "Null Controls And Sensitivity",
        report_payload,
        [
            "## Null-Calibrated Thresholds",
            markdown_table(threshold_frame),
            "",
            "## Null Controls",
            markdown_table(null_frame),
            "",
            "## Sensitivity",
            f"- Sensitivity rows: {len(sensitivity)}",
            f"- Decision label: `{payload['decision_label']}`",
        ],
        overwrite=overwrite,
    )


def read_many_tables(directory: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(directory.glob(pattern))
    if not paths:
        return pd.DataFrame()
    return pd.concat([read_table(path) for path in paths], ignore_index=True)


def build_niche_outputs(full_root: Path, assignments: dict[str, pd.DataFrame], memberships: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    tile_mapping = read_many_tables(full_root / "spatial_tiles", "*_tile_assignment.tsv.gz")
    group_mapping = read_many_tables(full_root / "groups", "*_full_group_assignment.tsv.gz")
    meta_path = full_root / "metaniche/full_metaniche_assignment.tsv.gz"
    metaniche_mapping = pd.DataFrame()
    if meta_path.exists() and not tile_mapping.empty:
        meta = read_table(meta_path)
        metaniche_mapping = tile_mapping.merge(meta[["sample_id", "slice_id", "section_order", "tile_id", "metaniche_id"]], on=["sample_id", "slice_id", "section_order", "tile_id"], how="inner")
    frames: dict[str, pd.DataFrame] = {}
    for clone_set in ["high_confidence", "expanded"]:
        tile_comp, tile_summary = aggregate_clone_membership(tile_mapping, assignments[clone_set], memberships[clone_set], ["sample_id", "slice_id", "section_order", "tile_id"])
        group_comp, group_summary = aggregate_clone_membership(group_mapping, assignments[clone_set], memberships[clone_set], ["sample_id", "slice_id", "section_order", "group_id"])
        meta_comp, meta_summary = aggregate_clone_membership(metaniche_mapping, assignments[clone_set], memberships[clone_set], ["metaniche_id"])
        for frame in [tile_comp, tile_summary, group_comp, group_summary, meta_comp, meta_summary]:
            if not frame.empty and "clone_set" not in frame.columns:
                frame.insert(0, "clone_set", clone_set)
        for name, frame in [
            ("tile_clone_composition_v2", tile_comp),
            ("tile_clone_summary_v2", tile_summary),
            ("group_clone_composition_v2", group_comp),
            ("group_clone_summary_v2", group_summary),
            ("metaniche_clone_composition_v2", meta_comp),
            ("metaniche_clone_summary_v2", meta_summary),
        ]:
            frames[name] = pd.concat([frames.get(name, pd.DataFrame()), frame], ignore_index=True)
    payload = {
        "generated_at_utc": utc_now(),
        "tile_rows": int(len(frames.get("tile_clone_summary_v2", pd.DataFrame()))),
        "group_rows": int(len(frames.get("group_clone_summary_v2", pd.DataFrame()))),
        "metaniche_rows": int(len(frames.get("metaniche_clone_summary_v2", pd.DataFrame()))),
        "tile_summaries_primary": True,
        "group_summaries_supplemental_overlapping": True,
    }
    return frames, payload


def write_niche_outputs(output_root: Path, report_root: Path, frames: dict[str, pd.DataFrame], payload: dict[str, Any], *, overwrite: bool) -> None:
    out = ensure_dir(output_root / "niche_clone")
    for name, frame in frames.items():
        atomic_write_tsv_gz(out / f"{name}.tsv.gz", frame, overwrite=overwrite)
    write_report_pair(
        report_root,
        "07_NICHE_CLONE_COMPOSITION",
        "Niche Clone Composition",
        payload,
        [
            "## Spatial Clone Composition",
            f"- Tile summary rows: {payload['tile_rows']}",
            f"- Group summary rows: {payload['group_rows']}",
            f"- Metaniche summary rows: {payload['metaniche_rows']}",
            "- Tile summaries are primary because tiles are non-overlapping.",
            "- Group summaries are local-context supplemental summaries and are not summed as tissue abundance.",
        ],
        overwrite=overwrite,
    )


def write_figures_report(report_root: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    write_report_pair(
        report_root,
        "08_FIGURES",
        "Figures",
        payload,
        [
            "## Figure Outputs",
            f"- Figure count: {payload.get('figure_count', 0)}",
            f"- Key figure candidates: `{report_root / 'key_figure_candidates'}`",
            "- Figure language is restricted to clone assignment, clone class, spatial clone composition, clone diversity, and QC comparisons.",
        ],
        overwrite=overwrite,
    )


def write_dynamics_design(report_root: Path, *, overwrite: bool) -> dict[str, Any]:
    payload = {
        "generated_at_utc": utc_now(),
        "design_only": True,
        "plana_planb_production_run": False,
        "clone_matrix": ["C[cellbin, clone]", "C[niche, clone]"],
    }
    write_report_pair(
        report_root,
        "09_CLONE_TO_DYNAMICS_DESIGN",
        "Clone To Dynamics Design",
        payload,
        [
            "## Part 1: Clone Matrix Definition",
            "- `C[cellbin, clone]` stores sparse membership weights with clone confidence and clone class.",
            "- `C[niche, clone]` stores membership-weighted clone composition per spatial unit.",
            "- Assignment ambiguity remains an uncertainty variable.",
            "",
            "## Part 2: Niche Clone State",
            "- Each niche state has clone composition, clone entropy, dominant clone fraction, clone richness, clone class composition, and assigned/ambiguous/unassigned fractions.",
            "",
            "## Part 3: PlanA Integration",
            "- For future time-anchored or perturbation-anchored data, same-clone observations in source and target can support candidate transition hypotheses.",
            "- A future kernel can use `W_clone(i,j)` as clone composition similarity.",
            "- Suggested form: `K(i,j) ∝ exp(-cost_expr(i,j)) × exp(lambda_clone * clone_support(i,j)) × direction_gate(i,j) × confidence_weight(i,j)`.",
            "- Direction must still come from time, perturbation, or biological prior, not clone alone.",
            "",
            "## Part 4: PlanB / BranchSBM Integration",
            "- Clone composition can regularize source-target coupling and carry mass across niche states in future time-anchored data.",
            "- Same clone across multiple target branches indicates possible clone-level branch divergence to test, not a claim from L126 alone.",
            "- Niche fate is represented in future data as a clone-weighted branch distribution, not expression-state transport alone.",
            "",
            "## Part 5: Niche Fate Probability Definition",
            "- Future `P(branch b | niche n)` combines model transport probability, clone composition support, assignment confidence, branch target clone enrichment, and uncertainty penalty.",
            "",
            "## Part 6: What L126 Can And Cannot Do",
            "- L126 can validate clone/niche integration and spatial clone characterization.",
            "- L126 cannot validate temporal niche fate probability because sections are not timepoints.",
            "- Future time or perturbation DARLIN data are needed for temporal inference.",
        ],
        overwrite=overwrite,
    )
    return payload


def decide_final(signatures: pd.DataFrame, summaries: dict[str, dict[str, Any]], null_frame: pd.DataFrame, sensitivity_payload: dict[str, Any], round1: dict[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    warnings: list[str] = []
    expanded_summary = summaries.get("expanded", {})
    high_summary = summaries.get("high_confidence", {})
    expanded_fraction = float(expanded_summary.get("assigned_cellbin_fraction", 0.0))
    high_fraction = float(high_summary.get("assigned_cellbin_fraction", 0.0))
    largest_fraction = float(expanded_summary.get("largest_clone_weighted_cellbins", 0.0)) / max(int(expanded_summary.get("n_cellbins", 1)), 1)
    high_real_clones = int(signatures["clone_set_high_confidence"].sum()) if not signatures.empty else 0
    expanded_real_clones = int(signatures["clone_set_expanded"].sum()) if not signatures.empty else 0
    null_recap = False
    if not null_frame.empty:
        for clone_set, real_count in [("high_confidence", high_real_clones), ("expanded", expanded_real_clones)]:
            subset = null_frame.loc[null_frame["clone_set"].eq(clone_set)]
            if subset.empty or real_count == 0:
                continue
            comparable_count = subset["n_clones"].max() >= 0.75 * real_count
            threshold = float(summaries[clone_set].get("min_assignment_score", 0.0))
            comparable_score = subset["score_q99"].max() >= threshold and threshold > 0
            if comparable_count and comparable_score:
                null_recap = True
    if high_real_clones == 0 and expanded_real_clones == 0:
        label = "L126_CLONE_SIGNATURE_MODEL_HOLD_FOR_LOW_ASSIGNMENT"
    elif null_recap:
        label = "L126_CLONE_SIGNATURE_MODEL_HOLD_FOR_NULL_RECAPITULATION"
    elif largest_fraction > 0.05:
        label = "L126_CLONE_SIGNATURE_MODEL_HOLD_FOR_OVERMERGING"
    elif not bool(sensitivity_payload.get("sensitivity_stable", False)):
        label = "L126_CLONE_SIGNATURE_MODEL_HOLD_FOR_SENSITIVITY"
    elif expanded_fraction <= float(round1.get("round1_assigned_cellbin_fraction", 0.0)):
        label = "L126_CLONE_SIGNATURE_MODEL_HOLD_FOR_LOW_ASSIGNMENT"
    else:
        b_count = int(signatures["clone_class"].eq("single_locus_recurrent_clone").sum()) if not signatures.empty else 0
        if b_count > 0:
            warnings.append("single_locus_recurrent_clone contributes to expanded clone set and is reported separately")
        if expanded_fraction < 0.05:
            warnings.append("assignment coverage improves over Round 1 but remains incomplete")
        warnings.append("empirical rarity remains a fallback because official DARLIN rarity schema was not found")
        label = "L126_CLONE_SIGNATURE_MODEL_READY_WITH_WARNINGS" if warnings else "L126_CLONE_SIGNATURE_MODEL_READY"
    payload = {
        "final_decision_label": label,
        "warnings": warnings,
        "null_recapitulation_flag": bool(null_recap),
        "largest_clone_fraction": largest_fraction,
        "high_confidence_assigned_fraction": high_fraction,
        "expanded_assigned_fraction": expanded_fraction,
    }
    return label, warnings, payload


def write_final_decision(report_root: Path, payload: dict[str, Any], signatures: pd.DataFrame, summaries: dict[str, dict[str, Any]], round1: dict[str, Any], null_frame: pd.DataFrame, sensitivity_payload: dict[str, Any], niche_payload: dict[str, Any], figure_payload: dict[str, Any], *, overwrite: bool) -> None:
    class_counts = signatures["clone_class"].value_counts().reset_index() if not signatures.empty else pd.DataFrame(columns=["clone_class", "count"])
    class_counts.columns = ["clone_class", "n_clones"]
    body = [
        "## Final Decision",
        f"- Label: `{payload['final_decision_label']}`",
        "",
        "## Required Answers",
        "- Clones are defined as validated signatures passing Class A/B/C empirical rules.",
        "- CA/TA/RA are integrated only through assay-scoped feature evidence and compatibility support.",
        f"- High-confidence assigned fraction: {payload['high_confidence_assigned_fraction']:.6f}",
        f"- Expanded assigned fraction: {payload['expanded_assigned_fraction']:.6f}",
        f"- Round 1 assigned fraction: {round1['round1_assigned_cellbin_fraction']:.6f}",
        f"- Null recap flag: {payload['null_recapitulation_flag']}",
        f"- Sensitivity stable: {sensitivity_payload.get('sensitivity_stable')}",
        f"- Niche/tile integration rows: {niche_payload.get('tile_rows', 0)}",
        f"- Key figures path: `{Path('reports/l126_darlin_clone_signature_round2/key_figure_candidates')}`",
        f"- PlanA/PlanB design path: `reports/l126_darlin_clone_signature_round2/09_CLONE_TO_DYNAMICS_DESIGN.md`",
        "",
        "## Clone Counts By Class",
        markdown_table(class_counts),
        "",
        "## Warnings",
        markdown_table(pd.DataFrame({"warning": payload.get("warnings", [])})) if payload.get("warnings") else "- None.",
    ]
    write_report_pair(report_root, "10_CLONE_SIGNATURE_DECISION", "CloneSignature Decision", payload, body, overwrite=overwrite)


def main() -> None:
    args = parse_args()
    reject_forbidden_paths(args.input_packet_root, args.barcode_root, args.full_characterization_root, args.round1_clone_root, args.output_root, args.report_root)
    params = CloneSignatureParams(
        rare_threshold=args.rare_threshold,
        low_frequency_threshold=args.low_frequency_threshold,
        min_single_feature_cellbins=args.min_single_feature_cellbins,
        min_feature_cooccurrence_cellbins=args.min_feature_cooccurrence_cellbins,
    )
    line_path = lineage_evidence_path(args.input_packet_root)
    allele_path = allele_annotation_path(args.input_packet_root)
    missing = [str(path) for path in [line_path, allele_path] if not path.exists()]
    if missing:
        report_incomplete_inputs(args, missing)
        return
    ensure_dir(args.output_root)
    ensure_dir(args.report_root)
    before = snapshot_files([line_path, allele_path], include_sha256=True)

    write_problem_framing(args.report_root, True, overwrite=args.overwrite)
    lineage = load_cellbin_lineage_evidence(line_path)
    allele = load_feature_allele_annotation(allele_path)
    full_cellbins = load_full_cellbins(args.full_characterization_root, args.barcode_root, lineage)
    if args.mode == "contract_only":
        write_contract(PROJECT_ROOT / "configs/darlin_clone/l126_clone_signature_v2_contract.draft.json", args.report_root, params, overwrite=args.overwrite)
        return
    evidence, feature_ref, complexity, evidence_payload = build_canonical_evidence(lineage, allele, full_cellbins, params)
    write_evidence_outputs(args.output_root, args.report_root, evidence, feature_ref, complexity, evidence_payload, overwrite=args.overwrite)
    write_contract(PROJECT_ROOT / "configs/darlin_clone/l126_clone_signature_v2_contract.draft.json", args.report_root, params, overwrite=args.overwrite)
    if args.mode == "evidence_only":
        return
    edges, graph_summary, components, graph_payload = build_feature_compatibility_graph(evidence, feature_ref, complexity, params)
    write_graph_outputs(args.output_root, args.report_root, edges, graph_summary, components, graph_payload, overwrite=args.overwrite)
    if args.mode == "compatibility_graph_only":
        return
    signatures, signature_membership, filtered_candidates, signature_payload = build_clone_signatures(evidence, feature_ref, edges, components, complexity, params)
    signatures = annotate_spatial_qc(signatures, evidence)
    write_signature_outputs(args.output_root, args.report_root, signatures, signature_membership, filtered_candidates, signature_payload, overwrite=args.overwrite)
    if args.mode == "signatures_only":
        return

    real_scores = {
        clone_set: candidate_clone_scores(evidence, signatures, signature_membership, params, clone_set=clone_set)
        for clone_set in ["high_confidence", "expanded"]
    }
    null_frames = []
    null_scores: dict[str, list[pd.DataFrame]] = {"high_confidence": [], "expanded": []}
    if args.run_null_controls or args.mode in {"all", "null_controls_only"}:
        for clone_set in ["high_confidence", "expanded"]:
            frame, scores, _ = run_null_controls(lineage, allele, full_cellbins, params, clone_set=clone_set)
            null_frames.append(frame)
            null_scores[clone_set] = scores
    null_frame = pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
    thresholds = {}
    for clone_set in ["high_confidence", "expanded"]:
        threshold = calibrate_assignment_thresholds(null_scores[clone_set], real_scores[clone_set], clone_set=clone_set)
        if args.min_assignment_score != "auto":
            threshold["min_assignment_score"] = float(args.min_assignment_score)
            threshold["score_threshold_source"] = "cli_override"
        if args.min_score_margin != "auto":
            threshold["min_score_margin"] = float(args.min_score_margin)
            threshold["margin_threshold_source"] = "cli_override"
        thresholds[clone_set] = threshold

    assignments: dict[str, pd.DataFrame] = {}
    topks: dict[str, pd.DataFrame] = {}
    memberships: dict[str, pd.DataFrame] = {}
    matrices: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for clone_set in ["high_confidence", "expanded"]:
        assignment, topk, cell_membership, matrix, summary = assign_cellbins_to_clones(
            full_cellbins,
            evidence,
            real_scores[clone_set],
            thresholds[clone_set],
            params,
            clone_set=clone_set,
        )
        assignments[clone_set] = assignment
        topks[clone_set] = topk
        memberships[clone_set] = cell_membership
        matrices[clone_set] = matrix
        summaries[clone_set] = summary
    round1 = read_round1_summary(args.round1_clone_root, int(len(full_cellbins)))
    write_assignment_outputs(args.output_root, args.report_root, assignments, topks, memberships, matrices, summaries, round1, overwrite=args.overwrite)
    if args.mode == "assignment_only":
        return

    sensitivity = pd.DataFrame()
    bridge_sensitivity = pd.DataFrame()
    sensitivity_payload = {"decision_label": "CLONE_SIGNATURE_MODEL_READY_WITH_WARNINGS", "sensitivity_stable": True}
    if args.run_sensitivity or args.mode in {"all", "null_controls_only"}:
        sensitivity, _, bridge_sensitivity, sensitivity_payload = run_sensitivity_grid(
            lineage,
            allele,
            full_cellbins,
            params,
            baseline_signatures=signatures,
            baseline_edges=edges,
            baseline_feature_reference=feature_ref,
            baseline_complexity=complexity,
        )
    null_decision = "CLONE_SIGNATURE_MODEL_READY_WITH_WARNINGS"
    if not null_frame.empty:
        high_real = max(int(signatures["clone_set_high_confidence"].sum()) if not signatures.empty else 0, 1)
        high_null = int(null_frame.loc[null_frame["clone_set"].eq("high_confidence"), "n_clones"].max()) if (null_frame["clone_set"].eq("high_confidence")).any() else 0
        null_decision = "HOLD_FOR_NULL_RECAPITULATION" if high_null >= 0.75 * high_real and high_real > 1 else "CLONE_SIGNATURE_MODEL_READY_WITH_WARNINGS"
    sensitivity_payload["decision_label"] = null_decision if null_decision.startswith("HOLD") else ("CLONE_SIGNATURE_MODEL_STABLE" if sensitivity_payload.get("sensitivity_stable") else "HOLD_FOR_SENSITIVITY")
    if sensitivity.empty:
        sensitivity = pd.DataFrame()
    if bridge_sensitivity.empty:
        bridge_sensitivity = pd.DataFrame()
    write_null_sensitivity_outputs(args.output_root, args.report_root, null_frame, sensitivity, bridge_sensitivity, thresholds, sensitivity_payload, overwrite=args.overwrite)
    if args.mode == "null_controls_only":
        return

    niche_frames, niche_payload = build_niche_outputs(args.full_characterization_root, assignments, memberships)
    write_niche_outputs(args.output_root, args.report_root, niche_frames, niche_payload, overwrite=args.overwrite)
    if args.mode == "niche_aggregation_only":
        return

    figure_payload: dict[str, Any] = {"figure_count": 0, "key_figure_count": 0}
    if args.make_figures or args.mode in {"all", "figures_only"}:
        tile_summary = niche_frames.get("tile_clone_summary_v2", pd.DataFrame())
        _, figure_payload = make_round2_figures(
            args.report_root,
            evidence,
            feature_ref,
            edges,
            signatures,
            assignments["expanded"],
            memberships["expanded"],
            tile_summary.loc[tile_summary["clone_set"].eq("expanded")] if not tile_summary.empty and "clone_set" in tile_summary else tile_summary,
            null_frame,
            bridge_sensitivity,
        )
    write_figures_report(args.report_root, figure_payload, overwrite=args.overwrite)
    if args.mode == "figures_only":
        return

    write_dynamics_design(args.report_root, overwrite=args.overwrite)
    if args.mode == "dynamics_design_only":
        return
    final_label, warnings, decision_payload = decide_final(signatures, summaries, null_frame, sensitivity_payload, round1)
    decision_payload.update(
        {
            "generated_at_utc": utc_now(),
            "clone_counts_by_class": signatures["clone_class"].value_counts().to_dict() if not signatures.empty else {},
            "high_confidence_clone_count": int(signatures["clone_set_high_confidence"].sum()) if not signatures.empty else 0,
            "expanded_clone_count": int(signatures["clone_set_expanded"].sum()) if not signatures.empty else 0,
            "round1_assigned_cellbin_fraction": round1["round1_assigned_cellbin_fraction"],
            "null_control_summary": null_frame.to_dict(orient="records") if not null_frame.empty else [],
            "sensitivity_summary": sensitivity_payload,
            "niche_clone_integration_status": "generated" if niche_payload.get("tile_rows", 0) else "missing",
            "key_figures_path": str(args.report_root / "key_figure_candidates"),
            "plana_planb_dynamics_design_path": str(args.report_root / "09_CLONE_TO_DYNAMICS_DESIGN.md"),
        }
    )
    write_final_decision(args.report_root, decision_payload, signatures, summaries, round1, null_frame, sensitivity_payload, niche_payload, figure_payload, overwrite=args.overwrite)

    after = snapshot_files([line_path, allele_path], include_sha256=True)
    diff = compare_file_snapshots(before, after)
    validation_payload = validate_round2_outputs(
        args.output_root,
        args.report_root,
        bool(diff["changed"].any()),
        figures_required=bool(args.make_figures or args.mode == "all"),
    )
    write_report_pair(
        args.report_root,
        "11_VALIDATION",
        "Validation",
        validation_payload,
        [
            "## Validation",
            f"- Status: `{validation_payload['validation_status']}`",
            f"- JSON parse: {validation_payload['json_parse']}",
            f"- TSV/gzip readability: {validation_payload['tsv_gzip_readability']}",
            f"- Source input packet unchanged: {validation_payload['source_input_packet_unchanged']}",
            f"- Positive claim-language check: {validation_payload['no_positive_fate_terminal_transition_claims']}",
        ],
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
