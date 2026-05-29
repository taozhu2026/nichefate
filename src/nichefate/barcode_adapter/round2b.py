from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .round2_qc import distribution_summary
from .spatial_neighborhood import GROUP_TYPE


GROUP_ASSIGNMENT_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "slice_id",
    "section_order",
    "group_id",
    "group_type",
    "niche_id",
    "anchor_cellbin_id",
    "anchor_x",
    "anchor_y",
    "cellbin_id",
    "x",
    "y",
    "role",
)

ALLOWED_PLANA_READINESS_LABELS: tuple[str, ...] = (
    "L126_PLANA_BARCODE_ROUTE_A_READY",
    "L126_PLANA_BARCODE_ROUTE_B_READY_WITH_WARNINGS",
    "L126_PLANA_BARCODE_HOLD_FOR_M0_M1_ADAPTATION",
    "L126_PLANA_BARCODE_HOLD_FOR_METANICHE_CONTRACT",
    "L126_PLANA_BARCODE_HOLD_FOR_LINEAGE_AGGREGATION",
)


def parse_sample_list(raw: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",")]
    else:
        values = [str(item).strip() for item in raw]
    samples = tuple(item for item in values if item)
    if not samples:
        raise ValueError("At least one sample must be provided")
    return samples


def validate_round2b_group_assignment(
    assignment: pd.DataFrame,
    cellbin_table: pd.DataFrame,
    *,
    sample_id: str,
    k_neighbors: int = 16,
) -> dict[str, Any]:
    missing = [column for column in GROUP_ASSIGNMENT_COLUMNS if column not in assignment.columns]
    if missing:
        return {
            "sample_id": sample_id,
            "missing_required_columns": missing,
            "required_columns_present": False,
            "validation_passed": False,
        }
    h5ad_ids = set(cellbin_table["cellbin_id"].astype(str))
    sample_assignment = assignment.loc[assignment["sample_id"].astype(str) == sample_id].copy()
    group_sizes = sample_assignment.groupby("group_id").size()
    center_rows = sample_assignment.loc[sample_assignment["role"].astype(str) == "center"]
    centers_per_group = center_rows.groupby("group_id")["anchor_cellbin_id"].nunique()
    expected_group_id = (
        sample_assignment["sample_id"].astype(str)
        + "__anchor__"
        + sample_assignment["anchor_cellbin_id"].astype(str)
        if not sample_assignment.empty and "anchor_cellbin_id" in sample_assignment
        else pd.Series(dtype=object)
    )
    validation_passed = bool(
        len(sample_assignment)
        and set(assignment["sample_id"].astype(str)) == {sample_id}
        and sample_assignment["cellbin_id"].notna().all()
        and set(sample_assignment["cellbin_id"].astype(str)).issubset(h5ad_ids)
        and set(sample_assignment["anchor_cellbin_id"].astype(str)).issubset(h5ad_ids)
        and (sample_assignment["group_id"].astype(str) == expected_group_id).all()
        and (sample_assignment["niche_id"].astype(str) == sample_assignment["group_id"].astype(str)).all()
        and sample_assignment["group_type"].astype(str).eq(GROUP_TYPE).all()
        and (centers_per_group == 1).all()
        and len(centers_per_group) == sample_assignment["group_id"].nunique()
        and np.isfinite(group_sizes.to_numpy(dtype=float)).all()
        and group_sizes.eq(k_neighbors).all()
        and sample_assignment.groupby("group_id")["slice_id"].nunique().eq(1).all()
    )
    return {
        "sample_id": sample_id,
        "missing_required_columns": missing,
        "required_columns_present": not missing,
        "has_rows": bool(len(sample_assignment)),
        "sample_only": bool(set(assignment["sample_id"].astype(str)) == {sample_id}) if "sample_id" in assignment else False,
        "no_missing_cellbin_id": bool(sample_assignment["cellbin_id"].notna().all()) if "cellbin_id" in sample_assignment else False,
        "all_group_members_in_h5ad": bool(set(sample_assignment["cellbin_id"].astype(str)).issubset(h5ad_ids))
        if "cellbin_id" in sample_assignment
        else False,
        "all_anchors_in_h5ad": bool(set(sample_assignment["anchor_cellbin_id"].astype(str)).issubset(h5ad_ids))
        if "anchor_cellbin_id" in sample_assignment
        else False,
        "stable_group_id": bool((sample_assignment["group_id"].astype(str) == expected_group_id).all())
        if len(sample_assignment)
        else False,
        "niche_id_equals_group_id": bool((sample_assignment["niche_id"].astype(str) == sample_assignment["group_id"].astype(str)).all())
        if "niche_id" in sample_assignment and "group_id" in sample_assignment
        else False,
        "group_type_expected": bool(sample_assignment["group_type"].astype(str).eq(GROUP_TYPE).all())
        if "group_type" in sample_assignment
        else False,
        "every_group_has_one_anchor": bool((centers_per_group == 1).all() and len(centers_per_group) == sample_assignment["group_id"].nunique())
        if len(sample_assignment)
        else False,
        "group_sizes_finite": bool(np.isfinite(group_sizes.to_numpy(dtype=float)).all()) if len(group_sizes) else False,
        "group_size_matches_k": bool(group_sizes.eq(k_neighbors).all()) if len(group_sizes) else False,
        "no_cross_section_mixing": bool(sample_assignment.groupby("group_id")["slice_id"].nunique().eq(1).all())
        if "slice_id" in sample_assignment and len(sample_assignment)
        else False,
        "validation_passed": validation_passed,
    }


def distribution_summary_by_sample(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    rows = []
    for sample_id, group in frame.groupby("sample_id", sort=True):
        summary = distribution_summary(group, value_column)
        summary.insert(0, "sample_id", str(sample_id))
        rows.append(summary)
    all_summary = distribution_summary(frame, value_column)
    all_summary.insert(0, "sample_id", "ALL")
    rows.append(all_summary)
    return pd.concat(rows, ignore_index=True)


def section_summary_row(
    *,
    sample_id: str,
    h5ad_n_obs: int,
    assignment: pd.DataFrame,
    group_summary: pd.DataFrame,
    multiplicity: pd.DataFrame,
    coverage_metrics: dict[str, Any],
) -> dict[str, Any]:
    group_sizes = assignment.groupby("group_id").size()
    return {
        "sample_id": sample_id,
        "n_h5ad_cellbins": int(h5ad_n_obs),
        "n_sampled_cellbins": int(coverage_metrics["sampled_cellbins"]),
        "n_sampled_lineage_positive_cellbins": int(coverage_metrics["sampled_lineage_positive_cellbins"]),
        "sampled_lineage_positive_fraction": float(coverage_metrics["fraction_sampled_cellbins_with_lineage_evidence"]),
        "n_groups": int(coverage_metrics["number_of_groups"]),
        "n_group_assignment_rows": int(len(assignment)),
        "median_group_size": float(group_sizes.median()),
        "mean_member_multiplicity": float(multiplicity["groups_per_member_cellbin"].mean()),
        "median_member_multiplicity": float(multiplicity["groups_per_member_cellbin"].median()),
        "max_member_multiplicity": int(multiplicity["groups_per_member_cellbin"].max()),
        "groups_with_ge1_lineage_positive_member": int(coverage_metrics["groups_with_ge1_lineage_positive_member"]),
        "groups_with_ge3_lineage_positive_members": int(coverage_metrics["groups_with_ge3_lineage_positive_members"]),
        "fraction_groups_with_ge1_lineage_positive_member": float(
            coverage_metrics["groups_with_ge1_lineage_positive_member"] / coverage_metrics["number_of_groups"]
        ),
        "fraction_groups_with_ge3_lineage_positive_members": float(
            coverage_metrics["groups_with_ge3_lineage_positive_members"] / coverage_metrics["number_of_groups"]
        ),
        "median_fraction_member_cellbins_with_lineage": float(
            coverage_metrics["median_fraction_member_cellbins_with_lineage"]
        ),
        "median_total_lineage_count_per_group": float(coverage_metrics["median_total_lineage_count_per_group"]),
        "median_detected_feature_count_per_group": float(coverage_metrics["median_detected_feature_count_per_group"]),
        "median_feature_entropy": float(group_summary["feature_entropy"].median()),
        "median_dominant_feature_fraction": float(group_summary["dominant_feature_fraction"].median()),
    }


def build_plana_barcode_readiness_audit(project_root: str | Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    root = Path(project_root).expanduser().resolve()
    artifacts = [
        {
            "artifact": "PlanA full result packet",
            "path": root / "reports/planA_k_full_result_packet/00_FULL_RESULT_PACKET_SUMMARY.md",
            "role": "frozen PlanA-K summary and claim guardrails",
        },
        {
            "artifact": "PlanA full M2.5 production",
            "path": root / "reports/planA_k_full_m2_5_production/00_FULL_M2_5_PRODUCTION_SUMMARY.md",
            "role": "metaniche/state construction contract",
        },
        {
            "artifact": "PlanA full GPCCA",
            "path": root / "reports/planA_k_full_gpcca/00_FULL_GPCCA_SUMMARY.md",
            "role": "macrostate and GPCCA input/output contract",
        },
    ]
    artifact_table = pd.DataFrame(
        [
            {
                "artifact": item["artifact"],
                "path": str(item["path"]),
                "exists": item["path"].exists(),
                "role": item["role"],
            }
            for item in artifacts
        ]
    )
    all_artifacts_present = bool(artifact_table["exists"].all())
    route_table = pd.DataFrame(
        [
            {
                "route": "Route A",
                "label": "Conservative post-hoc annotation route",
                "description": "Aggregate barcode evidence to barcode-free niche/metaniche or macrostate outputs and annotate macrostates with barcode metrics.",
                "risk": "lowest",
                "recommended_next": True,
            },
            {
                "route": "Route B",
                "label": "Barcode-aware kernel route",
                "description": "Add barcode composition similarity, entropy, or overlap terms to a bounded Markov kernel before GPCCA smoke.",
                "risk": "higher",
                "recommended_next": False,
            },
        ]
    )
    label = (
        "L126_PLANA_BARCODE_ROUTE_A_READY"
        if all_artifacts_present
        else "L126_PLANA_BARCODE_HOLD_FOR_METANICHE_CONTRACT"
    )
    payload = {
        "readiness_label": label,
        "recommended_route": "Route A - Conservative post-hoc annotation route",
        "route_a_ready": bool(all_artifacts_present),
        "route_b_ready_with_warnings": False,
        "planA_priority_over_planB": True,
        "planA_priority_reason": "PlanA is more mature/frozen and has completed M2.5/Kmix_A/GPCCA result-packet guardrails.",
        "serial_sections_not_timepoints": True,
        "terminal_biological_fate_claim_allowed": False,
        "gpcca_wording_for_serial_sections": "macrostate / sink-like / reachability-like only, not validated temporal fate",
        "barcode_free_vs_barcode_aware_comparison_is_key_novelty": True,
        "artifacts_present": bool(all_artifacts_present),
        "allowed_label": label in ALLOWED_PLANA_READINESS_LABELS,
    }
    return payload, artifact_table, route_table
