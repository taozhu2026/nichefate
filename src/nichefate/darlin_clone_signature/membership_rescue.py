from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import CELL_COLUMNS, entropy_from_counts, make_cell_key, simpson_from_counts, summarize_top_items
from .reporting import ensure_dir, positive_claim_hits, read_table


HIGH_CONFIDENCE_CLASSES = {"cross_locus_clone", "multi_feature_single_locus_clone"}
CLASS_B = "single_locus_recurrent_clone"
SUPPORT_STATUSES = {"supported", "weak_supported", "ambiguous_multi"}


def clone_set_for_class(clone_class: str) -> str:
    return "expanded" if clone_class == CLASS_B else "high_confidence"


def required_round2_paths(round2_root: Path, round2_report_root: Path) -> dict[str, Path]:
    return {
        "clone_signatures": round2_root / "signatures/clone_signatures.tsv.gz",
        "signature_feature_membership": round2_root / "signatures/clone_signature_feature_membership.tsv.gz",
        "evidence": round2_root / "evidence/cellbin_feature_evidence.tsv.gz",
        "high_confidence_scores": round2_root / "assignments/high_confidence_cellbin_clone_score_topk.tsv.gz",
        "expanded_scores": round2_root / "assignments/expanded_cellbin_clone_score_topk.tsv.gz",
        "high_confidence_assignment": round2_root / "assignments/high_confidence_cellbin_clone_assignment_v2.tsv.gz",
        "expanded_assignment": round2_root / "assignments/expanded_cellbin_clone_assignment_v2.tsv.gz",
        "clone_by_cellbin_matrix": round2_root / "assignments/clone_by_cellbin_matrix.tsv.gz",
        "null_control_comparison": round2_root / "sensitivity/null_control_comparison.tsv",
        "clone_signature_sensitivity": round2_root / "sensitivity/clone_signature_sensitivity.tsv",
        "tile_clone_summary": round2_root / "niche_clone/tile_clone_summary_v2.tsv.gz",
        "group_clone_summary": round2_root / "niche_clone/group_clone_summary_v2.tsv.gz",
        "metaniche_clone_summary": round2_root / "niche_clone/metaniche_clone_summary_v2.tsv.gz",
        "decision_json": round2_report_root / "10_CLONE_SIGNATURE_DECISION.json",
        "validation_json": round2_report_root / "11_VALIDATION.json",
        "decision_md": round2_report_root / "10_CLONE_SIGNATURE_DECISION.md",
        "validation_md": round2_report_root / "11_VALIDATION.md",
    }


def missing_round2_paths(paths: dict[str, Path]) -> list[str]:
    return [str(path) for path in paths.values() if not path.exists()]


def build_signature_overlap(evidence: pd.DataFrame, signature_membership: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty or signature_membership.empty:
        return pd.DataFrame(
            columns=[
                "cell_key",
                "clone_id",
                "clone_class",
                "raw_support_count",
                "n_supporting_features",
                "n_supporting_loci",
            ]
        )
    ev_cols = [
        "cell_key",
        "assay_scoped_feature_id",
        "assay",
        "count",
    ]
    work = evidence[ev_cols].copy()
    members = signature_membership[["clone_id", "clone_class", "assay_scoped_feature_id"]].drop_duplicates()
    joined = work.merge(members, on="assay_scoped_feature_id", how="inner")
    if joined.empty:
        return pd.DataFrame()
    return (
        joined.groupby(["cell_key", "clone_id", "clone_class"], as_index=False)
        .agg(
            raw_support_count=("count", "sum"),
            n_supporting_features=("assay_scoped_feature_id", "nunique"),
            n_supporting_loci=("assay", "nunique"),
        )
        .reset_index(drop=True)
    )


def _assignment_for_overlap(overlap: pd.DataFrame, assignments: dict[str, pd.DataFrame]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for clone_set, classes in [
        ("high_confidence", HIGH_CONFIDENCE_CLASSES),
        ("expanded", {CLASS_B}),
    ]:
        subset = overlap.loc[overlap["clone_class"].isin(classes)].copy()
        assignment = assignments.get(clone_set, pd.DataFrame())
        if subset.empty or assignment.empty:
            continue
        cols = ["cell_key", "clone_id", "assignment_status", "reason_if_not_assigned", "assignment_score", "score_margin"]
        joined = subset.merge(
            assignment[cols].rename(columns={"clone_id": "hard_clone_id"}),
            on="cell_key",
            how="left",
        )
        joined["hard_clone_set"] = clone_set
        chunks.append(joined)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    out["assignment_status"] = out["assignment_status"].fillna("unassigned")
    out["reason_if_not_assigned"] = out["reason_if_not_assigned"].fillna("missing_hard_assignment_row")
    out["hard_clone_id"] = out["hard_clone_id"].fillna("")
    out["assigned_to_this_clone"] = out["assignment_status"].isin(["assigned_single", "assigned_multi"]) & out["hard_clone_id"].eq(out["clone_id"])
    out["lost_reason"] = out["assignment_status"]
    out.loc[out["assignment_status"].isin(["assigned_single", "assigned_multi"]) & ~out["assigned_to_this_clone"], "lost_reason"] = "assigned_to_other_clone"
    return out


def audit_signature_assignment_loss(
    signatures: pd.DataFrame,
    overlap: pd.DataFrame,
    assignments: dict[str, pd.DataFrame],
    score_tables: dict[str, pd.DataFrame],
    null_comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    assigned_overlap = _assignment_for_overlap(overlap, assignments)
    raw_counts = (
        overlap.groupby("clone_id", as_index=False)
        .agg(
            n_supporting_cellbins_by_raw_feature_overlap=("cell_key", "nunique"),
            n_cellbins_with_any_signature_feature=("cell_key", "nunique"),
            n_cellbins_with_multiple_signature_features=("n_supporting_features", lambda s: int((s.astype(int) >= 2).sum())),
        )
        if not overlap.empty
        else pd.DataFrame(columns=["clone_id"])
    )
    hard_counts = (
        assigned_overlap.loc[assigned_overlap["assigned_to_this_clone"]]
        .groupby("clone_id", as_index=False)
        .agg(n_cellbins_assigned_hard=("cell_key", "nunique"))
        if not assigned_overlap.empty
        else pd.DataFrame(columns=["clone_id", "n_cellbins_assigned_hard"])
    )
    loss_counts = (
        assigned_overlap.loc[~assigned_overlap["assigned_to_this_clone"]]
        .groupby(["clone_id", "lost_reason"], as_index=False)
        .agg(n_lost_cellbins=("cell_key", "nunique"))
        if not assigned_overlap.empty
        else pd.DataFrame(columns=["clone_id", "lost_reason", "n_lost_cellbins"])
    )
    loss_pivot = (
        loss_counts.pivot_table(index="clone_id", columns="lost_reason", values="n_lost_cellbins", fill_value=0).reset_index()
        if not loss_counts.empty
        else pd.DataFrame(columns=["clone_id"])
    )
    score_chunks: list[pd.DataFrame] = []
    for clone_set, table in score_tables.items():
        if table.empty:
            continue
        score_chunks.append(
            table.groupby("clone_id", as_index=False).agg(
                median_best_score=("score", "median"),
                n_candidate_score_rows=("cell_key", "size"),
                n_candidate_score_cellbins=("cell_key", "nunique"),
            )
        )
    score_stats = pd.concat(score_chunks, ignore_index=True) if score_chunks else pd.DataFrame(columns=["clone_id"])
    score_stats = score_stats.sort_values("n_candidate_score_rows", ascending=False).drop_duplicates("clone_id") if not score_stats.empty else score_stats

    margins: list[pd.DataFrame] = []
    for clone_set, assignment in assignments.items():
        if assignment.empty:
            continue
        assigned = assignment.loc[assignment["clone_id"].astype(str).ne("")]
        if assigned.empty:
            continue
        margins.append(assigned.groupby("clone_id", as_index=False).agg(median_score_margin=("score_margin", "median")))
    margin_stats = pd.concat(margins, ignore_index=True) if margins else pd.DataFrame(columns=["clone_id", "median_score_margin"])
    margin_stats = margin_stats.drop_duplicates("clone_id") if not margin_stats.empty else margin_stats

    audit = signatures.copy()
    audit = audit.merge(raw_counts, on="clone_id", how="left")
    audit = audit.merge(hard_counts, on="clone_id", how="left")
    audit = audit.merge(loss_pivot, on="clone_id", how="left")
    audit = audit.merge(score_stats, on="clone_id", how="left")
    audit = audit.merge(margin_stats, on="clone_id", how="left")
    numeric_defaults = [
        "n_supporting_cellbins_by_raw_feature_overlap",
        "n_cellbins_with_any_signature_feature",
        "n_cellbins_with_multiple_signature_features",
        "n_cellbins_assigned_hard",
        "median_best_score",
        "median_score_margin",
        "n_candidate_score_rows",
        "n_candidate_score_cellbins",
    ]
    for col in numeric_defaults:
        if col not in audit:
            audit[col] = 0
        audit[col] = audit[col].fillna(0)
    for reason in ["ambiguous", "unassigned", "filtered", "assigned_to_other_clone"]:
        if reason not in audit:
            audit[reason] = 0
        audit[reason] = audit[reason].fillna(0)
    raw = audit["n_cellbins_with_any_signature_feature"].replace(0, np.nan)
    audit["assignment_conversion_rate"] = (audit["n_cellbins_assigned_hard"] / raw).fillna(0.0)
    audit["fraction_lost_to_ambiguous"] = (audit["ambiguous"] / raw).fillna(0.0)
    audit["fraction_lost_to_unassigned"] = (audit["unassigned"] / raw).fillna(0.0)
    audit["fraction_lost_to_filtered"] = (audit["filtered"] / raw).fillna(0.0)
    audit["null_enrichment_score"] = _class_null_enrichment(signatures, null_comparison).reindex(audit["clone_class"]).to_numpy()
    audit["hard_assignment_clone_set"] = audit["clone_class"].map(clone_set_for_class)

    class_level = (
        audit.groupby("clone_class", as_index=False)
        .agg(
            n_signatures=("clone_id", "nunique"),
            n_zero_hard_assigned=("n_cellbins_assigned_hard", lambda s: int((s.astype(float) == 0).sum())),
            n_one_hard_assigned=("n_cellbins_assigned_hard", lambda s: int((s.astype(float) == 1).sum())),
            median_raw_support_cellbins=("n_cellbins_with_any_signature_feature", "median"),
            median_hard_assigned_cellbins=("n_cellbins_assigned_hard", "median"),
            median_assignment_conversion_rate=("assignment_conversion_rate", "median"),
            median_candidate_score=("median_best_score", "median"),
        )
        .reset_index(drop=True)
    )
    lost_summary = (
        assigned_overlap.loc[~assigned_overlap["assigned_to_this_clone"]]
        .groupby(["clone_class", "hard_clone_set", "lost_reason"], as_index=False)
        .agg(n_lost_cellbins=("cell_key", "nunique"))
        if not assigned_overlap.empty
        else pd.DataFrame(columns=["clone_class", "hard_clone_set", "lost_reason", "n_lost_cellbins"])
    )
    payload = {
        "n_signatures_audited": int(len(audit)),
        "n_signatures_with_zero_hard_assignment": int((audit["n_cellbins_assigned_hard"].astype(float) == 0).sum()),
        "n_signatures_with_one_hard_assignment": int((audit["n_cellbins_assigned_hard"].astype(float) == 1).sum()),
        "median_assignment_conversion_rate": float(audit["assignment_conversion_rate"].median()) if not audit.empty else 0.0,
    }
    return audit, class_level, lost_summary, payload


def _class_null_enrichment(signatures: pd.DataFrame, null_comparison: pd.DataFrame) -> pd.Series:
    real_counts = signatures["clone_class"].value_counts() if not signatures.empty else pd.Series(dtype=float)
    high_null = 0.0
    expanded_null = 0.0
    if not null_comparison.empty and "clone_set" in null_comparison:
        high = null_comparison.loc[null_comparison["clone_set"].eq("high_confidence"), "n_clones"]
        expanded = null_comparison.loc[null_comparison["clone_set"].eq("expanded"), "n_clones"]
        high_null = float(high.max()) if len(high) else 0.0
        expanded_null = float(expanded.max()) if len(expanded) else 0.0
    values = {
        "cross_locus_clone": float(real_counts.get("cross_locus_clone", 0) / max(high_null, 1.0)),
        "multi_feature_single_locus_clone": float(real_counts.get("multi_feature_single_locus_clone", 0) / max(high_null, 1.0)),
        "single_locus_recurrent_clone": float(real_counts.get("single_locus_recurrent_clone", 0) / max(expanded_null, 1.0)),
    }
    return pd.Series(values)


def _null_stats(null_comparison: pd.DataFrame, clone_set: str) -> dict[str, float]:
    subset = null_comparison.loc[null_comparison["clone_set"].eq(clone_set)] if "clone_set" in null_comparison else pd.DataFrame()
    if subset.empty:
        return {"q95": 0.0, "q99": 0.0, "max": 0.0, "candidate_scores": 0.0, "n_clones": 0.0}
    return {
        "q95": float(subset["score_q95"].max()),
        "q99": float(subset["score_q99"].max()),
        "max": float(subset["max_score"].max()),
        "candidate_scores": float(subset["n_candidate_scores"].max()),
        "n_clones": float(subset["n_clones"].max()),
    }


def _normal_from_quantiles(q95: float, q99: float) -> tuple[float, float]:
    if q99 <= q95:
        sigma = max(q99, q95, 1.0) / 3.0
        mu = max(0.0, q95 - 1.645 * sigma)
        return mu, max(sigma, 1e-6)
    sigma = (q99 - q95) / (2.326 - 1.645)
    mu = q95 - 1.645 * sigma
    return float(mu), float(max(sigma, 1e-6))


def calibrate_membership_thresholds(
    null_comparison: pd.DataFrame,
    score_tables: dict[str, pd.DataFrame],
    *,
    class_b_mode: str = "exploratory",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    high_null = _null_stats(null_comparison, "high_confidence")
    expanded_null = _null_stats(null_comparison, "expanded")
    rows: list[dict[str, Any]] = []
    for clone_class in ["cross_locus_clone", "multi_feature_single_locus_clone", "single_locus_recurrent_clone"]:
        clone_set = clone_set_for_class(clone_class)
        table = score_tables.get(clone_set, pd.DataFrame())
        class_scores = table.loc[table["clone_class"].eq(clone_class), "score"].astype(float) if not table.empty else pd.Series(dtype=float)
        stats = expanded_null if clone_class == CLASS_B else high_null
        null_mu, null_sigma = _normal_from_quantiles(stats["q95"], stats["q99"])
        if clone_class == CLASS_B:
            min_support = 10.0 if class_b_mode == "exploratory" else max(stats["q95"], 10.0)
            min_supported = max(stats["q95"], min_support)
            robust = max(stats["q99"], min_supported)
            evidence_layer = "expanded_exploratory"
            class_b_warning = True
        else:
            observed_floor = float(class_scores.quantile(0.05)) if len(class_scores) else 7.0
            min_support = max(7.0, min(observed_floor, 8.0))
            min_supported = max(stats["q95"], min_support)
            robust = max(stats["q99"], min_supported)
            evidence_layer = "high_confidence"
            class_b_warning = False
        rows.append(
            {
                "clone_class": clone_class,
                "clone_set": clone_set,
                "evidence_layer": evidence_layer,
                "min_support_score": float(min_support),
                "min_supported_score": float(min_supported),
                "robust_supported_score": float(robust),
                "min_membership_weight": 0.01,
                "null_score_q95": float(stats["q95"]),
                "null_score_q99": float(stats["q99"]),
                "null_score_max": float(stats["max"]),
                "null_candidate_score_rows_max": int(stats["candidate_scores"]),
                "null_clone_count_max": int(stats["n_clones"]),
                "null_mu_approx": float(null_mu),
                "null_sigma_approx": float(null_sigma),
                "real_candidate_score_rows": int(len(class_scores)),
                "real_candidate_cellbins": int(table.loc[table["clone_class"].eq(clone_class), "cell_key"].nunique()) if not table.empty else 0,
                "real_score_q50": float(class_scores.quantile(0.50)) if len(class_scores) else 0.0,
                "real_score_q95": float(class_scores.quantile(0.95)) if len(class_scores) else 0.0,
                "real_score_q99": float(class_scores.quantile(0.99)) if len(class_scores) else 0.0,
                "class_b_warning": class_b_warning,
            }
        )
    thresholds = pd.DataFrame(rows)
    null_calibration = thresholds[
        [
            "clone_class",
            "clone_set",
            "null_score_q95",
            "null_score_q99",
            "null_score_max",
            "null_candidate_score_rows_max",
            "real_candidate_score_rows",
            "real_candidate_cellbins",
        ]
    ].copy()
    class_calibration = thresholds.copy()
    high_real_rows = int(
        sum(
            len(score_tables.get("high_confidence", pd.DataFrame()).loc[lambda f: f["clone_class"].isin(HIGH_CONFIDENCE_CLASSES)])
            for _ in [0]
        )
    )
    high_null_rows = int(high_null["candidate_scores"])
    high_null_recap = bool(high_null_rows >= 0.75 * max(high_real_rows, 1))
    class_b_unseparated = bool(expanded_null["n_clones"] >= 0.75 * max(thresholds.loc[thresholds["clone_class"].eq(CLASS_B), "real_candidate_cellbins"].iloc[0], 1))
    label = "HOLD_FOR_NULL_RECAPITULATION" if high_null_recap else "MEMBERSHIP_READY_WITH_CLASS_B_WARNINGS"
    payload = {
        "decision_label": label,
        "high_confidence_null_recapitulation": high_null_recap,
        "high_confidence_real_candidate_score_rows": high_real_rows,
        "high_confidence_null_candidate_score_rows_max": high_null_rows,
        "class_b_mode": class_b_mode,
        "class_b_null_separation_warning": True,
        "class_b_unseparated_by_signature_count": class_b_unseparated,
    }
    return thresholds, null_calibration, class_calibration, payload


def build_membership_matrix(
    score_tables: dict[str, pd.DataFrame],
    assignments: dict[str, pd.DataFrame],
    signatures: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    high = score_tables.get("high_confidence", pd.DataFrame()).copy()
    expanded = score_tables.get("expanded", pd.DataFrame()).copy()
    chunks: list[pd.DataFrame] = []
    if not high.empty:
        chunks.append(high.loc[high["clone_class"].isin(HIGH_CONFIDENCE_CLASSES)].assign(clone_set="high_confidence"))
    if not expanded.empty:
        chunks.append(expanded.loc[expanded["clone_class"].eq(CLASS_B)].assign(clone_set="expanded"))
    if chunks:
        candidates = pd.concat(chunks, ignore_index=True)
    else:
        candidates = pd.DataFrame(columns=["cell_key", "clone_id", "clone_class", "score", "n_supporting_features", "n_supporting_loci"])
    candidates = candidates.merge(thresholds, on=["clone_class", "clone_set"], how="left")
    candidates["support_score"] = candidates["score"].astype(float)
    candidates = candidates.loc[candidates["support_score"].ge(candidates["min_support_score"].astype(float))].copy()
    if candidates.empty:
        cell_summary = _empty_cell_summary(assignments)
        clone_summary = _empty_clone_summary(signatures)
        return candidates, cell_summary, clone_summary, {"membership_supported_cellbin_fraction": 0.0}
    denom = candidates.groupby("cell_key")["support_score"].transform("sum").replace(0, np.nan)
    candidates["membership_weight"] = (candidates["support_score"] / denom).fillna(0.0)
    candidates["support_score_null_z"] = (
        (candidates["support_score"] - candidates["null_mu_approx"].astype(float)) / candidates["null_sigma_approx"].replace(0, np.nan).astype(float)
    ).fillna(0.0)
    candidates["raw_support_count"] = candidates["n_supporting_features"].astype(int)
    candidates["rarity_weighted_support"] = candidates["support_score"]
    candidates["membership_status"] = "null_like"
    high_mask = candidates["clone_class"].isin(HIGH_CONFIDENCE_CLASSES)
    b_mask = candidates["clone_class"].eq(CLASS_B)
    candidates.loc[
        high_mask & candidates["support_score"].ge(candidates["min_supported_score"].astype(float)) & candidates["membership_weight"].ge(candidates["min_membership_weight"].astype(float)),
        "membership_status",
    ] = "supported"
    candidates.loc[
        high_mask & candidates["membership_status"].eq("null_like") & candidates["membership_weight"].ge(candidates["min_membership_weight"].astype(float)),
        "membership_status",
    ] = "weak_supported"
    candidates.loc[
        b_mask
        & candidates["support_score"].ge(candidates["min_supported_score"].astype(float))
        & candidates["membership_weight"].ge(candidates["min_membership_weight"].astype(float)),
        "membership_status",
    ] = "weak_supported"
    candidates.loc[
        b_mask
        & candidates["support_score"].ge(candidates["robust_supported_score"].astype(float))
        & candidates["membership_weight"].ge(candidates["min_membership_weight"].astype(float)),
        "membership_status",
    ] = "weak_supported"

    support_mask = candidates["membership_status"].isin(["supported", "weak_supported"])
    support_counts = candidates.loc[support_mask].groupby("cell_key")["clone_id"].transform("nunique")
    support_weight_max = candidates.loc[support_mask].groupby("cell_key")["membership_weight"].transform("max")
    ambiguous_index = candidates.loc[support_mask].index[(support_counts > 1) & (support_weight_max < 0.75)]
    candidates.loc[ambiguous_index, "membership_status"] = "ambiguous_multi"
    cellbase = _cellbase_from_assignments(assignments)
    candidates = candidates.merge(cellbase, on="cell_key", how="left")
    ordered_cols = [
        *CELL_COLUMNS,
        "cell_key",
        "clone_id",
        "clone_class",
        "clone_set",
        "raw_support_count",
        "rarity_weighted_support",
        "n_supporting_features",
        "n_supporting_loci",
        "support_score",
        "support_score_null_z",
        "membership_weight",
        "membership_status",
        "supporting_features",
        "evidence_layer",
    ]
    for col in ordered_cols:
        if col not in candidates:
            candidates[col] = ""
    membership = candidates[ordered_cols].sort_values(["cell_key", "support_score", "clone_id"], ascending=[True, False, True])
    cell_summary = summarize_cell_membership(cellbase, membership)
    clone_summary = summarize_clone_membership(signatures, membership)
    supported_cells = int(cell_summary["assignment_mode"].isin(["single_clone_dominant", "multi_clone_supported", "ambiguous"]).sum())
    payload = {
        "n_membership_rows": int(len(membership)),
        "n_cellbins": int(len(cell_summary)),
        "n_cellbins_with_clone_membership": supported_cells,
        "membership_supported_cellbin_fraction": float(supported_cells / max(len(cell_summary), 1)),
        "n_null_like_rows": int(membership["membership_status"].eq("null_like").sum()),
        "n_weak_supported_rows": int(membership["membership_status"].eq("weak_supported").sum()),
        "n_ambiguous_multi_rows": int(membership["membership_status"].eq("ambiguous_multi").sum()),
        "n_supported_rows": int(membership["membership_status"].eq("supported").sum()),
        "class_b_rows": int(membership["clone_class"].eq(CLASS_B).sum()),
    }
    return membership, cell_summary, clone_summary, payload


def _cellbase_from_assignments(assignments: dict[str, pd.DataFrame]) -> pd.DataFrame:
    for key in ["expanded", "high_confidence"]:
        frame = assignments.get(key, pd.DataFrame())
        if not frame.empty:
            cols = [*CELL_COLUMNS, "cell_key"]
            return frame[cols].drop_duplicates("cell_key").copy()
    return pd.DataFrame(columns=[*CELL_COLUMNS, "cell_key"])


def _empty_cell_summary(assignments: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cellbase = _cellbase_from_assignments(assignments)
    for col in [
        "n_supported_clones",
        "n_high_confidence_supported_clones",
        "n_expanded_supported_clones",
        "total_clone_support_score",
        "max_clone_membership_weight",
        "clone_membership_entropy",
    ]:
        cellbase[col] = 0
    cellbase["primary_clone_id"] = ""
    cellbase["primary_clone_class"] = ""
    cellbase["assignment_mode"] = "no_clone_signal"
    return cellbase


def _empty_clone_summary(signatures: pd.DataFrame) -> pd.DataFrame:
    out = signatures[["clone_id", "clone_class"]].copy() if not signatures.empty else pd.DataFrame(columns=["clone_id", "clone_class"])
    for col in ["n_member_cellbins_any", "n_supported_cellbins", "weighted_membership_cellbins", "total_support_score"]:
        out[col] = 0
    return out


def summarize_cell_membership(cellbase: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    support = membership.loc[membership["membership_status"].isin(SUPPORT_STATUSES)].copy()
    any_rows = membership.groupby("cell_key", as_index=False).agg(n_any_membership_rows=("clone_id", "size")) if not membership.empty else pd.DataFrame(columns=["cell_key", "n_any_membership_rows"])
    if support.empty:
        out = _empty_cell_summary({"expanded": cellbase})
        out = out.merge(any_rows, on="cell_key", how="left")
        out["n_any_membership_rows"] = out["n_any_membership_rows"].fillna(0).astype(int)
        out.loc[out["n_any_membership_rows"].gt(0), "assignment_mode"] = "null_like"
        return out
    primary = support.sort_values(["cell_key", "membership_weight", "support_score"], ascending=[True, False, False]).drop_duplicates("cell_key")
    grouped = support.groupby("cell_key").agg(
        n_supported_clones=("clone_id", "nunique"),
        n_high_confidence_supported_clones=("clone_set", lambda s: int((s == "high_confidence").sum())),
        n_expanded_supported_clones=("clone_set", lambda s: int((s == "expanded").sum())),
        total_clone_support_score=("support_score", "sum"),
        max_clone_membership_weight=("membership_weight", "max"),
        clone_membership_entropy=("membership_weight", lambda s: entropy_from_counts(s.tolist())),
    ).reset_index()
    grouped = grouped.merge(
        primary[["cell_key", "clone_id", "clone_class"]].rename(columns={"clone_id": "primary_clone_id", "clone_class": "primary_clone_class"}),
        on="cell_key",
        how="left",
    )
    grouped = grouped.merge(any_rows, on="cell_key", how="left")
    grouped["n_any_membership_rows"] = grouped["n_any_membership_rows"].fillna(0).astype(int)
    out = cellbase.merge(grouped, on="cell_key", how="left")
    if not any_rows.empty:
        out = out.merge(any_rows.rename(columns={"n_any_membership_rows": "n_any_membership_rows_all"}), on="cell_key", how="left")
        out["n_any_membership_rows"] = out["n_any_membership_rows"].fillna(out["n_any_membership_rows_all"])
        out = out.drop(columns=["n_any_membership_rows_all"])
    numeric = [
        "n_supported_clones",
        "n_high_confidence_supported_clones",
        "n_expanded_supported_clones",
        "total_clone_support_score",
        "max_clone_membership_weight",
        "clone_membership_entropy",
        "n_any_membership_rows",
    ]
    for col in numeric:
        out[col] = out[col].fillna(0)
    out["primary_clone_id"] = out["primary_clone_id"].fillna("")
    out["primary_clone_class"] = out["primary_clone_class"].fillna("")
    out["assignment_mode"] = "no_clone_signal"
    out.loc[out["n_any_membership_rows"].gt(0) & out["n_supported_clones"].eq(0), "assignment_mode"] = "null_like"
    out.loc[out["n_supported_clones"].eq(1), "assignment_mode"] = "single_clone_dominant"
    out.loc[out["n_supported_clones"].gt(1), "assignment_mode"] = "multi_clone_supported"
    out.loc[out["n_supported_clones"].gt(1) & out["max_clone_membership_weight"].lt(0.50), "assignment_mode"] = "ambiguous"
    return out


def summarize_clone_membership(signatures: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    base = signatures[["clone_id", "clone_class", "clone_set_high_confidence", "clone_set_expanded"]].copy()
    if membership.empty:
        return _empty_clone_summary(signatures)
    support = membership.loc[membership["membership_status"].isin(SUPPORT_STATUSES)].copy()
    any_summary = membership.groupby("clone_id", as_index=False).agg(
        n_member_cellbins_any=("cell_key", "nunique"),
        n_null_like_cellbins=("membership_status", lambda s: int((s == "null_like").sum())),
    )
    if support.empty:
        return base.merge(any_summary, on="clone_id", how="left").fillna(0)
    support_summary = support.groupby("clone_id", as_index=False).agg(
        n_supported_cellbins=("cell_key", "nunique"),
        weighted_membership_cellbins=("membership_weight", "sum"),
        total_support_score=("support_score", "sum"),
        median_support_score_null_z=("support_score_null_z", "median"),
        max_membership_weight=("membership_weight", "max"),
    )
    status_counts = (
        membership.groupby(["clone_id", "membership_status"], as_index=False)
        .agg(n_rows=("cell_key", "nunique"))
        .pivot_table(index="clone_id", columns="membership_status", values="n_rows", fill_value=0)
        .reset_index()
    )
    out = base.merge(any_summary, on="clone_id", how="left").merge(support_summary, on="clone_id", how="left").merge(status_counts, on="clone_id", how="left")
    for col in out.columns:
        if col not in {"clone_id", "clone_class"}:
            out[col] = out[col].fillna(0)
    return out


def write_sparse_membership(output_dir: Path, membership: pd.DataFrame, cell_summary: pd.DataFrame, signatures: pd.DataFrame) -> dict[str, Any]:
    ensure_dir(output_dir)
    if membership.empty:
        return {"sparse_matrix_written": False, "reason": "empty_membership"}
    try:
        from scipy import sparse
    except Exception as exc:  # pragma: no cover - depends on local environment
        return {"sparse_matrix_written": False, "reason": f"scipy_unavailable:{exc}"}
    row_index = cell_summary[["cell_key"]].drop_duplicates().reset_index(drop=True)
    col_index = signatures[["clone_id", "clone_class"]].drop_duplicates("clone_id").reset_index(drop=True)
    row_lookup = pd.Series(row_index.index.to_numpy(), index=row_index["cell_key"]).to_dict()
    col_lookup = pd.Series(col_index.index.to_numpy(), index=col_index["clone_id"]).to_dict()
    rows = membership["cell_key"].map(row_lookup)
    cols = membership["clone_id"].map(col_lookup)
    valid = rows.notna() & cols.notna()
    matrix = sparse.coo_matrix(
        (
            membership.loc[valid, "membership_weight"].astype(float).to_numpy(),
            (rows.loc[valid].astype(int).to_numpy(), cols.loc[valid].astype(int).to_numpy()),
        ),
        shape=(len(row_index), len(col_index)),
    ).tocsr()
    sparse.save_npz(output_dir / "clone_membership_sparse.npz", matrix)
    row_index.to_csv(output_dir / "clone_membership_sparse_rows.tsv.gz", sep="\t", index=False, compression="gzip")
    col_index.to_csv(output_dir / "clone_membership_sparse_cols.tsv.gz", sep="\t", index=False, compression="gzip")
    return {"sparse_matrix_written": True, "shape": [int(matrix.shape[0]), int(matrix.shape[1])], "nnz": int(matrix.nnz)}


def aggregate_membership_to_units(
    mapping: pd.DataFrame,
    cell_summary: pd.DataFrame,
    membership: pd.DataFrame,
    unit_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mapping.empty:
        return pd.DataFrame(), pd.DataFrame()
    mapping = mapping.copy()
    if "cell_key" not in mapping:
        mapping["cell_key"] = make_cell_key(mapping)
    base = mapping.groupby(unit_cols, dropna=False).agg(n_cellbins=("cell_key", "nunique")).reset_index()
    mode_join = mapping[unit_cols + ["cell_key"]].merge(
        cell_summary[["cell_key", "assignment_mode", "n_supported_clones", "total_clone_support_score"]],
        on="cell_key",
        how="left",
    )
    mode_join["assignment_mode"] = mode_join["assignment_mode"].fillna("no_clone_signal")
    support_modes = ["single_clone_dominant", "multi_clone_supported", "ambiguous"]
    mode_join["has_clone_membership"] = mode_join["assignment_mode"].isin(support_modes)
    cell_metrics = mode_join.groupby(unit_cols, dropna=False).agg(
        n_cellbins_with_clone_membership=("has_clone_membership", "sum"),
        total_clone_support=("total_clone_support_score", "sum"),
        multi_clone_cellbins=("assignment_mode", lambda s: int((s == "multi_clone_supported").sum())),
        ambiguous_cellbins=("assignment_mode", lambda s: int((s == "ambiguous").sum())),
    ).reset_index()
    support = membership.loc[membership["membership_status"].isin(SUPPORT_STATUSES)].copy()
    if support.empty:
        summary = base.merge(cell_metrics, on=unit_cols, how="left")
        return pd.DataFrame(), _finish_membership_unit_summary(summary, pd.DataFrame(), unit_cols)
    comp_join = mapping[unit_cols + ["cell_key"]].merge(
        support[["cell_key", "clone_id", "clone_class", "clone_set", "membership_weight", "support_score"]],
        on="cell_key",
        how="inner",
    )
    comp = (
        comp_join.groupby([*unit_cols, "clone_id", "clone_class", "clone_set"], dropna=False, as_index=False)
        .agg(
            clone_membership_weight=("membership_weight", "sum"),
            total_clone_support=("support_score", "sum"),
            n_member_cellbins=("cell_key", "nunique"),
        )
        .sort_values([*unit_cols, "clone_membership_weight"], ascending=[True] * len(unit_cols) + [False])
    )
    summary = base.merge(cell_metrics, on=unit_cols, how="left")
    summary = _finish_membership_unit_summary(summary, comp, unit_cols)
    return comp, summary


def _finish_membership_unit_summary(summary: pd.DataFrame, comp: pd.DataFrame, unit_cols: list[str]) -> pd.DataFrame:
    for col in ["n_cellbins_with_clone_membership", "total_clone_support", "multi_clone_cellbins", "ambiguous_cellbins"]:
        if col not in summary:
            summary[col] = 0
        summary[col] = summary[col].fillna(0)
    summary["fraction_cellbins_with_clone_membership"] = (
        summary["n_cellbins_with_clone_membership"].astype(float) / summary["n_cellbins"].replace(0, np.nan).astype(float)
    ).fillna(0.0)
    summary["multi_clone_fraction"] = (summary["multi_clone_cellbins"].astype(float) / summary["n_cellbins"].replace(0, np.nan).astype(float)).fillna(0.0)
    summary["ambiguous_membership_fraction"] = (summary["ambiguous_cellbins"].astype(float) / summary["n_cellbins"].replace(0, np.nan).astype(float)).fillna(0.0)
    if comp.empty:
        defaults: dict[str, Any] = {
            "n_supported_clones": 0,
            "n_supported_class_A_clones": 0,
            "n_supported_class_B_clones": 0,
            "n_supported_class_C_clones": 0,
            "dominant_clone_id": "",
            "dominant_clone_class": "",
            "dominant_clone_membership_fraction": 0.0,
            "clone_membership_entropy": 0.0,
            "simpson_clone_diversity": 0.0,
            "clone_richness": 0,
            "high_confidence_clone_support_fraction": 0.0,
            "expanded_clone_support_fraction": 0.0,
        }
        for col, value in defaults.items():
            summary[col] = value
        return summary
    richness = comp.groupby(unit_cols, dropna=False).agg(n_supported_clones=("clone_id", "nunique")).reset_index()
    for clone_class, col in [
        ("cross_locus_clone", "n_supported_class_A_clones"),
        ("single_locus_recurrent_clone", "n_supported_class_B_clones"),
        ("multi_feature_single_locus_clone", "n_supported_class_C_clones"),
    ]:
        counts = comp.loc[comp["clone_class"].eq(clone_class)].groupby(unit_cols, dropna=False)["clone_id"].nunique().rename(col).reset_index()
        richness = richness.merge(counts, on=unit_cols, how="left")
    dominant = comp.sort_values([*unit_cols, "clone_membership_weight"], ascending=[True] * len(unit_cols) + [False]).drop_duplicates(unit_cols)
    dominant = dominant[[*unit_cols, "clone_id", "clone_class", "clone_membership_weight"]].rename(
        columns={"clone_id": "dominant_clone_id", "clone_class": "dominant_clone_class", "clone_membership_weight": "dominant_clone_weight"}
    )
    total_weight = comp.groupby(unit_cols, dropna=False)["clone_membership_weight"].sum().rename("total_membership_weight").reset_index()
    diversity = (
        comp.groupby(unit_cols, dropna=False)["clone_membership_weight"]
        .agg(
            clone_membership_entropy=lambda s: entropy_from_counts(s.tolist()),
            simpson_clone_diversity=lambda s: simpson_from_counts(s.tolist()),
        )
        .reset_index()
    )
    set_weights = (
        comp.groupby([*unit_cols, "clone_set"], dropna=False)["clone_membership_weight"]
        .sum()
        .reset_index()
        .pivot_table(index=unit_cols, columns="clone_set", values="clone_membership_weight", fill_value=0)
        .reset_index()
    )
    summary = summary.merge(richness, on=unit_cols, how="left").merge(dominant, on=unit_cols, how="left").merge(total_weight, on=unit_cols, how="left").merge(diversity, on=unit_cols, how="left").merge(set_weights, on=unit_cols, how="left")
    for col in ["n_supported_clones", "n_supported_class_A_clones", "n_supported_class_B_clones", "n_supported_class_C_clones"]:
        summary[col] = summary[col].fillna(0).astype(int)
    summary["clone_richness"] = summary["n_supported_clones"]
    summary["dominant_clone_id"] = summary["dominant_clone_id"].fillna("")
    summary["dominant_clone_class"] = summary["dominant_clone_class"].fillna("")
    summary["total_membership_weight"] = summary["total_membership_weight"].fillna(0.0)
    summary["dominant_clone_weight"] = summary["dominant_clone_weight"].fillna(0.0)
    summary["dominant_clone_membership_fraction"] = (
        summary["dominant_clone_weight"] / summary["total_membership_weight"].replace(0, np.nan)
    ).fillna(0.0)
    for col in ["high_confidence", "expanded"]:
        if col not in summary:
            summary[col] = 0.0
    summary["high_confidence_clone_support_fraction"] = (summary["high_confidence"] / summary["total_membership_weight"].replace(0, np.nan)).fillna(0.0)
    summary["expanded_clone_support_fraction"] = (summary["expanded"] / summary["total_membership_weight"].replace(0, np.nan)).fillna(0.0)
    summary["clone_membership_entropy"] = summary["clone_membership_entropy"].fillna(0.0)
    summary["simpson_clone_diversity"] = summary["simpson_clone_diversity"].fillna(0.0)
    return summary


def make_round2_1_figures(
    report_root: Path,
    coverage: dict[str, float],
    loss_audit: pd.DataFrame,
    membership: pd.DataFrame,
    cell_summary: pd.DataFrame,
    tile_summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    null_calibration: pd.DataFrame,
    cell_coordinates: pd.DataFrame,
) -> tuple[list[Path], dict[str, Any]]:
    figure_dir = ensure_dir(report_root / "figures")
    paths: list[Path] = []

    plt.figure(figsize=(7, 4))
    pd.Series(coverage).plot(kind="bar", color=["#476c9b", "#7aa95c", "#d18c45", "#8a6f9e"])
    plt.ylabel("Cellbin fraction")
    plt.title("Clone support coverage comparison")
    paths.append(_save_figure(figure_dir / "coverage_comparison.png"))

    plt.figure(figsize=(7, 4))
    if not loss_audit.empty:
        loss_audit.groupby("clone_class")["assignment_conversion_rate"].median().plot(kind="bar", color="#5b8e7d")
    plt.ylabel("Median conversion rate")
    plt.title("Signature-to-hard-assignment loss")
    paths.append(_save_figure(figure_dir / "signature_assignment_loss_by_class.png"))

    plt.figure(figsize=(6, 5))
    coords = cell_coordinates.merge(cell_summary[["cell_key", "assignment_mode"]], on="cell_key", how="left") if not cell_coordinates.empty else pd.DataFrame()
    if not coords.empty and {"x", "y"}.issubset(coords.columns):
        supported = coords["assignment_mode"].isin(["single_clone_dominant", "multi_clone_supported", "ambiguous"])
        plt.scatter(coords.loc[~supported, "x"], coords.loc[~supported, "y"], s=1, c="#d8d8d8", linewidths=0)
        plt.scatter(coords.loc[supported, "x"], coords.loc[supported, "y"], s=2, c="#2a9d8f", linewidths=0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Membership-supported cellbins")
    paths.append(_save_figure(figure_dir / "membership_supported_cellbin_spatial_map.png"))

    for metric, filename, title in [
        ("fraction_cellbins_with_clone_membership", "tile_clone_membership_coverage.png", "Tile clone membership coverage"),
        ("clone_membership_entropy", "tile_clone_membership_entropy.png", "Tile clone membership entropy"),
        ("dominant_clone_membership_fraction", "tile_dominant_clone_membership_fraction.png", "Tile dominant clone membership fraction"),
    ]:
        plt.figure(figsize=(6, 4))
        if not tile_summary.empty and metric in tile_summary:
            tile_summary[metric].astype(float).hist(bins=40)
        plt.xlabel(metric)
        plt.ylabel("Tiles")
        plt.title(title)
        paths.append(_save_figure(figure_dir / filename))

    plt.figure(figsize=(7, 4))
    if not tile_summary.empty:
        class_cols = ["n_supported_class_A_clones", "n_supported_class_B_clones", "n_supported_class_C_clones"]
        tile_summary[class_cols].sum().plot(kind="bar", color=["#476c9b", "#d18c45", "#5b8e7d"])
    plt.ylabel("Supported clone count across tiles")
    plt.title("Tile Class A/B/C contribution")
    paths.append(_save_figure(figure_dir / "tile_clone_class_contribution.png"))

    plt.figure(figsize=(7, 4))
    if not membership.empty:
        for clone_class, group in membership.groupby("clone_class"):
            group["support_score"].astype(float).plot(kind="kde", label=clone_class)
        plt.legend(fontsize=7)
    plt.xlabel("Support score")
    plt.title("Null-calibrated support score distributions")
    paths.append(_save_figure(figure_dir / "support_score_distributions.png"))

    plt.figure(figsize=(6, 4))
    if not null_calibration.empty:
        b = null_calibration.loc[null_calibration["clone_class"].eq(CLASS_B)]
        if not b.empty:
            pd.Series(
                {
                    "real Class B score rows": float(b["real_candidate_score_rows"].iloc[0]),
                    "max null expanded score rows": float(b["null_candidate_score_rows_max"].iloc[0]),
                }
            ).plot(kind="bar", color=["#d18c45", "#7678ed"])
    plt.ylabel("Candidate score rows")
    plt.title("Class B real vs null caution")
    paths.append(_save_figure(figure_dir / "class_b_caution_real_vs_null.png"))

    plt.figure(figsize=(8, 5))
    robust = membership.loc[membership["membership_status"].isin(SUPPORT_STATUSES)]
    if not robust.empty and not tile_summary.empty:
        top_clones = robust.groupby("clone_id")["membership_weight"].sum().sort_values(ascending=False).head(20).index
        heat = robust.loc[robust["clone_id"].isin(top_clones)].pivot_table(index="clone_id", columns="cell_key", values="membership_weight", aggfunc="sum", fill_value=0)
        if not heat.empty:
            plt.imshow(heat.iloc[:, : min(200, heat.shape[1])], aspect="auto", interpolation="nearest", cmap="viridis")
            plt.yticks(range(len(heat.index)), heat.index, fontsize=5)
    plt.title("Selected robust clone membership heatmap")
    paths.append(_save_figure(figure_dir / "selected_clone_membership_heatmap.png"))

    key_dir = ensure_dir(report_root / "key_figure_candidates")
    key_paths = []
    for path in paths:
        target = key_dir / path.name
        shutil.copy2(path, target)
        key_paths.append(target)
    payload = {
        "figure_count": len(paths),
        "key_figure_count": len(key_paths),
        "figures": [str(path) for path in paths],
        "key_figure_candidates": [str(path) for path in key_paths],
    }
    return paths, payload


def _save_figure(path: Path) -> Path:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def validate_round2_1_outputs(
    output_root: Path,
    report_root: Path,
    *,
    figures_required: bool,
    source_input_unchanged: bool = True,
) -> dict[str, Any]:
    json_ok = True
    for path in sorted(report_root.glob("*.json")):
        try:
            import json

            json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            json_ok = False
    table_paths = [
        output_root / "audit/signature_assignment_loss.tsv.gz",
        output_root / "membership/cellbin_clone_membership.tsv.gz",
        output_root / "membership/cellbin_clone_membership_summary.tsv.gz",
        output_root / "membership/membership_thresholds.tsv",
        output_root / "niche_membership/tile_clone_membership_summary.tsv.gz",
    ]
    tsv_ok = True
    for path in table_paths:
        try:
            if not path.exists():
                tsv_ok = False
            else:
                read_table(path, nrows=5)
        except Exception:
            tsv_ok = False
    matrix_ok = (output_root / "membership/clone_membership_sparse.npz").exists()
    if matrix_ok:
        try:
            from scipy import sparse

            sparse.load_npz(output_root / "membership/clone_membership_sparse.npz")
        except Exception:
            matrix_ok = False
    weak_not_assigned = True
    membership_path = output_root / "membership/cellbin_clone_membership.tsv.gz"
    if membership_path.exists():
        membership = read_table(membership_path, nrows=1000)
        weak_not_assigned = not membership["membership_status"].astype(str).str.contains("assigned", case=False, regex=False).any()
    figures = sorted((report_root / "figures").glob("*.png"))
    figures_ok = (not figures_required) or (bool(figures) and all(path.stat().st_size > 0 for path in figures))
    text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(report_root.glob("*.md")))
    claim_hits = positive_claim_hits(text)
    payload = {
        "validation_status": "PASS"
        if all([json_ok, tsv_ok, matrix_ok, weak_not_assigned, figures_ok, source_input_unchanged, not claim_hits])
        else "FAIL",
        "json_parse": bool(json_ok),
        "tsv_gzip_readability": bool(tsv_ok),
        "membership_matrix_readability": bool(matrix_ok),
        "no_weak_support_labeled_assigned_clone": bool(weak_not_assigned),
        "null_calibration_outputs_present": bool((output_root / "membership/membership_null_calibration.tsv").exists()),
        "tile_niche_membership_composition_present": bool((output_root / "niche_membership/tile_clone_membership_summary.tsv.gz").exists()),
        "figures_non_empty": bool(figures_ok),
        "source_input_packet_unchanged": bool(source_input_unchanged),
        "no_ssd": "/ssd/" not in text,
        "no_fastq": "processed raw fastq" not in text.lower(),
        "no_darlin_recalling": "darlin allele calling was rerun" not in text.lower(),
        "no_directed_gpcca": "directed gpcca was run" not in text.lower(),
        "no_plana_planb_production": "plana production was run" not in text.lower()
        and "planb production was run" not in text.lower(),
        "no_positive_fate_terminal_transition_claims": bool(not claim_hits),
        "positive_claim_hits": claim_hits,
    }
    return payload


def finite_fraction(frame: pd.DataFrame, numerator: str, denominator: str = "n_cellbins") -> float:
    if frame.empty or numerator not in frame or denominator not in frame:
        return 0.0
    return float(frame[numerator].sum() / max(float(frame[denominator].sum()), 1.0))
