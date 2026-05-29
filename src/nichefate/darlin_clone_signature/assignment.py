from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import CELL_COLUMNS, CloneSignatureParams, make_cell_key, summarize_top_items


def _membership_for_set(membership: pd.DataFrame, clone_set: str) -> pd.DataFrame:
    if membership.empty:
        return membership.copy()
    if clone_set == "high_confidence":
        return membership.loc[membership["clone_set_high_confidence"].astype(bool)].copy()
    if clone_set == "expanded":
        return membership.loc[membership["clone_set_expanded"].astype(bool)].copy()
    raise ValueError(f"Unsupported clone set: {clone_set}")


def candidate_clone_scores(
    evidence: pd.DataFrame,
    signatures: pd.DataFrame,
    membership: pd.DataFrame,
    params: CloneSignatureParams,
    *,
    clone_set: str = "expanded",
) -> pd.DataFrame:
    """Compute unthresholded clone scores for cellbin-clone candidates."""

    members = _membership_for_set(membership, clone_set)
    if evidence.empty or signatures.empty or members.empty:
        return pd.DataFrame(
            columns=[
                "cell_key",
                "clone_id",
                "clone_class",
                "score_raw",
                "n_supporting_features",
                "n_supporting_loci",
                "supporting_features",
            ]
        )
    work = evidence.loc[evidence["valid_for_signature"].astype(bool)].copy()
    joined = work.merge(
        members[["clone_id", "clone_class", "assay_scoped_feature_id"]],
        on="assay_scoped_feature_id",
        how="inner",
    )
    if joined.empty:
        return pd.DataFrame()
    joined["weighted_count_score"] = joined["empirical_rarity_weight"].astype(float) * np.log1p(joined["count"].astype(float))
    grouped = (
        joined.groupby(["cell_key", "clone_id", "clone_class"], as_index=False)
        .agg(
            score_feature=("weighted_count_score", "sum"),
            n_supporting_features=("assay_scoped_feature_id", "nunique"),
            n_supporting_loci=("assay", "nunique"),
            supporting_features=("assay_scoped_feature_id", lambda s: ";".join(sorted(set(s.astype(str))))),
        )
    )
    sig_lookup = signatures.set_index("clone_id")[["n_features", "n_loci", "bridge_dependency_score"]]
    grouped = grouped.merge(sig_lookup, on="clone_id", how="left")
    grouped["locus_support_bonus"] = 0.75 * grouped["n_supporting_loci"].astype(float)
    grouped["signature_completeness_bonus"] = (
        grouped["n_supporting_features"].astype(float) / grouped["n_features"].replace(0, np.nan).astype(float)
    ).fillna(0.0)
    grouped["bridge_ambiguity_penalty"] = grouped["bridge_dependency_score"].fillna(0.0).clip(0, 1) * 0.25
    grouped["common_feature_penalty"] = 0.0
    grouped["score_raw"] = (
        grouped["score_feature"]
        + grouped["locus_support_bonus"]
        + grouped["signature_completeness_bonus"]
        - grouped["common_feature_penalty"]
        - grouped["bridge_ambiguity_penalty"]
    )
    return grouped.sort_values(["cell_key", "score_raw", "clone_id"], ascending=[True, False, True]).reset_index(drop=True)


def score_topk(scores: pd.DataFrame, params: CloneSignatureParams) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(
            columns=[
                "cell_key",
                "clone_id",
                "clone_class",
                "rank",
                "score",
                "n_supporting_features",
                "n_supporting_loci",
                "supporting_features",
            ]
        )
    out = scores.copy()
    out["rank"] = out.groupby("cell_key")["score_raw"].rank(method="first", ascending=False).astype(int)
    out = out.loc[out["rank"].le(params.topk_scores)].copy()
    out = out.rename(columns={"score_raw": "score"})
    return out[
        [
            "cell_key",
            "clone_id",
            "clone_class",
            "rank",
            "score",
            "n_supporting_features",
            "n_supporting_loci",
            "supporting_features",
        ]
    ].sort_values(["cell_key", "rank"])


def calibrate_assignment_thresholds(
    null_score_tables: list[pd.DataFrame],
    real_scores: pd.DataFrame,
    *,
    clone_set: str,
) -> dict[str, Any]:
    null_best_values = []
    null_margin_values = []
    for table in null_score_tables:
        if table.empty:
            continue
        top = score_topk(table, CloneSignatureParams(topk_scores=2))
        if top.empty:
            continue
        pivot = top.pivot_table(index="cell_key", columns="rank", values="score", aggfunc="first").fillna(0.0)
        if 1 in pivot:
            null_best_values.extend(pivot[1].astype(float).tolist())
        if 1 in pivot:
            second = pivot[2] if 2 in pivot else 0.0
            null_margin_values.extend((pivot[1] - second).astype(float).tolist())
    real_top = score_topk(real_scores, CloneSignatureParams(topk_scores=2))
    real_positive = real_top.loc[real_top["rank"].eq(1), "score"].astype(float)
    if null_best_values:
        min_score = float(np.quantile(null_best_values, 0.99))
        source = "null_q99_best_score"
    elif len(real_positive):
        min_score = float(max(0.0, real_positive.quantile(0.05)))
        source = "real_low_tail_fallback_no_null_candidate_scores"
    else:
        min_score = 0.0
        source = "no_candidate_scores"
    if null_margin_values:
        min_margin = float(np.quantile(null_margin_values, 0.95))
        margin_source = "null_q95_score_margin"
    elif len(real_positive):
        min_margin = float(max(0.0, real_positive.quantile(0.01) * 0.10))
        margin_source = "real_low_tail_margin_fallback_no_null_candidate_scores"
    else:
        min_margin = 0.0
        margin_source = "no_candidate_scores"
    return {
        "clone_set": clone_set,
        "min_assignment_score": min_score,
        "min_score_margin": min_margin,
        "score_threshold_source": source,
        "margin_threshold_source": margin_source,
        "n_null_candidate_scores": int(len(null_best_values)),
        "n_real_candidate_scores": int(len(real_positive)),
        "null_best_score_q99": float(np.quantile(null_best_values, 0.99)) if null_best_values else 0.0,
        "null_margin_q95": float(np.quantile(null_margin_values, 0.95)) if null_margin_values else 0.0,
    }


def _status_for_unscored_cell(row: pd.Series) -> str:
    if bool(row.get("has_lineage_evidence", False)) and int(row.get("n_valid_signature_features", 0)) == 0:
        return "filtered"
    return "unassigned"


def assign_cellbins_to_clones(
    full_cellbins: pd.DataFrame,
    evidence: pd.DataFrame,
    scores: pd.DataFrame,
    thresholds: dict[str, Any],
    params: CloneSignatureParams,
    *,
    clone_set: str = "expanded",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Assign primary clone status and weighted multi-clone memberships."""

    cellbase = full_cellbins[CELL_COLUMNS].drop_duplicates().copy()
    cellbase["cell_key"] = make_cell_key(cellbase)
    evidence_summary = (
        evidence.groupby("cell_key", as_index=False)
        .agg(
            has_lineage_evidence=("assay_scoped_feature_id", "size"),
            n_valid_signature_features=("valid_for_signature", "sum"),
        )
    )
    evidence_summary["has_lineage_evidence"] = evidence_summary["has_lineage_evidence"].gt(0)
    cellbase = cellbase.merge(evidence_summary, on="cell_key", how="left")
    cellbase["has_lineage_evidence"] = cellbase["has_lineage_evidence"].fillna(False).astype(bool)
    cellbase["n_valid_signature_features"] = cellbase["n_valid_signature_features"].fillna(0).astype(int)
    topk = score_topk(scores, params)
    min_score = float(thresholds.get("min_assignment_score", 0.0))
    min_margin = float(thresholds.get("min_score_margin", 0.0))
    top1 = topk.loc[topk["rank"].eq(1)].copy()
    top2 = topk.loc[topk["rank"].eq(2), ["cell_key", "score", "supporting_features"]].rename(
        columns={"score": "second_best_score", "supporting_features": "second_supporting_features"}
    )
    assignment = cellbase.merge(top1.drop(columns=["rank"], errors="ignore"), on="cell_key", how="left")
    assignment = assignment.merge(top2, on="cell_key", how="left")
    assignment["clone_set"] = clone_set
    assignment["score"] = assignment["score"].fillna(0.0)
    assignment["second_best_score"] = assignment["second_best_score"].fillna(0.0)
    assignment["score_margin"] = assignment["score"] - assignment["second_best_score"]
    assignment["clone_id"] = assignment["clone_id"].fillna("")
    assignment["clone_class"] = assignment["clone_class"].fillna("")
    assignment["supporting_features"] = assignment["supporting_features"].fillna("")
    assignment["second_supporting_features"] = assignment["second_supporting_features"].fillna("")
    assignment["n_supporting_features"] = assignment["n_supporting_features"].fillna(0).astype(int)
    assignment["n_supporting_loci"] = assignment["n_supporting_loci"].fillna(0).astype(int)

    has_score = assignment["score"].gt(0)
    passes = assignment["score"].ge(min_score) & has_score
    default_filtered = assignment["has_lineage_evidence"] & assignment["n_valid_signature_features"].eq(0)
    assignment["assignment_status"] = np.where(default_filtered, "filtered", "unassigned")
    assignment["reason_if_not_assigned"] = np.where(default_filtered, "only_filtered_or_common_features", "no_valid_clone_signature_evidence")
    assignment.loc[has_score & ~passes, "reason_if_not_assigned"] = "best_score_below_null_calibrated_threshold"

    multi_candidate = passes & assignment["second_best_score"].ge(min_score) & assignment["second_best_score"].ge(assignment["score"] * params.membership_ratio)
    overlap_values = pd.Series(0.0, index=assignment.index)
    for idx, row in assignment.loc[multi_candidate, ["supporting_features", "second_supporting_features"]].iterrows():
        left = set(item for item in str(row["supporting_features"]).split(";") if item)
        right = set(item for item in str(row["second_supporting_features"]).split(";") if item)
        overlap_values.loc[idx] = len(left & right) / max(len(left | right), 1)
    overlap_ambiguous = multi_candidate & overlap_values.ge(params.shared_support_overlap_ambiguity)
    margin_ambiguous = passes & assignment["second_best_score"].gt(0) & assignment["score_margin"].lt(min_margin)
    assigned_multi = multi_candidate & ~overlap_ambiguous & ~margin_ambiguous
    ambiguous = passes & ~assigned_multi & (overlap_ambiguous | margin_ambiguous)
    assigned_single = passes & ~assigned_multi & ~ambiguous
    assignment.loc[assigned_single, "assignment_status"] = "assigned_single"
    assignment.loc[assigned_multi, "assignment_status"] = "assigned_multi"
    assignment.loc[ambiguous, "assignment_status"] = "ambiguous"
    assignment.loc[assigned_single | assigned_multi, "reason_if_not_assigned"] = ""
    assignment.loc[ambiguous & margin_ambiguous, "reason_if_not_assigned"] = "score_margin_below_null_calibrated_threshold"
    assignment.loc[ambiguous & overlap_ambiguous, "reason_if_not_assigned"] = "top_clone_scores_share_supporting_features"
    assignment.loc[~passes, ["clone_id", "clone_class", "supporting_features"]] = ""
    assignment.loc[~passes, ["n_supporting_features", "n_supporting_loci"]] = 0
    assignment = assignment.rename(columns={"score": "assignment_score"})

    single_members = assignment.loc[
        assignment["assignment_status"].eq("assigned_single"),
        [*CELL_COLUMNS, "cell_key", "clone_set", "clone_id", "clone_class", "assignment_score", "n_supporting_features", "n_supporting_loci"],
    ].copy()
    single_members["membership_weight"] = 1.0
    multi_cells = assignment.loc[assignment["assignment_status"].eq("assigned_multi"), ["cell_key", "assignment_score"]].rename(
        columns={"assignment_score": "best_score"}
    )
    if not multi_cells.empty:
        multi_members = topk.merge(multi_cells, on="cell_key", how="inner")
        multi_members = multi_members.loc[
            multi_members["score"].ge(min_score)
            & multi_members["score"].ge(multi_members["best_score"] * params.membership_ratio)
        ].copy()
        multi_members = multi_members.merge(cellbase[CELL_COLUMNS + ["cell_key"]], on="cell_key", how="left")
        denom = multi_members.groupby("cell_key")["score"].transform("sum").replace(0, np.nan)
        multi_members["membership_weight"] = (multi_members["score"] / denom).fillna(0.0)
        multi_members["clone_set"] = clone_set
        multi_members = multi_members.rename(columns={"score": "assignment_score"})
        multi_members = multi_members[[*CELL_COLUMNS, "cell_key", "clone_set", "clone_id", "clone_class", "membership_weight", "assignment_score", "n_supporting_features", "n_supporting_loci"]]
    else:
        multi_members = pd.DataFrame(columns=[*CELL_COLUMNS, "cell_key", "clone_set", "clone_id", "clone_class", "membership_weight", "assignment_score", "n_supporting_features", "n_supporting_loci"])
    membership = pd.concat(
        [
            single_members[[*CELL_COLUMNS, "cell_key", "clone_set", "clone_id", "clone_class", "membership_weight", "assignment_score", "n_supporting_features", "n_supporting_loci"]],
            multi_members,
        ],
        ignore_index=True,
    )
    assignment = assignment[
        [
            *CELL_COLUMNS,
            "cell_key",
            "clone_set",
            "clone_id",
            "clone_class",
            "assignment_status",
            "assignment_score",
            "second_best_score",
            "score_margin",
            "n_supporting_features",
            "n_supporting_loci",
            "supporting_features",
            "reason_if_not_assigned",
        ]
    ]
    matrix = membership[
        ["cell_key", "clone_id", "clone_set", "membership_weight", "assignment_score"]
    ].copy() if not membership.empty else pd.DataFrame(columns=["cell_key", "clone_id", "clone_set", "membership_weight", "assignment_score"])
    assigned_mask = assignment["assignment_status"].isin(["assigned_single", "assigned_multi"])
    summary = {
        "clone_set": clone_set,
        "n_cellbins": int(len(assignment)),
        "n_assigned_cellbins": int(assigned_mask.sum()),
        "assigned_cellbin_fraction": float(assigned_mask.mean()) if len(assignment) else 0.0,
        "n_assigned_single": int(assignment["assignment_status"].eq("assigned_single").sum()),
        "n_assigned_multi": int(assignment["assignment_status"].eq("assigned_multi").sum()),
        "n_ambiguous": int(assignment["assignment_status"].eq("ambiguous").sum()),
        "ambiguous_fraction": float(assignment["assignment_status"].eq("ambiguous").mean()) if len(assignment) else 0.0,
        "n_unassigned": int(assignment["assignment_status"].eq("unassigned").sum()),
        "n_filtered": int(assignment["assignment_status"].eq("filtered").sum()),
        "largest_clone_weighted_cellbins": float(membership.groupby("clone_id")["membership_weight"].sum().max()) if not membership.empty else 0.0,
        "status_distribution": assignment["assignment_status"].value_counts().to_dict(),
        "top_clone_weighted_memberships": summarize_top_items(membership.groupby("clone_id")["membership_weight"].sum().sort_values(ascending=False)) if not membership.empty else "",
        **thresholds,
    }
    return assignment, topk, membership, matrix, summary
