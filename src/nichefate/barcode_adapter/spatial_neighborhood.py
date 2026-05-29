from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


GROUP_TYPE = "BOUNDED_SPATIAL_NEIGHBORHOOD_SMOKE_NOT_FULL_M1"


def spatially_stratified_subset(
    cellbins: pd.DataFrame,
    *,
    max_cellbins: int,
    seed: int = 271828,
    bins: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(cellbins) <= max_cellbins:
        subset = cellbins.copy()
        return subset.reset_index(drop=True), {
            "method": "all_cellbins",
            "requested": int(max_cellbins),
            "selected": int(len(subset)),
        }
    rng = np.random.default_rng(seed)
    frame = cellbins.copy()
    frame["_x_bin"] = pd.qcut(frame["x"].rank(method="first"), q=bins, labels=False, duplicates="drop")
    frame["_y_bin"] = pd.qcut(frame["y"].rank(method="first"), q=bins, labels=False, duplicates="drop")
    selected = []
    groups = list(frame.groupby(["_x_bin", "_y_bin"], sort=True, observed=True))
    base_quota = max(1, int(np.floor(max_cellbins / max(1, len(groups)))))
    for _, group in groups:
        take = min(base_quota, len(group))
        if take:
            selected.extend(rng.choice(group.index.to_numpy(), size=take, replace=False).tolist())
    if len(selected) < max_cellbins:
        remaining = frame.index.difference(pd.Index(selected)).to_numpy()
        take = min(max_cellbins - len(selected), len(remaining))
        selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
    subset = frame.loc[selected].drop(columns=["_x_bin", "_y_bin"]).copy()
    subset = subset.sort_values(["sample_id", "slice_id", "cellbin_id"]).reset_index(drop=True)
    return subset, {
        "method": "xy_quantile_stratified",
        "requested": int(max_cellbins),
        "selected": int(len(subset)),
        "seed": int(seed),
        "bins": int(bins),
    }


def build_spatial_neighborhood_groups(
    cellbins: pd.DataFrame,
    *,
    k_neighbors: int = 16,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if cellbins.empty:
        raise ValueError("Cannot build spatial groups from an empty cellbin table")
    all_rows = []
    for (_, _), section in cellbins.groupby(["sample_id", "slice_id"], sort=False, observed=True):
        coords = section[["x", "y"]].to_numpy(dtype=float)
        k = min(int(k_neighbors), len(section))
        tree = cKDTree(coords)
        distances, indices = tree.query(coords, k=k)
        if k == 1:
            indices = indices[:, None]
            distances = distances[:, None]
        section_rows = section.reset_index(drop=True)
        for anchor_pos, neighbor_positions in enumerate(indices):
            anchor = section_rows.iloc[int(anchor_pos)]
            group_id = f"{anchor.sample_id}__anchor__{anchor.cellbin_id}"
            for member_pos in neighbor_positions:
                member = section_rows.iloc[int(member_pos)]
                role = "center" if int(member_pos) == int(anchor_pos) else "member"
                all_rows.append(
                    {
                        "sample_id": str(anchor.sample_id),
                        "slice_id": str(anchor.slice_id),
                        "section_order": int(anchor.section_order),
                        "group_id": group_id,
                        "group_type": GROUP_TYPE,
                        "niche_id": group_id,
                        "anchor_cellbin_id": str(anchor.cellbin_id),
                        "anchor_x": float(anchor.x),
                        "anchor_y": float(anchor.y),
                        "cellbin_id": str(member.cellbin_id),
                        "x": float(member.x),
                        "y": float(member.y),
                        "role": role,
                    }
                )
    assignment = pd.DataFrame(all_rows)
    payload = {
        "group_type": GROUP_TYPE,
        "k_neighbors_requested": int(k_neighbors),
        "groups": int(assignment["group_id"].nunique()),
        "rows": int(len(assignment)),
        "overlapping_neighborhoods": True,
        "interpretation": "local-context summaries only; do not sum across groups as tissue-level abundance",
    }
    return assignment, payload


def group_membership_multiplicity(group_assignment: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = (
        group_assignment.groupby(["sample_id", "slice_id", "cellbin_id"], as_index=False)
        .agg(groups_per_member_cellbin=("group_id", "nunique"))
        .sort_values(["sample_id", "slice_id", "cellbin_id"])
    )
    anchors = group_assignment.loc[group_assignment["role"].astype(str) == "center"]
    anchors_per_group = anchors.groupby("group_id")["anchor_cellbin_id"].nunique()
    original_cellbins = set(group_assignment["cellbin_id"].astype(str))
    anchor_traceable = anchors["anchor_cellbin_id"].astype(str).isin(original_cellbins)
    payload = {
        "mean_groups_per_member_cellbin": float(counts["groups_per_member_cellbin"].mean()),
        "median_groups_per_member_cellbin": float(counts["groups_per_member_cellbin"].median()),
        "max_groups_per_member_cellbin": int(counts["groups_per_member_cellbin"].max()),
        "anchor_cellbin_count": int(anchors["anchor_cellbin_id"].nunique()),
        "member_cellbin_count": int(counts["cellbin_id"].nunique()),
        "group_count": int(group_assignment["group_id"].nunique()),
        "every_group_has_one_anchor": bool((anchors_per_group == 1).all()),
        "every_anchor_cellbin_traceable_to_h5ad_obs_cellbin_id": bool(anchor_traceable.all()),
        "overlapping_neighborhoods": True,
    }
    return counts, payload
