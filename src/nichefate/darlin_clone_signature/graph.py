from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .common import CloneSignatureParams


def bridge_filtered_cell_keys(complexity: pd.DataFrame, mode: str) -> set[str]:
    if complexity.empty or mode == "none":
        return set()
    if mode == "p99":
        return set(complexity.loc[complexity["bridge_flag_p99"], "cell_key"].astype(str))
    if mode in {"p995", "p99.5"}:
        return set(complexity.loc[complexity["bridge_flag_p995"], "cell_key"].astype(str))
    raise ValueError(f"Unsupported bridge filter mode: {mode}")


def _feature_section_counts(valid_evidence: pd.DataFrame) -> dict[tuple[str, object], int]:
    grouped = (
        valid_evidence.groupby(["assay_scoped_feature_id", "section_order"])["cell_key"]
        .nunique()
        .reset_index()
    )
    return {
        (str(row["assay_scoped_feature_id"]), row["section_order"]): int(row["cell_key"])
        for row in grouped.to_dict(orient="records")
    }


def _section_cell_counts(valid_evidence: pd.DataFrame) -> dict[object, int]:
    grouped = valid_evidence.groupby("section_order")["cell_key"].nunique()
    return {section: int(count) for section, count in grouped.items()}


def _expected_shared(
    left: str,
    right: str,
    section_counts: dict[tuple[str, object], int],
    section_cell_counts: dict[object, int],
) -> float:
    expected = 0.0
    for section, n_cells in section_cell_counts.items():
        if n_cells <= 0:
            continue
        expected += section_counts.get((left, section), 0) * section_counts.get((right, section), 0) / n_cells
    return float(expected)


def build_feature_compatibility_graph(
    evidence: pd.DataFrame,
    feature_reference: pd.DataFrame,
    complexity: pd.DataFrame,
    params: CloneSignatureParams,
    *,
    bridge_filter_mode: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build sparse feature-feature compatibility edges from per-cell inverted lists."""

    mode = bridge_filter_mode or params.bridge_filter_mode
    valid_features = set(
        feature_reference.loc[feature_reference["valid_for_signature"], "assay_scoped_feature_id"].astype(str)
    )
    if not valid_features:
        empty_edges = pd.DataFrame()
        summary = pd.DataFrame([{"metric": "n_edges", "value": 0}])
        return empty_edges, summary, pd.DataFrame(), {"n_edges": 0, "bridge_filter_mode": mode}
    filtered_keys = bridge_filtered_cell_keys(complexity, mode)
    work = evidence.loc[
        evidence["valid_for_signature"].astype(bool)
        & evidence["assay_scoped_feature_id"].astype(str).isin(valid_features)
        & ~evidence["cell_key"].astype(str).isin(filtered_keys)
    ].copy()
    if work.empty:
        empty_edges = pd.DataFrame(
            columns=[
                "feature_left",
                "feature_right",
                "observed_shared_cellbins",
                "expected_shared_cellbins",
                "enrichment_score",
                "jaccard_overlap",
                "weighted_overlap",
                "bridge_dependency_score",
                "same_locus",
                "cross_locus",
                "compatible",
            ]
        )
        summary = pd.DataFrame([{"metric": "n_edges", "value": 0}])
        return empty_edges, summary, pd.DataFrame(), {"n_edges": 0, "bridge_filter_mode": mode}

    per_cell = (
        work.groupby(["cell_key", "section_order"])["assay_scoped_feature_id"]
        .agg(lambda s: sorted(set(s.astype(str))))
        .reset_index()
    )
    per_cell["n_valid_features"] = per_cell["assay_scoped_feature_id"].map(len)
    pair_events = int(((per_cell["n_valid_features"] * (per_cell["n_valid_features"] - 1)) // 2).sum())
    pair_counter: Counter[tuple[str, str]] = Counter()
    pair_sections: defaultdict[tuple[str, str], Counter] = defaultdict(Counter)
    pair_cell_examples: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for row in per_cell.to_dict(orient="records"):
        features = row["assay_scoped_feature_id"]
        if len(features) < 2:
            continue
        for left, right in combinations(features, 2):
            pair = (left, right) if left < right else (right, left)
            pair_counter[pair] += 1
            pair_sections[pair][row["section_order"]] += 1
            if len(pair_cell_examples[pair]) < 5:
                pair_cell_examples[pair].append(str(row["cell_key"]))

    if not pair_counter:
        edges = pd.DataFrame()
    else:
        ref = feature_reference.set_index("assay_scoped_feature_id")
        section_counts = _feature_section_counts(work)
        section_cell_counts = _section_cell_counts(work)
        rows: list[dict[str, Any]] = []
        for (left, right), observed in pair_counter.items():
            if observed < params.min_feature_cooccurrence_cellbins:
                continue
            left_row = ref.loc[left]
            right_row = ref.loc[right]
            if left_row["feature_class"] == "common_filtered" or right_row["feature_class"] == "common_filtered":
                continue
            expected = _expected_shared(left, right, section_counts, section_cell_counts)
            n_left = int(left_row["n_cellbins_detected"])
            n_right = int(right_row["n_cellbins_detected"])
            jaccard = observed / max(n_left + n_right - observed, 1)
            weight_left = float(left_row["empirical_rarity_weight"])
            weight_right = float(right_row["empirical_rarity_weight"])
            bridge_dependency = 1.0 / max(observed, 1)
            left_assay = str(left_row["assay"])
            right_assay = str(right_row["assay"])
            compatible = bool(
                observed >= params.min_feature_cooccurrence_cellbins
                and bridge_dependency <= params.max_bridge_dependency_score
            )
            rows.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "observed_shared_cellbins": int(observed),
                    "expected_shared_cellbins": float(expected),
                    "enrichment_score": float((observed + 1e-9) / (expected + 1e-9)),
                    "jaccard_overlap": float(jaccard),
                    "weighted_overlap": float(observed * (weight_left + weight_right) / 2.0),
                    "bridge_dependency_score": float(bridge_dependency),
                    "same_locus": bool(left_assay == right_assay),
                    "cross_locus": bool(left_assay != right_assay),
                    "left_assay": left_assay,
                    "right_assay": right_assay,
                    "shared_cellbin_examples": ";".join(pair_cell_examples[(left, right)]),
                    "compatible": compatible,
                }
            )
        edges = pd.DataFrame(rows)
    if edges.empty:
        edges = pd.DataFrame(
            columns=[
                "feature_left",
                "feature_right",
                "observed_shared_cellbins",
                "expected_shared_cellbins",
                "enrichment_score",
                "jaccard_overlap",
                "weighted_overlap",
                "bridge_dependency_score",
                "same_locus",
                "cross_locus",
                "left_assay",
                "right_assay",
                "shared_cellbin_examples",
                "compatible",
            ]
        )
    component_rows = _connected_components(edges.loc[edges["compatible"].astype(bool)] if not edges.empty else edges)
    candidates = pd.DataFrame(component_rows)
    summary_rows = [
        {"metric": "bridge_filter_mode", "value": mode},
        {"metric": "n_valid_features", "value": int(len(valid_features))},
        {"metric": "n_bridge_cellbins_filtered", "value": int(len(filtered_keys))},
        {"metric": "n_cells_used_for_pairing", "value": int(per_cell.shape[0])},
        {"metric": "estimated_pair_event_count_after_filter", "value": int(pair_events)},
        {"metric": "n_observed_pairs", "value": int(len(pair_counter))},
        {"metric": "n_compatible_edges", "value": int(edges["compatible"].sum()) if not edges.empty else 0},
        {"metric": "n_candidate_components", "value": int(candidates["component_id"].nunique()) if not candidates.empty else 0},
    ]
    payload = {
        "bridge_filter_mode": mode,
        "n_valid_features": int(len(valid_features)),
        "n_bridge_cellbins_filtered": int(len(filtered_keys)),
        "estimated_pair_event_count_after_filter": int(pair_events),
        "resource_warning": bool(pair_events > params.high_complexity_pair_warning),
        "n_observed_pairs": int(len(pair_counter)),
        "n_compatible_edges": int(edges["compatible"].sum()) if not edges.empty else 0,
    }
    return edges.sort_values(["observed_shared_cellbins", "feature_left", "feature_right"], ascending=[False, True, True]), pd.DataFrame(summary_rows), candidates, payload


def _connected_components(edges: pd.DataFrame) -> list[dict[str, Any]]:
    if edges.empty:
        return []
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for row in edges[["feature_left", "feature_right"]].itertuples(index=False):
        union(str(row.feature_left), str(row.feature_right))
    components: defaultdict[str, list[str]] = defaultdict(list)
    for node in sorted(parent):
        components[find(node)].append(node)
    rows = []
    for idx, features in enumerate(sorted(components.values(), key=lambda values: (-len(values), values[0])), start=1):
        component_id = f"candidate_component_{idx:06d}"
        assays = sorted({feature.split("::", 1)[0] for feature in features})
        rows.append(
            {
                "component_id": component_id,
                "n_features": int(len(features)),
                "n_loci": int(len(assays)),
                "loci_present": ";".join(assays),
                "features": ";".join(features),
            }
        )
    return rows
