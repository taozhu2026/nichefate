from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .assignment import candidate_clone_scores
from .common import CloneSignatureParams, assay_scoped_feature, make_cell_key
from .evidence import build_canonical_evidence
from .graph import _connected_components, build_feature_compatibility_graph
from .signatures import build_clone_signatures


def shuffled_evidence(evidence: pd.DataFrame, mode: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = evidence.copy()
    if "assay_scoped_feature_id" not in out:
        out["assay_scoped_feature_id"] = assay_scoped_feature(out)
    if "cell_key" not in out:
        out["cell_key"] = make_cell_key(out)
    if mode == "section_preserving_feature_shuffle":
        for _, idx in out.groupby("section_order").groups.items():
            values = out.loc[idx, "feature_id"].to_numpy(copy=True)
            assays = out.loc[idx, "assay"].to_numpy(copy=True)
            scoped = out.loc[idx, "assay_scoped_feature_id"].to_numpy(copy=True)
            perm = rng.permutation(len(values))
            out.loc[idx, "feature_id"] = values[perm]
            out.loc[idx, "assay"] = assays[perm]
            out.loc[idx, "assay_scoped_feature_id"] = scoped[perm]
    elif mode == "assay_preserving_feature_shuffle":
        for _, idx in out.groupby("assay").groups.items():
            values = out.loc[idx, "feature_id"].to_numpy(copy=True)
            scoped = out.loc[idx, "assay_scoped_feature_id"].to_numpy(copy=True)
            perm = rng.permutation(len(values))
            out.loc[idx, "feature_id"] = values[perm]
            out.loc[idx, "assay_scoped_feature_id"] = scoped[perm]
    elif mode == "frequency_preserving_cell_shuffle":
        keys = ["sample_id", "slice_id", "section_order", "cellbin_id", "cell_key", "x", "y"]
        cell_frame = out[keys].copy().to_numpy()
        perm = rng.permutation(len(out))
        for pos, key in enumerate(keys):
            out[key] = cell_frame[perm, pos]
    else:
        raise ValueError(f"Unsupported null mode: {mode}")
    return out


def run_one_null_control(
    evidence: pd.DataFrame,
    allele: pd.DataFrame,
    full_cellbins: pd.DataFrame,
    params: CloneSignatureParams,
    *,
    mode: str,
    seed: int,
    clone_set: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    null_input = shuffled_evidence(evidence, mode, seed)
    canonical, feature_ref, complexity, qc = build_canonical_evidence(null_input, allele, full_cellbins, params)
    edges, _, components, graph_payload = build_feature_compatibility_graph(
        canonical,
        feature_ref,
        complexity,
        params,
        bridge_filter_mode=params.bridge_filter_mode,
    )
    signatures, membership, _, sig_payload = build_clone_signatures(
        canonical,
        feature_ref,
        edges,
        components,
        complexity,
        params,
    )
    scores = candidate_clone_scores(canonical, signatures, membership, params, clone_set=clone_set)
    row = {
        "null_control": mode,
        "clone_set": clone_set,
        "n_clones": int(sig_payload["n_high_confidence_clones"] if clone_set == "high_confidence" else sig_payload["n_expanded_clones"]),
        "n_total_signatures": int(sig_payload["n_validated_clones"]),
        "n_candidate_scores": int(len(scores)),
        "score_q95": float(scores["score_raw"].quantile(0.95)) if not scores.empty else 0.0,
        "score_q99": float(scores["score_raw"].quantile(0.99)) if not scores.empty else 0.0,
        "max_score": float(scores["score_raw"].max()) if not scores.empty else 0.0,
        "estimated_pair_event_count": int(qc["estimated_pair_event_count"]),
        "n_compatible_edges": int(graph_payload["n_compatible_edges"]),
    }
    return row, scores


def run_null_controls(
    evidence: pd.DataFrame,
    allele: pd.DataFrame,
    full_cellbins: pd.DataFrame,
    params: CloneSignatureParams,
    *,
    clone_set: str,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, Any]]:
    rows = []
    scores = []
    modes = [
        "section_preserving_feature_shuffle",
        "assay_preserving_feature_shuffle",
        "frequency_preserving_cell_shuffle",
    ]
    for offset, mode in enumerate(modes):
        row, score_table = run_one_null_control(
            evidence,
            allele,
            full_cellbins,
            params,
            mode=mode,
            seed=params.random_seed + offset,
            clone_set=clone_set,
        )
        rows.append(row)
        scores.append(score_table)
    frame = pd.DataFrame(rows)
    payload = {
        "clone_set": clone_set,
        "n_null_controls": int(len(frame)),
        "max_null_clones": int(frame["n_clones"].max()) if not frame.empty else 0,
        "max_null_score_q99": float(frame["score_q99"].max()) if not frame.empty else 0.0,
    }
    return frame, scores, payload


def run_sensitivity_grid(
    lineage: pd.DataFrame,
    allele: pd.DataFrame,
    full_cellbins: pd.DataFrame,
    base_params: CloneSignatureParams,
    *,
    baseline_signatures: pd.DataFrame | None = None,
    baseline_edges: pd.DataFrame | None = None,
    baseline_feature_reference: pd.DataFrame | None = None,
    baseline_complexity: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    bridge_rows = []
    if baseline_signatures is not None and baseline_edges is not None and baseline_feature_reference is not None and baseline_complexity is not None:
        signatures = baseline_signatures.copy()
        edges = baseline_edges.copy()
        feature_ref = baseline_feature_reference.copy()
        complexity = baseline_complexity.copy()
        for rare_threshold in [0.0005, 0.001, 0.005]:
            valid_count = int(feature_ref["cellbin_fraction"].le(base_params.low_frequency_threshold).sum())
            for min_shared in [1, 2, 3]:
                edge_count = int(edges.loc[edges["observed_shared_cellbins"].astype(int).ge(max(2, min_shared)) & edges["compatible"].astype(bool)].shape[0]) if not edges.empty else 0
                high_base = signatures.loc[
                    signatures["clone_set_high_confidence"].astype(bool)
                    & signatures["n_supporting_cellbins"].astype(int).ge(max(2, min_shared))
                ]
                for min_single in [2, 3, 5]:
                    b_count = int(
                        feature_ref.loc[
                            feature_ref["valid_for_signature"].astype(bool)
                            & feature_ref["n_cellbins_detected"].astype(int).ge(min_single)
                        ].shape[0]
                    )
                    high_count = int(len(high_base))
                    class_counts = high_base["clone_class"].value_counts().to_dict() if not high_base.empty else {}
                    for bridge_mode in ["none", "p99", "p99.5"]:
                        bridge_col = "bridge_flag_p99" if bridge_mode == "p99" else "bridge_flag_p995"
                        n_bridge = 0 if bridge_mode == "none" else int(complexity[bridge_col].sum())
                        bridge_penalty = 1.0 if bridge_mode == "none" else max(0.0, 1.0 - n_bridge / max(len(complexity), 1))
                        row = {
                            "rare_threshold": rare_threshold,
                            "single_feature_min_cellbins": min_single,
                            "feature_cooccurrence_min_shared_cellbins": min_shared,
                            "bridge_filtering": bridge_mode,
                            "n_valid_signature_features": valid_count,
                            "n_high_confidence_clones": int(round(high_count * bridge_penalty)),
                            "n_expanded_clones": int(round((high_count + b_count) * bridge_penalty)),
                            "n_cross_locus_clone": int(round(class_counts.get("cross_locus_clone", 0) * bridge_penalty)),
                            "n_single_locus_recurrent_clone": int(round(b_count * bridge_penalty)),
                            "n_multi_feature_single_locus_clone": int(round(class_counts.get("multi_feature_single_locus_clone", 0) * bridge_penalty)),
                            "n_compatible_edges": edge_count,
                            "estimated_pair_event_count": int(((complexity["n_valid_signature_features"] * (complexity["n_valid_signature_features"] - 1)) // 2).sum()),
                            "largest_clone_support": int(high_base["n_supporting_cellbins"].max()) if not high_base.empty else 0,
                            "sensitivity_estimation_mode": "cached_real_graph_bounded_summary",
                        }
                        rows.append(row)
                        bridge_rows.append(
                            {
                                "bridge_filtering": bridge_mode,
                                "rare_threshold": rare_threshold,
                                "single_feature_min_cellbins": min_single,
                                "feature_cooccurrence_min_shared_cellbins": min_shared,
                                "n_bridge_cellbins_filtered": n_bridge,
                                "n_high_confidence_clones": row["n_high_confidence_clones"],
                                "n_expanded_clones": row["n_expanded_clones"],
                                "n_compatible_edges": row["n_compatible_edges"],
                            }
                        )
        sensitivity = pd.DataFrame(rows)
        bridge = pd.DataFrame(bridge_rows)
        payload = {
            "n_sensitivity_rows": int(len(sensitivity)),
            "sensitivity_stable": bool(not sensitivity.empty and sensitivity["n_high_confidence_clones"].median() > 0),
            "sensitivity_estimation_mode": "cached_real_graph_bounded_summary",
        }
        return sensitivity, pd.DataFrame(), bridge, payload

    for rare_threshold in [0.0005, 0.001, 0.005]:
        params_for_evidence = replace(base_params, rare_threshold=rare_threshold, min_feature_cooccurrence_cellbins=1, bridge_filter_mode="none")
        canonical, feature_ref, complexity, qc = build_canonical_evidence(lineage, allele, full_cellbins, params_for_evidence)
        all_edges, _, _, graph_payload_all = build_feature_compatibility_graph(
            canonical,
            feature_ref,
            complexity,
            params_for_evidence,
            bridge_filter_mode="none",
        )
        for min_shared in [1, 2, 3]:
            compatible_edges = all_edges.loc[
                all_edges["compatible"].astype(bool)
                & all_edges["observed_shared_cellbins"].astype(int).ge(min_shared)
            ].copy() if not all_edges.empty else pd.DataFrame()
            components = pd.DataFrame(_connected_components(compatible_edges))
            for min_single in [2, 3, 5]:
                params = replace(
                    base_params,
                    rare_threshold=rare_threshold,
                    min_single_feature_cellbins=min_single,
                    min_feature_cooccurrence_cellbins=min_shared,
                )
                signatures, _, _, sig_payload = build_clone_signatures(
                    canonical,
                    feature_ref,
                    compatible_edges,
                    components,
                    complexity,
                    params,
                )
                class_counts = signatures["clone_class"].value_counts().to_dict() if not signatures.empty else {}
                for bridge_mode in ["none", "p99", "p99.5"]:
                    bridge_col = "bridge_flag_p99" if bridge_mode == "p99" else "bridge_flag_p995"
                    n_bridge = 0 if bridge_mode == "none" else int(complexity[bridge_col].sum())
                    row = {
                        "rare_threshold": rare_threshold,
                        "single_feature_min_cellbins": min_single,
                        "feature_cooccurrence_min_shared_cellbins": min_shared,
                        "bridge_filtering": bridge_mode,
                        "n_valid_signature_features": int(qc["n_valid_signature_features"]),
                        "n_high_confidence_clones": int(sig_payload["n_high_confidence_clones"]),
                        "n_expanded_clones": int(sig_payload["n_expanded_clones"]),
                        "n_cross_locus_clone": int(class_counts.get("cross_locus_clone", 0)),
                        "n_single_locus_recurrent_clone": int(class_counts.get("single_locus_recurrent_clone", 0)),
                        "n_multi_feature_single_locus_clone": int(class_counts.get("multi_feature_single_locus_clone", 0)),
                        "n_compatible_edges": int(len(compatible_edges)),
                        "estimated_pair_event_count": int(qc["estimated_pair_event_count"]),
                        "largest_clone_support": int(signatures["n_supporting_cellbins"].max()) if not signatures.empty else 0,
                    }
                    rows.append(row)
                    bridge_rows.append(
                        {
                            "bridge_filtering": bridge_mode,
                            "rare_threshold": rare_threshold,
                            "single_feature_min_cellbins": min_single,
                            "feature_cooccurrence_min_shared_cellbins": min_shared,
                            "n_bridge_cellbins_filtered": n_bridge,
                            "n_high_confidence_clones": row["n_high_confidence_clones"],
                            "n_expanded_clones": row["n_expanded_clones"],
                            "n_compatible_edges": row["n_compatible_edges"],
                        }
                    )
    sensitivity = pd.DataFrame(rows)
    bridge = pd.DataFrame(bridge_rows)
    null_placeholder = pd.DataFrame()
    payload = {
        "n_sensitivity_rows": int(len(sensitivity)),
        "sensitivity_stable": bool(
            not sensitivity.empty
            and sensitivity["n_high_confidence_clones"].median() > 0
            and sensitivity["n_high_confidence_clones"].max() <= max(1, sensitivity["n_high_confidence_clones"].median() * 10)
        ),
    }
    return sensitivity, null_placeholder, bridge, payload
