from pathlib import Path

import numpy as np
import pandas as pd

from nichefate.transition import (
    build_candidate_neighbors,
    combine_scaled_evidence,
    evidence_schema_columns,
    infer_adjacent_time_pairs,
    pair_adaptive_temperature,
    resolve_transition_feature_groups,
    row_normalize_weights,
    safe_scale_vector,
)


def test_adjacent_time_pair_inference_uses_metadata_days() -> None:
    metadata = pd.DataFrame(
        {
            "time": ["early", "mid", "late"],
            "time_day": [2, 7, 19],
            "slice_id": ["s1", "s2", "s3"],
            "mouse_id": ["m1", "m2", "m3"],
            "rows": [10, 20, 30],
        }
    )

    pairs = infer_adjacent_time_pairs(metadata)

    assert [(p["source_time"], p["target_time"]) for p in pairs] == [
        ("early", "mid"),
        ("mid", "late"),
    ]
    assert [p["time_delta"] for p in pairs] == [5.0, 12.0]


def test_feature_group_selection_separates_retrieval_and_rerank_columns() -> None:
    schema = {
        "numeric_feature_columns": [
            "r__emb_mean_pc001",
            "r__ct_l1__a",
            "r__ct_l2__excluded",
            "r__ct_l3__c",
            "r__ct_l1_entropy",
            "r__mean_neighbor_distance",
        ]
    }
    config = {
        "molecular_state": {"include_patterns": ["__emb_mean_pc"]},
        "cell_type_composition": {
            "include_patterns": ["__ct_l1__", "__ct_l3__"],
            "exclude_patterns": ["__ct_l2__"],
        },
        "entropy": {"include_patterns": ["__ct_l1_entropy"]},
        "spatial_summary": {"include_patterns": ["__mean_neighbor_distance"]},
    }

    groups = resolve_transition_feature_groups(schema, config)

    retrieval = (
        groups["molecular_state"]
        + groups["cell_type_composition"]
        + groups["entropy"]
    )
    assert "r__emb_mean_pc001" in retrieval
    assert "r__ct_l1__a" in retrieval
    assert "r__ct_l3__c" in retrieval
    assert "r__ct_l2__excluded" not in retrieval
    assert "r__mean_neighbor_distance" not in retrieval


def test_neighbor_backend_abstraction_matches_numpy_fallback() -> None:
    source = np.array([[0.0, 0.0], [2.0, 0.0]])
    target = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])

    sklearn_result = build_candidate_neighbors(source, target, k=2, backend="sklearn_exact")
    numpy_result = build_candidate_neighbors(
        source,
        target,
        k=2,
        backend="numpy_chunked",
        chunk_size=1,
    )

    assert sklearn_result.backend == "sklearn_exact"
    np.testing.assert_array_equal(sklearn_result.indices, numpy_result.indices)
    np.testing.assert_allclose(sklearn_result.distances, numpy_result.distances)


def test_safe_scaling_fallback_and_zero_variance() -> None:
    scaled, stats = safe_scale_vector([1.0, 1.0, 1.0, 1.0, 3.0], min_scale=1e-6)
    zeros, zero_stats = safe_scale_vector([2.0, 2.0, 2.0], min_scale=1e-6)

    assert stats["scaling_method_used"] in {"mean_std", "min_range"}
    assert not stats["zero_variance"]
    np.testing.assert_allclose(zeros, [0.0, 0.0, 0.0])
    assert zero_stats["zero_variance"]


def test_pair_adaptive_temperature_and_scaled_cost() -> None:
    first = pd.DataFrame(
        {
            "scaled_molecular_distance": [0.1, 0.2, 0.3],
            "scaled_composition_distance": [0.1, 0.1, 0.1],
        }
    )
    second = pd.DataFrame(
        {
            "scaled_molecular_distance": [10.0, 20.0, 30.0],
            "scaled_composition_distance": [1.0, 1.0, 1.0],
        }
    )
    weights = {"molecular_state": 1.0, "cell_type_composition": 2.0}
    columns = {
        "molecular_state": "scaled_molecular_distance",
        "cell_type_composition": "scaled_composition_distance",
    }

    first_cost = combine_scaled_evidence(first, weights, columns)
    second_cost = combine_scaled_evidence(second, weights, columns)

    np.testing.assert_allclose(first_cost, [0.3, 0.4, 0.5])
    assert pair_adaptive_temperature(first_cost) != pair_adaptive_temperature(second_cost)


def test_row_normalization_uses_mass_adjusted_weight_and_preserves_raw_weight() -> None:
    edges = pd.DataFrame(
        {
            "source_anchor_id": ["s1", "s1", "s2", "s2"],
            "raw_edge_weight": [10.0, 1.0, 2.0, 2.0],
            "mass_adjusted_weight": [5.0, 5.0, 1.0, 3.0],
        }
    )

    edges["row_normalized_transition_prob"] = row_normalize_weights(edges)

    np.testing.assert_allclose(
        edges["row_normalized_transition_prob"].to_numpy(),
        [0.5, 0.5, 0.25, 0.75],
    )
    assert not np.allclose(edges["raw_edge_weight"], edges["mass_adjusted_weight"])


def test_evidence_schema_contains_future_placeholders_and_metadata() -> None:
    columns = evidence_schema_columns()

    for column in [
        "raw_pseudotime_score",
        "raw_barcode_score",
        "scaled_pseudotime_score",
        "scaled_barcode_score",
        "source_mass",
        "target_mass",
        "growth_prior",
        "unbalanced_weight",
        "raw_edge_weight",
        "mass_adjusted_weight",
        "row_normalized_transition_prob",
    ]:
        assert column in columns


def test_m3_core_has_no_hard_coded_time_or_benchmark_labels() -> None:
    text = (Path(__file__).resolve().parents[1] / "src/nichefate/transition.py").read_text(
        encoding="utf-8"
    )

    for token in ["D0", "D3", "D9", "D21", "D35", "Moffitt", "Cadinu", "DSS", "colon"]:
        assert token not in text
