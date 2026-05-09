import pandas as pd

from nichefate.io import load_config
from nichefate.transition import (
    build_full_transition_shards,
    edge_density_metrics,
    estimate_time_pair_memory,
    full_transition_schema_columns,
)


def toy_pairs() -> list[dict]:
    return [
        {
            "source_time": "t0",
            "target_time": "t1",
            "source_day": 0.0,
            "target_day": 2.0,
            "time_delta": 2.0,
            "source_row_count": 15,
            "target_row_count": 20,
            "source_slices": ["s0a", "s0b"],
            "target_slices": ["s1a"],
        },
        {
            "source_time": "t1",
            "target_time": "t2",
            "source_day": 2.0,
            "target_day": 9.0,
            "time_delta": 7.0,
            "source_row_count": 20,
            "target_row_count": 5,
            "source_slices": ["s1a"],
            "target_slices": ["s2a"],
        },
    ]


def toy_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slice_id": ["s0a", "s0b", "s1a", "s2a"],
            "output_rows": [10, 5, 20, 5],
        }
    )


def test_full_design_shard_rows_equal_source_rows_times_k() -> None:
    shards = build_full_transition_shards(toy_pairs(), toy_summary(), candidate_k=3)

    assert shards.loc[shards["source_slice_id"] == "s0a", "expected_edge_rows"].iloc[0] == 30
    assert shards.loc[shards["source_slice_id"] == "s0b", "expected_edge_rows"].iloc[0] == 15
    assert int(shards["expected_edge_rows"].sum()) == (15 + 20) * 3


def test_final_time_point_is_not_used_as_source() -> None:
    shards = build_full_transition_shards(toy_pairs(), toy_summary(), candidate_k=3)

    assert "t2" not in set(shards["source_time"])
    assert "t2" in set(shards["target_time"])


def test_full_schema_contains_required_weight_and_probability_columns() -> None:
    columns = full_transition_schema_columns()

    for column in [
        "raw_edge_weight",
        "mass_adjusted_weight",
        "row_normalized_transition_prob",
        "scaling_method_molecular",
        "zero_variance_topology",
    ]:
        assert column in columns


def test_full_m3_config_is_design_only_and_local_probability_scope() -> None:
    config = load_config("configs/m3_transition_kernel.yaml")
    full = config["full_m3"]

    assert full["enabled"] is False
    assert full["execution_mode"] == "design_only"
    assert full["write_global_kernel"] is False
    assert full["row_normalization_scope"] == "source_niche_candidate_set"
    assert full["sample_aware"] is True
    assert full["sample_paired"] is False


def test_fixed_k_and_adaptive_k_placeholders_exist() -> None:
    full = load_config("configs/m3_transition_kernel.yaml")["full_m3"]

    assert full["candidate_k_mode"] == "fixed"
    assert full["adaptive_k_options"]["fraction_of_target"] is None
    assert full["adaptive_k_options"]["min_k"] == 30
    assert full["adaptive_k_options"]["max_k"] == 100


def test_target_pool_density_metrics() -> None:
    density = edge_density_metrics(toy_pairs(), candidate_k=5)

    assert density.loc[0, "target_pool_size"] == 20
    assert density.loc[0, "k_over_target_pool"] == 0.25
    assert density.loc[1, "target_pool_size"] == 5
    assert density.loc[1, "expected_candidate_edge_density"] == 1.0


def test_concurrency_cap_is_bounded_by_memory_warning() -> None:
    shards = build_full_transition_shards(toy_pairs(), toy_summary(), candidate_k=3)
    memory = estimate_time_pair_memory(
        toy_pairs(),
        shards,
        retrieval_dimensions=10,
        rerank_dimensions=5,
        max_memory_gb=1e-6,
    )

    assert (memory["safe_single_node_concurrency"] >= 1).all()
    assert "per_worker_memory_gb" in memory.columns
