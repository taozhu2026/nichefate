from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.planA_k.production_preflight import (
    PRODUCTION_ORDER,
    build_coordinate_join_contract,
    build_full_feature_lock,
    build_full_kmix_A_config,
    build_full_m2_5_coarsening_strategy,
    build_full_run_blueprint,
    build_gpcca_feasibility,
    evaluate_coordinate_join_validation,
    render_simple_yaml,
)


def write_synthetic_schema(path: Path) -> list[str]:
    metadata = [
        "slice_id",
        "slice_file",
        "time",
        "time_day",
        "mouse_id",
        "anchor_index",
        "anchor_cell_id",
        "cell_type_l1",
        "cell_type_l2",
        "cell_type_l3",
    ]
    safe = [
        "radius_x2__ct_l1__epithelial",
        "radius_x2__ct_l2__fibro_1",
        "radius_x2__ct_l3__immune_a",
        "radius_x2__niche_entropy",
        "radius_x2__emb_mean_pc1",
        "radius_x4__emb_mean_pc2",
    ]
    deferred = [
        "radius_x2__emb_var_pc1",
        "radius_x2__n_neighbors",
        "radius_x2__spatial_density",
        "radius_x2__endpoint_score",
    ]
    path.write_text(
        json.dumps(
            {
                "metadata_columns": metadata,
                "numeric_feature_columns": safe + deferred,
                "output_columns": metadata + safe + deferred,
                "metadata_column_count": len(metadata),
                "numeric_feature_column_count": len(safe) + len(deferred),
                "output_column_count": len(metadata) + len(safe) + len(deferred),
            }
        ),
        encoding="utf-8",
    )
    return safe


def synthetic_input_summary() -> dict:
    return {
        "blockers": [],
        "observed": {
            "total_m2_anchors_from_parquet_metadata": 10_000,
            "timepoint_anchor_totals": {
                "D0": 1_000,
                "D3": 2_000,
                "D9": 2_500,
                "D21": 2_500,
                "D35": 2_000,
            },
        },
    }


def test_feature_lock_selects_safe_columns_and_reproducibility_requirements(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    safe = write_synthetic_schema(schema_path)

    frame, payload, config = build_full_feature_lock(schema_path=schema_path, production_root=tmp_path / "scratch")

    assert payload["safe_feature_column_count"] == len(safe)
    assert config["feature_columns"] == safe
    assert "endpoint_score" not in " ".join(config["feature_columns"])
    assert "slice_id" not in config["feature_columns"]
    assert set(frame["feature_group"]).issuperset(
        {"metadata", "niche_composition_features", "entropy_features", "embedding_mean_features"}
    )
    for required in [
        "exact feature list",
        "scaler parameters or scaler object",
        "PCA components or PCA object",
        "training sample manifest",
        "random seed",
        "software environment record",
    ]:
        assert required in config["reproducibility_requirements"]


def test_strategy_and_kmix_include_d35_without_locking_top_k(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    write_synthetic_schema(schema_path)
    _, feature_payload, _ = build_full_feature_lock(schema_path=schema_path, production_root=tmp_path / "scratch")

    _, strategy = build_full_m2_5_coarsening_strategy(synthetic_input_summary(), feature_payload)
    config = build_full_kmix_A_config(production_root=tmp_path / "scratch", report_root=tmp_path / "reports")

    assert "D35 metaniche count" in strategy["d35_qc_requirements"]
    assert config["include_d35"] is True
    assert {"source": "D21", "target": "D35"} in config["forward_edges"]
    assert config["forward_top_k"]["candidate"] == 20
    assert config["forward_top_k"]["qc_grid"] == [10, 20, 30]
    assert config["forward_top_k"]["locked_final_value"] is None


def test_coordinate_join_contract_blocks_duplicates_and_low_coverage() -> None:
    bounded = pd.DataFrame(
        [
            {
                "slice_id": "slice_D0",
                "join_coverage": 1.0,
                "m2_duplicate_key_rows": 0,
                "m1_duplicate_key_rows": 0,
            }
        ]
    )
    _, payload = build_coordinate_join_contract(
        bounded,
        {
            "sample_rows_per_slice": 2,
            "slice_count_sampled": 1,
            "min_sample_join_coverage": 1.0,
            "sample_duplicate_key_rows": 0,
        },
    )

    assert payload["production_blocking_rules"]["duplicate_join_keys_block"] is True
    assert payload["production_blocking_rules"]["minimum_join_coverage"] == 0.999
    assert evaluate_coordinate_join_validation(0.998, 0)["production_blocked"] is True
    assert evaluate_coordinate_join_validation(1.0, 1)["production_blocked"] is True
    assert evaluate_coordinate_join_validation(0.999, 0)["production_blocked"] is False


def test_blueprint_order_and_path_policy_use_scratch_for_production(tmp_path: Path) -> None:
    blueprint = build_full_run_blueprint(
        "DIRECT_FULL_RUN_READY_WITH_RESOURCE_CAUTION",
        production_root=tmp_path / "scratch" / "planA_k_production",
        report_root=tmp_path / "repo" / "reports",
    )

    assert blueprint["production_order"] == PRODUCTION_ORDER
    assert [row["phase"] for row in blueprint["commands"]] == PRODUCTION_ORDER
    assert "planA_k_23_run_full_m2_5_production.py" in blueprint["next_safe_command"]
    assert "gpcca" not in blueprint["next_safe_command"].lower()
    assert blueprint["output_path_policy"]["production_matrices_parquet_kmix_gpcca_under_scratch"] is True
    assert all("/scratch/" in row["production_output_root"] for row in blueprint["commands"])
    assert all("/reports" in row["lightweight_report_root"] for row in blueprint["commands"])


def test_gpcca_feasibility_keeps_resource_caution() -> None:
    strategy = {"estimate": {"target_metaniche_count": 1024}}
    payload = build_gpcca_feasibility(strategy, {"blockers": []})

    assert payload["decision_label"] in {
        "DIRECT_FULL_RUN_READY_WITH_RESOURCE_CAUTION",
        "NEEDS_SCALE_CONTROLLED_FALLBACK",
    }
    assert payload["estimated_sparse_nnz"] > 0
    assert any("Full GPCCA is not run" in item for item in payload["resource_caution"])


def test_yaml_renderer_and_compatibility_imports() -> None:
    text = render_simple_yaml({"a": True, "b": [1, {"c": "D21->D35"}]})
    assert "a: true" in text
    assert "D21->D35" in text

    from nichefate.planA_k_gpcca import build_full_run_blueprint as facade_blueprint

    assert facade_blueprint("DIRECT_FULL_RUN_READY")["decision_label"] == "DIRECT_FULL_RUN_READY"
