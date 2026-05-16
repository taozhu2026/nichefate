from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.planA_k_gpcca import (
    bounded_sample_m2_rows,
    classify_m2_feature_groups,
    compute_metaniche_qc_from_outputs,
    discover_m2_inventory,
    load_m2_feature_schema,
    run_sampled_metaniche_pilot,
    select_m2_feature_columns,
)


def tiny_schema() -> dict:
    metadata_columns = [
        "slice_id",
        "time",
        "time_day",
        "mouse_id",
        "anchor_index",
        "anchor_cell_id",
        "cell_type_l1",
        "cell_type_l2",
        "cell_type_l3",
    ]
    numeric = [
        "radius_x2__ct_l1__epithelial",
        "radius_x2__ct_l1__immune",
        "radius_x2__ct_l1_entropy",
        "radius_x2__emb_mean_pc001",
        "radius_x2__emb_mean_pc002",
        "radius_x2__emb_var_pc001",
        "radius_x2__n_neighbors",
        "radius_x2__local_topology_anchor_degree",
    ]
    return {
        "exists": True,
        "metadata_columns": metadata_columns,
        "metadata_column_count": len(metadata_columns),
        "numeric_feature_columns": numeric,
        "numeric_feature_column_count": len(numeric),
        "output_columns": [*metadata_columns, *numeric],
        "output_column_count": len(metadata_columns) + len(numeric),
        "expected_scales": ["radius_x2"],
        "row_granularity": "one_row_per_anchor",
    }


def write_tiny_m2_root(tmp_path: Path) -> tuple[Path, Path]:
    m2_root = tmp_path / "m2" / "by_slice"
    m2_root.mkdir(parents=True)
    schema = tiny_schema()
    schema_path = tmp_path / "m2" / "reports" / "m2_full_feature_schema.json"
    schema_path.parent.mkdir(parents=True)
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    rows = []
    rng = np.random.default_rng(4)
    for slice_idx, day in enumerate([0, 3]):
        slice_id = f"tiny_D{day}_slice_{slice_idx + 1}"
        slice_dir = m2_root / slice_id
        slice_dir.mkdir()
        frame = pd.DataFrame(
            {
                "slice_id": slice_id,
                "time": f"D{day}",
                "time_day": day,
                "mouse_id": f"m{slice_idx + 1}",
                "anchor_index": np.arange(30),
                "anchor_cell_id": [f"cell_{slice_idx}_{i}" for i in range(30)],
                "cell_type_l1": ["epithelial"] * 15 + ["immune"] * 15,
                "cell_type_l2": ["a"] * 10 + ["b"] * 10 + ["c"] * 10,
                "cell_type_l3": ["rare"] * 2 + ["common"] * 28,
                "radius_x2__ct_l1__epithelial": np.r_[np.ones(15), np.zeros(15)],
                "radius_x2__ct_l1__immune": np.r_[np.zeros(15), np.ones(15)],
                "radius_x2__ct_l1_entropy": rng.random(30),
                "radius_x2__emb_mean_pc001": rng.normal(size=30),
                "radius_x2__emb_mean_pc002": rng.normal(size=30),
                "radius_x2__emb_var_pc001": rng.random(30),
                "radius_x2__n_neighbors": rng.integers(5, 15, size=30),
                "radius_x2__local_topology_anchor_degree": rng.integers(1, 8, size=30),
            }
        )
        output_path = slice_dir / f"m2_representation_{slice_id}.parquet"
        frame.to_parquet(output_path, index=False)
        rows.append(
            {
                "slice_id": slice_id,
                "status": "completed",
                "output_path": str(output_path),
                "output_rows": len(frame),
                "output_columns": len(frame.columns),
            }
        )
    pd.DataFrame(rows).to_csv(m2_root / "completed_slices.csv", index=False)
    return m2_root, schema_path


def test_feature_group_selection_uses_safe_columns() -> None:
    schema = tiny_schema()
    audit = classify_m2_feature_groups(schema)
    safe = select_m2_feature_columns(schema, feature_mode="safe")

    assert "niche_composition_features" in set(audit["feature_group"])
    assert "radius_x2__emb_mean_pc001" in safe
    assert "radius_x2__emb_var_pc001" not in safe
    assert "time_day" not in safe


def test_inventory_and_bounded_sampling(tmp_path: Path) -> None:
    m2_root, schema_path = write_tiny_m2_root(tmp_path)
    inventory, summary = discover_m2_inventory(m2_root=m2_root, schema_path=schema_path)
    schema = load_m2_feature_schema(schema_path)
    safe = select_m2_feature_columns(schema, feature_mode="safe")
    sampled = bounded_sample_m2_rows(
        inventory=inventory,
        schema=schema,
        feature_columns=safe,
        max_slices=2,
        max_anchors_per_slice=5,
        seed=9,
    )

    assert summary["m2_file_count"] == 2
    assert len(sampled) == 10
    assert "anchor_id" in sampled.columns
    assert sampled["slice_id"].nunique() == 2


def test_sampled_pilot_and_qc_schema(tmp_path: Path) -> None:
    m2_root, schema_path = write_tiny_m2_root(tmp_path)
    inventory, _ = discover_m2_inventory(m2_root=m2_root, schema_path=schema_path)
    schema = load_m2_feature_schema(schema_path)
    output_dir = tmp_path / "reports"

    payload = run_sampled_metaniche_pilot(
        inventory=inventory,
        schema=schema,
        output_dir=output_dir,
        max_slices=2,
        max_anchors_per_slice=12,
        n_components=2,
        n_clusters=3,
        dry_run=False,
        overwrite=True,
        seed=3,
    )
    qc_frame, qc_payload = compute_metaniche_qc_from_outputs(output_dir)

    assert payload["pilot_run"] is True
    assert (output_dir / "pilot_outputs" / "anchor_to_metaniche.tsv").exists()
    assert qc_payload["pilot_outputs_found"] is True
    assert qc_payload["sampled_anchor_count"] == 24
    assert set(qc_frame["qc_metric"]) >= {"sampled_anchor_count", "metaniche_count"}


def test_missing_m2_inputs_are_handled(tmp_path: Path) -> None:
    inventory, summary = discover_m2_inventory(
        m2_root=tmp_path / "missing",
        schema_path=tmp_path / "missing_schema.json",
    )

    assert inventory.empty
    assert summary["m2_file_count"] == 0
    assert summary["safe_to_sample_count"] == 0
