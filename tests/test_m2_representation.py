from pathlib import Path

import pandas as pd
import pytest

from nichefate.representation import (
    feature_group_columns,
    pivot_scale_features,
    select_numeric_feature_columns,
)


SCALES = ["radius_x2", "radius_x4", "radius_x8"]
METADATA_COLUMNS = [
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
FEATURE_GROUPS = {
    "metadata": {"columns": ["slice_id", "slice_file", "anchor_index"]},
    "scale": {"values": SCALES},
    "neighborhood_size": {"columns": ["n_neighbors"]},
    "cell_type_composition": {"patterns": ["ct_l1__*", "ct_l2__*", "ct_l3__*"]},
    "entropy": {"columns": ["ct_l1_entropy", "ct_l2_entropy", "ct_l3_entropy"]},
    "molecular_state": {"ranges": ["emb_mean_pc001..pc002", "emb_var_pc001..pc002"]},
    "spatial_summary": {"columns": ["mean_neighbor_distance", "pseudo_local_density"]},
    "topology": {
        "columns": ["local_topology_anchor_degree", "local_topology_mean_member_degree"]
    },
}


def make_feature_table() -> pd.DataFrame:
    rows = []
    for anchor_index in [0, 1]:
        for scale_index, scale in enumerate(SCALES, start=1):
            rows.append(
                {
                    "slice_id": "s1",
                    "slice_file": "s1.m0.h5ad",
                    "scale": scale,
                    "time": "T0",
                    "time_day": 0,
                    "mouse_id": "m1",
                    "anchor_index": anchor_index,
                    "anchor_cell_id": f"cell-{anchor_index}",
                    "cell_type_l1": "type_a",
                    "cell_type_l2": "type_b",
                    "cell_type_l3": "type_c",
                    "n_neighbors": anchor_index + scale_index,
                    "ct_l1__a": 1.0,
                    "ct_l2__b": 0.5,
                    "ct_l3__c": 0.25,
                    "ct_l1_entropy": 0.0,
                    "ct_l2_entropy": 0.1,
                    "ct_l3_entropy": 0.2,
                    "emb_mean_pc001": float(scale_index),
                    "emb_mean_pc002": float(scale_index + 1),
                    "emb_var_pc001": 0.01,
                    "emb_var_pc002": 0.02,
                    "mean_neighbor_distance": 2.0,
                    "pseudo_local_density": 3.0,
                    "local_topology_anchor_degree": 4.0,
                    "local_topology_mean_member_degree": 5.0,
                    "non_numeric_note": "ignored",
                }
            )
    return pd.DataFrame(rows)


def test_scale_pivoting_outputs_prefixed_columns() -> None:
    table = make_feature_table()
    feature_columns = ["n_neighbors", "ct_l1__a", "emb_mean_pc001"]

    matrix = pivot_scale_features(
        table,
        feature_columns=feature_columns,
        expected_scales=SCALES,
        metadata_columns=METADATA_COLUMNS,
        anchor_keys=["slice_id", "anchor_index"],
    )

    assert "radius_x2__n_neighbors" in matrix.columns
    assert "radius_x8__emb_mean_pc001" in matrix.columns
    assert matrix.loc[0, "radius_x4__ct_l1__a"] == 1.0


def test_feature_group_selection_uses_configured_numeric_columns() -> None:
    table = make_feature_table()

    grouped = feature_group_columns(table.columns, FEATURE_GROUPS)
    selected = select_numeric_feature_columns(table, FEATURE_GROUPS)

    assert grouped["molecular_state"] == [
        "emb_mean_pc001",
        "emb_mean_pc002",
        "emb_var_pc001",
        "emb_var_pc002",
    ]
    assert "n_neighbors" in selected
    assert "ct_l3__c" in selected
    assert "anchor_index" not in selected
    assert "non_numeric_note" not in selected


def test_missing_scale_detection_raises() -> None:
    table = make_feature_table()
    table = table[
        ~((table["anchor_index"] == 1) & (table["scale"] == "radius_x8"))
    ].copy()

    with pytest.raises(ValueError, match="Missing scale rows"):
        pivot_scale_features(
            table,
            feature_columns=["n_neighbors"],
            expected_scales=SCALES,
            metadata_columns=METADATA_COLUMNS,
            anchor_keys=["slice_id", "anchor_index"],
        )


def test_pivot_outputs_one_row_per_anchor() -> None:
    table = make_feature_table()

    matrix = pivot_scale_features(
        table,
        feature_columns=["n_neighbors"],
        expected_scales=SCALES,
        metadata_columns=METADATA_COLUMNS,
        anchor_keys=["slice_id", "anchor_index"],
    )

    assert matrix.shape[0] == table[["slice_id", "anchor_index"]].drop_duplicates().shape[0]
    assert matrix["anchor_index"].tolist() == [0, 1]


def test_m2_files_do_not_hard_code_benchmark_labels() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = [
        project_root / "configs" / "m2_niche_representation.yaml",
        project_root / "scripts" / "m2_00_audit_m1_outputs.py",
        project_root / "scripts" / "m2_01_prepare_representation_matrix.py",
        project_root / "src" / "nichefate" / "representation.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)

    for label in ["moffitt", "cadinu", "dss"]:
        assert label not in text
