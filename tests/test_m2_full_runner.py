import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from nichefate.representation import build_m2_representation_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "m2_02_build_full_representation_by_slice.py"
SPEC = importlib.util.spec_from_file_location("m2_full_runner", RUNNER_PATH)
m2_full_runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(m2_full_runner)


SCALES = ["radius_x2", "radius_x4", "radius_x8"]
METADATA_COLUMNS = [
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
M1_COLUMNS = [
    *METADATA_COLUMNS[:2],
    "scale",
    *METADATA_COLUMNS[2:],
    "n_neighbors",
    "ct_l1__a",
    "ct_l2__b",
    "ct_l3__c",
    "ct_l1_entropy",
    "emb_mean_pc001",
    "mean_neighbor_distance",
    "local_topology_anchor_degree",
]
CONFIG = {
    "representation": {
        "version": "m2_v1",
        "row_granularity": "one_row_per_anchor",
        "metadata_columns": METADATA_COLUMNS,
        "anchor_keys": ["slice_id", "anchor_index"],
        "scale_column": "scale",
        "scale_prefix_separator": "__",
    },
    "expected": {"scales": SCALES},
    "feature_groups": {
        "metadata": {"columns": METADATA_COLUMNS},
        "scale": {"values": SCALES},
        "neighborhood_size": {"columns": ["n_neighbors"]},
        "cell_type_composition": {"patterns": ["ct_l1__*", "ct_l2__*", "ct_l3__*"]},
        "entropy": {"columns": ["ct_l1_entropy"]},
        "molecular_state": {"columns": ["emb_mean_pc001"]},
        "spatial_summary": {"columns": ["mean_neighbor_distance"]},
        "topology": {"columns": ["local_topology_anchor_degree"]},
    },
}


def make_m1_table(slice_id: str) -> pd.DataFrame:
    rows = []
    for anchor_index in [0, 1]:
        for scale_index, scale in enumerate(SCALES, start=1):
            rows.append(
                {
                    "slice_id": slice_id,
                    "slice_file": f"{slice_id}.m0.h5ad",
                    "scale": scale,
                    "time": "T0",
                    "time_day": 0,
                    "mouse_id": "m1",
                    "anchor_index": anchor_index,
                    "anchor_cell_id": f"{slice_id}-cell-{anchor_index}",
                    "cell_type_l1": "type_a",
                    "cell_type_l2": "type_b",
                    "cell_type_l3": "type_c",
                    "n_neighbors": anchor_index + scale_index,
                    "ct_l1__a": 1.0,
                    "ct_l2__b": 0.5,
                    "ct_l3__c": 0.25,
                    "ct_l1_entropy": 0.1,
                    "emb_mean_pc001": float(scale_index),
                    "mean_neighbor_distance": 2.0,
                    "local_topology_anchor_degree": 3.0,
                }
            )
    return pd.DataFrame(rows, columns=M1_COLUMNS)


def schema_info() -> dict:
    return m2_full_runner.build_schema_info(CONFIG, {"feature_columns": M1_COLUMNS})


def test_toy_by_slice_m1_feature_table_pivots_to_anchor_rows() -> None:
    info = schema_info()
    table = make_m1_table("s1")

    matrix = build_m2_representation_table(
        table,
        feature_columns=info["source_feature_columns"],
        expected_scales=info["expected_scales"],
        metadata_columns=info["metadata_columns"],
        anchor_keys=info["anchor_keys"],
    )

    assert matrix.shape[0] == 2
    assert "slice_file" in matrix.columns
    assert "radius_x8__n_neighbors" in matrix.columns


def test_aligned_output_columns_across_two_toy_slices() -> None:
    info = schema_info()
    first = build_m2_representation_table(
        make_m1_table("s1"),
        info["source_feature_columns"],
        SCALES,
        METADATA_COLUMNS,
        ["slice_id", "anchor_index"],
    )
    second = build_m2_representation_table(
        make_m1_table("s2"),
        info["source_feature_columns"],
        SCALES,
        METADATA_COLUMNS,
        ["slice_id", "anchor_index"],
    )

    assert list(first.columns) == info["output_columns"]
    assert list(second.columns) == info["output_columns"]


def test_resume_skip_validation_accepts_valid_output(tmp_path: Path) -> None:
    info = schema_info()
    matrix = build_m2_representation_table(
        make_m1_table("s1"),
        info["source_feature_columns"],
        SCALES,
        METADATA_COLUMNS,
        ["slice_id", "anchor_index"],
    )
    output_path = tmp_path / "m2_representation_s1.parquet"
    matrix.to_parquet(output_path, index=False)

    ok, summary, error = m2_full_runner.validate_existing_output(
        output_path,
        info["output_columns"],
        expected_rows=2,
        numeric_columns=info["numeric_feature_columns"],
    )

    assert ok
    assert summary["output_rows"] == 2
    assert error == ""


def test_missing_scale_detection_raises() -> None:
    info = schema_info()
    table = make_m1_table("s1")
    table = table[~((table["anchor_index"] == 1) & (table["scale"] == "radius_x8"))]

    with pytest.raises(ValueError, match="Missing scale rows"):
        build_m2_representation_table(
            table,
            info["source_feature_columns"],
            SCALES,
            METADATA_COLUMNS,
            ["slice_id", "anchor_index"],
        )


def test_duplicated_anchor_scale_rows_raise() -> None:
    info = schema_info()
    table = make_m1_table("s1")
    table = pd.concat([table, table.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate anchor-scale"):
        build_m2_representation_table(
            table,
            info["source_feature_columns"],
            SCALES,
            METADATA_COLUMNS,
            ["slice_id", "anchor_index"],
        )


def test_m2_full_runner_files_do_not_hard_code_benchmark_labels() -> None:
    paths = [
        PROJECT_ROOT / "configs" / "m2_niche_representation.yaml",
        PROJECT_ROOT / "scripts" / "m2_02_build_full_representation_by_slice.py",
        PROJECT_ROOT / "src" / "nichefate" / "representation.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)

    for label in ["moffitt", "cadinu", "dss", "colon"]:
        assert label not in text
