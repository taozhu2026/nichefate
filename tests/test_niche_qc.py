import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nichefate.niche_qc import (
    composition_columns,
    composition_sum_qc,
    dominant_composition,
    estimate_full_m1_storage,
    load_neighbor_metadata,
    summarize_feature_integrity,
    validate_neighbor_npz,
)


def test_feature_qc_detects_missing_and_infinite_values() -> None:
    table = pd.DataFrame(
        {
            "scale": ["radius_x2", "radius_x2"],
            "ct_l1__a": [0.5, np.nan],
            "ct_l1__b": [0.5, 1.0],
            "emb_mean_pc001": [1.0, np.inf],
            "n_neighbors": [3, 4],
        }
    )

    summary = summarize_feature_integrity(table).set_index("feature_group")

    assert summary.loc["composition_l1", "missing_values"] == 1
    assert summary.loc["embedding_mean", "infinite_values"] == 1


def test_composition_column_discovery_and_sum_qc() -> None:
    table = pd.DataFrame(
        {
            "scale": ["radius_x2", "radius_x4"],
            "ct_l1__a": [0.25, 0.40],
            "ct_l1__b": [0.75, 0.40],
            "ct_l2__na": [0.0, 1.0],
        }
    )

    assert composition_columns(table, "cell_type_l1") == ["ct_l1__a", "ct_l1__b"]
    qc = composition_sum_qc(table, "cell_type_l1").iloc[0]

    assert qc["rows_close_to_one"] == 1
    assert qc["rows_not_close_to_one"] == 1


def test_dominant_composition_summarizes_by_scale() -> None:
    table = pd.DataFrame(
        {
            "scale": ["radius_x2", "radius_x2", "radius_x4"],
            "ct_l1__a": [0.7, 0.2, 0.1],
            "ct_l1__b": [0.3, 0.8, 0.9],
        }
    )

    dominant = dominant_composition(table, "cell_type_l1", by=("scale",))

    assert set(dominant["dominant_label"]) == {"a", "b"}
    assert int(dominant["row_count"].sum()) == 3


def _write_neighbor_npz(path: Path, neighbor_indices: np.ndarray) -> Path:
    metadata = [
        {
            "entry": "entry_000",
            "slice_id": "s1",
            "slice_file": "slice_a.m0.h5ad",
            "scale": "radius_x2",
            "n_anchors": 2,
            "n_neighbor_links": int(len(neighbor_indices)),
        }
    ]
    np.savez_compressed(
        path,
        entry_000__anchor_indices=np.array([0, 2], dtype=np.int64),
        entry_000__indptr=np.array([0, 2, len(neighbor_indices)], dtype=np.int64),
        entry_000__neighbor_indices=neighbor_indices.astype(np.int64),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    return path


def test_neighbor_npz_integrity_checks_pass_on_valid_index(tmp_path: Path) -> None:
    path = _write_neighbor_npz(tmp_path / "neighbors.npz", np.array([0, 1, 0, 2, 3]))
    feature_table = pd.DataFrame(
        {
            "slice_file": ["slice_a.m0.h5ad", "slice_a.m0.h5ad"],
            "slice_id": ["s1", "s1"],
            "scale": ["radius_x2", "radius_x2"],
            "anchor_index": [0, 2],
            "n_neighbors": [2, 3],
        }
    )

    metadata = load_neighbor_metadata(path)
    validation = validate_neighbor_npz(
        path,
        feature_table=feature_table,
        slice_n_obs={"s1": 4},
        expected_entries=1,
        include_anchor=True,
    )

    assert metadata[0]["slice_id"] == "s1"
    assert bool(validation.loc[0, "ok"])
    assert bool(validation.loc[0, "avg_neighbors_match"])


def test_neighbor_npz_integrity_checks_detect_invalid_index(tmp_path: Path) -> None:
    path = _write_neighbor_npz(tmp_path / "bad_neighbors.npz", np.array([0, -1, 0, 2, 9]))

    validation = validate_neighbor_npz(
        path,
        slice_n_obs={"s1": 4},
        expected_entries=1,
        include_anchor=True,
    )

    assert not bool(validation.loc[0, "ok"])
    assert validation.loc[0, "negative_neighbor_count"] == 1
    assert not bool(validation.loc[0, "within_slice_bounds"])


def test_full_m1_size_estimator_runs_on_toy_data() -> None:
    estimate = estimate_full_m1_storage(
        full_anchors=10,
        scales=["radius_x2", "radius_x4"],
        prototype_rows=4,
        prototype_feature_bytes=400,
        avg_neighbors_by_scale={"radius_x2": 2.0, "radius_x4": 4.0},
        n_slices=2,
    )

    assert estimate["full_feature_rows"] == 20
    assert estimate["feature_csv_bytes"] == 2000
    assert estimate["neighbor_raw_bytes"] > 0


def test_qc_helpers_do_not_require_moffitt_specific_fields() -> None:
    table = pd.DataFrame(
        {
            "slice_id": ["generic_slice"],
            "scale": ["small_radius"],
            "anchor_index": [0],
            "ct_l1__generic_type": [1.0],
            "n_neighbors": [1],
        }
    )

    integrity = summarize_feature_integrity(table)
    composition = composition_sum_qc(table, "cell_type_l1")

    assert not integrity.empty
    assert composition.loc[0, "rows_close_to_one"] == 1


def test_niche_qc_does_not_import_forbidden_optional_dependencies() -> None:
    assert "squidpy" not in sys.modules
    assert "spatialdata" not in sys.modules
    assert "harmonypy" not in sys.modules
