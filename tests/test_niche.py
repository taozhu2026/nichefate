import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from nichefate.niche import (
    align_feature_table_to_schema,
    build_basic_niche_feature_table,
    compute_celltype_composition,
    compute_embedding_summary,
    compute_neighbor_index,
    compute_shannon_entropy_from_composition,
    get_graph_neighbors,
    safe_feature_token,
    write_neighbor_index_npz,
)


def make_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "slice_id": ["s1"] * 4,
            "time": ["D0"] * 4,
            "time_day": [0] * 4,
            "mouse_id": ["m1"] * 4,
            "cell_type_l1": ["A", "A", "B", "B"],
            "cell_type_l2": ["A1", "A1", "B1", "B2"],
            "cell_type_l3": ["A1a", "A1b", "B1a", "B2a"],
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
        },
        index=["c0", "c1", "c2", "c3"],
    )
    data = ad.AnnData(X=np.ones((4, 2)), obs=obs)
    data.obsm["X_pca_m0"] = np.arange(12, dtype=float).reshape(4, 3)
    data.obsm["X_spatial_norm"] = obs[["x", "y"]].to_numpy()
    graph = sparse.csr_matrix(
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ]
        )
    )
    data.obsp["radius_x2"] = graph
    data.obsp["delaunay"] = graph
    return data


def test_neighbor_index_extraction_and_include_anchor() -> None:
    data = make_adata()

    without_anchor = get_graph_neighbors(data, "radius_x2", [0], include_anchor=False)
    with_anchor = get_graph_neighbors(data, "radius_x2", [0], include_anchor=True)
    index = compute_neighbor_index(data, "radius_x2", [0], include_anchor=True)

    assert without_anchor[0].tolist() == [1, 2]
    assert with_anchor[0].tolist() == [0, 1, 2]
    assert index["indptr"].tolist() == [0, 3]
    assert index["neighbor_indices"].tolist() == [0, 1, 2]


def test_celltype_composition_entropy_and_embedding_summary() -> None:
    data = make_adata()
    index = compute_neighbor_index(data, "radius_x2", [0, 1], include_anchor=True)

    composition = compute_celltype_composition(data, index, "cell_type_l1")
    entropy = compute_shannon_entropy_from_composition(composition)
    embedding = compute_embedding_summary(data, index)

    np.testing.assert_allclose(composition.sum(axis=1).to_numpy(), [1.0, 1.0])
    assert np.isfinite(entropy).all()
    assert embedding.shape == (2, 6)
    assert "emb_mean_pc001" in embedding.columns
    assert "emb_var_pc001" in embedding.columns


def test_feature_rows_have_required_identity_columns() -> None:
    data = make_adata()
    index = compute_neighbor_index(data, "radius_x2", [0, 1], include_anchor=True)

    table = build_basic_niche_feature_table(
        data,
        index,
        scale="radius_x2",
        slice_file="toy.m0.h5ad",
        cell_type_keys=["cell_type_l1", "cell_type_l2", "cell_type_l3"],
    )

    for column in ("slice_id", "slice_file", "scale", "anchor_index", "anchor_cell_id"):
        assert column in table.columns
    assert "ct_l1__a" in table.columns
    assert "ct_l3_entropy" in table.columns
    assert "pseudo_local_density" in table.columns
    assert "local_topology_anchor_degree" in table.columns
    assert "local_topology_mean_member_degree" in table.columns


def test_feature_table_aligns_to_global_schema_columns() -> None:
    data = make_adata()
    index = compute_neighbor_index(data, "radius_x2", [0, 1], include_anchor=True)
    schema = {
        "composition_columns": [
            "ct_l1__a",
            "ct_l1__b",
            "ct_l1__c",
            "ct_l2__a1",
            "ct_l2__b1",
            "ct_l2__b2",
            "ct_l3__a1a",
            "ct_l3__a1b",
            "ct_l3__b1a",
            "ct_l3__b2a",
        ],
        "feature_columns": [
            "slice_id",
            "slice_file",
            "scale",
            "anchor_index",
            "anchor_cell_id",
            "time",
            "time_day",
            "mouse_id",
            "cell_type_l1",
            "cell_type_l2",
            "cell_type_l3",
            "x",
            "y",
            "ct_l1__a",
            "ct_l1__b",
            "ct_l1__c",
            "ct_l1_entropy",
            "ct_l2__a1",
            "ct_l2__b1",
            "ct_l2__b2",
            "ct_l2_entropy",
            "ct_l3__a1a",
            "ct_l3__a1b",
            "ct_l3__b1a",
            "ct_l3__b2a",
            "ct_l3_entropy",
            "emb_mean_pc001",
            "emb_mean_pc002",
            "emb_mean_pc003",
            "emb_var_pc001",
            "emb_var_pc002",
            "emb_var_pc003",
            "n_neighbors",
            "mean_neighbor_distance",
            "pseudo_local_density",
            "local_topology_anchor_degree",
            "local_topology_mean_member_degree",
        ],
    }

    table = build_basic_niche_feature_table(
        data,
        index,
        scale="radius_x2",
        slice_file="toy.m0.h5ad",
        cell_type_keys=["cell_type_l1", "cell_type_l2", "cell_type_l3"],
        global_schema=schema,
    )

    assert list(table.columns) == schema["feature_columns"]
    assert table["ct_l1__c"].tolist() == [0.0, 0.0]


def test_align_feature_table_preserves_slice_local_behavior_without_schema() -> None:
    table = pd.DataFrame({"scale": ["radius_x2"], "ct_l1__a": [1.0]})

    aligned = align_feature_table_to_schema(table, None)

    assert list(aligned.columns) == ["scale", "ct_l1__a"]
    assert safe_feature_token("A cell/type") == "a_cell_type"


def test_neighbor_index_npz_metadata_is_unambiguous(tmp_path: Path) -> None:
    data = make_adata()
    index = compute_neighbor_index(data, "radius_x2", [0, 1], include_anchor=True)
    path = write_neighbor_index_npz(
        [
            {
                "slice_id": "s1",
                "slice_file": "slice_a.m0.h5ad",
                "scale": "radius_x2",
                "neighbor_index": index,
            },
            {
                "slice_id": "s2",
                "slice_file": "slice_b.m0.h5ad",
                "scale": "radius_x4",
                "neighbor_index": index,
            },
        ],
        tmp_path / "neighbors.npz",
    )

    loaded = np.load(path)
    metadata = json.loads(str(loaded["metadata_json"]))

    assert metadata[0]["slice_id"] == "s1"
    assert metadata[1]["scale"] == "radius_x4"
    assert "entry_000__neighbor_indices" in loaded.files
    assert "entry_001__neighbor_indices" in loaded.files


def test_niche_code_does_not_import_optional_spatial_dependencies() -> None:
    assert "squidpy" not in sys.modules
    assert "spatialdata" not in sys.modules
    assert "harmonypy" not in sys.modules
