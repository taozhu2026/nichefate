import anndata as ad
import numpy as np
import pandas as pd
import pytest

from nichefate.spatial import normalize_spatial_by_slice, set_spatial_obsm


def make_spatial_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "x": [0.0, 1.0, 10.0, 12.0],
            "y": [0.0, 1.0, 20.0, 22.0],
            "slice_id": ["s1", "s1", "s2", "s2"],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    return ad.AnnData(X=np.ones((4, 2)), obs=obs)


def test_set_spatial_obsm_writes_expected_keys() -> None:
    adata = make_spatial_adata()

    set_spatial_obsm(adata)

    assert "spatial" in adata.obsm
    assert "X_spatial" in adata.obsm
    np.testing.assert_allclose(adata.obsm["spatial"], adata.obs[["x", "y"]].to_numpy())


def test_set_spatial_obsm_requires_coordinate_fields() -> None:
    adata = make_spatial_adata()
    del adata.obs["x"]

    with pytest.raises(ValueError, match="Missing spatial coordinate field"):
        set_spatial_obsm(adata)


def test_normalize_spatial_by_slice_is_per_slice() -> None:
    adata = make_spatial_adata()
    set_spatial_obsm(adata)

    params = normalize_spatial_by_slice(adata)

    assert "X_spatial_norm" in adata.obsm
    assert set(params["slice_id"]) == {"s1", "s2"}
    for slice_id in ("s1", "s2"):
        mask = adata.obs["slice_id"].to_numpy() == slice_id
        centered = adata.obsm["X_spatial_norm"][mask]
        np.testing.assert_allclose(np.median(centered, axis=0), [0.0, 0.0])
    assert {"center_x", "center_y", "scale", "n_cells"}.issubset(params.columns)
