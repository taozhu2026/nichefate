import anndata as ad
import numpy as np
import pandas as pd
import pytest

from nichefate.io import load_config
from nichefate.metadata import (
    build_time_mapping,
    ensure_day35_time_fallback,
    standardize_colitis_metadata,
    validate_required_fields,
)


def make_obs(include_sample_type: bool = True) -> pd.DataFrame:
    obs = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [5.0, 6.0, 7.0],
            "Mouse_ID": ["m1", "m1", "m2"],
            "Technical_repeat_number": [1, 1, 2],
            "Slice_ID": ["s1", "s1", "s2"],
            "FOV": ["f1", "f2", "f1"],
            "cell_IDs": ["c1", "c2", "c3"],
            "N_genes": [10, 11, 12],
            "Tier1": ["A", "B", "A"],
            "Tier2": ["A2", "B2", "A2"],
            "Tier3": ["A3", "B3", "A3"],
            "Leiden_neigh": ["n1", "n1", "n2"],
        },
        index=["cell1", "cell2", "cell3"],
    )
    if include_sample_type:
        obs["Sample_type"] = ["Healthy", "DSS3", "DSS9"]
    return obs


def make_adata(include_sample_type: bool = True) -> ad.AnnData:
    return ad.AnnData(
        X=np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        obs=make_obs(include_sample_type=include_sample_type),
    )


def test_time_mapping_works() -> None:
    config = load_config("configs/m0_merfish_colitis.yaml")
    mapping = build_time_mapping(config)

    assert mapping["Healthy"] == 0
    assert mapping["DSS21"] == 21
    assert mapping["Day35"] == 35


def test_day35_fallback_creates_sample_type() -> None:
    adata = make_adata(include_sample_type=False)
    adata.obs["dataset_part"] = "day35"
    adata.obs["source_file"] = "adata_day35.h5ad"

    ensure_day35_time_fallback(adata)

    assert set(adata.obs["Sample_type"]) == {"Day35"}


def test_missing_required_fields_raise_clear_error() -> None:
    adata = make_adata()
    del adata.obs["x"]

    with pytest.raises(ValueError, match="missing required obs fields: x"):
        validate_required_fields(adata, ["x", "y"], "toy")


def test_required_fields_match_real_m0_core_files() -> None:
    config = load_config("configs/m0_merfish_colitis.yaml")
    required = config["metadata"]["required_obs_fields"]
    optional = config["metadata"]["optional_obs_fields"]

    assert "Cell_ID" not in required
    assert "cell_IDs" not in required
    assert "N_genes" not in required
    assert "Cell_ID" in optional
    assert "cell_IDs" in optional
    assert "N_genes" in optional


def test_standardize_metadata_creates_m0_fields() -> None:
    config = load_config("configs/m0_merfish_colitis.yaml")
    adata = make_adata()

    standardize_colitis_metadata(adata, "adata.h5ad", "main", config)

    expected = {
        "time_day",
        "time",
        "mouse_id",
        "slice_id",
        "fov_id",
        "batch_id",
        "cell_type_l1",
        "cell_type_l2",
        "cell_type_l3",
        "neighborhood_original",
        "source_file",
        "dataset_part",
    }
    assert expected.issubset(set(adata.obs.columns))
    assert list(adata.obs["time_day"]) == [0, 3, 9]
    assert list(adata.obs["time"]) == ["D0", "D3", "D9"]


def test_standardize_metadata_fills_missing_tier2_with_na() -> None:
    config = load_config("configs/m0_merfish_colitis.yaml")
    adata = make_adata()
    del adata.obs["Tier2"]

    standardize_colitis_metadata(adata, "adata_day35.h5ad", "day35", config)

    assert set(adata.obs["cell_type_l2"]) == {"NA"}
