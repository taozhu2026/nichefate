from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_OBS_COLUMNS = ("sample_id", "slice_id", "cellbin_id", "x", "y")
REQUIRED_LAYERS = ("counts",)
REQUIRED_OBSM = ("spatial",)


def _close_h5ad(adata: Any) -> None:
    file_obj = getattr(adata, "file", None)
    close = getattr(file_obj, "close", None)
    if callable(close):
        close()


def validate_l126_h5ad_schema(path: str | Path) -> dict[str, Any]:
    import anndata as ad

    h5ad_path = Path(path).expanduser().resolve()
    data = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs_columns = set(data.obs.columns)
        missing_obs = [column for column in REQUIRED_OBS_COLUMNS if column not in obs_columns]
        missing_layers = [layer for layer in REQUIRED_LAYERS if layer not in data.layers.keys()]
        missing_obsm = [key for key in REQUIRED_OBSM if key not in data.obsm.keys()]
        sample_ids = sorted(data.obs["sample_id"].astype(str).unique().tolist()) if "sample_id" in data.obs else []
        slice_ids = sorted(data.obs["slice_id"].astype(str).unique().tolist()) if "slice_id" in data.obs else []
        return {
            "path": str(h5ad_path),
            "n_obs": int(data.n_obs),
            "n_vars": int(data.n_vars),
            "sample_ids": sample_ids,
            "slice_ids": slice_ids,
            "obs_columns": sorted(obs_columns),
            "missing_obs_columns": missing_obs,
            "missing_layers": missing_layers,
            "missing_obsm": missing_obsm,
            "has_counts_layer": "counts" in data.layers.keys(),
            "has_spatial_obsm": "spatial" in data.obsm.keys(),
            "schema_passed": not missing_obs and not missing_layers and not missing_obsm,
        }
    finally:
        _close_h5ad(data)


def load_l126_cellbin_table(path: str | Path, sample_id: str | None = None) -> pd.DataFrame:
    import anndata as ad

    h5ad_path = Path(path).expanduser().resolve()
    data = ad.read_h5ad(h5ad_path, backed="r")
    try:
        schema = validate_l126_h5ad_schema(h5ad_path)
        if not schema["schema_passed"]:
            raise ValueError(f"L126 h5ad schema validation failed: {schema}")
        obs = data.obs.reset_index(names="obs_index").copy()
        obs["obs_position"] = np.arange(len(obs), dtype=int)
        if "section_order" not in obs:
            obs["section_order"] = pd.NA
        keep = ["sample_id", "slice_id", "section_order", "cellbin_id", "x", "y", "obs_index", "obs_position"]
        table = obs[keep].copy()
        if sample_id is not None:
            table = table.loc[table["sample_id"].astype(str) == str(sample_id)].copy()
        table["section_order"] = pd.to_numeric(table["section_order"], errors="coerce").astype("Int64")
        table["x"] = pd.to_numeric(table["x"], errors="raise")
        table["y"] = pd.to_numeric(table["y"], errors="raise")
        duplicate_count = int(table.duplicated(["sample_id", "slice_id", "cellbin_id"]).sum())
        if duplicate_count:
            raise ValueError(f"Duplicate L126 cellbin primary keys: {duplicate_count}")
        return table.reset_index(drop=True)
    finally:
        _close_h5ad(data)


def h5ad_path_for_sample(packet_root: str | Path, sample_id: str) -> Path:
    return (
        Path(packet_root).expanduser().resolve()
        / "processed"
        / "h5ad"
        / f"{sample_id}.mRNA_processed.h5ad"
    )
