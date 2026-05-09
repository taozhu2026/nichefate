"""Spatial coordinate helpers for M0."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_spatial_coordinates(coordinates: object) -> None:
    """Validate that coordinates look like an n by d coordinate matrix."""

    shape = getattr(coordinates, "shape", None)
    if shape is None or len(shape) != 2 or shape[1] < 2:
        raise ValueError("Spatial coordinates must have shape (n_observations, >=2).")


def build_spatial_graph(
    _coordinates: object,
    *,
    radius: float | None = None,
    n_neighbors: int | None = None,
) -> object:
    """Build a spatial graph from coordinate data."""

    if radius is None and n_neighbors is None:
        raise ValueError("Either radius or n_neighbors must be provided.")
    raise NotImplementedError("Spatial graph construction is not implemented yet.")


def set_spatial_obsm(adata: Any, x_field: str = "x", y_field: str = "y") -> Any:
    """Set raw spatial coordinate arrays in AnnData obsm."""

    for field in (x_field, y_field):
        if field not in adata.obs.columns:
            raise ValueError(f"Missing spatial coordinate field in obs: {field}")
    coords = adata.obs[[x_field, y_field]].to_numpy(dtype=float)
    validate_spatial_coordinates(coords)
    adata.obsm["spatial"] = coords
    adata.obsm["X_spatial"] = coords.copy()
    return adata


def normalize_spatial_by_slice(
    adata: Any,
    slice_key: str = "slice_id",
) -> pd.DataFrame:
    """Center and scale spatial coordinates independently for each slice."""

    if "X_spatial" not in adata.obsm:
        raise ValueError('Missing obsm["X_spatial"]; call set_spatial_obsm first.')
    if slice_key not in adata.obs.columns:
        raise ValueError(f"Missing slice key in obs: {slice_key}")

    coords = np.asarray(adata.obsm["X_spatial"], dtype=float)
    normalized = np.zeros_like(coords, dtype=float)
    params: list[dict[str, object]] = []

    slice_values = adata.obs[slice_key].astype(str)
    for slice_id in sorted(slice_values.unique()):
        mask = slice_values.to_numpy() == slice_id
        slice_coords = coords[mask]
        center = np.nanmedian(slice_coords, axis=0)
        centered = slice_coords - center
        scale = float(np.nanmedian(np.linalg.norm(centered, axis=1)))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        normalized[mask] = centered / scale
        params.append(
            {
                "slice_id": slice_id,
                "n_cells": int(mask.sum()),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "scale": scale,
            }
        )

    adata.obsm["X_spatial_norm"] = normalized
    return pd.DataFrame(params)
