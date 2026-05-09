"""M1 spatial niche prototype helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


def _as_anchor_indices(n_obs: int, anchor_indices: Any | None) -> np.ndarray:
    if anchor_indices is None:
        return np.arange(n_obs, dtype=np.int64)
    anchors = np.asarray(anchor_indices, dtype=np.int64)
    if anchors.ndim != 1:
        raise ValueError("anchor_indices must be one-dimensional.")
    if anchors.size and (anchors.min() < 0 or anchors.max() >= n_obs):
        raise IndexError("anchor_indices are outside the AnnData observation range.")
    return anchors


def _graph_csr(adata: Any, graph_key: str) -> sparse.csr_matrix:
    if graph_key not in adata.obsp:
        raise KeyError(f"Missing graph in adata.obsp: {graph_key}")
    matrix = adata.obsp[graph_key]
    if not sparse.issparse(matrix):
        raise TypeError(f"Graph is not a scipy sparse matrix: {graph_key}")
    return matrix.tocsr()


def safe_feature_token(value: object) -> str:
    """Return the stable token used in M1 feature column names."""

    text = str(value)
    token = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return token or "na"


def cell_type_composition_prefix(cell_type_key: str) -> str:
    """Return the M1 composition prefix for a configured cell-type key."""

    return {
        "cell_type_l1": "ct_l1",
        "cell_type_l2": "ct_l2",
        "cell_type_l3": "ct_l3",
    }.get(cell_type_key, safe_feature_token(cell_type_key))


def _safe_token(value: object) -> str:
    return safe_feature_token(value)


def _cell_type_prefix(cell_type_key: str) -> str:
    return cell_type_composition_prefix(cell_type_key)


def load_global_feature_schema(path: str | Path | None) -> dict[str, Any] | None:
    """Load an optional global M1 feature schema."""

    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def align_feature_table_to_schema(
    table: pd.DataFrame,
    schema: dict[str, Any] | None,
) -> pd.DataFrame:
    """Align a slice-local feature table to an optional global M1 schema."""

    if not schema:
        return table
    expected_columns = list(schema.get("feature_columns", []))
    if not expected_columns:
        return table

    aligned = table.copy()
    composition_columns = set(schema.get("composition_columns", []))
    for column in expected_columns:
        if column not in aligned:
            aligned[column] = 0.0 if column in composition_columns else pd.NA
    extra_columns = [column for column in aligned.columns if column not in expected_columns]
    return aligned[expected_columns + extra_columns]


def get_graph_neighbors(
    adata: Any,
    graph_key: str,
    anchor_indices: Any | None = None,
    include_anchor: bool = True,
) -> list[np.ndarray]:
    """Return sparse-graph neighbors for each requested anchor."""

    graph = _graph_csr(adata, graph_key)
    anchors = _as_anchor_indices(adata.n_obs, anchor_indices)
    neighbors: list[np.ndarray] = []
    for anchor in anchors:
        row = graph.getrow(int(anchor))
        indices = row.indices.astype(np.int64, copy=False)
        if include_anchor:
            indices = np.union1d(indices, np.array([anchor], dtype=np.int64))
        else:
            indices = indices[indices != anchor]
        neighbors.append(np.sort(indices.astype(np.int64, copy=False)))
    return neighbors


def compute_neighbor_index(
    adata: Any,
    graph_key: str,
    anchor_indices: Any | None = None,
    include_anchor: bool = True,
) -> dict[str, np.ndarray]:
    """Return a ragged neighbor index for a graph and anchor set."""

    anchors = _as_anchor_indices(adata.n_obs, anchor_indices)
    neighbor_rows = get_graph_neighbors(adata, graph_key, anchors, include_anchor)
    lengths = np.array([len(row) for row in neighbor_rows], dtype=np.int64)
    indptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(lengths)
    flat = (
        np.concatenate(neighbor_rows).astype(np.int64, copy=False)
        if neighbor_rows
        else np.array([], dtype=np.int64)
    )
    return {"anchor_indices": anchors, "indptr": indptr, "neighbor_indices": flat}


def _iter_neighbor_rows(neighbor_index: dict[str, np.ndarray]):
    indptr = neighbor_index["indptr"]
    flat = neighbor_index["neighbor_indices"]
    for row_idx in range(len(indptr) - 1):
        yield flat[indptr[row_idx] : indptr[row_idx + 1]]


def compute_celltype_composition(
    adata: Any,
    neighbor_index: dict[str, np.ndarray],
    cell_type_key: str,
) -> pd.DataFrame:
    """Compute normalized cell-type composition for each anchor niche."""

    if cell_type_key not in adata.obs:
        raise KeyError(f"Missing obs field: {cell_type_key}")
    labels = adata.obs[cell_type_key].astype(str).to_numpy()
    levels = sorted(pd.unique(labels).tolist())
    prefix = _cell_type_prefix(cell_type_key)
    columns = [f"{prefix}__{_safe_token(level)}" for level in levels]
    data = np.zeros((len(neighbor_index["anchor_indices"]), len(levels)), dtype=float)
    level_pos = {level: idx for idx, level in enumerate(levels)}
    for row_idx, neighbors in enumerate(_iter_neighbor_rows(neighbor_index)):
        if len(neighbors) == 0:
            continue
        values, counts = np.unique(labels[neighbors], return_counts=True)
        for value, count in zip(values, counts, strict=False):
            data[row_idx, level_pos[value]] = count / len(neighbors)
    return pd.DataFrame(data, columns=columns)


def compute_shannon_entropy_from_composition(composition: pd.DataFrame) -> pd.Series:
    """Compute finite Shannon entropy from row-wise composition values."""

    values = composition.to_numpy(dtype=float)
    positive = values > 0
    logs = np.zeros_like(values)
    logs[positive] = np.log(values[positive])
    entropy = -(values * logs).sum(axis=1)
    return pd.Series(entropy, index=composition.index)


def compute_embedding_summary(
    adata: Any,
    neighbor_index: dict[str, np.ndarray],
    embedding_key: str = "X_pca_m0",
) -> pd.DataFrame:
    """Compute mean and variance of neighbor embeddings for each anchor."""

    if embedding_key not in adata.obsm:
        raise KeyError(f"Missing embedding in adata.obsm: {embedding_key}")
    embedding = np.asarray(adata.obsm[embedding_key], dtype=float)
    n_dims = embedding.shape[1]
    means = np.full((len(neighbor_index["anchor_indices"]), n_dims), np.nan)
    variances = np.full_like(means, np.nan)
    for row_idx, neighbors in enumerate(_iter_neighbor_rows(neighbor_index)):
        if len(neighbors) == 0:
            continue
        values = embedding[neighbors]
        means[row_idx] = values.mean(axis=0)
        variances[row_idx] = values.var(axis=0)
    mean_cols = [f"emb_mean_pc{idx:03d}" for idx in range(1, n_dims + 1)]
    var_cols = [f"emb_var_pc{idx:03d}" for idx in range(1, n_dims + 1)]
    return pd.DataFrame(np.hstack([means, variances]), columns=mean_cols + var_cols)


def compute_spatial_summary(
    adata: Any,
    neighbor_index: dict[str, np.ndarray],
    spatial_key: str = "X_spatial_norm",
) -> pd.DataFrame:
    """Compute distance summaries and a relative pseudo-density feature."""

    if spatial_key not in adata.obsm:
        raise KeyError(f"Missing spatial coordinates in adata.obsm: {spatial_key}")
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)[:, :2]
    anchors = neighbor_index["anchor_indices"]
    rows = []
    for anchor, neighbors in zip(anchors, _iter_neighbor_rows(neighbor_index), strict=False):
        other = neighbors[neighbors != anchor]
        if len(other) == 0:
            mean_distance = 0.0
            pseudo_density = 0.0
        else:
            distances = np.linalg.norm(coords[other] - coords[int(anchor)], axis=1)
            mean_distance = float(distances.mean())
            pseudo_density = float(len(neighbors) / max(mean_distance, 1e-12))
        rows.append(
            {
                "n_neighbors": int(len(neighbors)),
                "mean_neighbor_distance": mean_distance,
                "pseudo_local_density": pseudo_density,
            }
        )
    return pd.DataFrame(rows)


def compute_topology_summary(
    adata: Any,
    neighbor_index: dict[str, np.ndarray],
    topology_graph_key: str = "delaunay",
) -> pd.DataFrame:
    """Compute local Delaunay degree summaries for each anchor niche."""

    graph = _graph_csr(adata, topology_graph_key)
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    rows = []
    for anchor, neighbors in zip(
        neighbor_index["anchor_indices"], _iter_neighbor_rows(neighbor_index), strict=False
    ):
        mean_degree = float(degrees[neighbors].mean()) if len(neighbors) else 0.0
        rows.append(
            {
                "local_topology_anchor_degree": float(degrees[int(anchor)]),
                "local_topology_mean_member_degree": mean_degree,
            }
        )
    return pd.DataFrame(rows)


def build_basic_niche_feature_table(
    adata: Any,
    neighbor_index: dict[str, np.ndarray],
    *,
    scale: str,
    slice_file: str,
    cell_type_keys: list[str],
    embedding_key: str = "X_pca_m0",
    spatial_key: str = "X_spatial_norm",
    topology_graph_key: str = "delaunay",
    global_schema: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build one prototype M1 feature row per anchor for one scale."""

    anchors = neighbor_index["anchor_indices"]
    obs = adata.obs.iloc[anchors]
    table = pd.DataFrame(
        {
            "slice_id": obs["slice_id"].astype(str).to_numpy(),
            "slice_file": slice_file,
            "scale": scale,
            "anchor_index": anchors.astype(int),
            "anchor_cell_id": obs.index.astype(str),
        }
    )
    for key in ("time", "time_day", "mouse_id", "cell_type_l1", "cell_type_l2", "cell_type_l3", "x", "y"):
        if key in obs:
            table[key] = obs[key].to_numpy()
    for key in cell_type_keys:
        composition = compute_celltype_composition(adata, neighbor_index, key)
        table = pd.concat([table, composition], axis=1)
        table[f"{_cell_type_prefix(key)}_entropy"] = (
            compute_shannon_entropy_from_composition(composition).to_numpy()
        )
    features = pd.concat(
        [
            table,
            compute_embedding_summary(adata, neighbor_index, embedding_key),
            compute_spatial_summary(adata, neighbor_index, spatial_key),
            compute_topology_summary(adata, neighbor_index, topology_graph_key),
        ],
        axis=1,
    )
    return align_feature_table_to_schema(features, global_schema)


def write_neighbor_index_npz(entries: list[dict[str, Any]], path: str | Path) -> Path:
    """Write unambiguous per-slice/per-scale ragged neighbor indices."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    metadata = []
    for idx, entry in enumerate(entries):
        prefix = f"entry_{idx:03d}"
        neighbor_index = entry["neighbor_index"]
        arrays[f"{prefix}__anchor_indices"] = neighbor_index["anchor_indices"]
        arrays[f"{prefix}__indptr"] = neighbor_index["indptr"]
        arrays[f"{prefix}__neighbor_indices"] = neighbor_index["neighbor_indices"]
        metadata.append(
            {
                "entry": prefix,
                "slice_id": str(entry["slice_id"]),
                "slice_file": str(entry["slice_file"]),
                "scale": str(entry["scale"]),
                "n_anchors": int(len(neighbor_index["anchor_indices"])),
                "n_neighbor_links": int(len(neighbor_index["neighbor_indices"])),
            }
        )
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(output_path, **arrays)
    return output_path


def write_niche_feature_table_parquet_or_csv(table: pd.DataFrame, path: str | Path) -> Path:
    """Write a feature table, falling back to CSV when parquet is unavailable."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        try:
            table.to_parquet(output_path, index=False)
            return output_path
        except ImportError:
            output_path = output_path.with_suffix(".csv")
    table.to_csv(output_path, index=False)
    return output_path
