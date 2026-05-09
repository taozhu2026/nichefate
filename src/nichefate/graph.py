"""Sparse graph builders for M0 spatial neighborhoods."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay, QhullError
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph


def attach_graph_to_anndata(_adata: object, _graph: object, *, key: str) -> object:
    """Attach a graph object to an AnnData object under a named key."""

    raise NotImplementedError(f"Graph export is not implemented yet for key: {key}")


def _as_coordinates(coords: Any) -> np.ndarray:
    array = np.asarray(coords, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("Coordinates must have shape (n_cells, >=2).")
    return array[:, :2]


def compute_median_nn_distance(coords: Any) -> float:
    """Compute the median first-nearest-neighbor distance."""

    array = _as_coordinates(coords)
    if array.shape[0] < 2:
        return 0.0
    model = NearestNeighbors(n_neighbors=2)
    model.fit(array)
    distances, _indices = model.kneighbors(array)
    return float(np.median(distances[:, 1]))


def build_radius_graph(coords: Any, radius: float) -> sparse.csr_matrix:
    """Build an undirected radius graph as CSR."""

    if radius <= 0:
        raise ValueError("radius must be positive.")
    matrix = radius_neighbors_graph(
        _as_coordinates(coords),
        radius=radius,
        mode="connectivity",
        include_self=False,
    ).tocsr()
    matrix = matrix.maximum(matrix.T).tocsr()
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix


def build_knn_graph(coords: Any, k: int) -> sparse.csr_matrix:
    """Build an undirected kNN graph as CSR."""

    array = _as_coordinates(coords)
    if k <= 0:
        raise ValueError("k must be positive.")
    if array.shape[0] <= 1:
        return sparse.csr_matrix((array.shape[0], array.shape[0]))
    n_neighbors = min(k + 1, array.shape[0])
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(array)
    matrix = model.kneighbors_graph(array, mode="connectivity").tocsr()
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    matrix = matrix.maximum(matrix.T).tocsr()
    return matrix


def build_delaunay_graph(coords: Any) -> sparse.csr_matrix:
    """Build an undirected Delaunay adjacency graph as CSR."""

    array = _as_coordinates(coords)
    n_cells = array.shape[0]
    if n_cells < 3:
        return sparse.csr_matrix((n_cells, n_cells))
    try:
        triangulation = Delaunay(array)
    except QhullError:
        return sparse.csr_matrix((n_cells, n_cells))

    edges: set[tuple[int, int]] = set()
    for simplex in triangulation.simplices:
        for i, source in enumerate(simplex):
            for target in simplex[i + 1 :]:
                a, b = sorted((int(source), int(target)))
                edges.add((a, b))
    if not edges:
        return sparse.csr_matrix((n_cells, n_cells))
    rows, cols = zip(*edges)
    data = np.ones(len(edges), dtype=np.uint8)
    upper = sparse.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    return (upper + upper.T).tocsr()


def summarize_sparse_graph(
    matrix: sparse.spmatrix,
    graph_name: str,
    slice_id: str,
) -> dict[str, object]:
    """Return a compact degree summary for a sparse graph."""

    csr = matrix.tocsr()
    degrees = np.asarray(csr.sum(axis=1)).ravel()
    return {
        "slice_id": slice_id,
        "graph_name": graph_name,
        "n_nodes": int(csr.shape[0]),
        "n_edges": int(csr.nnz // 2),
        "mean_degree": float(degrees.mean()) if degrees.size else 0.0,
        "median_degree": float(np.median(degrees)) if degrees.size else 0.0,
        "max_degree": float(degrees.max()) if degrees.size else 0.0,
    }
