import sys
import importlib.util
from pathlib import Path

import numpy as np
from scipy import sparse

from nichefate.graph import (
    build_delaunay_graph,
    build_knn_graph,
    build_radius_graph,
    compute_median_nn_distance,
    summarize_sparse_graph,
)


def toy_coords() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )


def test_radius_graph_returns_csr() -> None:
    matrix = build_radius_graph(toy_coords(), radius=1.1)

    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (4, 4)
    assert matrix.nnz > 0


def test_knn_graph_returns_csr() -> None:
    matrix = build_knn_graph(toy_coords(), k=2)

    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (4, 4)
    assert matrix.nnz > 0


def test_delaunay_graph_returns_csr() -> None:
    matrix = build_delaunay_graph(toy_coords())

    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (4, 4)
    assert matrix.nnz > 0


def test_graph_summary_and_no_squidpy_dependency() -> None:
    matrix = build_knn_graph(toy_coords(), k=1)
    summary = summarize_sparse_graph(matrix, "knn_k1", "sliceA")

    assert summary["slice_id"] == "sliceA"
    assert summary["graph_name"] == "knn_k1"
    assert summary["n_nodes"] == 4
    assert compute_median_nn_distance(toy_coords()) > 0
    assert "squidpy" not in sys.modules


def test_graph_script_expected_names_and_failed_slice_parser(tmp_path: Path) -> None:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "m0_04_build_spatial_graphs.py"
    )
    spec = importlib.util.spec_from_file_location("m0_04_build_spatial_graphs", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    names = module.expected_graph_names(
        {
            "adaptive_radius_multipliers": [2, 4, 8],
            "knn_values": [6, 12],
            "build_delaunay": True,
        }
    )
    assert names == ["radius_x2", "radius_x4", "radius_x8", "knn_k6", "knn_k12", "delaunay"]

    failed_path = tmp_path / "failed_slices.txt"
    failed_path.write_text("sliceA\tValueError: bad\nsliceB\tRuntimeError: worse\n")
    assert module.failed_slices(failed_path) == {"sliceA", "sliceB"}
