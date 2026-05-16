from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csgraph

from .schemas import *


def build_sparse_matrix_stats(matrix: sp.csr_matrix, include_components: bool) -> dict[str, Any]:
    row_nnz = np.diff(matrix.indptr).astype(np.int64)
    row_sums = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    zero_rows = row_nnz == 0
    expected = np.where(zero_rows, 0.0, 1.0)
    row_error = np.abs(row_sums - expected)
    result: dict[str, Any] = {
        "state_count": int(matrix.shape[0]),
        "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
        "nnz": int(matrix.nnz),
        "nnz_per_row_min": int(row_nnz.min()) if row_nnz.size else 0,
        "nnz_per_row_median": float(np.median(row_nnz)) if row_nnz.size else 0.0,
        "nnz_per_row_p99": float(np.quantile(row_nnz, 0.99)) if row_nnz.size else 0.0,
        "nnz_per_row_max": int(row_nnz.max()) if row_nnz.size else 0,
        "zero_row_count": int(zero_rows.sum()),
        "row_sum_max_error": float(row_error.max()) if row_error.size else 0.0,
        "row_sum_p99_error": float(np.quantile(row_error, 0.99)) if row_error.size else 0.0,
        "negative_entries": int(np.count_nonzero(matrix.data < 0)),
        "self_loop_mass": float(matrix.diagonal().sum()) if matrix.shape[0] == matrix.shape[1] else None,
        "row_entropy_min": None,
        "row_entropy_median": None,
        "row_entropy_p99": None,
        "weak_component_count": None,
        "strong_component_count": None,
        "largest_weak_component_fraction": None,
        "component_summary": "skipped",
    }
    if matrix.nnz:
        row_ids = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), row_nnz)
        probs = matrix.data.astype(float) / row_sums[row_ids]
        probs = np.clip(probs, 1e-300, 1.0)
        entropy = np.bincount(
            row_ids,
            weights=-probs * np.log(probs),
            minlength=matrix.shape[0],
        )
        result.update(
            {
                "row_entropy_min": float(entropy.min()),
                "row_entropy_median": float(np.median(entropy)),
                "row_entropy_p99": float(np.quantile(entropy, 0.99)),
            }
        )
    if include_components:
        weak_count, weak_labels = csgraph.connected_components(
            matrix, directed=True, connection="weak", return_labels=True
        )
        strong_count, _ = csgraph.connected_components(
            matrix, directed=True, connection="strong", return_labels=True
        )
        weak_sizes = np.bincount(weak_labels)
        result.update(
            {
                "weak_component_count": int(weak_count),
                "strong_component_count": int(strong_count),
                "largest_weak_component_fraction": float(weak_sizes.max() / matrix.shape[0]),
                "component_summary": f"weak={int(weak_count)} strong={int(strong_count)}",
            }
        )
    return result


def sparse_matrix_suitability(stats: dict[str, Any], artifact_role: str) -> tuple[str, str]:
    if artifact_role == "gpcca_output_transition_matrix":
        return "already_gpcca_output", "This is a downstream GPCCA output, not an input kernel."
    if stats["state_count"] > MATRIX_STATE_COMPONENT_LIMIT or stats["nnz"] > MATRIX_NNZ_COMPONENT_LIMIT:
        return "no", (
            f"Row-stochastic but too large for a full GPCCA run in this sprint "
            f"({stats['state_count']:,} states; {stats['nnz']:,} nnz)."
        )
    if stats["row_sum_max_error"] > 1e-5:
        return "no", "Row-stochastic error exceeds the inspect-only tolerance."
    if stats.get("weak_component_count") not in (None, 0, 1):
        frac = stats.get("largest_weak_component_fraction")
        if frac is not None and frac >= 0.995:
            return "conditional", (
                f"Mostly connected but not fully weakly connected "
                f"({stats['weak_component_count']} weak components; largest fraction {frac:.4f})."
            )
        return "no", (
            f"Disconnected directed graph ({stats['weak_component_count']} weak components)."
        )
    return "yes", "Sparse row-stochastic kernel is suitable for a GPCCA pilot."


def sparse_kernel_qc(
    matrix: sp.csr_matrix,
    edge_table: pd.DataFrame,
    state_metadata: pd.DataFrame,
    matrix_path: Path | None = None,
) -> dict[str, Any]:
    stats = build_sparse_matrix_stats(matrix, include_components=matrix.shape[0] <= 10_000 and matrix.nnz <= 2_000_000)
    row_nnz = np.diff(matrix.indptr)
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    nonzero_rows = row_sums > 0
    edge_valid = True
    invalid_edges = 0
    if not edge_table.empty:
        invalid = edge_table[
            (edge_table["edge_kind"] == "adjacent_time_transition")
            & (pd.to_numeric(edge_table["target_time_day"]) <= pd.to_numeric(edge_table["source_time_day"]))
        ]
        invalid_edges = int(len(invalid))
        edge_valid = invalid_edges == 0
    in_degree = np.bincount(matrix.indices, minlength=matrix.shape[0]) if matrix.nnz else np.zeros(matrix.shape[0])
    qc = {
        **stats,
        "matrix_path": str(matrix_path) if matrix_path else None,
        "effective_outgoing_edges_mean": float(np.mean(row_nnz[nonzero_rows])) if np.any(nonzero_rows) else 0.0,
        "effective_outgoing_edges_median": float(np.median(row_nnz[nonzero_rows])) if np.any(nonzero_rows) else 0.0,
        "source_coverage_fraction": float(np.mean(row_nnz > 0)) if row_nnz.size else 0.0,
        "target_coverage_fraction": float(np.mean(in_degree > 0)) if in_degree.size else 0.0,
        "time_direction_valid": edge_valid,
        "invalid_time_edge_count": invalid_edges,
        "terminal_self_loop_rows": int((matrix.diagonal() > 0).sum()),
        "row_stochastic": bool(stats["row_sum_max_error"] <= 1e-10 and stats["zero_row_count"] == 0),
        "kernel_too_sparse": bool(stats["nnz_per_row_median"] < 5),
        "kernel_too_dense": bool(stats["nnz_per_row_median"] > 50),
        "state_count": int(matrix.shape[0]),
        "time_day_count": int(state_metadata["dominant_time_day"].nunique()) if "dominant_time_day" in state_metadata else None,
    }
    qc["kernel_qc_pass"] = bool(
        qc["row_stochastic"]
        and qc["time_direction_valid"]
        and not qc["kernel_too_sparse"]
        and not qc["kernel_too_dense"]
        and qc["source_coverage_fraction"] == 1.0
    )
    return qc


def _matrix_row_columns(matrix: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    row_nnz = np.diff(matrix.indptr)
    row_ids = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), row_nnz)
    col_ids = matrix.indices.astype(np.int64, copy=False)
    return row_ids, col_ids


def strong_component_closure_summary(matrix: sp.csr_matrix) -> dict[str, Any]:
    if matrix.shape[0] == 0:
        return {
            "strong_component_count": 0,
            "closed_class_count": 0,
            "largest_closed_class_size": 0,
            "singleton_closed_class_count": 0,
        }
    strong_count, strong_labels = csgraph.connected_components(
        matrix,
        directed=True,
        connection="strong",
        return_labels=True,
    )
    row_ids, col_ids = _matrix_row_columns(matrix)
    row_labels = strong_labels[row_ids]
    col_labels = strong_labels[col_ids]
    outgoing = np.zeros(strong_count, dtype=bool)
    cross_component = row_labels != col_labels
    if cross_component.any():
        outgoing[np.unique(row_labels[cross_component])] = True
    sizes = np.bincount(strong_labels, minlength=strong_count)
    closed_sizes = sizes[~outgoing]
    return {
        "strong_component_count": int(strong_count),
        "closed_class_count": int((~outgoing).sum()),
        "largest_closed_class_size": int(closed_sizes.max()) if closed_sizes.size else 0,
        "singleton_closed_class_count": int(np.count_nonzero(closed_sizes == 1)),
        "strong_component_sizes": [int(value) for value in sizes.tolist()],
    }


def transition_mass_summary(matrix: sp.csr_matrix, state_metadata: pd.DataFrame) -> dict[str, Any]:
    times = pd.to_numeric(state_metadata["dominant_time_day"], errors="coerce").to_numpy(dtype=float)
    row_ids, col_ids = _matrix_row_columns(matrix)
    if matrix.nnz == 0:
        return {
            "forward_time_mass": 0.0,
            "within_time_nonself_mass": 0.0,
            "backward_time_mass": 0.0,
            "self_loop_mass": 0.0,
            "forward_time_mass_fraction": 0.0,
            "within_time_nonself_mass_fraction": 0.0,
            "backward_time_mass_fraction": 0.0,
            "self_loop_mass_fraction": 0.0,
            "time_block_edges": [],
        }
    src_time = times[row_ids]
    dst_time = times[col_ids]
    data = matrix.data.astype(float, copy=False)
    is_self = row_ids == col_ids
    forward_mask = dst_time > src_time
    within_mask = (dst_time == src_time) & (~is_self)
    backward_mask = dst_time < src_time
    self_mask = is_self
    total_mass = float(matrix.shape[0]) if matrix.shape[0] else 1.0
    edge_frame = pd.DataFrame(
        {
            "source_time": src_time,
            "target_time": dst_time,
            "probability": data,
        }
    )
    block_frame = (
        edge_frame.groupby(["source_time", "target_time"], dropna=False)["probability"]
        .sum()
        .reset_index()
        .sort_values(["source_time", "target_time"])
    )
    return {
        "forward_time_mass": float(data[forward_mask].sum()),
        "within_time_nonself_mass": float(data[within_mask].sum()),
        "backward_time_mass": float(data[backward_mask].sum()),
        "self_loop_mass": float(data[self_mask].sum()),
        "forward_time_mass_fraction": float(data[forward_mask].sum() / total_mass),
        "within_time_nonself_mass_fraction": float(data[within_mask].sum() / total_mass),
        "backward_time_mass_fraction": float(data[backward_mask].sum() / total_mass),
        "self_loop_mass_fraction": float(data[self_mask].sum() / total_mass),
        "time_block_edges": block_frame.to_dict(orient="records"),
    }


def dense_eigen_diagnostics(matrix: sp.csr_matrix) -> dict[str, Any]:
    if matrix.shape[0] == 0 or matrix.shape[0] > 1_000:
        return {
            "computed": False,
            "reason": "matrix too large or empty",
        }
    dense = matrix.toarray().astype(float, copy=False)
    eigvals = np.linalg.eigvals(dense)
    abs_vals = np.abs(eigvals)
    sorted_idx = np.argsort(-abs_vals)
    top_abs = eigvals[sorted_idx[:10]]
    unique_one = int(np.count_nonzero(np.abs(abs_vals - 1.0) <= 1e-6))
    near_one = int(np.count_nonzero(np.abs(abs_vals - 1.0) <= 1e-3))
    second = float(abs_vals[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
    return {
        "computed": True,
        "top_eigenvalues_abs_sorted": [float(np.abs(value)) for value in top_abs],
        "top_eigenvalues_real_sorted": [float(np.real(value)) for value in top_abs],
        "unit_eigenvalue_count_tol_1e6": unique_one,
        "near_unit_eigenvalue_count_tol_1e3": near_one,
        "spectral_gap_abs": float(max(0.0, 1.0 - second)),
    }


def row_normalize_csr(matrix: sp.spmatrix, zero_row_self_loop: bool = False) -> sp.csr_matrix:
    matrix = matrix.tocsr(copy=True)
    if zero_row_self_loop:
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        zero_rows = np.where(row_sums <= 0)[0]
        if zero_rows.size:
            matrix = matrix.tolil(copy=False)
            matrix[zero_rows, zero_rows] = 1.0
            matrix = matrix.tocsr()
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sums, dtype=np.float64)
    mask = row_sums > 0
    inv[mask] = 1.0 / row_sums[mask]
    normalized = sp.diags(inv).dot(matrix).tocsr()
    normalized.eliminate_zeros()
    return normalized


def sparse_matrix_edge_table(matrix: sp.csr_matrix, state_metadata: pd.DataFrame, edge_kind: str) -> pd.DataFrame:
    row_ids, col_ids = _matrix_row_columns(matrix)
    if row_ids.size == 0:
        return pd.DataFrame(
            columns=[
                "source_state_index",
                "target_state_index",
                "source_metaniche_id",
                "target_metaniche_id",
                "source_time_day",
                "target_time_day",
                "edge_kind",
                "probability",
            ]
        )
    return pd.DataFrame(
        {
            "source_state_index": row_ids,
            "target_state_index": col_ids,
            "source_metaniche_id": state_metadata.iloc[row_ids]["metaniche_id"].to_numpy(),
            "target_metaniche_id": state_metadata.iloc[col_ids]["metaniche_id"].to_numpy(),
            "source_time_day": state_metadata.iloc[row_ids]["dominant_time_day"].to_numpy(),
            "target_time_day": state_metadata.iloc[col_ids]["dominant_time_day"].to_numpy(),
            "edge_kind": edge_kind,
            "probability": matrix.data.astype(float, copy=False),
        }
    )


__all__ = [name for name in globals() if not name.startswith("__")]
