"""M3 direction-aware transition evidence helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateNeighbors:
    """KNN candidate result."""

    indices: np.ndarray
    distances: np.ndarray
    backend: str
    metric: str


@dataclass(frozen=True)
class CandidateNeighborBackendStatus:
    """Runtime availability status for a candidate-neighbor backend."""

    backend: str
    metric: str
    package: str | None
    importable: bool
    usable: bool
    available: bool
    reason: str
    version: str | None = None


REQUIRED_NEIGHBOR_BACKENDS = {"sklearn_exact", "numpy_chunked"}
OPTIONAL_ANN_BACKENDS = {
    "faiss": "faiss",
    "hnswlib": "hnswlib",
    "pynndescent": "pynndescent",
}
SUPPORTED_NEIGHBOR_BACKENDS = REQUIRED_NEIGHBOR_BACKENDS | set(OPTIONAL_ANN_BACKENDS)
ANN_BACKENDS = set(OPTIONAL_ANN_BACKENDS)


def columns_by_patterns(
    columns: Sequence[str],
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str] | None = None,
) -> list[str]:
    """Select columns containing any include pattern and no exclude pattern."""

    excludes = list(exclude_patterns or [])
    selected = []
    for column in columns:
        if include_patterns and not any(pattern in column for pattern in include_patterns):
            continue
        if any(pattern in column for pattern in excludes):
            continue
        selected.append(column)
    return selected


def resolve_transition_feature_groups(
    m2_schema: dict[str, Any],
    feature_config: dict[str, Any],
) -> dict[str, list[str]]:
    """Resolve configured M3 feature groups against M2 numeric columns."""

    numeric_columns = list(m2_schema["numeric_feature_columns"])
    groups: dict[str, list[str]] = {}
    for group_name, group in feature_config.items():
        groups[group_name] = columns_by_patterns(
            numeric_columns,
            group.get("include_patterns", []) or [],
            group.get("exclude_patterns", []) or [],
        )
    return groups


def infer_adjacent_time_pairs(
    metadata: pd.DataFrame,
    time_column: str = "time",
    time_day_column: str = "time_day",
) -> list[dict[str, Any]]:
    """Infer adjacent time pairs from sorted unique numeric day metadata."""

    required = {time_column, time_day_column, "slice_id"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise KeyError(f"Missing metadata columns for time-pair inference: {missing}")
    unique_times = (
        metadata[[time_column, time_day_column]]
        .drop_duplicates()
        .sort_values(time_day_column)
        .reset_index(drop=True)
    )
    pairs = []
    for idx in range(len(unique_times) - 1):
        source = unique_times.iloc[idx]
        target = unique_times.iloc[idx + 1]
        source_rows = metadata[metadata[time_day_column] == source[time_day_column]]
        target_rows = metadata[metadata[time_day_column] == target[time_day_column]]
        pair = {
            "source_time": source[time_column],
            "target_time": target[time_column],
            "source_day": float(source[time_day_column]),
            "target_day": float(target[time_day_column]),
            "time_delta": float(target[time_day_column] - source[time_day_column]),
            "source_row_count": int(source_rows["rows"].sum())
            if "rows" in source_rows
            else int(len(source_rows)),
            "target_row_count": int(target_rows["rows"].sum())
            if "rows" in target_rows
            else int(len(target_rows)),
            "source_slices": sorted(source_rows["slice_id"].astype(str).unique().tolist()),
            "target_slices": sorted(target_rows["slice_id"].astype(str).unique().tolist()),
        }
        if "mouse_id" in metadata.columns:
            pair["source_samples"] = sorted(
                source_rows["mouse_id"].dropna().astype(str).unique().tolist()
            )
            pair["target_samples"] = sorted(
                target_rows["mouse_id"].dropna().astype(str).unique().tolist()
            )
        pairs.append(pair)
    return pairs


def standardize_feature_matrices(
    source_matrix: np.ndarray,
    target_matrix: np.ndarray,
    min_scale: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Standardize source and target matrices with combined-sample column stats."""

    combined = np.vstack([source_matrix, target_matrix]).astype(float, copy=False)
    means = np.nanmean(combined, axis=0)
    stds = np.nanstd(combined, axis=0)
    valid = stds >= min_scale
    safe_stds = np.where(valid, stds, 1.0)
    source = (source_matrix - means) / safe_stds
    target = (target_matrix - means) / safe_stds
    source[:, ~valid] = 0.0
    target[:, ~valid] = 0.0
    return (
        np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0),
        {"zero_variance_columns": int((~valid).sum()), "total_columns": int(len(valid))},
    )


def build_candidate_neighbors(
    source_matrix: np.ndarray,
    target_matrix: np.ndarray,
    k: int,
    backend: str = "sklearn_exact",
    metric: str = "euclidean",
    chunk_size: int | None = None,
    random_seed: int = 1,
) -> CandidateNeighbors | CandidateNeighborBackendStatus:
    """Return k target candidates per source using a configurable backend."""

    source = np.asarray(source_matrix, dtype=float)
    target = np.asarray(target_matrix, dtype=float)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source_matrix and target_matrix must be two-dimensional.")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source_matrix and target_matrix must have the same features.")
    if len(target) == 0 or len(source) == 0:
        raise ValueError("source_matrix and target_matrix must be non-empty.")
    k_eff = min(int(k), len(target))
    if k_eff <= 0:
        raise ValueError("k must be positive.")
    if backend == "sklearn_exact":
        from sklearn.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=k_eff, metric=metric)
        model.fit(target)
        distances, indices = model.kneighbors(source, return_distance=True)
        return CandidateNeighbors(indices=indices, distances=distances, backend=backend, metric=metric)
    if backend == "numpy_chunked":
        if metric != "euclidean":
            raise ValueError("numpy_chunked currently supports only euclidean metric.")
        return _numpy_chunked_neighbors(source, target, k_eff, chunk_size or 512, metric)
    if backend == "faiss":
        status = inspect_candidate_neighbor_backend(backend, metric=metric, run_toy_check=False)
        if not status.available:
            return status
        return _faiss_neighbors(source, target, k_eff, metric)
    if backend == "hnswlib":
        status = inspect_candidate_neighbor_backend(backend, metric=metric, run_toy_check=False)
        if not status.available:
            return status
        return _hnswlib_neighbors(source, target, k_eff, metric, random_seed)
    if backend == "pynndescent":
        status = inspect_candidate_neighbor_backend(backend, metric=metric, run_toy_check=False)
        if not status.available:
            return status
        return _pynndescent_neighbors(source, target, k_eff, metric, random_seed)
    raise ValueError(f"Unsupported candidate-neighbor backend: {backend}")


def inspect_candidate_neighbor_backend(
    backend: str,
    metric: str = "euclidean",
    run_toy_check: bool = False,
) -> CandidateNeighborBackendStatus:
    """Return importability/usability status without requiring optional ANN packages."""

    if backend not in SUPPORTED_NEIGHBOR_BACKENDS:
        raise ValueError(f"Unsupported candidate-neighbor backend: {backend}")
    package = _backend_package(backend)
    if not _backend_metric_supported(backend, metric):
        return CandidateNeighborBackendStatus(
            backend=backend,
            metric=metric,
            package=package,
            importable=_backend_importable(package),
            usable=False,
            available=False,
            reason=f"{backend} does not support metric {metric!r} in M3.",
            version=None,
        )
    importable, version, reason = _backend_import_status(package)
    if not importable:
        return CandidateNeighborBackendStatus(
            backend=backend,
            metric=metric,
            package=package,
            importable=False,
            usable=False,
            available=False,
            reason=reason,
            version=version,
        )
    if not run_toy_check:
        return CandidateNeighborBackendStatus(
            backend=backend,
            metric=metric,
            package=package,
            importable=True,
            usable=True,
            available=True,
            reason="importable",
            version=version,
        )
    try:
        source = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
        target = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=float)
        result = build_candidate_neighbors(
            source,
            target,
            k=2,
            backend=backend,
            metric=metric,
            chunk_size=1,
            random_seed=1,
        )
        if isinstance(result, CandidateNeighborBackendStatus):
            return result
        usable = result.indices.shape == (2, 2) and result.distances.shape == (2, 2)
        return CandidateNeighborBackendStatus(
            backend=backend,
            metric=metric,
            package=package,
            importable=True,
            usable=usable,
            available=usable,
            reason="toy KNN query succeeded" if usable else "toy KNN query returned unexpected shape",
            version=version,
        )
    except Exception as exc:  # noqa: BLE001
        return CandidateNeighborBackendStatus(
            backend=backend,
            metric=metric,
            package=package,
            importable=True,
            usable=False,
            available=False,
            reason=f"toy KNN query failed: {exc}",
            version=version,
        )


def _backend_package(backend: str) -> str | None:
    if backend == "sklearn_exact":
        return "sklearn"
    if backend == "numpy_chunked":
        return "numpy"
    return OPTIONAL_ANN_BACKENDS.get(backend)


def _backend_metric_supported(backend: str, metric: str) -> bool:
    if backend == "sklearn_exact":
        return True
    return metric == "euclidean"


def _backend_importable(package: str | None) -> bool:
    if package is None:
        return True
    return importlib.util.find_spec(package) is not None


def _backend_import_status(package: str | None) -> tuple[bool, str | None, str]:
    if package is None:
        return True, None, "no package import required"
    if importlib.util.find_spec(package) is None:
        return False, None, f"package {package!r} is not importable"
    try:
        module = importlib.import_module(package)
    except Exception as exc:  # noqa: BLE001
        return False, None, f"package {package!r} import failed: {exc}"
    version = getattr(module, "__version__", None)
    return True, str(version) if version is not None else None, "importable"


def _faiss_neighbors(
    source: np.ndarray,
    target: np.ndarray,
    k: int,
    metric: str,
) -> CandidateNeighbors:
    if metric != "euclidean":
        raise ValueError("faiss backend currently supports only euclidean metric.")
    faiss = importlib.import_module("faiss")
    target32 = np.ascontiguousarray(target.astype("float32", copy=False))
    source32 = np.ascontiguousarray(source.astype("float32", copy=False))
    index = faiss.IndexFlatL2(target32.shape[1])
    index.add(target32)
    distances2, indices = index.search(source32, k)
    distances = np.sqrt(np.maximum(distances2.astype(float, copy=False), 0.0))
    return CandidateNeighbors(indices=indices.astype(np.int64, copy=False), distances=distances, backend="faiss", metric=metric)


def _hnswlib_neighbors(
    source: np.ndarray,
    target: np.ndarray,
    k: int,
    metric: str,
    random_seed: int,
) -> CandidateNeighbors:
    if metric != "euclidean":
        raise ValueError("hnswlib backend currently supports only euclidean metric.")
    hnswlib = importlib.import_module("hnswlib")
    target32 = target.astype("float32", copy=False)
    source32 = source.astype("float32", copy=False)
    index = hnswlib.Index(space="l2", dim=target32.shape[1])
    index.init_index(
        max_elements=len(target32),
        ef_construction=max(100, k * 4),
        M=16,
        random_seed=int(random_seed),
    )
    index.add_items(target32, np.arange(len(target32), dtype=np.int64))
    index.set_ef(max(50, k * 2))
    indices, distances2 = index.knn_query(source32, k=k)
    distances = np.sqrt(np.maximum(distances2.astype(float, copy=False), 0.0))
    return CandidateNeighbors(indices=indices.astype(np.int64, copy=False), distances=distances, backend="hnswlib", metric=metric)


def _pynndescent_neighbors(
    source: np.ndarray,
    target: np.ndarray,
    k: int,
    metric: str,
    random_seed: int,
) -> CandidateNeighbors:
    if metric != "euclidean":
        raise ValueError("pynndescent backend currently supports only euclidean metric.")
    pynndescent = importlib.import_module("pynndescent")
    n_neighbors = min(max(k, 2), len(target))
    index = pynndescent.NNDescent(
        target.astype("float32", copy=False),
        metric=metric,
        n_neighbors=n_neighbors,
        random_state=int(random_seed),
    )
    indices, distances = index.query(source.astype("float32", copy=False), k=k)
    return CandidateNeighbors(indices=indices.astype(np.int64, copy=False), distances=distances.astype(float, copy=False), backend="pynndescent", metric=metric)


def _numpy_chunked_neighbors(
    source: np.ndarray,
    target: np.ndarray,
    k: int,
    chunk_size: int,
    metric: str,
) -> CandidateNeighbors:
    indices = np.empty((len(source), k), dtype=np.int64)
    distances = np.empty((len(source), k), dtype=float)
    target_norm = np.sum(target * target, axis=1)
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        chunk = source[start:stop]
        dist2 = np.sum(chunk * chunk, axis=1)[:, None] + target_norm[None, :]
        dist2 -= 2.0 * chunk @ target.T
        np.maximum(dist2, 0.0, out=dist2)
        part = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        part_dist = np.take_along_axis(dist2, part, axis=1)
        order = np.argsort(part_dist, axis=1)
        sorted_idx = np.take_along_axis(part, order, axis=1)
        sorted_dist = np.take_along_axis(part_dist, order, axis=1)
        indices[start:stop] = sorted_idx
        distances[start:stop] = np.sqrt(sorted_dist)
    return CandidateNeighbors(indices=indices, distances=distances, backend="numpy_chunked", metric=metric)


def pairwise_row_distance(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    columns: Sequence[str],
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute row-wise distances for selected source-target candidate pairs."""

    if not columns:
        return np.zeros(len(source_indices), dtype=float)
    src = source.iloc[source_indices][list(columns)].to_numpy(dtype=float)
    tgt = target.iloc[target_indices][list(columns)].to_numpy(dtype=float)
    diff = src - tgt
    if metric == "l1":
        values = np.abs(diff).sum(axis=1)
    else:
        values = np.sqrt(np.square(diff).sum(axis=1))
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def safe_scale_vector(values: Sequence[float], min_scale: float = 1e-6) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale evidence with robust fallback and zero-variance protection."""

    arr = np.asarray(values, dtype=float)
    median = float(np.nanmedian(arr)) if arr.size else 0.0
    q75, q25 = np.nanpercentile(arr, [75, 25]) if arr.size else (0.0, 0.0)
    iqr = float(q75 - q25)
    mean = float(np.nanmean(arr)) if arr.size else 0.0
    std = float(np.nanstd(arr)) if arr.size else 0.0
    min_value = float(np.nanmin(arr)) if arr.size else 0.0
    max_value = float(np.nanmax(arr)) if arr.size else 0.0
    span = max_value - min_value
    if iqr >= min_scale:
        scaled = (arr - median) / iqr
        method = "median_iqr"
        zero_variance = False
    elif std >= min_scale:
        scaled = (arr - mean) / std
        method = "mean_std"
        zero_variance = False
    elif span >= min_scale:
        scaled = (arr - min_value) / span
        method = "min_range"
        zero_variance = False
    else:
        scaled = np.zeros_like(arr, dtype=float)
        method = "zero_variance"
        zero_variance = True
    stats = {
        "median": median,
        "iqr": iqr,
        "std": std,
        "min": min_value,
        "max": max_value,
        "scaling_method_used": method,
        "zero_variance": zero_variance,
    }
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0), stats


def combine_scaled_evidence(
    evidence: pd.DataFrame,
    weights: dict[str, float],
    scaled_column_by_group: dict[str, str],
) -> np.ndarray:
    """Compute combined cost from scaled evidence columns only."""

    total = np.zeros(len(evidence), dtype=float)
    for group, column in scaled_column_by_group.items():
        weight = float(weights.get(group, 0.0))
        if weight == 0.0 or column not in evidence:
            continue
        total += weight * evidence[column].to_numpy(dtype=float)
    return np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)


def pair_adaptive_temperature(cost: Sequence[float], min_temperature: float = 1e-3) -> float:
    """Use median combined cost as pair-specific temperature floor."""

    values = np.asarray(cost, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(min_temperature)
    return float(max(np.median(finite), min_temperature))


def row_normalize_weights(
    frame: pd.DataFrame,
    source_column: str = "source_anchor_id",
    weight_column: str = "mass_adjusted_weight",
) -> pd.Series:
    """Normalize candidate weights within each source row."""

    sums = frame.groupby(source_column, observed=True)[weight_column].transform("sum")
    values = frame[weight_column] / sums.replace(0.0, np.nan)
    return values.fillna(0.0)


def transition_probability_diagnostics(
    frame: pd.DataFrame,
    source_column: str = "source_anchor_id",
    prob_column: str = "row_normalized_transition_prob",
) -> dict[str, float]:
    """Summarize local candidate-set transition probabilities."""

    grouped = frame.groupby(source_column, observed=True)[prob_column]
    row_sums = grouped.sum()
    entropy = grouped.apply(lambda s: float(-(s * np.log(s.clip(lower=1e-300))).sum()))
    top1 = grouped.max()
    effective = np.exp(entropy)
    return {
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "row_sum_mean": float(row_sums.mean()),
        "row_entropy_mean": float(entropy.mean()),
        "top1_probability_mean": float(top1.mean()),
        "effective_targets_mean": float(effective.mean()),
    }


def categorical_target_diagnostics(
    frame: pd.DataFrame,
    source_column: str,
    target_column: str,
) -> dict[str, float]:
    """Summarize target category concentration per source group."""

    entropies = []
    top_fractions = []
    for _, values in frame.groupby(source_column, observed=True)[target_column]:
        counts = values.astype(str).value_counts(normalize=True)
        probs = counts.to_numpy(dtype=float)
        entropies.append(float(-(probs * np.log(probs.clip(min=1e-300))).sum()))
        top_fractions.append(float(probs.max()))
    return {
        f"{target_column}_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        f"top_{target_column}_fraction_mean": float(np.mean(top_fractions))
        if top_fractions
        else 0.0,
    }


def evidence_schema_columns() -> list[str]:
    """Return the M3 sampled edge evidence schema."""

    groups = [
        "molecular",
        "composition",
        "entropy",
        "spatial_summary",
        "topology",
        "pseudotime",
        "barcode",
    ]
    return [
        "source_anchor_id",
        "target_anchor_id",
        "source_anchor_index",
        "target_anchor_index",
        "source_time",
        "target_time",
        "source_day",
        "target_day",
        "time_delta",
        "source_slice_id",
        "target_slice_id",
        "source_slice_file",
        "target_slice_file",
        "source_mouse_id",
        "target_mouse_id",
        "evidence_mode",
        *[f"raw_{group}_distance" for group in groups[:5]],
        "raw_pseudotime_score",
        "raw_barcode_score",
        *[f"scaled_{group}_distance" for group in groups[:5]],
        "scaled_pseudotime_score",
        "scaled_barcode_score",
        "combined_cost",
        "tau_pair",
        "raw_edge_weight",
        "source_mass",
        "target_mass",
        "growth_prior",
        "unbalanced_weight",
        "mass_adjusted_weight",
        "row_normalized_transition_prob",
    ]


def full_transition_schema_columns() -> list[str]:
    """Return the planned full M3 shard edge schema."""

    evidence_groups = [
        "molecular",
        "composition",
        "entropy",
        "spatial_summary",
        "topology",
    ]
    return [
        "source_anchor_id",
        "target_anchor_id",
        "source_anchor_index",
        "target_anchor_index",
        "source_time",
        "target_time",
        "source_day",
        "target_day",
        "time_delta",
        "source_slice_id",
        "target_slice_id",
        "source_slice_file",
        "target_slice_file",
        "source_mouse_id",
        "target_mouse_id",
        "evidence_mode",
        *[f"raw_{group}_distance" for group in evidence_groups],
        "raw_pseudotime_score",
        "raw_barcode_score",
        *[f"scaled_{group}_distance" for group in evidence_groups],
        "scaled_pseudotime_score",
        "scaled_barcode_score",
        "scaling_method_molecular",
        "scaling_method_composition",
        "scaling_method_entropy",
        "scaling_method_spatial_summary",
        "scaling_method_topology",
        "zero_variance_molecular",
        "zero_variance_composition",
        "zero_variance_entropy",
        "zero_variance_spatial_summary",
        "zero_variance_topology",
        "source_mass",
        "target_mass",
        "growth_prior",
        "unbalanced_weight",
        "mass_adjusted_weight",
        "combined_cost",
        "tau_pair",
        "raw_edge_weight",
        "row_normalized_transition_prob",
    ]


def build_full_transition_shards(
    time_pairs: Sequence[dict[str, Any]],
    m2_summary: pd.DataFrame,
    candidate_k: int,
) -> pd.DataFrame:
    """Plan one full-M3 shard per time pair and source slice."""

    rows = []
    summary = m2_summary.set_index("slice_id")
    for pair in time_pairs:
        source_slices = list(pair["source_slices"])
        target_slices = list(pair["target_slices"])
        target_rows = int(pair["target_row_count"])
        for source_slice in source_slices:
            source_row = summary.loc[source_slice]
            source_rows = int(source_row["output_rows"])
            rows.append(
                {
                    "source_time": pair["source_time"],
                    "target_time": pair["target_time"],
                    "source_day": pair["source_day"],
                    "target_day": pair["target_day"],
                    "time_delta": pair["time_delta"],
                    "source_slice_id": source_slice,
                    "source_slice_file": f"{source_slice}.m0.h5ad",
                    "source_rows": source_rows,
                    "target_time_rows": target_rows,
                    "target_slice_count": len(target_slices),
                    "candidate_k": int(candidate_k),
                    "expected_edge_rows": source_rows * int(candidate_k),
                }
            )
    return pd.DataFrame(rows)


def edge_density_metrics(
    time_pairs: Sequence[dict[str, Any]],
    candidate_k: int,
) -> pd.DataFrame:
    """Return fixed-K density metrics by adjacent time pair."""

    rows = []
    for pair in time_pairs:
        target_rows = int(pair["target_row_count"])
        rows.append(
            {
                "source_time": pair["source_time"],
                "target_time": pair["target_time"],
                "target_pool_size": target_rows,
                "candidate_k": int(candidate_k),
                "k_over_target_pool": float(candidate_k / target_rows),
                "expected_candidate_edge_density": float(candidate_k / target_rows),
            }
        )
    return pd.DataFrame(rows)


def matrix_memory_gb(rows: int, columns: int, bytes_per_value: int = 8) -> float:
    """Estimate dense matrix memory in GiB."""

    return float(rows) * float(columns) * float(bytes_per_value) / float(1024**3)


def estimate_time_pair_memory(
    time_pairs: Sequence[dict[str, Any]],
    shards: pd.DataFrame,
    retrieval_dimensions: int,
    rerank_dimensions: int,
    max_memory_gb: float,
    bytes_per_value: int = 8,
) -> pd.DataFrame:
    """Estimate per-worker memory and safe concurrency for each time pair."""

    rows = []
    for pair in time_pairs:
        pair_shards = shards[
            (shards["source_time"].astype(str) == str(pair["source_time"]))
            & (shards["target_time"].astype(str) == str(pair["target_time"]))
        ]
        max_source_rows = int(pair_shards["source_rows"].max())
        target_rows = int(pair["target_row_count"])
        target_retrieval = matrix_memory_gb(
            target_rows,
            retrieval_dimensions,
            bytes_per_value,
        )
        target_rerank = matrix_memory_gb(target_rows, rerank_dimensions, bytes_per_value)
        source_retrieval = matrix_memory_gb(
            max_source_rows,
            retrieval_dimensions,
            bytes_per_value,
        )
        source_rerank = matrix_memory_gb(max_source_rows, rerank_dimensions, bytes_per_value)
        per_worker = target_retrieval + target_rerank + source_retrieval + source_rerank
        safe_concurrency = max(1, int(max_memory_gb // per_worker)) if per_worker else 1
        rows.append(
            {
                "source_time": pair["source_time"],
                "target_time": pair["target_time"],
                "target_retrieval_matrix_gb": target_retrieval,
                "target_rerank_matrix_gb": target_rerank,
                "source_shard_matrix_gb": source_retrieval + source_rerank,
                "per_worker_memory_gb": per_worker,
                "safe_single_node_concurrency": safe_concurrency,
            }
        )
    return pd.DataFrame(rows)
