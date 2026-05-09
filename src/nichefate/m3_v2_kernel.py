"""M3-v2 pilot transition kernel utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScalingStats:
    """Robust scaling statistics for one feature block."""

    median: np.ndarray
    scale: np.ndarray
    zero_scale_columns: int


def robust_scale_fit(
    frames: Iterable[np.ndarray],
    min_scale: float = 1e-6,
) -> ScalingStats:
    """Fit median/IQR scaling over one or more feature matrices."""

    matrices = [np.asarray(frame, dtype=np.float32) for frame in frames if frame.size]
    if not matrices:
        raise ValueError("At least one non-empty feature matrix is required.")
    combined = np.vstack(matrices)
    median = np.nanmedian(combined, axis=0).astype(np.float32)
    q25 = np.nanpercentile(combined, 25, axis=0).astype(np.float32)
    q75 = np.nanpercentile(combined, 75, axis=0).astype(np.float32)
    iqr = q75 - q25
    std = np.nanstd(combined, axis=0).astype(np.float32)
    scale = np.where(iqr >= min_scale, iqr, std)
    valid = scale >= min_scale
    safe_scale = np.where(valid, scale, 1.0).astype(np.float32)
    return ScalingStats(
        median=median,
        scale=safe_scale,
        zero_scale_columns=int((~valid).sum()),
    )


def robust_scale_transform(matrix: np.ndarray, stats: ScalingStats) -> np.ndarray:
    """Apply fitted robust scaling and replace non-finite values with zero."""

    scaled = (np.asarray(matrix, dtype=np.float32) - stats.median) / stats.scale
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def pairwise_l2_for_edges(
    source_matrix: np.ndarray,
    target_matrix: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Compute source-target L2 distance for an existing edge list."""

    if source_matrix.shape[1] != target_matrix.shape[1]:
        raise ValueError("Source and target matrices must have the same feature count.")
    if len(source_indices) != len(target_indices):
        raise ValueError("source_indices and target_indices must have the same length.")
    distances = np.empty(len(source_indices), dtype=np.float32)
    for start in range(0, len(source_indices), int(chunk_size)):
        stop = min(start + int(chunk_size), len(source_indices))
        diff = source_matrix[source_indices[start:stop]] - target_matrix[target_indices[start:stop]]
        distances[start:stop] = np.sqrt(np.einsum("ij,ij->i", diff, diff)).astype(np.float32)
    return distances


def source_adaptive_tau(
    distances: np.ndarray,
    source_codes: np.ndarray,
    quantile: float = 0.5,
    min_tau: float = 1e-6,
) -> np.ndarray:
    """Return per-edge source-adaptive bandwidths from candidate distances."""

    distance_series = pd.Series(np.asarray(distances, dtype=np.float64))
    source_series = pd.Series(np.asarray(source_codes))
    positive = distance_series.where(distance_series > min_tau)
    per_source = positive.groupby(source_series, sort=False).transform(
        lambda values: float(np.nanquantile(values, quantile)) if values.notna().any() else np.nan
    )
    fallback = float(np.nanquantile(positive, quantile)) if positive.notna().any() else float(min_tau)
    tau = per_source.fillna(fallback).clip(lower=min_tau).to_numpy(dtype=np.float32)
    return tau


def exponential_gate(
    distances: np.ndarray,
    tau: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Convert distances and bandwidths into a soft exponential gate."""

    if strength <= 0:
        return np.ones(len(distances), dtype=np.float32)
    safe_tau = np.clip(np.asarray(tau, dtype=np.float32), 1e-12, None)
    gate = np.exp(-np.asarray(distances, dtype=np.float32) / safe_tau)
    if strength == 1.0:
        return gate.astype(np.float32)
    return ((1.0 - strength) + strength * gate).astype(np.float32)


def _balance_gate(values: pd.Series, strength: float) -> pd.Series:
    if strength <= 0 or values.nunique(dropna=True) <= 1:
        return pd.Series(np.ones(len(values), dtype=np.float32), index=values.index)
    frequencies = values.astype(str).value_counts(normalize=True)
    expected = 1.0 / float(len(frequencies))
    penalty = frequencies.map(lambda observed: min(1.0, float(np.sqrt(expected / observed))))
    mapped = values.astype(str).map(penalty).fillna(1.0).astype(float)
    return (1.0 - strength) + strength * mapped


def slice_mouse_gate(
    target_slice: pd.Series,
    target_mouse: pd.Series,
    strength: float = 0.25,
    min_gate: float = 0.2,
) -> np.ndarray:
    """Softly penalize overrepresented target slice/mouse categories."""

    slice_gate = _balance_gate(target_slice, strength)
    mouse_gate = _balance_gate(target_mouse, strength)
    combined = (slice_gate * mouse_gate).clip(lower=min_gate, upper=1.0)
    return combined.to_numpy(dtype=np.float32)


def row_normalize_weights(
    weights: np.ndarray,
    source_codes: np.ndarray,
) -> np.ndarray:
    """Normalize edge weights within each source candidate set."""

    work = pd.DataFrame(
        {
            "source_code": np.asarray(source_codes),
            "weight": np.nan_to_num(np.asarray(weights, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0),
        }
    )
    work.loc[work["weight"] < 0, "weight"] = 0.0
    sums = work.groupby("source_code", sort=False)["weight"].transform("sum")
    counts = work.groupby("source_code", sort=False)["weight"].transform("size")
    normalized = np.where(sums > 0, work["weight"] / sums, 1.0 / counts)
    return normalized.astype(np.float32)


def validate_probabilities(probabilities: np.ndarray, source_codes: np.ndarray, atol: float = 1e-5) -> dict[str, float | bool]:
    """Validate finite, non-negative, row-normalized transition probabilities."""

    probs = np.asarray(probabilities, dtype=np.float64)
    finite = bool(np.isfinite(probs).all())
    nonnegative = bool((probs >= -atol).all())
    row_sums = pd.Series(probs).groupby(pd.Series(source_codes), sort=False).sum()
    max_abs_error = float(np.abs(row_sums - 1.0).max()) if len(row_sums) else float("nan")
    return {
        "finite": finite,
        "nonnegative": nonnegative,
        "row_sum_max_abs_error": max_abs_error,
        "row_sum_pass": bool(finite and nonnegative and max_abs_error <= atol),
    }


def source_entropy_and_top1(probabilities: np.ndarray, source_codes: np.ndarray) -> pd.DataFrame:
    """Return per-source transition entropy and top1 probability."""

    probs = np.asarray(probabilities, dtype=np.float64)
    safe_probs = np.clip(probs, 1e-300, 1.0)
    work = pd.DataFrame({"source_code": source_codes, "probability": probs})
    entropy_terms = -safe_probs * np.log(safe_probs)
    work["entropy_term"] = entropy_terms
    grouped = work.groupby("source_code", sort=False)
    return pd.DataFrame(
        {
            "source_code": grouped.size().index.to_numpy(),
            "transition_entropy": grouped["entropy_term"].sum().to_numpy(dtype=float),
            "top1_probability": grouped["probability"].max().to_numpy(dtype=float),
        }
    )


def jensen_shannon_by_source(
    v1_probabilities: np.ndarray,
    v2_probabilities: np.ndarray,
    source_codes: np.ndarray,
) -> pd.DataFrame:
    """Compute Jensen-Shannon divergence for aligned candidate rows per source."""

    p = np.clip(np.asarray(v1_probabilities, dtype=np.float64), 1e-300, 1.0)
    q = np.clip(np.asarray(v2_probabilities, dtype=np.float64), 1e-300, 1.0)
    m = 0.5 * (p + q)
    terms = 0.5 * (p * np.log(p / m) + q * np.log(q / m))
    grouped = pd.Series(terms).groupby(pd.Series(source_codes), sort=False).sum()
    return pd.DataFrame(
        {
            "source_code": grouped.index.to_numpy(),
            "v1_v2_js_divergence": grouped.to_numpy(dtype=float),
        }
    )
