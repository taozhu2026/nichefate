#!/usr/bin/env python
"""Construct bounded K_gpcca pilot kernels and run gated standard pyGPCCA.

This script never constructs full-production K_gpcca matrices. It writes only
bounded pilot artifacts under ``/home/zhutao/scratch/nichefate/k_gpcca_pilot``.
Standard pyGPCCA is invoked only after kernel QC passes. CellRank-compatible
GPCCA remains a policy-compatible future option, but is not invoked here. No
custom GPCCA-like fallback is implemented.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import k_gpcca_01_pilot_kernel_preflight as k01


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "k_gpcca_pilot.yaml"
VALID_PHASES = {"construct_smoke", "construct_pilot", "gpcca_smoke", "gpcca_pilot"}
SMOKE_DEFAULT_MAX_NODES = 20_000
PILOT_DEFAULT_MAX_NODES = 100_000
DEFAULT_K_VALUES = [4, 6, 8, 10, 12]
PRIMARY_CANDIDATES = ["pilot_v1_balanced", "pilot_v2_balanced"]
TMPDIR = Path("/home/zhutao/tmp/k_gpcca")


@dataclass(frozen=True)
class Candidate:
    grid_id: str
    route: str
    cross_time_source: str
    alpha: float
    beta: float
    gamma: float
    delta: float
    within_time_k: int
    similarity_metric: str
    priority: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", required=True, choices=sorted(VALID_PHASES))
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--n-macrostates", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    k01.reject_ssd(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(k01.json_safe(payload), indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    k01.reject_ssd(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    root = k01.validate_output_root(config)
    paths = {
        "root": root,
        "kernels": root / "kernels",
        "gpcca": root / "gpcca",
        "reports": root / "reports",
        "figures": root / "reports" / "figures",
    }
    for path in paths.values():
        k01.reject_ssd(path)
        if path != root and not k01.is_relative_to(path, root):
            raise ValueError(f"Output path must be under root: {path}")
    return paths


def ensure_output_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def candidate_from_row(row: pd.Series) -> Candidate:
    return Candidate(
        grid_id=str(row["grid_id"]),
        route=str(row["route"]),
        cross_time_source=str(row["cross_time_source"]),
        alpha=float(row["alpha"]),
        beta=float(row["beta"]),
        gamma=float(row["gamma"]),
        delta=float(row["delta"]),
        within_time_k=int(row["within_time_k"]),
        similarity_metric=str(row["similarity_metric"]),
        priority=str(row["priority"]),
    )


def select_candidate(config: dict[str, Any], candidate_id: str | None) -> Candidate:
    design_root = k01.resolved(config["paths"]["design_root"])
    grid = k01.load_candidate_grid(design_root)
    if candidate_id is None:
        candidate_id = PRIMARY_CANDIDATES[0]
    row = grid[grid["grid_id"] == candidate_id]
    if row.empty:
        raise ValueError(f"Unknown candidate_id: {candidate_id}")
    candidate = candidate_from_row(row.iloc[0])
    if candidate.route != "full_resolution_subset":
        raise ValueError(f"Candidate is not executable in K_gpcca-02: {candidate.grid_id}")
    if "mixed" in candidate.cross_time_source.lower():
        raise ValueError(f"Mixed review-only candidate is not executable: {candidate.grid_id}")
    if "barcode" in candidate.cross_time_source.lower() or candidate.delta != 0.0:
        raise ValueError(f"Future barcode candidate is not executable: {candidate.grid_id}")
    if candidate.cross_time_source not in {"M3-v1", "M3-v2"}:
        raise ValueError(f"Unsupported cross-time source: {candidate.cross_time_source}")
    return candidate


def phase_label(phase: str) -> str:
    return "smoke" if "smoke" in phase else "pilot"


def max_nodes_for_phase(config: dict[str, Any], phase: str, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    if phase_label(phase) == "smoke":
        return SMOKE_DEFAULT_MAX_NODES
    return int(config["pilot"].get("target_max_nodes", PILOT_DEFAULT_MAX_NODES))


def kernel_paths(paths: dict[str, Path], candidate_id: str, label: str) -> dict[str, Path]:
    prefix = f"K_gpcca_{label}_{candidate_id}"
    return {
        "matrix": paths["kernels"] / f"{prefix}.npz",
        "node_table": paths["kernels"] / f"{prefix}_node_table.parquet",
        "qc_csv": paths["reports"] / f"k_gpcca_02_{label}_kernel_qc_{candidate_id}.csv",
        "report_md": paths["reports"] / f"k_gpcca_02_{label}_kernel_report_{candidate_id}.md",
    }


def choose_time_scope(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    return list(config["pilot"]["preferred_time_points"]), list(config["pilot"]["preferred_time_pairs"])


def prepare_selected_nodes(config: dict[str, Any], max_nodes: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    time_points, time_pairs = choose_time_scope(config)
    nodes = k01.read_node_table(config)
    selected = k01.deterministic_select_nodes(nodes, time_points, max_nodes).copy()
    selected = k01.read_optional_neighborhood(config, selected)
    selected = selected.reset_index(drop=True)
    selected["local_index"] = np.arange(len(selected), dtype=np.int64)
    return selected, time_points, time_pairs


def m2_feature_columns(config: dict[str, Any]) -> list[str]:
    schema = k01.load_json(k01.resolved(config["paths"]["m2_root"]) / "reports" / "m2_full_feature_schema.json")
    columns = schema.get("numeric_feature_columns", [])
    if not columns:
        raise ValueError("M2 numeric feature columns are missing")
    return [str(column) for column in columns]


def read_selected_m2_features(config: dict[str, Any], selected: pd.DataFrame) -> np.ndarray:
    feature_columns = m2_feature_columns(config)
    m2_by_slice = k01.resolved(config["paths"]["m2_root"]) / "by_slice"
    features = np.empty((len(selected), len(feature_columns)), dtype=np.float32)
    filled = np.zeros(len(selected), dtype=bool)
    for slice_id, group in selected.groupby("slice_id", sort=True):
        slice_id_text = str(slice_id)
        path = m2_by_slice / slice_id_text / f"m2_representation_{slice_id_text}.parquet"
        if not path.exists():
            matches = sorted((m2_by_slice / slice_id_text).glob("m2_representation_*.parquet"))
            if not matches:
                raise FileNotFoundError(f"Missing M2 representation parquet for slice {slice_id_text}")
            path = matches[0]
        anchors = sorted(int(value) for value in group["anchor_index"].tolist())
        frame = pd.read_parquet(
            path,
            columns=["anchor_index", *feature_columns],
            filters=[("anchor_index", "in", anchors)],
        )
        merged = group[["local_index", "anchor_index"]].merge(
            frame,
            on="anchor_index",
            how="left",
            validate="one_to_one",
        )
        values = merged[feature_columns].to_numpy(dtype=np.float32, copy=True)
        local_indices = merged["local_index"].to_numpy(dtype=np.int64)
        if np.isnan(values).any():
            raise ValueError(f"NaN values found in M2 features for slice {slice_id_text}")
        features[local_indices, :] = values
        filled[local_indices] = True
    if not bool(filled.all()):
        missing = int((~filled).sum())
        raise ValueError(f"Missing M2 features for {missing} selected nodes")
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite M2 feature values found")
    return features


def standardize_and_project(features: np.ndarray, n_components: int = 50) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD

    features = features.astype(np.float32, copy=True)
    means = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = features.std(axis=0, dtype=np.float64).astype(np.float32)
    stds[stds == 0] = 1.0
    features -= means
    features /= stds
    if features.shape[1] <= n_components or features.shape[0] <= n_components + 1:
        return features
    svd = TruncatedSVD(
        n_components=min(n_components, features.shape[1] - 1, features.shape[0] - 1),
        random_state=0,
        algorithm="randomized",
    )
    projected = svd.fit_transform(features).astype(np.float32, copy=False)
    return projected


def exact_knn_indices(data: np.ndarray, k: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    n_neighbors = min(k + 1, data.shape[0])
    metric_name = "cosine" if metric == "cosine" else "euclidean"
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric_name, algorithm="auto", n_jobs=-1)
    model.fit(data)
    distances, indices = model.kneighbors(data)
    return indices, distances


def approximate_knn_indices(data: np.ndarray, k: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pynndescent

        metric_name = "cosine" if metric == "cosine" else "euclidean"
        index = pynndescent.NNDescent(
            data,
            n_neighbors=min(k + 1, data.shape[0]),
            metric=metric_name,
            random_state=0,
            low_memory=True,
            n_jobs=-1,
        )
        indices, distances = index.neighbor_graph
        return indices, distances
    except Exception:
        return exact_knn_indices(data, k, metric)


def build_within_time_graph(
    selected: pd.DataFrame,
    feature_embedding: np.ndarray,
    k: int,
    metric: str,
) -> sparse.csr_matrix:
    n_nodes = len(selected)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for _, group in selected.groupby("time", sort=True):
        local_indices = group["local_index"].to_numpy(dtype=np.int64)
        if len(local_indices) <= 1:
            continue
        group_data = feature_embedding[local_indices]
        if len(local_indices) <= 5000:
            neighbor_indices, distances = exact_knn_indices(group_data, k, metric)
        else:
            neighbor_indices, distances = approximate_knn_indices(group_data, k, metric)
        for row_position, local_row in enumerate(local_indices):
            neigh = neighbor_indices[row_position]
            dist = distances[row_position]
            keep = neigh != row_position
            neigh = neigh[keep][:k]
            dist = dist[keep][:k]
            if len(neigh) == 0:
                continue
            local_cols = local_indices[neigh]
            weights = np.exp(-np.maximum(dist.astype(np.float64), 0.0)).astype(np.float32)
            rows.append(np.full(len(local_cols), local_row, dtype=np.int64))
            cols.append(local_cols.astype(np.int64, copy=False))
            data.append(weights)
    if not rows:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float32)
    matrix = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return row_normalize(matrix)


def row_normalize(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True)
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    nonzero = row_sums > 0
    inv = np.zeros_like(row_sums, dtype=np.float64)
    inv[nonzero] = 1.0 / row_sums[nonzero]
    normalized = sparse.diags(inv).dot(matrix).tocsr()
    normalized.eliminate_zeros()
    return normalized.astype(np.float32)


def build_cross_time_graph(
    config: dict[str, Any],
    selected: pd.DataFrame,
    candidate: Candidate,
    time_pairs: list[str],
) -> sparse.csr_matrix:
    import pyarrow.parquet as pq

    cross_cfg = config["cross_time"]
    if candidate.cross_time_source == "M3-v1":
        edge_root = k01.resolved(cross_cfg["m3_v1_edge_root"])
        probability_column = cross_cfg["m3_v1_probability_column"]
    elif candidate.cross_time_source == "M3-v2":
        edge_root = k01.resolved(cross_cfg["m3_v2_edge_root"])
        probability_column = cross_cfg["m3_v2_probability_column"]
    else:
        raise ValueError(f"Unsupported cross-time source: {candidate.cross_time_source}")
    anchor_to_local = dict(zip(selected["anchor_id"].astype(str), selected["local_index"].astype(int)))
    selected_anchors = set(anchor_to_local)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for path in k01.edge_files_for_time_pairs(edge_root, time_pairs):
        parquet_file = pq.ParquetFile(path)
        missing = k01.validate_cross_time_schema(list(parquet_file.schema.names), probability_column)
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        for batch in parquet_file.iter_batches(
            batch_size=int(cross_cfg["batch_rows"]),
            columns=["source_anchor_id", "target_anchor_id", probability_column],
        ):
            frame = batch.to_pandas()
            mask = frame["source_anchor_id"].isin(selected_anchors) & frame["target_anchor_id"].isin(selected_anchors)
            if not bool(mask.any()):
                continue
            sub = frame.loc[mask, ["source_anchor_id", "target_anchor_id", probability_column]]
            probabilities = sub[probability_column].to_numpy(dtype=np.float32, copy=False)
            if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
                raise ValueError(f"Invalid probabilities in cross-time source {candidate.cross_time_source}")
            rows.append(sub["source_anchor_id"].map(anchor_to_local).to_numpy(dtype=np.int64))
            cols.append(sub["target_anchor_id"].map(anchor_to_local).to_numpy(dtype=np.int64))
            data.append(probabilities)
    n_nodes = len(selected)
    if not rows:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float32)
    matrix = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return row_normalize(matrix)


def combine_components(
    within: sparse.csr_matrix,
    cross: sparse.csr_matrix,
    candidate: Candidate,
) -> tuple[sparse.csr_matrix, dict[str, float]]:
    n_nodes = within.shape[0]
    identity = sparse.identity(n_nodes, format="csr", dtype=np.float32)
    raw = (
        within.multiply(candidate.alpha)
        + cross.multiply(candidate.beta)
        + identity.multiply(candidate.gamma)
    ).tocsr()
    raw.sum_duplicates()
    raw.eliminate_zeros()
    raw_row_sums = np.asarray(raw.sum(axis=1)).ravel()
    kernel = row_normalize(raw)
    component_masses = {
        "within_time_mass": component_mass(within, candidate.alpha, raw_row_sums),
        "cross_time_mass": component_mass(cross, candidate.beta, raw_row_sums),
        "self_loop_mass": component_mass(identity, candidate.gamma, raw_row_sums),
    }
    total_mass = max(1.0, float(n_nodes))
    component_masses.update(
        {
            "within_time_mass_fraction": component_masses["within_time_mass"] / total_mass,
            "cross_time_mass_fraction": component_masses["cross_time_mass"] / total_mass,
            "self_loop_mass_fraction": component_masses["self_loop_mass"] / total_mass,
        }
    )
    return kernel, component_masses


def component_mass(component: sparse.csr_matrix, weight: float, raw_row_sums: np.ndarray) -> float:
    component = component.tocsr()
    rows = np.repeat(np.arange(component.shape[0], dtype=np.int64), np.diff(component.indptr))
    if len(rows) == 0:
        return 0.0
    denominators = raw_row_sums[rows]
    valid = denominators > 0
    values = component.data.astype(np.float64, copy=False) * float(weight)
    return float(np.sum(values[valid] / denominators[valid]))


def kernel_qc(
    kernel: sparse.csr_matrix,
    within: sparse.csr_matrix,
    cross: sparse.csr_matrix,
    selected: pd.DataFrame,
    candidate: Candidate,
    component_masses: dict[str, float],
    matrix_path: Path | None = None,
) -> dict[str, Any]:
    row_sums = np.asarray(kernel.sum(axis=1)).ravel()
    row_sum_errors = np.abs(row_sums - 1.0)
    invalid_entries = int((~np.isfinite(kernel.data)).sum())
    negative_entries = int((kernel.data < 0).sum())
    zero_outgoing_rows = int((row_sums == 0).sum())
    weak_count, weak_labels = csgraph.connected_components(
        kernel,
        directed=True,
        connection="weak",
        return_labels=True,
    )
    weak_counts = np.bincount(weak_labels)
    largest_weak_fraction = float(weak_counts.max() / max(1, kernel.shape[0]))
    strong_components: int | str
    if kernel.nnz <= 2_000_000:
        strong_components = int(
            csgraph.connected_components(
                kernel,
                directed=True,
                connection="strong",
                return_labels=False,
            )
        )
    else:
        strong_components = "not computed: nnz threshold"
    out_degree = np.diff(kernel.indptr)
    in_degree = np.bincount(kernel.indices, minlength=kernel.shape[0])
    slice_max_fraction = float(selected["slice_id"].value_counts(normalize=True).max())
    mouse_max_fraction = float(selected["mouse_id"].value_counts(normalize=True).max())
    disk_bytes = int(matrix_path.stat().st_size) if matrix_path and matrix_path.exists() else 0
    qc = {
        "candidate_id": candidate.grid_id,
        "alpha": candidate.alpha,
        "beta": candidate.beta,
        "gamma": candidate.gamma,
        "cross_time_source": candidate.cross_time_source,
        "selected_node_count": int(kernel.shape[0]),
        "matrix_shape": f"{kernel.shape[0]}x{kernel.shape[1]}",
        "nnz": int(kernel.nnz),
        "row_sum_max_error": float(row_sum_errors.max(initial=0.0)),
        "row_sum_p99_error": float(np.quantile(row_sum_errors, 0.99)),
        "invalid_entries": invalid_entries,
        "negative_entries": negative_entries,
        "zero_outgoing_rows": zero_outgoing_rows,
        "self_loop_count": int((kernel.diagonal() > 0).sum()),
        "within_time_nnz": int(within.nnz),
        "cross_time_nnz": int(cross.nnz),
        "weak_component_count": int(weak_count),
        "largest_weak_component_fraction": largest_weak_fraction,
        "strong_component_count": strong_components,
        "out_degree_min": int(out_degree.min(initial=0)),
        "out_degree_median": float(np.median(out_degree)),
        "out_degree_max": int(out_degree.max(initial=0)),
        "in_degree_min": int(in_degree.min(initial=0)),
        "in_degree_median": float(np.median(in_degree)),
        "in_degree_max": int(in_degree.max(initial=0)),
        "slice_max_fraction": slice_max_fraction,
        "mouse_max_fraction": mouse_max_fraction,
        "memory_estimate_mb": float((kernel.data.nbytes + kernel.indices.nbytes + kernel.indptr.nbytes) / (1024**2)),
        "disk_bytes": disk_bytes,
        "disk_mb": float(disk_bytes / (1024**2)),
        **component_masses,
    }
    for key in [
        "within_time_mass",
        "cross_time_mass",
        "self_loop_mass",
        "within_time_mass_fraction",
        "cross_time_mass_fraction",
        "self_loop_mass_fraction",
    ]:
        qc.setdefault(key, 0.0)
    qc["component_dominance_warning"] = bool(
        max(
            qc["within_time_mass_fraction"],
            qc["cross_time_mass_fraction"],
            qc["self_loop_mass_fraction"],
        )
        > 0.85
    )
    qc["slice_mouse_concentration_warning"] = bool(
        qc["slice_max_fraction"] > 0.25 or qc["mouse_max_fraction"] > 0.60
    )
    qc["kernel_qc_pass"] = bool(
        qc["row_sum_max_error"] <= 1e-5
        and invalid_entries == 0
        and negative_entries == 0
        and zero_outgoing_rows == 0
        and qc["self_loop_count"] == kernel.shape[0]
        and qc["within_time_nnz"] > 0
        and qc["cross_time_nnz"] > 0
        and not qc["component_dominance_warning"]
    )
    return qc


def write_kernel_report(path: Path, qc: dict[str, Any], label: str) -> None:
    lines = [
        f"# K_gpcca-02 {label.title()} Kernel Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## QC",
        "",
    ]
    for key, value in qc.items():
        lines.append(f"- `{key}`: {value}")
    atomic_write_text(path, "\n".join(lines) + "\n")


def construct_kernel_phase(
    config: dict[str, Any],
    paths: dict[str, Path],
    candidate: Candidate,
    label: str,
    max_nodes: int,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    outputs = kernel_paths(paths, candidate.grid_id, label)
    if outputs["matrix"].exists() and not overwrite and not resume:
        raise FileExistsError(f"Kernel already exists; use --resume or --overwrite: {outputs['matrix']}")
    selected, time_points, time_pairs = prepare_selected_nodes(config, max_nodes)
    features = read_selected_m2_features(config, selected)
    embedding = standardize_and_project(features)
    within = build_within_time_graph(
        selected,
        embedding,
        candidate.within_time_k,
        candidate.similarity_metric,
    )
    cross = build_cross_time_graph(config, selected, candidate, time_pairs)
    kernel, component_masses = combine_components(within, cross, candidate)
    sparse.save_npz(outputs["matrix"], kernel, compressed=True)
    selected.to_parquet(outputs["node_table"], index=False)
    qc = kernel_qc(kernel, within, cross, selected, candidate, component_masses, outputs["matrix"])
    atomic_write_csv(outputs["qc_csv"], pd.DataFrame([qc]))
    write_kernel_report(outputs["report_md"], qc, label)
    if label == "smoke":
        atomic_write_csv(paths["reports"] / "k_gpcca_02_smoke_kernel_qc_summary.csv", pd.DataFrame([qc]))
        atomic_write_text(
            paths["reports"] / "k_gpcca_02_smoke_kernel_qc_report.md",
            (outputs["report_md"].read_text(encoding="utf-8")),
        )
    return {
        "candidate_id": candidate.grid_id,
        "label": label,
        "time_points": time_points,
        "time_pairs": time_pairs,
        "matrix_path": outputs["matrix"],
        "node_table_path": outputs["node_table"],
        "qc": qc,
    }


def existing_kernel_path(paths: dict[str, Path], candidate_id: str, label: str) -> Path:
    path = kernel_paths(paths, candidate_id, label)["matrix"]
    if not path.exists():
        raise FileNotFoundError(f"Required {label} kernel is missing: {path}")
    return path


def existing_node_table_path(paths: dict[str, Path], candidate_id: str, label: str) -> Path:
    path = kernel_paths(paths, candidate_id, label)["node_table"]
    if not path.exists():
        raise FileNotFoundError(f"Required {label} node table is missing: {path}")
    return path


def detect_pygpcca_env() -> str:
    for env_name in ["nichefate-gpcca"]:
        result = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", env_name, "python", "-c", "import pygpcca"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return env_name
    raise RuntimeError("No conda environment with standard pyGPCCA was found")


def gpcca_k_values(requested: int | None) -> list[int]:
    if requested is not None:
        return [int(requested)]
    return DEFAULT_K_VALUES


def run_pygpcca_subprocess(
    matrix_path: Path,
    node_table_path: Path,
    output_dir: Path,
    candidate_id: str,
    label: str,
    k_values: list[int],
    write_outputs: bool,
) -> dict[str, Any]:
    env_name = detect_pygpcca_env()
    TMPDIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "matrix_path": str(matrix_path),
        "node_table_path": str(node_table_path),
        "output_dir": str(output_dir),
        "candidate_id": candidate_id,
        "label": label,
        "k_values": k_values,
        "write_outputs": bool(write_outputs),
    }
    code = r"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pygpcca
from scipy import sparse

payload = json.loads(sys.argv[1])
matrix = sparse.load_npz(payload["matrix_path"]).tocsr().astype(float)
row_sums = np.asarray(matrix.sum(axis=1)).ravel()
nonzero = row_sums > 0
inv = np.zeros_like(row_sums, dtype=float)
inv[nonzero] = 1.0 / row_sums[nonzero]
matrix = sparse.diags(inv).dot(matrix).tocsr()
node_table = pd.read_parquet(payload["node_table_path"])
rows = []
best = None
for k in payload["k_values"]:
    record = {
        "k": int(k),
        "success": False,
        "error": "",
        "memberships_shape": "",
        "macrostate_assignment_shape": "",
        "coarse_transition_shape": "",
        "membership_entropy_mean": np.nan,
        "membership_entropy_median": np.nan,
        "max_membership_mean": np.nan,
        "max_membership_median": np.nan,
        "macrostate_size_min": np.nan,
        "macrostate_size_max": np.nan,
    }
    try:
        gpcca = pygpcca.GPCCA(matrix, z="LM", method="krylov")
        gpcca.optimize(int(k))
        memberships = np.asarray(gpcca.memberships)
        assignments = np.asarray(gpcca.macrostate_assignment).reshape(-1)
        coarse = np.asarray(gpcca.coarse_grained_transition_matrix)
        memberships = np.real_if_close(memberships).astype(float)
        memberships[memberships < 0] = 0.0
        row_sums = memberships.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        memberships = memberships / row_sums
        entropy = -np.sum(np.where(memberships > 0, memberships * np.log(memberships), 0.0), axis=1)
        max_membership = memberships.max(axis=1)
        sizes = np.bincount(assignments.astype(int), minlength=memberships.shape[1])
        record.update({
            "success": True,
            "memberships_shape": "x".join(map(str, memberships.shape)),
            "macrostate_assignment_shape": "x".join(map(str, assignments.shape)),
            "coarse_transition_shape": "x".join(map(str, coarse.shape)),
            "membership_entropy_mean": float(np.mean(entropy)),
            "membership_entropy_median": float(np.median(entropy)),
            "max_membership_mean": float(np.mean(max_membership)),
            "max_membership_median": float(np.median(max_membership)),
            "macrostate_size_min": int(sizes.min()),
            "macrostate_size_max": int(sizes.max()),
        })
        if best is None or abs(int(k) - 8) < abs(int(best["k"]) - 8):
            best = {
                "k": int(k),
                "memberships": memberships,
                "assignments": assignments,
                "coarse": coarse,
                "entropy": entropy,
                "max_membership": max_membership,
            }
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    rows.append(record)

output_dir = Path(payload["output_dir"])
candidate_table = pd.DataFrame(rows)
candidate_csv = output_dir / f"k_gpcca_{payload['label']}_pygpcca_candidates_{payload['candidate_id']}.csv"
candidate_table.to_csv(candidate_csv, index=False)
written = {}
if payload["write_outputs"] and best is not None:
    k = best["k"]
    memberships = pd.DataFrame(best["memberships"], columns=[f"membership_{i}" for i in range(best["memberships"].shape[1])])
    memberships.insert(0, "global_node_index", node_table["global_node_index"].to_numpy())
    memberships.insert(1, "local_index", np.arange(len(memberships)))
    memberships_path = output_dir / f"k_gpcca_pilot_gpcca_memberships_{payload['candidate_id']}.parquet"
    memberships.to_parquet(memberships_path, index=False)
    macrostates = node_table[["global_node_index", "local_index", "time", "slice_id", "mouse_id", "cell_type_l3"]].copy()
    for optional in ["leiden_neigh", "cadinu_neighborhood_label", "x", "y"]:
        if optional in node_table.columns:
            macrostates[optional] = node_table[optional]
    macrostates["macrostate"] = best["assignments"].astype(int)
    macrostates["membership_entropy"] = best["entropy"]
    macrostates["max_membership"] = best["max_membership"]
    macro_path = output_dir / f"k_gpcca_pilot_gpcca_macrostates_{payload['candidate_id']}.csv"
    macrostates.to_csv(macro_path, index=False)
    coarse_path = output_dir / f"k_gpcca_pilot_gpcca_coarse_transition_{payload['candidate_id']}.csv"
    pd.DataFrame(best["coarse"]).to_csv(coarse_path, index=False)
    written = {
        "selected_k": k,
        "memberships_path": str(memberships_path),
        "macrostates_path": str(macro_path),
        "coarse_transition_path": str(coarse_path),
    }
result = {
    "env": "pygpcca",
    "candidate_csv": str(candidate_csv),
    "k_values": payload["k_values"],
    "succeeded": [int(row["k"]) for row in rows if row["success"]],
    "failed": [int(row["k"]) for row in rows if not row["success"]],
    "written": written,
}
print(json.dumps(result))
"""
    env = os.environ.copy()
    for key in ["TMPDIR", "TMP", "TEMP", "OMPI_MCA_orte_tmpdir_base", "OMPI_MCA_prte_tmpdir_base", "PRTE_MCA_prte_tmpdir_base", "PMIX_MCA_pmix_tmpdir_base"]:
        env[key] = str(TMPDIR)
    result = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", env_name, "python", "-c", code, json.dumps(payload)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        return {
            "environment": env_name,
            "tmpdir": str(TMPDIR),
            "returncode": result.returncode,
            "success": False,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "k_values": k_values,
            "succeeded": [],
            "failed": k_values,
            "written": {},
        }
    stdout_lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    parsed = json.loads(stdout_lines[-1]) if stdout_lines else {}
    parsed.update(
        {
            "environment": env_name,
            "tmpdir": str(TMPDIR),
            "returncode": result.returncode,
            "success": bool(parsed.get("succeeded")),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )
    return parsed


def kernel_qc_allows_gpcca(qc: dict[str, Any]) -> bool:
    return bool(qc.get("kernel_qc_pass", False))


def annotate_gpcca_outputs(paths: dict[str, Path], candidate_id: str) -> dict[str, Any]:
    macro_path = paths["gpcca"] / f"k_gpcca_pilot_gpcca_macrostates_{candidate_id}.csv"
    if not macro_path.exists():
        return {"annotation_status": "not computed", "reason": "macrostates file missing"}
    data = pd.read_csv(macro_path)
    endpoint_columns = [
        "global_node_index",
        "candidate_endpoint",
        "candidate_endpoint_label",
        "endpoint_biological_label",
        "endpoint_phenotype_class",
        "biological_confidence_tier",
    ]
    endpoint_path = k01.ROOT / "m4e" / "neighborhood_annotation" / "node_neighborhood_annotation.parquet"
    if endpoint_path.exists() and "candidate_endpoint" not in data.columns:
        available = k01.parquet_metadata(endpoint_path)[1]
        columns = [column for column in endpoint_columns if column in available]
        if "global_node_index" in columns:
            endpoint = pd.read_parquet(endpoint_path, columns=columns)
            data = data.merge(endpoint, on="global_node_index", how="left")
    rows = []
    for column in [
        "time",
        "leiden_neigh",
        "slice_id",
        "mouse_id",
        "cell_type_l3",
        "candidate_endpoint_label",
        "endpoint_biological_label",
    ]:
        if column not in data.columns:
            continue
        summary = (
            data.groupby(["macrostate", column])
            .size()
            .reset_index(name="node_count")
            .sort_values(["macrostate", "node_count"], ascending=[True, False])
        )
        summary["annotation_group"] = column
        rows.append(summary.rename(columns={column: "label"}))
    annotation = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    annotation_path = paths["reports"] / f"k_gpcca_02_macrostate_annotation_{candidate_id}.csv"
    if not annotation.empty:
        atomic_write_csv(annotation_path, annotation)
        data.to_csv(macro_path, index=False)
    return {
        "annotation_status": "computed" if not annotation.empty else "not computed",
        "annotation_path": str(annotation_path) if not annotation.empty else "",
        "macrostate_count": int(data["macrostate"].nunique()),
        "macrostate_size_min": int(data["macrostate"].value_counts().min()),
        "macrostate_size_max": int(data["macrostate"].value_counts().max()),
        "membership_entropy_mean": float(data["membership_entropy"].mean()),
        "membership_entropy_median": float(data["membership_entropy"].median()),
        "max_membership_mean": float(data["max_membership"].mean()),
        "max_membership_median": float(data["max_membership"].median()),
    }


def generate_figures(paths: dict[str, Path], candidate_id: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    macro_path = paths["gpcca"] / f"k_gpcca_pilot_gpcca_macrostates_{candidate_id}.csv"
    coarse_path = paths["gpcca"] / f"k_gpcca_pilot_gpcca_coarse_transition_{candidate_id}.csv"
    if not macro_path.exists():
        return []
    data = pd.read_csv(macro_path)
    figures: list[Path] = []
    size_path = paths["figures"] / f"k_gpcca_02_macrostate_sizes_{candidate_id}.png"
    data["macrostate"].value_counts().sort_index().plot(kind="bar")
    plt.title("Macrostate sizes")
    plt.tight_layout()
    plt.savefig(size_path, dpi=150)
    plt.close()
    figures.append(size_path)

    entropy_path = paths["figures"] / f"k_gpcca_02_membership_entropy_{candidate_id}.png"
    data["membership_entropy"].hist(bins=40)
    plt.title("Membership entropy")
    plt.tight_layout()
    plt.savefig(entropy_path, dpi=150)
    plt.close()
    figures.append(entropy_path)

    maxmem_path = paths["figures"] / f"k_gpcca_02_max_membership_{candidate_id}.png"
    data["max_membership"].hist(bins=40)
    plt.title("Max membership")
    plt.tight_layout()
    plt.savefig(maxmem_path, dpi=150)
    plt.close()
    figures.append(maxmem_path)

    if "time" in data.columns:
        heat = pd.crosstab(data["macrostate"], data["time"], normalize="index")
        heat_path = paths["figures"] / f"k_gpcca_02_macrostate_by_time_{candidate_id}.png"
        plt.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
        plt.yticks(range(len(heat.index)), heat.index)
        plt.xticks(range(len(heat.columns)), heat.columns, rotation=45)
        plt.colorbar(label="fraction")
        plt.title("Macrostate by time")
        plt.tight_layout()
        plt.savefig(heat_path, dpi=150)
        plt.close()
        figures.append(heat_path)
    if "leiden_neigh" in data.columns:
        top = data["leiden_neigh"].value_counts().head(20).index
        heat = pd.crosstab(
            data.loc[data["leiden_neigh"].isin(top), "macrostate"],
            data.loc[data["leiden_neigh"].isin(top), "leiden_neigh"],
            normalize="index",
        )
        neigh_path = paths["figures"] / f"k_gpcca_02_macrostate_by_neighborhood_{candidate_id}.png"
        plt.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
        plt.yticks(range(len(heat.index)), heat.index)
        plt.xticks(range(len(heat.columns)), heat.columns, rotation=90)
        plt.colorbar(label="fraction")
        plt.title("Macrostate by neighborhood")
        plt.tight_layout()
        plt.savefig(neigh_path, dpi=150)
        plt.close()
        figures.append(neigh_path)
    if coarse_path.exists():
        coarse = pd.read_csv(coarse_path).to_numpy()
        coarse_fig = paths["figures"] / f"k_gpcca_02_coarse_transition_{candidate_id}.png"
        plt.imshow(coarse, aspect="auto", cmap="magma")
        plt.colorbar(label="transition")
        plt.title("Coarse transition")
        plt.tight_layout()
        plt.savefig(coarse_fig, dpi=150)
        plt.close()
        figures.append(coarse_fig)
    return figures


def scan_kernel_inventory(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for matrix_path in sorted(paths["kernels"].glob("K_gpcca_*.npz")):
        node_path = matrix_path.with_name(matrix_path.stem + "_node_table.parquet")
        rows.append(
            {
                "artifact": matrix_path.name,
                "path": str(matrix_path),
                "bytes": int(matrix_path.stat().st_size),
                "node_table_exists": node_path.exists(),
                "node_table_path": str(node_path),
            }
        )
    return pd.DataFrame(rows)


def scan_gpcca_inventory(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for path in sorted(paths["gpcca"].glob("*")):
        if path.is_file():
            rows.append({"artifact": path.name, "path": str(path), "bytes": int(path.stat().st_size)})
    return pd.DataFrame(rows)


def write_global_reports(
    paths: dict[str, Path],
    summary: dict[str, Any],
    kernel_inventory: pd.DataFrame,
    gpcca_inventory: pd.DataFrame,
) -> None:
    execution = [
        "# K_gpcca-02 Execution Report",
        "",
        f"Generated: {utc_now()}",
        "",
    ]
    for key, value in summary.items():
        if key.endswith("_diffs") or key == "stdout" or key == "stderr":
            continue
        execution.append(f"- `{key}`: {value}")
    atomic_write_text(paths["reports"] / "k_gpcca_02_execution_report.md", "\n".join(execution) + "\n")

    kernel_decision = summary.get("kernel_decision_category", "not computed")
    atomic_write_text(
        paths["reports"] / "k_gpcca_02_kernel_qc_decision_report.md",
        f"# K_gpcca-02 Kernel QC Decision\n\nDecision: `{kernel_decision}`\n\nKernel inventory rows: {len(kernel_inventory)}\n",
    )
    pygpcca_decision = summary.get("pygpcca_decision_category", "not computed")
    atomic_write_text(
        paths["reports"] / "k_gpcca_02_pygpcca_decision_report.md",
        f"# K_gpcca-02 pyGPCCA Decision\n\nDecision: `{pygpcca_decision}`\n\nGPCCA inventory rows: {len(gpcca_inventory)}\n",
    )
    annotation_lines = [
        "# K_gpcca-02 Biological Annotation Report",
        "",
        f"Annotation status: {summary.get('annotation_status', 'not computed')}",
        f"Reason: {summary.get('annotation_reason', 'available after successful pyGPCCA pilot')}",
        "",
    ]
    annotation_path = summary.get("annotation_path", "")
    if annotation_path and Path(annotation_path).exists():
        annotation = pd.read_csv(annotation_path)
        for group in [
            "time",
            "leiden_neigh",
            "slice_id",
            "mouse_id",
            "candidate_endpoint_label",
            "endpoint_biological_label",
        ]:
            sub = annotation[annotation["annotation_group"] == group].head(12)
            if not sub.empty:
                annotation_lines.append(f"## Macrostate by {group}")
                annotation_lines.append("")
                annotation_lines.append(k01.markdown_table(sub, max_rows=12) if hasattr(k01, "markdown_table") else sub.to_string(index=False))
                annotation_lines.append("")
    atomic_write_text(paths["reports"] / "k_gpcca_02_biological_annotation_report.md", "\n".join(annotation_lines) + "\n")
    atomic_write_text(
        paths["reports"] / "k_gpcca_02_next_step_recommendation.md",
        "# K_gpcca-02 Next Step Recommendation\n\n"
        f"{summary.get('next_recommended_step', 'Run the next gated K_gpcca task only after reviewing K_gpcca-02 reports.')}\n",
    )
    atomic_write_csv(paths["root"] / "k_gpcca_02_kernel_inventory.csv", kernel_inventory)
    atomic_write_csv(paths["root"] / "k_gpcca_02_gpcca_output_inventory.csv", gpcca_inventory)


def safety_snapshot_roots(config: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    protected = [k01.resolved(path) for path in config.get("protected_roots", [])]
    forbidden = [k01.resolved(path) for path in config.get("forbidden_downstream_roots", [])]
    return protected, forbidden


def phase_summary_base(
    phase: str,
    candidate: Candidate,
    paths: dict[str, Path],
    protected_before: dict[str, dict[str, Any]],
    forbidden_before: dict[str, dict[str, Any]],
    protected_roots: list[Path],
    forbidden_roots: list[Path],
) -> dict[str, Any]:
    protected_after = k01.snapshot(protected_roots)
    forbidden_after = k01.snapshot(forbidden_roots)
    upstream_diffs = k01.diff_snapshot(protected_before, protected_after)
    forbidden_diffs = k01.diff_snapshot(forbidden_before, forbidden_after)
    return {
        "stage": "K_gpcca-02",
        "generated_at_utc": utc_now(),
        "phase": phase,
        "candidate_id": candidate.grid_id,
        "alpha": candidate.alpha,
        "beta": candidate.beta,
        "gamma": candidate.gamma,
        "cross_time_source": candidate.cross_time_source,
        "output_root": paths["root"],
        "kernels_dir": paths["kernels"],
        "gpcca_dir": paths["gpcca"],
        "reports_dir": paths["reports"],
        "upstream_metadata_diff_count": len(upstream_diffs),
        "upstream_metadata_diffs": upstream_diffs,
        "forbidden_downstream_diff_count": len(forbidden_diffs),
        "forbidden_downstream_diffs": forbidden_diffs,
        "ssd_output_count": k01.count_ssd_outputs(paths["root"]),
        "custom_fallback_used": False,
    }


def run_phase(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    config = k01.load_config(args.config)
    paths = output_dirs(config)
    ensure_output_dirs(paths)
    candidate = select_candidate(config, args.candidate_id)
    protected_roots, forbidden_roots = safety_snapshot_roots(config)
    protected_before = k01.snapshot(protected_roots)
    forbidden_before = k01.snapshot(forbidden_roots)
    label = phase_label(args.phase)
    max_nodes = max_nodes_for_phase(config, args.phase, args.max_nodes)
    result: dict[str, Any] = {}

    if args.phase in {"construct_smoke", "construct_pilot"}:
        constructed = construct_kernel_phase(
            config,
            paths,
            candidate,
            label,
            max_nodes,
            args.resume,
            args.overwrite,
        )
        qc = constructed["qc"]
        result.update(qc)
        result.update(
            {
                "status": "PASSED" if qc["kernel_qc_pass"] else "FAILED",
                "kernel_decision_category": "kernel_pass_gpcca_ready"
                if qc["kernel_qc_pass"]
                else (
                    "kernel_fail_component_balance"
                    if qc["component_dominance_warning"]
                    else "kernel_fail_safety_or_schema"
                ),
                "pygpcca_decision_category": "not_run_kernel_phase",
                "matrix_path": constructed["matrix_path"],
                "node_table_path": constructed["node_table_path"],
                "time_points": constructed["time_points"],
                "time_pairs": constructed["time_pairs"],
                "k_gpcca_constructed": True,
                "pygpcca_executed": False,
                "cellrank_executed": False,
                "next_recommended_step": "Run gated standard pyGPCCA smoke/pilot only if kernel QC passes.",
            }
        )
    else:
        matrix_path = existing_kernel_path(paths, candidate.grid_id, label)
        node_table_path = existing_node_table_path(paths, candidate.grid_id, label)
        qc_path = kernel_paths(paths, candidate.grid_id, label)["qc_csv"]
        if not qc_path.exists():
            raise FileNotFoundError(f"Kernel QC is required before pyGPCCA: {qc_path}")
        qc = pd.read_csv(qc_path).iloc[0].to_dict()
        if not kernel_qc_allows_gpcca(qc):
            raise RuntimeError("pyGPCCA phase blocked because kernel QC did not pass")
        gpcca = run_pygpcca_subprocess(
            matrix_path,
            node_table_path,
            paths["gpcca"],
            candidate.grid_id,
            label,
            gpcca_k_values(args.n_macrostates),
            write_outputs=args.phase == "gpcca_pilot",
        )
        candidates_csv = Path(gpcca.get("candidate_csv", "")) if gpcca.get("candidate_csv") else None
        if candidates_csv and candidates_csv.exists():
            dest = paths["reports"] / (
                "k_gpcca_02_pygpcca_smoke_candidates.csv"
                if args.phase == "gpcca_smoke"
                else "k_gpcca_02_pygpcca_candidate_table.csv"
            )
            atomic_write_csv(dest, pd.read_csv(candidates_csv))
        annotation = annotate_gpcca_outputs(paths, candidate.grid_id) if args.phase == "gpcca_pilot" else {
            "annotation_status": "not computed",
            "reason": "smoke phase only",
        }
        figures = generate_figures(paths, candidate.grid_id) if args.phase == "gpcca_pilot" and gpcca.get("success") else []
        succeeded = gpcca.get("succeeded", [])
        failed = gpcca.get("failed", [])
        result.update(
            {
                "status": "PASSED" if succeeded else "FAILED",
                "kernel_decision_category": "kernel_pass_gpcca_ready",
                "pygpcca_decision_category": "standard_pygpcca_pilot_success"
                if args.phase == "gpcca_pilot" and succeeded and not failed
                else (
                    "standard_pygpcca_partial_success"
                    if succeeded
                    else "standard_pygpcca_failed_no_fallback"
                ),
                "environment_used": gpcca.get("environment", "not available"),
                "tmpdir_used": gpcca.get("tmpdir", str(TMPDIR)),
                "k_values_tested": gpcca.get("k_values", []),
                "k_values_succeeded": succeeded,
                "k_values_failed": failed,
                "selected_k": gpcca.get("written", {}).get("selected_k", succeeded[0] if succeeded else None),
                "gpcca_stdout": gpcca.get("stdout", ""),
                "gpcca_stderr": gpcca.get("stderr", ""),
                "gpcca_returncode": gpcca.get("returncode", None),
                "fate_probabilities_computed": False,
                "fate_probability_label": "not computed",
                "annotation_status": annotation.get("annotation_status", "not computed"),
                "annotation_reason": annotation.get("reason", ""),
                "figure_count": len(figures),
                "k_gpcca_constructed": False,
                "pygpcca_executed": True,
                "cellrank_executed": False,
                "next_recommended_step": "Proceed to K_gpcca-03 biological benchmark and pilot fate-probability interpretation."
                if succeeded
                else "Proceed to K_gpcca kernel redesign or supernode strategy review; do not use a heuristic fallback.",
            }
        )
        result.update(annotation)

    result["runtime_seconds"] = time.perf_counter() - start
    result.update(
        phase_summary_base(
            args.phase,
            candidate,
            paths,
            protected_before,
            forbidden_before,
            protected_roots,
            forbidden_roots,
        )
    )
    kernel_inventory = scan_kernel_inventory(paths)
    gpcca_inventory = scan_gpcca_inventory(paths)
    write_global_reports(paths, result, kernel_inventory, gpcca_inventory)
    atomic_write_json(paths["root"] / "k_gpcca_02_summary.json", result)
    return result


def main() -> None:
    args = parse_args()
    result = run_phase(args)
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "phase": result.get("phase"),
                "candidate_id": result.get("candidate_id"),
                "kernel_decision_category": result.get("kernel_decision_category"),
                "pygpcca_decision_category": result.get("pygpcca_decision_category"),
                "selected_node_count": result.get("selected_node_count"),
                "row_sum_max_error": result.get("row_sum_max_error"),
                "k_values_succeeded": result.get("k_values_succeeded"),
                "k_values_failed": result.get("k_values_failed"),
                "upstream_metadata_diff_count": result.get("upstream_metadata_diff_count"),
                "forbidden_downstream_diff_count": result.get("forbidden_downstream_diff_count"),
                "ssd_output_count": result.get("ssd_output_count"),
                "custom_fallback_used": result.get("custom_fallback_used"),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
