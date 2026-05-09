#!/usr/bin/env python
"""Assemble M4A global sparse transition objects from frozen M3 edge shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config


DEFAULT_CONFIG = "configs/m4a_markov_assembly.yaml"
DEFAULT_BATCH_ROWS = 1_000_000
DEFAULT_PREFLIGHT_ROWS_PER_SHARD = 100_000
NO_DOWNSTREAM_FLAGS = {
    "no_gpcca": True,
    "no_terminal_state_inference": True,
    "no_fate_probability": True,
    "no_absorption_probability": True,
    "no_branched_nicheflow": True,
    "no_m5": True,
    "no_regulator_analysis": True,
}
NODE_METADATA_COLUMNS = [
    "slice_id",
    "slice_file",
    "time",
    "time_day",
    "mouse_id",
    "anchor_index",
    "anchor_cell_id",
    "cell_type_l1",
    "cell_type_l2",
    "cell_type_l3",
]
REQUIRED_NODE_COLUMNS = {"slice_id", "time", "time_day", "anchor_index"}
OPTIONAL_EDGE_METADATA_COLUMNS = [
    "source_time",
    "target_time",
    "source_slice_id",
    "target_slice_id",
    "source_mouse_id",
    "target_mouse_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument("--preflight-rows-per-shard", type=int, default=DEFAULT_PREFLIGHT_ROWS_PER_SHARD)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(json_safe(payload), indent=2) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_save_npz(path: Path, matrix: sp.spmatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    sp.save_npz(tmp, matrix)
    os.replace(tmp, path)


def assert_no_ssd_path(path: Path, label: str) -> None:
    resolved = str(path.expanduser().resolve())
    if resolved == "/ssd" or resolved.startswith("/ssd/"):
        raise ValueError(f"Refusing to use /ssd for {label}: {path}")


def configured_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    for key, path in paths.items():
        assert_no_ssd_path(path, f"paths.{key}")
    return paths


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {"paths", "assembly", "validation", "visualization"}
    missing = sorted(required_sections - set(config))
    if missing:
        raise KeyError(f"M4A config is missing required sections: {missing}")
    assembly = config["assembly"]
    if assembly["terminal_time_policy"] != "final_time_no_outgoing":
        raise ValueError("M4A-01 supports terminal_time_policy=final_time_no_outgoing only.")
    if str(assembly["dtype"]) not in {"float32", "float64"}:
        raise ValueError("assembly.dtype must be float32 or float64.")
    if str(assembly["index_dtype"]) != "int64":
        raise ValueError("M4A-01 is scoped to assembly.index_dtype=int64.")


def parquet_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.read_schema(path).names)


def iter_parquet_batches(path: Path, columns: list[str], batch_rows: int) -> Any:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    yield from parquet.iter_batches(batch_size=int(batch_rows), columns=columns)


def m2_representation_paths(m2_root: Path) -> list[Path]:
    completed = m2_root / "completed_slices.csv"
    if completed.exists():
        summary = pd.read_csv(completed)
        if "output_path" not in summary.columns:
            raise KeyError("M2 completed_slices.csv is missing output_path.")
        paths = [Path(value) for value in summary["output_path"].astype(str)]
    else:
        paths = sorted(m2_root.glob("*/m2_representation_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No M2 representation parquets found under {m2_root}.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing M2 representation parquets: {missing[:5]}")
    return paths


def build_node_table_from_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No node metadata frames were provided.")
    table = pd.concat(frames, ignore_index=True)
    missing = sorted(REQUIRED_NODE_COLUMNS - set(table.columns))
    if missing:
        raise KeyError(f"M2 node metadata is missing required columns: {missing}")
    for column in NODE_METADATA_COLUMNS:
        if column not in table.columns:
            table[column] = pd.NA
    if int(table[["slice_id", "anchor_index", "time", "time_day"]].isna().sum().sum()):
        raise ValueError("M2 node metadata has missing required identity/time values.")
    table = table[NODE_METADATA_COLUMNS].copy()
    table["anchor_id"] = table["slice_id"].astype(str) + "::" + table["anchor_index"].astype(str)
    if bool(table["anchor_id"].duplicated().any()):
        examples = table.loc[table["anchor_id"].duplicated(), "anchor_id"].head(5).tolist()
        raise ValueError(f"Duplicate anchor IDs in M2 node table: {examples}")
    table = table.sort_values(["time_day", "time", "slice_id", "anchor_index"], kind="mergesort").reset_index(drop=True)
    table.insert(0, "global_node_index", np.arange(len(table), dtype=np.int64))
    final_time_day, final_time = infer_final_time(table)
    table["is_final_time"] = np.isclose(table["time_day"].astype(float), final_time_day) & (
        table["time"].astype(str) == final_time
    )
    if bool(table["global_node_index"].duplicated().any()):
        raise ValueError("Duplicate global node indices were produced.")
    return table[
        [
            "global_node_index",
            "anchor_id",
            "slice_id",
            "anchor_index",
            "anchor_cell_id",
            "time",
            "time_day",
            "mouse_id",
            "cell_type_l1",
            "cell_type_l2",
            "cell_type_l3",
            "is_final_time",
        ]
    ]


def build_global_node_table(m2_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in m2_representation_paths(m2_root):
        columns = parquet_columns(path)
        missing = sorted(REQUIRED_NODE_COLUMNS - columns)
        if missing:
            raise KeyError(f"{path} is missing required M2 node columns: {missing}")
        read_columns = [column for column in NODE_METADATA_COLUMNS if column in columns]
        frames.append(pd.read_parquet(path, columns=read_columns))
    return build_node_table_from_frames(frames)


def infer_final_time(node_table: pd.DataFrame) -> tuple[float, str]:
    max_day = float(node_table["time_day"].astype(float).max())
    labels = sorted(
        node_table.loc[np.isclose(node_table["time_day"].astype(float), max_day), "time"].dropna().astype(str).unique()
    )
    if len(labels) != 1:
        raise ValueError(f"Expected exactly one final time label for max time_day {max_day}, found {labels}.")
    return max_day, labels[0]


def expected_node_count_from_m2_summary(m2_root: Path) -> int | None:
    completed = m2_root / "completed_slices.csv"
    if not completed.exists():
        return None
    summary = pd.read_csv(completed)
    if "output_rows" not in summary.columns:
        return None
    return int(summary["output_rows"].sum())


def load_m3_manifest(path: Path, edge_root: Path, expected_edges: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen M3 manifest: {path}")
    manifest = pd.read_csv(path)
    required = {"shard_id", "observed_edge_rows", "output_parquet", "m3_16_status"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise KeyError(f"Frozen M3 manifest is missing columns: {missing}")
    if not bool((manifest["m3_16_status"].astype(str) == "FINAL_QC_VALIDATED").all()):
        raise ValueError("All M3 manifest rows must have m3_16_status=FINAL_QC_VALIDATED.")
    if int(manifest["observed_edge_rows"].sum()) != int(expected_edges):
        raise ValueError("Frozen M3 manifest edge-row sum does not match M4A config.")
    root = edge_root.resolve()
    for value in manifest["output_parquet"].astype(str):
        path_value = Path(value)
        if not path_value.exists():
            raise FileNotFoundError(f"Missing M3 edge shard: {path_value}")
        assert_no_ssd_path(path_value, "M3 edge shard")
        try:
            path_value.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"M3 edge shard is outside configured edge root: {path_value}") from exc
    return manifest.sort_values("shard_id").reset_index(drop=True)


def load_m3_schema(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen M3 schema: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = set(payload.get("edge_schema_columns", []))
    required = {
        "source_anchor_id",
        "target_anchor_id",
        "row_normalized_transition_prob",
        "raw_edge_weight",
        "mass_adjusted_weight",
    }
    missing = sorted(required - columns)
    if missing:
        raise KeyError(f"Frozen M3 schema is missing required M4A columns: {missing}")
    return payload


def edge_read_columns(config: dict[str, Any], available_columns: set[str]) -> list[str]:
    assembly = config["assembly"]
    required = [
        assembly["source_id_column"],
        assembly["target_id_column"],
        assembly["edge_probability_column"],
        assembly["raw_weight_column"],
        assembly["mass_adjusted_weight_column"],
    ]
    missing = sorted(set(required) - available_columns)
    if missing:
        raise KeyError(f"M3 edge shard is missing required columns: {missing}")
    return required + [column for column in OPTIONAL_EDGE_METADATA_COLUMNS if column in available_columns and column not in required]


def validate_edge_values(frame: pd.DataFrame, prob_col: str, raw_col: str, mass_col: str, config: dict[str, Any]) -> None:
    values = frame[[prob_col, raw_col, mass_col]].to_numpy(dtype=float)
    if bool(config["validation"].get("fail_on_nan", True)) and int((~np.isfinite(values)).sum()):
        raise ValueError("M3 edge batch has NaN or infinite probability/weight values.")
    if bool(config["validation"].get("fail_on_negative_probability", True)) and bool((frame[prob_col].astype(float) < 0).any()):
        raise ValueError("M3 edge batch has negative probabilities.")


def map_edge_batch(frame: pd.DataFrame, anchor_index: pd.Index, source_col: str, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    source = anchor_index.get_indexer(frame[source_col].astype(str))
    target = anchor_index.get_indexer(frame[target_col].astype(str))
    if int((source < 0).sum()) or int((target < 0).sum()):
        bad_source = frame.loc[source < 0, source_col].astype(str).head(5).tolist()
        bad_target = frame.loc[target < 0, target_col].astype(str).head(5).tolist()
        raise ValueError(f"M3 edge endpoints do not map to M2 node table. source={bad_source}, target={bad_target}")
    return source.astype(np.int64, copy=False), target.astype(np.int64, copy=False)


def preflight_edge_endpoint_mapping(
    manifest: pd.DataFrame,
    anchor_index: pd.Index,
    config: dict[str, Any],
    rows_per_shard: int,
) -> dict[str, Any]:
    assembly = config["assembly"]
    checked_rows = 0
    for row in manifest.to_dict("records"):
        path = Path(row["output_parquet"])
        columns = edge_read_columns(config, parquet_columns(path))
        id_columns = [assembly["source_id_column"], assembly["target_id_column"]]
        for batch in iter_parquet_batches(path, id_columns, rows_per_shard):
            frame = batch.to_pandas()
            map_edge_batch(frame, anchor_index, id_columns[0], id_columns[1])
            checked_rows += int(len(frame))
            break
    return {"preflight_shards_checked": int(len(manifest)), "preflight_rows_checked": checked_rows}


def check_duplicate_edge_pairs(source_idx: np.ndarray, target_idx: np.ndarray, n_nodes: int) -> None:
    if len(source_idx) != len(target_idx):
        raise ValueError("source_idx and target_idx must have equal length.")
    keys = source_idx.astype(np.int64, copy=False) * np.int64(n_nodes) + target_idx.astype(np.int64, copy=False)
    keys.sort()
    duplicate_count = int((keys[1:] == keys[:-1]).sum()) if len(keys) > 1 else 0
    if duplicate_count:
        raise ValueError(f"Duplicate (source_node_index, target_node_index) pairs detected before sparse conversion: {duplicate_count}")


def assemble_sparse_matrices(
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    probabilities: np.ndarray,
    raw_weights: np.ndarray,
    mass_weights: np.ndarray,
    n_nodes: int,
    final_node_indices: np.ndarray,
    dtype: str,
    final_time_self_loop_weight: float,
    write_absorbing: bool,
) -> dict[str, sp.csr_matrix]:
    check_duplicate_edge_pairs(source_idx.copy(), target_idx, n_nodes)
    shape = (int(n_nodes), int(n_nodes))
    p_forward = sp.coo_matrix((probabilities.astype(dtype, copy=False), (source_idx, target_idx)), shape=shape).tocsr()
    w_raw = sp.coo_matrix((raw_weights.astype(dtype, copy=False), (source_idx, target_idx)), shape=shape).tocsr()
    w_mass = sp.coo_matrix((mass_weights.astype(dtype, copy=False), (source_idx, target_idx)), shape=shape).tocsr()
    matrices = {
        "P_forward_no_terminal_selfloops": p_forward,
        "W_raw_edge_weight": w_raw,
        "W_mass_adjusted_weight": w_mass,
    }
    if write_absorbing:
        final_mask = np.zeros(int(n_nodes), dtype=bool)
        final_mask[final_node_indices] = True
        diagonal_overlap = final_mask[source_idx] & (source_idx == target_idx)
        if bool(diagonal_overlap.any()):
            raise ValueError("Final-time self-loop coordinates overlap existing forward edges.")
        loops = np.full(len(final_node_indices), float(final_time_self_loop_weight), dtype=dtype)
        absorbing_source = np.concatenate([source_idx, final_node_indices])
        absorbing_target = np.concatenate([target_idx, final_node_indices])
        absorbing_data = np.concatenate([probabilities.astype(dtype, copy=False), loops])
        matrices["P_absorbing_terminal_selfloops"] = sp.coo_matrix(
            (absorbing_data, (absorbing_source, absorbing_target)),
            shape=shape,
        ).tocsr()
    return matrices


def sparse_patterns_equal(left: sp.csr_matrix, right: sp.csr_matrix) -> bool:
    return (
        left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
    )


def row_sum_qc(
    p_forward: sp.csr_matrix,
    p_absorbing: sp.csr_matrix,
    node_table: pd.DataFrame,
    tolerance: float,
) -> dict[str, Any]:
    final_mask = node_table["is_final_time"].to_numpy(dtype=bool)
    non_final_mask = ~final_mask
    forward_row_sums = np.asarray(p_forward.sum(axis=1)).ravel()
    absorbing_row_sums = np.asarray(p_absorbing.sum(axis=1)).ravel()
    nonfinal_forward_error = np.abs(forward_row_sums[non_final_mask] - 1.0)
    final_forward_abs = np.abs(forward_row_sums[final_mask])
    absorbing_error = np.abs(absorbing_row_sums - 1.0)

    def error_stats(values: np.ndarray) -> dict[str, Any]:
        return {
            "max": float(values.max()) if len(values) else 0.0,
            "p99": float(np.quantile(values, 0.99)) if len(values) else 0.0,
            "rows_exceeding_1e_6": int((values > 1e-6).sum()),
            "rows_exceeding_1e_5": int((values > 1e-5).sum()),
        }

    qc = {
        "forward_nonfinal_row_sum_error": error_stats(nonfinal_forward_error),
        "forward_final_row_sum_abs": error_stats(final_forward_abs),
        "absorbing_all_row_sum_error": error_stats(absorbing_error),
        "forward_nonfinal_rows_within_tolerance": bool((nonfinal_forward_error <= tolerance).all()),
        "forward_final_rows_zero": bool((final_forward_abs <= tolerance).all()),
        "absorbing_rows_within_tolerance": bool((absorbing_error <= tolerance).all()),
    }
    if not qc["forward_nonfinal_rows_within_tolerance"]:
        raise ValueError("P_forward non-final source rows are not row-stochastic within tolerance.")
    if not qc["forward_final_rows_zero"]:
        raise ValueError("P_forward final-time rows are not zero within tolerance.")
    if not qc["absorbing_rows_within_tolerance"]:
        raise ValueError("P_absorbing rows are not row-stochastic within tolerance.")
    return qc


def degree_summary(matrix: sp.csr_matrix, node_table: pd.DataFrame) -> pd.DataFrame:
    out_degree = np.diff(matrix.indptr)
    in_degree = np.asarray((matrix != 0).sum(axis=0)).ravel()
    frame = node_table[["time", "time_day", "is_final_time"]].copy()
    frame["out_degree"] = out_degree
    frame["in_degree"] = in_degree
    return (
        frame.groupby(["time", "time_day", "is_final_time"], observed=True)
        .agg(
            nodes=("time", "size"),
            out_degree_min=("out_degree", "min"),
            out_degree_mean=("out_degree", "mean"),
            out_degree_max=("out_degree", "max"),
            in_degree_min=("in_degree", "min"),
            in_degree_mean=("in_degree", "mean"),
            in_degree_max=("in_degree", "max"),
        )
        .reset_index()
        .sort_values("time_day")
    )


def update_dict_sum(target: dict[str, float], keys: pd.Series, values: np.ndarray) -> None:
    grouped = pd.DataFrame({"key": keys.astype(str), "value": values}).groupby("key", observed=True)["value"].sum()
    for key, value in grouped.items():
        target[str(key)] += float(value)


def stream_edges(
    manifest: pd.DataFrame,
    node_table: pd.DataFrame,
    config: dict[str, Any],
    batch_rows: int,
) -> dict[str, Any]:
    assembly = config["assembly"]
    total_edges = int(manifest["observed_edge_rows"].sum())
    dtype = np.dtype(str(assembly["dtype"]))
    source_idx = np.empty(total_edges, dtype=np.int64)
    target_idx = np.empty(total_edges, dtype=np.int64)
    probabilities = np.empty(total_edges, dtype=dtype)
    raw_weights = np.empty(total_edges, dtype=dtype)
    mass_weights = np.empty(total_edges, dtype=dtype)
    anchor_index = pd.Index(node_table["anchor_id"].astype(str))

    edge_manifest_rows: list[dict[str, Any]] = []
    transition_mass = defaultdict(float)
    target_slice_mass = defaultdict(float)
    target_slice_edges = defaultdict(float)
    target_mouse_mass = defaultdict(float)
    target_mouse_edges = defaultdict(float)
    edge_weight_samples = {"probability": [], "raw_edge_weight": [], "mass_adjusted_weight": []}

    cursor = 0
    for shard in manifest.to_dict("records"):
        path = Path(shard["output_parquet"])
        columns = edge_read_columns(config, parquet_columns(path))
        shard_rows = 0
        shard_prob_sum = 0.0
        shard_raw_sum = 0.0
        shard_mass_sum = 0.0
        source_time = str(shard.get("source_time", ""))
        target_time = str(shard.get("target_time", ""))
        for batch in iter_parquet_batches(path, columns, batch_rows):
            frame = batch.to_pandas()
            validate_edge_values(
                frame,
                assembly["edge_probability_column"],
                assembly["raw_weight_column"],
                assembly["mass_adjusted_weight_column"],
                config,
            )
            src, tgt = map_edge_batch(frame, anchor_index, assembly["source_id_column"], assembly["target_id_column"])
            stop = cursor + len(frame)
            source_idx[cursor:stop] = src
            target_idx[cursor:stop] = tgt
            prob_values = frame[assembly["edge_probability_column"]].to_numpy(dtype=dtype, copy=False)
            raw_values = frame[assembly["raw_weight_column"]].to_numpy(dtype=dtype, copy=False)
            mass_values = frame[assembly["mass_adjusted_weight_column"]].to_numpy(dtype=dtype, copy=False)
            probabilities[cursor:stop] = prob_values
            raw_weights[cursor:stop] = raw_values
            mass_weights[cursor:stop] = mass_values
            shard_rows += int(len(frame))
            shard_prob_sum += float(prob_values.astype(float).sum())
            shard_raw_sum += float(raw_values.astype(float).sum())
            shard_mass_sum += float(mass_values.astype(float).sum())
            if "source_time" in frame.columns and "target_time" in frame.columns:
                pair_keys = frame["source_time"].astype(str) + "->" + frame["target_time"].astype(str)
                update_dict_sum(transition_mass, pair_keys, prob_values.astype(float))
            if "target_slice_id" in frame.columns:
                update_dict_sum(target_slice_mass, frame["target_slice_id"], prob_values.astype(float))
                update_dict_sum(target_slice_edges, frame["target_slice_id"], np.ones(len(frame), dtype=float))
            if "target_mouse_id" in frame.columns:
                update_dict_sum(target_mouse_mass, frame["target_mouse_id"], prob_values.astype(float))
                update_dict_sum(target_mouse_edges, frame["target_mouse_id"], np.ones(len(frame), dtype=float))
            sample_count = min(5000, len(frame))
            if sample_count:
                sample_idx = np.linspace(0, len(frame) - 1, sample_count, dtype=np.int64)
                edge_weight_samples["probability"].append(prob_values[sample_idx].astype(float))
                edge_weight_samples["raw_edge_weight"].append(raw_values[sample_idx].astype(float))
                edge_weight_samples["mass_adjusted_weight"].append(mass_values[sample_idx].astype(float))
            cursor = stop
        if shard_rows != int(shard["observed_edge_rows"]):
            raise ValueError(f"Shard {shard['shard_id']} rows {shard_rows} != manifest {shard['observed_edge_rows']}.")
        edge_manifest_rows.append(
            {
                "shard_id": shard["shard_id"],
                "source_time": source_time,
                "target_time": target_time,
                "edge_rows": shard_rows,
                "probability_sum": shard_prob_sum,
                "raw_edge_weight_sum": shard_raw_sum,
                "mass_adjusted_weight_sum": shard_mass_sum,
                "output_parquet": str(path),
                "read_status": "READ_OK",
            }
        )
    if cursor != total_edges:
        raise ValueError(f"Read {cursor} total edges, expected {total_edges}.")
    samples = {
        key: np.concatenate(values) if values else np.array([], dtype=float)
        for key, values in edge_weight_samples.items()
    }
    return {
        "source_idx": source_idx,
        "target_idx": target_idx,
        "probabilities": probabilities,
        "raw_weights": raw_weights,
        "mass_weights": mass_weights,
        "edge_manifest": pd.DataFrame(edge_manifest_rows),
        "transition_mass": dict(transition_mass),
        "target_slice_summary": diagnostic_summary(target_slice_edges, target_slice_mass, "target_slice_id"),
        "target_mouse_summary": diagnostic_summary(target_mouse_edges, target_mouse_mass, "target_mouse_id"),
        "edge_weight_samples": samples,
    }


def diagnostic_summary(edge_counts: dict[str, float], mass: dict[str, float], key_name: str) -> pd.DataFrame:
    keys = sorted(set(edge_counts) | set(mass))
    return pd.DataFrame(
        {
            key_name: keys,
            "incoming_edges": [int(edge_counts.get(key, 0.0)) for key in keys],
            "incoming_probability_mass": [float(mass.get(key, 0.0)) for key in keys],
        }
    )


def validate_assembled_matrices(
    matrices: dict[str, sp.csr_matrix],
    expected_edges: int,
    final_node_count: int,
    node_count: int,
) -> dict[str, Any]:
    p_forward = matrices["P_forward_no_terminal_selfloops"]
    p_absorbing = matrices["P_absorbing_terminal_selfloops"]
    w_raw = matrices["W_raw_edge_weight"]
    w_mass = matrices["W_mass_adjusted_weight"]
    expected_shape = (node_count, node_count)
    for name, matrix in matrices.items():
        if matrix.shape != expected_shape:
            raise ValueError(f"{name} shape {matrix.shape} != {expected_shape}.")
    if int(p_forward.nnz) != int(expected_edges):
        raise ValueError(f"P_forward nnz {p_forward.nnz} != expected edges {expected_edges}.")
    expected_absorbing_nnz = int(expected_edges + final_node_count)
    if int(p_absorbing.nnz) != expected_absorbing_nnz:
        diag_overlap = int(p_forward.diagonal().astype(bool).sum())
        raise ValueError(
            "P_absorbing nnz does not equal edge rows plus final node count: "
            f"observed={p_absorbing.nnz}, expected={expected_absorbing_nnz}, "
            f"p_forward_diagonal_nnz={diag_overlap}, final_node_count={final_node_count}."
        )
    if not sparse_patterns_equal(p_forward, w_raw) or not sparse_patterns_equal(p_forward, w_mass):
        raise ValueError("Weight matrices do not match P_forward edge pattern.")
    return {
        "shape": list(expected_shape),
        "p_forward_nnz": int(p_forward.nnz),
        "p_absorbing_nnz": int(p_absorbing.nnz),
        "w_raw_nnz": int(w_raw.nnz),
        "w_mass_adjusted_nnz": int(w_mass.nnz),
    }


def transition_mass_frame(transition_mass: dict[str, float]) -> pd.DataFrame:
    rows = []
    for pair, mass in sorted(transition_mass.items()):
        source, target = pair.split("->", 1) if "->" in pair else (pair, "")
        rows.append({"source_time": source, "target_time": target, "transition_probability_mass": float(mass)})
    return pd.DataFrame(rows)


def qc_summary_frame(qc: dict[str, Any], matrix_qc: dict[str, Any], node_table: pd.DataFrame, edge_count: int) -> pd.DataFrame:
    rows = [
        {"metric": "global_nodes", "value": int(len(node_table))},
        {"metric": "final_time_nodes", "value": int(node_table["is_final_time"].sum())},
        {"metric": "assembled_edges", "value": int(edge_count)},
        {"metric": "matrix_shape_rows", "value": int(matrix_qc["shape"][0])},
        {"metric": "matrix_shape_cols", "value": int(matrix_qc["shape"][1])},
        {"metric": "p_forward_nnz", "value": int(matrix_qc["p_forward_nnz"])},
        {"metric": "p_absorbing_nnz", "value": int(matrix_qc["p_absorbing_nnz"])},
        {
            "metric": "forward_nonfinal_row_sum_error_max",
            "value": qc["forward_nonfinal_row_sum_error"]["max"],
        },
        {
            "metric": "forward_nonfinal_row_sum_error_p99",
            "value": qc["forward_nonfinal_row_sum_error"]["p99"],
        },
        {
            "metric": "forward_nonfinal_rows_exceeding_1e_6",
            "value": qc["forward_nonfinal_row_sum_error"]["rows_exceeding_1e_6"],
        },
        {
            "metric": "forward_nonfinal_rows_exceeding_1e_5",
            "value": qc["forward_nonfinal_row_sum_error"]["rows_exceeding_1e_5"],
        },
        {"metric": "absorbing_all_row_sum_error_max", "value": qc["absorbing_all_row_sum_error"]["max"]},
        {"metric": "absorbing_all_row_sum_error_p99", "value": qc["absorbing_all_row_sum_error"]["p99"]},
        {
            "metric": "absorbing_rows_exceeding_1e_6",
            "value": qc["absorbing_all_row_sum_error"]["rows_exceeding_1e_6"],
        },
        {
            "metric": "absorbing_rows_exceeding_1e_5",
            "value": qc["absorbing_all_row_sum_error"]["rows_exceeding_1e_5"],
        },
    ]
    return pd.DataFrame(rows)


def report_markdown(
    config: dict[str, Any],
    paths: dict[str, Path],
    final_time: str,
    final_time_day: float,
    manifest: pd.DataFrame,
    node_table: pd.DataFrame,
    qc: dict[str, Any],
    matrix_qc: dict[str, Any],
    degree: pd.DataFrame,
    transition_mass: pd.DataFrame,
    figure_warnings: list[str],
    runtime_seconds: float,
) -> str:
    lines = [
        "# M4A-01 Global Sparse Transition Object Assembly",
        "",
        "This stage assembled sparse structural transition objects from frozen M3 local edge shards.",
        "The absorbing terminal self-loop object is a Markov-ready structural variant only; it is not a fate result and does not compute absorption probabilities.",
        "",
        "## Inputs",
        f"- M3 manifest: {paths['m3_manifest']}",
        f"- M3 edge root: {paths['m3_edge_root']}",
        f"- M2 node root: {paths['m2_by_slice_root']}",
        f"- M3 shards read: {len(manifest)}",
        f"- M3 edge rows read: {int(manifest['observed_edge_rows'].sum())}",
        "",
        "## Node And Matrix Summary",
        f"- global nodes: {len(node_table)}",
        f"- final time inferred from max time_day: {final_time} ({final_time_day:g})",
        f"- final-time nodes: {int(node_table['is_final_time'].sum())}",
        f"- sparse matrix shape: {tuple(matrix_qc['shape'])}",
        f"- P_forward nnz: {matrix_qc['p_forward_nnz']}",
        f"- P_absorbing nnz: {matrix_qc['p_absorbing_nnz']}",
        f"- W_raw_edge_weight nnz: {matrix_qc['w_raw_nnz']}",
        f"- W_mass_adjusted_weight nnz: {matrix_qc['w_mass_adjusted_nnz']}",
        f"- runtime seconds: {runtime_seconds:.3f}",
        "",
        "## Row-Sum QC",
        f"- forward non-final max error: {qc['forward_nonfinal_row_sum_error']['max']:.6g}",
        f"- forward non-final p99 error: {qc['forward_nonfinal_row_sum_error']['p99']:.6g}",
        f"- forward non-final rows > 1e-6: {qc['forward_nonfinal_row_sum_error']['rows_exceeding_1e_6']}",
        f"- forward non-final rows > 1e-5: {qc['forward_nonfinal_row_sum_error']['rows_exceeding_1e_5']}",
        f"- forward final max row sum: {qc['forward_final_row_sum_abs']['max']:.6g}",
        f"- absorbing max row-sum error: {qc['absorbing_all_row_sum_error']['max']:.6g}",
        f"- absorbing p99 row-sum error: {qc['absorbing_all_row_sum_error']['p99']:.6g}",
        f"- absorbing rows > 1e-6: {qc['absorbing_all_row_sum_error']['rows_exceeding_1e_6']}",
        f"- absorbing rows > 1e-5: {qc['absorbing_all_row_sum_error']['rows_exceeding_1e_5']}",
        "",
        "## Degree QC By Time",
    ]
    for row in degree.to_dict("records"):
        lines.append(
            "- "
            f"{row['time']}: nodes={int(row['nodes'])}, "
            f"out_degree_min/mean/max={row['out_degree_min']:.0f}/{row['out_degree_mean']:.3f}/{row['out_degree_max']:.0f}, "
            f"in_degree_min/mean/max={row['in_degree_min']:.0f}/{row['in_degree_mean']:.3f}/{row['in_degree_max']:.0f}"
        )
    lines.extend(["", "## Transition Mass By Time Pair"])
    for row in transition_mass.to_dict("records"):
        lines.append(f"- {row['source_time']}->{row['target_time']}: {row['transition_probability_mass']:.6g}")
    lines.extend(
        [
            "",
            "## Outputs",
            f"- transition objects: {paths['output_root'] / 'transition_objects'}",
            f"- node table: {paths['output_root'] / 'node_table' / 'global_node_table.parquet'}",
            f"- edge manifest: {paths['output_root'] / 'edge_manifest' / 'm4a_edge_shard_manifest.csv'}",
            f"- reports: {paths['reports_dir']}",
            "",
            "## Downstream Boundary",
            "- no GPCCA was run",
            "- no terminal-state inference was run",
            "- no fate probability was computed",
            "- no absorption probability was computed",
            "- no Branched NicheFlow was run",
            "- no M5 was run",
            "- no regulator analysis was run",
        ]
    )
    if figure_warnings:
        lines.extend(["", "## Figure Warnings", *[f"- {warning}" for warning in figure_warnings]])
    return "\n".join(lines).rstrip() + "\n"


def make_figures(
    figures_dir: Path,
    matrices: dict[str, sp.csr_matrix],
    node_table: pd.DataFrame,
    degree: pd.DataFrame,
    transition_mass: pd.DataFrame,
    edge_samples: dict[str, np.ndarray],
    target_slice_summary: pd.DataFrame,
    target_mouse_summary: pd.DataFrame,
    figure_failure_is_warning: bool,
) -> list[str]:
    warnings: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir.mkdir(parents=True, exist_ok=True)
        forward_sums = np.asarray(matrices["P_forward_no_terminal_selfloops"].sum(axis=1)).ravel()
        absorbing_sums = np.asarray(matrices["P_absorbing_terminal_selfloops"].sum(axis=1)).ravel()
        row_frame = node_table[["time", "time_day"]].copy()
        row_frame["forward_sum"] = forward_sums
        row_frame["absorbing_sum"] = absorbing_sums
        labels = degree["time"].astype(str)

        fig, ax = plt.subplots(figsize=(8, 4))
        for time, group in row_frame.groupby("time", sort=False):
            ax.hist(group["forward_sum"], bins=40, alpha=0.45, label=str(time), histtype="stepfilled")
        ax.set_title("Forward row-sum distribution by time")
        ax.set_xlabel("row sum")
        ax.set_ylabel("nodes")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures_dir / "m4a_row_sum_distribution_forward.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        for time, group in row_frame.groupby("time", sort=False):
            ax.hist(group["absorbing_sum"], bins=40, alpha=0.45, label=str(time), histtype="stepfilled")
        ax.set_title("Absorbing row-sum distribution by time")
        ax.set_xlabel("row sum")
        ax.set_ylabel("nodes")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures_dir / "m4a_row_sum_distribution_absorbing.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, degree["out_degree_mean"])
        ax.errorbar(
            labels,
            degree["out_degree_mean"],
            yerr=[degree["out_degree_mean"] - degree["out_degree_min"], degree["out_degree_max"] - degree["out_degree_mean"]],
            fmt="none",
            color="black",
            linewidth=1,
        )
        ax.set_title("Forward out-degree by time")
        ax.set_ylabel("degree")
        fig.tight_layout()
        fig.savefig(figures_dir / "m4a_out_degree_by_time.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, degree["in_degree_mean"])
        ax.set_title("Forward in-degree by time")
        ax.set_ylabel("mean in-degree")
        fig.tight_layout()
        fig.savefig(figures_dir / "m4a_in_degree_by_time.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        tm_labels = transition_mass["source_time"].astype(str) + "->" + transition_mass["target_time"].astype(str)
        ax.bar(tm_labels, transition_mass["transition_probability_mass"])
        ax.set_title("Transition probability mass by time pair")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(figures_dir / "m4a_transition_mass_by_time_pair.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        for label, values in edge_samples.items():
            if len(values):
                ax.hist(np.log10(np.clip(values.astype(float), 1e-12, None)), bins=80, alpha=0.45, label=label)
        ax.set_title("Edge weight distribution")
        ax.set_xlabel("log10(value)")
        ax.set_ylabel("sampled edges")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures_dir / "m4a_edge_weight_distribution.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].bar(labels, degree["out_degree_mean"])
        axes[0, 0].set_title("Out-degree mean")
        axes[0, 1].bar(labels, degree["in_degree_mean"])
        axes[0, 1].set_title("In-degree mean")
        axes[1, 0].bar(tm_labels, transition_mass["transition_probability_mass"])
        axes[1, 0].tick_params(axis="x", rotation=25)
        axes[1, 0].set_title("Transition mass")
        axes[1, 1].hist(forward_sums, bins=50)
        axes[1, 1].set_title("Forward row sums")
        fig.tight_layout()
        fig.savefig(figures_dir / "m4a_transition_object_qc_dashboard.png", dpi=140)
        plt.close(fig)

        for frame, key, name, title in [
            (target_slice_summary, "target_slice_id", "m4a_target_slice_incoming_mass.png", "Incoming mass by target slice"),
            (target_mouse_summary, "target_mouse_id", "m4a_target_mouse_incoming_mass.png", "Incoming mass by target mouse"),
        ]:
            if len(frame):
                top = frame.sort_values("incoming_probability_mass", ascending=False).head(30)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(top[key].astype(str), top["incoming_probability_mass"])
                ax.set_title(title)
                ax.tick_params(axis="x", rotation=90, labelsize=6)
                fig.tight_layout()
                fig.savefig(figures_dir / name, dpi=140)
                plt.close(fig)

        if len(transition_mass):
            pivot = transition_mass.pivot_table(
                index="source_time",
                columns="target_time",
                values="transition_probability_mass",
                fill_value=0.0,
            )
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_title("Source-time to target-time mass")
            fig.tight_layout()
            fig.savefig(figures_dir / "m4a_source_target_time_block_heatmap.png", dpi=140)
            plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        if not figure_failure_is_warning:
            raise
        warnings.append(f"Figure generation failed after assembly/QC passed: {exc}")
    return warnings


def write_outputs(
    config: dict[str, Any],
    paths: dict[str, Path],
    manifest: pd.DataFrame,
    node_table: pd.DataFrame,
    matrices: dict[str, sp.csr_matrix],
    stream: dict[str, Any],
    qc: dict[str, Any],
    matrix_qc: dict[str, Any],
    preflight: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Path]:
    output_root = paths["output_root"]
    transition_dir = output_root / "transition_objects"
    node_dir = output_root / "node_table"
    edge_manifest_dir = output_root / "edge_manifest"
    reports_dir = paths["reports_dir"]
    figures_dir = paths["figures_dir"]
    degree = degree_summary(matrices["P_forward_no_terminal_selfloops"], node_table)
    transition_mass = transition_mass_frame(stream["transition_mass"])
    figure_warnings: list[str] = []
    if bool(config["visualization"].get("make_figures", True)):
        figure_warnings = make_figures(
            figures_dir,
            matrices,
            node_table,
            degree,
            transition_mass,
            stream["edge_weight_samples"],
            stream["target_slice_summary"],
            stream["target_mouse_summary"],
            bool(config["visualization"].get("figure_failure_is_warning", True)),
        )

    output_paths = {
        "p_forward": transition_dir / "P_forward_no_terminal_selfloops.npz",
        "p_absorbing": transition_dir / "P_absorbing_terminal_selfloops.npz",
        "w_raw": transition_dir / "W_raw_edge_weight.npz",
        "w_mass": transition_dir / "W_mass_adjusted_weight.npz",
        "node_table": node_dir / "global_node_table.parquet",
        "edge_manifest": edge_manifest_dir / "m4a_edge_shard_manifest.csv",
        "report": reports_dir / "m4a_assembly_report.md",
        "qc_summary": reports_dir / "m4a_assembly_qc_summary.csv",
        "schema": reports_dir / "m4a_transition_object_schema.json",
        "degree_summary": reports_dir / "m4a_degree_summary_by_time.csv",
        "transition_mass": reports_dir / "m4a_transition_mass_by_time_pair.csv",
        "target_slice_summary": reports_dir / "m4a_target_slice_incoming_summary.csv",
        "target_mouse_summary": reports_dir / "m4a_target_mouse_incoming_summary.csv",
    }
    atomic_save_npz(output_paths["p_forward"], matrices["P_forward_no_terminal_selfloops"])
    atomic_save_npz(output_paths["p_absorbing"], matrices["P_absorbing_terminal_selfloops"])
    atomic_save_npz(output_paths["w_raw"], matrices["W_raw_edge_weight"])
    atomic_save_npz(output_paths["w_mass"], matrices["W_mass_adjusted_weight"])
    atomic_write_parquet(output_paths["node_table"], node_table)
    atomic_write_csv(output_paths["edge_manifest"], stream["edge_manifest"])
    atomic_write_csv(output_paths["degree_summary"], degree)
    atomic_write_csv(output_paths["transition_mass"], transition_mass)
    atomic_write_csv(output_paths["target_slice_summary"], stream["target_slice_summary"])
    atomic_write_csv(output_paths["target_mouse_summary"], stream["target_mouse_summary"])
    atomic_write_csv(output_paths["qc_summary"], qc_summary_frame(qc, matrix_qc, node_table, int(manifest["observed_edge_rows"].sum())))

    final_time_day, final_time = infer_final_time(node_table)
    atomic_write_json(
        output_paths["schema"],
        {
            "schema_version": "m4a_transition_object_schema_v1",
            "generated_at_utc": utc_now_iso(),
            "assembly_config": config["assembly"],
            "validation_config": config["validation"],
            "node_count": int(len(node_table)),
            "final_time": final_time,
            "final_time_day": final_time_day,
            "matrix_qc": matrix_qc,
            "row_sum_qc": qc,
            "preflight": preflight,
            "outputs": {key: str(value) for key, value in output_paths.items()},
            "absorbing_terminal_selfloops_semantics": "Markov-ready structural variant only; no fate or absorption probability was computed.",
            **NO_DOWNSTREAM_FLAGS,
        },
    )
    atomic_write_text(
        output_paths["report"],
        report_markdown(
            config,
            paths,
            final_time,
            final_time_day,
            manifest,
            node_table,
            qc,
            matrix_qc,
            degree,
            transition_mass,
            figure_warnings,
            runtime_seconds,
        ),
    )
    return output_paths


def run(args: argparse.Namespace) -> int:
    start = time.monotonic()
    config = load_config(args.config)
    validate_config(config)
    paths = configured_paths(config)
    load_m3_schema(paths["m3_schema"])
    manifest = load_m3_manifest(paths["m3_manifest"], paths["m3_edge_root"], int(config["validation"]["expected_edge_rows"]))
    node_table = build_global_node_table(paths["m2_by_slice_root"])
    expected_nodes = expected_node_count_from_m2_summary(paths["m2_by_slice_root"])
    if expected_nodes is not None and int(len(node_table)) != expected_nodes:
        raise ValueError(f"Global node count {len(node_table)} != expected M2 output rows {expected_nodes}.")
    final_time_day, final_time = infer_final_time(node_table)
    anchor_index = pd.Index(node_table["anchor_id"].astype(str))
    preflight = preflight_edge_endpoint_mapping(manifest, anchor_index, config, int(args.preflight_rows_per_shard))
    stream = stream_edges(manifest, node_table, config, int(args.batch_rows))
    final_indices = node_table.loc[node_table["is_final_time"], "global_node_index"].to_numpy(dtype=np.int64)
    matrices = assemble_sparse_matrices(
        stream["source_idx"],
        stream["target_idx"],
        stream["probabilities"],
        stream["raw_weights"],
        stream["mass_weights"],
        int(len(node_table)),
        final_indices,
        str(config["assembly"]["dtype"]),
        float(config["assembly"]["final_time_self_loop_weight"]),
        bool(config["assembly"]["write_absorbing_terminal_variant"]),
    )
    matrix_qc = validate_assembled_matrices(
        matrices,
        int(config["validation"]["expected_edge_rows"]),
        int(len(final_indices)),
        int(len(node_table)),
    )
    qc = row_sum_qc(
        matrices["P_forward_no_terminal_selfloops"],
        matrices["P_absorbing_terminal_selfloops"],
        node_table,
        float(config["validation"]["row_sum_tolerance"]),
    )
    runtime_seconds = time.monotonic() - start
    outputs = write_outputs(config, paths, manifest, node_table, matrices, stream, qc, matrix_qc, preflight, runtime_seconds)

    print("M4A_01_ASSEMBLY_COMPLETE")
    print(f"M3_SHARDS_READ {len(manifest)}")
    print(f"TOTAL_EDGES_ASSEMBLED {int(config['validation']['expected_edge_rows'])}")
    print(f"GLOBAL_NODES {len(node_table)}")
    print(f"FINAL_TIME {final_time}")
    print(f"FINAL_TIME_DAY {final_time_day:g}")
    print(f"P_FORWARD_SHAPE {tuple(matrix_qc['shape'])}")
    print(f"P_FORWARD_NNZ {matrix_qc['p_forward_nnz']}")
    print(f"P_ABSORBING_NNZ {matrix_qc['p_absorbing_nnz']}")
    print(f"FORWARD_ROW_SUM_ERROR_MAX {qc['forward_nonfinal_row_sum_error']['max']:.9g}")
    print(f"FORWARD_ROW_SUM_ERROR_P99 {qc['forward_nonfinal_row_sum_error']['p99']:.9g}")
    print(f"FORWARD_ROWS_EXCEEDING_1E_6 {qc['forward_nonfinal_row_sum_error']['rows_exceeding_1e_6']}")
    print(f"FORWARD_ROWS_EXCEEDING_1E_5 {qc['forward_nonfinal_row_sum_error']['rows_exceeding_1e_5']}")
    print(f"ABSORBING_ROW_SUM_ERROR_MAX {qc['absorbing_all_row_sum_error']['max']:.9g}")
    print(f"REPORT {outputs['report']}")
    print("P_ABSORBING_IS_STRUCTURAL_MARKOV_READY_VARIANT True")
    print("NO_GPCCA True")
    print("NO_FATE_PROBABILITY True")
    print("NO_ABSORPTION_PROBABILITY True")
    print("NO_BRANCHED_NICHEFLOW True")
    print("NO_M5 True")
    print("NO_REGULATOR_ANALYSIS True")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
