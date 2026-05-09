#!/usr/bin/env python
"""M4A-v2 sparse matrix assembler with guarded dry-run/preflight support."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _thread_var in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_thread_var, "1")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "m4a_v2_assembly.yaml"
REQUIRED_EDGE_COLUMNS = {
    "source_anchor_id",
    "target_anchor_id",
    "v2_row_normalized_transition_prob",
    "v2_unnormalized_weight",
}
OPTIONAL_EDGE_COLUMNS = [
    "source_time",
    "target_time",
    "source_slice_id",
    "target_slice_id",
    "source_mouse_id",
    "target_mouse_id",
]
DRYRUN_OUTPUT_NAMES = [
    "m4a_v2_01_preflight_report.md",
    "m4a_v2_01_dryrun_assembly_plan.csv",
    "m4a_v2_01_dryrun_summary.json",
    "m4a_v2_01_input_schema_validation.csv",
    "m4a_v2_01_node_mapping_validation.csv",
]
SMOKE_REPORT_NAME = "m4a_v2_02_smoke_validation_report.md"
NEXT_STEP_TEXT = (
    "Recommended next step: M4A-v2 full QC / M4A-v1 vs M4A-v2 matrix benchmark. "
    "If the matrix-level comparison is already sufficient, proceed to M4C-v2 planning only; "
    "do not start M4C-v2 execution from this assembly step."
)


@dataclass(frozen=True)
class NodeIndex:
    node_table: pd.DataFrame
    anchor_index: pd.Index
    global_indices: np.ndarray
    final_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = resolved(left)
    right_resolved = resolved(right)
    return is_relative_to(left_resolved, right_resolved) or is_relative_to(right_resolved, left_resolved)


def reject_ssd(path: Path, label: str) -> None:
    path = resolved(path)
    if path == Path("/ssd") or Path("/ssd") in path.parents:
        raise ValueError(f"Refusing /ssd path for {label}: {path}")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
    if isinstance(value, np.bool_):
        return bool(value)
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


def is_smoke_mode(args: argparse.Namespace) -> bool:
    return (not bool(args.dry_run)) and args.max_shards is not None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


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


def config_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for section in ["paths", "inputs"]:
        for key, value in config.get(section, {}).items():
            paths[key] = resolved(value)
    return paths


def validate_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = config_paths(config)
    required = {"output_root", "reports_dir", "tmp_dir", "transition_objects_dir", "node_table_dir"}
    missing = sorted(required - set(paths))
    if missing:
        raise KeyError(f"M4A-v2 config missing output paths: {missing}")
    protected_roots = [resolved(path) for path in config.get("protected_roots", [])]
    output_paths = [paths[name] for name in required]
    for label, path in [(name, paths[name]) for name in required]:
        reject_ssd(path, label)
        if not is_relative_to(path, paths["output_root"]):
            raise ValueError(f"{label} must be under output_root: {path}")
        for protected in protected_roots:
            if paths_overlap(path, protected):
                raise ValueError(f"{label} overlaps protected root {protected}: {path}")
    for name, path in paths.items():
        if name not in required:
            reject_ssd(path, f"input {name}")
    if len({str(path) for path in output_paths}) != len(output_paths):
        raise ValueError("M4A-v2 output paths must be distinct.")
    return paths


def validate_config(config: dict[str, Any]) -> dict[str, Path]:
    required_sections = {"paths", "inputs", "assembly", "expected", "validation", "protected_roots"}
    missing = sorted(required_sections - set(config))
    if missing:
        raise KeyError(f"M4A-v2 config missing sections: {missing}")
    assembly = config["assembly"]
    for key in ["source_id_column", "target_id_column", "probability_column", "weight_column"]:
        if not assembly.get(key):
            raise KeyError(f"M4A-v2 assembly config missing {key}")
    if assembly["terminal_time_policy"] != "final_time_no_outgoing":
        raise ValueError("M4A-v2 supports terminal_time_policy=final_time_no_outgoing only.")
    if str(assembly["dtype"]) not in {"float32", "float64"}:
        raise ValueError("assembly.dtype must be float32 or float64.")
    if str(assembly["index_dtype"]) != "int64":
        raise ValueError("assembly.index_dtype must be int64.")
    return validate_output_paths(config)


def snapshot(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for root in paths:
        if not root.exists():
            out[str(root)] = {"exists": False, "size": -1, "mtime_ns": -1, "is_dir": False}
            continue
        entries = [root, *sorted(path for path in root.rglob("*") if path.exists())] if root.is_dir() else [root]
        for path in entries:
            stat = path.stat()
            out[str(path)] = {
                "exists": True,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "is_dir": path.is_dir(),
            }
    return out


def diff_snapshot(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[str]:
    diffs = []
    for path in sorted(set(before) | set(after)):
        if path not in before:
            diffs.append(f"ADDED\t{path}")
        elif path not in after:
            diffs.append(f"REMOVED\t{path}")
        elif before[path] != after[path]:
            diffs.append(f"CHANGED\t{path}\tbefore={before[path]}\tafter={after[path]}")
    return diffs


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def iter_parquet_batches(path: Path, columns: list[str], batch_rows: int) -> Any:
    parquet = pq.ParquetFile(path)
    yield from parquet.iter_batches(batch_size=int(batch_rows), columns=columns)


def load_m3_v2_qc(config: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    qc_path = paths["m3_v2_qc_summary"]
    if not qc_path.is_file():
        raise FileNotFoundError(f"Missing M3-v2 QC summary: {qc_path}")
    qc = pd.read_csv(qc_path)
    required = {
        "shard_id",
        "time_pair",
        "m3_v2_output_parquet",
        "source_count",
        "retained_v2_edges",
        "targets_per_source_max",
        "row_sum_max_abs_error",
        "probability_nonfinite_count",
        "probability_negative_count",
        "weight_nonfinite_count",
        "weight_negative_count",
    }
    missing = sorted(required - set(qc.columns))
    if missing:
        raise KeyError(f"M3-v2 QC summary missing columns: {missing}")
    expected = config["expected"]
    checks = {
        "shards": int(len(qc)) == int(expected["shards"]),
        "source_rows": int(qc["source_count"].sum()) == int(expected["source_rows"]),
        "retained_v2_edges": int(qc["retained_v2_edges"].sum()) == int(expected["retained_v2_edges"]),
        "targets_per_source_max": int(qc["targets_per_source_max"].max()) <= 10,
        "row_sum_tolerance": float(qc["row_sum_max_abs_error"].max()) <= float(config["validation"]["row_sum_tolerance"]),
        "probability_nonfinite": int(qc["probability_nonfinite_count"].sum()) == 0,
        "probability_negative": int(qc["probability_negative_count"].sum()) == 0,
        "weight_nonfinite": int(qc["weight_nonfinite_count"].sum()) == 0,
        "weight_negative": int(qc["weight_negative_count"].sum()) == 0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"M3-v2 QC checks failed before M4A-v2 preflight: {failed}")
    root = paths["m3_v2_edge_root"]
    for value in qc["m3_v2_output_parquet"].astype(str):
        path = resolved(value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing M3-v2 edge shard: {path}")
        if not is_relative_to(path, root):
            raise ValueError(f"M3-v2 edge shard outside configured root: {path}")
    return qc.sort_values("shard_id", kind="mergesort").reset_index(drop=True)


def validate_benchmark_summary(paths: dict[str, Path]) -> dict[str, Any]:
    path = paths["m3_v2_benchmark_summary"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing M3-v2 benchmark summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASSED":
        raise ValueError(f"M3-v2 benchmark summary status is not PASSED: {payload.get('status')}")
    if payload.get("v2_probability_column") != "v2_row_normalized_transition_prob":
        raise ValueError("M3-v2 benchmark summary does not confirm v2_row_normalized_transition_prob.")
    return payload


def select_shards(qc: pd.DataFrame, max_shards: int | None) -> pd.DataFrame:
    if max_shards is None:
        return qc.copy()
    if max_shards <= 0:
        raise ValueError("--max-shards must be positive when provided.")
    return qc.head(int(max_shards)).copy().reset_index(drop=True)


def load_node_index(config: dict[str, Any], paths: dict[str, Path]) -> NodeIndex:
    path = paths["m4a_v1_node_table"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing M4A-v1 node table: {path}")
    required = ["global_node_index", "anchor_id", "time", "time_day", "is_final_time"]
    node_table = pd.read_parquet(path, columns=required)
    expected = config["expected"]
    if int(len(node_table)) != int(expected["nodes"]):
        raise ValueError(f"Node count {len(node_table)} != expected {expected['nodes']}")
    if not bool(node_table["anchor_id"].is_unique):
        raise ValueError("M4A-v1 node table anchor_id is not unique.")
    if not bool(node_table["global_node_index"].is_unique):
        raise ValueError("M4A-v1 node table global_node_index is not unique.")
    expected_indices = np.arange(len(node_table), dtype=np.int64)
    observed_indices = node_table["global_node_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed_indices, expected_indices):
        raise ValueError(
            "M4A-v1 node table global_node_index must be contiguous and row-aligned."
        )
    final_mask = node_table["is_final_time"].to_numpy(dtype=bool)
    final_count = int(final_mask.sum())
    if final_count != int(expected["final_time_nodes"]):
        raise ValueError(f"Final-time node count {final_count} != expected {expected['final_time_nodes']}")
    final_label = str(config["assembly"].get("final_time_label", "D35"))
    final_labels = set(node_table.loc[final_mask, "time"].astype(str).unique())
    if final_labels != {final_label}:
        raise ValueError(f"Expected final-time label {final_label}, found {sorted(final_labels)}")
    return NodeIndex(
        node_table=node_table,
        anchor_index=pd.Index(node_table["anchor_id"].astype(str)),
        global_indices=node_table["global_node_index"].to_numpy(dtype=np.int64),
        final_mask=final_mask,
    )


def map_edge_batch(
    frame: pd.DataFrame,
    node_index: NodeIndex,
    source_col: str,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    source_pos = node_index.anchor_index.get_indexer(frame[source_col].astype(str))
    target_pos = node_index.anchor_index.get_indexer(frame[target_col].astype(str))
    source_missing = int((source_pos < 0).sum())
    target_missing = int((target_pos < 0).sum())
    if source_missing or target_missing:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), source_missing, target_missing
    return (
        node_index.global_indices[source_pos].astype(np.int64, copy=False),
        node_index.global_indices[target_pos].astype(np.int64, copy=False),
        0,
        0,
    )


def coordinate_keys(source_idx: np.ndarray, target_idx: np.ndarray, node_count: int) -> np.ndarray:
    return source_idx.astype(np.int64, copy=False) * np.int64(node_count) + target_idx.astype(np.int64, copy=False)


def duplicate_coordinate_count(source_idx: np.ndarray, target_idx: np.ndarray, node_count: int) -> int:
    if len(source_idx) != len(target_idx):
        raise ValueError("source_idx and target_idx must have the same length.")
    if len(source_idx) <= 1:
        return 0
    keys = coordinate_keys(source_idx, target_idx, node_count)
    keys.sort()
    return int((keys[1:] == keys[:-1]).sum())


def assert_no_duplicate_coordinates(source_idx: np.ndarray, target_idx: np.ndarray, node_count: int) -> None:
    duplicates = duplicate_coordinate_count(source_idx, target_idx, node_count)
    if duplicates:
        raise ValueError(f"Duplicate source-target matrix coordinates detected: {duplicates}")


def validate_edge_values(frame: pd.DataFrame, probability_col: str, weight_col: str) -> dict[str, int]:
    probabilities = frame[probability_col].to_numpy(dtype=np.float64)
    weights = frame[weight_col].to_numpy(dtype=np.float64)
    return {
        "probability_nonfinite_count": int((~np.isfinite(probabilities)).sum()),
        "probability_negative_count": int((probabilities < 0).sum()),
        "weight_nonfinite_count": int((~np.isfinite(weights)).sum()),
        "weight_negative_count": int((weights < 0).sum()),
    }


def construct_sparse_matrix(
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    values: np.ndarray,
    node_count: int,
    dtype: str = "float32",
) -> sp.csr_matrix:
    assert_no_duplicate_coordinates(source_idx.copy(), target_idx, node_count)
    return sp.coo_matrix(
        (values.astype(dtype, copy=False), (source_idx, target_idx)),
        shape=(int(node_count), int(node_count)),
    ).tocsr()


def stored_diagonal_coordinate_count(matrix: sp.spmatrix, indices: np.ndarray) -> int:
    csr = matrix.tocsr()
    diagonal_indices = set(indices.astype(np.int64, copy=False).tolist())
    conflicts = 0
    for row in diagonal_indices:
        start = int(csr.indptr[row])
        stop = int(csr.indptr[row + 1])
        if row in csr.indices[start:stop]:
            conflicts += 1
    return conflicts


def add_final_self_loops(
    p_forward: sp.csr_matrix,
    final_node_indices: np.ndarray,
    weight: float = 1.0,
    dtype: str = "float32",
) -> sp.csr_matrix:
    final_overlap = stored_diagonal_coordinate_count(p_forward, final_node_indices)
    if final_overlap:
        raise ValueError(f"Final-time self-loop conflict count: {final_overlap}")
    loops = sp.coo_matrix(
        (
            np.full(len(final_node_indices), float(weight), dtype=dtype),
            (final_node_indices, final_node_indices),
        ),
        shape=p_forward.shape,
    ).tocsr()
    return (p_forward + loops).tocsr()


def row_sum_qc(
    p_forward: sp.csr_matrix,
    p_absorbing: sp.csr_matrix,
    final_mask: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    forward_sums = np.asarray(p_forward.sum(axis=1)).ravel()
    absorbing_sums = np.asarray(p_absorbing.sum(axis=1)).ravel()
    non_final_mask = ~final_mask
    nonfinal_error = np.abs(forward_sums[non_final_mask] - 1.0)
    final_abs = np.abs(forward_sums[final_mask])
    absorbing_error = np.abs(absorbing_sums - 1.0)
    qc = {
        "forward_nonfinal_row_sum_max_error": float(nonfinal_error.max()) if len(nonfinal_error) else 0.0,
        "forward_final_row_sum_max_abs": float(final_abs.max()) if len(final_abs) else 0.0,
        "absorbing_row_sum_max_error": float(absorbing_error.max()) if len(absorbing_error) else 0.0,
        "forward_nonfinal_rows_exceeding_tolerance": int((nonfinal_error > tolerance).sum()),
        "forward_final_rows_exceeding_tolerance": int((final_abs > tolerance).sum()),
        "absorbing_rows_exceeding_tolerance": int((absorbing_error > tolerance).sum()),
    }
    if qc["forward_nonfinal_rows_exceeding_tolerance"]:
        raise ValueError("Forward non-final row sums exceed tolerance.")
    if qc["forward_final_rows_exceeding_tolerance"]:
        raise ValueError("Forward final-time rows are not zero.")
    if qc["absorbing_rows_exceeding_tolerance"]:
        raise ValueError("Absorbing row sums exceed tolerance.")
    return qc


def sparse_entry_qc(matrix: sp.spmatrix, name: str) -> dict[str, Any]:
    data = matrix.data
    nonfinite_count = int((~np.isfinite(data)).sum())
    negative_count = int((data < 0).sum())
    qc = {
        f"{name}_nonfinite_count": nonfinite_count,
        f"{name}_negative_count": negative_count,
        f"{name}_data_min": float(data.min()) if len(data) else 0.0,
        f"{name}_data_max": float(data.max()) if len(data) else 0.0,
    }
    if nonfinite_count or negative_count:
        raise ValueError(f"{name} has invalid matrix entries: {qc}")
    return qc


def sparse_basic_stats(matrix: sp.spmatrix, name: str) -> dict[str, Any]:
    data = matrix.data
    return {
        "matrix": name,
        "shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
        "nnz": int(matrix.nnz),
        "dtype": str(data.dtype),
        "data_sum": float(data.sum()) if len(data) else 0.0,
        "data_min": float(data.min()) if len(data) else 0.0,
        "data_max": float(data.max()) if len(data) else 0.0,
        "nonfinite_count": int((~np.isfinite(data)).sum()),
        "negative_count": int((data < 0).sum()),
    }


def row_activity_summary(p_forward: sp.csr_matrix, final_mask: np.ndarray) -> dict[str, int]:
    row_nnz = np.diff(p_forward.indptr)
    outgoing_mask = row_nnz > 0
    return {
        "source_rows": int(outgoing_mask.sum()),
        "final_time_forward_zero_outgoing_rows": int((~outgoing_mask & final_mask).sum()),
        "final_time_rows_with_outgoing_edges": int((outgoing_mask & final_mask).sum()),
        "non_final_zero_outgoing_rows": int((~outgoing_mask & ~final_mask).sum()),
    }


def validate_expected_full_counts(
    config: dict[str, Any],
    p_forward: sp.csr_matrix,
    p_absorbing: sp.csr_matrix,
    node_index: NodeIndex,
) -> dict[str, int | str]:
    expected = config["expected"]
    activity = row_activity_summary(p_forward, node_index.final_mask)
    final_indices = node_index.node_table.loc[
        node_index.final_mask, "global_node_index"
    ].to_numpy(dtype=np.int64)
    final_self_loop_count = stored_diagonal_coordinate_count(p_absorbing, final_indices)
    checks = {
        "matrix_shape": f"{p_forward.shape[0]}x{p_forward.shape[1]}",
        "forward_nnz": int(p_forward.nnz),
        "absorbing_nnz": int(p_absorbing.nnz),
        "final_time_self_loop_count": int(final_self_loop_count),
        **activity,
    }
    expected_shape = f"{expected['nodes']}x{expected['nodes']}"
    failures = []
    if checks["matrix_shape"] != expected_shape:
        failures.append("matrix_shape")
    if checks["forward_nnz"] != int(expected["retained_v2_edges"]):
        failures.append("forward_nnz")
    if checks["absorbing_nnz"] != int(expected["absorbing_nnz"]):
        failures.append("absorbing_nnz")
    if checks["final_time_self_loop_count"] != int(expected["final_time_nodes"]):
        failures.append("final_time_self_loop_count")
    if checks["source_rows"] != int(expected["source_rows"]):
        failures.append("source_rows")
    if checks["final_time_forward_zero_outgoing_rows"] != int(expected["final_time_nodes"]):
        failures.append("final_time_forward_zero_outgoing_rows")
    if checks["non_final_zero_outgoing_rows"] != 0:
        failures.append("non_final_zero_outgoing_rows")
    if failures:
        raise ValueError(f"M4A-v2 full production count checks failed: {failures}")
    return checks


def planned_production_outputs(paths: dict[str, Path]) -> dict[str, Path]:
    transition_dir = paths["transition_objects_dir"]
    node_dir = paths["node_table_dir"]
    reports_dir = paths["reports_dir"]
    return {
        "p_forward": transition_dir / "P_forward_no_terminal_selfloops_v2.npz",
        "p_absorbing": transition_dir / "P_absorbing_terminal_selfloops_v2.npz",
        "w_v2": transition_dir / "W_v2_unnormalized_weight.npz",
        "node_table": node_dir / "global_node_table.parquet",
        "assembly_report": reports_dir / "m4a_v2_02_full_assembly_report.md",
        "qc_summary": reports_dir / "m4a_v2_02_qc_summary.csv",
        "output_inventory": reports_dir / "m4a_v2_02_output_inventory.csv",
        "matrix_comparison": reports_dir / "m4a_v2_02_v1_v2_matrix_comparison.csv",
        "next_step": reports_dir / "m4a_v2_02_next_step_recommendation.md",
        "completed_manifest": reports_dir / "m4a_v2_02_completed_manifest.csv",
        "failed_manifest": reports_dir / "m4a_v2_02_failed_manifest.csv",
    }


def validate_no_existing_production_outputs(paths: dict[str, Path], overwrite: bool, resume: bool) -> None:
    existing = [path for path in planned_production_outputs(paths).values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Existing M4A-v2 production outputs found; pass explicit --overwrite: "
            + ", ".join(str(path) for path in existing[:5])
        )


def schema_validation_for_shards(plan: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    required = {
        config["assembly"]["source_id_column"],
        config["assembly"]["target_id_column"],
        config["assembly"]["probability_column"],
        config["assembly"]["weight_column"],
    }
    for row in plan.to_dict("records"):
        path = resolved(row["m3_v2_output_parquet"])
        pf = pq.ParquetFile(path)
        columns = set(pf.schema_arrow.names)
        missing = sorted(required - columns)
        optional_present = [column for column in OPTIONAL_EDGE_COLUMNS if column in columns]
        rows.append(
            {
                "shard_id": row["shard_id"],
                "time_pair": row["time_pair"],
                "path": str(path),
                "metadata_rows": int(pf.metadata.num_rows),
                "qc_retained_v2_edges": int(row["retained_v2_edges"]),
                "required_columns_present": not missing,
                "missing_required_columns": ";".join(missing),
                "optional_metadata_columns_present": ";".join(optional_present),
                "status": "PASS" if not missing else "FAIL",
            }
        )
        if missing:
            raise KeyError(f"M3-v2 shard {path} missing required columns: {missing}")
        if int(pf.metadata.num_rows) != int(row["retained_v2_edges"]):
            raise ValueError(f"Parquet row count mismatch for {path}")
    return pd.DataFrame(rows)


def global_duplicate_count(keys_parts: list[np.ndarray]) -> int:
    if not keys_parts:
        return 0
    keys = np.concatenate(keys_parts)
    if len(keys) <= 1:
        return 0
    keys.sort()
    return int((keys[1:] == keys[:-1]).sum())


def stream_preflight_edges(
    plan: pd.DataFrame,
    node_index: NodeIndex,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assembly = config["assembly"]
    validation = config["validation"]
    source_col = assembly["source_id_column"]
    target_col = assembly["target_id_column"]
    prob_col = assembly["probability_column"]
    weight_col = assembly["weight_column"]
    batch_rows = int(validation.get("batch_rows", 500_000))
    node_count = int(len(node_index.node_table))
    row_sums = np.zeros(node_count, dtype=np.float64)
    out_counts = np.zeros(node_count, dtype=np.int32)
    global_key_parts: list[np.ndarray] = []
    mapping_rows: list[dict[str, Any]] = []
    total_rows = 0
    total_source_missing = 0
    total_target_missing = 0
    total_probability_nonfinite = 0
    total_probability_negative = 0
    total_weight_nonfinite = 0
    total_weight_negative = 0

    for shard in plan.to_dict("records"):
        path = resolved(shard["m3_v2_output_parquet"])
        columns = [source_col, target_col, prob_col, weight_col]
        shard_rows = 0
        shard_source_parts: list[np.ndarray] = []
        shard_key_parts: list[np.ndarray] = []
        shard_counts = {
            "probability_nonfinite_count": 0,
            "probability_negative_count": 0,
            "weight_nonfinite_count": 0,
            "weight_negative_count": 0,
        }
        shard_source_missing = 0
        shard_target_missing = 0
        for batch in iter_parquet_batches(path, columns, batch_rows):
            frame = batch.to_pandas()
            counts = validate_edge_values(frame, prob_col, weight_col)
            for key, value in counts.items():
                shard_counts[key] += int(value)
            source_idx, target_idx, source_missing, target_missing = map_edge_batch(
                frame,
                node_index,
                source_col,
                target_col,
            )
            shard_source_missing += source_missing
            shard_target_missing += target_missing
            if source_missing or target_missing:
                continue
            probabilities = frame[prob_col].to_numpy(dtype=np.float64)
            np.add.at(row_sums, source_idx, probabilities)
            np.add.at(out_counts, source_idx, 1)
            keys = coordinate_keys(source_idx, target_idx, node_count)
            shard_key_parts.append(keys)
            if bool(validation.get("full_duplicate_check", True)):
                global_key_parts.append(keys)
            shard_source_parts.append(source_idx)
            shard_rows += int(len(frame))
        total_rows += shard_rows
        total_source_missing += shard_source_missing
        total_target_missing += shard_target_missing
        total_probability_nonfinite += shard_counts["probability_nonfinite_count"]
        total_probability_negative += shard_counts["probability_negative_count"]
        total_weight_nonfinite += shard_counts["weight_nonfinite_count"]
        total_weight_negative += shard_counts["weight_negative_count"]
        shard_duplicate_count = global_duplicate_count(shard_key_parts)
        shard_sources = int(len(np.unique(np.concatenate(shard_source_parts)))) if shard_source_parts else 0
        row = {
            "shard_id": shard["shard_id"],
            "time_pair": shard["time_pair"],
            "path": str(path),
            "rows_checked": shard_rows,
            "qc_retained_v2_edges": int(shard["retained_v2_edges"]),
            "source_rows_observed": shard_sources,
            "qc_source_count": int(shard["source_count"]),
            "source_anchor_missing_from_node_table": shard_source_missing,
            "target_anchor_missing_from_node_table": shard_target_missing,
            "duplicate_source_target_pairs": shard_duplicate_count,
            **shard_counts,
            "status": "PASS",
        }
        failure_fields = [
            "source_anchor_missing_from_node_table",
            "target_anchor_missing_from_node_table",
            "duplicate_source_target_pairs",
            "probability_nonfinite_count",
            "probability_negative_count",
            "weight_nonfinite_count",
            "weight_negative_count",
        ]
        if any(int(row[field]) for field in failure_fields) or shard_rows != int(shard["retained_v2_edges"]):
            row["status"] = "FAIL"
        mapping_rows.append(row)

    global_duplicates = global_duplicate_count(global_key_parts) if bool(validation.get("full_duplicate_check", True)) else -1
    observed_mask = out_counts > 0
    observed_errors = np.abs(row_sums[observed_mask] - 1.0)
    final_mask = node_index.final_mask
    non_final_mask = ~final_mask
    stats = {
        "selected_edges_checked": int(total_rows),
        "selected_source_rows_observed": int(observed_mask.sum()),
        "selected_forward_row_sum_max_error": float(observed_errors.max()) if len(observed_errors) else 0.0,
        "selected_forward_rows_exceeding_tolerance": int((observed_errors > float(validation["row_sum_tolerance"])).sum()),
        "selected_non_final_zero_outgoing_rows": int(((out_counts == 0) & non_final_mask).sum()),
        "selected_final_forward_zero_outgoing_rows": int(((out_counts == 0) & final_mask).sum()),
        "selected_final_rows_with_outgoing_edges": int(((out_counts > 0) & final_mask).sum()),
        "global_duplicate_source_target_pairs": int(global_duplicates),
        "source_anchor_missing_from_node_table": int(total_source_missing),
        "target_anchor_missing_from_node_table": int(total_target_missing),
        "probability_nonfinite_count": int(total_probability_nonfinite),
        "probability_negative_count": int(total_probability_negative),
        "weight_nonfinite_count": int(total_weight_nonfinite),
        "weight_negative_count": int(total_weight_negative),
    }
    fail_counts = [
        stats["source_anchor_missing_from_node_table"],
        stats["target_anchor_missing_from_node_table"],
        stats["probability_nonfinite_count"],
        stats["probability_negative_count"],
        stats["weight_nonfinite_count"],
        stats["weight_negative_count"],
        stats["selected_forward_rows_exceeding_tolerance"],
        stats["selected_final_rows_with_outgoing_edges"],
    ]
    if stats["global_duplicate_source_target_pairs"] > 0:
        fail_counts.append(stats["global_duplicate_source_target_pairs"])
    if any(fail_counts):
        raise ValueError(f"M4A-v2 preflight edge validation failed: {stats}")
    return pd.DataFrame(mapping_rows), stats


def build_dryrun_plan(
    full_plan: pd.DataFrame,
    selected_plan: pd.DataFrame,
    node_index: NodeIndex,
    config: dict[str, Any],
    stream_stats: dict[str, Any],
) -> pd.DataFrame:
    expected = config["expected"]
    full_selected = len(selected_plan) == len(full_plan)
    planned = {
        "planned_matrix_shape": f"{expected['nodes']}x{expected['nodes']}",
        "planned_forward_nnz": int(expected["retained_v2_edges"]),
        "planned_absorbing_nnz": int(expected["absorbing_nnz"]),
        "planned_final_time_self_loops": int(expected["final_time_nodes"]),
        "planned_non_final_zero_outgoing_rows": 0,
        "planned_final_time_forward_zero_outgoing_rows": int(expected["final_time_nodes"]),
        "planned_source_rows": int(expected["source_rows"]),
        "planned_probability_column": config["assembly"]["probability_column"],
        "planned_weight_column": config["assembly"]["weight_column"],
        "selected_shards": int(len(selected_plan)),
        "selected_edges_checked": int(stream_stats["selected_edges_checked"]),
        "selected_source_rows_observed": int(stream_stats["selected_source_rows_observed"]),
        "full_preflight": bool(full_selected),
    }
    return pd.DataFrame([{"field": key, "value": value} for key, value in planned.items()])


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    work = frame[columns].copy() if columns else frame.copy()
    for col in work.columns:
        work[col] = work[col].map(lambda value: str(value).replace("\n", " "))
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for row in work.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def preflight_report(summary: dict[str, Any], schema: pd.DataFrame, mapping: pd.DataFrame) -> str:
    schema_counts = schema["status"].value_counts().to_dict()
    mapping_counts = mapping["status"].value_counts().to_dict()
    lines = [
        "# M4A-v2-01 Dry-Run/Preflight Report",
        "",
        "## Execution Boundary",
        "- dry_run: true",
        "- M4A-v2 production matrix assembly: not run",
        "- M4C-v2, GPCCA/K_gpcca, M4D, barcode, M5, and BranchSBM: not run",
        "- production sparse .npz outputs written: 0",
        "",
        "## Planned Assembly",
        f"- matrix shape: {summary['planned_matrix_shape']}",
        f"- planned forward nnz: {summary['expected_forward_nnz']}",
        f"- planned absorbing nnz: {summary['expected_absorbing_nnz']}",
        f"- planned final-time self-loops: {summary['expected_final_time_self_loops']}",
        f"- planned source rows: {summary['expected_source_rows']}",
        f"- probability column: `{summary['probability_column']}`",
        f"- weight column: `{summary['weight_column']}`",
        "",
        "## Dry-Run Scope",
        f"- selected shards: {summary['selected_shards']}",
        f"- full shards available: {summary['full_shards']}",
        f"- selected edges checked: {summary['selected_edges_checked']}",
        f"- selected source rows observed: {summary['selected_source_rows_observed']}",
        f"- full preflight: {summary['full_preflight']}",
        "",
        "## Validation Results",
        f"- schema validation status counts: {schema_counts}",
        f"- node mapping validation status counts: {mapping_counts}",
        f"- source mapping missing count: {summary['source_anchor_missing_from_node_table']}",
        f"- target mapping missing count: {summary['target_anchor_missing_from_node_table']}",
        f"- duplicate source-target coordinates: {summary['global_duplicate_source_target_pairs']}",
        f"- selected row-sum max error: {summary['selected_forward_row_sum_max_error']}",
        f"- upstream metadata diff count: {summary['upstream_metadata_diff_count']}",
        f"- forbidden downstream metadata diff count: {summary['forbidden_downstream_metadata_diff_count']}",
        "",
        "## Time-Pair Counts",
        markdown_table(pd.DataFrame(summary["time_pair_counts"])),
    ]
    return "\n".join(lines).rstrip() + "\n"


def count_npz(output_root: Path) -> int:
    return len(list(output_root.glob("**/*.npz"))) if output_root.exists() else 0


def write_dryrun_outputs(
    paths: dict[str, Path],
    dryrun_plan: pd.DataFrame,
    summary: dict[str, Any],
    schema_validation: pd.DataFrame,
    node_mapping_validation: pd.DataFrame,
) -> None:
    reports_dir = paths["reports_dir"]
    atomic_write_csv(reports_dir / "m4a_v2_01_dryrun_assembly_plan.csv", dryrun_plan)
    atomic_write_csv(reports_dir / "m4a_v2_01_input_schema_validation.csv", schema_validation)
    atomic_write_csv(reports_dir / "m4a_v2_01_node_mapping_validation.csv", node_mapping_validation)
    atomic_write_json(reports_dir / "m4a_v2_01_dryrun_summary.json", summary)
    atomic_write_text(
        reports_dir / "m4a_v2_01_preflight_report.md",
        preflight_report(summary, schema_validation, node_mapping_validation),
    )


def validate_dryrun_outputs(paths: dict[str, Path]) -> None:
    reports_dir = paths["reports_dir"]
    required = [reports_dir / name for name in DRYRUN_OUTPUT_NAMES]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty dry-run outputs: {missing}")
    json.loads((reports_dir / "m4a_v2_01_dryrun_summary.json").read_text(encoding="utf-8"))


def watched_roots(config: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    protected = [resolved(path) for path in config.get("protected_roots", [])]
    downstream = [resolved(path) for path in config.get("forbidden_downstream_roots", [])]
    return protected, downstream


def run_dryrun(config: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    protected, downstream = watched_roots(config)
    npz_before = count_npz(paths["output_root"])
    before_protected = snapshot(protected)
    before_downstream = snapshot(downstream)
    benchmark = validate_benchmark_summary(paths)
    full_plan = load_m3_v2_qc(config, paths)
    selected_plan = select_shards(full_plan, args.max_shards)
    node_index = load_node_index(config, paths)
    schema_validation = schema_validation_for_shards(selected_plan, config)
    node_mapping_validation, stream_stats = stream_preflight_edges(selected_plan, node_index, config)
    full_preflight = len(selected_plan) == len(full_plan)
    expected = config["expected"]
    if full_preflight:
        if stream_stats["selected_edges_checked"] != int(expected["retained_v2_edges"]):
            raise ValueError("Full preflight edge count does not match expected retained v2 edges.")
        if stream_stats["selected_source_rows_observed"] != int(expected["source_rows"]):
            raise ValueError("Full preflight source row count does not match expected source rows.")
        if stream_stats["selected_non_final_zero_outgoing_rows"] != 0:
            raise ValueError("Full preflight found non-final zero-outgoing rows.")
    after_protected = snapshot(protected)
    after_downstream = snapshot(downstream)
    upstream_diffs = diff_snapshot(before_protected, after_protected)
    downstream_diffs = diff_snapshot(before_downstream, after_downstream)
    npz_after = count_npz(paths["output_root"])
    if npz_after != npz_before or npz_after != 0:
        raise RuntimeError(f"Dry-run sparse .npz count changed or is nonzero: before={npz_before}, after={npz_after}")
    time_pair_counts = (
        full_plan.groupby("time_pair", observed=True)
        .agg(shards=("shard_id", "size"), sources=("source_count", "sum"), retained_edges=("retained_v2_edges", "sum"))
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "stage": "M4A-v2-01",
        "status": "PASSED" if not upstream_diffs and not downstream_diffs else "FAILED",
        "dry_run": True,
        "resume": bool(args.resume),
        "stop_on_error": bool(args.stop_on_error),
        "overwrite": bool(args.overwrite),
        "full_preflight": bool(full_preflight),
        "selected_shards": int(len(selected_plan)),
        "full_shards": int(len(full_plan)),
        "expected_nodes": int(expected["nodes"]),
        "expected_source_rows": int(expected["source_rows"]),
        "expected_forward_nnz": int(expected["retained_v2_edges"]),
        "expected_final_time_self_loops": int(expected["final_time_nodes"]),
        "expected_absorbing_nnz": int(expected["absorbing_nnz"]),
        "planned_matrix_shape": f"{expected['nodes']}x{expected['nodes']}",
        "planned_non_final_zero_outgoing_rows": 0,
        "planned_final_time_forward_zero_outgoing_rows": int(expected["final_time_nodes"]),
        "probability_column": config["assembly"]["probability_column"],
        "weight_column": config["assembly"]["weight_column"],
        "benchmark_decision_category": benchmark.get("decision_category"),
        "node_count": int(len(node_index.node_table)),
        "final_time_node_count": int(node_index.final_mask.sum()),
        "time_pair_counts": time_pair_counts,
        "production_npz_count_before": int(npz_before),
        "production_npz_count_after": int(npz_after),
        "upstream_metadata_diff_count": len(upstream_diffs),
        "forbidden_downstream_metadata_diff_count": len(downstream_diffs),
        "upstream_metadata_diffs": upstream_diffs,
        "forbidden_downstream_metadata_diffs": downstream_diffs,
        "runtime_seconds": float(time.monotonic() - started),
        "generated_at_utc": utc_now(),
        **stream_stats,
    }
    dryrun_plan = build_dryrun_plan(full_plan, selected_plan, node_index, config, stream_stats)
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    write_dryrun_outputs(paths, dryrun_plan, summary, schema_validation, node_mapping_validation)
    validate_dryrun_outputs(paths)
    if upstream_diffs or downstream_diffs:
        raise RuntimeError("Protected upstream or forbidden downstream metadata changed during dry-run.")
    return summary


def stream_production_arrays(
    plan: pd.DataFrame,
    node_index: NodeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    assembly = config["assembly"]
    batch_rows = int(config["validation"].get("batch_rows", 500_000))
    total_edges = int(plan["retained_v2_edges"].sum())
    dtype = np.dtype(str(assembly["dtype"]))
    source_idx = np.empty(total_edges, dtype=np.int64)
    target_idx = np.empty(total_edges, dtype=np.int64)
    probabilities = np.empty(total_edges, dtype=dtype)
    weights = np.empty(total_edges, dtype=dtype)
    cursor = 0
    value_counts = {
        "source_anchor_missing_from_node_table": 0,
        "target_anchor_missing_from_node_table": 0,
        "probability_nonfinite_count": 0,
        "probability_negative_count": 0,
        "weight_nonfinite_count": 0,
        "weight_negative_count": 0,
    }
    for shard in plan.to_dict("records"):
        path = resolved(shard["m3_v2_output_parquet"])
        columns = [
            assembly["source_id_column"],
            assembly["target_id_column"],
            assembly["probability_column"],
            assembly["weight_column"],
        ]
        for batch in iter_parquet_batches(path, columns, batch_rows):
            frame = batch.to_pandas()
            counts = validate_edge_values(
                frame,
                assembly["probability_column"],
                assembly["weight_column"],
            )
            for key, value in counts.items():
                value_counts[key] += int(value)
            src, tgt, src_missing, tgt_missing = map_edge_batch(
                frame,
                node_index,
                assembly["source_id_column"],
                assembly["target_id_column"],
            )
            value_counts["source_anchor_missing_from_node_table"] += src_missing
            value_counts["target_anchor_missing_from_node_table"] += tgt_missing
            if src_missing or tgt_missing:
                raise ValueError(f"Missing node mapping during production: source={src_missing}, target={tgt_missing}")
            stop = cursor + len(frame)
            source_idx[cursor:stop] = src
            target_idx[cursor:stop] = tgt
            probabilities[cursor:stop] = frame[assembly["probability_column"]].to_numpy(dtype=dtype, copy=False)
            weights[cursor:stop] = frame[assembly["weight_column"]].to_numpy(dtype=dtype, copy=False)
            cursor = stop
    if cursor != total_edges:
        raise ValueError(f"Production stream read {cursor} edges, expected {total_edges}")
    invalid_counts = [
        value_counts["source_anchor_missing_from_node_table"],
        value_counts["target_anchor_missing_from_node_table"],
        value_counts["probability_nonfinite_count"],
        value_counts["probability_negative_count"],
        value_counts["weight_nonfinite_count"],
        value_counts["weight_negative_count"],
    ]
    if any(invalid_counts):
        raise ValueError(f"M4A-v2 production value validation failed: {value_counts}")
    return {
        "source_idx": source_idx,
        "target_idx": target_idx,
        "probabilities": probabilities,
        "weights": weights,
        "value_qc": value_counts,
    }


def assemble_matrices_from_arrays(
    arrays: dict[str, Any],
    node_index: NodeIndex,
    config: dict[str, Any],
) -> dict[str, sp.csr_matrix]:
    node_count = int(len(node_index.node_table))
    dtype = str(config["assembly"]["dtype"])
    p_forward = construct_sparse_matrix(
        arrays["source_idx"],
        arrays["target_idx"],
        arrays["probabilities"],
        node_count,
        dtype,
    )
    w_v2 = construct_sparse_matrix(
        arrays["source_idx"],
        arrays["target_idx"],
        arrays["weights"],
        node_count,
        dtype,
    )
    final_indices = node_index.node_table.loc[
        node_index.final_mask, "global_node_index"
    ].to_numpy(dtype=np.int64)
    p_absorbing = add_final_self_loops(
        p_forward,
        final_indices,
        float(config["assembly"]["final_time_self_loop_weight"]),
        dtype,
    )
    return {"p_forward": p_forward, "p_absorbing": p_absorbing, "w_v2": w_v2}


def smoke_matrix_qc(
    arrays: dict[str, Any],
    matrices: dict[str, sp.csr_matrix],
    node_index: NodeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    tolerance = float(config["validation"]["row_sum_tolerance"])
    p_forward = matrices["p_forward"]
    p_absorbing = matrices["p_absorbing"]
    w_v2 = matrices["w_v2"]
    final_indices = node_index.node_table.loc[
        node_index.final_mask, "global_node_index"
    ].to_numpy(dtype=np.int64)
    source_rows = np.unique(arrays["source_idx"])
    forward_sums = np.asarray(p_forward.sum(axis=1)).ravel()
    absorbing_sums = np.asarray(p_absorbing.sum(axis=1)).ravel()
    source_errors = np.abs(forward_sums[source_rows] - 1.0)
    final_forward_abs = np.abs(forward_sums[final_indices])
    final_absorbing_errors = np.abs(absorbing_sums[final_indices] - 1.0)
    final_self_loop_count = stored_diagonal_coordinate_count(p_absorbing, final_indices)
    qc = {
        "smoke_source_rows": int(len(source_rows)),
        "smoke_forward_nnz": int(p_forward.nnz),
        "smoke_absorbing_nnz": int(p_absorbing.nnz),
        "smoke_w_v2_nnz": int(w_v2.nnz),
        "smoke_forward_source_row_sum_max_error": float(source_errors.max()) if len(source_errors) else 0.0,
        "smoke_forward_source_rows_exceeding_tolerance": int((source_errors > tolerance).sum()),
        "smoke_forward_final_row_sum_max_abs": float(final_forward_abs.max()) if len(final_forward_abs) else 0.0,
        "smoke_final_rows_with_outgoing_edges": int((final_forward_abs > tolerance).sum()),
        "smoke_absorbing_final_row_sum_max_error": float(final_absorbing_errors.max()) if len(final_absorbing_errors) else 0.0,
        "smoke_absorbing_final_rows_exceeding_tolerance": int((final_absorbing_errors > tolerance).sum()),
        "smoke_final_time_self_loop_count": int(final_self_loop_count),
        "global_duplicate_source_target_pairs": 0,
        **arrays["value_qc"],
        **sparse_entry_qc(p_forward, "p_forward"),
        **sparse_entry_qc(p_absorbing, "p_absorbing"),
        **sparse_entry_qc(w_v2, "w_v2"),
    }
    failures = []
    if qc["smoke_forward_source_rows_exceeding_tolerance"]:
        failures.append("smoke_forward_source_rows_exceeding_tolerance")
    if qc["smoke_final_rows_with_outgoing_edges"]:
        failures.append("smoke_final_rows_with_outgoing_edges")
    if qc["smoke_absorbing_final_rows_exceeding_tolerance"]:
        failures.append("smoke_absorbing_final_rows_exceeding_tolerance")
    if qc["smoke_final_time_self_loop_count"] != int(node_index.final_mask.sum()):
        failures.append("smoke_final_time_self_loop_count")
    if failures:
        raise ValueError(f"M4A-v2 smoke matrix QC failed: {failures}")
    return qc


def output_inventory(outputs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, path in outputs.items():
        rows.append(
            {
                "output_name": name,
                "path": str(path),
                "exists": bool(path.exists()),
                "bytes": int(path.stat().st_size) if path.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def v1_transition_paths(paths: dict[str, Path]) -> dict[str, Path]:
    root = paths["m4a_v1_node_table"].parents[1]
    transition_dir = root / "transition_objects"
    return {
        "p_forward": transition_dir / "P_forward_no_terminal_selfloops.npz",
        "p_absorbing": transition_dir / "P_absorbing_terminal_selfloops.npz",
        "w_raw": transition_dir / "W_raw_edge_weight.npz",
        "w_mass": transition_dir / "W_mass_adjusted_weight.npz",
    }


def sparse_stats_from_path(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "matrix": name,
            "shape": "",
            "nnz": -1,
            "dtype": "",
            "data_sum": np.nan,
            "data_min": np.nan,
            "data_max": np.nan,
            "nonfinite_count": -1,
            "negative_count": -1,
        }
    return sparse_basic_stats(sp.load_npz(path), name)


def build_v1_v2_matrix_comparison(
    paths: dict[str, Path],
    matrices: dict[str, sp.csr_matrix],
) -> pd.DataFrame:
    v1_paths = v1_transition_paths(paths)
    comparisons = [
        ("P_forward", "p_forward", "p_forward"),
        ("P_absorbing", "p_absorbing", "p_absorbing"),
        ("W_v2_vs_W_raw", "w_raw", "w_v2"),
        ("W_v2_vs_W_mass", "w_mass", "w_v2"),
    ]
    rows = []
    for comparison, v1_key, v2_key in comparisons:
        v1_stats = sparse_stats_from_path(v1_paths[v1_key], f"v1_{v1_key}")
        v2_stats = sparse_basic_stats(matrices[v2_key], f"v2_{v2_key}")
        rows.append(
            {
                "comparison": comparison,
                "v1_path": str(v1_paths[v1_key]),
                "v1_shape": v1_stats["shape"],
                "v2_shape": v2_stats["shape"],
                "v1_nnz": int(v1_stats["nnz"]),
                "v2_nnz": int(v2_stats["nnz"]),
                "nnz_delta_v2_minus_v1": int(v2_stats["nnz"]) - int(v1_stats["nnz"]),
                "v1_data_sum": v1_stats["data_sum"],
                "v2_data_sum": v2_stats["data_sum"],
                "data_sum_delta_v2_minus_v1": v2_stats["data_sum"] - v1_stats["data_sum"],
                "v1_nonfinite_count": int(v1_stats["nonfinite_count"]),
                "v2_nonfinite_count": int(v2_stats["nonfinite_count"]),
                "v1_negative_count": int(v1_stats["negative_count"]),
                "v2_negative_count": int(v2_stats["negative_count"]),
                "status": "PASS" if int(v2_stats["nonfinite_count"]) == 0 else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def smoke_report(summary: dict[str, Any]) -> str:
    lines = [
        "# M4A-v2-02 Smoke Validation Report",
        "",
        "## Status",
        f"- status: {summary['status']}",
        f"- execution_mode: {summary['execution_mode']}",
        f"- selected_shards: {summary['selected_shards']}",
        f"- selected_edges: {summary['selected_edges']}",
        "",
        "## Matrix QC",
        f"- smoke_forward_nnz: {summary['smoke_forward_nnz']}",
        f"- smoke_absorbing_nnz: {summary['smoke_absorbing_nnz']}",
        f"- smoke_source_rows: {summary['smoke_source_rows']}",
        f"- source row-sum max error: {summary['smoke_forward_source_row_sum_max_error']}",
        f"- final self-loop count: {summary['smoke_final_time_self_loop_count']}",
        f"- source mapping missing count: {summary['source_anchor_missing_from_node_table']}",
        f"- target mapping missing count: {summary['target_anchor_missing_from_node_table']}",
        f"- duplicate source-target coordinates: {summary['global_duplicate_source_target_pairs']}",
        "",
        "## Safety Checks",
        f"- canonical production outputs created: {summary['canonical_outputs_created_count']}",
        f"- upstream metadata diff count: {summary['upstream_metadata_diff_count']}",
        f"- forbidden downstream metadata diff count: {summary['forbidden_downstream_metadata_diff_count']}",
        f"- /ssd output count: {summary['ssd_output_count']}",
    ]
    return "\n".join(lines).rstrip() + "\n"


def full_assembly_report(summary: dict[str, Any]) -> str:
    lines = [
        "# M4A-v2-02 Full Assembly Report",
        "",
        "## Status",
        f"- status: {summary['status']}",
        f"- execution_mode: {summary['execution_mode']}",
        f"- matrix_shape: {summary['matrix_shape']}",
        "",
        "## Matrix Counts",
        f"- forward_nnz: {summary['forward_nnz']}",
        f"- absorbing_nnz: {summary['absorbing_nnz']}",
        f"- W_v2_nnz: {summary['w_v2_nnz']}",
        f"- D35 self-loop count: {summary['final_time_self_loop_count']}",
        f"- source rows: {summary['source_rows']}",
        f"- final-time forward zero-outgoing rows: {summary['final_time_forward_zero_outgoing_rows']}",
        f"- non-final zero-outgoing rows: {summary['non_final_zero_outgoing_rows']}",
        "",
        "## QC",
        f"- forward non-final row-sum max error: {summary['forward_nonfinal_row_sum_max_error']}",
        f"- forward final row-sum max abs: {summary['forward_final_row_sum_max_abs']}",
        f"- absorbing row-sum max error: {summary['absorbing_row_sum_max_error']}",
        f"- probability nonfinite count: {summary['probability_nonfinite_count']}",
        f"- probability negative count: {summary['probability_negative_count']}",
        f"- weight nonfinite count: {summary['weight_nonfinite_count']}",
        f"- weight negative count: {summary['weight_negative_count']}",
        f"- source mapping missing count: {summary['source_anchor_missing_from_node_table']}",
        f"- target mapping missing count: {summary['target_anchor_missing_from_node_table']}",
        f"- duplicate source-target coordinates: {summary['global_duplicate_source_target_pairs']}",
        "",
        "## Safety Checks",
        f"- upstream metadata diff count: {summary['upstream_metadata_diff_count']}",
        f"- forbidden downstream metadata diff count: {summary['forbidden_downstream_metadata_diff_count']}",
        f"- /ssd output count: {summary['ssd_output_count']}",
        "",
        "## Next Step",
        f"- {NEXT_STEP_TEXT}",
    ]
    return "\n".join(lines).rstrip() + "\n"


def count_ssd_outputs(output_root: Path) -> int:
    if not output_root.exists():
        return 0
    return sum(1 for path in output_root.rglob("*") if str(path).startswith("/ssd/"))


def write_completed_manifests(outputs: dict[str, Path], summary: dict[str, Any]) -> None:
    completed = pd.DataFrame(
        [
            {"step": "stream_m3_v2_edges", "status": "COMPLETED", "rows": summary["forward_nnz"]},
            {"step": "assemble_sparse_matrices", "status": "COMPLETED", "rows": summary["forward_nnz"]},
            {"step": "validate_matrix_qc", "status": "COMPLETED", "rows": summary["source_rows"]},
            {"step": "write_m4a_v2_outputs", "status": "COMPLETED", "rows": len(outputs)},
        ]
    )
    failed = pd.DataFrame(columns=["step", "status", "error"])
    atomic_write_csv(outputs["completed_manifest"], completed)
    atomic_write_csv(outputs["failed_manifest"], failed)


def run_smoke_test(config: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    protected, downstream = watched_roots(config)
    before_protected = snapshot(protected)
    before_downstream = snapshot(downstream)
    canonical_before = {
        name: path.exists() for name, path in planned_production_outputs(paths).items()
    }
    benchmark = validate_benchmark_summary(paths)
    full_plan = load_m3_v2_qc(config, paths)
    selected_plan = select_shards(full_plan, args.max_shards)
    node_index = load_node_index(config, paths)
    schema_validation_for_shards(selected_plan, config)
    stream_preflight_edges(selected_plan, node_index, config)
    arrays = stream_production_arrays(selected_plan, node_index, config)
    matrices = assemble_matrices_from_arrays(arrays, node_index, config)
    qc = smoke_matrix_qc(arrays, matrices, node_index, config)
    after_protected = snapshot(protected)
    after_downstream = snapshot(downstream)
    upstream_diffs = diff_snapshot(before_protected, after_protected)
    downstream_diffs = diff_snapshot(before_downstream, after_downstream)
    canonical_after = {
        name: path.exists() for name, path in planned_production_outputs(paths).items()
    }
    created_canonical = [
        name for name, exists in canonical_after.items() if exists and not canonical_before[name]
    ]
    summary = {
        "stage": "M4A-v2-02",
        "status": "PASSED",
        "execution_mode": "production_smoke",
        "benchmark_decision_category": benchmark.get("decision_category"),
        "selected_shards": int(len(selected_plan)),
        "selected_edges": int(selected_plan["retained_v2_edges"].sum()),
        "canonical_outputs_created_count": len(created_canonical),
        "canonical_outputs_created": created_canonical,
        "upstream_metadata_diff_count": len(upstream_diffs),
        "forbidden_downstream_metadata_diff_count": len(downstream_diffs),
        "upstream_metadata_diffs": upstream_diffs,
        "forbidden_downstream_metadata_diffs": downstream_diffs,
        "ssd_output_count": count_ssd_outputs(paths["output_root"]),
        "runtime_seconds": float(time.monotonic() - started),
        "generated_at_utc": utc_now(),
        **qc,
    }
    failed = []
    if created_canonical:
        failed.append("canonical_outputs_created")
    if upstream_diffs:
        failed.append("upstream_metadata_diff")
    if downstream_diffs:
        failed.append("forbidden_downstream_metadata_diff")
    if summary["ssd_output_count"]:
        failed.append("ssd_outputs")
    if failed:
        summary["status"] = "FAILED"
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    atomic_write_text(paths["reports_dir"] / SMOKE_REPORT_NAME, smoke_report(summary))
    if failed:
        raise RuntimeError(f"M4A-v2 smoke validation failed: {failed}")
    return summary


def run_production(config: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    if is_smoke_mode(args):
        return run_smoke_test(config, paths, args)
    validate_no_existing_production_outputs(paths, args.overwrite, args.resume)
    started = time.monotonic()
    protected, downstream = watched_roots(config)
    before_protected = snapshot(protected)
    before_downstream = snapshot(downstream)
    benchmark = validate_benchmark_summary(paths)
    plan = load_m3_v2_qc(config, paths)
    node_index = load_node_index(config, paths)
    arrays = stream_production_arrays(plan, node_index, config)
    matrices = assemble_matrices_from_arrays(arrays, node_index, config)
    p_forward = matrices["p_forward"]
    p_absorbing = matrices["p_absorbing"]
    w_v2 = matrices["w_v2"]
    qc = row_sum_qc(
        p_forward,
        p_absorbing,
        node_index.final_mask,
        float(config["validation"]["row_sum_tolerance"]),
    )
    expected_counts = validate_expected_full_counts(config, p_forward, p_absorbing, node_index)
    entry_qc = {
        **sparse_entry_qc(p_forward, "p_forward"),
        **sparse_entry_qc(p_absorbing, "p_absorbing"),
        **sparse_entry_qc(w_v2, "w_v2"),
    }
    duplicate_count = duplicate_coordinate_count(
        arrays["source_idx"].copy(),
        arrays["target_idx"],
        int(len(node_index.node_table)),
    )
    if duplicate_count:
        raise ValueError(f"Duplicate source-target matrix coordinates detected: {duplicate_count}")
    outputs = planned_production_outputs(paths)
    atomic_save_npz(outputs["p_forward"], p_forward)
    atomic_save_npz(outputs["p_absorbing"], p_absorbing)
    atomic_save_npz(outputs["w_v2"], w_v2)
    atomic_write_parquet(outputs["node_table"], node_index.node_table)
    comparison = build_v1_v2_matrix_comparison(paths, matrices)
    after_protected = snapshot(protected)
    after_downstream = snapshot(downstream)
    upstream_diffs = diff_snapshot(before_protected, after_protected)
    downstream_diffs = diff_snapshot(before_downstream, after_downstream)
    summary = {
        "stage": "M4A-v2-02",
        "status": "COMPLETED",
        "execution_mode": "full_production",
        "benchmark_decision_category": benchmark.get("decision_category"),
        "selected_shards": int(len(plan)),
        "matrix_shape": expected_counts["matrix_shape"],
        "forward_nnz": int(p_forward.nnz),
        "absorbing_nnz": int(p_absorbing.nnz),
        "w_v2_nnz": int(w_v2.nnz),
        "global_duplicate_source_target_pairs": int(duplicate_count),
        "upstream_metadata_diff_count": len(upstream_diffs),
        "forbidden_downstream_metadata_diff_count": len(downstream_diffs),
        "upstream_metadata_diffs": upstream_diffs,
        "forbidden_downstream_metadata_diffs": downstream_diffs,
        "ssd_output_count": count_ssd_outputs(paths["output_root"]),
        "runtime_seconds": float(time.monotonic() - started),
        "generated_at_utc": utc_now(),
        **arrays["value_qc"],
        **qc,
        **expected_counts,
        **entry_qc,
    }
    failed = []
    if upstream_diffs:
        failed.append("upstream_metadata_diff")
    if downstream_diffs:
        failed.append("forbidden_downstream_metadata_diff")
    if summary["ssd_output_count"]:
        failed.append("ssd_outputs")
    if failed:
        summary["status"] = "FAILED"
    atomic_write_csv(outputs["qc_summary"], pd.DataFrame([summary]))
    atomic_write_csv(outputs["matrix_comparison"], comparison)
    atomic_write_text(outputs["assembly_report"], full_assembly_report(summary))
    atomic_write_text(outputs["next_step"], NEXT_STEP_TEXT + "\n")
    write_completed_manifests(outputs, summary)
    inventory = output_inventory(outputs)
    atomic_write_csv(outputs["output_inventory"], inventory)
    atomic_write_csv(outputs["output_inventory"], output_inventory(outputs))
    if failed:
        raise RuntimeError(f"M4A-v2 full production validation failed: {failed}")
    return {"status": "COMPLETED", "outputs": outputs, "row_sum_qc": qc, "summary": summary}


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_config(args.config)
    paths = validate_config(config)
    if args.dry_run:
        return run_dryrun(config, paths, args)
    return run_production(config, paths, args)


def main() -> None:
    print(json.dumps(json_safe(run(parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
