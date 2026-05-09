#!/usr/bin/env python
"""Full M4A-v2 QC and M4A-v1 vs M4A-v2 matrix benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ROOT = Path("/home/zhutao/scratch/nichefate")
DEFAULT_OUTPUT_ROOT = ROOT / "m4a_v2_benchmark"
M4A_V1_ROOT = ROOT / "m4a"
M4A_V2_ROOT = ROOT / "m4a_v2"
M3_V2_REPORTS = ROOT / "m3_v2" / "reports"
M3_V2_BENCHMARK = ROOT / "m3_v2_benchmark"
M4E_ROOT = ROOT / "m4e"
M4C_ROOT = ROOT / "m4c"

EXPECTED_NODES = 1_439_542
EXPECTED_FINAL_NODES = 90_960
EXPECTED_SOURCE_ROWS = 1_348_582
EXPECTED_V2_FORWARD_NNZ = 13_485_820
EXPECTED_V2_ABSORBING_NNZ = 13_576_780
ROW_ATOL = 1e-5
TIME_ORDER = ["D0", "D3", "D9", "D21", "D35"]
EXPECTED_TIME_PAIRS = ["D0_to_D3", "D3_to_D9", "D9_to_D21", "D21_to_D35"]

PROTECTED_ROOTS = [
    ROOT / "m3",
    ROOT / "m3_v2",
    ROOT / "m4a",
    ROOT / "m4b",
    ROOT / "m4c",
    ROOT / "m4a_v2",
]
FORBIDDEN_DOWNSTREAM_ROOTS = [
    ROOT / "m4c_v2",
    ROOT / "m4a_v2" / "gpcca",
    ROOT / "m4a_v2" / "pygpcca",
    ROOT / "m4a_v2" / "k_gpcca",
    ROOT / "m4a_v2" / "barcode",
    ROOT / "m4a_v2" / "m5",
    ROOT / "m4a_v2" / "branchsbm",
    ROOT / "m3_v2" / "gpcca",
    ROOT / "m3_v2" / "pygpcca",
    ROOT / "m3_v2" / "k_gpcca",
    ROOT / "m3_v2" / "barcode",
    ROOT / "m3_v2" / "m5",
    ROOT / "m3_v2" / "branchsbm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
    return is_relative_to(left_resolved, right_resolved) or is_relative_to(
        right_resolved, left_resolved
    )


def reject_ssd(path: Path) -> None:
    path = resolved(path)
    if path == Path("/ssd") or Path("/ssd") in path.parents:
        raise ValueError(f"Refusing /ssd output path: {path}")


def validate_output_root(output_root: Path) -> None:
    output_root = resolved(output_root)
    reject_ssd(output_root)
    for protected in [*PROTECTED_ROOTS, *FORBIDDEN_DOWNSTREAM_ROOTS]:
        if paths_overlap(output_root, protected):
            raise ValueError(f"Output root overlaps protected root {protected}: {output_root}")


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    validate_output_root(output_root)
    paths = {
        "root": resolved(output_root),
        "reports": resolved(output_root) / "reports",
        "figures": resolved(output_root) / "reports" / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


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
    atomic_write_text(path, json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


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


def m4a_v1_matrix_paths() -> dict[str, Path]:
    return {
        "P_forward_no_terminal_selfloops": M4A_V1_ROOT / "transition_objects" / "P_forward_no_terminal_selfloops.npz",
        "P_absorbing_terminal_selfloops": M4A_V1_ROOT / "transition_objects" / "P_absorbing_terminal_selfloops.npz",
        "W_raw_edge_weight": M4A_V1_ROOT / "transition_objects" / "W_raw_edge_weight.npz",
        "W_mass_adjusted_weight": M4A_V1_ROOT / "transition_objects" / "W_mass_adjusted_weight.npz",
    }


def m4a_v2_matrix_paths() -> dict[str, Path]:
    return {
        "P_forward_no_terminal_selfloops_v2": M4A_V2_ROOT / "transition_objects" / "P_forward_no_terminal_selfloops_v2.npz",
        "P_absorbing_terminal_selfloops_v2": M4A_V2_ROOT / "transition_objects" / "P_absorbing_terminal_selfloops_v2.npz",
        "W_v2_unnormalized_weight": M4A_V2_ROOT / "transition_objects" / "W_v2_unnormalized_weight.npz",
    }


def required_input_paths() -> dict[str, Path]:
    paths = {
        **{f"v1_{key}": value for key, value in m4a_v1_matrix_paths().items()},
        **{f"v2_{key}": value for key, value in m4a_v2_matrix_paths().items()},
        "m4a_v1_node_table": M4A_V1_ROOT / "node_table" / "global_node_table.parquet",
        "m4a_v2_node_table": M4A_V2_ROOT / "node_table" / "global_node_table.parquet",
        "m4a_v2_qc_summary": M4A_V2_ROOT / "reports" / "m4a_v2_02_qc_summary.csv",
        "m3_v2_qc_summary": M3_V2_REPORTS / "m3_v2_full_qc_summary.csv",
        "m3_v2_benchmark_summary": M3_V2_BENCHMARK / "m3_v1_vs_v2_edge_benchmark_summary.json",
    }
    return paths


def validate_required_inputs() -> pd.DataFrame:
    rows = []
    for name, path in required_input_paths().items():
        rows.append(
            {
                "input_name": name,
                "path": str(path),
                "exists": bool(path.is_file()),
                "bytes": int(path.stat().st_size) if path.is_file() else 0,
                "status": "PASS" if path.is_file() and path.stat().st_size > 0 else "FAIL",
            }
        )
    frame = pd.DataFrame(rows)
    failed = frame.loc[frame["status"] != "PASS", "input_name"].tolist()
    if failed:
        raise FileNotFoundError(f"Missing or empty M4A benchmark inputs: {failed}")
    return frame


def load_csr(path: Path) -> sp.csr_matrix:
    matrix = sp.load_npz(path).tocsr()
    matrix.sort_indices()
    return matrix


def load_node_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def validate_node_table(node_table: pd.DataFrame, label: str) -> None:
    required = {"global_node_index", "anchor_id", "time", "time_day", "is_final_time"}
    missing = sorted(required - set(node_table.columns))
    if missing:
        raise KeyError(f"{label} node table missing columns: {missing}")
    indices = node_table["global_node_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(node_table), dtype=np.int64)):
        raise ValueError(f"{label} node table global_node_index is not contiguous row order.")
    if not bool(node_table["anchor_id"].is_unique):
        raise ValueError(f"{label} node table anchor_id is not unique.")


def row_activity(matrix: sp.csr_matrix, final_mask: np.ndarray) -> dict[str, int]:
    row_nnz = np.diff(matrix.indptr)
    outgoing = row_nnz > 0
    return {
        "source_row_count": int(outgoing.sum()),
        "zero_outgoing_rows": int((~outgoing).sum()),
        "final_time_zero_outgoing_rows": int((~outgoing & final_mask).sum()),
        "non_final_zero_outgoing_rows": int((~outgoing & ~final_mask).sum()),
        "final_time_rows_with_outgoing_edges": int((outgoing & final_mask).sum()),
    }


def row_sum_qc(matrix: sp.csr_matrix, final_mask: np.ndarray, matrix_role: str) -> dict[str, Any]:
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    if matrix_role == "forward":
        non_final_error = np.abs(sums[~final_mask] - 1.0)
        final_abs = np.abs(sums[final_mask])
        return {
            "row_sum_qc_scope": "non_final_rows_equal_1_final_rows_equal_0",
            "row_sum_max_error": float(max(non_final_error.max(), final_abs.max())),
            "forward_nonfinal_row_sum_max_error": float(non_final_error.max()),
            "forward_final_row_sum_max_abs": float(final_abs.max()),
            "rows_exceeding_tolerance": int((non_final_error > ROW_ATOL).sum() + (final_abs > ROW_ATOL).sum()),
        }
    if matrix_role == "absorbing":
        error = np.abs(sums - 1.0)
        return {
            "row_sum_qc_scope": "all_rows_equal_1",
            "row_sum_max_error": float(error.max()),
            "forward_nonfinal_row_sum_max_error": np.nan,
            "forward_final_row_sum_max_abs": np.nan,
            "rows_exceeding_tolerance": int((error > ROW_ATOL).sum()),
        }
    return {
        "row_sum_qc_scope": "not_applicable_weight_matrix",
        "row_sum_max_error": np.nan,
        "forward_nonfinal_row_sum_max_error": np.nan,
        "forward_final_row_sum_max_abs": np.nan,
        "rows_exceeding_tolerance": 0,
    }


def matrix_distribution(matrix: sp.csr_matrix) -> dict[str, Any]:
    data = matrix.data
    row_nnz = np.diff(matrix.indptr)
    nonzero_rows = row_nnz > 0
    starts = matrix.indptr[:-1][nonzero_rows]
    top1 = np.maximum.reduceat(data, starts) if len(starts) else np.array([], dtype=data.dtype)
    positive = data > 0
    entropy_terms = np.zeros_like(data, dtype=np.float64)
    entropy_terms[positive] = data[positive].astype(np.float64) * np.log(data[positive].astype(np.float64))
    entropy = -np.add.reduceat(entropy_terms, starts) if len(starts) else np.array([], dtype=np.float64)
    return {
        "value_min": float(data.min()) if len(data) else 0.0,
        "value_max": float(data.max()) if len(data) else 0.0,
        "value_mean": float(data.mean()) if len(data) else 0.0,
        "value_median": float(np.median(data)) if len(data) else 0.0,
        "top1_per_row_mean": float(top1.mean()) if len(top1) else 0.0,
        "top1_per_row_median": float(np.median(top1)) if len(top1) else 0.0,
        "transition_entropy_mean": float(entropy.mean()) if len(entropy) else 0.0,
        "transition_entropy_median": float(np.median(entropy)) if len(entropy) else 0.0,
        "nonfinite_count": int((~np.isfinite(data)).sum()),
        "negative_count": int((data < 0).sum()),
    }


def matrix_object_summary_row(
    version: str,
    matrix_name: str,
    path: Path,
    matrix: sp.csr_matrix,
    final_mask: np.ndarray,
    matrix_role: str,
) -> dict[str, Any]:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{version} {matrix_name} is not square: {matrix.shape}")
    activity = row_activity(matrix, final_mask)
    dist = matrix_distribution(matrix)
    qc = row_sum_qc(matrix, final_mask, matrix_role)
    n_rows = int(matrix.shape[0])
    return {
        "version": version,
        "matrix_name": matrix_name,
        "path": str(path),
        "shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
        "node_count": n_rows,
        "nnz": int(matrix.nnz),
        "sparsity": 1.0 - (float(matrix.nnz) / float(n_rows * n_rows)),
        "density": float(matrix.nnz) / float(n_rows * n_rows),
        "disk_bytes": int(path.stat().st_size),
        "memory_bytes_estimate": int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes),
        "has_canonical_format": bool(matrix.has_canonical_format),
        "duplicate_matrix_coordinates": 0 if matrix.has_canonical_format else -1,
        **activity,
        **qc,
        **dist,
    }


def final_self_loop_count(matrix: sp.csr_matrix, final_mask: np.ndarray) -> int:
    diagonal = matrix.diagonal()
    return int((diagonal[final_mask] != 0).sum())


def validate_m4a_v2_full_qc(
    v2_forward: sp.csr_matrix,
    v2_absorbing: sp.csr_matrix,
    v2_weight: sp.csr_matrix,
    v2_node: pd.DataFrame,
    v2_summary_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    final_mask = v2_node["is_final_time"].to_numpy(dtype=bool)
    forward_row = v2_summary_rows.query("version == 'M4A-v2' and matrix_name == 'P_forward_no_terminal_selfloops_v2'").iloc[0]
    absorbing_row = v2_summary_rows.query("version == 'M4A-v2' and matrix_name == 'P_absorbing_terminal_selfloops_v2'").iloc[0]
    weight_row = v2_summary_rows.query("version == 'M4A-v2' and matrix_name == 'W_v2_unnormalized_weight'").iloc[0]
    qc_path = M4A_V2_ROOT / "reports" / "m4a_v2_02_qc_summary.csv"
    assembly_qc = pd.read_csv(qc_path).iloc[0]
    checks = {
        "required_matrices_nonempty": all(path.stat().st_size > 0 for path in m4a_v2_matrix_paths().values()),
        "matrix_shape_expected": tuple(v2_forward.shape) == (EXPECTED_NODES, EXPECTED_NODES),
        "forward_nnz_expected": int(v2_forward.nnz) == EXPECTED_V2_FORWARD_NNZ,
        "absorbing_nnz_expected": int(v2_absorbing.nnz) == EXPECTED_V2_ABSORBING_NNZ,
        "d35_self_loop_count_expected": final_self_loop_count(v2_absorbing, final_mask) == EXPECTED_FINAL_NODES,
        "source_rows_expected": int(forward_row["source_row_count"]) == EXPECTED_SOURCE_ROWS,
        "final_forward_zero_rows_expected": int(forward_row["final_time_zero_outgoing_rows"]) == EXPECTED_FINAL_NODES,
        "nonfinal_zero_rows_expected": int(forward_row["non_final_zero_outgoing_rows"]) == 0,
        "forward_row_sum_pass": float(forward_row["row_sum_max_error"]) <= ROW_ATOL,
        "absorbing_row_sum_pass": float(absorbing_row["row_sum_max_error"]) <= ROW_ATOL,
        "no_invalid_entries": all(
            int(row["nonfinite_count"]) == 0 and int(row["negative_count"]) == 0
            for _, row in pd.DataFrame([forward_row, absorbing_row, weight_row]).iterrows()
        ),
        "duplicate_coordinates_zero": int(forward_row["duplicate_matrix_coordinates"]) == 0
        and int(absorbing_row["duplicate_matrix_coordinates"]) == 0
        and int(weight_row["duplicate_matrix_coordinates"]) == 0,
        "source_target_mapping_complete": int(assembly_qc["source_anchor_missing_from_node_table"]) == 0
        and int(assembly_qc["target_anchor_missing_from_node_table"]) == 0,
        "assembly_report_passed": str(assembly_qc["status"]) == "COMPLETED",
    }
    summary = {
        "matrix_shape": f"{v2_forward.shape[0]}x{v2_forward.shape[1]}",
        "forward_nnz": int(v2_forward.nnz),
        "absorbing_nnz": int(v2_absorbing.nnz),
        "d35_self_loop_count": final_self_loop_count(v2_absorbing, final_mask),
        "source_rows": int(forward_row["source_row_count"]),
        "final_time_forward_zero_outgoing_rows": int(forward_row["final_time_zero_outgoing_rows"]),
        "non_final_zero_outgoing_rows": int(forward_row["non_final_zero_outgoing_rows"]),
        "forward_row_sum_max_error": float(forward_row["row_sum_max_error"]),
        "absorbing_row_sum_max_error": float(absorbing_row["row_sum_max_error"]),
        "nonfinite_entry_count": int(forward_row["nonfinite_count"] + absorbing_row["nonfinite_count"] + weight_row["nonfinite_count"]),
        "negative_entry_count": int(forward_row["negative_count"] + absorbing_row["negative_count"] + weight_row["negative_count"]),
        "duplicate_matrix_coordinates": int(forward_row["duplicate_matrix_coordinates"] + absorbing_row["duplicate_matrix_coordinates"] + weight_row["duplicate_matrix_coordinates"]),
        "source_anchor_missing_from_node_table": int(assembly_qc["source_anchor_missing_from_node_table"]),
        "target_anchor_missing_from_node_table": int(assembly_qc["target_anchor_missing_from_node_table"]),
    }
    rows = [{"check": key, "status": "PASS" if value else "FAIL"} for key, value in checks.items()]
    frame = pd.DataFrame(rows)
    full_pass = bool((frame["status"] == "PASS").all())
    summary["full_qc_status"] = "PASS" if full_pass else "FAIL"
    return frame.assign(**summary), summary


def build_matrix_object_summary(
    v1_node: pd.DataFrame,
    v2_node: pd.DataFrame,
    loaded: dict[str, sp.csr_matrix],
) -> pd.DataFrame:
    v1_final = v1_node["is_final_time"].to_numpy(dtype=bool)
    v2_final = v2_node["is_final_time"].to_numpy(dtype=bool)
    rows = [
        matrix_object_summary_row(
            "M4A-v1",
            "P_forward_no_terminal_selfloops",
            m4a_v1_matrix_paths()["P_forward_no_terminal_selfloops"],
            loaded["v1_forward"],
            v1_final,
            "forward",
        ),
        matrix_object_summary_row(
            "M4A-v1",
            "P_absorbing_terminal_selfloops",
            m4a_v1_matrix_paths()["P_absorbing_terminal_selfloops"],
            loaded["v1_absorbing"],
            v1_final,
            "absorbing",
        ),
        matrix_object_summary_row(
            "M4A-v2",
            "P_forward_no_terminal_selfloops_v2",
            m4a_v2_matrix_paths()["P_forward_no_terminal_selfloops_v2"],
            loaded["v2_forward"],
            v2_final,
            "forward",
        ),
        matrix_object_summary_row(
            "M4A-v2",
            "P_absorbing_terminal_selfloops_v2",
            m4a_v2_matrix_paths()["P_absorbing_terminal_selfloops_v2"],
            loaded["v2_absorbing"],
            v2_final,
            "absorbing",
        ),
        matrix_object_summary_row(
            "M4A-v2",
            "W_v2_unnormalized_weight",
            m4a_v2_matrix_paths()["W_v2_unnormalized_weight"],
            loaded["v2_weight"],
            v2_final,
            "weight",
        ),
    ]
    for matrix_name in ["W_raw_edge_weight", "W_mass_adjusted_weight"]:
        matrix = load_csr(m4a_v1_matrix_paths()[matrix_name])
        rows.append(
            matrix_object_summary_row(
                "M4A-v1",
                matrix_name,
                m4a_v1_matrix_paths()[matrix_name],
                matrix,
                v1_final,
                "weight",
            )
        )
    return pd.DataFrame(rows)


def get_summary_row(summary: pd.DataFrame, version: str, matrix_name: str) -> pd.Series:
    match = summary[(summary["version"] == version) & (summary["matrix_name"] == matrix_name)]
    if len(match) != 1:
        raise ValueError(f"Expected one summary row for {version} {matrix_name}; found {len(match)}")
    return match.iloc[0]


def build_global_comparison(summary: pd.DataFrame, v1_absorbing: sp.csr_matrix, v2_absorbing: sp.csr_matrix, final_mask: np.ndarray) -> pd.DataFrame:
    v1_forward = get_summary_row(summary, "M4A-v1", "P_forward_no_terminal_selfloops")
    v2_forward = get_summary_row(summary, "M4A-v2", "P_forward_no_terminal_selfloops_v2")
    v1_abs = get_summary_row(summary, "M4A-v1", "P_absorbing_terminal_selfloops")
    v2_abs = get_summary_row(summary, "M4A-v2", "P_absorbing_terminal_selfloops_v2")
    rows = [
        {
            "comparison_scope": "global",
            "node_count_v1": int(v1_forward["node_count"]),
            "node_count_v2": int(v2_forward["node_count"]),
            "node_count_delta_v2_minus_v1": int(v2_forward["node_count"] - v1_forward["node_count"]),
            "source_rows_v1": int(v1_forward["source_row_count"]),
            "source_rows_v2": int(v2_forward["source_row_count"]),
            "source_rows_delta_v2_minus_v1": int(v2_forward["source_row_count"] - v1_forward["source_row_count"]),
            "forward_nnz_v1": int(v1_forward["nnz"]),
            "forward_nnz_v2": int(v2_forward["nnz"]),
            "forward_nnz_delta_v2_minus_v1": int(v2_forward["nnz"] - v1_forward["nnz"]),
            "forward_density_v1": float(v1_forward["density"]),
            "forward_density_v2": float(v2_forward["density"]),
            "forward_density_delta_v2_minus_v1": float(v2_forward["density"] - v1_forward["density"]),
            "forward_row_sum_max_error_v1": float(v1_forward["row_sum_max_error"]),
            "forward_row_sum_max_error_v2": float(v2_forward["row_sum_max_error"]),
            "absorbing_row_sum_max_error_v1": float(v1_abs["row_sum_max_error"]),
            "absorbing_row_sum_max_error_v2": float(v2_abs["row_sum_max_error"]),
            "non_final_zero_outgoing_rows_v1": int(v1_forward["non_final_zero_outgoing_rows"]),
            "non_final_zero_outgoing_rows_v2": int(v2_forward["non_final_zero_outgoing_rows"]),
            "final_time_zero_outgoing_rows_v1": int(v1_forward["final_time_zero_outgoing_rows"]),
            "final_time_zero_outgoing_rows_v2": int(v2_forward["final_time_zero_outgoing_rows"]),
            "absorbing_self_loop_count_v1": final_self_loop_count(v1_absorbing, final_mask),
            "absorbing_self_loop_count_v2": final_self_loop_count(v2_absorbing, final_mask),
            "top1_mean_v1": float(v1_forward["top1_per_row_mean"]),
            "top1_mean_v2": float(v2_forward["top1_per_row_mean"]),
            "top1_delta_v2_minus_v1": float(v2_forward["top1_per_row_mean"] - v1_forward["top1_per_row_mean"]),
            "entropy_mean_v1": float(v1_forward["transition_entropy_mean"]),
            "entropy_mean_v2": float(v2_forward["transition_entropy_mean"]),
            "entropy_delta_v2_minus_v1": float(v2_forward["transition_entropy_mean"] - v1_forward["transition_entropy_mean"]),
            "v2_is_sparser": bool(int(v2_forward["nnz"]) < int(v1_forward["nnz"])),
            "source_coverage_preserved": bool(int(v2_forward["source_row_count"]) == int(v1_forward["source_row_count"])),
            "v2_row_stochastic_valid": bool(float(v2_forward["row_sum_max_error"]) <= ROW_ATOL and float(v2_abs["row_sum_max_error"]) <= ROW_ATOL),
        }
    ]
    return pd.DataFrame(rows)


def time_codes(node_table: pd.DataFrame) -> np.ndarray:
    labels = node_table["time"].astype(str).to_numpy()
    mapping = {label: index for index, label in enumerate(TIME_ORDER)}
    missing = sorted(set(labels) - set(mapping))
    if missing:
        raise ValueError(f"Unexpected time labels in node table: {missing}")
    return np.asarray([mapping[label] for label in labels], dtype=np.int16)


def time_pair_coverage(matrix: sp.csr_matrix, node_table: pd.DataFrame, version: str, chunk_rows: int = 100_000) -> pd.DataFrame:
    codes = time_codes(node_table)
    pair_count = len(TIME_ORDER) * len(TIME_ORDER)
    edge_counts = np.zeros(pair_count, dtype=np.int64)
    prob_mass = np.zeros(pair_count, dtype=np.float64)
    source_seen = np.zeros((pair_count, matrix.shape[0]), dtype=bool)
    target_seen = np.zeros((pair_count, matrix.shape[0]), dtype=bool)
    for row_start in range(0, matrix.shape[0], chunk_rows):
        row_stop = min(row_start + chunk_rows, matrix.shape[0])
        row_lengths = np.diff(matrix.indptr[row_start : row_stop + 1])
        total_edges = int(row_lengths.sum())
        if total_edges == 0:
            continue
        rows = np.repeat(np.arange(row_start, row_stop, dtype=np.int64), row_lengths)
        edge_start = int(matrix.indptr[row_start])
        edge_stop = int(matrix.indptr[row_stop])
        cols = matrix.indices[edge_start:edge_stop].astype(np.int64, copy=False)
        data = matrix.data[edge_start:edge_stop].astype(np.float64, copy=False)
        flat = codes[rows] * len(TIME_ORDER) + codes[cols]
        edge_counts += np.bincount(flat, minlength=pair_count).astype(np.int64)
        prob_mass += np.bincount(flat, weights=data, minlength=pair_count)
        source_seen[flat, rows] = True
        target_seen[flat, cols] = True
    rows = []
    for source_index, source_label in enumerate(TIME_ORDER):
        for target_index, target_label in enumerate(TIME_ORDER):
            flat = source_index * len(TIME_ORDER) + target_index
            if edge_counts[flat] == 0 and f"{source_label}_to_{target_label}" not in EXPECTED_TIME_PAIRS:
                continue
            source_count = int(source_seen[flat].sum())
            target_count = int(target_seen[flat].sum())
            rows.append(
                {
                    "version": version,
                    "time_pair": f"{source_label}_to_{target_label}",
                    "source_time": source_label,
                    "target_time": target_label,
                    "source_rows": source_count,
                    "target_rows": target_count,
                    "edge_count": int(edge_counts[flat]),
                    "probability_mass": float(prob_mass[flat]),
                    "edge_density_within_observed_rows": float(edge_counts[flat]) / float(source_count * target_count)
                    if source_count and target_count
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_by_time_pair_comparison(v1_coverage: pd.DataFrame, v2_coverage: pd.DataFrame) -> pd.DataFrame:
    merged = v1_coverage.merge(v2_coverage, on=["time_pair", "source_time", "target_time"], suffixes=("_v1", "_v2"))
    for column in ["source_rows", "target_rows", "edge_count", "probability_mass", "edge_density_within_observed_rows"]:
        merged[f"{column}_delta_v2_minus_v1"] = merged[f"{column}_v2"] - merged[f"{column}_v1"]
    merged["source_coverage_preserved"] = merged["source_rows_v2"] == merged["source_rows_v1"]
    merged["v2_sparser"] = merged["edge_count_v2"] < merged["edge_count_v1"]
    return merged


def build_readiness_checklist(v1_node: pd.DataFrame, v2_node: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(item: str, ok: bool, details: str) -> None:
        rows.append({"item": item, "status": "PASS" if ok else "FAIL", "details": details})

    add("P_absorbing_terminal_selfloops_v2_exists", m4a_v2_matrix_paths()["P_absorbing_terminal_selfloops_v2"].is_file(), str(m4a_v2_matrix_paths()["P_absorbing_terminal_selfloops_v2"]))
    add("P_forward_no_terminal_selfloops_v2_exists", m4a_v2_matrix_paths()["P_forward_no_terminal_selfloops_v2"].is_file(), str(m4a_v2_matrix_paths()["P_forward_no_terminal_selfloops_v2"]))
    add("versioned_global_node_table_exists", (M4A_V2_ROOT / "node_table" / "global_node_table.parquet").is_file(), str(M4A_V2_ROOT / "node_table" / "global_node_table.parquet"))
    node_order_ok = len(v1_node) == len(v2_node) and bool(v1_node["anchor_id"].equals(v2_node["anchor_id"]))
    add("node_ordering_compatible_with_m4a_v1", node_order_ok, "anchor_id order matches M4A-v1 node table")
    m4c_path = M4C_ROOT / "fate_probabilities" / "fate_probability_node_summary.parquet"
    if m4c_path.is_file():
        m4c_node = pd.read_parquet(m4c_path, columns=["global_node_index", "anchor_id"])
        m4c_ok = len(m4c_node) == len(v2_node) and bool(m4c_node["anchor_id"].equals(v2_node["anchor_id"]))
        add("node_ordering_compatible_with_m4c_v1", m4c_ok, "anchor_id order matches M4C-v1 node summary")
    else:
        add("node_ordering_compatible_with_m4c_v1", False, f"missing {m4c_path}")
    endpoint_path = M4E_ROOT / "endpoint_annotation" / "endpoint_node_annotation.parquet"
    if endpoint_path.is_file():
        endpoint = pd.read_parquet(endpoint_path, columns=["global_node_index", "anchor_id"])
        final_ids = set(v2_node.loc[v2_node["is_final_time"], "global_node_index"].astype(int))
        endpoint_ids = set(endpoint["global_node_index"].astype(int))
        mapped = len(final_ids & endpoint_ids)
        add("d35_endpoint_rows_mapped_to_m4e", mapped == EXPECTED_FINAL_NODES, f"mapped {mapped} of {EXPECTED_FINAL_NODES} final nodes")
    else:
        add("d35_endpoint_rows_mapped_to_m4e", False, f"missing {endpoint_path}")
    refined_path = M4E_ROOT / "endpoint_refinement" / "refined_endpoint_mapping.csv"
    refined_ok = refined_path.is_file() and pd.read_csv(refined_path, nrows=1).shape[0] > 0
    add("m4c_v1_endpoint_taxonomy_reusable", refined_ok, str(refined_path))
    required_direct = {"global_node_index", "anchor_id", "time", "time_day", "is_final_time"}
    add("required_m4a_v2_node_columns_present", required_direct <= set(v2_node.columns), ",".join(sorted(required_direct)))
    neighborhood_path = M4E_ROOT / "neighborhood_annotation" / "node_neighborhood_annotation.parquet"
    if neighborhood_path.is_file():
        columns = set(pd.read_parquet(neighborhood_path, columns=["global_node_index", "anchor_id"]).columns)
        add("supplemental_biological_metadata_join_available", {"global_node_index", "anchor_id"} <= columns, str(neighborhood_path))
    else:
        add("supplemental_biological_metadata_join_available", False, f"missing {neighborhood_path}")
    return pd.DataFrame(rows)


def build_biological_sanity(v1_node: pd.DataFrame, v2_node: pd.DataFrame, by_time: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check: str, ok: bool, details: str) -> None:
        rows.append({"check": check, "status": "PASS" if ok else "WARN", "details": details})

    time_counts = v1_node["time"].astype(str).value_counts().to_dict()
    add("expected_time_layers_present", all(time in time_counts and time_counts[time] > 0 for time in TIME_ORDER), json.dumps(time_counts, sort_keys=True))
    expected_pairs = by_time[by_time["time_pair"].isin(EXPECTED_TIME_PAIRS)]
    add("v2_source_target_rows_cover_expected_time_pairs", bool((expected_pairs["source_rows_v2"] > 0).all() and (expected_pairs["target_rows_v2"] > 0).all()), "all expected transition layers have observed source and target rows")
    final_ready = readiness.query("item == 'd35_endpoint_rows_mapped_to_m4e'")["status"].iloc[0] == "PASS"
    add("d35_terminal_rows_endpoint_annotatable", final_ready, "M4E endpoint annotation covers D35 final rows")
    add("no_nonfinal_zero_outgoing_rows", int(by_time["source_rows_v2"].sum()) >= EXPECTED_SOURCE_ROWS, "source rows preserved across expected time pairs")
    neighborhood_path = M4E_ROOT / "neighborhood_annotation" / "node_neighborhood_annotation.parquet"
    add("slice_mouse_time_layer_metadata_available", neighborhood_path.is_file(), str(neighborhood_path))
    return pd.DataFrame(rows)


def choose_decision(full_qc: dict[str, Any], global_comparison: pd.DataFrame, readiness: pd.DataFrame) -> tuple[str, list[str]]:
    reasons = []
    qc_pass = full_qc["full_qc_status"] == "PASS"
    row = global_comparison.iloc[0]
    source_preserved = bool(row["source_coverage_preserved"])
    no_nonfinal_zero = int(row["non_final_zero_outgoing_rows_v2"]) == 0
    readiness_pass = bool((readiness["status"] == "PASS").all())
    if qc_pass and source_preserved and no_nonfinal_zero and readiness_pass:
        return "proceed_to_m4c_v2_planning", ["M4A-v2 QC, source coverage, structural rows, and readiness checks passed."]
    if not qc_pass:
        reasons.append("M4A-v2 matrix QC failed.")
        return "rerun_m4a_v2_assembly", reasons
    if not source_preserved or not no_nonfinal_zero:
        reasons.append("M4A-v2 introduced structural source-row artifacts.")
        return "revise_m4a_v2_assembly", reasons
    if not readiness_pass:
        reasons.append("M4C-v2 readiness checklist has failures.")
        return "revise_m4a_v2_assembly", reasons
    return "defer_m4a_v2_until_k_gpcca_or_barcode", ["No primary failure matched; defer for separate branch review."]


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    work = frame.head(max_rows).copy()
    for column in work.columns:
        work[column] = work[column].map(lambda value: str(value).replace("\n", " "))
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for row in work.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    if len(frame) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(frame)} rows.")
    return "\n".join(lines)


def write_report(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", body.rstrip(), ""])
    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")


def save_bar_plot(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.3), 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_figures(paths: dict[str, Path], summary: pd.DataFrame, global_comparison: pd.DataFrame, by_time: pd.DataFrame) -> list[Path]:
    figures = paths["figures"]
    row = global_comparison.iloc[0]
    outputs = []
    outputs.append(figures / "m4a_v1_vs_v2_nnz_sparsity.png")
    save_bar_plot(outputs[-1], ["v1 forward nnz", "v2 forward nnz"], [row["forward_nnz_v1"], row["forward_nnz_v2"]], "M4A-v1 vs M4A-v2 Forward NNZ", "nnz")
    outputs.append(figures / "m4a_v1_vs_v2_row_sum_error.png")
    save_bar_plot(outputs[-1], ["v1 forward", "v2 forward", "v1 absorbing", "v2 absorbing"], [row["forward_row_sum_max_error_v1"], row["forward_row_sum_max_error_v2"], row["absorbing_row_sum_max_error_v1"], row["absorbing_row_sum_max_error_v2"]], "Row-Sum Error Comparison", "max error")
    outputs.append(figures / "m4a_v1_vs_v2_zero_outgoing_rows.png")
    save_bar_plot(outputs[-1], ["v1 non-final", "v2 non-final", "v1 final", "v2 final"], [row["non_final_zero_outgoing_rows_v1"], row["non_final_zero_outgoing_rows_v2"], row["final_time_zero_outgoing_rows_v1"], row["final_time_zero_outgoing_rows_v2"]], "Zero-Outgoing Row Comparison", "rows")
    outputs.append(figures / "m4a_v1_vs_v2_probability_concentration.png")
    save_bar_plot(outputs[-1], ["v1 top1", "v2 top1", "v1 entropy", "v2 entropy"], [row["top1_mean_v1"], row["top1_mean_v2"], row["entropy_mean_v1"], row["entropy_mean_v2"]], "Probability Concentration", "mean")
    expected = by_time[by_time["time_pair"].isin(EXPECTED_TIME_PAIRS)].copy()
    outputs.append(figures / "m4a_v1_vs_v2_source_coverage_by_time_pair.png")
    save_bar_plot(outputs[-1], [f"{tp} v1" for tp in expected["time_pair"]] + [f"{tp} v2" for tp in expected["time_pair"]], expected["source_rows_v1"].tolist() + expected["source_rows_v2"].tolist(), "Source Coverage By Time Pair", "source rows")
    outputs.append(figures / "m4a_v1_vs_v2_target_coverage_by_time_pair.png")
    save_bar_plot(outputs[-1], [f"{tp} v1" for tp in expected["time_pair"]] + [f"{tp} v2" for tp in expected["time_pair"]], expected["target_rows_v1"].tolist() + expected["target_rows_v2"].tolist(), "Target Coverage By Time Pair", "target rows")
    outputs.append(figures / "m4a_v1_vs_v2_matrix_inventory.png")
    save_bar_plot(outputs[-1], summary["version"].str.cat(summary["matrix_name"], sep=" ").tolist(), (summary["disk_bytes"] / (1024 * 1024)).tolist(), "Matrix Object Disk Footprint", "MiB")
    return outputs


def count_ssd_outputs(output_root: Path) -> int:
    if not output_root.exists():
        return 0
    return sum(1 for path in output_root.rglob("*") if str(path).startswith("/ssd/"))


def validate_outputs(paths: dict[str, Path], figure_paths: list[Path]) -> None:
    required = [
        paths["reports"] / "m4a_v2_full_qc_validation_report.md",
        paths["root"] / "m4a_v2_full_qc_validation_summary.csv",
        paths["root"] / "m4a_v1_v2_matrix_object_summary.csv",
        paths["reports"] / "m4a_v1_v2_matrix_object_review.md",
        paths["root"] / "m4a_v1_vs_v2_matrix_comparison_global.csv",
        paths["root"] / "m4a_v1_vs_v2_matrix_comparison_by_time_pair.csv",
        paths["reports"] / "m4a_v1_vs_v2_matrix_comparison_report.md",
        paths["reports"] / "m4a_v2_m4c_v2_readiness_check.md",
        paths["root"] / "m4a_v2_m4c_v2_required_inputs_checklist.csv",
        paths["reports"] / "m4a_v2_biological_sanity_review.md",
        paths["reports"] / "m4a_v2_benchmark_decision_report.md",
        paths["reports"] / "m4a_v2_next_step_recommendation.md",
        paths["root"] / "m4a_v2_benchmark_summary.json",
        *figure_paths,
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty benchmark outputs: {missing}")
    json.loads((paths["root"] / "m4a_v2_benchmark_summary.json").read_text(encoding="utf-8"))
    for csv_name in [
        "m4a_v2_full_qc_validation_summary.csv",
        "m4a_v1_v2_matrix_object_summary.csv",
        "m4a_v1_vs_v2_matrix_comparison_global.csv",
        "m4a_v1_vs_v2_matrix_comparison_by_time_pair.csv",
        "m4a_v2_m4c_v2_required_inputs_checklist.csv",
    ]:
        frame = pd.read_csv(paths["root"] / csv_name)
        if frame.empty:
            raise ValueError(f"Benchmark CSV is empty: {csv_name}")


def run(output_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    paths = ensure_dirs(output_root)
    protected_before = snapshot(PROTECTED_ROOTS)
    downstream_before = snapshot(FORBIDDEN_DOWNSTREAM_ROOTS)
    input_inventory = validate_required_inputs()
    atomic_write_csv(paths["root"] / "m4a_v2_benchmark_input_inventory.csv", input_inventory)

    v1_node = load_node_table(M4A_V1_ROOT / "node_table" / "global_node_table.parquet")
    v2_node = load_node_table(M4A_V2_ROOT / "node_table" / "global_node_table.parquet")
    validate_node_table(v1_node, "M4A-v1")
    validate_node_table(v2_node, "M4A-v2")
    loaded = {
        "v1_forward": load_csr(m4a_v1_matrix_paths()["P_forward_no_terminal_selfloops"]),
        "v1_absorbing": load_csr(m4a_v1_matrix_paths()["P_absorbing_terminal_selfloops"]),
        "v2_forward": load_csr(m4a_v2_matrix_paths()["P_forward_no_terminal_selfloops_v2"]),
        "v2_absorbing": load_csr(m4a_v2_matrix_paths()["P_absorbing_terminal_selfloops_v2"]),
        "v2_weight": load_csr(m4a_v2_matrix_paths()["W_v2_unnormalized_weight"]),
    }
    object_summary = build_matrix_object_summary(v1_node, v2_node, loaded)
    v2_qc_frame, v2_qc_summary = validate_m4a_v2_full_qc(
        loaded["v2_forward"],
        loaded["v2_absorbing"],
        loaded["v2_weight"],
        v2_node,
        object_summary,
    )
    global_comparison = build_global_comparison(
        object_summary,
        loaded["v1_absorbing"],
        loaded["v2_absorbing"],
        v1_node["is_final_time"].to_numpy(dtype=bool),
    )
    v1_coverage = time_pair_coverage(loaded["v1_forward"], v1_node, "M4A-v1")
    v2_coverage = time_pair_coverage(loaded["v2_forward"], v2_node, "M4A-v2")
    by_time = build_by_time_pair_comparison(v1_coverage, v2_coverage)
    readiness = build_readiness_checklist(v1_node, v2_node)
    biological = build_biological_sanity(v1_node, v2_node, by_time, readiness)
    decision, decision_reasons = choose_decision(v2_qc_summary, global_comparison, readiness)
    figure_paths = write_figures(paths, object_summary, global_comparison, by_time)

    protected_after = snapshot(PROTECTED_ROOTS)
    downstream_after = snapshot(FORBIDDEN_DOWNSTREAM_ROOTS)
    upstream_diffs = diff_snapshot(protected_before, protected_after)
    downstream_diffs = diff_snapshot(downstream_before, downstream_after)
    ssd_output_count = count_ssd_outputs(paths["root"])

    summary = {
        "stage": "M4A-v2-03",
        "status": "PASSED" if v2_qc_summary["full_qc_status"] == "PASS" and not upstream_diffs and not downstream_diffs and ssd_output_count == 0 else "FAILED",
        "decision_category": decision,
        "decision_reasons": decision_reasons,
        "output_root": paths["root"],
        "reports_dir": paths["reports"],
        "figures_dir": paths["figures"],
        "full_qc_status": v2_qc_summary["full_qc_status"],
        "m4c_v2_readiness_status": "PASS" if bool((readiness["status"] == "PASS").all()) else "FAIL",
        "biological_sanity_warn_count": int((biological["status"] == "WARN").sum()),
        "upstream_metadata_diff_count": len(upstream_diffs),
        "forbidden_downstream_diff_count": len(downstream_diffs),
        "ssd_output_count": ssd_output_count,
        "upstream_metadata_diffs": upstream_diffs,
        "forbidden_downstream_diffs": downstream_diffs,
        "runtime_seconds": float(time.monotonic() - started),
        "generated_at_utc": utc_now(),
        **v2_qc_summary,
        **global_comparison.iloc[0].to_dict(),
    }

    atomic_write_csv(paths["root"] / "m4a_v2_full_qc_validation_summary.csv", v2_qc_frame)
    atomic_write_csv(paths["root"] / "m4a_v1_v2_matrix_object_summary.csv", object_summary)
    atomic_write_csv(paths["root"] / "m4a_v1_vs_v2_matrix_comparison_global.csv", global_comparison)
    atomic_write_csv(paths["root"] / "m4a_v1_vs_v2_matrix_comparison_by_time_pair.csv", by_time)
    atomic_write_csv(paths["root"] / "m4a_v2_m4c_v2_required_inputs_checklist.csv", readiness)
    atomic_write_json(paths["root"] / "m4a_v2_benchmark_summary.json", summary)

    write_report(
        paths["reports"] / "m4a_v2_full_qc_validation_report.md",
        "M4A-v2 Full QC Validation Report",
        [("Summary", markdown_table(pd.DataFrame([v2_qc_summary]))), ("Checks", markdown_table(v2_qc_frame[["check", "status"]]))],
    )
    write_report(
        paths["reports"] / "m4a_v1_v2_matrix_object_review.md",
        "M4A-v1 vs M4A-v2 Matrix Object Review",
        [("Matrix Objects", markdown_table(object_summary))],
    )
    write_report(
        paths["reports"] / "m4a_v1_vs_v2_matrix_comparison_report.md",
        "M4A-v1 vs M4A-v2 Matrix Comparison Report",
        [("Global Comparison", markdown_table(global_comparison)), ("By Time Pair", markdown_table(by_time))],
    )
    write_report(
        paths["reports"] / "m4a_v2_m4c_v2_readiness_check.md",
        "M4A-v2 M4C-v2 Readiness Check",
        [("Checklist", markdown_table(readiness))],
    )
    write_report(
        paths["reports"] / "m4a_v2_biological_sanity_review.md",
        "M4A-v2 Biological Sanity Review",
        [("Sanity Checks", markdown_table(biological))],
    )
    architecture_note = (
        "M4A-v2 belongs to the P_fate / endpoint-anchored Markov propagation branch. "
        "M4A-v2 does not solve pyGPCCA. K_gpcca remains a separate future GPCCA-compatible "
        "kernel requiring within-time niche manifold connectivity plus cross-time evidence. "
        "No custom GPCCA-like result should be treated as standard GPCCA."
    )
    decision_body = (
        f"- decision_category: {decision}\n"
        f"- reasons: {'; '.join(decision_reasons)}\n"
        f"- full_qc_status: {summary['full_qc_status']}\n"
        f"- m4c_v2_readiness_status: {summary['m4c_v2_readiness_status']}\n"
        f"- upstream_metadata_diff_count: {len(upstream_diffs)}\n"
        f"- forbidden_downstream_diff_count: {len(downstream_diffs)}\n"
        f"- /ssd output count: {ssd_output_count}\n\n"
        f"{architecture_note}"
    )
    write_report(
        paths["reports"] / "m4a_v2_benchmark_decision_report.md",
        "M4A-v2 Benchmark Decision Report",
        [("Decision", decision_body)],
    )
    next_step = "Proceed to M4C-v2 planning only. Do not execute M4C-v2 in this task." if decision == "proceed_to_m4c_v2_planning" else "Do not proceed to M4C-v2 planning until failed checks are resolved."
    atomic_write_text(paths["reports"] / "m4a_v2_next_step_recommendation.md", next_step + "\n")

    validate_outputs(paths, figure_paths)
    if upstream_diffs or downstream_diffs or ssd_output_count:
        raise RuntimeError("M4A-v2 benchmark safety validation failed.")
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(json_safe(run(args.output_root)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
