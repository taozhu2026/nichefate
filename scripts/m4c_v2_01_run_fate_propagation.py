#!/usr/bin/env python
"""Run M4C-v2 endpoint-anchored Markov fate propagation.

The runner supports dry-run/preflight, bounded smoke propagation, and full
production propagation. It consumes M4A-v2 transition objects and M4E endpoint
annotations, preserves raw terminal endpoint columns, and writes only under the
configured M4C-v2 output root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for thread_var in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(thread_var, "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config


DEFAULT_CONFIG = "configs/m4c_v2_fate_propagation.yaml"
ROOT = Path("/home/zhutao/scratch/nichefate")
NO_DOWNSTREAM_FLAGS = {
    "no_gpcca": True,
    "no_m4d_diagnostics": True,
    "no_k_gpcca": True,
    "no_barcode_preprocessing": True,
    "no_m5": True,
    "no_regulator_analysis": True,
    "no_branchsbm": True,
    "no_branched_nicheflow": True,
}
REQUIRED_NODE_COLUMNS = {
    "global_node_index",
    "anchor_id",
    "time",
    "time_day",
    "is_final_time",
}
REQUIRED_ENDPOINT_MAPPING_COLUMNS = {
    "raw_terminal_macrostate",
    "raw_terminal_macrostate_label",
    "refined_endpoint_id",
    "refined_endpoint_label",
    "confidence_tier_after_refinement",
}
OPTIONAL_NEIGHBORHOOD_COLUMNS = [
    "global_node_index",
    "slice_id",
    "anchor_index",
    "anchor_cell_id",
    "mouse_id",
    "cell_type_l1",
    "cell_type_l2",
    "cell_type_l3",
    "leiden_neigh",
    "cadinu_neighborhood_label",
    "x",
    "y",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
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
        right_resolved,
        left_resolved,
    )


def reject_ssd(path: Path, label: str = "path") -> None:
    path = resolved(path)
    if path == Path("/ssd") or Path("/ssd") in path.parents:
        raise ValueError(f"Refusing /ssd {label}: {path}")


def validate_config(config: dict[str, Any]) -> dict[str, Path]:
    for section in ["paths", "inputs", "fate", "validation"]:
        if section not in config:
            raise KeyError(f"Missing config section: {section}")
    paths = {key: resolved(value) for key, value in config["paths"].items()}
    inputs = {key: resolved(value) for key, value in config["inputs"].items()}
    protected = [resolved(path) for path in config.get("protected_roots", [])]
    forbidden = [resolved(path) for path in config.get("forbidden_downstream_roots", [])]

    for key, path in {**paths, **inputs}.items():
        reject_ssd(path, key)

    output_root = paths["output_root"]
    for protected_root in protected:
        if paths_overlap(output_root, protected_root):
            raise ValueError(
                f"Output root overlaps protected root {protected_root}: {output_root}"
            )
    for key in ["reports_dir", "tmp_dir", "fate_dir"]:
        if not is_relative_to(paths[key], output_root):
            raise ValueError(f"paths.{key} must be under output_root: {paths[key]}")
    for key, path in paths.items():
        for forbidden_root in forbidden:
            if is_relative_to(path, forbidden_root):
                raise ValueError(
                    f"Configured output path {key} falls under forbidden root "
                    f"{forbidden_root}: {path}"
                )
    if config["fate"].get("method") != "time_layered_backward_propagation":
        raise ValueError("M4C-v2 supports time_layered_backward_propagation only.")
    if not bool(config["fate"].get("preserve_raw_terminal_columns", False)):
        raise ValueError("M4C-v2 requires preserve_raw_terminal_columns=true.")
    return {**paths, **{f"input_{key}": value for key, value in inputs.items()}}


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


def atomic_write_text(path: Path, text: str) -> None:
    reject_ssd(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    reject_ssd(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    reject_ssd(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    reject_ssd(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_save_npz(path: Path, matrix: sp.spmatrix) -> None:
    reject_ssd(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    sp.save_npz(tmp, matrix, compressed=True)
    os.replace(tmp, path)


def snapshot(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for root in paths:
        if not root.exists():
            out[str(root)] = {"exists": False, "size": -1, "mtime_ns": -1, "is_dir": False}
            continue
        entries = [root, *sorted(path for path in root.rglob("*") if path.exists())]
        for path in entries:
            stat = path.stat()
            out[str(path)] = {
                "exists": True,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "is_dir": path.is_dir(),
            }
    return out


def diff_snapshot(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> list[str]:
    diffs: list[str] = []
    for path in sorted(set(before) | set(after)):
        if path not in before:
            diffs.append(f"ADDED\t{path}")
        elif path not in after:
            diffs.append(f"REMOVED\t{path}")
        elif before[path] != after[path]:
            diffs.append(
                f"CHANGED\t{path}\tbefore={before[path]}\tafter={after[path]}"
            )
    return diffs


def protected_roots(config: dict[str, Any]) -> list[Path]:
    return [resolved(path) for path in config.get("protected_roots", [])]


def forbidden_roots(config: dict[str, Any]) -> list[Path]:
    return [resolved(path) for path in config.get("forbidden_downstream_roots", [])]


def dryrun_outputs(paths: dict[str, Path]) -> dict[str, Path]:
    reports = paths["reports_dir"]
    return {
        "preflight_report": reports / "m4c_v2_01_preflight_report.md",
        "dryrun_summary": reports / "m4c_v2_01_dryrun_summary.json",
        "input_validation": reports / "m4c_v2_01_input_validation.csv",
        "endpoint_mapping_validation": reports
        / "m4c_v2_01_endpoint_mapping_validation.csv",
    }


def smoke_outputs(paths: dict[str, Path]) -> dict[str, Path]:
    reports = paths["reports_dir"]
    return {
        "smoke_report": reports / "m4c_v2_02_smoke_validation_report.md",
    }


def production_outputs(paths: dict[str, Path]) -> dict[str, Path]:
    fate_dir = paths["fate_dir"]
    reports = paths["reports_dir"]
    return {
        "fate_matrix": fate_dir / "fate_probability_matrix_v2.npz",
        "node_summary": fate_dir / "node_fate_summary_v2.parquet",
        "dominant_endpoint": fate_dir / "dominant_endpoint_assignment_v2.parquet",
        "plasticity": fate_dir / "plasticity_score_v2.parquet",
        "by_time": fate_dir / "fate_probability_by_time_summary_v2.csv",
        "by_slice": fate_dir / "fate_probability_by_slice_summary_v2.csv",
        "by_mouse": fate_dir / "fate_probability_by_mouse_summary_v2.csv",
        "by_neighborhood": fate_dir / "fate_probability_by_neighborhood_summary_v2.csv",
        "endpoint_composition": fate_dir / "endpoint_composition_summary_v2.csv",
        "report": reports / "m4c_v2_02_full_propagation_report.md",
        "qc_summary": reports / "m4c_v2_02_qc_summary.csv",
        "output_inventory": reports / "m4c_v2_02_output_inventory.csv",
        "next_step": reports / "m4c_v2_02_next_step_recommendation.md",
        "completed_manifest": reports / "m4c_v2_02_completed_manifest.csv",
        "failed_manifest": reports / "m4c_v2_02_failed_manifest.csv",
    }


def validate_no_existing_production_outputs(
    outputs: dict[str, Path],
    overwrite: bool,
) -> None:
    if overwrite:
        return
    existing = [
        str(path)
        for key, path in outputs.items()
        if key != "failed_manifest" and path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Existing M4C-v2 production outputs require --overwrite: "
            + ", ".join(existing[:8])
        )


def validate_output_paths(paths: dict[str, Path], config: dict[str, Any]) -> None:
    all_outputs = {
        **dryrun_outputs(paths),
        **smoke_outputs(paths),
        **production_outputs(paths),
    }
    output_root = paths["output_root"]
    forbidden = forbidden_roots(config)
    for key, path in all_outputs.items():
        reject_ssd(path, key)
        if not is_relative_to(resolved(path), output_root):
            raise ValueError(f"Output {key} is outside output_root: {path}")
        for forbidden_root in forbidden:
            if is_relative_to(resolved(path), forbidden_root):
                raise ValueError(
                    f"Output {key} falls under forbidden downstream root "
                    f"{forbidden_root}: {path}"
                )


def input_validation_frame(paths: dict[str, Path]) -> pd.DataFrame:
    input_keys = sorted(key for key in paths if key.startswith("input_"))
    rows = []
    for key in input_keys:
        path = paths[key]
        exists = path.is_file()
        rows.append(
            {
                "input_name": key.removeprefix("input_"),
                "path": str(path),
                "exists": bool(exists),
                "bytes": int(path.stat().st_size) if exists else 0,
                "status": "PASS" if exists and path.stat().st_size > 0 else "FAIL",
            }
        )
    frame = pd.DataFrame(rows)
    failed = frame.loc[frame["status"] != "PASS", "input_name"].tolist()
    if failed:
        raise FileNotFoundError(f"Missing or empty M4C-v2 inputs: {failed}")
    return frame


def load_sparse(path: Path) -> sp.csr_matrix:
    matrix = sp.load_npz(path).tocsr()
    matrix.sort_indices()
    return matrix


def sparse_invalid_counts(matrix: sp.csr_matrix) -> dict[str, Any]:
    data = matrix.data
    return {
        "nonfinite_count": int((~np.isfinite(data)).sum()),
        "negative_count": int((data < 0).sum()),
        "data_min": float(data.min()) if data.size else 0.0,
        "data_max": float(data.max()) if data.size else 0.0,
    }


def validate_node_table(
    node_table: pd.DataFrame,
    expected_nodes: int,
    expected_final_nodes: int,
    final_time_label: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    missing = sorted(REQUIRED_NODE_COLUMNS - set(node_table.columns))
    if missing:
        raise KeyError(f"M4A-v2 node table missing columns: {missing}")
    table = node_table.copy()
    if len(table) != int(expected_nodes):
        raise ValueError(f"Expected {expected_nodes} nodes, found {len(table)}.")
    if bool(table["global_node_index"].isna().any()):
        raise ValueError("global_node_index contains missing values.")
    expected = np.arange(len(table), dtype=np.int64)
    observed = table["global_node_index"].to_numpy(dtype=np.int64, copy=True)
    if not np.array_equal(observed, expected):
        raise ValueError("M4A-v2 node table must preserve row i == global_node_index i.")
    if bool(table["anchor_id"].duplicated().any()):
        raise ValueError("M4A-v2 node table anchor_id values must be unique.")
    final_mask = table["is_final_time"].astype(bool).to_numpy()
    if int(final_mask.sum()) != int(expected_final_nodes):
        raise ValueError(
            f"Expected {expected_final_nodes} final nodes, found {int(final_mask.sum())}."
        )
    final_times = set(table.loc[final_mask, "time"].astype(str).unique())
    if final_times != {str(final_time_label)}:
        raise ValueError(f"Final-time labels must be {final_time_label}, found {final_times}.")
    return table, final_mask


def endpoint_labels(mapping: pd.DataFrame) -> pd.DataFrame:
    mapping = mapping.sort_values("raw_terminal_macrostate", kind="mergesort").reset_index(drop=True)
    return mapping


def validate_endpoint_mapping(
    mapping: pd.DataFrame,
    expected_endpoint_count: int,
) -> pd.DataFrame:
    missing = sorted(REQUIRED_ENDPOINT_MAPPING_COLUMNS - set(mapping.columns))
    if missing:
        raise KeyError(f"Endpoint mapping missing columns: {missing}")
    if len(mapping) != int(expected_endpoint_count):
        raise ValueError(
            f"Expected {expected_endpoint_count} endpoint mapping rows, found {len(mapping)}."
        )
    table = endpoint_labels(mapping.copy())
    raw = table["raw_terminal_macrostate"].to_numpy(dtype=np.int64)
    expected = np.arange(int(expected_endpoint_count), dtype=np.int64)
    if not np.array_equal(raw, expected):
        raise ValueError("Raw terminal endpoints must be contiguous 0..endpoint_count-1.")
    return table


def validate_endpoint_assignments(
    endpoint_nodes: pd.DataFrame,
    node_table: pd.DataFrame,
    final_mask: np.ndarray,
    endpoint_mapping: pd.DataFrame,
    macrostate_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"global_node_index", "anchor_id", macrostate_column}
    missing = sorted(required - set(endpoint_nodes.columns))
    if missing:
        raise KeyError(f"Endpoint node annotation missing columns: {missing}")
    assignments = endpoint_nodes[["global_node_index", "anchor_id", macrostate_column]].copy()
    assignments = assignments.rename(columns={macrostate_column: "terminal_macrostate"})
    if bool(assignments["global_node_index"].duplicated().any()):
        raise ValueError("Endpoint annotations contain duplicate global_node_index values.")
    if bool(assignments["terminal_macrostate"].isna().any()):
        raise ValueError("Endpoint annotations contain missing terminal macrostate IDs.")
    assignments["global_node_index"] = assignments["global_node_index"].astype(np.int64)
    assignments["terminal_macrostate"] = assignments["terminal_macrostate"].astype(np.int32)
    final_indices = node_table.loc[final_mask, "global_node_index"].to_numpy(dtype=np.int64)
    assignment_indices = assignments["global_node_index"].to_numpy(dtype=np.int64)
    missing_final = np.setdiff1d(final_indices, assignment_indices, assume_unique=False)
    extra = np.setdiff1d(assignment_indices, final_indices, assume_unique=False)
    missing_anchor = 0
    if len(missing_final) or len(extra):
        missing_anchor += int(len(missing_final) + len(extra))
    aligned = node_table.loc[assignments["global_node_index"], ["anchor_id"]].reset_index(drop=True)
    missing_anchor += int((aligned["anchor_id"].astype(str) != assignments["anchor_id"].astype(str)).sum())
    allowed = set(endpoint_mapping["raw_terminal_macrostate"].astype(int).tolist())
    invalid_macro = sorted(set(assignments["terminal_macrostate"].astype(int)) - allowed)
    if invalid_macro:
        raise ValueError(f"Endpoint annotations contain invalid macrostates: {invalid_macro}")
    if missing_anchor:
        raise ValueError(f"Endpoint mapping missing/mismatched count: {missing_anchor}")
    label_frame = endpoint_mapping[
        [
            "raw_terminal_macrostate",
            "raw_terminal_macrostate_label",
            "refined_endpoint_id",
            "refined_endpoint_label",
            "confidence_tier_after_refinement",
        ]
    ].rename(columns={"raw_terminal_macrostate": "terminal_macrostate"})
    assignments = assignments.merge(label_frame, on="terminal_macrostate", how="left")
    if bool(assignments["refined_endpoint_id"].isna().any()):
        raise ValueError("Endpoint assignments failed to join refined endpoint metadata.")
    assignments = assignments.sort_values("global_node_index", kind="mergesort").reset_index(drop=True)
    validation = pd.DataFrame(
        [
            {
                "check": "endpoint_node_rows",
                "observed": int(len(assignments)),
                "expected": int(final_mask.sum()),
                "status": "PASS" if len(assignments) == int(final_mask.sum()) else "FAIL",
            },
            {
                "check": "missing_endpoint_mappings",
                "observed": int(missing_anchor),
                "expected": 0,
                "status": "PASS" if missing_anchor == 0 else "FAIL",
            },
            {
                "check": "raw_terminal_endpoint_columns",
                "observed": int(endpoint_mapping["raw_terminal_macrostate"].nunique()),
                "expected": int(len(endpoint_mapping)),
                "status": "PASS",
            },
            {
                "check": "unique_refined_endpoint_ids",
                "observed": int(endpoint_mapping["refined_endpoint_id"].nunique()),
                "expected": int(endpoint_mapping["refined_endpoint_id"].nunique()),
                "status": "PASS",
            },
            {
                "check": "merge_candidate_mapping_rows",
                "observed": int(endpoint_mapping["refined_endpoint_id"].duplicated(keep=False).sum()),
                "expected": "metadata_only",
                "status": "PASS",
            },
        ]
    )
    return assignments, validation


def matrix_metadata_validation(
    p_forward: sp.csr_matrix,
    p_absorbing: sp.csr_matrix,
    final_mask: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    validation = config["validation"]
    n_nodes = int(validation["expected_global_nodes"])
    expected_forward_nnz = int(validation["expected_forward_nnz"])
    expected_absorbing_nnz = int(validation["expected_absorbing_nnz"])
    if p_forward.shape != (n_nodes, n_nodes):
        raise ValueError(f"P_forward shape {p_forward.shape} != {(n_nodes, n_nodes)}")
    if p_absorbing.shape != (n_nodes, n_nodes):
        raise ValueError(f"P_absorbing shape {p_absorbing.shape} != {(n_nodes, n_nodes)}")
    if int(p_forward.nnz) != expected_forward_nnz:
        raise ValueError(f"P_forward nnz {p_forward.nnz} != {expected_forward_nnz}")
    if int(p_absorbing.nnz) != expected_absorbing_nnz:
        raise ValueError(f"P_absorbing nnz {p_absorbing.nnz} != {expected_absorbing_nnz}")
    forward_invalid = sparse_invalid_counts(p_forward)
    absorbing_invalid = sparse_invalid_counts(p_absorbing)
    final_outgoing = int(p_forward[final_mask, :].nnz)
    final_absorbing_diag = int(p_absorbing[final_mask, :][:, final_mask].diagonal().sum())
    if final_outgoing:
        raise ValueError(f"P_forward final-time rows have outgoing nnz: {final_outgoing}")
    if forward_invalid["nonfinite_count"] or absorbing_invalid["nonfinite_count"]:
        raise ValueError("M4A-v2 matrices contain non-finite entries.")
    if forward_invalid["negative_count"] or absorbing_invalid["negative_count"]:
        raise ValueError("M4A-v2 matrices contain negative entries.")
    return {
        "matrix_shape": f"{n_nodes}x{n_nodes}",
        "forward_nnz": int(p_forward.nnz),
        "absorbing_nnz": int(p_absorbing.nnz),
        "forward_final_time_outgoing_nnz": final_outgoing,
        "absorbing_final_diagonal_sum": final_absorbing_diag,
        **{f"forward_{key}": value for key, value in forward_invalid.items()},
        **{f"absorbing_{key}": value for key, value in absorbing_invalid.items()},
    }


def validate_m4a_v2_qc(qc_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    qc = pd.read_csv(qc_path)
    if qc.empty:
        raise ValueError("M4A-v2 QC summary is empty.")
    row = qc.iloc[0].to_dict()
    validation = config["validation"]
    checks = {
        "status": str(row.get("status")) == "COMPLETED",
        "matrix_shape": str(row.get("matrix_shape"))
        == f"{validation['expected_global_nodes']}x{validation['expected_global_nodes']}",
        "forward_nnz": int(row.get("forward_nnz", -1)) == int(validation["expected_forward_nnz"]),
        "absorbing_nnz": int(row.get("absorbing_nnz", -1)) == int(validation["expected_absorbing_nnz"]),
        "invalid_entries": int(row.get("p_forward_nonfinite_count", 0)) == 0
        and int(row.get("p_forward_negative_count", 0)) == 0
        and int(row.get("p_absorbing_nonfinite_count", 0)) == 0
        and int(row.get("p_absorbing_negative_count", 0)) == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"M4A-v2 QC summary failed checks: {failed}")
    return row


def time_layers(node_table: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (time_day, time_label), group in node_table.groupby(["time_day", "time"], sort=True, observed=True):
        rows.append(
            {
                "time_day": float(time_day),
                "time": str(time_label),
                "indices": group["global_node_index"].to_numpy(dtype=np.int64),
            }
        )
    rows = sorted(rows, key=lambda row: (row["time_day"], row["time"]))
    if len(rows) < 2:
        raise ValueError("M4C-v2 requires at least two time layers.")
    return rows


def initialize_terminal_fates(
    n_nodes: int,
    n_endpoints: int,
    assignments: pd.DataFrame,
    dtype: np.dtype,
) -> np.ndarray:
    fate = np.zeros((n_nodes, n_endpoints), dtype=dtype)
    indices = assignments["global_node_index"].to_numpy(dtype=np.int64)
    endpoints = assignments["terminal_macrostate"].to_numpy(dtype=np.int64)
    fate[indices, endpoints] = np.array(1.0, dtype=dtype)
    return fate


def compute_fate_probabilities(
    p_forward: sp.csr_matrix,
    node_table: pd.DataFrame,
    assignments: pd.DataFrame,
    n_endpoints: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n_nodes = int(len(node_table))
    fate = initialize_terminal_fates(n_nodes, n_endpoints, assignments, dtype)
    layers = time_layers(node_table)
    time_day_by_index = node_table["time_day"].astype(float).to_numpy()
    propagation_steps: list[dict[str, Any]] = []
    for layer_position in range(len(layers) - 2, -1, -1):
        layer = layers[layer_position]
        next_layer = layers[layer_position + 1]
        indices = layer["indices"]
        block = p_forward[indices, :]
        if block.nnz:
            target_days = time_day_by_index[block.indices]
            if not np.allclose(target_days, float(next_layer["time_day"])):
                raise ValueError(
                    "P_forward is not an adjacent time-layer DAG for "
                    f"{layer['time']} -> {next_layer['time']}."
                )
        fate[indices, :] = block.dot(fate).astype(dtype, copy=False)
        propagation_steps.append(
            {
                "source_time": layer["time"],
                "source_time_day": layer["time_day"],
                "target_time": next_layer["time"],
                "source_nodes": int(len(indices)),
                "transition_nnz": int(block.nnz),
            }
        )
    propagation_steps.reverse()
    return fate, propagation_steps


def validate_fate_matrix(
    fate: np.ndarray,
    final_mask: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    nonfinite = int((~np.isfinite(fate)).sum())
    negative = int((fate < 0).sum())
    if nonfinite:
        raise ValueError(f"Fate matrix contains non-finite values: {nonfinite}")
    if negative:
        raise ValueError(f"Fate matrix contains negative values: {negative}")
    row_sums = fate.sum(axis=1, dtype=np.float64)
    row_error = np.abs(row_sums - 1.0)
    nonfinal_error = row_error[~final_mask]
    final_error = row_error[final_mask]
    final_rows = fate[final_mask]
    final_onehot_error = (
        np.minimum(np.abs(final_rows - 0.0), np.abs(final_rows - 1.0)).max(axis=1)
        if len(final_rows)
        else np.array([0.0])
    )
    rows_exceeding = int((row_error > tolerance).sum())
    final_onehot_fail = int((final_onehot_error > tolerance).sum())
    if rows_exceeding:
        raise ValueError(f"{rows_exceeding} fate rows exceed row-sum tolerance {tolerance}.")
    if final_onehot_fail:
        raise ValueError(f"{final_onehot_fail} final rows are not one-hot.")
    return {
        "row_sum_tolerance": float(tolerance),
        "row_sum_max_error": float(row_error.max()) if len(row_error) else 0.0,
        "row_sum_p99_error": float(np.quantile(row_error, 0.99)) if len(row_error) else 0.0,
        "nonfinal_row_sum_max_error": float(nonfinal_error.max()) if len(nonfinal_error) else 0.0,
        "final_row_sum_max_error": float(final_error.max()) if len(final_error) else 0.0,
        "final_onehot_error_max": float(final_onehot_error.max()) if len(final_onehot_error) else 0.0,
        "rows_exceeding_tolerance": rows_exceeding,
        "nan_or_inf_values": nonfinite,
        "negative_values": negative,
    }


def fate_metrics(
    fate: np.ndarray,
    endpoint_mapping: pd.DataFrame,
) -> pd.DataFrame:
    probs = fate.astype(np.float64, copy=False)
    positive = probs > 0.0
    entropy_terms = np.where(positive, probs * np.log(np.clip(probs, 1e-300, None)), 0.0)
    entropy = -entropy_terms.sum(axis=1)
    entropy = np.maximum(entropy, 0.0)
    normalizer = float(np.log(fate.shape[1])) if fate.shape[1] > 1 else 1.0
    normalized_entropy = np.clip(entropy / normalizer, 0.0, 1.0) if normalizer else entropy
    dominant_col = fate.argmax(axis=1)
    top1 = fate[np.arange(fate.shape[0]), dominant_col].astype(np.float64, copy=False)
    top2 = (
        np.partition(fate, -2, axis=1)[:, -2].astype(np.float64, copy=False)
        if fate.shape[1] > 1
        else np.zeros(fate.shape[0], dtype=np.float64)
    )
    raw_ids = endpoint_mapping["raw_terminal_macrostate"].to_numpy(dtype=np.int32)
    raw_labels = endpoint_mapping["raw_terminal_macrostate_label"].astype(str).to_numpy(dtype=object)
    refined_ids = endpoint_mapping["refined_endpoint_id"].astype(str).to_numpy(dtype=object)
    refined_labels = endpoint_mapping["refined_endpoint_label"].astype(str).to_numpy(dtype=object)
    confidence = endpoint_mapping["confidence_tier_after_refinement"].astype(str).to_numpy(dtype=object)
    return pd.DataFrame(
        {
            "plasticity_entropy": entropy,
            "normalized_plasticity_entropy": normalized_entropy,
            "dominant_endpoint": raw_ids[dominant_col],
            "dominant_endpoint_label": raw_labels[dominant_col],
            "dominant_refined_endpoint_id": refined_ids[dominant_col],
            "dominant_refined_endpoint_label": refined_labels[dominant_col],
            "dominant_endpoint_confidence_tier": confidence[dominant_col],
            "dominant_endpoint_probability": top1,
            "fate_margin_top1_minus_top2": top1 - top2,
        }
    )


def read_optional_neighborhood(path: Path, n_nodes: int) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame({"global_node_index": np.arange(n_nodes, dtype=np.int64)})
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema.names)
    except Exception:  # noqa: BLE001
        available = set(pd.read_parquet(path).columns)
    columns = [column for column in OPTIONAL_NEIGHBORHOOD_COLUMNS if column in available]
    if "global_node_index" not in columns:
        return pd.DataFrame({"global_node_index": np.arange(n_nodes, dtype=np.int64)})
    frame = pd.read_parquet(path, columns=columns)
    if len(frame) != n_nodes:
        return pd.DataFrame({"global_node_index": np.arange(n_nodes, dtype=np.int64)})
    frame = frame.sort_values("global_node_index", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(frame["global_node_index"].to_numpy(dtype=np.int64), np.arange(n_nodes, dtype=np.int64)):
        return pd.DataFrame({"global_node_index": np.arange(n_nodes, dtype=np.int64)})
    return frame


def build_node_summary(
    node_table: pd.DataFrame,
    assignments: pd.DataFrame,
    endpoint_mapping: pd.DataFrame,
    metrics: pd.DataFrame,
    neighborhood: pd.DataFrame,
    directionality_evidence_source: str,
    barcode_compatible_contract: bool,
) -> pd.DataFrame:
    summary = node_table[["global_node_index", "anchor_id", "time", "time_day", "is_final_time"]].copy()
    if not neighborhood.empty:
        extra = neighborhood.drop(columns=[column for column in ["anchor_id", "time", "time_day", "is_final_time"] if column in neighborhood.columns], errors="ignore")
        summary = summary.merge(extra, on="global_node_index", how="left", sort=False)
    n_nodes = len(summary)
    terminal_macrostate = np.full(n_nodes, -1, dtype=np.int32)
    terminal_label = np.full(n_nodes, "non_terminal", dtype=object)
    refined_id = np.full(n_nodes, "non_terminal", dtype=object)
    refined_label = np.full(n_nodes, "non_terminal", dtype=object)
    confidence = np.full(n_nodes, "non_terminal", dtype=object)
    indices = assignments["global_node_index"].to_numpy(dtype=np.int64)
    terminal_macrostate[indices] = assignments["terminal_macrostate"].to_numpy(dtype=np.int32)
    terminal_label[indices] = assignments["raw_terminal_macrostate_label"].astype(str).to_numpy(dtype=object)
    refined_id[indices] = assignments["refined_endpoint_id"].astype(str).to_numpy(dtype=object)
    refined_label[indices] = assignments["refined_endpoint_label"].astype(str).to_numpy(dtype=object)
    confidence[indices] = assignments["confidence_tier_after_refinement"].astype(str).to_numpy(dtype=object)
    summary["terminal_macrostate"] = terminal_macrostate
    summary["terminal_macrostate_label"] = terminal_label
    summary["terminal_refined_endpoint_id"] = refined_id
    summary["terminal_refined_endpoint_label"] = refined_label
    summary["terminal_endpoint_confidence_tier"] = confidence
    for column in metrics.columns:
        summary[column] = metrics[column].to_numpy()
    summary["endpoint_column_count"] = int(len(endpoint_mapping))
    summary["unique_refined_endpoint_count"] = int(endpoint_mapping["refined_endpoint_id"].nunique())
    summary["directionality_evidence_source"] = directionality_evidence_source
    summary["barcode_compatible_contract"] = bool(barcode_compatible_contract)
    return summary


def group_probability_summary(
    node_summary: pd.DataFrame,
    fate: np.ndarray,
    group_columns: list[str],
    endpoint_mapping: pd.DataFrame,
) -> pd.DataFrame:
    if any(column not in node_summary.columns for column in group_columns):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in node_summary.groupby(group_key, sort=True, dropna=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        indices = group["global_node_index"].to_numpy(dtype=np.int64)
        probs = fate[indices, :].astype(np.float64, copy=False)
        sums = probs.sum(axis=0)
        denom = float(sums.sum())
        dominant_counts = group["dominant_endpoint"].value_counts()
        base = {column: value for column, value in zip(group_columns, key, strict=True)}
        for col_idx, endpoint in endpoint_mapping.iterrows():
            endpoint_id = int(endpoint["raw_terminal_macrostate"])
            rows.append(
                {
                    **base,
                    "n_nodes": int(len(group)),
                    "terminal_macrostate": endpoint_id,
                    "terminal_macrostate_label": str(endpoint["raw_terminal_macrostate_label"]),
                    "refined_endpoint_id": str(endpoint["refined_endpoint_id"]),
                    "refined_endpoint_label": str(endpoint["refined_endpoint_label"]),
                    "mean_probability": float(probs[:, col_idx].mean()) if len(group) else 0.0,
                    "sum_probability": float(sums[col_idx]),
                    "normalized_mass_fraction": float(sums[col_idx] / denom) if denom else 0.0,
                    "dominant_endpoint_fraction": float(dominant_counts.get(endpoint_id, 0) / len(group)) if len(group) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def endpoint_composition_summary(
    fate: np.ndarray,
    node_summary: pd.DataFrame,
    assignments: pd.DataFrame,
    endpoint_mapping: pd.DataFrame,
) -> pd.DataFrame:
    terminal_counts = assignments["terminal_macrostate"].value_counts()
    dominant_counts = node_summary["dominant_endpoint"].value_counts()
    masses = fate.sum(axis=0, dtype=np.float64)
    rows = []
    for col_idx, endpoint in endpoint_mapping.iterrows():
        raw = int(endpoint["raw_terminal_macrostate"])
        rows.append(
            {
                "terminal_macrostate": raw,
                "terminal_macrostate_label": str(endpoint["raw_terminal_macrostate_label"]),
                "refined_endpoint_id": str(endpoint["refined_endpoint_id"]),
                "refined_endpoint_label": str(endpoint["refined_endpoint_label"]),
                "confidence_tier_after_refinement": str(endpoint["confidence_tier_after_refinement"]),
                "terminal_node_count": int(terminal_counts.get(raw, 0)),
                "dominant_node_count": int(dominant_counts.get(raw, 0)),
                "total_probability_mass": float(masses[col_idx]),
                "mean_probability": float(masses[col_idx] / fate.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def select_smoke_subset(
    p_forward: sp.csr_matrix,
    node_table: pd.DataFrame,
    max_sources: int,
    max_nodes: int | None,
) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    final_day = float(node_table.loc[node_table["is_final_time"].astype(bool), "time_day"].max())
    nonfinal = node_table.loc[node_table["time_day"].astype(float) < final_day]
    latest_day = float(nonfinal["time_day"].astype(float).max())
    candidates = nonfinal.loc[np.isclose(nonfinal["time_day"].astype(float), latest_day), "global_node_index"].to_numpy(dtype=np.int64)
    row_nnz = p_forward.indptr[candidates + 1] - p_forward.indptr[candidates]
    source_indices = candidates[row_nnz > 0][: int(max_sources)]
    if len(source_indices) == 0:
        raise ValueError("Smoke subset has no source rows with outgoing transitions.")
    while True:
        block = p_forward[source_indices, :]
        target_indices = np.unique(block.indices.astype(np.int64))
        subset_size = int(len(source_indices) + len(target_indices))
        if max_nodes is None or subset_size <= int(max_nodes) or len(source_indices) <= 1:
            break
        source_indices = source_indices[: max(1, len(source_indices) // 2)]
    target_final = node_table.loc[target_indices, "is_final_time"].astype(bool).to_numpy()
    if not bool(target_final.all()):
        raise ValueError("Smoke latest-layer targets must all be final-time nodes.")
    subset_indices = np.concatenate([source_indices, target_indices]).astype(np.int64)
    local_index = {int(global_idx): local_idx for local_idx, global_idx in enumerate(subset_indices)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for local_row, global_row in enumerate(source_indices):
        start, end = p_forward.indptr[global_row], p_forward.indptr[global_row + 1]
        for global_col, value in zip(p_forward.indices[start:end], p_forward.data[start:end], strict=True):
            if int(global_col) in local_index:
                rows.append(local_row)
                cols.append(local_index[int(global_col)])
                data.append(float(value))
    local = sp.csr_matrix(
        (
            np.asarray(data, dtype=p_forward.dtype),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(len(subset_indices), len(subset_indices)),
    )
    return subset_indices, source_indices, local


def compute_smoke_fate(
    p_forward: sp.csr_matrix,
    node_table: pd.DataFrame,
    assignments: pd.DataFrame,
    endpoint_mapping: pd.DataFrame,
    max_sources: int,
    max_nodes: int | None,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    subset_indices, source_indices, local = select_smoke_subset(
        p_forward,
        node_table,
        max_sources,
        max_nodes,
    )
    n_endpoints = len(endpoint_mapping)
    fate = np.zeros((len(subset_indices), n_endpoints), dtype=dtype)
    local_index = {int(global_idx): local_idx for local_idx, global_idx in enumerate(subset_indices)}
    assignment_lookup = dict(
        zip(
            assignments["global_node_index"].astype(int),
            assignments["terminal_macrostate"].astype(int),
            strict=True,
        )
    )
    final_mask = node_table.loc[subset_indices, "is_final_time"].astype(bool).to_numpy()
    missing = 0
    for local_idx, global_idx in enumerate(subset_indices):
        if final_mask[local_idx]:
            endpoint = assignment_lookup.get(int(global_idx))
            if endpoint is None:
                missing += 1
            else:
                fate[local_idx, endpoint] = np.array(1.0, dtype=dtype)
    if missing:
        raise ValueError(f"Smoke subset missing endpoint assignments: {missing}")
    source_local = np.array([local_index[int(global_idx)] for global_idx in source_indices], dtype=np.int64)
    fate[source_local, :] = local[source_local, :].dot(fate).astype(dtype, copy=False)
    summary = {
        "smoke_subset_nodes": int(len(subset_indices)),
        "smoke_source_rows": int(len(source_indices)),
        "smoke_local_nnz": int(local.nnz),
    }
    return fate, subset_indices, final_mask, summary


def output_inventory(outputs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, path in outputs.items():
        if path.exists():
            rows.append(
                {
                    "output_name": name,
                    "path": str(path),
                    "exists": True,
                    "bytes": int(path.stat().st_size),
                }
            )
    return pd.DataFrame(rows)


def qc_summary_payload(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "propagation_steps"}


def count_ssd_outputs(output_root: Path) -> int:
    if not output_root.exists():
        return 0
    return int(sum(str(path.resolve()).startswith("/ssd/") for path in output_root.rglob("*")))


def write_dryrun_outputs(
    outputs: dict[str, Path],
    input_validation: pd.DataFrame,
    endpoint_validation: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    atomic_write_csv(outputs["input_validation"], input_validation)
    atomic_write_csv(outputs["endpoint_mapping_validation"], endpoint_validation)
    atomic_write_json(outputs["dryrun_summary"], summary)
    lines = [
        "# M4C-v2-01 Preflight Report",
        "",
        f"- status: {summary['status']}",
        f"- planned fate matrix shape: {summary['planned_fate_matrix_shape']}",
        f"- endpoint count: {summary['endpoint_count']}",
        f"- missing endpoint mappings: {summary['missing_endpoint_mapping_count']}",
        f"- M4C-v2 execution run: {summary['m4c_v2_execution_run']}",
        f"- fate matrix generated: {summary['fate_matrix_generated']}",
        f"- upstream metadata diff count: {summary['upstream_metadata_diff_count']}",
        f"- forbidden downstream diff count: {summary['forbidden_downstream_diff_count']}",
        f"- /ssd output count: {summary['ssd_output_count']}",
        "",
        "## Not Run",
        "- Full M4C-v2 fate propagation was not run.",
        "- Smoke propagation was not run.",
        "- GPCCA, K_gpcca, M4D, barcode, M5, BranchSBM, and Branched NicheFlow were not run.",
    ]
    atomic_write_text(outputs["preflight_report"], "\n".join(lines).rstrip() + "\n")


def write_smoke_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# M4C-v2-02 Smoke Validation Report",
        "",
        f"- status: {summary['status']}",
        f"- smoke subset shape: {summary['fate_matrix_shape']}",
        f"- endpoint count: {summary['endpoint_count']}",
        f"- row-sum max error: {summary['row_sum_max_error']:.6g}",
        f"- invalid entries: {summary['invalid_entry_count']}",
        f"- missing endpoint mappings: {summary['missing_endpoint_mapping_count']}",
        f"- dominant endpoint assigned rows: {summary['dominant_endpoint_assigned_rows']}",
        f"- plasticity finite rows: {summary['plasticity_finite_rows']}",
        f"- upstream metadata diff count: {summary['upstream_metadata_diff_count']}",
        f"- forbidden downstream diff count: {summary['forbidden_downstream_diff_count']}",
        f"- /ssd output count: {summary['ssd_output_count']}",
        "",
        "No full M4C-v2 production fate matrix was written in smoke mode.",
    ]
    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")


def full_report_text(summary: dict[str, Any], propagation_steps: list[dict[str, Any]]) -> str:
    lines = [
        "# M4C-v2-02 Full Fate Propagation Report",
        "",
        "M4C-v2 computed endpoint-anchored Markov fate probabilities using M4A-v2 transitions.",
        "This is not a GPCCA result and does not run K_gpcca, M4D, barcode, M5, or BranchSBM.",
        "",
        "## QC",
        f"- fate matrix shape: {summary['fate_matrix_shape']}",
        f"- endpoint count: {summary['endpoint_count']}",
        f"- row-sum max error: {summary['row_sum_max_error']:.6g}",
        f"- invalid entries: {summary['invalid_entry_count']}",
        f"- missing endpoint mappings: {summary['missing_endpoint_mapping_count']}",
        f"- dominant endpoint assigned rows: {summary['dominant_endpoint_assigned_rows']}",
        f"- plasticity finite rows: {summary['plasticity_finite_rows']}",
        "",
        "## Propagation Steps",
    ]
    for step in propagation_steps:
        lines.append(
            "- "
            f"{step['source_time']} -> {step['target_time']}: "
            f"source_nodes={step['source_nodes']}, nnz={step['transition_nnz']}"
        )
    lines.extend(
        [
            "",
            "## Safety",
            f"- upstream metadata diff count: {summary['upstream_metadata_diff_count']}",
            f"- forbidden downstream diff count: {summary['forbidden_downstream_diff_count']}",
            f"- /ssd output count: {summary['ssd_output_count']}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def next_step_text() -> str:
    return (
        "# M4C-v2 Next Step Recommendation\n\n"
        "Run M4C-v2-03 full M4C-v1 vs M4C-v2 fate-level benchmark and "
        "visualization QC. Do not start M4C-v2 benchmark execution in this task.\n"
    )


def write_full_outputs(
    outputs: dict[str, Path],
    fate: np.ndarray,
    node_summary: pd.DataFrame,
    dominant: pd.DataFrame,
    plasticity: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    endpoint_summary: pd.DataFrame,
    endpoint_mapping: pd.DataFrame,
    summary: dict[str, Any],
    propagation_steps: list[dict[str, Any]],
) -> None:
    atomic_savez(
        outputs["fate_matrix"],
        probabilities=fate,
        global_node_index=np.arange(fate.shape[0], dtype=np.int64),
        terminal_macrostate_ids=endpoint_mapping["raw_terminal_macrostate"].to_numpy(dtype=np.int32),
        terminal_macrostate_labels=endpoint_mapping["raw_terminal_macrostate_label"].astype(str).to_numpy(dtype=object),
        refined_endpoint_ids=endpoint_mapping["refined_endpoint_id"].astype(str).to_numpy(dtype=object),
        refined_endpoint_labels=endpoint_mapping["refined_endpoint_label"].astype(str).to_numpy(dtype=object),
        confidence_tiers=endpoint_mapping["confidence_tier_after_refinement"].astype(str).to_numpy(dtype=object),
    )
    atomic_write_parquet(outputs["node_summary"], node_summary)
    atomic_write_parquet(outputs["dominant_endpoint"], dominant)
    atomic_write_parquet(outputs["plasticity"], plasticity)
    for name, frame in summaries.items():
        if not frame.empty:
            atomic_write_csv(outputs[name], frame)
    atomic_write_csv(outputs["endpoint_composition"], endpoint_summary)
    atomic_write_csv(outputs["qc_summary"], pd.DataFrame([qc_summary_payload(summary)]))
    atomic_write_text(outputs["report"], full_report_text(summary, propagation_steps))
    atomic_write_text(outputs["next_step"], next_step_text())
    inventory = output_inventory(outputs)
    atomic_write_csv(outputs["output_inventory"], inventory)
    atomic_write_csv(outputs["completed_manifest"], inventory)
    atomic_write_csv(outputs["failed_manifest"], pd.DataFrame(columns=["step", "error"]))


def collect_validated_inputs(
    config: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    input_validation = input_validation_frame(paths)
    validate_m4a_v2_qc(paths["input_m4a_v2_qc_summary"], config)
    node_table = pd.read_parquet(paths["input_m4a_v2_node_table"])
    node_table, final_mask = validate_node_table(
        node_table,
        int(config["validation"]["expected_global_nodes"]),
        int(config["validation"]["expected_terminal_nodes"]),
        str(config["fate"]["final_time_label"]),
    )
    endpoint_mapping = validate_endpoint_mapping(
        pd.read_csv(paths["input_endpoint_mapping"]),
        int(config["validation"]["expected_endpoint_count"]),
    )
    endpoint_nodes = pd.read_parquet(paths["input_endpoint_node_annotation"])
    assignments, endpoint_validation = validate_endpoint_assignments(
        endpoint_nodes,
        node_table,
        final_mask,
        endpoint_mapping,
        str(config["fate"]["endpoint_macrostate_column"]),
    )
    p_forward = load_sparse(paths["input_p_forward"])
    p_absorbing = load_sparse(paths["input_p_absorbing"])
    matrix_meta = matrix_metadata_validation(p_forward, p_absorbing, final_mask, config)
    return {
        "input_validation": input_validation,
        "node_table": node_table,
        "final_mask": final_mask,
        "endpoint_mapping": endpoint_mapping,
        "assignments": assignments,
        "endpoint_validation": endpoint_validation,
        "p_forward": p_forward,
        "p_absorbing": p_absorbing,
        "matrix_meta": matrix_meta,
    }


def base_summary(
    mode: str,
    config: dict[str, Any],
    paths: dict[str, Path],
    validated: dict[str, Any],
    start: float,
) -> dict[str, Any]:
    n_nodes = int(config["validation"]["expected_global_nodes"])
    endpoint_count = int(config["validation"]["expected_endpoint_count"])
    missing_endpoint = int(
        (validated["endpoint_validation"].query("check == 'missing_endpoint_mappings'")["observed"].iloc[0])
    )
    return {
        "stage": "M4C-v2-01/02",
        "status": "PASSED",
        "execution_mode": mode,
        "generated_at_utc": utc_now(),
        "runtime_seconds": time.monotonic() - start,
        "output_root": str(paths["output_root"]),
        "reports_dir": str(paths["reports_dir"]),
        "planned_fate_matrix_shape": f"{n_nodes}x{endpoint_count}",
        "endpoint_count": endpoint_count,
        "raw_terminal_endpoint_columns": endpoint_count,
        "unique_refined_endpoint_count": int(validated["endpoint_mapping"]["refined_endpoint_id"].nunique()),
        "merge_candidate_mapping_rows": int(validated["endpoint_mapping"]["refined_endpoint_id"].duplicated(keep=False).sum()),
        "missing_endpoint_mapping_count": missing_endpoint,
        "m4c_v2_execution_run": mode in {"smoke", "full_production"},
        "fate_matrix_generated": mode == "full_production",
        **validated["matrix_meta"],
        **NO_DOWNSTREAM_FLAGS,
    }


def run_dryrun(
    config: dict[str, Any],
    paths: dict[str, Path],
    validated: dict[str, Any],
    safety: dict[str, Any],
    start: float,
) -> dict[str, Any]:
    summary = {
        **base_summary("dry_run", config, paths, validated, start),
        **safety,
    }
    write_dryrun_outputs(
        dryrun_outputs(paths),
        validated["input_validation"],
        validated["endpoint_validation"],
        summary,
    )
    return summary


def run_smoke(
    args: argparse.Namespace,
    config: dict[str, Any],
    paths: dict[str, Path],
    validated: dict[str, Any],
    safety: dict[str, Any],
    start: float,
) -> dict[str, Any]:
    max_sources = int(args.max_sources or config.get("smoke", {}).get("default_max_sources", 5000))
    dtype = np.dtype(str(config["fate"]["probability_dtype"]))
    fate, subset_indices, final_mask, smoke_meta = compute_smoke_fate(
        validated["p_forward"],
        validated["node_table"],
        validated["assignments"],
        validated["endpoint_mapping"],
        max_sources,
        args.max_nodes,
        dtype,
    )
    qc = validate_fate_matrix(
        fate,
        final_mask,
        float(config["validation"]["row_sum_tolerance"]),
    )
    metrics = fate_metrics(fate, validated["endpoint_mapping"])
    finite_plasticity = int(np.isfinite(metrics["normalized_plasticity_entropy"].to_numpy(dtype=float)).sum())
    summary = {
        **base_summary("smoke", config, paths, validated, start),
        **smoke_meta,
        **qc,
        "fate_matrix_shape": f"{fate.shape[0]}x{fate.shape[1]}",
        "invalid_entry_count": int(qc["nan_or_inf_values"] + qc["negative_values"]),
        "dominant_endpoint_assigned_rows": int(metrics["dominant_endpoint"].notna().sum()),
        "plasticity_finite_rows": finite_plasticity,
        "smoke_subset_global_min": int(subset_indices.min()),
        "smoke_subset_global_max": int(subset_indices.max()),
        **safety,
    }
    write_smoke_report(smoke_outputs(paths)["smoke_report"], summary)
    return summary


def run_full(
    config: dict[str, Any],
    paths: dict[str, Path],
    validated: dict[str, Any],
    safety: dict[str, Any],
    start: float,
) -> dict[str, Any]:
    dtype = np.dtype(str(config["fate"]["probability_dtype"]))
    fate, propagation_steps = compute_fate_probabilities(
        validated["p_forward"],
        validated["node_table"],
        validated["assignments"],
        len(validated["endpoint_mapping"]),
        dtype,
    )
    qc = validate_fate_matrix(
        fate,
        validated["final_mask"],
        float(config["validation"]["row_sum_tolerance"]),
    )
    metrics = fate_metrics(fate, validated["endpoint_mapping"])
    neighborhood = read_optional_neighborhood(
        paths["input_neighborhood_annotation"],
        len(validated["node_table"]),
    )
    node_summary = build_node_summary(
        validated["node_table"],
        validated["assignments"],
        validated["endpoint_mapping"],
        metrics,
        neighborhood,
        str(config["fate"]["directionality_evidence_source"]),
        bool(config["fate"].get("barcode_compatible_contract", False)),
    )
    dominant = node_summary[
        [
            "global_node_index",
            "anchor_id",
            "time",
            "time_day",
            "dominant_endpoint",
            "dominant_endpoint_label",
            "dominant_refined_endpoint_id",
            "dominant_refined_endpoint_label",
            "dominant_endpoint_probability",
        ]
    ].copy()
    plasticity = node_summary[
        [
            "global_node_index",
            "plasticity_entropy",
            "normalized_plasticity_entropy",
            "fate_margin_top1_minus_top2",
            "dominant_endpoint_probability",
        ]
    ].copy()
    summaries = {
        "by_time": group_probability_summary(
            node_summary,
            fate,
            ["time_day", "time"],
            validated["endpoint_mapping"],
        ),
        "by_slice": group_probability_summary(
            node_summary,
            fate,
            ["time_day", "time", "slice_id"],
            validated["endpoint_mapping"],
        ),
        "by_mouse": group_probability_summary(
            node_summary,
            fate,
            ["time_day", "time", "mouse_id"],
            validated["endpoint_mapping"],
        ),
        "by_neighborhood": group_probability_summary(
            node_summary,
            fate,
            ["time_day", "time", "leiden_neigh"],
            validated["endpoint_mapping"],
        ),
    }
    endpoint_summary = endpoint_composition_summary(
        fate,
        node_summary,
        validated["assignments"],
        validated["endpoint_mapping"],
    )
    finite_plasticity = int(np.isfinite(node_summary["normalized_plasticity_entropy"].to_numpy(dtype=float)).sum())
    summary = {
        **base_summary("full_production", config, paths, validated, start),
        **qc,
        "fate_matrix_shape": f"{fate.shape[0]}x{fate.shape[1]}",
        "invalid_entry_count": int(qc["nan_or_inf_values"] + qc["negative_values"]),
        "dominant_endpoint_assigned_rows": int(node_summary["dominant_endpoint"].notna().sum()),
        "plasticity_finite_rows": finite_plasticity,
        "endpoint_mass_summaries_generated": int(
            sum(not frame.empty for frame in summaries.values())
        ),
        "propagation_steps": propagation_steps,
        **safety,
    }
    write_full_outputs(
        production_outputs(paths),
        fate,
        node_summary,
        dominant,
        plasticity,
        summaries,
        endpoint_summary,
        validated["endpoint_mapping"],
        summary,
        propagation_steps,
    )
    return summary


def validate_mode(args: argparse.Namespace) -> str:
    if args.dry_run and args.smoke:
        raise ValueError("--dry-run and --smoke are mutually exclusive.")
    if args.dry_run:
        return "dry_run"
    if args.smoke:
        return "smoke"
    return "full_production"


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.monotonic()
    config = load_config(args.config)
    mode = validate_mode(args)
    paths = validate_config(config)
    validate_output_paths(paths, config)
    before_upstream = snapshot(protected_roots(config))
    before_forbidden = snapshot(forbidden_roots(config))
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    if mode == "full_production":
        validate_no_existing_production_outputs(
            production_outputs(paths),
            overwrite=bool(args.overwrite),
        )
    validated = collect_validated_inputs(config, paths)
    provisional_safety = {
        "upstream_metadata_diff_count": 0,
        "upstream_metadata_diffs": [],
        "forbidden_downstream_diff_count": 0,
        "forbidden_downstream_diffs": [],
        "ssd_output_count": count_ssd_outputs(paths["output_root"]),
    }
    if mode == "dry_run":
        summary = run_dryrun(config, paths, validated, provisional_safety, start)
    elif mode == "smoke":
        summary = run_smoke(args, config, paths, validated, provisional_safety, start)
    else:
        summary = run_full(config, paths, validated, provisional_safety, start)

    after_upstream = snapshot(protected_roots(config))
    after_forbidden = snapshot(forbidden_roots(config))
    safety = {
        "upstream_metadata_diff_count": len(diff_snapshot(before_upstream, after_upstream)),
        "upstream_metadata_diffs": diff_snapshot(before_upstream, after_upstream),
        "forbidden_downstream_diff_count": len(diff_snapshot(before_forbidden, after_forbidden)),
        "forbidden_downstream_diffs": diff_snapshot(before_forbidden, after_forbidden),
        "ssd_output_count": count_ssd_outputs(paths["output_root"]),
    }
    if mode == "dry_run":
        summary = run_dryrun(config, paths, validated, safety, start)
    elif mode == "smoke":
        summary.update(safety)
        write_smoke_report(smoke_outputs(paths)["smoke_report"], summary)
    else:
        summary.update(safety)
        atomic_write_csv(
            production_outputs(paths)["qc_summary"],
            pd.DataFrame([qc_summary_payload(summary)]),
        )
        atomic_write_text(
            production_outputs(paths)["report"],
            full_report_text(summary, summary.get("propagation_steps", [])),
        )
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = run(args)
    except Exception as exc:  # noqa: BLE001
        if args.stop_on_error:
            raise
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    return 0 if summary.get("status") == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
