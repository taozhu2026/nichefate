#!/usr/bin/env python
"""Compute M4C Markov fate probabilities by time-layered backward propagation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config


DEFAULT_CONFIG = "configs/m4c_fate_probability.yaml"
NO_DOWNSTREAM_FLAGS = {
    "no_gpcca": True,
    "no_terminal_state_inference": True,
    "no_dense_absorbing_markov_solve": True,
    "no_absorption_probability": True,
    "no_branched_nicheflow_training": True,
    "no_branchsbm_training": True,
    "no_m5": True,
    "no_regulator_analysis": True,
}
REQUIRED_NODE_COLUMNS = {
    "global_node_index",
    "anchor_id",
    "slice_id",
    "anchor_index",
    "time",
    "time_day",
}
NODE_SUMMARY_COLUMNS = [
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
OPTIONAL_COORDINATE_COLUMNS = ["x", "y", "spatial_x", "spatial_y", "center_x", "center_y"]
REQUIRED_FIGURE_NAMES = [
    "m4c_fate_probability_entropy_by_time.png",
    "m4c_normalized_plasticity_by_time.png",
    "m4c_dominant_fate_composition_by_time.png",
    "m4c_dominant_fate_composition_by_slice_heatmap.png",
    "m4c_terminal_macrostate_probability_heatmap_by_time.png",
    "m4c_top1_fate_probability_by_time.png",
    "m4c_fate_margin_by_time.png",
    "m4c_source_time_to_terminal_macrostate_flow.png",
    "m4c_fate_probability_qc_dashboard.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
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


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    np.savez_compressed(tmp, **arrays)
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
    required_sections = {"paths", "fate", "validation", "visualization"}
    missing = sorted(required_sections - set(config))
    if missing:
        raise KeyError(f"M4C config is missing required sections: {missing}")
    fate = config["fate"]
    if fate.get("method") != "time_layered_backward_propagation":
        raise ValueError("M4C supports fate.method=time_layered_backward_propagation only.")
    if fate.get("entropy_base") != "natural":
        raise ValueError("M4C supports fate.entropy_base=natural only.")
    if str(fate.get("probability_dtype")) not in {"float32", "float64"}:
        raise ValueError("fate.probability_dtype must be float32 or float64.")
    if not bool(fate.get("barcode_compatible_contract", False)):
        raise ValueError("M4C requires fate.barcode_compatible_contract=true.")


def infer_final_time(node_table: pd.DataFrame) -> tuple[float, str]:
    if "time_day" not in node_table.columns or "time" not in node_table.columns:
        raise KeyError("Node table must contain time_day and time.")
    max_day = float(node_table["time_day"].astype(float).max())
    labels = sorted(
        node_table.loc[np.isclose(node_table["time_day"].astype(float), max_day), "time"]
        .dropna()
        .astype(str)
        .unique()
    )
    if len(labels) != 1:
        raise ValueError(f"Expected one final time label for max time_day {max_day}, found {labels}.")
    return max_day, labels[0]


def validate_global_node_table(
    node_table: pd.DataFrame,
    expected_global_nodes: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, float, str]:
    missing = sorted(REQUIRED_NODE_COLUMNS - set(node_table.columns))
    if missing:
        raise KeyError(f"M4A node table is missing required columns: {missing}")
    if bool(node_table["global_node_index"].isna().any()):
        raise ValueError("global_node_index contains missing values.")
    if bool(node_table["global_node_index"].duplicated().any()):
        examples = node_table.loc[node_table["global_node_index"].duplicated(), "global_node_index"].head(5).tolist()
        raise ValueError(f"Duplicate global_node_index values: {examples}")
    n_nodes = int(len(node_table))
    if expected_global_nodes is not None and n_nodes != int(expected_global_nodes):
        raise ValueError(f"Expected {expected_global_nodes} global nodes, found {n_nodes}.")
    observed = np.sort(node_table["global_node_index"].to_numpy(dtype=np.int64, copy=True))
    expected = np.arange(n_nodes, dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError("global_node_index must be contiguous from 0 to n_nodes-1.")
    table = node_table.sort_values("global_node_index", kind="mergesort").reset_index(drop=True).copy()
    if not np.array_equal(table["global_node_index"].to_numpy(dtype=np.int64), expected):
        raise ValueError("Node table row order could not be aligned to global_node_index.")
    final_time_day, final_time = infer_final_time(table)
    final_mask = np.isclose(table["time_day"].astype(float), final_time_day) & (
        table["time"].astype(str) == final_time
    )
    if "is_final_time" in table.columns:
        existing = table["is_final_time"].astype(bool).to_numpy()
        if not np.array_equal(existing, final_mask.to_numpy(dtype=bool)):
            raise ValueError("Existing is_final_time column disagrees with max-time_day final-time inference.")
    else:
        table["is_final_time"] = final_mask.to_numpy(dtype=bool)
    return table, final_mask.to_numpy(dtype=bool), final_time_day, final_time


def validate_sparse_transition(matrix: sp.spmatrix, n_nodes: int, fail_on_nan: bool, fail_on_negative: bool) -> sp.csr_matrix:
    csr = matrix.tocsr()
    if csr.shape != (n_nodes, n_nodes):
        raise ValueError(f"P_forward shape {csr.shape} does not match n_nodes x n_nodes ({n_nodes}, {n_nodes}).")
    if fail_on_nan and bool(np.isnan(csr.data).any()):
        raise ValueError("P_forward contains NaN probabilities.")
    if fail_on_negative and bool((csr.data < 0).any()):
        raise ValueError("P_forward contains negative probabilities.")
    return csr


def validate_terminal_assignments(
    node_table: pd.DataFrame,
    final_mask: np.ndarray,
    assignments: pd.DataFrame,
    macrostate_column: str,
    expected_terminal_nodes: int,
    expected_terminal_macrostates: int,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    required = {"global_node_index", macrostate_column}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise KeyError(f"Terminal assignments are missing required columns: {missing}")
    if bool(assignments["global_node_index"].isna().any()):
        raise ValueError("Terminal assignments contain missing global_node_index values.")
    if bool(assignments["global_node_index"].duplicated().any()):
        examples = assignments.loc[assignments["global_node_index"].duplicated(), "global_node_index"].head(5).tolist()
        raise ValueError(f"Terminal assignments contain duplicate final nodes: {examples}")
    assignment_indices_float = assignments["global_node_index"].astype(float).to_numpy()
    assignment_indices = assignment_indices_float.astype(np.int64)
    if not np.allclose(assignment_indices_float, assignment_indices):
        raise ValueError("Terminal assignment global_node_index values must be integers.")
    final_indices = node_table.loc[final_mask, "global_node_index"].to_numpy(dtype=np.int64)
    if len(final_indices) != int(expected_terminal_nodes):
        raise ValueError(f"Expected {expected_terminal_nodes} final nodes, found {len(final_indices)}.")
    extra = np.setdiff1d(assignment_indices, final_indices, assume_unique=False)
    missing_final = np.setdiff1d(final_indices, assignment_indices, assume_unique=False)
    if len(extra):
        raise ValueError(f"Terminal assignments include non-final nodes: {extra[:5].tolist()}")
    if len(missing_final):
        raise ValueError(f"Terminal assignments are missing final nodes: {missing_final[:5].tolist()}")
    if bool(assignments[macrostate_column].isna().any()):
        raise ValueError(f"{macrostate_column} contains missing values.")
    macro_values_float = assignments[macrostate_column].astype(float).to_numpy()
    macro_values = macro_values_float.astype(np.int64)
    if not np.allclose(macro_values_float, macro_values):
        raise ValueError(f"{macrostate_column} must contain integer macrostate IDs.")
    unique_macros = np.sort(np.unique(macro_values))
    expected_macros = np.arange(int(expected_terminal_macrostates), dtype=np.int64)
    if not np.array_equal(unique_macros, expected_macros):
        raise ValueError(
            f"{macrostate_column} must map cleanly to columns 0..{expected_terminal_macrostates - 1}; "
            f"found {unique_macros.tolist()}."
        )
    sorted_assignments = assignments.sort_values("global_node_index", kind="mergesort").reset_index(drop=True).copy()
    sorted_assignments["terminal_macrostate"] = sorted_assignments[macrostate_column].astype(np.int32)
    if "terminal_macrostate_label" not in sorted_assignments.columns:
        sorted_assignments["terminal_macrostate_label"] = [
            f"terminal_macrostate_{value:02d}" for value in sorted_assignments["terminal_macrostate"].to_numpy()
        ]
    labels = (
        sorted_assignments[["terminal_macrostate", "terminal_macrostate_label"]]
        .drop_duplicates()
        .sort_values("terminal_macrostate")
    )
    if len(labels) != int(expected_terminal_macrostates):
        raise ValueError("Terminal macrostate labels do not map one-to-one to macrostate IDs.")
    return sorted_assignments, expected_macros.astype(np.int32), labels["terminal_macrostate_label"].astype(str).tolist()


def validate_terminal_summary(
    terminal_summary: pd.DataFrame,
    macrostate_ids: np.ndarray,
    macrostate_labels: list[str],
    expected_terminal_nodes: int,
    expected_terminal_macrostates: int,
    final_time_day: float,
    final_time: str,
) -> pd.DataFrame:
    required = {"terminal_macrostate_id", "terminal_macrostate_label", "n_nodes"}
    missing = sorted(required - set(terminal_summary.columns))
    if missing:
        raise KeyError(f"Terminal macrostate summary is missing required columns: {missing}")
    if bool(terminal_summary["terminal_macrostate_id"].isna().any()):
        raise ValueError("Terminal macrostate summary contains missing terminal_macrostate_id values.")
    if bool(terminal_summary["terminal_macrostate_id"].duplicated().any()):
        raise ValueError("Terminal macrostate summary contains duplicate terminal_macrostate_id values.")
    summary = terminal_summary.sort_values("terminal_macrostate_id", kind="mergesort").reset_index(drop=True).copy()
    summary_ids = summary["terminal_macrostate_id"].astype(np.int64).to_numpy()
    if len(summary_ids) != int(expected_terminal_macrostates):
        raise ValueError(f"Expected {expected_terminal_macrostates} terminal summary rows, found {len(summary_ids)}.")
    if not np.array_equal(summary_ids, macrostate_ids.astype(np.int64)):
        raise ValueError("Terminal summary macrostate IDs do not match terminal assignments.")
    summary_labels = summary["terminal_macrostate_label"].astype(str).tolist()
    if summary_labels != list(macrostate_labels):
        raise ValueError("Terminal summary labels do not match terminal assignments.")
    node_count = int(summary["n_nodes"].astype(np.int64).sum())
    if node_count != int(expected_terminal_nodes):
        raise ValueError(f"Terminal summary n_nodes sum {node_count} does not match final node count {expected_terminal_nodes}.")
    if "time_day" in summary.columns and not np.allclose(summary["time_day"].astype(float), float(final_time_day)):
        raise ValueError("Terminal summary time_day values do not match inferred final time_day.")
    if "time" in summary.columns and set(summary["time"].astype(str)) != {str(final_time)}:
        raise ValueError("Terminal summary time values do not match inferred final time.")
    return summary


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
        raise ValueError("M4C requires at least two time layers.")
    return rows


def initialize_terminal_fates(
    n_nodes: int,
    n_macrostates: int,
    assignments: pd.DataFrame,
    dtype: np.dtype,
) -> np.ndarray:
    fate = np.zeros((n_nodes, n_macrostates), dtype=dtype)
    indices = assignments["global_node_index"].to_numpy(dtype=np.int64)
    macrostates = assignments["terminal_macrostate"].to_numpy(dtype=np.int64)
    fate[indices, macrostates] = np.array(1.0, dtype=dtype)
    return fate


def compute_fate_probabilities(
    p_forward: sp.csr_matrix,
    node_table: pd.DataFrame,
    assignments: pd.DataFrame,
    n_macrostates: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n_nodes = int(len(node_table))
    fate = initialize_terminal_fates(n_nodes, n_macrostates, assignments, dtype)
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
        values = block.dot(fate).astype(dtype, copy=False)
        fate[indices, :] = values
        propagation_steps.append(
            {
                "source_time": layer["time"],
                "source_time_day": layer["time_day"],
                "source_nodes": int(len(indices)),
                "transition_nnz": int(block.nnz),
            }
        )
    propagation_steps.reverse()
    return fate, propagation_steps


def validate_fate_matrix(
    fate: np.ndarray,
    final_mask: np.ndarray,
    row_sum_tolerance: float,
    fail_on_nan: bool,
    fail_on_negative: bool,
) -> dict[str, Any]:
    if fail_on_nan and bool(np.isnan(fate).any()):
        raise ValueError("Fate probability matrix contains NaN values.")
    if fail_on_negative and bool((fate < 0).any()):
        raise ValueError("Fate probability matrix contains negative probabilities.")
    row_sums = fate.sum(axis=1, dtype=np.float64)
    nonfinal_error = np.abs(row_sums[~final_mask] - 1.0)
    final_rows = fate[final_mask]
    final_row_sums = row_sums[final_mask]
    final_onehot_error = np.minimum(
        np.abs(final_rows - 0.0),
        np.abs(final_rows - 1.0),
    ).max(axis=1)
    final_onehot_pass = np.isclose(final_rows.max(axis=1), 1.0, atol=row_sum_tolerance) & (
        np.isclose(final_row_sums, 1.0, atol=row_sum_tolerance)
    )
    nonfinal_exceed = int((nonfinal_error > row_sum_tolerance).sum())
    final_fail = int((~final_onehot_pass).sum())
    if nonfinal_exceed:
        raise ValueError(f"{nonfinal_exceed} non-final fate rows exceed row_sum_tolerance={row_sum_tolerance}.")
    if final_fail:
        raise ValueError(f"{final_fail} final-time fate rows are not one-hot within tolerance.")
    return {
        "row_sum_tolerance": float(row_sum_tolerance),
        "nonfinal_rows": int((~final_mask).sum()),
        "final_rows": int(final_mask.sum()),
        "nonfinal_row_sum_error_max": float(nonfinal_error.max()) if len(nonfinal_error) else 0.0,
        "nonfinal_row_sum_error_p99": float(np.quantile(nonfinal_error, 0.99)) if len(nonfinal_error) else 0.0,
        "nonfinal_rows_exceed_1e_minus_5": int((nonfinal_error > 1e-5).sum()),
        "nonfinal_rows_exceed_1e_minus_6": int((nonfinal_error > 1e-6).sum()),
        "final_row_sum_error_max": float(np.abs(final_row_sums - 1.0).max()) if len(final_row_sums) else 0.0,
        "final_onehot_error_max": float(final_onehot_error.max()) if len(final_onehot_error) else 0.0,
        "nan_values": int(np.isnan(fate).sum()),
        "negative_values": int((fate < 0).sum()),
    }


def fate_metrics(fate: np.ndarray, macrostate_ids: np.ndarray, macrostate_labels: list[str]) -> pd.DataFrame:
    probs64 = fate.astype(np.float64, copy=False)
    entropy = np.zeros(fate.shape[0], dtype=np.float64)
    positive = probs64 > 0.0
    entropy[positive.any(axis=1)] = -(np.where(positive, probs64 * np.log(np.clip(probs64, 1e-300, None)), 0.0)).sum(
        axis=1
    )[positive.any(axis=1)]
    entropy = np.maximum(entropy, 0.0)
    normalizer = float(np.log(fate.shape[1])) if fate.shape[1] > 1 else 1.0
    normalized_entropy = entropy / normalizer if normalizer else entropy
    normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)
    dominant_col = fate.argmax(axis=1)
    top1 = fate[np.arange(fate.shape[0]), dominant_col].astype(np.float64, copy=False)
    if fate.shape[1] > 1:
        top2 = np.partition(fate, -2, axis=1)[:, -2].astype(np.float64, copy=False)
    else:
        top2 = np.zeros(fate.shape[0], dtype=np.float64)
    macro_ids = macrostate_ids[dominant_col].astype(np.int32, copy=False)
    labels = np.asarray(macrostate_labels, dtype=object)[dominant_col]
    return pd.DataFrame(
        {
            "plasticity_entropy": entropy,
            "normalized_plasticity_entropy": normalized_entropy,
            "dominant_fate": macro_ids,
            "dominant_fate_label": labels,
            "dominant_fate_probability": top1,
            "fate_margin_top1_minus_top2": top1 - top2,
        }
    )


def build_node_summary(
    node_table: pd.DataFrame,
    assignments: pd.DataFrame,
    metrics: pd.DataFrame,
    directionality_evidence_source: str,
    barcode_compatible_contract: bool,
) -> pd.DataFrame:
    table = node_table.copy()
    for column in NODE_SUMMARY_COLUMNS:
        if column not in table.columns:
            table[column] = pd.NA
    coordinate_columns = [column for column in OPTIONAL_COORDINATE_COLUMNS if column in table.columns]
    summary = table[NODE_SUMMARY_COLUMNS + coordinate_columns].copy()
    n_nodes = len(summary)
    terminal_macrostate = np.full(n_nodes, -1, dtype=np.int32)
    terminal_labels = np.full(n_nodes, "non_terminal", dtype=object)
    terminal_indices = assignments["global_node_index"].to_numpy(dtype=np.int64)
    terminal_macrostate[terminal_indices] = assignments["terminal_macrostate"].to_numpy(dtype=np.int32)
    terminal_labels[terminal_indices] = assignments["terminal_macrostate_label"].astype(str).to_numpy()
    summary["terminal_macrostate"] = terminal_macrostate
    summary["terminal_macrostate_label"] = terminal_labels
    for column in metrics.columns:
        summary[column] = metrics[column].to_numpy()
    summary["directionality_evidence_source"] = directionality_evidence_source
    summary["barcode_compatible_contract"] = bool(barcode_compatible_contract)
    if not np.array_equal(summary["global_node_index"].to_numpy(dtype=np.int64), np.arange(n_nodes, dtype=np.int64)):
        raise ValueError("node_summary does not preserve row i == global_node_index i mapping.")
    return summary


def group_probability_summary(
    node_summary: pd.DataFrame,
    fate: np.ndarray,
    group_columns: list[str],
    macrostate_ids: np.ndarray,
    macrostate_labels: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_keys: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in node_summary.groupby(group_keys, sort=True, dropna=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        indices = group["global_node_index"].to_numpy(dtype=np.int64)
        probs = fate[indices, :].astype(np.float64, copy=False)
        sums = probs.sum(axis=0)
        denom = float(sums.sum())
        dominant_counts = group["dominant_fate"].value_counts()
        base = {column: value for column, value in zip(group_columns, key, strict=True)}
        for col_idx, macro_id in enumerate(macrostate_ids):
            dominant_count = int(dominant_counts.get(int(macro_id), 0))
            rows.append(
                {
                    **base,
                    "n_nodes": int(len(group)),
                    "terminal_macrostate": int(macro_id),
                    "terminal_macrostate_label": macrostate_labels[col_idx],
                    "mean_probability": float(probs[:, col_idx].mean()) if len(group) else 0.0,
                    "sum_probability": float(sums[col_idx]),
                    "normalized_mass_fraction": float(sums[col_idx] / denom) if denom else 0.0,
                    "dominant_fate_fraction": float(dominant_count / len(group)) if len(group) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def output_paths(paths: dict[str, Path]) -> dict[str, Path]:
    fate_dir = paths["output_root"] / "fate_probabilities"
    reports_dir = paths["reports_dir"]
    return {
        "fate_matrix": fate_dir / "fate_probability_matrix.npz",
        "node_summary": fate_dir / "fate_probability_node_summary.parquet",
        "by_time": fate_dir / "fate_probability_by_time_summary.csv",
        "by_slice": fate_dir / "fate_probability_by_slice_summary.csv",
        "by_mouse": fate_dir / "fate_probability_by_mouse_summary.csv",
        "report": reports_dir / "m4c_markov_fate_probability_report.md",
        "schema": reports_dir / "m4c_fate_probability_schema.json",
        "qc": reports_dir / "m4c_fate_probability_qc_summary.csv",
    }


def validate_no_forbidden_output_paths(paths: dict[str, Path]) -> None:
    forbidden = ["gpcca", "branched_nicheflow", "branchsbm", "m5", "regulator"]
    offenders = [str(path) for path in paths.values() if any(token in str(path).lower() for token in forbidden)]
    if offenders:
        raise ValueError(f"M4C output paths include forbidden downstream targets: {offenders}")


def metric_quantiles(node_summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (time_day, time_label), group in node_summary.groupby(["time_day", "time"], sort=True, observed=True):
        values = group[metric].to_numpy(dtype=float)
        rows.append(
            {
                "time_day": float(time_day),
                "time": str(time_label),
                "mean": float(np.mean(values)),
                "p25": float(np.quantile(values, 0.25)),
                "p50": float(np.quantile(values, 0.50)),
                "p75": float(np.quantile(values, 0.75)),
            }
        )
    return pd.DataFrame(rows).sort_values(["time_day", "time"]).reset_index(drop=True)


def make_figures(
    figures_dir: Path,
    node_summary: pd.DataFrame,
    by_time: pd.DataFrame,
    by_slice: pd.DataFrame,
    qc: dict[str, Any],
    warning_only: bool,
) -> tuple[list[str], bool]:
    warnings: list[str] = []
    spatial_generated = False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir.mkdir(parents=True, exist_ok=True)

        def plot_metric(metric: str, title: str, path_name: str) -> None:
            frame = metric_quantiles(node_summary, metric)
            x = np.arange(len(frame))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, frame["mean"], marker="o", label="mean")
            ax.fill_between(x, frame["p25"], frame["p75"], alpha=0.25, label="IQR")
            ax.set_xticks(x)
            ax.set_xticklabels(frame["time"].astype(str), rotation=30)
            ax.set_title(title)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(figures_dir / path_name, dpi=140)
            plt.close(fig)

        plot_metric("plasticity_entropy", "Fate probability entropy by time", "m4c_fate_probability_entropy_by_time.png")
        plot_metric(
            "normalized_plasticity_entropy",
            "Normalized plasticity by time",
            "m4c_normalized_plasticity_by_time.png",
        )
        plot_metric("dominant_fate_probability", "Top-1 fate probability by time", "m4c_top1_fate_probability_by_time.png")
        plot_metric("fate_margin_top1_minus_top2", "Fate margin by time", "m4c_fate_margin_by_time.png")

        dominant = by_time.pivot(
            index="time",
            columns="terminal_macrostate_label",
            values="dominant_fate_fraction",
        ).fillna(0.0)
        dominant = dominant.loc[
            by_time[["time_day", "time"]].drop_duplicates().sort_values(["time_day", "time"])["time"].astype(str)
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = np.zeros(len(dominant), dtype=float)
        for label in dominant.columns:
            values = dominant[label].to_numpy(dtype=float)
            ax.bar(dominant.index.astype(str), values, bottom=bottom, label=str(label))
            bottom += values
        ax.set_title("Dominant fate composition by time")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=6, ncol=3)
        fig.tight_layout()
        fig.savefig(figures_dir / "m4c_dominant_fate_composition_by_time.png", dpi=140)
        plt.close(fig)

        slice_heat = by_slice.pivot_table(
            index="slice_id",
            columns="terminal_macrostate_label",
            values="dominant_fate_fraction",
            fill_value=0.0,
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(10, max(5, min(14, 0.18 * len(slice_heat)))))
        im = ax.imshow(slice_heat.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title("Dominant fate fraction by slice")
        ax.set_xticks(np.arange(len(slice_heat.columns)))
        ax.set_xticklabels(slice_heat.columns.astype(str), rotation=45, ha="right", fontsize=7)
        y_step = max(1, len(slice_heat) // 25)
        ax.set_yticks(np.arange(0, len(slice_heat), y_step))
        ax.set_yticklabels(slice_heat.index.astype(str)[::y_step], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.03, label="fraction")
        fig.tight_layout()
        fig.savefig(figures_dir / "m4c_dominant_fate_composition_by_slice_heatmap.png", dpi=140)
        plt.close(fig)

        prob_heat = by_time.pivot(index="time", columns="terminal_macrostate_label", values="mean_probability").fillna(0.0)
        prob_heat = prob_heat.loc[dominant.index]
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(prob_heat.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="magma")
        ax.set_title("Mean terminal macrostate probability by time")
        ax.set_xticks(np.arange(len(prob_heat.columns)))
        ax.set_xticklabels(prob_heat.columns.astype(str), rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(prob_heat.index)))
        ax.set_yticklabels(prob_heat.index.astype(str))
        fig.colorbar(im, ax=ax, fraction=0.04, label="mean probability")
        fig.tight_layout()
        fig.savefig(figures_dir / "m4c_terminal_macrostate_probability_heatmap_by_time.png", dpi=140)
        plt.close(fig)

        flow = by_time.pivot(index="time", columns="terminal_macrostate_label", values="normalized_mass_fraction").fillna(0.0)
        flow = flow.loc[dominant.index]
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(flow.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="cividis")
        ax.set_title("Source time to terminal macrostate flow")
        ax.set_xticks(np.arange(len(flow.columns)))
        ax.set_xticklabels(flow.columns.astype(str), rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(flow.index)))
        ax.set_yticklabels(flow.index.astype(str))
        fig.colorbar(im, ax=ax, fraction=0.04, label="normalized mass fraction")
        fig.tight_layout()
        fig.savefig(figures_dir / "m4c_source_time_to_terminal_macrostate_flow.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].bar(["nodes", "terminal"], [qc["global_nodes"], qc["terminal_nodes"]])
        axes[0, 0].set_title("Node counts")
        axes[0, 1].bar(
            ["max", "p99"],
            [qc["nonfinal_row_sum_error_max"], qc["nonfinal_row_sum_error_p99"]],
        )
        axes[0, 1].set_title("Non-final row-sum error")
        axes[1, 0].bar(["macrostates"], [qc["terminal_macrostates"]])
        axes[1, 0].set_title("Terminal macrostates")
        entropy_frame = metric_quantiles(node_summary, "normalized_plasticity_entropy")
        axes[1, 1].plot(entropy_frame["time"].astype(str), entropy_frame["mean"], marker="o")
        axes[1, 1].tick_params(axis="x", rotation=30)
        axes[1, 1].set_title("Mean normalized plasticity")
        fig.tight_layout()
        fig.savefig(figures_dir / "m4c_fate_probability_qc_dashboard.png", dpi=140)
        plt.close(fig)

        coordinate_pairs = [("x", "y"), ("spatial_x", "spatial_y"), ("center_x", "center_y")]
        for x_col, y_col in coordinate_pairs:
            if x_col in node_summary.columns and y_col in node_summary.columns:
                representative = node_summary.loc[node_summary["is_final_time"].astype(bool)].head(20000)
                if len(representative):
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sc = ax.scatter(
                        representative[x_col],
                        representative[y_col],
                        c=representative["dominant_fate"],
                        s=2,
                        cmap="tab20",
                    )
                    ax.set_title("Representative spatial dominant fate")
                    fig.colorbar(sc, ax=ax, fraction=0.04, label="dominant fate")
                    fig.tight_layout()
                    fig.savefig(figures_dir / "m4c_optional_spatial_dominant_fate_representative.png", dpi=140)
                    plt.close(fig)
                    spatial_generated = True
                break
    except Exception as exc:  # noqa: BLE001
        if not warning_only:
            raise
        warnings.append(f"Figure generation failed after M4C computation/QC passed: {exc}")
    return warnings, spatial_generated


def schema_payload(
    config: dict[str, Any],
    paths: dict[str, Path],
    outputs: dict[str, Path],
    final_time: str,
    final_time_day: float,
    macrostate_ids: np.ndarray,
    macrostate_labels: list[str],
    qc: dict[str, Any],
    propagation_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "m4c_fate_probability_schema_v1",
        "generated_at_utc": utc_now_iso(),
        "method": config["fate"]["method"],
        "computation_semantics": "time-layered DAG-style backward propagation; no dense absorbing Markov solve",
        "input_p_forward": str(paths["p_forward"]),
        "structural_absorbing_p_reference": str(paths["p_absorbing"]),
        "p_absorbing_used_for_computation": False,
        "node_table": str(paths["m4a_node_table"]),
        "terminal_assignments": str(paths["terminal_assignments"]),
        "terminal_summary": str(paths["terminal_summary"]),
        "final_time": final_time,
        "final_time_day": final_time_day,
        "terminal_macrostate_ids": macrostate_ids.astype(int).tolist(),
        "terminal_macrostate_labels": macrostate_labels,
        "fate_probability_matrix": {
            "path": str(outputs["fate_matrix"]),
            "format": "numpy savez_compressed",
            "array_key": "probabilities",
            "shape": [int(qc["global_nodes"]), int(qc["terminal_macrostates"])],
            "dtype": str(config["fate"]["probability_dtype"]),
            "row_alignment": "row i corresponds to global_node_index i",
        },
        "node_summary": {
            "path": str(outputs["node_summary"]),
            "row_alignment": "global_node_index is contiguous and sorted",
        },
        "summaries": {
            "by_time": str(outputs["by_time"]),
            "by_slice": str(outputs["by_slice"]),
            "by_mouse": str(outputs["by_mouse"]),
        },
        "directionality_evidence_source": config["fate"]["directionality_evidence_source"],
        "barcode_compatible_contract": bool(config["fate"]["barcode_compatible_contract"]),
        "propagation_steps": propagation_steps,
        "qc": qc,
        **NO_DOWNSTREAM_FLAGS,
    }


def report_text(
    final_time: str,
    final_time_day: float,
    qc: dict[str, Any],
    by_time: pd.DataFrame,
    outputs: dict[str, Path],
    figure_warnings: list[str],
    spatial_generated: bool,
    runtime_seconds: float,
) -> str:
    dominant = (
        by_time.sort_values(["time_day", "time", "dominant_fate_fraction"], ascending=[True, True, False])
        .groupby(["time_day", "time"], sort=True)
        .head(3)
    )
    lines = [
        "# M4C-01 Markov Fate Probability Computation",
        "",
        "This stage computed pseudo-lineage/time-coupled Markov fate probabilities to the M4B candidate terminal niche macrostates.",
        "The computation used time-layered DAG-style backward propagation and did not compute a dense absorbing Markov solve.",
        "",
        "## Inputs And Semantics",
        f"- final time inferred from max time_day: {final_time} ({final_time_day:g})",
        "- transition object used: P_forward_no_terminal_selfloops",
        "- P_absorbing_terminal_selfloops was retained only as a structural reference and was not used for computation.",
        "- terminal macrostates are candidate terminal niche states from M4B, not final biological fate labels.",
        "- future barcode-aware M3 transition evidence can replace or supplement pseudo-lineage evidence without changing this M4C fate interface.",
        "",
        "## QC",
        f"- global nodes: {qc['global_nodes']}",
        f"- terminal nodes: {qc['terminal_nodes']}",
        f"- terminal macrostates: {qc['terminal_macrostates']}",
        f"- fate matrix shape: {qc['fate_matrix_shape']}",
        f"- non-final row-sum error max: {qc['nonfinal_row_sum_error_max']:.6g}",
        f"- non-final row-sum error p99: {qc['nonfinal_row_sum_error_p99']:.6g}",
        f"- non-final rows exceeding 1e-5: {qc['nonfinal_rows_exceed_1e_minus_5']}",
        f"- final one-hot error max: {qc['final_onehot_error_max']:.6g}",
        "",
        "## Dominant Fate Composition By Time",
    ]
    for row in dominant.to_dict("records"):
        lines.append(
            "- "
            f"{row['time']}: {row['terminal_macrostate_label']} "
            f"dominant_fate_fraction={row['dominant_fate_fraction']:.6g}, "
            f"mean_probability={row['mean_probability']:.6g}"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            *[f"- {key}: {value}" for key, value in outputs.items()],
            "",
            "## Figures",
            *[f"- {outputs['report'].parent / 'figures' / figure_name}" for figure_name in REQUIRED_FIGURE_NAMES],
            f"- optional spatial fate maps generated: {spatial_generated}",
        ]
    )
    if figure_warnings:
        lines.extend(["", "## Figure Warnings", *[f"- {warning}" for warning in figure_warnings]])
    lines.extend(
        [
            "",
            "## Runtime",
            f"- runtime seconds: {runtime_seconds:.3f}",
            "",
            "## Not Run",
            "- GPCCA was not run.",
            "- Terminal-state inference was not run.",
            "- Dense absorbing Markov solve and absorption probabilities were not computed.",
            "- Branched NicheFlow / BranchSBM training was not run.",
            "- M5 and regulator analysis were not run.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    config: dict[str, Any],
    paths: dict[str, Path],
    outputs: dict[str, Path],
    fate: np.ndarray,
    macrostate_ids: np.ndarray,
    macrostate_labels: list[str],
    node_summary: pd.DataFrame,
    by_time: pd.DataFrame,
    by_slice: pd.DataFrame,
    by_mouse: pd.DataFrame,
    qc: dict[str, Any],
    final_time: str,
    final_time_day: float,
    propagation_steps: list[dict[str, Any]],
    figure_warnings: list[str],
    spatial_generated: bool,
    runtime_seconds: float,
) -> None:
    atomic_savez(
        outputs["fate_matrix"],
        probabilities=fate,
        global_node_index=np.arange(fate.shape[0], dtype=np.int64),
        terminal_macrostate_ids=macrostate_ids.astype(np.int32),
        terminal_macrostate_labels=np.asarray(macrostate_labels, dtype=object),
        directionality_evidence_source=np.asarray([config["fate"]["directionality_evidence_source"]], dtype=object),
        barcode_compatible_contract=np.asarray([bool(config["fate"]["barcode_compatible_contract"])]),
    )
    atomic_write_parquet(outputs["node_summary"], node_summary)
    atomic_write_csv(outputs["by_time"], by_time)
    atomic_write_csv(outputs["by_slice"], by_slice)
    atomic_write_csv(outputs["by_mouse"], by_mouse)
    atomic_write_csv(outputs["qc"], pd.DataFrame([qc]))
    atomic_write_json(
        outputs["schema"],
        schema_payload(config, paths, outputs, final_time, final_time_day, macrostate_ids, macrostate_labels, qc, propagation_steps),
    )
    atomic_write_text(
        outputs["report"],
        report_text(final_time, final_time_day, qc, by_time, outputs, figure_warnings, spatial_generated, runtime_seconds),
    )


def run(args: argparse.Namespace) -> int:
    start = time.monotonic()
    config = load_config(args.config)
    validate_config(config)
    paths = configured_paths(config)
    outputs = output_paths(paths)
    validate_no_forbidden_output_paths(outputs)
    for key in ["m4a_node_table", "p_forward", "p_absorbing", "terminal_assignments", "terminal_summary"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing required M4C input paths.{key}: {paths[key]}")

    validation = config["validation"]
    node_table = pd.read_parquet(paths["m4a_node_table"])
    node_table, final_mask, final_time_day, final_time = validate_global_node_table(
        node_table,
        int(validation["expected_global_nodes"]),
    )
    assignments = pd.read_parquet(paths["terminal_assignments"])
    assignments, macrostate_ids, macrostate_labels = validate_terminal_assignments(
        node_table,
        final_mask,
        assignments,
        str(config["fate"]["terminal_macrostate_column"]),
        int(validation["expected_terminal_nodes"]),
        int(validation["expected_terminal_macrostates"]),
    )
    terminal_summary = pd.read_csv(paths["terminal_summary"])
    terminal_summary = validate_terminal_summary(
        terminal_summary,
        macrostate_ids,
        macrostate_labels,
        int(validation["expected_terminal_nodes"]),
        int(validation["expected_terminal_macrostates"]),
        final_time_day,
        final_time,
    )
    p_forward = validate_sparse_transition(
        sp.load_npz(paths["p_forward"]),
        len(node_table),
        bool(validation.get("fail_on_nan", True)),
        bool(validation.get("fail_on_negative_probability", True)),
    )
    final_outgoing_nnz = int(p_forward[final_mask, :].nnz)
    if final_outgoing_nnz:
        raise ValueError(f"P_forward final-time rows must have no outgoing transitions; found {final_outgoing_nnz}.")
    dtype = np.dtype(str(config["fate"]["probability_dtype"]))
    fate, propagation_steps = compute_fate_probabilities(
        p_forward,
        node_table,
        assignments,
        int(validation["expected_terminal_macrostates"]),
        dtype,
    )
    fate_qc = validate_fate_matrix(
        fate,
        final_mask,
        float(validation["row_sum_tolerance"]),
        bool(validation.get("fail_on_nan", True)),
        bool(validation.get("fail_on_negative_probability", True)),
    )
    metrics = fate_metrics(fate, macrostate_ids, macrostate_labels)
    if bool(metrics[["plasticity_entropy", "normalized_plasticity_entropy"]].isna().any().any()):
        raise ValueError("Plasticity entropy metrics contain missing values.")
    if not np.isfinite(metrics[["plasticity_entropy", "normalized_plasticity_entropy"]].to_numpy()).all():
        raise ValueError("Plasticity entropy metrics are not finite.")
    node_summary = build_node_summary(
        node_table,
        assignments,
        metrics,
        str(config["fate"]["directionality_evidence_source"]),
        bool(config["fate"]["barcode_compatible_contract"]),
    )
    by_time = group_probability_summary(
        node_summary,
        fate,
        ["time_day", "time"],
        macrostate_ids,
        macrostate_labels,
    )
    by_slice = group_probability_summary(
        node_summary,
        fate,
        ["time_day", "time", "slice_id"],
        macrostate_ids,
        macrostate_labels,
    )
    by_mouse = group_probability_summary(
        node_summary,
        fate,
        ["time_day", "time", "mouse_id"],
        macrostate_ids,
        macrostate_labels,
    )
    qc = {
        "schema_version": "m4c_fate_probability_qc_v1",
        "generated_at_utc": utc_now_iso(),
        "global_nodes": int(len(node_table)),
        "terminal_nodes": int(final_mask.sum()),
        "terminal_macrostates": int(len(macrostate_ids)),
        "terminal_summary_rows": int(len(terminal_summary)),
        "terminal_summary_n_nodes": int(terminal_summary["n_nodes"].astype(np.int64).sum()),
        "fate_matrix_shape": f"{fate.shape[0]}x{fate.shape[1]}",
        "p_forward_shape": f"{p_forward.shape[0]}x{p_forward.shape[1]}",
        "p_forward_nnz": int(p_forward.nnz),
        "p_forward_final_time_outgoing_nnz": final_outgoing_nnz,
        "global_node_index_contiguous": True,
        "fate_row_i_matches_global_node_index_i": True,
        "directionality_evidence_source": str(config["fate"]["directionality_evidence_source"]),
        "barcode_compatible_contract": bool(config["fate"]["barcode_compatible_contract"]),
        "optional_spatial_maps_generated": False,
        **fate_qc,
        **NO_DOWNSTREAM_FLAGS,
    }
    figure_warnings: list[str] = []
    spatial_generated = False
    if bool(config["visualization"].get("make_figures", True)):
        figure_warnings, spatial_generated = make_figures(
            paths["figures_dir"],
            node_summary,
            by_time,
            by_slice,
            qc,
            bool(config["visualization"].get("figure_failure_is_warning", True)),
        )
    qc["figure_warnings"] = int(len(figure_warnings))
    qc["optional_spatial_maps_generated"] = bool(spatial_generated)
    runtime = time.monotonic() - start
    qc["runtime_seconds"] = float(runtime)
    write_outputs(
        config,
        paths,
        outputs,
        fate,
        macrostate_ids,
        macrostate_labels,
        node_summary,
        by_time,
        by_slice,
        by_mouse,
        qc,
        final_time,
        final_time_day,
        propagation_steps,
        figure_warnings,
        spatial_generated,
        runtime,
    )
    print("M4C_01_MARKOV_FATE_PROBABILITY_COMPLETE")
    print(f"GLOBAL_NODES {len(node_table)}")
    print(f"FINAL_TIME {final_time}")
    print(f"FINAL_TIME_DAY {final_time_day:g}")
    print(f"TERMINAL_NODES {int(final_mask.sum())}")
    print(f"TERMINAL_MACROSTATES {len(macrostate_ids)}")
    print(f"FATE_MATRIX_SHAPE {fate.shape[0]}x{fate.shape[1]}")
    print(f"NONFINAL_ROW_SUM_ERROR_MAX {qc['nonfinal_row_sum_error_max']:.6g}")
    print(f"OPTIONAL_SPATIAL_MAPS_GENERATED {spatial_generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
