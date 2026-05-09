#!/usr/bin/env python
"""Benchmark M4C-v1 vs M4C-v2 fate probabilities and visualization QC."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ROOT = Path("/home/zhutao/scratch/nichefate")
DEFAULT_OUTPUT_ROOT = ROOT / "m4c_v2_benchmark"
M4C_V1_ROOT = ROOT / "m4c"
M4C_V2_ROOT = ROOT / "m4c_v2"
M4E_ROOT = ROOT / "m4e"

EXPECTED_NODES = 1_439_542
EXPECTED_ENDPOINTS = 12
ROW_ATOL = 1e-5
TIMES = ["D0", "D3", "D9", "D21", "D35"]
MASS_SHIFT_WARN = 0.10
GROUP_SHIFT_WARN = 0.15
ENDPOINT_COLLAPSE_FRACTION = 0.70
LOW_SIZE_INFLATION_WARN = 0.05
SPEARMAN_SAMPLE_ROWS = 100_000

PROTECTED_ROOTS = [
    ROOT / "m3",
    ROOT / "m3_v2",
    ROOT / "m4a",
    ROOT / "m4a_v2",
    ROOT / "m4b",
    ROOT / "m4c",
    ROOT / "m4c_v2",
]
FORBIDDEN_DOWNSTREAM_ROOTS = [
    ROOT / "m4c_v2_benchmark" / "gpcca",
    ROOT / "m4c_v2_benchmark" / "pygpcca",
    ROOT / "m4c_v2_benchmark" / "k_gpcca",
    ROOT / "m4c_v2_benchmark" / "barcode",
    ROOT / "m4c_v2_benchmark" / "m5",
    ROOT / "m4c_v2_benchmark" / "branchsbm",
    ROOT / "m4c_v2" / "gpcca",
    ROOT / "m4c_v2" / "pygpcca",
    ROOT / "m4c_v2" / "k_gpcca",
    ROOT / "m4c_v2" / "barcode",
    ROOT / "m4c_v2" / "m5",
    ROOT / "m4c_v2" / "branchsbm",
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
        right_resolved,
        left_resolved,
    )


def reject_ssd(path: Path) -> None:
    path = resolved(path)
    if path == Path("/ssd") or Path("/ssd") in path.parents:
        raise ValueError(f"Refusing /ssd output path: {path}")


def validate_output_root(output_root: Path) -> Path:
    output_root = resolved(output_root)
    reject_ssd(output_root)
    for protected in PROTECTED_ROOTS:
        if paths_overlap(output_root, protected):
            raise ValueError(f"Output root overlaps protected root {protected}: {output_root}")
    for forbidden in FORBIDDEN_DOWNSTREAM_ROOTS:
        if is_relative_to(output_root, forbidden):
            raise ValueError(f"Output root falls under forbidden root {forbidden}: {output_root}")
    return output_root


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    root = validate_output_root(output_root)
    paths = {
        "root": root,
        "reports": root / "reports",
        "figures": root / "reports" / "figures",
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


def count_ssd_outputs(output_root: Path) -> int:
    if not output_root.exists():
        return 0
    return int(sum(str(path.resolve()).startswith("/ssd/") for path in output_root.rglob("*")))


def required_input_paths() -> dict[str, Path]:
    return {
        "m4c_v1_fate_matrix": M4C_V1_ROOT / "fate_probabilities" / "fate_probability_matrix.npz",
        "m4c_v1_node_summary": M4C_V1_ROOT / "fate_probabilities" / "fate_probability_node_summary.parquet",
        "m4c_v2_fate_matrix": M4C_V2_ROOT / "fate_probabilities" / "fate_probability_matrix_v2.npz",
        "m4c_v2_node_summary": M4C_V2_ROOT / "fate_probabilities" / "node_fate_summary_v2.parquet",
        "m4c_v2_qc_summary": M4C_V2_ROOT / "reports" / "m4c_v2_02_qc_summary.csv",
        "endpoint_mapping": M4E_ROOT / "endpoint_refinement" / "refined_endpoint_mapping.csv",
        "neighborhood_annotation": M4E_ROOT / "neighborhood_annotation" / "node_neighborhood_annotation.parquet",
        "m4a_v2_benchmark_summary": ROOT / "m4a_v2_benchmark" / "m4a_v2_benchmark_summary.json",
        "m3_v2_benchmark_summary": ROOT / "m3_v2_benchmark" / "m3_v1_vs_v2_edge_benchmark_summary.json",
    }


def validate_required_inputs() -> pd.DataFrame:
    rows = []
    for name, path in required_input_paths().items():
        exists = path.is_file()
        rows.append(
            {
                "input_name": name,
                "path": str(path),
                "exists": bool(exists),
                "bytes": int(path.stat().st_size) if exists else 0,
                "status": "PASS" if exists and path.stat().st_size > 0 else "FAIL",
            }
        )
    frame = pd.DataFrame(rows)
    failed = frame.loc[frame["status"] != "PASS", "input_name"].tolist()
    if failed:
        raise FileNotFoundError(f"Missing or empty M4C benchmark inputs: {failed}")
    return frame


def load_fate_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.keys()}


def row_entropy(probabilities: np.ndarray) -> np.ndarray:
    probs64 = probabilities.astype(np.float64, copy=False)
    positive = probs64 > 0.0
    terms = np.where(positive, probs64 * np.log(np.clip(probs64, 1e-300, None)), 0.0)
    entropy = -terms.sum(axis=1)
    return np.maximum(entropy, 0.0)


def top1(probabilities: np.ndarray) -> np.ndarray:
    return probabilities.max(axis=1).astype(np.float64, copy=False)


def row_sum_qc(probabilities: np.ndarray, label: str) -> dict[str, Any]:
    row_sums = probabilities.sum(axis=1, dtype=np.float64)
    errors = np.abs(row_sums - 1.0)
    nonfinite = int((~np.isfinite(probabilities)).sum())
    negative = int((probabilities < 0).sum())
    return {
        f"{label}_row_sum_max_error": float(errors.max()),
        f"{label}_row_sum_p99_error": float(np.quantile(errors, 0.99)),
        f"{label}_rows_exceeding_tolerance": int((errors > ROW_ATOL).sum()),
        f"{label}_nonfinite_values": nonfinite,
        f"{label}_negative_values": negative,
    }


def jensen_shannon_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left64 = left.astype(np.float64, copy=False)
    right64 = right.astype(np.float64, copy=False)
    midpoint = 0.5 * (left64 + right64)
    left_term = np.zeros_like(left64)
    right_term = np.zeros_like(right64)
    left_mask = left64 > 0.0
    right_mask = right64 > 0.0
    left_term[left_mask] = left64[left_mask] * np.log(
        np.clip(left64[left_mask] / midpoint[left_mask], 1e-300, None)
    )
    right_term[right_mask] = right64[right_mask] * np.log(
        np.clip(right64[right_mask] / midpoint[right_mask], 1e-300, None)
    )
    return 0.5 * (left_term.sum(axis=1) + right_term.sum(axis=1))


def row_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left64 = left.astype(np.float64, copy=False)
    right64 = right.astype(np.float64, copy=False)
    left_centered = left64 - left64.mean(axis=1, keepdims=True)
    right_centered = right64 - right64.mean(axis=1, keepdims=True)
    numerator = (left_centered * right_centered).sum(axis=1)
    denominator = np.sqrt((left_centered**2).sum(axis=1) * (right_centered**2).sum(axis=1))
    equal_constant = np.isclose(denominator, 0.0) & np.isclose(left64, right64).all(axis=1)
    corr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0)
    corr[equal_constant] = 1.0
    return corr


def rank_rows(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    row_indices = np.arange(values.shape[0])[:, None]
    ranks[row_indices, order] = np.arange(values.shape[1], dtype=np.float64)
    return ranks


def sampled_spearman(left: np.ndarray, right: np.ndarray, sample_rows: int = SPEARMAN_SAMPLE_ROWS) -> float:
    n_rows = left.shape[0]
    if n_rows <= sample_rows:
        indices = np.arange(n_rows)
    else:
        indices = np.linspace(0, n_rows - 1, sample_rows, dtype=np.int64)
    corr = row_pearson(rank_rows(left[indices]), rank_rows(right[indices]))
    return float(np.mean(corr))


def refined_column_groups(endpoint_mapping: pd.DataFrame) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for _, row in endpoint_mapping.iterrows():
        refined_id = str(row["refined_endpoint_id"])
        raw_id = int(row["raw_terminal_macrostate"])
        groups.setdefault(refined_id, []).append(raw_id)
    return groups


def raw_endpoint_mass(
    probabilities: np.ndarray,
    endpoint_mapping: pd.DataFrame,
    version: str,
) -> pd.DataFrame:
    masses = probabilities.sum(axis=0, dtype=np.float64)
    total = float(masses.sum())
    rows = []
    for _, endpoint in endpoint_mapping.iterrows():
        raw_id = int(endpoint["raw_terminal_macrostate"])
        rows.append(
            {
                "version": version,
                "terminal_macrostate": raw_id,
                "terminal_macrostate_label": str(endpoint["raw_terminal_macrostate_label"]),
                "refined_endpoint_id": str(endpoint["refined_endpoint_id"]),
                "refined_endpoint_label": str(endpoint["refined_endpoint_label"]),
                "confidence_tier_after_refinement": str(endpoint["confidence_tier_after_refinement"]),
                "mass": float(masses[raw_id]),
                "mass_fraction": float(masses[raw_id] / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def refined_endpoint_mass(
    probabilities: np.ndarray,
    endpoint_mapping: pd.DataFrame,
    version: str,
) -> pd.DataFrame:
    masses = probabilities.sum(axis=0, dtype=np.float64)
    total = float(masses.sum())
    rows = []
    for refined_id, columns in refined_column_groups(endpoint_mapping).items():
        subset = endpoint_mapping.loc[endpoint_mapping["refined_endpoint_id"].astype(str) == refined_id]
        mass = float(masses[columns].sum())
        rows.append(
            {
                "version": version,
                "refined_endpoint_id": refined_id,
                "refined_endpoint_label": str(subset["refined_endpoint_label"].iloc[0]),
                "raw_terminal_columns": ",".join(map(str, columns)),
                "mass": mass,
                "mass_fraction": float(mass / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def endpoint_mass_comparison(
    v1_prob: np.ndarray,
    v2_prob: np.ndarray,
    endpoint_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_v1 = raw_endpoint_mass(v1_prob, endpoint_mapping, "M4C-v1")
    raw_v2 = raw_endpoint_mass(v2_prob, endpoint_mapping, "M4C-v2")
    raw = raw_v1.merge(
        raw_v2,
        on=[
            "terminal_macrostate",
            "terminal_macrostate_label",
            "refined_endpoint_id",
            "refined_endpoint_label",
            "confidence_tier_after_refinement",
        ],
        suffixes=("_v1", "_v2"),
    )
    raw["mass_delta_v2_minus_v1"] = raw["mass_v2"] - raw["mass_v1"]
    raw["mass_fraction_delta_v2_minus_v1"] = raw["mass_fraction_v2"] - raw["mass_fraction_v1"]
    refined_v1 = refined_endpoint_mass(v1_prob, endpoint_mapping, "M4C-v1")
    refined_v2 = refined_endpoint_mass(v2_prob, endpoint_mapping, "M4C-v2")
    refined = refined_v1.merge(
        refined_v2,
        on=["refined_endpoint_id", "refined_endpoint_label", "raw_terminal_columns"],
        suffixes=("_v1", "_v2"),
    )
    refined["mass_delta_v2_minus_v1"] = refined["mass_v2"] - refined["mass_v1"]
    refined["mass_fraction_delta_v2_minus_v1"] = refined["mass_fraction_v2"] - refined["mass_fraction_v1"]
    return raw, refined


def build_node_metrics(
    v1_prob: np.ndarray,
    v2_prob: np.ndarray,
    node_table: pd.DataFrame,
    endpoint_mapping: pd.DataFrame,
) -> pd.DataFrame:
    entropy_v1 = row_entropy(v1_prob)
    entropy_v2 = row_entropy(v2_prob)
    top1_v1 = top1(v1_prob)
    top1_v2 = top1(v2_prob)
    dominant_v1 = v1_prob.argmax(axis=1).astype(np.int16)
    dominant_v2 = v2_prob.argmax(axis=1).astype(np.int16)
    refined_lookup = endpoint_mapping.sort_values("raw_terminal_macrostate")["refined_endpoint_id"].astype(str).to_numpy()
    refined_v1 = refined_lookup[dominant_v1]
    refined_v2 = refined_lookup[dominant_v2]
    normalizer = float(np.log(v1_prob.shape[1]))
    metrics = node_table[
        [
            "global_node_index",
            "time",
            "time_day",
            "slice_id",
            "mouse_id",
            "leiden_neigh",
            "cell_type_l3",
            "x",
            "y",
        ]
    ].copy()
    metrics["dominant_endpoint_v1"] = dominant_v1
    metrics["dominant_endpoint_v2"] = dominant_v2
    metrics["dominant_refined_endpoint_v1"] = refined_v1
    metrics["dominant_refined_endpoint_v2"] = refined_v2
    metrics["dominant_endpoint_agreement"] = dominant_v1 == dominant_v2
    metrics["dominant_refined_endpoint_agreement"] = refined_v1 == refined_v2
    metrics["entropy_v1"] = entropy_v1
    metrics["entropy_v2"] = entropy_v2
    metrics["entropy_delta_v2_minus_v1"] = entropy_v2 - entropy_v1
    metrics["normalized_plasticity_v1"] = entropy_v1 / normalizer
    metrics["normalized_plasticity_v2"] = entropy_v2 / normalizer
    metrics["plasticity_delta_v2_minus_v1"] = metrics["normalized_plasticity_v2"] - metrics["normalized_plasticity_v1"]
    metrics["top1_probability_v1"] = top1_v1
    metrics["top1_probability_v2"] = top1_v2
    metrics["top1_delta_v2_minus_v1"] = top1_v2 - top1_v1
    metrics["js_divergence"] = jensen_shannon_rows(v1_prob, v2_prob)
    metrics["pearson_correlation"] = row_pearson(v1_prob, v2_prob)
    return metrics


def validate_m4c_v2_production(qc_path: Path, v2_payload: dict[str, Any], node_v2: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    qc = pd.read_csv(qc_path)
    if qc.empty:
        raise ValueError("M4C-v2 QC summary is empty.")
    row = qc.iloc[0].to_dict()
    probabilities = v2_payload["probabilities"]
    checks = [
        ("fate_matrix_shape", probabilities.shape == (EXPECTED_NODES, EXPECTED_ENDPOINTS), str(probabilities.shape)),
        ("node_summary_rows", len(node_v2) == EXPECTED_NODES, len(node_v2)),
        ("dominant_endpoint_assigned_rows", int(row.get("dominant_endpoint_assigned_rows", -1)) == EXPECTED_NODES, row.get("dominant_endpoint_assigned_rows")),
        ("plasticity_finite_rows", int(row.get("plasticity_finite_rows", -1)) == EXPECTED_NODES, row.get("plasticity_finite_rows")),
        ("row_sum_max_error", float(row.get("row_sum_max_error", 1.0)) <= ROW_ATOL, row.get("row_sum_max_error")),
        ("invalid_entries", int(row.get("invalid_entry_count", -1)) == 0, row.get("invalid_entry_count")),
        ("missing_endpoint_mappings", int(row.get("missing_endpoint_mapping_count", -1)) == 0, row.get("missing_endpoint_mapping_count")),
        ("raw_terminal_columns", int(row.get("raw_terminal_endpoint_columns", -1)) == EXPECTED_ENDPOINTS, row.get("raw_terminal_endpoint_columns")),
        ("unique_refined_endpoint_ids", int(row.get("unique_refined_endpoint_count", -1)) == 11, row.get("unique_refined_endpoint_count")),
        ("upstream_metadata_diff", int(row.get("upstream_metadata_diff_count", -1)) == 0, row.get("upstream_metadata_diff_count")),
        ("forbidden_downstream_diff", int(row.get("forbidden_downstream_diff_count", -1)) == 0, row.get("forbidden_downstream_diff_count")),
        ("ssd_output_count", int(row.get("ssd_output_count", -1)) == 0, row.get("ssd_output_count")),
    ]
    frame = pd.DataFrame(
        [
            {
                "check": name,
                "observed": observed,
                "status": "PASS" if passed else "FAIL",
            }
            for name, passed, observed in checks
        ]
    )
    failed = frame.loc[frame["status"] != "PASS", "check"].tolist()
    if failed:
        raise ValueError(f"M4C-v2 production QC failed: {failed}")
    summary = {
        "m4c_v2_qc_status": "PASS",
        "fate_matrix_shape": f"{probabilities.shape[0]}x{probabilities.shape[1]}",
        "endpoint_count": int(probabilities.shape[1]),
        "unique_refined_endpoint_count": int(row.get("unique_refined_endpoint_count", 11)),
        "row_sum_max_error": float(row.get("row_sum_max_error")),
        "invalid_entry_count": int(row.get("invalid_entry_count")),
        "missing_endpoint_mapping_count": int(row.get("missing_endpoint_mapping_count")),
    }
    return frame, summary


def endpoint_mapping_check(v1_payload: dict[str, Any], v2_payload: dict[str, Any], endpoint_mapping: pd.DataFrame) -> pd.DataFrame:
    v1_ids = v1_payload["terminal_macrostate_ids"].astype(int)
    v2_ids = v2_payload["terminal_macrostate_ids"].astype(int)
    rows = []
    for _, endpoint in endpoint_mapping.sort_values("raw_terminal_macrostate").iterrows():
        raw_id = int(endpoint["raw_terminal_macrostate"])
        rows.append(
            {
                "terminal_macrostate": raw_id,
                "v1_column_present": bool(raw_id in set(v1_ids.tolist())),
                "v2_column_present": bool(raw_id in set(v2_ids.tolist())),
                "column_index_v1": int(np.where(v1_ids == raw_id)[0][0]) if raw_id in set(v1_ids.tolist()) else -1,
                "column_index_v2": int(np.where(v2_ids == raw_id)[0][0]) if raw_id in set(v2_ids.tolist()) else -1,
                "raw_terminal_macrostate_label": str(endpoint["raw_terminal_macrostate_label"]),
                "refined_endpoint_id": str(endpoint["refined_endpoint_id"]),
                "refined_endpoint_label": str(endpoint["refined_endpoint_label"]),
                "confidence_tier_after_refinement": str(endpoint["confidence_tier_after_refinement"]),
                "status": "PASS" if raw_id in set(v1_ids.tolist()) and raw_id in set(v2_ids.tolist()) else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def validate_comparability(
    v1_payload: dict[str, Any],
    v2_payload: dict[str, Any],
    node_v1: pd.DataFrame,
    node_v2: pd.DataFrame,
    endpoint_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    index_match = np.array_equal(v1_payload["global_node_index"], v2_payload["global_node_index"])
    node_index_match = np.array_equal(
        node_v1["global_node_index"].to_numpy(dtype=np.int64),
        node_v2["global_node_index"].to_numpy(dtype=np.int64),
    )
    anchor_match = np.array_equal(node_v1["anchor_id"].astype(str).to_numpy(), node_v2["anchor_id"].astype(str).to_numpy())
    endpoint_check = endpoint_mapping_check(v1_payload, v2_payload, endpoint_mapping)
    checks = [
        ("matrix_global_node_index_order", index_match, "global_node_index arrays"),
        ("node_summary_global_node_index_order", node_index_match, "node summary order"),
        ("anchor_id_order", anchor_match, "anchor_id arrays"),
        ("endpoint_columns_explicitly_mappable", bool((endpoint_check["status"] == "PASS").all()), "raw endpoint IDs"),
        ("time_metadata_joinable", {"time", "time_day"} <= set(node_v2.columns), "time,time_day"),
        ("slice_mouse_metadata_joinable", {"slice_id", "mouse_id"} <= set(node_v2.columns), "slice_id,mouse_id"),
        ("neighborhood_metadata_joinable", "leiden_neigh" in node_v2.columns, "leiden_neigh"),
    ]
    frame = pd.DataFrame(
        [{"check": name, "observed": observed, "status": "PASS" if passed else "FAIL"} for name, passed, observed in checks]
    )
    failed = frame.loc[frame["status"] != "PASS", "check"].tolist()
    if failed:
        raise ValueError(f"M4C-v1/v2 comparability failed: {failed}")
    return endpoint_check, {"comparability_status": "PASS"}


def global_benchmark(
    v1_prob: np.ndarray,
    v2_prob: np.ndarray,
    metrics: pd.DataFrame,
    endpoint_mass: pd.DataFrame,
    refined_mass: pd.DataFrame,
) -> pd.DataFrame:
    v1_qc = row_sum_qc(v1_prob, "v1")
    v2_qc = row_sum_qc(v2_prob, "v2")
    return pd.DataFrame(
        [
            {
                "comparison_scope": "global",
                "nodes": int(v1_prob.shape[0]),
                "endpoint_count": int(v1_prob.shape[1]),
                **v1_qc,
                **v2_qc,
                "dominant_endpoint_agreement": float(metrics["dominant_endpoint_agreement"].mean()),
                "dominant_refined_endpoint_agreement": float(metrics["dominant_refined_endpoint_agreement"].mean()),
                "entropy_mean_v1": float(metrics["entropy_v1"].mean()),
                "entropy_mean_v2": float(metrics["entropy_v2"].mean()),
                "entropy_delta_v2_minus_v1": float(metrics["entropy_delta_v2_minus_v1"].mean()),
                "normalized_plasticity_mean_v1": float(metrics["normalized_plasticity_v1"].mean()),
                "normalized_plasticity_mean_v2": float(metrics["normalized_plasticity_v2"].mean()),
                "normalized_plasticity_delta_v2_minus_v1": float(metrics["plasticity_delta_v2_minus_v1"].mean()),
                "top1_mean_v1": float(metrics["top1_probability_v1"].mean()),
                "top1_mean_v2": float(metrics["top1_probability_v2"].mean()),
                "top1_delta_v2_minus_v1": float(metrics["top1_delta_v2_minus_v1"].mean()),
                "js_divergence_mean": float(metrics["js_divergence"].mean()),
                "js_divergence_p95": float(metrics["js_divergence"].quantile(0.95)),
                "pearson_correlation_mean": float(metrics["pearson_correlation"].mean()),
                "spearman_correlation_sampled_mean": sampled_spearman(v1_prob, v2_prob),
                "max_raw_endpoint_mass_fraction_v2": float(endpoint_mass["mass_fraction_v2"].max()),
                "max_raw_endpoint_mass_fraction_abs_delta": float(endpoint_mass["mass_fraction_delta_v2_minus_v1"].abs().max()),
                "max_refined_endpoint_mass_fraction_abs_delta": float(refined_mass["mass_fraction_delta_v2_minus_v1"].abs().max()),
            }
        ]
    )


def group_mass_shift(
    v1_prob: np.ndarray,
    v2_prob: np.ndarray,
    indices: np.ndarray,
    endpoint_mapping: pd.DataFrame,
) -> tuple[float, float]:
    v1_mass = v1_prob[indices].sum(axis=0, dtype=np.float64)
    v2_mass = v2_prob[indices].sum(axis=0, dtype=np.float64)
    denom = float(len(indices))
    raw_delta = np.abs((v2_mass / denom) - (v1_mass / denom)).max() if denom else 0.0
    refined_deltas = []
    for columns in refined_column_groups(endpoint_mapping).values():
        refined_deltas.append(abs(float(v2_mass[columns].sum() - v1_mass[columns].sum()) / denom) if denom else 0.0)
    return float(raw_delta), float(max(refined_deltas) if refined_deltas else 0.0)


def stratified_benchmark(
    metrics: pd.DataFrame,
    v1_prob: np.ndarray,
    v2_prob: np.ndarray,
    endpoint_mapping: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    if any(column not in metrics.columns for column in group_columns):
        return pd.DataFrame()
    rows = []
    group_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in metrics.groupby(group_key, sort=True, dropna=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        indices = group.index.to_numpy(dtype=np.int64)
        raw_shift, refined_shift = group_mass_shift(v1_prob, v2_prob, indices, endpoint_mapping)
        row = {column: value for column, value in zip(group_columns, key, strict=True)}
        row.update(
            {
                "n_nodes": int(len(group)),
                "dominant_endpoint_agreement": float(group["dominant_endpoint_agreement"].mean()),
                "dominant_refined_endpoint_agreement": float(group["dominant_refined_endpoint_agreement"].mean()),
                "entropy_delta_v2_minus_v1": float(group["entropy_delta_v2_minus_v1"].mean()),
                "plasticity_delta_v2_minus_v1": float(group["plasticity_delta_v2_minus_v1"].mean()),
                "top1_delta_v2_minus_v1": float(group["top1_delta_v2_minus_v1"].mean()),
                "js_divergence_mean": float(group["js_divergence"].mean()),
                "endpoint_mass_shift_max_abs": raw_shift,
                "refined_endpoint_mass_shift_max_abs": refined_shift,
                "slice_mouse_artifact_flag": bool(raw_shift >= GROUP_SHIFT_WARN),
                "neighborhood_artifact_flag": bool(raw_shift >= GROUP_SHIFT_WARN),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def endpoint_shift_flags(endpoint_mass: pd.DataFrame, refined_mass: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in endpoint_mass.iterrows():
        tier = str(row["confidence_tier_after_refinement"])
        delta = float(row["mass_fraction_delta_v2_minus_v1"])
        abs_delta = abs(delta)
        flags = []
        if abs_delta >= MASS_SHIFT_WARN:
            flags.append("large_mass_shift")
        if "slice" in tier or "mouse" in tier:
            flags.append("slice_mouse_associated_endpoint")
            if delta >= GROUP_SHIFT_WARN:
                flags.append("slice_mouse_associated_expansion")
        if ("rare" in tier or "low" in tier or "merge_candidate" in tier) and delta >= LOW_SIZE_INFLATION_WARN:
            flags.append("low_size_endpoint_inflation")
        rows.append(
            {
                **row.to_dict(),
                "abs_mass_fraction_delta": abs_delta,
                "flag_category": ";".join(flags) if flags else "none",
                "severity": "WARN" if flags else "PASS",
            }
        )
    refined = refined_mass.copy()
    refined["abs_mass_fraction_delta"] = refined["mass_fraction_delta_v2_minus_v1"].abs()
    refined["severity"] = np.where(refined["abs_mass_fraction_delta"] >= MASS_SHIFT_WARN, "WARN", "PASS")
    return pd.DataFrame(rows), refined


def artifact_flags(
    global_frame: pd.DataFrame,
    endpoint_flags: pd.DataFrame,
    by_slice: pd.DataFrame,
    by_mouse: pd.DataFrame,
    by_neighborhood: pd.DataFrame,
    endpoint_mapping: pd.DataFrame,
) -> pd.DataFrame:
    global_row = global_frame.iloc[0]
    rows = [
        {
            "artifact": "endpoint_collapse",
            "status": "WARN" if global_row["max_raw_endpoint_mass_fraction_v2"] >= ENDPOINT_COLLAPSE_FRACTION else "PASS",
            "metric": "max_raw_endpoint_mass_fraction_v2",
            "value": float(global_row["max_raw_endpoint_mass_fraction_v2"]),
            "threshold": ENDPOINT_COLLAPSE_FRACTION,
        },
        {
            "artifact": "large_endpoint_mass_shift",
            "status": "WARN" if bool((endpoint_flags["severity"] == "WARN").any()) else "PASS",
            "metric": "flagged_endpoint_count",
            "value": int((endpoint_flags["severity"] == "WARN").sum()),
            "threshold": MASS_SHIFT_WARN,
        },
        {
            "artifact": "slice_artifact",
            "status": "WARN" if not by_slice.empty and bool((by_slice["endpoint_mass_shift_max_abs"] >= GROUP_SHIFT_WARN).any()) else "PASS",
            "metric": "max_slice_shift",
            "value": float(by_slice["endpoint_mass_shift_max_abs"].max()) if not by_slice.empty else 0.0,
            "threshold": GROUP_SHIFT_WARN,
        },
        {
            "artifact": "mouse_artifact",
            "status": "WARN" if not by_mouse.empty and bool((by_mouse["endpoint_mass_shift_max_abs"] >= GROUP_SHIFT_WARN).any()) else "PASS",
            "metric": "max_mouse_shift",
            "value": float(by_mouse["endpoint_mass_shift_max_abs"].max()) if not by_mouse.empty else 0.0,
            "threshold": GROUP_SHIFT_WARN,
        },
        {
            "artifact": "neighborhood_artifact",
            "status": "WARN" if not by_neighborhood.empty and bool((by_neighborhood["endpoint_mass_shift_max_abs"] >= GROUP_SHIFT_WARN).any()) else "PASS",
            "metric": "max_neighborhood_shift",
            "value": float(by_neighborhood["endpoint_mass_shift_max_abs"].max()) if not by_neighborhood.empty else 0.0,
            "threshold": GROUP_SHIFT_WARN,
        },
        {
            "artifact": "refined_endpoint_merge_ambiguity",
            "status": "WARN" if endpoint_mapping["refined_endpoint_id"].nunique() < len(endpoint_mapping) else "PASS",
            "metric": "raw_columns_minus_unique_refined_ids",
            "value": int(len(endpoint_mapping) - endpoint_mapping["refined_endpoint_id"].nunique()),
            "threshold": 0,
        },
        {
            "artifact": "spatial_incoherence",
            "status": "REVIEW",
            "metric": "representative_maps_generated",
            "value": 1,
            "threshold": "manual_review",
        },
    ]
    return pd.DataFrame(rows)


def choose_decision(global_frame: pd.DataFrame, artifact_frame: pd.DataFrame) -> tuple[str, list[str]]:
    global_row = global_frame.iloc[0]
    severe_artifacts = artifact_frame.loc[
        (artifact_frame["status"] == "WARN")
        & (artifact_frame["artifact"].isin(["endpoint_collapse", "slice_artifact", "mouse_artifact"]))
    ]
    sharper = bool(global_row["top1_delta_v2_minus_v1"] > 0 and global_row["entropy_delta_v2_minus_v1"] < 0)
    row_qc_pass = bool(global_row["v2_row_sum_max_error"] <= ROW_ATOL and global_row["v2_nonfinite_values"] == 0 and global_row["v2_negative_values"] == 0)
    if row_qc_pass and sharper and severe_artifacts.empty:
        return (
            "keep_v1_and_v2_as_complementary_p_fate_branch",
            ["M4C-v2 is sharper, row-stochastic, and no severe collapse/slice/mouse artifacts were detected."],
        )
    if not row_qc_pass:
        return ("revise_m4c_v2", ["M4C-v2 fate matrix QC failed."])
    if not severe_artifacts.empty:
        return (
            "keep_v1_as_main_and_v2_as_diagnostic",
            ["M4C-v2 sharpened probabilities but triggered artifact warnings."],
        )
    return (
        "keep_v1_as_main_and_v2_as_diagnostic",
        ["M4C-v2 did not clearly improve fate-level sharpness."],
    )


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    work = frame.head(max_rows).copy().fillna("")
    for column in work.columns:
        work[column] = work[column].map(lambda value: str(value).replace("\n", " "))
    lines = [
        "| " + " | ".join(work.columns) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
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


def save_bar_comparison(path: Path, frame: pd.DataFrame, label_col: str, left_col: str, right_col: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = frame[label_col].astype(str).tolist()
    positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(max(8, len(labels) * 0.75), 4.5))
    axis.bar(positions - 0.2, frame[left_col], width=0.4, label="M4C-v1")
    axis.bar(positions + 0.2, frame[right_col], width=0.4, label="M4C-v2")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axis.set_title(title)
    axis.set_ylabel("mass fraction")
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_histogram(path: Path, values: list[np.ndarray], labels: list[str], title: str, xlabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4.5))
    for value_array, label in zip(values, labels, strict=True):
        axis.hist(value_array, bins=60, alpha=0.45, density=True, label=label)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("density")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_heatmap(path: Path, frame: pd.DataFrame, index_col: str, column_col: str, value_col: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if frame.empty:
        return
    heat = frame.pivot_table(index=index_col, columns=column_col, values=value_col, fill_value=0.0, aggfunc="mean")
    fig, axis = plt.subplots(figsize=(max(8, 0.55 * len(heat.columns)), max(4, min(12, 0.3 * len(heat)))))
    image = axis.imshow(heat.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set_xticks(np.arange(len(heat.columns)))
    axis.set_xticklabels(heat.columns.astype(str), rotation=45, ha="right", fontsize=7)
    y_step = max(1, len(heat) // 30)
    axis.set_yticks(np.arange(0, len(heat), y_step))
    axis.set_yticklabels(heat.index.astype(str)[::y_step], fontsize=7)
    axis.set_title(title)
    fig.colorbar(image, ax=axis, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def endpoint_colors(endpoint_mapping: pd.DataFrame) -> dict[int, Any]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    return {int(raw): cmap(int(raw) % 20) for raw in endpoint_mapping["raw_terminal_macrostate"]}


def representative_slices(metrics: pd.DataFrame) -> pd.DataFrame:
    finite = metrics.loc[np.isfinite(metrics["x"]) & np.isfinite(metrics["y"])].copy()
    if finite.empty:
        return pd.DataFrame(columns=["time", "slice_id", "n_nodes"])
    counts = (
        finite.groupby(["time", "slice_id"], observed=True)
        .size()
        .reset_index(name="n_nodes")
        .sort_values(["time", "n_nodes", "slice_id"], ascending=[True, False, True])
    )
    selected = []
    for time_label in TIMES:
        group = counts.loc[counts["time"].astype(str) == time_label]
        if not group.empty:
            selected.append(group.iloc[0].to_dict())
    return pd.DataFrame(selected)


def save_scatter_endpoint(path: Path, data: pd.DataFrame, column: str, colors: dict[int, Any], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 5))
    color_values = [colors.get(int(value), (0.5, 0.5, 0.5, 1.0)) for value in data[column]]
    axis.scatter(data["x"], data["y"], c=color_values, s=2, linewidths=0)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=str(endpoint))
        for endpoint, color in colors.items()
    ]
    axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6, ncol=1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_scatter_continuous(path: Path, data: pd.DataFrame, column: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 5))
    scatter = axis.scatter(data["x"], data["y"], c=data[column], s=2, linewidths=0, cmap="magma")
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    fig.colorbar(scatter, ax=axis, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_figures(
    paths: dict[str, Path],
    endpoint_mass: pd.DataFrame,
    refined_mass: pd.DataFrame,
    metrics: pd.DataFrame,
    by_time: pd.DataFrame,
    endpoint_mapping: pd.DataFrame,
) -> pd.DataFrame:
    figures = paths["figures"]
    rows: list[dict[str, Any]] = []

    def record(name: str, path: Path, kind: str) -> None:
        rows.append({"figure_name": name, "path": str(path), "figure_type": kind, "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0})

    path = figures / "global_endpoint_mass_v1_vs_v2.png"
    save_bar_comparison(path, endpoint_mass, "terminal_macrostate_label", "mass_fraction_v1", "mass_fraction_v2", "Global raw endpoint mass")
    record(path.name, path, "bar")
    path = figures / "refined_endpoint_mass_v1_vs_v2.png"
    save_bar_comparison(path, refined_mass, "refined_endpoint_id", "mass_fraction_v1", "mass_fraction_v2", "Global refined endpoint mass")
    record(path.name, path, "bar")
    for name, values, labels, title, xlabel in [
        ("entropy_distribution_v1_vs_v2.png", [metrics["entropy_v1"].to_numpy(), metrics["entropy_v2"].to_numpy()], ["M4C-v1", "M4C-v2"], "Entropy distribution", "entropy"),
        ("plasticity_distribution_v1_vs_v2.png", [metrics["normalized_plasticity_v1"].to_numpy(), metrics["normalized_plasticity_v2"].to_numpy()], ["M4C-v1", "M4C-v2"], "Normalized plasticity distribution", "normalized plasticity"),
        ("top1_probability_distribution_v1_vs_v2.png", [metrics["top1_probability_v1"].to_numpy(), metrics["top1_probability_v2"].to_numpy()], ["M4C-v1", "M4C-v2"], "Top1 probability distribution", "top1 probability"),
        ("js_divergence_distribution.png", [metrics["js_divergence"].to_numpy()], ["JS"], "JS divergence distribution", "JS divergence"),
    ]:
        path = figures / name
        save_histogram(path, values, labels, title, xlabel)
        record(path.name, path, "histogram")
    path = figures / "dominant_endpoint_agreement_by_time.png"
    save_bar_comparison(path, by_time, "time", "dominant_endpoint_agreement", "dominant_refined_endpoint_agreement", "Dominant endpoint agreement by time")
    record(path.name, path, "bar")

    raw_melt = pd.concat(
        [
            endpoint_mass.assign(version_label="v1", mass_fraction=endpoint_mass["mass_fraction_v1"]),
            endpoint_mass.assign(version_label="v2", mass_fraction=endpoint_mass["mass_fraction_v2"]),
        ],
        ignore_index=True,
    )
    path = figures / "endpoint_mass_heatmap_global_v1_v2.png"
    save_heatmap(path, raw_melt, "version_label", "terminal_macrostate_label", "mass_fraction", "Endpoint mass heatmap")
    record(path.name, path, "heatmap")

    time_mass = by_time[["time", "endpoint_mass_shift_max_abs", "refined_endpoint_mass_shift_max_abs"]].copy()
    time_melt = time_mass.melt(id_vars=["time"], var_name="metric", value_name="value")
    path = figures / "endpoint_mass_shift_heatmap_by_time.png"
    save_heatmap(path, time_melt, "time", "metric", "value", "Endpoint mass shift by time")
    record(path.name, path, "heatmap")

    selected = representative_slices(metrics)
    colors = endpoint_colors(endpoint_mapping)
    for _, selected_row in selected.iterrows():
        slice_id = str(selected_row["slice_id"])
        time_label = str(selected_row["time"])
        safe_slice = slice_id.replace("/", "_")
        data = metrics.loc[metrics["slice_id"].astype(str) == slice_id].copy()
        if len(data) > 25_000:
            data = data.sample(n=25_000, random_state=13).sort_index()
        figure_specs = [
            ("dominant_endpoint_v1", "dominant_v1", "endpoint"),
            ("dominant_endpoint_v2", "dominant_v2", "endpoint"),
            ("dominant_endpoint_agreement", "agreement", "continuous"),
            ("normalized_plasticity_v1", "plasticity_v1", "continuous"),
            ("normalized_plasticity_v2", "plasticity_v2", "continuous"),
            ("plasticity_delta_v2_minus_v1", "plasticity_delta", "continuous"),
        ]
        for column, suffix, plot_kind in figure_specs:
            path = figures / f"representative_{time_label}_{safe_slice}_{suffix}.png"
            if plot_kind == "endpoint":
                save_scatter_endpoint(path, data, column, colors, f"{time_label} {slice_id} {suffix}")
            else:
                save_scatter_continuous(path, data, column, f"{time_label} {slice_id} {suffix}")
            record(path.name, path, f"representative_{plot_kind}")
    return pd.DataFrame(rows)


def validate_outputs(paths: dict[str, Path], figures: pd.DataFrame) -> None:
    required = [
        paths["reports"] / "m4c_v2_production_qc_validation_report.md",
        paths["root"] / "m4c_v2_production_qc_validation_summary.csv",
        paths["reports"] / "m4c_v1_v2_comparability_report.md",
        paths["root"] / "m4c_v1_v2_endpoint_mapping_check.csv",
        paths["root"] / "m4c_v1_vs_v2_global_fate_benchmark.csv",
        paths["root"] / "m4c_v1_vs_v2_endpoint_mass_global.csv",
        paths["reports"] / "m4c_v1_vs_v2_global_fate_benchmark_report.md",
        paths["root"] / "m4c_v1_vs_v2_benchmark_by_time.csv",
        paths["root"] / "m4c_v1_vs_v2_benchmark_by_slice.csv",
        paths["root"] / "m4c_v1_vs_v2_benchmark_by_mouse.csv",
        paths["root"] / "m4c_v1_vs_v2_benchmark_by_neighborhood.csv",
        paths["reports"] / "m4c_v1_vs_v2_stratified_benchmark_report.md",
        paths["reports"] / "m4c_v2_biological_interpretability_review.md",
        paths["root"] / "m4c_v1_vs_v2_endpoint_shift_flags.csv",
        paths["root"] / "m4c_v1_vs_v2_refined_endpoint_shift_summary.csv",
        paths["reports"] / "m4c_v2_visualization_qc_report.md",
        paths["root"] / "m4c_v2_visualization_inventory.csv",
        paths["root"] / "m4c_v2_artifact_flags.csv",
        paths["reports"] / "m4c_v2_artifact_and_collapse_review.md",
        paths["reports"] / "m4c_v2_benchmark_decision_report.md",
        paths["reports"] / "m4c_v2_next_step_recommendation.md",
        paths["root"] / "m4c_v2_benchmark_summary.json",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty M4C-v2 benchmark outputs: {missing}")
    if figures.empty or not bool(figures["exists"].all()):
        raise ValueError("Visualization inventory has missing figures.")
    json.loads((paths["root"] / "m4c_v2_benchmark_summary.json").read_text(encoding="utf-8"))


def run(output_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    paths = ensure_dirs(output_root)
    protected_before = snapshot(PROTECTED_ROOTS)
    forbidden_before = snapshot(FORBIDDEN_DOWNSTREAM_ROOTS)
    input_inventory = validate_required_inputs()
    atomic_write_csv(paths["root"] / "m4c_v2_benchmark_input_inventory.csv", input_inventory)

    v1_payload = load_fate_npz(required_input_paths()["m4c_v1_fate_matrix"])
    v2_payload = load_fate_npz(required_input_paths()["m4c_v2_fate_matrix"])
    v1_prob = v1_payload["probabilities"]
    v2_prob = v2_payload["probabilities"]
    node_v1 = pd.read_parquet(required_input_paths()["m4c_v1_node_summary"])
    node_v2 = pd.read_parquet(required_input_paths()["m4c_v2_node_summary"])
    endpoint_mapping = pd.read_csv(required_input_paths()["endpoint_mapping"]).sort_values("raw_terminal_macrostate").reset_index(drop=True)

    qc_frame, qc_summary = validate_m4c_v2_production(required_input_paths()["m4c_v2_qc_summary"], v2_payload, node_v2)
    endpoint_check, comparability_summary = validate_comparability(v1_payload, v2_payload, node_v1, node_v2, endpoint_mapping)
    metrics = build_node_metrics(v1_prob, v2_prob, node_v2, endpoint_mapping)
    endpoint_mass, refined_mass = endpoint_mass_comparison(v1_prob, v2_prob, endpoint_mapping)
    global_frame = global_benchmark(v1_prob, v2_prob, metrics, endpoint_mass, refined_mass)
    by_time = stratified_benchmark(metrics, v1_prob, v2_prob, endpoint_mapping, ["time"])
    by_slice = stratified_benchmark(metrics, v1_prob, v2_prob, endpoint_mapping, ["time", "slice_id"])
    by_mouse = stratified_benchmark(metrics, v1_prob, v2_prob, endpoint_mapping, ["time", "mouse_id"])
    by_neighborhood = stratified_benchmark(metrics, v1_prob, v2_prob, endpoint_mapping, ["time", "leiden_neigh"])
    by_fine = stratified_benchmark(metrics, v1_prob, v2_prob, endpoint_mapping, ["time", "cell_type_l3"])
    endpoint_flags, refined_flags = endpoint_shift_flags(endpoint_mass, refined_mass)
    artifact_frame = artifact_flags(global_frame, endpoint_flags, by_slice, by_mouse, by_neighborhood, endpoint_mapping)
    decision, decision_reasons = choose_decision(global_frame, artifact_frame)
    figures = write_figures(paths, endpoint_mass, refined_mass, metrics, by_time, endpoint_mapping)

    protected_after = snapshot(PROTECTED_ROOTS)
    forbidden_after = snapshot(FORBIDDEN_DOWNSTREAM_ROOTS)
    upstream_diffs = diff_snapshot(protected_before, protected_after)
    forbidden_diffs = diff_snapshot(forbidden_before, forbidden_after)
    ssd_output_count = count_ssd_outputs(paths["root"])

    summary = {
        "stage": "M4C-v2-03",
        "status": "PASSED" if not upstream_diffs and not forbidden_diffs and ssd_output_count == 0 else "FAILED",
        "generated_at_utc": utc_now(),
        "runtime_seconds": float(time.monotonic() - started),
        "output_root": paths["root"],
        "reports_dir": paths["reports"],
        "figures_dir": paths["figures"],
        "m4c_v2_qc_status": qc_summary["m4c_v2_qc_status"],
        "comparability_status": comparability_summary["comparability_status"],
        "decision_category": decision,
        "decision_reasons": decision_reasons,
        "artifact_warn_count": int((artifact_frame["status"] == "WARN").sum()),
        "visualization_figure_count": int(len(figures)),
        "upstream_metadata_diff_count": len(upstream_diffs),
        "forbidden_downstream_diff_count": len(forbidden_diffs),
        "ssd_output_count": ssd_output_count,
        "upstream_metadata_diffs": upstream_diffs,
        "forbidden_downstream_diffs": forbidden_diffs,
        **qc_summary,
        **global_frame.iloc[0].to_dict(),
    }

    atomic_write_csv(paths["root"] / "m4c_v2_production_qc_validation_summary.csv", qc_frame)
    atomic_write_csv(paths["root"] / "m4c_v1_v2_endpoint_mapping_check.csv", endpoint_check)
    atomic_write_csv(paths["root"] / "m4c_v1_vs_v2_global_fate_benchmark.csv", global_frame)
    atomic_write_csv(paths["root"] / "m4c_v1_vs_v2_endpoint_mass_global.csv", endpoint_mass)
    atomic_write_csv(paths["root"] / "m4c_v1_vs_v2_refined_endpoint_shift_summary.csv", refined_flags)
    atomic_write_csv(paths["root"] / "m4c_v1_vs_v2_benchmark_by_time.csv", by_time)
    atomic_write_csv(paths["root"] / "m4c_v1_vs_v2_benchmark_by_slice.csv", by_slice)
    atomic_write_csv(paths["root"] / "m4c_v1_vs_v2_benchmark_by_mouse.csv", by_mouse)
    atomic_write_csv(paths["root"] / "m4c_v1_vs_v2_benchmark_by_neighborhood.csv", by_neighborhood)
    if not by_fine.empty:
        atomic_write_csv(paths["root"] / "m4c_v1_vs_v2_benchmark_by_fine_cluster.csv", by_fine)
    atomic_write_csv(paths["root"] / "m4c_v1_vs_v2_endpoint_shift_flags.csv", endpoint_flags)
    atomic_write_csv(paths["root"] / "m4c_v2_artifact_flags.csv", artifact_frame)
    atomic_write_csv(paths["root"] / "m4c_v2_visualization_inventory.csv", figures)
    atomic_write_json(paths["root"] / "m4c_v2_benchmark_summary.json", summary)

    write_report(paths["reports"] / "m4c_v2_production_qc_validation_report.md", "M4C-v2 Production QC Validation Report", [("Summary", markdown_table(pd.DataFrame([qc_summary]))), ("Checks", markdown_table(qc_frame))])
    write_report(paths["reports"] / "m4c_v1_v2_comparability_report.md", "M4C-v1 vs M4C-v2 Comparability Report", [("Endpoint Mapping", markdown_table(endpoint_check)), ("Comparability", markdown_table(pd.DataFrame([comparability_summary])))])
    write_report(paths["reports"] / "m4c_v1_vs_v2_global_fate_benchmark_report.md", "M4C-v1 vs M4C-v2 Global Fate Benchmark Report", [("Global Metrics", markdown_table(global_frame)), ("Endpoint Mass", markdown_table(endpoint_mass))])
    write_report(paths["reports"] / "m4c_v1_vs_v2_stratified_benchmark_report.md", "M4C-v1 vs M4C-v2 Stratified Benchmark Report", [("By Time", markdown_table(by_time)), ("By Slice", markdown_table(by_slice)), ("By Mouse", markdown_table(by_mouse)), ("By Neighborhood", markdown_table(by_neighborhood))])
    write_report(paths["reports"] / "m4c_v2_biological_interpretability_review.md", "M4C-v2 Biological Interpretability Review", [("Endpoint Shift Flags", markdown_table(endpoint_flags)), ("Refined Endpoint Shifts", markdown_table(refined_flags)), ("Interpretation", "M4C-v2 is evaluated as a pseudo-only endpoint-attraction fate map. It is not barcode-aware or lineage-validated.")])
    write_report(paths["reports"] / "m4c_v2_visualization_qc_report.md", "M4C-v2 Visualization QC Report", [("Figure Inventory", markdown_table(figures, max_rows=40)), ("Representative Maps", "Representative slices were selected deterministically by maximum finite-coordinate node count per time point. Spatial incoherence remains a manual visual QC item.")])
    write_report(paths["reports"] / "m4c_v2_artifact_and_collapse_review.md", "M4C-v2 Artifact And Collapse Review", [("Artifact Flags", markdown_table(artifact_frame)), ("Endpoint Flags", markdown_table(endpoint_flags))])
    architecture_note = (
        "M4C-v1/v2 are endpoint-anchored Markov propagation outputs in the P_fate branch. "
        "They are not standard GPCCA outputs. K_gpcca remains a separate required branch for "
        "standard pyGPCCA / CellRank-compatible macrostate discovery. DARLIN barcode evidence "
        "will later test whether barcode/hybrid transition adds information beyond pseudo-only v1/v2."
    )
    decision_body = (
        f"- decision_category: {decision}\n"
        f"- reasons: {'; '.join(decision_reasons)}\n"
        f"- artifact_warn_count: {summary['artifact_warn_count']}\n"
        f"- upstream_metadata_diff_count: {len(upstream_diffs)}\n"
        f"- forbidden_downstream_diff_count: {len(forbidden_diffs)}\n"
        f"- /ssd output count: {ssd_output_count}\n\n"
        f"{architecture_note}"
    )
    write_report(paths["reports"] / "m4c_v2_benchmark_decision_report.md", "M4C-v2 Benchmark Decision Report", [("Decision", decision_body)])
    atomic_write_text(paths["reports"] / "m4c_v2_next_step_recommendation.md", "Freeze the P_fate branch and write a Plan A architecture correction/freeze memo separating P_fate from K_gpcca. Do not start K_gpcca implementation in this task.\n")

    validate_outputs(paths, figures)
    if upstream_diffs or forbidden_diffs or ssd_output_count:
        raise RuntimeError("M4C-v2 benchmark safety validation failed.")
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(json_safe(run(args.output_root)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
