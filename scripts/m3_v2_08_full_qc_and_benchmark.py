#!/usr/bin/env python
"""Validate M3-v2 full production and benchmark M3-v1 against M3-v2 edges."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.m3_v2_kernel import jensen_shannon_by_source, source_entropy_and_top1


ROOT = Path("/home/zhutao/scratch/nichefate")
DEFAULT_OUTPUT_ROOT = ROOT / "m3_v2_benchmark"
M3_V1_ROOT = ROOT / "m3" / "full_by_shard"
M3_V2_ROOT = ROOT / "m3_v2" / "full_by_shard"
M3_V2_REPORTS = ROOT / "m3_v2" / "reports"
M4E_ANNOTATION = ROOT / "m4e" / "neighborhood_annotation" / "node_neighborhood_annotation.parquet"
M4C_NODE_SUMMARY = ROOT / "m4c" / "fate_probabilities" / "fate_probability_node_summary.parquet"
REFINED_ENDPOINT_MAPPING = ROOT / "m4e" / "endpoint_refinement" / "refined_endpoint_mapping.csv"
M4A_NODE_TABLE = ROOT / "m4a" / "node_table" / "global_node_table.parquet"

EXPECTED_SHARDS = 52
EXPECTED_SOURCES = 1_348_582
EXPECTED_V1_EDGES = 40_457_460
EXPECTED_V2_EDGES = 13_485_820
ROW_ATOL = 1e-5
JOIN_MISSING_RATE_MAX = 0.05
V2_PROBABILITY_COLUMN = "v2_row_normalized_transition_prob"
V1_PROBABILITY_COLUMN = "row_normalized_transition_prob"
EXPECTED_TIME_PAIRS = ["D0_to_D3", "D3_to_D9", "D9_to_D21", "D21_to_D35"]
DOWNSTREAM_ROOTS = [
    ROOT / "m4a_v2",
    ROOT / "m4c_v2",
    ROOT / "m3_v2" / "pygpcca",
    ROOT / "m3_v2" / "gpcca",
    ROOT / "m3_v2" / "barcode",
    ROOT / "m3_v2" / "m5",
    ROOT / "m3_v2" / "branchsbm",
]
PROTECTED_ROOTS = [
    ROOT / "m3",
    ROOT / "m4a",
    ROOT / "m4b",
    ROOT / "m4c",
    ROOT / "m3_v2",
    *DOWNSTREAM_ROOTS,
]


@dataclass(frozen=True)
class AnnotationJoin:
    name: str
    annotation_key: str
    source_key: str
    target_key: str
    composite: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--v1-probability-column", default=None)
    return parser.parse_args()


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def paths_overlap(left: Path, right: Path) -> bool:
    l_path = resolved(left)
    r_path = resolved(right)
    return is_relative_to(l_path, r_path) or is_relative_to(r_path, l_path)


def reject_ssd(path: Path) -> None:
    path = resolved(path)
    if path == Path("/ssd") or Path("/ssd") in path.parents:
        raise ValueError(f"Refusing /ssd output path: {path}")


def validate_output_root(output_root: Path) -> None:
    output_root = resolved(output_root)
    reject_ssd(output_root)
    for root in PROTECTED_ROOTS:
        if paths_overlap(output_root, root):
            raise ValueError(f"Output root overlaps protected root {root}: {output_root}")


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def detect_v1_probability_column(columns: list[str], configured: str | None = None) -> str:
    if configured is not None:
        if configured not in columns:
            raise ValueError(f"Configured v1 probability column is missing: {configured}")
        return configured
    plausible = [
        col
        for col in columns
        if col == V1_PROBABILITY_COLUMN
        or col.endswith("_transition_prob")
        or "transition_probability" in col.lower()
    ]
    if plausible == [V1_PROBABILITY_COLUMN]:
        return V1_PROBABILITY_COLUMN
    if V1_PROBABILITY_COLUMN in plausible and len(plausible) == 1:
        return V1_PROBABILITY_COLUMN
    if not plausible:
        raise ValueError("No v1 probability column detected in frozen M3-v1 schema.")
    raise ValueError(f"Ambiguous v1 probability columns detected: {plausible}")


def validate_probability_values(frame: pd.DataFrame, column: str, label: str) -> dict[str, Any]:
    if column not in frame.columns:
        raise ValueError(f"{label} probability column missing: {column}")
    values = frame[column].to_numpy(dtype=np.float64)
    nonfinite = int((~np.isfinite(values)).sum())
    negative = int((values < -ROW_ATOL).sum())
    if nonfinite or negative:
        raise ValueError(f"{label} probability values invalid: nonfinite={nonfinite}, negative={negative}")
    return {f"{label}_probability_nonfinite_count": nonfinite, f"{label}_probability_negative_count": negative}


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


def diff_path(diff_line: str) -> Path:
    parts = diff_line.split("\t", 2)
    if len(parts) < 2:
        raise ValueError(f"Malformed metadata diff line: {diff_line}")
    return resolved(parts[1])


def load_completed() -> pd.DataFrame:
    path = M3_V2_REPORTS / "completed_shards.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed_shards.csv: {path}")
    return pd.read_csv(path)


def validate_production_completeness(max_shards: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = load_completed()
    failed = M3_V2_REPORTS / "failed_shards.txt"
    failed_text = failed.read_text(encoding="utf-8").strip() if failed.is_file() else ""
    if failed_text:
        raise RuntimeError(f"failed_shards.txt is non-empty: {failed}")
    required = {
        "shard_id",
        "time_pair",
        "m3_v1_shard_path",
        "m3_v2_output_parquet",
        "source_count",
        "v1_candidate_edges",
        "retained_v2_edges",
        "targets_per_source_max",
        "row_sum_max_abs_error",
        "probability_nonfinite_count",
        "probability_negative_count",
        "weight_nonfinite_count",
        "weight_negative_count",
    }
    missing = sorted(required - set(completed.columns))
    if missing:
        raise ValueError(f"completed_shards.csv missing required columns: {missing}")
    full_checks = {
        "completed_shards": len(completed) == EXPECTED_SHARDS,
        "total_sources": int(completed["source_count"].sum()) == EXPECTED_SOURCES,
        "total_v1_candidate_edges": int(completed["v1_candidate_edges"].sum()) == EXPECTED_V1_EDGES,
        "total_retained_v2_edges": int(completed["retained_v2_edges"].sum()) == EXPECTED_V2_EDGES,
        "targets_per_source_max": int(completed["targets_per_source_max"].max()) <= 10,
        "row_sum_max_abs_error": float(completed["row_sum_max_abs_error"].max()) <= ROW_ATOL,
        "probability_nonfinite_count": int(completed["probability_nonfinite_count"].sum()) == 0,
        "probability_negative_count": int(completed["probability_negative_count"].sum()) == 0,
        "weight_nonfinite_count": int(completed["weight_nonfinite_count"].sum()) == 0,
        "weight_negative_count": int(completed["weight_negative_count"].sum()) == 0,
    }
    failed_checks = [name for name, ok in full_checks.items() if not ok]
    if failed_checks:
        raise RuntimeError(f"M3-v2 production completeness checks failed: {failed_checks}")
    for _, row in completed.iterrows():
        v1_path = resolved(row["m3_v1_shard_path"])
        v2_path = resolved(row["m3_v2_output_parquet"])
        if not v1_path.is_file():
            raise FileNotFoundError(f"Missing M3-v1 shard parquet: {v1_path}")
        if not v2_path.is_file():
            raise FileNotFoundError(f"Missing M3-v2 shard parquet: {v2_path}")
        if not is_relative_to(v1_path, resolved(M3_V1_ROOT)):
            raise ValueError(f"M3-v1 shard path is outside frozen M3-v1 root: {v1_path}")
        if not is_relative_to(v2_path, resolved(ROOT / "m3_v2")):
            raise ValueError(f"M3-v2 shard path is outside m3_v2 root: {v2_path}")
        reject_ssd(v1_path)
        reject_ssd(v2_path)
    selected = completed.head(int(max_shards)).copy() if max_shards is not None else completed.copy()
    summary = pd.DataFrame(
        [
            {"check": name, "passed": ok, "max_shards": max_shards if max_shards is not None else "all"}
            for name, ok in full_checks.items()
        ]
    )
    return completed, selected.reset_index(drop=True), summary


def read_refined_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(REFINED_ENDPOINT_MAPPING)
    return mapping[
        [
            "raw_terminal_macrostate",
            "refined_endpoint_id",
            "refined_endpoint_label",
            "confidence_tier_after_refinement",
        ]
    ].rename(columns={"raw_terminal_macrostate": "dominant_fate"})


def load_annotations() -> pd.DataFrame:
    m4e_cols = [
        "global_node_index",
        "anchor_id",
        "slice_id",
        "anchor_cell_id",
        "time_label",
        "mouse_id",
        "leiden_neigh",
        "cell_type_l1",
        "cell_type_l3",
    ]
    m4c_cols = [
        "global_node_index",
        "anchor_id",
        "dominant_fate",
        "dominant_fate_label",
        "dominant_fate_probability",
        "normalized_plasticity_entropy",
    ]
    m4e = pd.read_parquet(M4E_ANNOTATION, columns=m4e_cols)
    m4c = pd.read_parquet(M4C_NODE_SUMMARY, columns=m4c_cols)
    m4c = m4c.merge(read_refined_mapping(), on="dominant_fate", how="left")
    m4c = m4c.drop(columns=["global_node_index"], errors="ignore")
    out = m4e.merge(m4c, on="anchor_id", how="left")
    if out["anchor_id"].duplicated().any():
        raise ValueError("Annotation anchor_id is not globally unique after M4E/M4C merge.")
    return out


def choose_annotation_join(edge_columns: set[str], annotations: pd.DataFrame) -> AnnotationJoin:
    if {"source_global_node_index", "target_global_node_index"} <= edge_columns and "global_node_index" in annotations:
        return AnnotationJoin("global_node_index", "global_node_index", "source_global_node_index", "target_global_node_index")
    if "global_node_index" in edge_columns and "global_node_index" in annotations:
        return AnnotationJoin("global_node_index", "global_node_index", "global_node_index", "global_node_index")
    if {"source_anchor_id", "target_anchor_id"} <= edge_columns and annotations["anchor_id"].is_unique:
        return AnnotationJoin("anchor_id", "anchor_id", "source_anchor_id", "target_anchor_id")
    composite_source = {"source_slice_id", "source_anchor_cell_id"} <= edge_columns
    composite_target = {"target_slice_id", "target_anchor_cell_id"} <= edge_columns
    if composite_source and composite_target and {"slice_id", "anchor_cell_id"} <= set(annotations.columns):
        return AnnotationJoin("slice_id_anchor_cell_id", "slice_id_anchor_cell_id", "source_slice_id", "target_slice_id", True)
    raise ValueError("Could not choose a stable annotation join key for benchmark edges.")


def annotation_lookup(annotations: pd.DataFrame, join: AnnotationJoin) -> pd.DataFrame:
    if join.composite:
        work = annotations.copy()
        work["slice_id_anchor_cell_id"] = work["slice_id"].astype(str) + "::" + work["anchor_cell_id"].astype(str)
        return work.set_index("slice_id_anchor_cell_id", drop=False)
    return annotations.set_index(join.annotation_key, drop=False)


def edge_keys(frame: pd.DataFrame, join: AnnotationJoin, side: str) -> pd.Series:
    if not join.composite:
        return frame[join.source_key if side == "source" else join.target_key]
    slice_col = "source_slice_id" if side == "source" else "target_slice_id"
    cell_col = "source_anchor_cell_id" if side == "source" else "target_anchor_cell_id"
    return frame[slice_col].astype(str) + "::" + frame[cell_col].astype(str)


def map_annotation(frame: pd.DataFrame, lookup: pd.DataFrame, join: AnnotationJoin, side: str, column: str) -> pd.Series:
    keys = edge_keys(frame, join, side)
    return keys.map(lookup[column])


def normalized_entropy(distribution: pd.Series) -> float:
    values = distribution[distribution > 0].to_numpy(dtype=float)
    if len(values) <= 1:
        return 0.0
    probs = values / float(values.sum())
    return float(-(probs * np.log(np.clip(probs, 1e-300, None))).sum() / np.log(len(probs)))


def weighted_distribution(categories: pd.Series, weights: pd.Series) -> pd.Series:
    work = pd.DataFrame({"category": categories.fillna("NA").astype(str), "weight": weights.astype(float)})
    return work.groupby("category", sort=False)["weight"].sum()


def top_targets(frame: pd.DataFrame, probability_column: str) -> pd.DataFrame:
    idx = frame.groupby("source_anchor_id", sort=False)[probability_column].idxmax()
    return frame.loc[idx].copy()


def agreement_counts(
    top: pd.DataFrame,
    lookup: pd.DataFrame,
    join: AnnotationJoin,
    source_col: str,
    target_col: str,
) -> tuple[int, int]:
    source_values = map_annotation(top, lookup, join, "source", source_col)
    target_values = map_annotation(top, lookup, join, "target", target_col)
    valid = source_values.notna() & target_values.notna()
    if not bool(valid.any()):
        return 0, 0
    return int((source_values[valid].astype(str) == target_values[valid].astype(str)).sum()), int(valid.sum())


def method_metrics(
    frame: pd.DataFrame,
    probability_column: str,
    lookup: pd.DataFrame,
    join: AnnotationJoin,
    method: str,
) -> tuple[dict[str, Any], dict[str, pd.Series]]:
    stats = source_entropy_and_top1(
        frame[probability_column].to_numpy(dtype=np.float64),
        pd.factorize(frame["source_anchor_id"], sort=False)[0],
    )
    top = top_targets(frame, probability_column)
    source_missing = map_annotation(frame, lookup, join, "source", "leiden_neigh").isna().mean()
    target_leiden = map_annotation(frame, lookup, join, "target", "leiden_neigh")
    target_fine = map_annotation(frame, lookup, join, "target", "cell_type_l3")
    target_endpoint = map_annotation(frame, lookup, join, "target", "refined_endpoint_id")
    target_missing = target_leiden.isna().mean()
    leiden_match, leiden_valid = agreement_counts(top, lookup, join, "leiden_neigh", "leiden_neigh")
    fine_match, fine_valid = agreement_counts(top, lookup, join, "cell_type_l3", "cell_type_l3")
    endpoint_match, endpoint_valid = agreement_counts(top, lookup, join, "refined_endpoint_id", "refined_endpoint_id")
    weights = frame[probability_column].astype(float)
    dists = {
        "endpoint": weighted_distribution(target_endpoint, weights),
        "leiden": weighted_distribution(target_leiden, weights),
        "fine_cluster": weighted_distribution(target_fine, weights),
        "slice": weighted_distribution(frame["target_slice_id"], weights),
        "mouse": weighted_distribution(frame["target_mouse_id"], weights),
    }
    slice_conc = float(dists["slice"].max() / dists["slice"].sum()) if float(dists["slice"].sum()) > 0 else float("nan")
    mouse_conc = float(dists["mouse"].max() / dists["mouse"].sum()) if float(dists["mouse"].sum()) > 0 else float("nan")
    metrics = {
        f"{method}_entropy_mean": float(stats["transition_entropy"].mean()),
        f"{method}_top1_probability_mean": float(stats["top1_probability"].mean()),
        f"{method}_source_annotation_missing_rate": float(source_missing),
        f"{method}_target_annotation_missing_rate": float(target_missing),
        f"{method}_leiden_match_count": leiden_match,
        f"{method}_leiden_valid_count": leiden_valid,
        f"{method}_fine_match_count": fine_match,
        f"{method}_fine_valid_count": fine_valid,
        f"{method}_endpoint_match_count": endpoint_match,
        f"{method}_endpoint_valid_count": endpoint_valid,
        f"{method}_leiden_consistency": float(leiden_match / leiden_valid) if leiden_valid else float("nan"),
        f"{method}_fine_cluster_consistency": float(fine_match / fine_valid) if fine_valid else float("nan"),
        f"{method}_refined_endpoint_plausibility": float(endpoint_match / endpoint_valid) if endpoint_valid else float("nan"),
        f"{method}_target_slice_concentration": slice_conc,
        f"{method}_target_mouse_concentration": mouse_conc,
        f"{method}_slice_mouse_collapse": max(slice_conc, mouse_conc),
        f"{method}_target_neighborhood_diversity": normalized_entropy(dists["leiden"]),
    }
    return metrics, dists


def validate_corresponding_paths(row: pd.Series) -> tuple[Path, Path]:
    v1 = resolved(row["m3_v1_shard_path"])
    v2 = resolved(row["m3_v2_output_parquet"])
    if not v1.is_file():
        raise FileNotFoundError(f"Missing v1 shard: {v1}")
    if not v2.is_file():
        raise FileNotFoundError(f"Missing v2 shard: {v2}")
    if not is_relative_to(v1, resolved(M3_V1_ROOT)):
        raise ValueError(f"v1 shard is outside frozen M3-v1 root: {v1}")
    if not is_relative_to(v2, resolved(M3_V2_ROOT)):
        raise ValueError(f"v2 shard is outside M3-v2 production root: {v2}")
    return v1, v2


def benchmark_shard(
    row: pd.Series,
    lookup: pd.DataFrame,
    join: AnnotationJoin,
    configured_v1_prob: str | None,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], pd.Series]]:
    v1_path, v2_path = validate_corresponding_paths(row)
    v1_prob = detect_v1_probability_column(parquet_columns(v1_path), configured_v1_prob)
    v2_columns = parquet_columns(v2_path)
    if V2_PROBABILITY_COLUMN not in v2_columns:
        raise ValueError(f"Missing v2 probability column {V2_PROBABILITY_COLUMN}: {v2_path}")
    common_columns = [
        "source_anchor_id",
        "target_anchor_id",
        "source_slice_id",
        "target_slice_id",
        "source_mouse_id",
        "target_mouse_id",
    ]
    v1 = pd.read_parquet(v1_path, columns=[*common_columns, v1_prob])
    v2 = pd.read_parquet(v2_path, columns=[*common_columns, V2_PROBABILITY_COLUMN])
    validate_probability_values(v1, v1_prob, "v1")
    validate_probability_values(v2, V2_PROBABILITY_COLUMN, "v2")
    v1_metrics, v1_dists = method_metrics(v1, v1_prob, lookup, join, "v1")
    v2_metrics, v2_dists = method_metrics(v2, V2_PROBABILITY_COLUMN, lookup, join, "v2")

    v2_pairs = v2[["source_anchor_id", "target_anchor_id", V2_PROBABILITY_COLUMN]]
    aligned = v1[["source_anchor_id", "target_anchor_id", v1_prob]].merge(
        v2_pairs,
        on=["source_anchor_id", "target_anchor_id"],
        how="left",
    )
    aligned[V2_PROBABILITY_COLUMN] = aligned[V2_PROBABILITY_COLUMN].fillna(0.0)
    source_codes, _ = pd.factorize(aligned["source_anchor_id"], sort=False)
    js = jensen_shannon_by_source(
        aligned[v1_prob].to_numpy(dtype=np.float64),
        aligned[V2_PROBABILITY_COLUMN].to_numpy(dtype=np.float64),
        source_codes.astype(np.int32),
    )
    v1_top10 = (
        v1.sort_values(["source_anchor_id", v1_prob], ascending=[True, False])
        .assign(v1_rank=lambda frame: frame.groupby("source_anchor_id", sort=False).cumcount() + 1)
        .loc[lambda frame: frame["v1_rank"] <= 10, ["source_anchor_id", "target_anchor_id"]]
        .copy()
    )
    v1_top10["in_v1_top10"] = True
    v2_overlap = v2[["source_anchor_id", "target_anchor_id"]].merge(
        v1_top10,
        on=["source_anchor_id", "target_anchor_id"],
        how="left",
    )
    v2_in_v1 = v2[["source_anchor_id", "target_anchor_id"]].merge(
        v1[["source_anchor_id", "target_anchor_id"]],
        on=["source_anchor_id", "target_anchor_id"],
        how="left",
        indicator=True,
    )
    v1_sources = int(v1["source_anchor_id"].nunique())
    v2_sources = int(v2["source_anchor_id"].nunique())
    row_out = {
        "shard_id": row["shard_id"],
        "time_pair": row["time_pair"],
        "source_slice_id": row["source_slice_id"],
        "v1_probability_column": v1_prob,
        "v2_probability_column": V2_PROBABILITY_COLUMN,
        "annotation_join_key_used": join.name,
        "source_count_v1": v1_sources,
        "source_count_v2": v2_sources,
        "source_coverage_equal": bool(v1_sources == v2_sources),
        "v1_candidate_edges": int(len(v1)),
        "v2_retained_edges": int(len(v2)),
        "v1_unique_targets": int(v1["target_anchor_id"].nunique()),
        "v2_unique_targets": int(v2["target_anchor_id"].nunique()),
        "v2_edges_in_v1_fraction": float((v2_in_v1["_merge"] == "both").mean()),
        "v2_retained_in_v1_top10_fraction": float(v2_overlap["in_v1_top10"].fillna(False).astype(bool).mean()),
        "mean_js_divergence_from_v1": float(js["v1_v2_js_divergence"].mean()),
        "targets_per_source_v2_max": int(v2.groupby("source_anchor_id", observed=True).size().max()),
        **v1_metrics,
        **v2_metrics,
    }
    row_out.update(delta_metrics(row_out))
    dist_rows = {}
    for method, dists in [("v1", v1_dists), ("v2", v2_dists)]:
        for category_type, series in dists.items():
            dist_rows[(row["time_pair"], method, category_type)] = series
    return row_out, dist_rows


def delta_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "entropy_delta_v2_minus_v1": row["v2_entropy_mean"] - row["v1_entropy_mean"],
        "top1_delta_v2_minus_v1": row["v2_top1_probability_mean"] - row["v1_top1_probability_mean"],
        "leiden_delta_v2_minus_v1": row["v2_leiden_consistency"] - row["v1_leiden_consistency"],
        "fine_delta_v2_minus_v1": row["v2_fine_cluster_consistency"] - row["v1_fine_cluster_consistency"],
        "endpoint_delta_v2_minus_v1": row["v2_refined_endpoint_plausibility"] - row["v1_refined_endpoint_plausibility"],
        "collapse_delta_v2_minus_v1": row["v2_slice_mouse_collapse"] - row["v1_slice_mouse_collapse"],
        "diversity_delta_v2_minus_v1": row["v2_target_neighborhood_diversity"] - row["v1_target_neighborhood_diversity"],
    }


def add_distribution(acc: dict[tuple[str, str, str], pd.Series], key: tuple[str, str, str], values: pd.Series) -> None:
    if key in acc:
        acc[key] = acc[key].add(values, fill_value=0.0)
    else:
        acc[key] = values.copy()


def distribution_frame(acc: dict[tuple[str, str, str], pd.Series], category_type: str) -> pd.DataFrame:
    rows = []
    for (time_pair, method, observed_type), series in acc.items():
        if observed_type != category_type:
            continue
        total = float(series.sum())
        for category, weight in series.sort_values(ascending=False).items():
            rows.append(
                {
                    "time_pair": time_pair,
                    "method": method,
                    "category_type": category_type,
                    "category": category,
                    "weight": float(weight),
                    "fraction": float(weight / total) if total > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def aggregate_group(group: pd.DataFrame, label: str) -> dict[str, Any]:
    weights = group["source_count_v1"].astype(float)
    row: dict[str, Any] = {
        "time_pair": label,
        "shards": int(len(group)),
        "source_count": int(group["source_count_v1"].sum()),
        "v1_candidate_edges": int(group["v1_candidate_edges"].sum()),
        "v2_retained_edges": int(group["v2_retained_edges"].sum()),
        "annotation_join_key_used": ",".join(sorted(set(group["annotation_join_key_used"].astype(str)))),
        "v1_source_annotation_missing_rate": float(np.average(group["v1_source_annotation_missing_rate"], weights=weights)),
        "v2_source_annotation_missing_rate": float(np.average(group["v2_source_annotation_missing_rate"], weights=weights)),
        "v1_target_annotation_missing_rate": float(np.average(group["v1_target_annotation_missing_rate"], weights=weights)),
        "v2_target_annotation_missing_rate": float(np.average(group["v2_target_annotation_missing_rate"], weights=weights)),
        "mean_js_divergence_from_v1": float(np.average(group["mean_js_divergence_from_v1"], weights=weights)),
    }
    for method in ["v1", "v2"]:
        row[f"{method}_entropy_mean"] = float(np.average(group[f"{method}_entropy_mean"], weights=weights))
        row[f"{method}_top1_probability_mean"] = float(np.average(group[f"{method}_top1_probability_mean"], weights=weights))
        for metric in ["leiden", "fine", "endpoint"]:
            match_col = f"{method}_{'fine' if metric == 'fine' else metric}_match_count"
            valid_col = f"{method}_{'fine' if metric == 'fine' else metric}_valid_count"
            out_col = {
                "leiden": "leiden_consistency",
                "fine": "fine_cluster_consistency",
                "endpoint": "refined_endpoint_plausibility",
            }[metric]
            valid = int(group[valid_col].sum())
            row[f"{method}_{out_col}"] = float(group[match_col].sum() / valid) if valid else float("nan")
    return row


def apply_distribution_metrics(rows: pd.DataFrame, acc: dict[tuple[str, str, str], pd.Series]) -> pd.DataFrame:
    out = rows.copy()
    for idx, row in out.iterrows():
        label = row["time_pair"]
        for method in ["v1", "v2"]:
            leiden = acc.get((label, method, "leiden"), pd.Series(dtype=float))
            slice_dist = acc.get((label, method, "slice"), pd.Series(dtype=float))
            mouse_dist = acc.get((label, method, "mouse"), pd.Series(dtype=float))
            out.loc[idx, f"{method}_target_neighborhood_diversity"] = normalized_entropy(leiden)
            slice_conc = float(slice_dist.max() / slice_dist.sum()) if float(slice_dist.sum()) > 0 else float("nan")
            mouse_conc = float(mouse_dist.max() / mouse_dist.sum()) if float(mouse_dist.sum()) > 0 else float("nan")
            out.loc[idx, f"{method}_target_slice_concentration"] = slice_conc
            out.loc[idx, f"{method}_target_mouse_concentration"] = mouse_conc
            out.loc[idx, f"{method}_slice_mouse_collapse"] = max(slice_conc, mouse_conc)
        updated = delta_metrics(out.loc[idx].to_dict())
        for key, value in updated.items():
            out.loc[idx, key] = value
    return out


def aggregate_outputs(shard_rows: pd.DataFrame, dist_acc: dict[tuple[str, str, str], pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_rows = [aggregate_group(group, time_pair) for time_pair, group in shard_rows.groupby("time_pair", sort=True)]
    time_frame = apply_distribution_metrics(pd.DataFrame(time_rows), dist_acc)
    global_acc: dict[tuple[str, str, str], pd.Series] = {}
    for (time_pair, method, category_type), series in dist_acc.items():
        add_distribution(global_acc, ("global", method, category_type), series)
    global_frame = apply_distribution_metrics(pd.DataFrame([aggregate_group(shard_rows, "global")]), global_acc)
    return time_frame, global_frame


def add_acceptance_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["endpoint_ok"] = out["endpoint_delta_v2_minus_v1"] >= -0.02
    out["leiden_ok"] = out["leiden_delta_v2_minus_v1"] >= -0.03
    out["entropy_ok"] = out["entropy_delta_v2_minus_v1"] < 0.0
    out["top1_ok"] = out["top1_delta_v2_minus_v1"] > 0.0
    out["collapse_ok"] = out["collapse_delta_v2_minus_v1"] <= 0.005
    out["diversity_ok"] = out["diversity_delta_v2_minus_v1"] >= -0.03
    out["source_join_ok"] = out[["v1_source_annotation_missing_rate", "v2_source_annotation_missing_rate"]].max(axis=1) <= JOIN_MISSING_RATE_MAX
    out["target_join_ok"] = out[["v1_target_annotation_missing_rate", "v2_target_annotation_missing_rate"]].max(axis=1) <= JOIN_MISSING_RATE_MAX
    out["all_thresholds_pass"] = out[
        ["endpoint_ok", "leiden_ok", "entropy_ok", "top1_ok", "collapse_ok", "diversity_ok", "source_join_ok", "target_join_ok"]
    ].all(axis=1)
    return out


def choose_decision(global_frame: pd.DataFrame, time_frame: pd.DataFrame) -> tuple[str, list[str]]:
    global_flags = add_acceptance_flags(global_frame).iloc[0]
    time_flags = add_acceptance_flags(time_frame)
    reasons: list[str] = []
    failing_pairs = time_flags[~time_flags["all_thresholds_pass"]]
    if len(failing_pairs):
        for row in failing_pairs.itertuples(index=False):
            failed = [
                name
                for name in [
                    "endpoint_ok",
                    "leiden_ok",
                    "entropy_ok",
                    "top1_ok",
                    "collapse_ok",
                    "diversity_ok",
                    "source_join_ok",
                    "target_join_ok",
                ]
                if not bool(getattr(row, name))
            ]
            reasons.append(f"{row.time_pair}: failed {','.join(failed)}")
        severe_artifact = bool((~failing_pairs[["collapse_ok", "diversity_ok", "source_join_ok", "target_join_ok"]]).any(axis=None))
        if severe_artifact:
            return "revise_v2_and_repeat_full_or_partial", reasons
        return "keep_v1_as_main_baseline", reasons
    if not bool(global_flags["all_thresholds_pass"]):
        reasons.append("global metrics failed one or more thresholds")
        return "keep_v1_as_main_baseline", reasons
    if float(global_flags["endpoint_delta_v2_minus_v1"]) >= 0 and float(global_flags["leiden_delta_v2_minus_v1"]) >= 0:
        return "adopt_v2_as_default_pseudo_mode", ["global and per-time-pair thresholds passed with nonnegative plausibility deltas"]
    return "keep_v1_and_v2_as_complementary", ["v2 sharpens transitions and passes plausibility thresholds but does not dominate v1"]


def write_distribution_csvs(paths: dict[str, Path], dist_acc: dict[tuple[str, str, str], pd.Series]) -> dict[str, Path]:
    outputs = {
        "endpoint": paths["root"] / "m3_v1_vs_v2_target_endpoint_by_time_pair.csv",
        "leiden": paths["root"] / "m3_v1_vs_v2_target_leiden_by_time_pair.csv",
        "fine_cluster": paths["root"] / "m3_v1_vs_v2_target_fine_cluster_by_time_pair.csv",
    }
    for category_type, path in outputs.items():
        distribution_frame(dist_acc, category_type).to_csv(path, index=False)
    return outputs


def plot_bar(time_frame: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(time_frame))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - width / 2, time_frame[f"v1_{metric}"], width, label="v1")
    ax.bar(x + width / 2, time_frame[f"v2_{metric}"], width, label="v2")
    ax.set_xticks(x)
    ax.set_xticklabels(time_frame["time_pair"].astype(str), rotation=25, ha="right")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_retained_edges(time_frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(time_frame["time_pair"].astype(str), time_frame["v2_retained_edges"])
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Retained M3-v2 edges by time pair")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap(distribution: pd.DataFrame, title: str, path: Path, top_n: int = 20) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if distribution.empty:
        return
    top_categories = (
        distribution.groupby("category")["weight"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    )
    work = distribution[distribution["category"].isin(top_categories)].copy()
    work["row"] = work["time_pair"].astype(str) + "__" + work["method"].astype(str)
    pivot = work.pivot_table(index="row", columns="category", values="fraction", aggfunc="sum", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_figures(paths: dict[str, Path], time_frame: pd.DataFrame, dist_outputs: dict[str, Path]) -> list[str]:
    warnings = []
    specs = [
        ("entropy_mean", "entropy_v1_vs_v2_by_time_pair.png", "Entropy v1 vs v2 by time pair"),
        ("top1_probability_mean", "top1_v1_vs_v2_by_time_pair.png", "Top1 probability v1 vs v2 by time pair"),
        ("refined_endpoint_plausibility", "endpoint_plausibility_v1_vs_v2_by_time_pair.png", "Endpoint plausibility v1 vs v2"),
        ("leiden_consistency", "leiden_consistency_v1_vs_v2_by_time_pair.png", "Leiden consistency v1 vs v2"),
        ("slice_mouse_collapse", "slice_mouse_collapse_v1_vs_v2_by_time_pair.png", "Slice/mouse collapse v1 vs v2"),
    ]
    try:
        for metric, filename, title in specs:
            plot_bar(time_frame, metric, title, paths["figures"] / filename)
        plot_retained_edges(time_frame, paths["figures"] / "retained_edge_count_by_time_pair.png")
        plot_heatmap(
            pd.read_csv(dist_outputs["endpoint"]),
            "Target endpoint distribution v1 vs v2",
            paths["figures"] / "target_endpoint_distribution_heatmap.png",
        )
        plot_heatmap(
            pd.read_csv(dist_outputs["leiden"]),
            "Target Leiden distribution v1 vs v2",
            paths["figures"] / "target_leiden_distribution_heatmap.png",
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Figure generation failed: {exc}")
    return warnings


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    work = frame[columns].copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: f"{float(value):.4g}" if pd.notna(value) else "NA")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in work.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_reports(
    paths: dict[str, Path],
    qc_summary: pd.DataFrame,
    shard_frame: pd.DataFrame,
    time_frame: pd.DataFrame,
    global_frame: pd.DataFrame,
    decision: str,
    decision_reasons: list[str],
    validation_diffs: dict[str, int],
    figure_warnings: list[str],
) -> None:
    full_mode = "all" if len(shard_frame) == EXPECTED_SHARDS else f"max_shards={len(shard_frame)}"
    (paths["reports"] / "m3_v2_full_qc_validation_report.md").write_text(
        "# M3-v2 Full QC Validation Report\n\n"
        f"- benchmark_scope: {full_mode}\n"
        f"- validation_checks_passed: {bool(qc_summary['passed'].all())}\n"
        f"- completed_shards_available: {EXPECTED_SHARDS}\n"
        f"- benchmarked_shards: {len(shard_frame)}\n"
        f"- upstream_metadata_diff_count: {validation_diffs['upstream']}\n"
        f"- forbidden_downstream_metadata_diff_count: {validation_diffs['downstream']}\n",
        encoding="utf-8",
    )
    (paths["reports"] / "m3_v2_full_vs_pilot_consistency_report.md").write_text(
        "# M3-v2 Full vs Pilot Consistency Report\n\n"
        f"{markdown_table(time_frame, ['time_pair', 'entropy_delta_v2_minus_v1', 'top1_delta_v2_minus_v1', 'endpoint_delta_v2_minus_v1', 'leiden_delta_v2_minus_v1', 'collapse_delta_v2_minus_v1', 'diversity_delta_v2_minus_v1'])}\n\n"
        "The same acceptance thresholds used for the decision report are evaluated globally and per time pair.\n",
        encoding="utf-8",
    )
    artifact_flags = add_acceptance_flags(time_frame)
    (paths["reports"] / "m3_v2_biological_interpretability_review.md").write_text(
        "# M3-v2 Biological Interpretability Review\n\n"
        f"{markdown_table(artifact_flags, ['time_pair', 'endpoint_ok', 'leiden_ok', 'collapse_ok', 'diversity_ok', 'source_join_ok', 'target_join_ok'])}\n",
        encoding="utf-8",
    )
    (paths["reports"] / "m3_v2_full_benchmark_decision_report.md").write_text(
        "# M3-v2 Full Benchmark Decision Report\n\n"
        f"- decision_category: `{decision}`\n"
        f"- global_summary:\n\n{markdown_table(global_frame, ['time_pair', 'source_count', 'v1_candidate_edges', 'v2_retained_edges', 'entropy_delta_v2_minus_v1', 'top1_delta_v2_minus_v1', 'endpoint_delta_v2_minus_v1', 'leiden_delta_v2_minus_v1', 'collapse_delta_v2_minus_v1', 'diversity_delta_v2_minus_v1'])}\n\n"
        "## Decision Reasons\n\n"
        + "\n".join(f"- {reason}" for reason in decision_reasons)
        + ("\n\n## Figure Warnings\n\n" + "\n".join(f"- {warning}" for warning in figure_warnings) if figure_warnings else "")
        + "\n",
        encoding="utf-8",
    )
    next_step = (
        "M4A-v2 assembly planning only; do not execute M4A-v2 without explicit approval."
        if decision in {"adopt_v2_as_default_pseudo_mode", "keep_v1_and_v2_as_complementary"}
        else "Do not proceed to M4A-v2 planning; review the benchmark failure/artifact flags first."
    )
    (paths["reports"] / "m3_v2_next_step_recommendation.md").write_text(
        "# M3-v2 Next Step Recommendation\n\n"
        f"- decision_category: `{decision}`\n"
        f"- exact_next_step: {next_step}\n"
        "- not_run: M4A-v2, M4C-v2, pyGPCCA, K_gpcca, M4D diagnostics, barcode preprocessing, M5, BranchSBM.\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    paths = ensure_dirs(args.output_root)
    watched_roots = [ROOT / "m3", ROOT / "m4a", ROOT / "m4b", ROOT / "m4c", *DOWNSTREAM_ROOTS]
    before = snapshot(watched_roots)
    completed, selected, qc_summary = validate_production_completeness(args.max_shards)
    annotations = load_annotations()
    first_v1, first_v2 = validate_corresponding_paths(selected.iloc[0])
    v1_prob = detect_v1_probability_column(parquet_columns(first_v1), args.v1_probability_column)
    if V2_PROBABILITY_COLUMN not in parquet_columns(first_v2):
        raise ValueError(f"Missing v2 probability column: {V2_PROBABILITY_COLUMN}")
    join = choose_annotation_join(set(parquet_columns(first_v1)).union(parquet_columns(first_v2)), annotations)
    lookup = annotation_lookup(annotations, join)
    shard_rows: list[dict[str, Any]] = []
    dist_acc: dict[tuple[str, str, str], pd.Series] = {}
    for row in selected.itertuples(index=False):
        record, shard_dists = benchmark_shard(pd.Series(row._asdict()), lookup, join, v1_prob)
        shard_rows.append(record)
        for key, series in shard_dists.items():
            add_distribution(dist_acc, key, series)
        print(
            f"M3_V2_08_SHARD {record['shard_id']} {record['time_pair']} "
            f"entropy_delta={record['entropy_delta_v2_minus_v1']:.6g} "
            f"top1_delta={record['top1_delta_v2_minus_v1']:.6g}",
            flush=True,
        )
    shard_frame = pd.DataFrame(shard_rows)
    time_frame, global_frame = aggregate_outputs(shard_frame, dist_acc)
    decision, reasons = choose_decision(global_frame, time_frame)

    qc_summary.to_csv(paths["reports"] / "m3_v2_full_qc_validation_summary.csv", index=False)
    shard_frame.to_csv(paths["root"] / "m3_v1_vs_v2_edge_benchmark_by_shard.csv", index=False)
    time_frame.to_csv(paths["root"] / "m3_v1_vs_v2_edge_benchmark_by_time_pair.csv", index=False)
    global_frame.to_csv(paths["root"] / "m3_v1_vs_v2_edge_benchmark_global_summary.csv", index=False)
    dist_outputs = write_distribution_csvs(paths, dist_acc)
    figure_warnings = write_figures(paths, time_frame, dist_outputs)
    after = snapshot(watched_roots)
    diffs = diff_snapshot(before, after)
    upstream_roots = [resolved(ROOT / "m3"), resolved(ROOT / "m4a"), resolved(ROOT / "m4b"), resolved(ROOT / "m4c")]
    upstream_diffs = [
        diff
        for diff in diffs
        if any(is_relative_to(diff_path(diff), root) for root in upstream_roots)
    ]
    downstream_diffs = [diff for diff in diffs if diff not in upstream_diffs]
    validation_diffs = {"upstream": len(upstream_diffs), "downstream": len(downstream_diffs)}
    write_reports(
        paths,
        qc_summary,
        shard_frame,
        time_frame,
        global_frame,
        decision,
        reasons,
        validation_diffs,
        figure_warnings,
    )
    payload = {
        "status": "PASSED" if not upstream_diffs and not downstream_diffs else "FAILED",
        "max_shards": args.max_shards,
        "benchmarked_shards": int(len(shard_frame)),
        "available_completed_shards": int(len(completed)),
        "annotation_join_key_used": join.name,
        "v1_probability_column": v1_prob,
        "v2_probability_column": V2_PROBABILITY_COLUMN,
        "decision_category": decision,
        "decision_reasons": reasons,
        "global_summary": global_frame.iloc[0].to_dict(),
        "time_pair_summary": time_frame.to_dict(orient="records"),
        "upstream_metadata_diff_count": len(upstream_diffs),
        "forbidden_downstream_metadata_diff_count": len(downstream_diffs),
        "runtime_seconds": float(time.monotonic() - started),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(paths["root"] / "m3_v1_vs_v2_edge_benchmark_summary.json", payload)
    if upstream_diffs or downstream_diffs:
        raise RuntimeError("Metadata diff detected for protected upstream/downstream roots.")
    return payload


def main() -> None:
    print(json.dumps(json_safe(run(parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
