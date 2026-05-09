#!/usr/bin/env python
"""M3-v2 full-by-shard runner with dry-run/preflight safeguards."""

from __future__ import annotations

import argparse
import json
import os
import resource
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

from nichefate.io import load_config
from nichefate.m3_v2_kernel import (
    exponential_gate,
    pairwise_l2_for_edges,
    robust_scale_fit,
    robust_scale_transform,
    row_normalize_weights,
    slice_mouse_gate,
    source_adaptive_tau,
    source_entropy_and_top1,
    validate_probabilities,
)


MODE_NAME = "constrained_v1prior_sharpening"
EXPECTED_TIME_PAIRS = ["D0_to_D3", "D3_to_D9", "D9_to_D21", "D21_to_D35"]
ROW_QC_ATOL = 1e-5
MIN_SCALE = 1e-6
MIN_TAU = 1e-6
TAU_QUANTILE = 0.5
DISTANCE_CHUNK_SIZE = 100_000

M3_V1_EDGE_COLUMNS = [
    "source_anchor_id",
    "target_anchor_id",
    "source_time",
    "target_time",
    "source_slice_id",
    "target_slice_id",
    "source_mouse_id",
    "target_mouse_id",
    "row_normalized_transition_prob",
]
M3_V1_REQUIRED_COLUMNS = set(M3_V1_EDGE_COLUMNS)
M2_REQUIRED_COLUMNS = {"slice_id", "anchor_index", "time", "mouse_id"}
M4A_REQUIRED_COLUMNS = {
    "anchor_id",
    "global_node_index",
    "slice_id",
    "anchor_index",
    "time",
    "mouse_id",
}
M4C_REQUIRED_COLUMNS = {
    "anchor_id",
    "dominant_fate",
    "dominant_fate_probability",
    "normalized_plasticity_entropy",
}
M4E_REQUIRED_COLUMNS = {
    "anchor_id",
    "time_label",
    "slice_id",
    "mouse_id",
    "leiden_neigh",
    "cell_type_l1",
    "cell_type_l3",
    "x",
    "y",
}
V2_COMPONENT_COLUMNS = {
    "source_anchor_id",
    "target_anchor_id",
    "row_normalized_transition_prob",
    "v2_d_state",
    "v2_tau_state",
    "v2_g_composition",
    "v2_g_spatial_topology",
    "v2_g_slice_mouse",
}
V2_OUTPUT_REQUIRED_COLUMNS = {
    "source_anchor_id",
    "target_anchor_id",
    "source_slice_id",
    "target_slice_id",
    "source_mouse_id",
    "target_mouse_id",
    "v1_row_normalized_transition_prob",
    "v2_row_normalized_transition_prob",
    "v2_unnormalized_weight",
    "v2_rank_within_source",
}
FORBIDDEN_DOWNSTREAM_OUTPUT_TOKENS = [
    "m4a_v2",
    "m4c_v2",
    "pygpcca",
    "k_gpcca",
    "m5",
    "branchsbm",
    "branched_nicheflow",
    "barcode",
]


@dataclass(frozen=True)
class LockedParameters:
    lambda_value: float
    tau_scale: float
    top_k: int
    g_barcode: float
    row_normalization: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_v2_full_production.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = resolved(left)
    right_resolved = resolved(right)
    return path_is_relative_to(left_resolved, right_resolved) or path_is_relative_to(
        right_resolved, left_resolved
    )


def reject_ssd_path(path: Path) -> None:
    path_resolved = resolved(path)
    if path_resolved == Path("/ssd") or Path("/ssd") in path_resolved.parents:
        raise ValueError(f"Refusing to use /ssd output path: {path_resolved}")


def path_dict(config: dict[str, Any], section: str) -> dict[str, Path]:
    values = config.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"Config section {section!r} must be a mapping.")
    return {key: resolved(value) for key, value in values.items() if isinstance(value, str)}


def load_runner_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    for section in ["mode", "paths", "inputs", "expected"]:
        if section not in config:
            raise KeyError(f"Missing config section: {section}")
    return config


def locked_parameters_from_mapping(mapping: dict[str, Any]) -> LockedParameters:
    return LockedParameters(
        lambda_value=float(mapping["lambda"]),
        tau_scale=float(mapping["tau_scale"]),
        top_k=int(mapping["top_k"]),
        g_barcode=float(mapping["G_barcode"]),
        row_normalization=str(mapping["row_normalization"]),
    )


def load_mode_schema(config: dict[str, Any]) -> tuple[dict[str, Any], LockedParameters]:
    mode = config["mode"]
    schema_path = resolved(mode["schema"])
    if not schema_path.is_file():
        raise FileNotFoundError(f"Missing M3-v2 mode schema: {schema_path}")
    schema = json.loads(schema_path.read_text())
    if schema.get("mode_name") != MODE_NAME:
        raise ValueError(f"Unexpected mode schema name: {schema.get('mode_name')}")
    params = locked_parameters_from_mapping(schema["validated_pseudo_only_parameters"])
    expected = locked_parameters_from_mapping(mode["locked_parameters"])
    if params != expected:
        raise ValueError(f"Mode schema locked parameters differ from config: {params} != {expected}")
    return schema, params


def validate_output_path_separation(config: dict[str, Any]) -> None:
    paths = path_dict(config, "paths")
    output_keys = {"output_root", "full_by_shard_dir", "reports_dir", "logs_dir", "figures_dir"}
    unexpected = sorted(set(paths) - output_keys)
    if unexpected:
        raise ValueError(f"Unexpected output path keys in runner config: {unexpected}")
    protected_roots = [resolved(path) for path in config.get("protected_roots", [])]
    for key, path in paths.items():
        reject_ssd_path(path)
        for root in protected_roots:
            if paths_overlap(path, root):
                raise ValueError(f"Output path {key} overlaps protected root {root}: {path}")
    output_root = paths["output_root"]
    for key in ["full_by_shard_dir", "reports_dir", "logs_dir", "figures_dir"]:
        if not path_is_relative_to(paths[key], output_root):
            raise ValueError(f"Output path {key} must be under output_root: {paths[key]}")
    serialized = json.dumps(config.get("paths", {}), sort_keys=True).lower()
    forbidden = [token for token in FORBIDDEN_DOWNSTREAM_OUTPUT_TOKENS if token in serialized]
    if forbidden:
        raise ValueError(f"Forbidden downstream output token(s) in output paths: {forbidden}")


def parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def validate_parquet_columns(path: Path, required: set[str], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    missing = sorted(required - parquet_columns(path))
    if missing:
        raise ValueError(f"{label} is missing required columns {missing}: {path}")


def max_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / float(1024**2)


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


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def validate_input_paths(config: dict[str, Any], plan: pd.DataFrame) -> dict[str, Any]:
    inputs = path_dict(config, "inputs")
    root = inputs["m3_v1_candidate_edges_root"]
    if not root.is_dir():
        raise FileNotFoundError(f"Missing M3-v1 edge shard root: {root}")
    m2_root = inputs["m2_by_slice_dir"]
    if not m2_root.is_dir():
        raise FileNotFoundError(f"Missing M2 by-slice root: {m2_root}")
    validate_parquet_columns(inputs["m4a_global_node_table"], M4A_REQUIRED_COLUMNS, "M4A node table")
    validate_parquet_columns(inputs["m4c_v1_node_summary"], M4C_REQUIRED_COLUMNS, "M4C-v1 node summary")
    validate_parquet_columns(
        inputs["m4e_node_neighborhood_annotation"],
        M4E_REQUIRED_COLUMNS,
        "M4E neighborhood annotation",
    )
    endpoint = inputs["m4e_refined_endpoint_mapping"]
    if not endpoint.is_file() or endpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing M4E refined endpoint mapping: {endpoint}")
    feature_groups = inputs["m3_feature_groups_json"]
    if not feature_groups.is_file() or feature_groups.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing M3 feature-groups JSON: {feature_groups}")
    checked_m2 = 0
    for slice_id in sorted(plan["source_slice_id"].astype(str).unique()):
        m2_path = m2_root / slice_id / f"m2_representation_{slice_id}.parquet"
        validate_parquet_columns(m2_path, M2_REQUIRED_COLUMNS, "M2 source-slice representation")
        checked_m2 += 1
    return {
        "m3_v1_root": str(root),
        "m2_source_slice_files_checked": checked_m2,
        "m4a_node_table": str(inputs["m4a_global_node_table"]),
        "m4c_v1_node_summary": str(inputs["m4c_v1_node_summary"]),
        "m4e_node_neighborhood_annotation": str(inputs["m4e_node_neighborhood_annotation"]),
        "m4e_refined_endpoint_mapping": str(endpoint),
        "m3_feature_groups_json": str(feature_groups),
    }


def parse_time_pair(time_pair: str) -> tuple[str, str]:
    if "_to_" not in time_pair:
        raise ValueError(f"Invalid time-pair directory name: {time_pair}")
    source_time, target_time = time_pair.split("_to_", 1)
    return source_time, target_time


def discover_m3_v1_shards(config: dict[str, Any]) -> pd.DataFrame:
    inputs = path_dict(config, "inputs")
    outputs = path_dict(config, "paths")
    expected = config["expected"]
    root = inputs["m3_v1_candidate_edges_root"]
    candidate_k = int(expected["candidate_k"])
    expected_pairs = list(expected["time_pairs"])
    pair_rank = {pair: idx for idx, pair in enumerate(expected_pairs)}
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*/candidate_edges_*.parquet")):
        time_pair = path.parent.parent.name
        if time_pair not in pair_rank:
            continue
        source_time, target_time = parse_time_pair(time_pair)
        columns = parquet_columns(path)
        missing = sorted(M3_V1_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"M3-v1 shard missing required columns {missing}: {path}")
        parquet = pq.ParquetFile(path)
        row_count = int(parquet.metadata.num_rows)
        if row_count % candidate_k:
            raise ValueError(f"Shard row count is not divisible by candidate_k={candidate_k}: {path}")
        source_count = row_count // candidate_k
        retained_count = source_count * int(expected["retained_top10_edges"] / expected["full_sources"])
        source_slice = path.parent.name
        output_dir = outputs["full_by_shard_dir"] / time_pair / source_slice
        qc_path = outputs["logs_dir"] / "shard_qc" / f"m3_v2_qc_{time_pair}__{source_slice}.json"
        rows.append(
            {
                "time_pair": time_pair,
                "source_time": source_time,
                "target_time": target_time,
                "source_slice_id": source_slice,
                "m3_v1_shard_path": str(path),
                "m3_v1_row_count": row_count,
                "source_count": int(source_count),
                "retained_v2_edge_count": int(retained_count),
                "candidate_k": candidate_k,
                "top_k": int(expected["retained_top10_edges"] / expected["full_sources"]),
                "m3_v2_output_dir": str(output_dir),
                "m3_v2_output_parquet": str(output_dir / path.name),
                "m3_v2_shard_qc_json": str(qc_path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No M3-v1 candidate edge shards found under {root}")
    plan = pd.DataFrame(rows)
    plan["_time_pair_rank"] = plan["time_pair"].map(pair_rank)
    plan = plan.sort_values(["_time_pair_rank", "source_slice_id"]).reset_index(drop=True)
    plan.insert(0, "shard_id", [f"m3_v2_full_{idx + 1:04d}" for idx in range(len(plan))])
    plan = plan.drop(columns=["_time_pair_rank"])
    validate_plan_totals(plan, config)
    return plan


def validate_plan_totals(plan: pd.DataFrame, config: dict[str, Any]) -> None:
    expected = config["expected"]
    observed = {
        "full_shards": int(len(plan)),
        "full_sources": int(plan["source_count"].sum()),
        "v1_candidate_edges": int(plan["m3_v1_row_count"].sum()),
        "retained_top10_edges": int(plan["retained_v2_edge_count"].sum()),
    }
    for key, value in observed.items():
        if value != int(expected[key]):
            raise ValueError(f"Plan {key}={value} does not match expected {expected[key]}.")
    if set(plan["time_pair"].astype(str)) != set(expected["time_pairs"]):
        raise ValueError("Discovered time pairs do not match expected full M3-v2 scope.")


def select_plan(plan: pd.DataFrame, max_shards: int | None) -> pd.DataFrame:
    if max_shards is None:
        return plan.copy()
    if int(max_shards) <= 0:
        raise ValueError("--max-shards must be positive when provided.")
    return plan.head(int(max_shards)).copy()


def validate_existing_v2_shard(path: Path, expected_rows: int) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    try:
        parquet = pq.ParquetFile(path)
        missing = sorted(V2_OUTPUT_REQUIRED_COLUMNS - set(parquet.schema_arrow.names))
        if missing:
            return False, f"missing columns: {missing}"
        if int(parquet.metadata.num_rows) != int(expected_rows):
            return False, f"row count {parquet.metadata.num_rows} != expected {expected_rows}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, "valid existing shard"


def add_resume_status(plan: pd.DataFrame, resume: bool, overwrite: bool) -> pd.DataFrame:
    out = plan.copy()
    statuses: list[str] = []
    reasons: list[str] = []
    for row in out.itertuples(index=False):
        output = Path(row.m3_v2_output_parquet)
        if output.exists() and overwrite:
            statuses.append("OVERWRITE_REQUESTED")
            reasons.append("existing output will be overwritten only because --overwrite was passed")
        elif output.exists() and resume:
            valid, reason = validate_existing_v2_shard(output, int(row.retained_v2_edge_count))
            statuses.append("SKIP_VALID_EXISTING" if valid else "INVALID_EXISTING_REQUIRES_OVERWRITE")
            reasons.append(reason)
        elif output.exists():
            statuses.append("OUTPUT_EXISTS_REQUIRES_RESUME_OR_OVERWRITE")
            reasons.append("existing output found and overwrite is false")
        else:
            statuses.append("PENDING")
            reasons.append("no existing M3-v2 shard output")
    out["resume_status"] = statuses
    out["resume_reason"] = reasons
    return out


def source_codes(frame: pd.DataFrame) -> np.ndarray:
    codes, _ = pd.factorize(frame["source_anchor_id"], sort=False)
    return codes.astype(np.int32)


def selected_feature_columns(config: dict[str, Any]) -> dict[str, list[str]]:
    inputs = path_dict(config, "inputs")
    payload = json.loads(inputs["m3_feature_groups_json"].read_text())
    groups = payload["feature_groups"]
    spatial_topology = list(dict.fromkeys(groups["spatial_summary"] + groups["topology"]))
    return {
        "state": list(groups["molecular_state"][:150]),
        "composition": list(groups["cell_type_composition"][:120]),
        "spatial_topology": spatial_topology,
    }


def m2_slice_path(config: dict[str, Any], slice_id: str) -> Path:
    return path_dict(config, "inputs")["m2_by_slice_dir"] / slice_id / f"m2_representation_{slice_id}.parquet"


def load_m2_features(
    config: dict[str, Any],
    slice_ids: list[str],
    anchor_ids: set[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    frames = []
    read_columns = ["slice_id", "anchor_index", *feature_columns]
    for slice_id in sorted(set(slice_ids)):
        path = m2_slice_path(config, slice_id)
        validate_parquet_columns(path, {"slice_id", "anchor_index", *feature_columns}, "M2 feature table")
        frame = pd.read_parquet(path, columns=read_columns)
        frame["anchor_id"] = frame["slice_id"].astype(str) + "::" + frame["anchor_index"].astype(str)
        frame = frame[frame["anchor_id"].isin(anchor_ids)].copy()
        frames.append(frame[["anchor_id", *feature_columns]])
    out = pd.concat(frames, ignore_index=True)
    if out["anchor_id"].duplicated().any():
        raise ValueError("Duplicate M2 anchor_id rows after feature loading.")
    return out.set_index("anchor_id")


def aligned_feature_matrix(frame: pd.DataFrame, anchor_order: pd.Index, columns: list[str]) -> np.ndarray:
    missing = sorted(set(anchor_order) - set(frame.index))
    if missing:
        raise KeyError(f"Missing {len(missing)} M2 feature rows; first missing anchor: {missing[0]}")
    return frame.loc[anchor_order, columns].to_numpy(dtype=np.float32)


def compute_distance_block(
    source_features: pd.DataFrame,
    target_features: pd.DataFrame,
    source_order: pd.Index,
    target_order: pd.Index,
    source_pos: np.ndarray,
    target_pos: np.ndarray,
    columns: list[str],
) -> tuple[np.ndarray, int]:
    if not columns:
        return np.zeros(len(source_pos), dtype=np.float32), 0
    source_matrix = aligned_feature_matrix(source_features, source_order, columns)
    target_matrix = aligned_feature_matrix(target_features, target_order, columns)
    stats = robust_scale_fit([source_matrix, target_matrix], min_scale=MIN_SCALE)
    source_scaled = robust_scale_transform(source_matrix, stats)
    target_scaled = robust_scale_transform(target_matrix, stats)
    distances = pairwise_l2_for_edges(
        source_scaled,
        target_scaled,
        source_pos,
        target_pos,
        DISTANCE_CHUNK_SIZE,
    )
    return distances, stats.zero_scale_columns


def compute_v2_components(edges: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_cols = selected_feature_columns(config)
    all_feature_cols = list(
        dict.fromkeys(feature_cols["state"] + feature_cols["composition"] + feature_cols["spatial_topology"])
    )
    source_order = pd.Index(pd.factorize(edges["source_anchor_id"], sort=False)[1].astype(str))
    target_order = pd.Index(pd.factorize(edges["target_anchor_id"], sort=False)[1].astype(str))
    source_index = pd.Series(np.arange(len(source_order), dtype=np.int32), index=source_order)
    target_index = pd.Series(np.arange(len(target_order), dtype=np.int32), index=target_order)
    edge_source_pos = edges["source_anchor_id"].astype(str).map(source_index).to_numpy(dtype=np.int32)
    edge_target_pos = edges["target_anchor_id"].astype(str).map(target_index).to_numpy(dtype=np.int32)
    source_features = load_m2_features(
        config,
        sorted(edges["source_slice_id"].astype(str).unique()),
        set(source_order),
        all_feature_cols,
    )
    target_features = load_m2_features(
        config,
        sorted(edges["target_slice_id"].astype(str).unique()),
        set(target_order),
        all_feature_cols,
    )
    d_state, zero_state = compute_distance_block(
        source_features,
        target_features,
        source_order,
        target_order,
        edge_source_pos,
        edge_target_pos,
        feature_cols["state"],
    )
    tau_state = source_adaptive_tau(d_state, edge_source_pos, quantile=TAU_QUANTILE, min_tau=MIN_TAU)
    d_comp, zero_comp = compute_distance_block(
        source_features,
        target_features,
        source_order,
        target_order,
        edge_source_pos,
        edge_target_pos,
        feature_cols["composition"],
    )
    tau_comp = source_adaptive_tau(d_comp, edge_source_pos, quantile=TAU_QUANTILE, min_tau=MIN_TAU)
    d_spatial, zero_spatial = compute_distance_block(
        source_features,
        target_features,
        source_order,
        target_order,
        edge_source_pos,
        edge_target_pos,
        feature_cols["spatial_topology"],
    )
    tau_spatial = source_adaptive_tau(d_spatial, edge_source_pos, quantile=TAU_QUANTILE, min_tau=MIN_TAU)
    out = edges.copy()
    out["v2_d_state"] = d_state
    out["v2_tau_state"] = tau_state
    out["v2_g_composition"] = exponential_gate(d_comp, tau_comp, strength=1.0)
    out["v2_g_spatial_topology"] = exponential_gate(d_spatial, tau_spatial, strength=1.0)
    out["v2_g_slice_mouse"] = slice_mouse_gate(
        out["target_slice_id"],
        out["target_mouse_id"],
        strength=0.25,
        min_gate=0.2,
    )
    return out, {
        "zero_scale_state_columns": int(zero_state),
        "zero_scale_composition_columns": int(zero_comp),
        "zero_scale_spatial_topology_columns": int(zero_spatial),
        "unique_source_anchors": int(len(source_order)),
        "unique_target_anchors": int(len(target_order)),
    }


def top_k_mask(weights: np.ndarray, codes: np.ndarray, top_k: int) -> np.ndarray:
    work = pd.DataFrame({"source_code": codes, "weight": np.asarray(weights, dtype=np.float64)})
    ranks = work.groupby("source_code", sort=False)["weight"].rank(method="first", ascending=False)
    return ranks.to_numpy(dtype=np.float64) <= float(top_k)


def reweight_edges_with_components(
    edges: pd.DataFrame,
    params: LockedParameters,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = sorted(V2_COMPONENT_COLUMNS - set(edges.columns))
    if missing:
        raise ValueError(f"Cannot reweight edges without component columns: {missing}")
    codes = source_codes(edges)
    tau = np.clip(edges["v2_tau_state"].to_numpy(dtype=np.float64) * params.tau_scale, 1e-12, None)
    p_v1 = np.clip(edges["row_normalized_transition_prob"].to_numpy(dtype=np.float64), 1e-300, None)
    weights = np.power(p_v1, params.lambda_value)
    weights *= np.exp(-edges["v2_d_state"].to_numpy(dtype=np.float64) / tau)
    weights *= np.clip(edges["v2_g_composition"].to_numpy(dtype=np.float64), 0.0, None)
    weights *= np.clip(edges["v2_g_spatial_topology"].to_numpy(dtype=np.float64), 0.0, None)
    weights *= np.clip(edges["v2_g_slice_mouse"].to_numpy(dtype=np.float64), 0.0, None)
    weights *= params.g_barcode
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    ranks = pd.Series(weights).groupby(pd.Series(codes), sort=False).rank(method="first", ascending=False)
    keep = ranks.to_numpy() <= int(params.top_k)
    retained = edges.loc[keep].copy().reset_index(drop=True)
    retained_codes = source_codes(retained)
    retained_weights = weights[keep]
    probabilities = row_normalize_weights(retained_weights, retained_codes)
    qc = validate_probabilities(probabilities, retained_codes, atol=ROW_QC_ATOL)
    retained["v1_row_normalized_transition_prob"] = retained["row_normalized_transition_prob"].to_numpy(
        dtype=np.float64
    )
    retained["v2_unnormalized_weight"] = retained_weights
    retained["v2_rank_within_source"] = ranks.to_numpy(dtype=np.int32)[keep]
    retained["v2_row_normalized_transition_prob"] = probabilities
    retained["v2_mode_name"] = MODE_NAME
    retained["v2_lambda"] = params.lambda_value
    retained["v2_tau_scale"] = params.tau_scale
    retained["v2_top_k"] = params.top_k
    retained["v2_g_barcode"] = params.g_barcode
    qc.update(
        {
            "input_edges": int(len(edges)),
            "retained_edges": int(len(retained)),
            "source_count": int(retained["source_anchor_id"].nunique()),
            "weight_finite": bool(np.isfinite(retained_weights).all()),
            "weight_nonnegative": bool((retained_weights >= 0).all()),
        }
    )
    return retained, qc


def load_v1_shard(shard: pd.Series) -> pd.DataFrame:
    path = Path(shard["m3_v1_shard_path"])
    validate_parquet_columns(path, M3_V1_REQUIRED_COLUMNS, "M3-v1 edge shard")
    frame = pd.read_parquet(path, columns=M3_V1_EDGE_COLUMNS)
    if len(frame) != int(shard["m3_v1_row_count"]):
        raise ValueError(f"Observed {len(frame)} v1 rows, expected {shard['m3_v1_row_count']}.")
    source_count = int(frame["source_anchor_id"].nunique())
    if source_count != int(shard["source_count"]):
        raise ValueError(f"Observed {source_count} source anchors, expected {shard['source_count']}.")
    counts = frame.groupby("source_anchor_id", observed=True).size()
    if int(counts.min()) != int(shard["candidate_k"]) or int(counts.max()) != int(shard["candidate_k"]):
        raise ValueError("M3-v1 shard does not have exactly candidate_k edges per source.")
    if not bool((frame["source_time"].astype(str) == str(shard["source_time"])).all()):
        raise ValueError("M3-v1 shard source_time does not match plan.")
    if not bool((frame["target_time"].astype(str) == str(shard["target_time"])).all()):
        raise ValueError("M3-v1 shard target_time does not match plan.")
    return frame


def validate_v2_output_frame(frame: pd.DataFrame, shard: pd.Series) -> dict[str, Any]:
    missing = sorted(V2_OUTPUT_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"M3-v2 output is missing required columns: {missing}")
    expected_sources = int(shard["source_count"])
    observed_sources = int(frame["source_anchor_id"].nunique())
    if observed_sources != expected_sources:
        raise ValueError(f"Observed {observed_sources} source anchors, expected {expected_sources}.")
    counts = frame.groupby("source_anchor_id", observed=True).size()
    if int(counts.max()) > int(shard["top_k"]):
        raise ValueError("M3-v2 output retained more than top_k targets for at least one source.")
    if int(counts.min()) <= 0:
        raise ValueError("M3-v2 output has a source with no retained targets.")
    probabilities = frame["v2_row_normalized_transition_prob"].to_numpy(dtype=np.float64)
    weights = frame["v2_unnormalized_weight"].to_numpy(dtype=np.float64)
    probability_nonfinite = int((~np.isfinite(probabilities)).sum())
    weight_nonfinite = int((~np.isfinite(weights)).sum())
    probability_negative = int((probabilities < -ROW_QC_ATOL).sum())
    weight_negative = int((weights < -ROW_QC_ATOL).sum())
    if probability_nonfinite or weight_nonfinite:
        raise ValueError("M3-v2 output has non-finite weights or probabilities.")
    if probability_negative or weight_negative:
        raise ValueError("M3-v2 output has negative weights or probabilities.")
    row_sums = frame.groupby("source_anchor_id", observed=True)["v2_row_normalized_transition_prob"].sum()
    row_sum_max_abs_error = float((row_sums - 1.0).abs().max())
    if row_sum_max_abs_error > ROW_QC_ATOL:
        raise ValueError(f"M3-v2 row-sum max error {row_sum_max_abs_error} exceeds {ROW_QC_ATOL}.")
    source_stats = source_entropy_and_top1(probabilities, source_codes(frame))
    return {
        "source_count": observed_sources,
        "v1_candidate_edges": int(shard["m3_v1_row_count"]),
        "retained_v2_edges": int(len(frame)),
        "targets_per_source_min": int(counts.min()),
        "targets_per_source_max": int(counts.max()),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "row_sum_max_abs_error": row_sum_max_abs_error,
        "row_sum_pass": True,
        "probability_nonfinite_count": probability_nonfinite,
        "probability_negative_count": probability_negative,
        "weight_nonfinite_count": weight_nonfinite,
        "weight_negative_count": weight_negative,
        "weight_finite": True,
        "weight_nonnegative": True,
        "transition_entropy_mean": float(source_stats["transition_entropy"].mean()),
        "top1_probability_mean": float(source_stats["top1_probability"].mean()),
    }


def completed_record(
    shard: pd.Series,
    status: str,
    metrics: dict[str, Any],
    runtime_seconds: float,
    output_bytes: int,
) -> dict[str, Any]:
    return {
        "shard_id": shard["shard_id"],
        "time_pair": shard["time_pair"],
        "source_time": shard["source_time"],
        "target_time": shard["target_time"],
        "source_slice_id": shard["source_slice_id"],
        "status": status,
        "m3_v1_shard_path": shard["m3_v1_shard_path"],
        "m3_v2_output_parquet": shard["m3_v2_output_parquet"],
        "m3_v2_shard_qc_json": shard["m3_v2_shard_qc_json"],
        "runtime_seconds": float(runtime_seconds),
        "max_rss_gib": max_rss_gib(),
        "output_bytes": int(output_bytes),
        **metrics,
    }


def existing_shard_record(shard: pd.Series) -> dict[str, Any]:
    path = Path(shard["m3_v2_output_parquet"])
    frame = pd.read_parquet(path)
    metrics = validate_v2_output_frame(frame, shard)
    return completed_record(
        shard,
        "SKIPPED_VALID_EXISTING",
        metrics,
        runtime_seconds=0.0,
        output_bytes=path.stat().st_size,
    )


def run_one_shard(
    shard: pd.Series,
    config: dict[str, Any],
    params: LockedParameters,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str | None]:
    output_path = Path(shard["m3_v2_output_parquet"])
    qc_path = Path(shard["m3_v2_shard_qc_json"])
    if output_path.exists() and args.overwrite:
        pass
    elif output_path.exists() and args.resume:
        valid, reason = validate_existing_v2_shard(output_path, int(shard["retained_v2_edge_count"]))
        if valid:
            record = existing_shard_record(shard)
            atomic_write_json(qc_path, record)
            return record, None
        return None, f"{shard['shard_id']}\tINVALID_EXISTING_REQUIRES_OVERWRITE\t{reason}"
    elif output_path.exists():
        return None, f"{shard['shard_id']}\tOUTPUT_EXISTS_REQUIRES_RESUME_OR_OVERWRITE\t{output_path}"

    started = time.monotonic()
    v1_edges = load_v1_shard(shard)
    edges_with_components, component_qc = compute_v2_components(v1_edges, config)
    retained, reweight_qc = reweight_edges_with_components(edges_with_components, params)
    metrics = validate_v2_output_frame(retained, shard)
    metrics.update(component_qc)
    metrics.update(
        {
            "reweight_weight_finite": bool(reweight_qc["weight_finite"]),
            "reweight_weight_nonnegative": bool(reweight_qc["weight_nonnegative"]),
            "reweight_row_sum_max_abs_error": float(reweight_qc["row_sum_max_abs_error"]),
        }
    )
    atomic_write_parquet(retained, output_path)
    runtime_seconds = time.monotonic() - started
    record = completed_record(
        shard,
        "COMPLETED",
        metrics,
        runtime_seconds=runtime_seconds,
        output_bytes=output_path.stat().st_size,
    )
    atomic_write_json(qc_path, record)
    return record, None


def write_failed(path: Path, failures: list[str]) -> None:
    text = "\n".join(failures)
    if text:
        text += "\n"
    atomic_write_text(path, text)


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "completed_shard_count": 0,
            "failed_shard_count": 0,
            "total_sources": 0,
            "total_v1_candidate_edges": 0,
            "total_retained_v2_edges": 0,
            "row_sum_max_abs_error": float("nan"),
            "probability_nonfinite_count": 0,
            "probability_negative_count": 0,
            "weight_nonfinite_count": 0,
            "weight_negative_count": 0,
            "runtime_seconds": 0.0,
            "peak_rss_gib": 0.0,
        }
    return {
        "completed_shard_count": len(records),
        "failed_shard_count": 0,
        "total_sources": int(sum(record["source_count"] for record in records)),
        "total_v1_candidate_edges": int(sum(record["v1_candidate_edges"] for record in records)),
        "total_retained_v2_edges": int(sum(record["retained_v2_edges"] for record in records)),
        "row_sum_max_abs_error": float(max(record["row_sum_max_abs_error"] for record in records)),
        "probability_nonfinite_count": int(sum(record["probability_nonfinite_count"] for record in records)),
        "probability_negative_count": int(sum(record["probability_negative_count"] for record in records)),
        "weight_nonfinite_count": int(sum(record["weight_nonfinite_count"] for record in records)),
        "weight_negative_count": int(sum(record["weight_negative_count"] for record in records)),
        "runtime_seconds": float(sum(record["runtime_seconds"] for record in records)),
        "peak_rss_gib": float(max(record["max_rss_gib"] for record in records)),
    }


def production_report_text(summary: dict[str, Any], failures: list[str], args: argparse.Namespace) -> str:
    lines = [
        "# M3-v2 Full Production Report",
        "",
        "This run produced versioned M3-v2 edge shards only. It did not run M4A-v2, M4C-v2, GPCCA, M4D, M5, BranchSBM, barcode preprocessing, or downstream analyses.",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        "- execution_mode: direct",
        f"- resume: {bool(args.resume)}",
        f"- stop_on_error: {bool(args.stop_on_error)}",
        f"- max_shards: {args.max_shards if args.max_shards is not None else 'all'}",
        f"- completed_or_skipped_valid_shards: {summary['completed_shard_count']}",
        f"- failed_shards: {len(failures)}",
        f"- total_sources: {summary['total_sources']:,}",
        f"- total_v1_candidate_edges: {summary['total_v1_candidate_edges']:,}",
        f"- total_retained_v2_edges: {summary['total_retained_v2_edges']:,}",
        f"- row_sum_max_abs_error: {summary['row_sum_max_abs_error']:.6g}",
        f"- probability_nonfinite_count: {summary['probability_nonfinite_count']}",
        f"- probability_negative_count: {summary['probability_negative_count']}",
        f"- weight_nonfinite_count: {summary['weight_nonfinite_count']}",
        f"- weight_negative_count: {summary['weight_negative_count']}",
        f"- runtime_seconds_sum: {summary['runtime_seconds']:.3f}",
        f"- peak_rss_gib: {summary['peak_rss_gib']:.4f}",
        "",
        "## Guardrails",
        "",
        "- output_root: `/home/zhutao/scratch/nichefate/m3_v2/`",
        "- no_ssd_outputs_declared: True",
        "- no_downstream_outputs_declared: True",
    ]
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{failure}`" for failure in failures)
    return "\n".join(lines).rstrip() + "\n"


def write_production_outputs(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    failures: list[str],
    args: argparse.Namespace,
) -> dict[str, Path]:
    paths = path_dict(config, "paths")
    reports = paths["reports_dir"]
    reports.mkdir(parents=True, exist_ok=True)
    completed_csv = reports / "completed_shards.csv"
    failed_txt = reports / "failed_shards.txt"
    qc_summary_csv = reports / "m3_v2_full_qc_summary.csv"
    inventory_csv = reports / "m3_v2_full_output_inventory.csv"
    report_md = reports / "m3_v2_full_production_report.md"
    next_step_md = reports / "m3_v2_full_next_step_recommendation.md"

    records_frame = pd.DataFrame(records)
    if len(records_frame):
        records_frame.to_csv(completed_csv, index=False)
        records_frame.to_csv(qc_summary_csv, index=False)
        inventory_cols = [
            "shard_id",
            "time_pair",
            "source_slice_id",
            "status",
            "m3_v2_output_parquet",
            "output_bytes",
            "source_count",
            "v1_candidate_edges",
            "retained_v2_edges",
        ]
        records_frame[inventory_cols].to_csv(inventory_csv, index=False)
    else:
        pd.DataFrame().to_csv(completed_csv, index=False)
        pd.DataFrame().to_csv(qc_summary_csv, index=False)
        pd.DataFrame().to_csv(inventory_csv, index=False)
    write_failed(failed_txt, failures)
    summary = aggregate_records(records)
    summary["failed_shard_count"] = len(failures)
    atomic_write_text(report_md, production_report_text(summary, failures, args))
    atomic_write_text(
        next_step_md,
        "# M3-v2 Full Next Step Recommendation\n\n"
        "If full production completed with zero failed shards, the next step is M3-v2 full QC and v1-v2 full edge-level benchmark. Do not start M4A-v2, M4C-v2, GPCCA, barcode, M5, or BranchSBM work from this runner.\n",
    )
    return {
        "completed_shards": completed_csv,
        "failed_shards": failed_txt,
        "qc_summary": qc_summary_csv,
        "output_inventory": inventory_csv,
        "production_report": report_md,
        "next_step_recommendation": next_step_md,
    }


def run_production(
    config: dict[str, Any],
    selected: pd.DataFrame,
    params: LockedParameters,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    started = time.monotonic()
    for _, shard in selected.iterrows():
        print(
            f"M3_V2_SHARD_START {shard['shard_id']} {shard['time_pair']} {shard['source_slice_id']}",
            flush=True,
        )
        try:
            record, failure = run_one_shard(shard, config, params, args)
            if record is not None:
                records.append(record)
                print(
                    f"M3_V2_SHARD_{record['status']} {record['shard_id']} "
                    f"RETAINED {record['retained_v2_edges']} ROW_ERR {record['row_sum_max_abs_error']:.6g}",
                    flush=True,
                )
            if failure is not None:
                failures.append(failure)
                print(f"M3_V2_SHARD_FAILED {failure}", flush=True)
                if args.stop_on_error:
                    break
        except Exception as exc:  # noqa: BLE001
            failure = f"{shard['shard_id']}\tFAILED\t{type(exc).__name__}: {exc}"
            failures.append(failure)
            print(f"M3_V2_SHARD_FAILED {failure}", flush=True)
            if args.stop_on_error:
                break
    outputs = write_production_outputs(config, records, failures, args)
    summary = aggregate_records(records)
    summary["failed_shard_count"] = len(failures)
    summary["wall_runtime_seconds"] = float(time.monotonic() - started)
    return {
        "status": "PRODUCTION_COMPLETE" if not failures else "PRODUCTION_FAILED",
        "selected_shards": int(len(selected)),
        **summary,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }


def write_dryrun_outputs(
    config: dict[str, Any],
    full_plan: pd.DataFrame,
    selected_plan: pd.DataFrame,
    schema: dict[str, Any],
    params: LockedParameters,
    input_status: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Path]:
    paths = path_dict(config, "paths")
    reports = paths["reports_dir"]
    reports.mkdir(parents=True, exist_ok=True)
    selected_ids = set(selected_plan["shard_id"])
    plan = full_plan.copy()
    plan["selected_for_this_dry_run"] = plan["shard_id"].isin(selected_ids)
    plan_csv = reports / "m3_v2_full_dryrun_plan.csv"
    summary_json = reports / "m3_v2_full_dryrun_summary.json"
    report_md = reports / "m3_v2_full_preflight_report.md"
    plan.to_csv(plan_csv, index=False)
    summary = {
        "status": "PASSED",
        "dry_run": bool(args.dry_run),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode_name": schema["mode_name"],
        "locked_parameters": {
            "lambda": params.lambda_value,
            "tau_scale": params.tau_scale,
            "top_k": params.top_k,
            "G_barcode": params.g_barcode,
            "row_normalization": params.row_normalization,
        },
        "selected_shards": int(len(selected_plan)),
        "planned_full_shards": int(len(full_plan)),
        "planned_full_sources": int(full_plan["source_count"].sum()),
        "planned_v1_candidate_edges": int(full_plan["m3_v1_row_count"].sum()),
        "planned_retained_v2_edges_after_top10": int(full_plan["retained_v2_edge_count"].sum()),
        "time_pairs": EXPECTED_TIME_PAIRS,
        "reports": {
            "plan_csv": str(plan_csv),
            "summary_json": str(summary_json),
            "preflight_report": str(report_md),
        },
        "input_status": input_status,
        "dry_run_created_edge_parquet": False,
        "no_downstream_outputs_declared": True,
        "full_production_executed": False,
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    report_md.write_text(preflight_markdown(summary, selected_plan))
    return {"plan_csv": plan_csv, "summary_json": summary_json, "preflight_report": report_md}


def preflight_markdown(summary: dict[str, Any], selected_plan: pd.DataFrame) -> str:
    lines = [
        "# M3-v2 Full Dry-Run Preflight Report",
        "",
        f"- status: {summary['status']}",
        f"- dry_run: {summary['dry_run']}",
        f"- mode_name: `{summary['mode_name']}`",
        f"- lambda: {summary['locked_parameters']['lambda']}",
        f"- tau_scale: {summary['locked_parameters']['tau_scale']}",
        f"- top_k: {summary['locked_parameters']['top_k']}",
        f"- G_barcode: {summary['locked_parameters']['G_barcode']}",
        f"- selected_shards: {summary['selected_shards']}",
        f"- planned_full_shards: {summary['planned_full_shards']}",
        f"- planned_full_sources: {summary['planned_full_sources']:,}",
        f"- planned_v1_candidate_edges: {summary['planned_v1_candidate_edges']:,}",
        f"- planned_retained_v2_edges_after_top10: {summary['planned_retained_v2_edges_after_top10']:,}",
        "- full_production_executed: False",
        "- edge_parquet_written: False",
        "",
        "## Selected Shards",
        "",
        "| shard_id | time_pair | source_slice_id | source_count | v1_edges | retained_v2_edges | resume_status |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in selected_plan.itertuples(index=False):
        lines.append(
            f"| {row.shard_id} | {row.time_pair} | {row.source_slice_id} | "
            f"{row.source_count} | {row.m3_v1_row_count} | {row.retained_v2_edge_count} | "
            f"{row.resume_status} |"
        )
    lines.extend(
        [
            "",
            "## Scope Guardrails",
            "",
            "- The dry-run did not create M3-v2 shard edge parquet files.",
            "- The dry-run did not declare M4A-v2, M4C-v2, pyGPCCA, K_gpcca, M5, BranchSBM, or barcode outputs.",
            "- Output path checks used resolved pathlib ancestor checks, not string prefix matching.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_runner_config(args.config)
    validate_output_path_separation(config)
    schema, params = load_mode_schema(config)
    full_plan = discover_m3_v1_shards(config)
    input_status = validate_input_paths(config, full_plan)
    selected = select_plan(full_plan, args.max_shards)
    selected = add_resume_status(selected, args.resume, args.overwrite)
    if args.dry_run:
        outputs = write_dryrun_outputs(config, full_plan, selected, schema, params, input_status, args)
        return {
            "status": "DRY_RUN_COMPLETE",
            "selected_shards": int(len(selected)),
            "planned_full_shards": int(len(full_plan)),
            "planned_full_sources": int(full_plan["source_count"].sum()),
            "planned_v1_candidate_edges": int(full_plan["m3_v1_row_count"].sum()),
            "planned_retained_v2_edges_after_top10": int(full_plan["retained_v2_edge_count"].sum()),
            "outputs": {key: str(value) for key, value in outputs.items()},
        }
    return run_production(config, selected, params, args)


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
