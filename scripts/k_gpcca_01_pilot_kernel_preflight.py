#!/usr/bin/env python
"""K_gpcca-01 pilot kernel constructor dry-run/preflight.

This runner validates the planned pilot subset, K_within_time inputs,
P_cross_time evidence, self-loop settings, and future output contracts. It is
dry-run only for this stage: it does not construct K_gpcca matrices and does
not run pyGPCCA or CellRank.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ROOT = Path("/home/zhutao/scratch/nichefate")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "k_gpcca_pilot.yaml"

REPORT_NAMES = [
    "k_gpcca_01_preflight_report.md",
    "k_gpcca_01_input_validation_report.md",
    "k_gpcca_01_pilot_subset_plan.md",
    "k_gpcca_01_within_time_graph_plan.md",
    "k_gpcca_01_cross_time_evidence_validation.md",
    "k_gpcca_01_self_loop_plan.md",
    "k_gpcca_01_pyGPCCA_readiness_plan.md",
]

CSV_NAMES = [
    "k_gpcca_01_candidate_preflight_summary.csv",
    "k_gpcca_01_pilot_subset_summary.csv",
    "k_gpcca_01_expected_kernel_inventory.csv",
]

SUMMARY_NAME = "k_gpcca_01_dryrun_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--max-time-pairs", type=int, default=None)
    parser.add_argument("--candidate-id", "--grid-row-id", dest="candidate_id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
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


def reject_ssd(path: Path) -> None:
    path = resolved(path)
    if path == Path("/ssd") or Path("/ssd") in path.parents:
        raise ValueError(f"Refusing /ssd path: {path}")


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load K_gpcca pilot config")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config did not parse to a mapping: {path}")
    return config


def validate_output_root(config: dict[str, Any]) -> Path:
    output_root = resolved(config["paths"]["output_root"])
    reject_ssd(output_root)
    protected_roots = [resolved(path) for path in config.get("protected_roots", [])]
    forbidden_roots = [
        resolved(path) for path in config.get("forbidden_downstream_roots", [])
    ]
    for protected in protected_roots:
        if paths_overlap(output_root, protected):
            raise ValueError(
                f"Output root overlaps protected production root {protected}: {output_root}"
            )
    for forbidden in forbidden_roots:
        if paths_overlap(output_root, forbidden):
            raise ValueError(
                f"Output root overlaps forbidden downstream root {forbidden}: {output_root}"
            )
    return output_root


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    output_root = validate_output_root(config)
    reports_dir = resolved(config["paths"].get("reports_dir", output_root / "reports"))
    reject_ssd(reports_dir)
    if not is_relative_to(reports_dir, output_root):
        raise ValueError(f"Reports dir must be under output root: {reports_dir}")
    paths = {"root": output_root, "reports": reports_dir, "summary": output_root / SUMMARY_NAME}
    for name in REPORT_NAMES:
        paths[name] = reports_dir / name
    for name in CSV_NAMES:
        paths[name] = output_root / name
    return paths


def ensure_dirs(paths: dict[str, Path]) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
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
    metadata: dict[str, dict[str, Any]] = {}
    for root in paths:
        if not root.exists():
            metadata[str(root)] = {
                "exists": False,
                "size": -1,
                "mtime_ns": -1,
                "is_dir": False,
            }
            continue
        entries = [root, *sorted(path for path in root.rglob("*") if path.exists())]
        for path in entries:
            stat = path.stat()
            metadata[str(path)] = {
                "exists": True,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "is_dir": path.is_dir(),
            }
    return metadata


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
    output_root = resolved(output_root)
    if output_root == Path("/ssd") or Path("/ssd") in output_root.parents:
        return 1
    if not output_root.exists():
        return 0
    return int(
        sum(
            path.resolve() == Path("/ssd") or Path("/ssd") in path.resolve().parents
            for path in output_root.rglob("*")
        )
    )


def count_npz_outputs(output_root: Path) -> int:
    if not output_root.exists():
        return 0
    return int(sum(1 for _ in output_root.rglob("*.npz")))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidate_grid(design_root: Path, candidate_id: str | None = None) -> pd.DataFrame:
    grid_path = design_root / "k_gpcca_candidate_parameter_grid.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing K_gpcca candidate grid: {grid_path}")
    grid = pd.read_csv(grid_path)
    required = {
        "grid_id",
        "route",
        "cross_time_source",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "within_time_k",
        "similarity_metric",
        "scope",
        "priority",
        "rationale",
    }
    missing = sorted(required - set(grid.columns))
    if missing:
        raise ValueError(f"Candidate grid missing required columns: {missing}")
    if candidate_id:
        grid = grid[grid["grid_id"] == candidate_id].copy()
        if grid.empty:
            raise ValueError(f"Candidate grid row not found: {candidate_id}")
    return grid


def candidate_status(row: pd.Series) -> tuple[str, str]:
    route = str(row["route"])
    source = str(row["cross_time_source"])
    priority = str(row["priority"])
    if route == "future_barcode" or "barcode" in source.lower():
        return "SKIP_FUTURE_BARCODE", "Future placeholder; not executable before DARLIN processed clone tables."
    if "mixed" in source.lower() or priority == "review_only":
        return "REVIEW_PLANNING_ONLY", "Mixed v1/v2 evidence is design-only in K_gpcca-01."
    if route == "supernode":
        return "FALLBACK_PLANNING_ONLY", "Supernode route is a future fallback if full-resolution subset is infeasible."
    return "DRYRUN_PREFLIGHT", "Eligible for dry-run feasibility estimation."


def build_candidate_preflight_summary(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in grid.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        status, note = candidate_status(row_series)
        rows.append(
            {
                "grid_id": row.grid_id,
                "route": row.route,
                "cross_time_source": row.cross_time_source,
                "alpha": row.alpha,
                "beta": row.beta,
                "gamma": row.gamma,
                "delta": row.delta,
                "within_time_k": row.within_time_k,
                "similarity_metric": row.similarity_metric,
                "status": status,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def parquet_metadata(path: Path) -> tuple[int, list[str]]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    return int(parquet_file.metadata.num_rows), list(parquet_file.schema.names)


def read_node_table(config: dict[str, Any]) -> pd.DataFrame:
    node_path = resolved(config["paths"]["m4a_root"]) / "node_table" / "global_node_table.parquet"
    columns = [
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
    return pd.read_parquet(node_path, columns=columns)


def deterministic_select_nodes(
    nodes: pd.DataFrame,
    time_points: list[str],
    max_nodes: int,
) -> pd.DataFrame:
    if max_nodes <= 0:
        raise ValueError("--max-nodes must be positive")
    eligible = nodes[nodes["time"].isin(time_points)].copy()
    if eligible.empty:
        raise ValueError(f"No nodes found for pilot time points: {time_points}")
    sort_cols = [
        column
        for column in ["time_day", "time", "slice_id", "mouse_id", "global_node_index"]
        if column in eligible.columns
    ]
    eligible = eligible.sort_values(sort_cols).reset_index(drop=True)
    selected_parts = []
    per_time_quota = max(1, max_nodes // max(1, len(time_points)))
    selected_indices: set[int] = set()
    for time_point in time_points:
        time_rows = eligible[eligible["time"] == time_point]
        if time_rows.empty:
            continue
        take = min(per_time_quota, len(time_rows))
        positions = np.linspace(0, len(time_rows) - 1, num=take, dtype=int)
        part = time_rows.iloc[positions]
        selected_parts.append(part)
        selected_indices.update(int(idx) for idx in part.index)
    selected = pd.concat(selected_parts, ignore_index=False) if selected_parts else eligible.iloc[[]]
    if len(selected) < max_nodes:
        remaining = eligible[~eligible.index.isin(selected_indices)]
        selected = pd.concat([selected, remaining.head(max_nodes - len(selected))], ignore_index=False)
    selected = selected.head(max_nodes).sort_values("global_node_index").reset_index(drop=True)
    return selected


def read_optional_neighborhood(config: dict[str, Any], selected: pd.DataFrame) -> pd.DataFrame:
    m4e_path = (
        resolved(config["paths"]["m4e_root"])
        / "neighborhood_annotation"
        / "node_neighborhood_annotation.parquet"
    )
    if not m4e_path.exists():
        return selected
    columns = [
        "global_node_index",
        "leiden_neigh",
        "cadinu_neighborhood_label",
        "x",
        "y",
    ]
    available = parquet_metadata(m4e_path)[1]
    columns = [column for column in columns if column in available]
    if "global_node_index" not in columns:
        return selected
    annotation = pd.read_parquet(m4e_path, columns=columns)
    return selected.merge(annotation, on="global_node_index", how="left")


def build_subset_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_specs = [
        ("time", "time"),
        ("slice", "slice_id"),
        ("mouse", "mouse_id"),
        ("cell_type_l3", "cell_type_l3"),
        ("leiden_neigh", "leiden_neigh"),
    ]
    for group_name, column in group_specs:
        if column not in selected.columns:
            rows.append(
                {
                    "summary_group": group_name,
                    "label": "UNAVAILABLE",
                    "node_count": 0,
                    "fraction": 0.0,
                }
            )
            continue
        counts = selected[column].fillna("NA").value_counts().sort_index()
        for label, count in counts.items():
            rows.append(
                {
                    "summary_group": group_name,
                    "label": label,
                    "node_count": int(count),
                    "fraction": float(count / max(1, len(selected))),
                }
            )
    return pd.DataFrame(rows)


def validate_m2_features(config: dict[str, Any]) -> dict[str, Any]:
    m2_root = resolved(config["paths"]["m2_root"])
    schema_path = m2_root / "reports" / "m2_full_feature_schema.json"
    schema = load_json(schema_path)
    metadata_cols = schema.get("metadata_columns", [])
    numeric_cols = schema.get("numeric_feature_columns", [])
    composition_cols = [column for column in numeric_cols if "__ct_" in column]
    spatial_cols = [
        column
        for column in numeric_cols
        if any(token in column.lower() for token in ["spatial", "topology", "neighbor"])
    ]
    molecular_cols = [
        column
        for column in numeric_cols
        if any(token in column.lower() for token in ["molecular", "embedding", "pca"])
    ]
    missing_metadata = [
        column
        for column in ["time", "slice_id", "mouse_id", "anchor_index", "anchor_cell_id"]
        if column not in metadata_cols
    ]
    parquet_files = sorted((m2_root / "by_slice").rglob("*.parquet"))
    sample_files = parquet_files[: min(3, len(parquet_files))]
    nonfinite_count = 0
    sampled_rows = 0
    sampled_feature_cols = numeric_cols[: min(50, len(numeric_cols))]
    for path in sample_files:
        frame = pd.read_parquet(path, columns=sampled_feature_cols)
        sampled_rows += len(frame)
        values = frame.to_numpy(dtype=float, copy=False)
        nonfinite_count += int((~np.isfinite(values)).sum())
    return {
        "schema_path": schema_path,
        "m2_parquet_file_count": len(parquet_files),
        "metadata_column_count": len(metadata_cols),
        "numeric_feature_column_count": len(numeric_cols),
        "composition_feature_column_count": len(composition_cols),
        "spatial_topology_feature_column_count": len(spatial_cols),
        "molecular_embedding_feature_column_count": len(molecular_cols),
        "missing_metadata_columns": missing_metadata,
        "sampled_rows": sampled_rows,
        "sampled_feature_columns": len(sampled_feature_cols),
        "sampled_nonfinite_feature_values": nonfinite_count,
        "standardization_required": True,
        "pca_recommended": len(numeric_cols) > 100,
    }


def within_time_graph_plan(
    selected: pd.DataFrame,
    k_values: list[int],
    bytes_per_nnz: int = 16,
) -> pd.DataFrame:
    time_counts = selected["time"].value_counts().to_dict()
    rows = []
    for k_value in k_values:
        expected_nnz = sum(count * min(k_value, max(0, count - 1)) for count in time_counts.values())
        rows.append(
            {
                "component": "K_within_time",
                "k": int(k_value),
                "time_graph_count": int(len(time_counts)),
                "selected_node_count": int(len(selected)),
                "expected_nnz": int(expected_nnz),
                "row_coverage_expected": bool(all(count > 1 for count in time_counts.values())),
                "estimated_memory_mb": float(expected_nnz * bytes_per_nnz / (1024**2)),
                "construction_status": "PLANNED_DRYRUN_ONLY",
            }
        )
    return pd.DataFrame(rows)


def self_loop_plan(selected_node_count: int, gamma_values: list[float]) -> pd.DataFrame:
    rows = []
    for gamma in gamma_values:
        rows.append(
            {
                "component": "I_self",
                "gamma": float(gamma),
                "selected_node_count": int(selected_node_count),
                "expected_self_loop_nnz": int(selected_node_count),
                "enters_before_final_row_normalization": True,
                "dominance_warning": bool(gamma >= 0.10),
                "construction_status": "PLANNED_DRYRUN_ONLY",
            }
        )
    return pd.DataFrame(rows)


def edge_files_for_time_pairs(edge_root: Path, time_pairs: list[str]) -> list[Path]:
    files: list[Path] = []
    for time_pair in time_pairs:
        time_pair_root = edge_root / time_pair
        if time_pair_root.exists():
            files.extend(sorted(time_pair_root.rglob("*.parquet")))
    return files


def validate_cross_time_schema(columns: list[str], probability_column: str) -> list[str]:
    required = [
        "source_anchor_id",
        "target_anchor_id",
        "source_time",
        "target_time",
        probability_column,
    ]
    return sorted(column for column in required if column not in columns)


def inspect_cross_time_source(
    label: str,
    edge_root: Path,
    probability_column: str,
    selected_anchor_ids: set[str],
    time_pairs: list[str],
    batch_rows: int,
    max_scan_rows: int,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    files = edge_files_for_time_pairs(edge_root, time_pairs)
    total_rows = 0
    missing_column_files = 0
    missing_columns: set[str] = set()
    scanned_rows = 0
    in_pilot_edges = 0
    negative_probabilities = 0
    nonfinite_probabilities = 0
    source_mapped = 0
    target_mapped = 0
    for path in files:
        parquet_file = pq.ParquetFile(path)
        columns = list(parquet_file.schema.names)
        total_rows += int(parquet_file.metadata.num_rows)
        missing = validate_cross_time_schema(columns, probability_column)
        if missing:
            missing_column_files += 1
            missing_columns.update(missing)
            continue
        columns_to_read = [
            "source_anchor_id",
            "target_anchor_id",
            "source_time",
            "target_time",
            probability_column,
        ]
        for batch in parquet_file.iter_batches(
            batch_size=batch_rows,
            columns=columns_to_read,
        ):
            if max_scan_rows and scanned_rows >= max_scan_rows:
                break
            frame = batch.to_pandas()
            if max_scan_rows:
                remaining = max_scan_rows - scanned_rows
                frame = frame.head(max(0, remaining))
            if frame.empty:
                break
            scanned_rows += len(frame)
            probabilities = frame[probability_column].to_numpy(dtype=float, copy=False)
            negative_probabilities += int((probabilities < 0).sum())
            nonfinite_probabilities += int((~np.isfinite(probabilities)).sum())
            source_mask = frame["source_anchor_id"].isin(selected_anchor_ids)
            target_mask = frame["target_anchor_id"].isin(selected_anchor_ids)
            source_mapped += int(source_mask.sum())
            target_mapped += int(target_mask.sum())
            in_pilot_edges += int((source_mask & target_mask).sum())
        if max_scan_rows and scanned_rows >= max_scan_rows:
            break
    estimated_in_pilot = (
        int(round(in_pilot_edges * total_rows / scanned_rows))
        if scanned_rows and scanned_rows < total_rows
        else int(in_pilot_edges)
    )
    return {
        "source": label,
        "edge_root": edge_root,
        "probability_column": probability_column,
        "time_pairs": ";".join(time_pairs),
        "parquet_file_count": len(files),
        "total_rows_metadata": int(total_rows),
        "scanned_rows": int(scanned_rows),
        "scan_is_complete": bool(scanned_rows == total_rows),
        "missing_column_files": int(missing_column_files),
        "missing_columns": ";".join(sorted(missing_columns)),
        "source_mapped_in_scan": int(source_mapped),
        "target_mapped_in_scan": int(target_mapped),
        "in_pilot_edges_in_scan": int(in_pilot_edges),
        "estimated_in_pilot_edges": int(estimated_in_pilot),
        "negative_probabilities_in_scan": int(negative_probabilities),
        "nonfinite_probabilities_in_scan": int(nonfinite_probabilities),
        "status": "PASS"
        if files
        and not missing_columns
        and negative_probabilities == 0
        and nonfinite_probabilities == 0
        else "FAIL",
    }


def build_expected_kernel_inventory(
    candidate_summary: pd.DataFrame,
    selected_node_count: int,
    within_plan: pd.DataFrame,
    cross_time: pd.DataFrame,
    self_plan: pd.DataFrame,
    bytes_per_nnz: int,
) -> pd.DataFrame:
    executable = candidate_summary[
        candidate_summary["status"].isin(["DRYRUN_PREFLIGHT", "FALLBACK_PLANNING_ONLY", "REVIEW_PLANNING_ONLY"])
    ].copy()
    cross_lookup = {
        row.source: int(row.estimated_in_pilot_edges)
        for row in cross_time.itertuples(index=False)
    }
    within_lookup = {
        int(row.k): int(row.expected_nnz) for row in within_plan.itertuples(index=False)
    }
    rows = []
    for row in executable.itertuples(index=False):
        if row.cross_time_source == "M3-v1":
            cross_nnz = cross_lookup.get("M3-v1", 0)
        elif row.cross_time_source == "M3-v2":
            cross_nnz = cross_lookup.get("M3-v2", 0)
        elif row.cross_time_source == "M3-v1_v2_mixed":
            cross_nnz = max(cross_lookup.get("M3-v1", 0), cross_lookup.get("M3-v2", 0))
        else:
            cross_nnz = 0
        within_nnz = within_lookup.get(int(row.within_time_k), 0)
        self_nnz = int(selected_node_count)
        raw_nnz = int(within_nnz + cross_nnz + self_nnz)
        coalesced_nnz_upper_bound = raw_nnz
        rows.append(
            {
                "candidate_id": row.grid_id,
                "route": row.route,
                "cross_time_source": row.cross_time_source,
                "selected_node_count": int(selected_node_count),
                "expected_within_time_nnz": int(within_nnz),
                "expected_cross_time_nnz": int(cross_nnz),
                "expected_self_loop_nnz": int(self_nnz),
                "expected_raw_nnz_before_coalescing": int(raw_nnz),
                "expected_nnz_after_coalescing_upper_bound": int(coalesced_nnz_upper_bound),
                "estimated_sparse_memory_mb": float(raw_nnz * bytes_per_nnz / (1024**2)),
                "future_matrix_object": f"K_gpcca_pilot_{row.grid_id}.npz",
                "future_node_table": f"K_gpcca_pilot_node_table_{row.grid_id}.parquet",
                "future_qc_table": f"K_gpcca_pilot_qc_{row.grid_id}.csv",
                "created_in_this_task": False,
                "status": row.status,
            }
        )
    return pd.DataFrame(rows)


def build_input_validation_table(
    config: dict[str, Any],
    design_summary: dict[str, Any],
    grid: pd.DataFrame,
    m2_features: dict[str, Any],
) -> pd.DataFrame:
    checks = [
        ("config", "config_parse", True, "Config parsed successfully."),
        ("design", "design_summary_exists", bool(design_summary), "K_gpcca-00 design summary loaded."),
        ("design", "candidate_grid_exists", not grid.empty, "Candidate grid loaded."),
        ("design", "candidate_grid_has_m3_v1", bool((grid["cross_time_source"] == "M3-v1").any()), "M3-v1 candidate rows present."),
        ("design", "candidate_grid_has_m3_v2", bool((grid["cross_time_source"] == "M3-v2").any()), "M3-v2 candidate rows present."),
        ("design", "future_barcode_placeholder_only", bool((grid["route"] == "future_barcode").any()), "Future barcode rows are skipped in K_gpcca-01."),
        ("m2", "m2_numeric_features_present", m2_features["numeric_feature_column_count"] > 0, "M2 numeric niche features available."),
        ("m2", "m2_metadata_present", not m2_features["missing_metadata_columns"], f"Missing metadata: {m2_features['missing_metadata_columns']}"),
        ("paths", "output_root_safe", True, str(validate_output_root(config))),
    ]
    rows = []
    for group, check, passed, detail in checks:
        rows.append(
            {
                "group": group,
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    shown = frame.head(max_rows).copy()
    columns = [str(column) for column in shown.columns]
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for record in shown.astype(str).to_dict(orient="records"):
        values = [
            record[column].replace("|", "\\|").replace("\n", " ")
            for column in columns
        ]
        rows.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        rows.append(f"\nShowing {max_rows} of {len(frame)} rows.")
    return "\n".join(rows)


def build_reports(state: dict[str, Any]) -> dict[str, str]:
    selected_node_count = len(state["selected"])
    candidate_count = len(state["candidate_summary"])
    executable_count = int((state["candidate_summary"]["status"] == "DRYRUN_PREFLIGHT").sum())
    cross_time = state["cross_time"]
    m2_features = state["m2_features"]
    max_memory = (
        state["kernel_inventory"]["estimated_sparse_memory_mb"].max()
        if not state["kernel_inventory"].empty
        else 0.0
    )
    reports: dict[str, str] = {}
    reports["k_gpcca_01_preflight_report.md"] = f"""# K_gpcca-01 Pilot Kernel Preflight

Generated: {utc_now()}

## Scope

This is dry-run/preflight only. No K_gpcca matrix was constructed, no `.npz` matrix was written, and pyGPCCA/CellRank were not executed.

## Pilot

- Time points: {', '.join(state['time_points'])}
- Time pairs: {', '.join(state['time_pairs'])}
- Selected node count: {selected_node_count}
- Candidate grid rows evaluated: {candidate_count}
- Executable dry-run candidate rows: {executable_count}
- Estimated max sparse memory: {max_memory:.2f} MB

## Feasibility

- K_within_time feasibility: PASS if M2 features are available and each selected time has more than one node.
- P_cross_time feasibility M3-v1: {cross_time.loc[cross_time['source'] == 'M3-v1', 'status'].iloc[0] if 'M3-v1' in set(cross_time['source']) else 'NA'}
- P_cross_time feasibility M3-v2: {cross_time.loc[cross_time['source'] == 'M3-v2', 'status'].iloc[0] if 'M3-v2' in set(cross_time['source']) else 'NA'}
- Self-loop plan: selected node count self-loops for each gamma.

## Safety

- Production output writes: none.
- `.npz` files under output root: {state['safety']['npz_output_count']}
- Upstream metadata diff count: {state['safety']['upstream_metadata_diff_count']}
- Forbidden downstream diff count: {state['safety']['forbidden_downstream_diff_count']}
- `/ssd` output count: {state['safety']['ssd_output_count']}
"""
    reports["k_gpcca_01_input_validation_report.md"] = f"""# K_gpcca-01 Input Validation

## Validation Checks

{markdown_table(state['input_validation'])}

## M2 Feature Summary

- M2 parquet files: {m2_features['m2_parquet_file_count']}
- Numeric feature columns: {m2_features['numeric_feature_column_count']}
- Composition feature columns: {m2_features['composition_feature_column_count']}
- Spatial/topology feature columns: {m2_features['spatial_topology_feature_column_count']}
- Molecular/embedding feature columns: {m2_features['molecular_embedding_feature_column_count']}
- Sampled nonfinite feature values: {m2_features['sampled_nonfinite_feature_values']}
- Standardization required: {m2_features['standardization_required']}
- PCA recommended: {m2_features['pca_recommended']}
"""
    reports["k_gpcca_01_pilot_subset_plan.md"] = f"""# K_gpcca-01 Pilot Subset Plan

## Deterministic Selection

Nodes are selected from the requested time points using stable sorting by time, slice, mouse, and global node index. Sampling is stratified by time with deterministic spread across sorted rows.

## Summary

{markdown_table(state['subset_summary'])}
"""
    reports["k_gpcca_01_within_time_graph_plan.md"] = f"""# K_gpcca-01 Within-Time Graph Plan

K_within_time is planned as same-time kNN only. It will not contain cross-time edges.

## Planned kNN Settings

{markdown_table(state['within_plan'])}

## Construction Requirements

- Candidate k values: 10, 20, 30, 50.
- Candidate metrics: cosine and euclidean/RBF after scaling.
- Row-normalize each graph and final combined kernel.
- Validate same-time-only edges, degree distribution, row coverage, and memory before construction.
"""
    reports["k_gpcca_01_cross_time_evidence_validation.md"] = f"""# K_gpcca-01 Cross-Time Evidence Validation

## Evidence Sources

{markdown_table(cross_time)}

## Rules

- M3-v1 remains conservative cross-time evidence.
- M3-v2 remains sharpened cross-time evidence.
- Mixed v1/v2 rows are planning-only in this task.
- Future barcode rows are skipped.
"""
    reports["k_gpcca_01_self_loop_plan.md"] = f"""# K_gpcca-01 Self-Loop Plan

Self-loops enter before final row normalization as `gamma * I_self`.

{markdown_table(state['self_plan'])}

Gamma values at or above 0.10 are flagged for dominance review in later construction.
"""
    reports["k_gpcca_01_pyGPCCA_readiness_plan.md"] = """# K_gpcca-01 pyGPCCA Readiness Plan

Do not run pyGPCCA in K_gpcca-01.

K_gpcca-02 must validate before standard pyGPCCA:

- row-stochasticity
- no NaN, inf, or negative entries
- weak/strong component diagnostics
- irreducibility and aperiodicity diagnostics
- zero-outgoing rows
- degree distribution
- within-time versus cross-time edge mass
- self-loop mass distribution
- time-layer mixing ratio
- slice/mouse concentration diagnostics

Formal GPCCA outputs must use standard pyGPCCA or a CellRank-compatible GPCCA estimator. If pyGPCCA fails, report the failure and inspect K_gpcca diagnostics; do not use a heuristic fallback as a formal result.
"""
    return reports


def validate_generated_outputs(paths: dict[str, Path]) -> dict[str, Any]:
    required = [paths[name] for name in REPORT_NAMES + CSV_NAMES] + [paths["summary"]]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.stat().st_size == 0]
    return {
        "required_output_count": len(required),
        "missing_required_outputs": missing,
        "empty_required_outputs": empty,
    }


def choose_time_scope(config: dict[str, Any], max_time_pairs: int | None) -> tuple[list[str], list[str]]:
    time_points = list(config["pilot"]["preferred_time_points"])
    time_pairs = list(config["pilot"]["preferred_time_pairs"])
    if max_time_pairs is not None:
        time_pairs = time_pairs[:max_time_pairs]
        used_times = set()
        for time_pair in time_pairs:
            left, right = time_pair.split("_to_")
            used_times.add(left)
            used_times.add(right)
        time_points = [time for time in time_points if time in used_times]
    return time_points, time_pairs


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.dry_run:
        raise RuntimeError("K_gpcca-01 only supports --dry-run; construction is not implemented.")
    start = time.perf_counter()
    config = load_config(args.config)
    paths = output_paths(config)
    output_root = paths["root"]
    protected_roots = [resolved(path) for path in config.get("protected_roots", [])]
    forbidden_roots = [resolved(path) for path in config.get("forbidden_downstream_roots", [])]
    protected_before = snapshot(protected_roots)
    forbidden_before = snapshot(forbidden_roots)

    ensure_dirs(paths)

    design_root = resolved(config["paths"]["design_root"])
    design_summary = load_json(design_root / "k_gpcca_design_summary.json")
    grid = load_candidate_grid(design_root, args.candidate_id)
    candidate_summary = build_candidate_preflight_summary(grid)
    max_nodes = int(args.max_nodes or config["pilot"]["target_max_nodes"])
    time_points, time_pairs = choose_time_scope(config, args.max_time_pairs)

    nodes = read_node_table(config)
    selected = deterministic_select_nodes(nodes, time_points, max_nodes)
    selected = read_optional_neighborhood(config, selected)
    subset_summary = build_subset_summary(selected)
    m2_features = validate_m2_features(config)
    input_validation = build_input_validation_table(config, design_summary, grid, m2_features)
    within_plan = within_time_graph_plan(
        selected,
        [int(value) for value in config["kernel"]["within_time_k_values"]],
        int(config["kernel"]["bytes_per_sparse_nnz_estimate"]),
    )
    self_plan = self_loop_plan(
        len(selected),
        [float(value) for value in config["kernel"]["gamma_values"]],
    )

    selected_anchor_ids = set(selected["anchor_id"].astype(str))
    cross_cfg = config["cross_time"]
    cross_time = pd.DataFrame(
        [
            inspect_cross_time_source(
                "M3-v1",
                resolved(cross_cfg["m3_v1_edge_root"]),
                cross_cfg["m3_v1_probability_column"],
                selected_anchor_ids,
                time_pairs,
                int(cross_cfg["batch_rows"]),
                int(cross_cfg.get("max_edge_scan_rows_per_source", 0)),
            ),
            inspect_cross_time_source(
                "M3-v2",
                resolved(cross_cfg["m3_v2_edge_root"]),
                cross_cfg["m3_v2_probability_column"],
                selected_anchor_ids,
                time_pairs,
                int(cross_cfg["batch_rows"]),
                int(cross_cfg.get("max_edge_scan_rows_per_source", 0)),
            ),
        ]
    )
    kernel_inventory = build_expected_kernel_inventory(
        candidate_summary,
        len(selected),
        within_plan,
        cross_time,
        self_plan,
        int(config["kernel"]["bytes_per_sparse_nnz_estimate"]),
    )

    protected_after = snapshot(protected_roots)
    forbidden_after = snapshot(forbidden_roots)
    safety = {
        "upstream_metadata_diffs": diff_snapshot(protected_before, protected_after),
        "forbidden_downstream_diffs": diff_snapshot(forbidden_before, forbidden_after),
        "ssd_output_count": count_ssd_outputs(output_root),
        "npz_output_count": count_npz_outputs(output_root),
    }
    safety["upstream_metadata_diff_count"] = len(safety["upstream_metadata_diffs"])
    safety["forbidden_downstream_diff_count"] = len(safety["forbidden_downstream_diffs"])

    state = {
        "time_points": time_points,
        "time_pairs": time_pairs,
        "selected": selected,
        "subset_summary": subset_summary,
        "m2_features": m2_features,
        "input_validation": input_validation,
        "candidate_summary": candidate_summary,
        "within_plan": within_plan,
        "cross_time": cross_time,
        "self_plan": self_plan,
        "kernel_inventory": kernel_inventory,
        "safety": safety,
    }
    reports = build_reports(state)
    for name, body in reports.items():
        atomic_write_text(paths[name], body)
    atomic_write_csv(paths["k_gpcca_01_candidate_preflight_summary.csv"], candidate_summary)
    atomic_write_csv(paths["k_gpcca_01_pilot_subset_summary.csv"], subset_summary)
    atomic_write_csv(paths["k_gpcca_01_expected_kernel_inventory.csv"], kernel_inventory)

    runtime_seconds = time.perf_counter() - start
    summary = {
        "stage": "K_gpcca-01",
        "status": "PASSED"
        if safety["upstream_metadata_diff_count"] == 0
        and safety["forbidden_downstream_diff_count"] == 0
        and safety["ssd_output_count"] == 0
        and safety["npz_output_count"] == 0
        and set(cross_time["status"]) == {"PASS"}
        else "REVIEW",
        "generated_at_utc": utc_now(),
        "runtime_seconds": runtime_seconds,
        "mode": "dry_run_preflight",
        "output_root": output_root,
        "reports_dir": paths["reports"],
        "config": args.config,
        "dry_run": True,
        "k_gpcca_constructed": False,
        "pygpcca_executed": False,
        "cellrank_executed": False,
        "npz_written": False,
        "selected_time_points": time_points,
        "selected_time_pairs": time_pairs,
        "selected_node_count": int(len(selected)),
        "candidate_grid_rows_loaded": int(len(grid)),
        "candidate_grid_rows_evaluated": int(len(candidate_summary)),
        "dryrun_candidate_rows": int((candidate_summary["status"] == "DRYRUN_PREFLIGHT").sum()),
        "future_barcode_rows_skipped": int((candidate_summary["status"] == "SKIP_FUTURE_BARCODE").sum()),
        "m2_numeric_feature_column_count": int(m2_features["numeric_feature_column_count"]),
        "m2_sampled_nonfinite_feature_values": int(m2_features["sampled_nonfinite_feature_values"]),
        "within_time_min_expected_nnz": int(within_plan["expected_nnz"].min()),
        "within_time_max_expected_nnz": int(within_plan["expected_nnz"].max()),
        "m3_v1_cross_time_estimated_in_pilot_edges": int(
            cross_time.loc[cross_time["source"] == "M3-v1", "estimated_in_pilot_edges"].iloc[0]
        ),
        "m3_v2_cross_time_estimated_in_pilot_edges": int(
            cross_time.loc[cross_time["source"] == "M3-v2", "estimated_in_pilot_edges"].iloc[0]
        ),
        "self_loop_nnz": int(len(selected)),
        "max_estimated_sparse_memory_mb": float(
            kernel_inventory["estimated_sparse_memory_mb"].max()
            if not kernel_inventory.empty
            else 0.0
        ),
        "pygpcca_readiness_plan_written": True,
        **safety,
    }
    atomic_write_json(paths["summary"], summary)
    summary["output_validation"] = validate_generated_outputs(paths)
    atomic_write_json(paths["summary"], summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "mode": summary["mode"],
                "selected_node_count": summary["selected_node_count"],
                "candidate_grid_rows_evaluated": summary["candidate_grid_rows_evaluated"],
                "m3_v1_cross_time_estimated_in_pilot_edges": summary[
                    "m3_v1_cross_time_estimated_in_pilot_edges"
                ],
                "m3_v2_cross_time_estimated_in_pilot_edges": summary[
                    "m3_v2_cross_time_estimated_in_pilot_edges"
                ],
                "npz_written": summary["npz_written"],
                "upstream_metadata_diff_count": summary["upstream_metadata_diff_count"],
                "forbidden_downstream_diff_count": summary["forbidden_downstream_diff_count"],
                "ssd_output_count": summary["ssd_output_count"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
