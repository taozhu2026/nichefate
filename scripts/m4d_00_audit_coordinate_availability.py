#!/usr/bin/env python
"""Audit tissue-space coordinate availability for M4D visualization."""

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
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config


DEFAULT_CONFIG = "configs/m4d_markov_macrostate_visualization.yaml"
COORDINATE_CANDIDATES = ["x", "y", "x_raw", "y_raw", "spatial_x", "spatial_y", "center_x", "center_y"]
M4_NODE_COLUMNS = ["global_node_index", "anchor_id", "slice_id", "anchor_index", "anchor_cell_id"]
M1_COLUMNS = ["slice_id", "anchor_index", "anchor_cell_id", "scale", "x", "y"]
NO_DOWNSTREAM_FLAGS = {
    "no_gpcca": True,
    "no_branched_nicheflow_training": True,
    "no_branchsbm_training": True,
    "no_m5": True,
    "no_regulator_analysis": True,
}


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


def assert_no_ssd_path(path: Path, label: str) -> None:
    resolved = str(path.expanduser().resolve())
    if resolved == "/ssd" or resolved.startswith("/ssd/"):
        raise ValueError(f"Refusing to use /ssd for {label}: {path}")


def configured_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    if bool(config.get("validation", {}).get("fail_on_ssd_path", True)):
        for key, path in paths.items():
            assert_no_ssd_path(path, f"paths.{key}")
    return paths


def output_paths(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "audit_md": paths["reports_dir"] / "m4d_coordinate_availability_audit.md",
        "audit_csv": paths["reports_dir"] / "m4d_coordinate_availability_audit.csv",
        "coordinate_cache": paths["visualization_dir"] / "node_coordinates.parquet",
    }


def parquet_columns(path: Path) -> list[str]:
    return list(pq.read_schema(path).names)


def coordinate_columns(columns: list[str]) -> list[str]:
    lower = {column.lower(): column for column in columns}
    pairs = [("x", "y"), ("x_raw", "y_raw"), ("spatial_x", "spatial_y"), ("center_x", "center_y")]
    found: list[str] = []
    for x_col, y_col in pairs:
        if x_col in lower and y_col in lower:
            found.extend([lower[x_col], lower[y_col]])
    return found


def inspect_parquet_source(name: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    columns = parquet_columns(path) if exists else []
    coords = coordinate_columns(columns)
    join_keys = [column for column in ["global_node_index", "anchor_id", "slice_id", "anchor_index", "anchor_cell_id"] if column in columns]
    return {
        "source_priority": name,
        "source_artifact": str(path),
        "source_exists": exists,
        "available_coordinate_columns": ";".join(coords),
        "available_join_keys": ";".join(join_keys),
        "status": "coordinates_available" if coords else ("present_no_coordinates" if exists else "missing"),
        "selected_coordinate_source": False,
        "matched_nodes": 0,
        "missing_coordinates": pd.NA,
        "tissue_space_maps_enabled": False,
        "cross_time_physical_arrows_allowed": False,
        "state_space_only_visualization": True,
    }


def first_m2_file(m2_root: Path) -> Path | None:
    files = sorted(m2_root.glob("*/m2_representation_*.parquet"))
    return files[0] if files else None


def m0_fallback_path(paths: dict[str, Path]) -> Path:
    return paths.get("m0_final_h5ad", paths.get("m0_processed_dir", Path("")))


def m1_files(m1_root: Path) -> list[Path]:
    return sorted(m1_root.glob("*/niche_features_*.parquet"))


def inspect_sources(paths: dict[str, Path]) -> pd.DataFrame:
    rows = [
        inspect_parquet_source("1_m4a_node_table", paths["m4a_node_table"]),
        inspect_parquet_source("2_m4c_node_summary", paths["m4c_node_summary"]),
    ]
    m2_file = first_m2_file(paths["m2_by_slice_root"])
    if m2_file is None:
        rows.append(
            {
                "source_priority": "3_m2_by_slice_representation",
                "source_artifact": str(paths["m2_by_slice_root"]),
                "source_exists": False,
                "available_coordinate_columns": "",
                "available_join_keys": "",
                "status": "missing",
                "selected_coordinate_source": False,
                "matched_nodes": 0,
                "missing_coordinates": pd.NA,
                "tissue_space_maps_enabled": False,
                "cross_time_physical_arrows_allowed": False,
                "state_space_only_visualization": True,
            }
        )
    else:
        row = inspect_parquet_source("3_m2_by_slice_representation", m2_file)
        row["source_artifact"] = str(paths["m2_by_slice_root"])
        rows.append(row)
    files = m1_files(paths["m1_by_slice_root"])
    if files:
        columns = parquet_columns(files[0])
        coords = coordinate_columns(columns)
        rows.append(
            {
                "source_priority": "4_m1_by_slice_niche_features",
                "source_artifact": str(paths["m1_by_slice_root"]),
                "source_exists": True,
                "available_coordinate_columns": ";".join(coords),
                "available_join_keys": ";".join([column for column in ["slice_id", "anchor_index", "anchor_cell_id"] if column in columns]),
                "status": "coordinates_available" if coords else "present_no_coordinates",
                "selected_coordinate_source": bool(coords),
                "matched_nodes": 0,
                "missing_coordinates": pd.NA,
                "tissue_space_maps_enabled": False,
                "cross_time_physical_arrows_allowed": False,
                "state_space_only_visualization": True,
            }
        )
    else:
        rows.append(
            {
                "source_priority": "4_m1_by_slice_niche_features",
                "source_artifact": str(paths["m1_by_slice_root"]),
                "source_exists": False,
                "available_coordinate_columns": "",
                "available_join_keys": "",
                "status": "missing",
                "selected_coordinate_source": False,
                "matched_nodes": 0,
                "missing_coordinates": pd.NA,
                "tissue_space_maps_enabled": False,
                "cross_time_physical_arrows_allowed": False,
                "state_space_only_visualization": True,
            }
        )
    fallback_path = m0_fallback_path(paths)
    rows.append(
        {
            "source_priority": "5_m0_h5ad_fallback",
            "source_artifact": str(fallback_path),
            "source_exists": fallback_path.exists(),
            "available_coordinate_columns": "not_loaded",
            "available_join_keys": "not_loaded",
            "status": "skipped_cheaper_m1_coordinates_available" if files else "not_loaded",
            "selected_coordinate_source": False,
            "matched_nodes": 0,
            "missing_coordinates": pd.NA,
            "tissue_space_maps_enabled": False,
            "cross_time_physical_arrows_allowed": False,
            "state_space_only_visualization": True,
        }
    )
    return pd.DataFrame(rows)


def validate_m4a_node_table(node_table: pd.DataFrame, expected_nodes: int) -> pd.DataFrame:
    missing = sorted(set(M4_NODE_COLUMNS) - set(node_table.columns))
    if missing:
        raise KeyError(f"M4A node table missing columns needed for coordinate join: {missing}")
    if len(node_table) != int(expected_nodes):
        raise ValueError(f"Expected {expected_nodes} M4A nodes, found {len(node_table)}.")
    if bool(node_table["global_node_index"].duplicated().any()):
        raise ValueError("M4A node table contains duplicate global_node_index values.")
    sorted_table = node_table.sort_values("global_node_index", kind="mergesort").reset_index(drop=True).copy()
    expected = np.arange(len(sorted_table), dtype=np.int64)
    if not np.array_equal(sorted_table["global_node_index"].to_numpy(dtype=np.int64), expected):
        raise ValueError("M4A global_node_index must be contiguous and row-aligned.")
    return sorted_table[M4_NODE_COLUMNS].copy()


def check_coordinate_consistency(frame: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        frame.groupby(["slice_id", "anchor_index"], sort=False, observed=True)
        .agg(
            x_min=("x", "min"),
            x_max=("x", "max"),
            y_min=("y", "min"),
            y_max=("y", "max"),
            n_scale_rows=("scale", "nunique"),
            n_anchor_cell_ids=("anchor_cell_id", "nunique"),
        )
        .reset_index()
    )
    grouped["x_range"] = grouped["x_max"] - grouped["x_min"]
    grouped["y_range"] = grouped["y_max"] - grouped["y_min"]
    bad = grouped[(grouped["x_range"].abs() > tolerance) | (grouped["y_range"].abs() > tolerance)].copy()
    return grouped, bad


def reduce_m1_coordinates(
    frame: pd.DataFrame,
    source_path: Path,
    coordinate_scale: str,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    consistency, bad = check_coordinate_consistency(frame, tolerance)
    if len(bad):
        examples = bad.head(5).to_dict("records")
        raise ValueError(f"M1 coordinates are not identical across scale rows for {len(bad)} anchors; examples={examples}")
    selected = frame.loc[frame["scale"].astype(str) == str(coordinate_scale)].copy()
    if selected.empty:
        available = sorted(frame["scale"].dropna().astype(str).unique().tolist())
        raise ValueError(f"Configured M1 coordinate scale {coordinate_scale!r} not found; available scales: {available}")
    duplicate_mask = selected.duplicated(["slice_id", "anchor_index"], keep=False)
    if bool(duplicate_mask.any()):
        examples = selected.loc[duplicate_mask, ["slice_id", "anchor_index"]].head(5).to_dict("records")
        raise ValueError(f"M1 selected coordinate scale has duplicate anchor rows: {examples}")
    selected = selected.rename(columns={"x": "x_raw", "y": "y_raw"})
    selected["coordinate_source"] = "m1_by_slice_niche_features"
    selected["coordinate_source_path"] = str(source_path)
    selected["coordinate_scale_used"] = str(coordinate_scale)
    selected["coordinate_join_key"] = "slice_id+anchor_index"
    summary = {
        "anchors_checked": int(len(consistency)),
        "nonidentical_coordinate_anchors": int(len(bad)),
        "selected_coordinate_rows": int(len(selected)),
        "scale_rows_per_anchor_min": int(consistency["n_scale_rows"].min()) if len(consistency) else 0,
        "scale_rows_per_anchor_max": int(consistency["n_scale_rows"].max()) if len(consistency) else 0,
        "anchor_cell_id_variants_max": int(consistency["n_anchor_cell_ids"].max()) if len(consistency) else 0,
    }
    return (
        selected[
            [
                "slice_id",
                "anchor_index",
                "anchor_cell_id",
                "x_raw",
                "y_raw",
                "coordinate_source",
                "coordinate_source_path",
                "coordinate_scale_used",
                "coordinate_join_key",
            ]
        ],
        summary,
    )


def load_m1_coordinate_rows(
    m1_root: Path,
    coordinate_scale: str,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    files = m1_files(m1_root)
    if not files:
        raise FileNotFoundError(f"No M1 niche feature parquet files found under {m1_root}")
    for path in files:
        frame = pd.read_parquet(path, columns=M1_COLUMNS)
        missing = sorted(set(M1_COLUMNS) - set(frame.columns))
        if missing:
            raise KeyError(f"M1 file {path} missing coordinate audit columns: {missing}")
        reduced, summary = reduce_m1_coordinates(frame, path, coordinate_scale, tolerance)
        rows.append(reduced)
        summary["source_path"] = str(path)
        summaries.append(summary)
    coordinate_rows = pd.concat(rows, ignore_index=True)
    duplicate_mask = coordinate_rows.duplicated(["slice_id", "anchor_index"], keep=False)
    if bool(duplicate_mask.any()):
        examples = coordinate_rows.loc[duplicate_mask, ["slice_id", "anchor_index"]].head(5).to_dict("records")
        raise ValueError(f"Reduced M1 coordinate table has duplicate slice_id+anchor_index keys: {examples}")
    return coordinate_rows, pd.DataFrame(summaries)


def add_slice_normalized_coordinates(cache: pd.DataFrame) -> pd.DataFrame:
    result = cache.copy()
    result["x_centered_by_slice"] = np.nan
    result["y_centered_by_slice"] = np.nan
    result["x_scaled_by_slice"] = np.nan
    result["y_scaled_by_slice"] = np.nan
    matched = result["coordinate_join_status"] == "matched"
    for _, idx in result.loc[matched].groupby("slice_id", sort=False, observed=True).groups.items():
        x = result.loc[idx, "x_raw"].to_numpy(dtype=float)
        y = result.loc[idx, "y_raw"].to_numpy(dtype=float)
        x_center = float(np.nanmedian(x))
        y_center = float(np.nanmedian(y))
        x_centered = x - x_center
        y_centered = y - y_center
        scale = float(np.nanmedian(np.sqrt(x_centered**2 + y_centered**2)))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        result.loc[idx, "x_centered_by_slice"] = x_centered
        result.loc[idx, "y_centered_by_slice"] = y_centered
        result.loc[idx, "x_scaled_by_slice"] = x_centered / scale
        result.loc[idx, "y_scaled_by_slice"] = y_centered / scale
    return result


def build_coordinate_cache(
    node_table: pd.DataFrame,
    m1_coordinates: pd.DataFrame,
) -> pd.DataFrame:
    merged = node_table.merge(
        m1_coordinates,
        on=["slice_id", "anchor_index"],
        how="left",
        suffixes=("", "_m1"),
        sort=False,
    )
    if len(merged) != len(node_table):
        raise ValueError(f"Coordinate join changed row count from {len(node_table)} to {len(merged)}.")
    if bool(merged["global_node_index"].duplicated().any()):
        raise ValueError("Coordinate cache contains duplicate global_node_index values.")
    expected = np.arange(len(merged), dtype=np.int64)
    if not np.array_equal(merged["global_node_index"].to_numpy(dtype=np.int64), expected):
        raise ValueError("Coordinate cache global_node_index is not row-aligned to M4A.")
    matched = merged["x_raw"].notna() & merged["y_raw"].notna()
    merged["coordinate_join_status"] = np.where(matched, "matched", "missing")
    if "anchor_cell_id_m1" in merged.columns:
        mismatch = matched & merged["anchor_cell_id_m1"].notna() & (merged["anchor_cell_id"].astype(str) != merged["anchor_cell_id_m1"].astype(str))
        if bool(mismatch.any()):
            examples = merged.loc[mismatch, ["slice_id", "anchor_index", "anchor_cell_id", "anchor_cell_id_m1"]].head(5).to_dict("records")
            raise ValueError(f"Coordinate join anchor_cell_id validation failed: {examples}")
        merged = merged.drop(columns=["anchor_cell_id_m1"])
    for column in ["coordinate_source", "coordinate_source_path", "coordinate_scale_used", "coordinate_join_key"]:
        merged[column] = merged[column].fillna("missing")
    merged = add_slice_normalized_coordinates(merged)
    return merged[
        [
            "global_node_index",
            "anchor_id",
            "slice_id",
            "anchor_index",
            "anchor_cell_id",
            "x_raw",
            "y_raw",
            "x_centered_by_slice",
            "y_centered_by_slice",
            "x_scaled_by_slice",
            "y_scaled_by_slice",
            "coordinate_source",
            "coordinate_source_path",
            "coordinate_scale_used",
            "coordinate_join_key",
            "coordinate_join_status",
        ]
    ]


def cache_qc(cache: pd.DataFrame, expected_nodes: int) -> dict[str, Any]:
    matched = int((cache["coordinate_join_status"] == "matched").sum())
    missing = int((cache["coordinate_join_status"] == "missing").sum())
    return {
        "expected_rows": int(expected_nodes),
        "cache_rows": int(len(cache)),
        "global_node_index_unique": bool(cache["global_node_index"].is_unique),
        "global_node_index_aligned": bool(
            np.array_equal(cache["global_node_index"].to_numpy(dtype=np.int64), np.arange(len(cache), dtype=np.int64))
        ),
        "matched_nodes": matched,
        "missing_coordinates": missing,
        "tissue_space_maps_enabled": bool(missing == 0 and len(cache) == int(expected_nodes)),
        "cross_time_physical_arrows_allowed": False,
        "tissue_space_arrows_allowed": False,
        "state_space_only_visualization": bool(missing > 0),
    }


def update_audit_rows(audit: pd.DataFrame, qc: dict[str, Any]) -> pd.DataFrame:
    result = audit.copy()
    m1_mask = result["source_priority"] == "4_m1_by_slice_niche_features"
    result.loc[m1_mask, "selected_coordinate_source"] = True
    result.loc[m1_mask, "matched_nodes"] = int(qc["matched_nodes"])
    result.loc[m1_mask, "missing_coordinates"] = int(qc["missing_coordinates"])
    result.loc[m1_mask, "tissue_space_maps_enabled"] = bool(qc["tissue_space_maps_enabled"])
    result.loc[m1_mask, "cross_time_physical_arrows_allowed"] = False
    result.loc[m1_mask, "state_space_only_visualization"] = bool(qc["state_space_only_visualization"])
    return result


def audit_report_text(
    audit: pd.DataFrame,
    scale_summary: pd.DataFrame,
    qc: dict[str, Any],
    cache_path: Path,
    runtime_seconds: float,
) -> str:
    selected = audit.loc[audit["selected_coordinate_source"].astype(bool)]
    selected_source = selected["source_priority"].iloc[0] if len(selected) else "none"
    lines = [
        "# M4D-00 Coordinate Availability Audit",
        "",
        "This audit selected the cheapest valid tissue-space coordinate source for M4D visualization.",
        "M0 h5ad files were not loaded because M1 by-slice parquet files contain x/y coordinates.",
        "",
        "## Source Priority Results",
    ]
    for row in audit.to_dict("records"):
        lines.append(
            f"- {row['source_priority']}: status={row['status']}, "
            f"coordinate_columns={row['available_coordinate_columns']}, join_keys={row['available_join_keys']}"
        )
    lines.extend(
        [
            "",
            "## Selected Coordinate Source",
            f"- selected source: {selected_source}",
            "- join key: slice_id + anchor_index",
            f"- coordinate cache: {cache_path}",
            f"- expected rows: {qc['expected_rows']}",
            f"- cache rows: {qc['cache_rows']}",
            f"- matched nodes: {qc['matched_nodes']}",
            f"- missing coordinates: {qc['missing_coordinates']}",
            f"- global_node_index unique: {qc['global_node_index_unique']}",
            f"- global_node_index aligned: {qc['global_node_index_aligned']}",
            f"- tissue-space maps enabled: {qc['tissue_space_maps_enabled']}",
            "- cross-time physical arrows allowed: False",
            "- tissue-space arrows allowed: False",
            f"- only state-space visualization possible: {qc['state_space_only_visualization']}",
            "",
            "## M1 Scale Consistency",
            f"- M1 files checked: {len(scale_summary)}",
            f"- anchors checked: {int(scale_summary['anchors_checked'].sum()) if len(scale_summary) else 0}",
            f"- non-identical coordinate anchors: {int(scale_summary['nonidentical_coordinate_anchors'].sum()) if len(scale_summary) else 0}",
            "",
            "## Not Run",
            "- M0 h5ad fallback was not loaded.",
            "- M1/M2/M3/M4A/M4B/M4C outputs were not modified.",
            "- GPCCA, Branched NicheFlow / BranchSBM, M5, and regulator analysis were not run.",
            "",
            "## Runtime",
            f"- runtime seconds: {runtime_seconds:.3f}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def validate_cache_contract(cache: pd.DataFrame, qc: dict[str, Any]) -> None:
    if qc["cache_rows"] != qc["expected_rows"]:
        raise ValueError(f"Coordinate cache must have exactly {qc['expected_rows']} rows; found {qc['cache_rows']}.")
    if not qc["global_node_index_unique"]:
        raise ValueError("Coordinate cache global_node_index must be unique.")
    if not qc["global_node_index_aligned"]:
        raise ValueError("Coordinate cache global_node_index must remain aligned with M4A node identity.")


def run(args: argparse.Namespace) -> int:
    start = time.monotonic()
    config = load_config(args.config)
    paths = configured_paths(config)
    outputs = output_paths(paths)
    coord_cfg = config["coordinates"]
    expected_nodes = int(coord_cfg["expected_global_nodes"])
    coordinate_scale = str(coord_cfg["m1_coordinate_scale"])
    tolerance = float(coord_cfg["coordinate_consistency_tolerance"])

    audit = inspect_sources(paths)
    node_table = validate_m4a_node_table(pd.read_parquet(paths["m4a_node_table"], columns=M4_NODE_COLUMNS), expected_nodes)
    m1_coordinates, scale_summary = load_m1_coordinate_rows(paths["m1_by_slice_root"], coordinate_scale, tolerance)
    cache = build_coordinate_cache(node_table, m1_coordinates)
    qc = cache_qc(cache, expected_nodes)
    validate_cache_contract(cache, qc)
    audit = update_audit_rows(audit, qc)
    runtime = time.monotonic() - start

    atomic_write_parquet(outputs["coordinate_cache"], cache)
    audit_with_meta = audit.copy()
    audit_with_meta["generated_at_utc"] = utc_now_iso()
    audit_with_meta["coordinate_cache"] = str(outputs["coordinate_cache"])
    audit_with_meta["coordinate_scale_used"] = coordinate_scale
    audit_with_meta["m0_h5ad_loaded"] = False
    for key, value in NO_DOWNSTREAM_FLAGS.items():
        audit_with_meta[key] = value
    atomic_write_csv(outputs["audit_csv"], audit_with_meta)
    atomic_write_text(outputs["audit_md"], audit_report_text(audit_with_meta, scale_summary, qc, outputs["coordinate_cache"], runtime))

    print("M4D_00_COORDINATE_AVAILABILITY_AUDIT_COMPLETE")
    print(f"EXPECTED_ROWS {qc['expected_rows']}")
    print(f"CACHE_ROWS {qc['cache_rows']}")
    print(f"MATCHED_NODES {qc['matched_nodes']}")
    print(f"MISSING_COORDINATES {qc['missing_coordinates']}")
    print(f"TISSUE_SPACE_MAPS_ENABLED {qc['tissue_space_maps_enabled']}")
    print("CROSS_TIME_PHYSICAL_ARROWS_ALLOWED False")
    print("M0_H5AD_LOADED False")
    print(f"COORDINATE_CACHE {outputs['coordinate_cache']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
