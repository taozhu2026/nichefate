"""Coordinate rescue and spatial QC helpers for PlanA-K metaniches."""

from __future__ import annotations

import getpass
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.sparse import csgraph

from .schemas import *
from .io import *
from .reporting import *
from .validation import *
from .kernel_qc import *
from .m2_inventory import *


def collect_metaniche_hardening_preflight_payload() -> tuple[pd.DataFrame, dict[str, Any]]:
    key_files = [
        METANICHE_REPORT_ROOT / "00_M2_5_METANICHE_PILOT_SUMMARY.md",
        METANICHE_REPORT_ROOT / "05_metaniche_qc.md",
        METANICHE_REPORT_ROOT / "06_next_sparse_k_pilot_design.md",
        DOC_ROOT / "05_m2_5_metaniche_pilot_protocol.md",
        PILOT_OUTPUT_ROOT / "anchor_to_metaniche.tsv",
        PILOT_OUTPUT_ROOT / "metaniche_table.tsv",
    ]
    inventory = pd.DataFrame([file_summary(path) for path in key_files])
    payload = {
        "generated_at_utc": utc_now(),
        "environment": {
            "hostname": platform.node() or "unknown",
            "date_utc": utc_now(),
            "user": getpass.getuser(),
            "pwd": str(Path.cwd()),
            "git_root": git_root(),
            "git_branch": git_branch(),
            "git_status_short": git_status_short(),
            "disk_usage": {
                "repo_root": disk_usage(PROJECT_ROOT),
                "/home": disk_usage(Path("/home")),
                "/data": disk_usage(Path("/data")) if Path("/data").exists() else None,
                "/ssd": disk_usage(Path("/ssd")) if Path("/ssd").exists() else None,
            },
            "memory": read_memory_info(),
        },
        "key_file_inventory": inventory.to_dict(orient="records"),
        "scope": {
            "goal": "Harden M2.5 metaniche state contract before sparse-K construction.",
            "primary_coordinate_source_hypothesis": "M1 by-slice niche_features parquet contains x/y.",
            "forbidden_actions": [
                "DARLIN processing",
                "raw data modification",
                "frozen P_fate modification",
                "full GPCCA",
                "BranchSBM training",
                "Slurm submission",
                "/ssd output",
            ],
        },
    }
    return inventory, payload


def pilot_anchor_map_path(pilot_root: Path = PILOT_OUTPUT_ROOT) -> Path:
    return pilot_root / "anchor_to_metaniche.tsv"


def load_pilot_anchor_map(pilot_root: Path = PILOT_OUTPUT_ROOT) -> pd.DataFrame:
    path = pilot_anchor_map_path(pilot_root)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def m1_niche_path_for_slice(slice_id: str, m1_root: Path = M1_BY_SLICE_ROOT) -> Path:
    return m1_root / slice_id / f"niche_features_{slice_id}.parquet"


def m2_representation_path_for_slice(slice_id: str, m2_root: Path = M2_BY_SLICE_ROOT) -> Path:
    return m2_root / slice_id / f"m2_representation_{slice_id}.parquet"


def inspect_table_columns(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "format": path.suffix.lstrip("."),
            "columns": [],
            "row_count": None,
            "inspection_error": "missing",
        }
    try:
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            return {
                "path": str(path),
                "exists": True,
                "format": "parquet",
                "columns": list(parquet.schema_arrow.names),
                "row_count": int(parquet.metadata.num_rows),
                "inspection_error": "",
            }
        if path.suffix in {".csv", ".tsv"}:
            sep = "\t" if path.suffix == ".tsv" else ","
            frame = pd.read_csv(path, sep=sep, nrows=5)
            return {
                "path": str(path),
                "exists": True,
                "format": path.suffix.lstrip("."),
                "columns": list(frame.columns),
                "row_count": None,
                "inspection_error": "",
            }
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "format": path.suffix.lstrip("."),
            "columns": [],
            "row_count": None,
            "inspection_error": str(exc),
        }
    return {
        "path": str(path),
        "exists": True,
        "format": path.suffix.lstrip("."),
        "columns": [],
        "row_count": None,
        "inspection_error": "unsupported format",
    }


def discover_coordinate_sources(
    pilot_root: Path = PILOT_OUTPUT_ROOT,
    m1_root: Path = M1_BY_SLICE_ROOT,
    m2_root: Path = M2_BY_SLICE_ROOT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pilot = load_pilot_anchor_map(pilot_root)
    pilot_slices = sorted(pilot["slice_id"].dropna().astype(str).unique()) if "slice_id" in pilot else []
    candidates: list[Path] = []
    for slice_id in pilot_slices:
        candidates.extend(
            [
                m1_niche_path_for_slice(slice_id, m1_root),
                m2_representation_path_for_slice(slice_id, m2_root),
            ]
        )
    for path in [
        SCRATCH_ROOT / "m0" / "reports" / "spatial_normalization_params.csv",
        SCRATCH_ROOT / "m0" / "reports" / "full_graph_preflight_summary.csv",
        SCRATCH_ROOT / "m1" / "reports" / "m1_global_schema.json",
        SCRATCH_ROOT / "m1" / "reports" / "m1_global_schema.md",
        SCRATCH_ROOT / "m1" / "reports" / "m1_m0_input_audit.csv",
        SCRATCH_ROOT / "m2" / "reports" / "m2_full_feature_schema.json",
    ]:
        candidates.append(path)

    rows: list[dict[str, Any]] = []
    pilot_keys = {"slice_id", "anchor_index", "anchor_cell_id"}
    for path in dict.fromkeys(candidates):
        info = inspect_table_columns(path)
        columns = list(info["columns"])
        column_set = set(columns)
        coord_cols = [
            column
            for column in columns
            if column.lower() in {"x", "y"} or "coord" in column.lower() or "spatial" in column.lower()
        ]
        join_keys = [column for column in ["anchor_id", "slice_id", "anchor_index", "anchor_cell_id", "cell_id", "spot_id"] if column in column_set]
        slice_cols = [column for column in ["slice_id", "slice_file", "time", "time_day", "mouse_id", "scale"] if column in column_set]
        can_join = bool({"x", "y"}.issubset(column_set) and pilot_keys.issubset(column_set))
        rows.append(
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
                "format": info["format"],
                "row_count_if_cheap": info["row_count"],
                "columns": ";".join(columns[:60]),
                "likely_join_keys": ";".join(join_keys),
                "coordinate_columns": ";".join(coord_cols),
                "slice_timepoint_columns": ";".join(slice_cols),
                "can_join_to_m2_pilot_anchor_ids": can_join,
                "inspection_note": info["inspection_error"] or "schema/header inspected only",
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "generated_at_utc": utc_now(),
        "pilot_anchor_count": int(len(pilot)),
        "pilot_slice_count": int(len(pilot_slices)),
        "pilot_slices": pilot_slices,
        "candidate_file_count": int(len(frame)),
        "joinable_coordinate_file_count": int(frame["can_join_to_m2_pilot_anchor_ids"].sum()) if not frame.empty else 0,
        "primary_coordinate_source": "M1 by-slice niche_features parquet",
        "recommended_join_key": "slice_id + anchor_index + anchor_cell_id",
    }
    return frame, summary


def coordinate_source_inventory_markdown(frame: pd.DataFrame, summary: dict[str, Any]) -> str:
    preview = frame[
        [
            "path",
            "format",
            "row_count_if_cheap",
            "likely_join_keys",
            "coordinate_columns",
            "can_join_to_m2_pilot_anchor_ids",
            "inspection_note",
        ]
    ].copy()
    return dedent(
        f"""
        # Coordinate Source Inventory

        - Pilot anchors: {summary["pilot_anchor_count"]:,}
        - Pilot slices: {summary["pilot_slice_count"]}
        - Candidate files inspected: {summary["candidate_file_count"]}
        - Joinable coordinate files: {summary["joinable_coordinate_file_count"]}
        - Primary source: {summary["primary_coordinate_source"]}
        - Recommended join key: `{summary["recommended_join_key"]}`

        {dataframe_to_markdown(preview)}
        """
    ).strip() + "\n"


def load_m1_coordinates_for_slices(
    slice_ids: list[str],
    m1_root: Path = M1_BY_SLICE_ROOT,
    scale: str = "radius_x2",
) -> pd.DataFrame:
    columns = [
        "slice_id",
        "scale",
        "anchor_index",
        "anchor_cell_id",
        "time",
        "time_day",
        "mouse_id",
        "cell_type_l1",
        "cell_type_l2",
        "cell_type_l3",
        "x",
        "y",
    ]
    chunks: list[pd.DataFrame] = []
    for slice_id in slice_ids:
        path = m1_niche_path_for_slice(slice_id, m1_root)
        if not path.exists():
            continue
        parquet = pq.ParquetFile(path)
        available = [column for column in columns if column in parquet.schema_arrow.names]
        if not {"slice_id", "anchor_index", "anchor_cell_id", "x", "y"}.issubset(available):
            continue
        table = pq.read_table(path, columns=available)
        frame = table.to_pandas()
        if "scale" in frame.columns and scale in set(frame["scale"].astype(str)):
            frame = frame[frame["scale"].astype(str) == scale].copy()
        frame = frame.drop_duplicates(["slice_id", "anchor_index", "anchor_cell_id"])
        frame["source_m1_path"] = str(path)
        chunks.append(frame)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def audit_coordinate_join_keys(
    pilot_root: Path = PILOT_OUTPUT_ROOT,
    m1_root: Path = M1_BY_SLICE_ROOT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pilot = load_pilot_anchor_map(pilot_root)
    if pilot.empty or "slice_id" not in pilot.columns:
        frame = pd.DataFrame(
            [
                {
                    "candidate_key": "none",
                    "key_available": False,
                    "pilot_unique_rate": 0.0,
                    "coordinate_unique_rate": 0.0,
                    "overlap_rate": 0.0,
                    "duplicate_rate": 1.0,
                    "missing_rate_after_join": 1.0,
                    "status": "FAIL",
                    "reason": "pilot anchor map missing",
                }
            ]
        )
        return frame, {"safe_join_identified": False, "best_join_key": None}
    slice_ids = sorted(pilot["slice_id"].astype(str).unique())
    coords = load_m1_coordinates_for_slices(slice_ids, m1_root=m1_root)
    key_options = [
        ("slice_id+anchor_index+anchor_cell_id", ["slice_id", "anchor_index", "anchor_cell_id"]),
        ("slice_id+anchor_index", ["slice_id", "anchor_index"]),
        ("slice_id+anchor_cell_id", ["slice_id", "anchor_cell_id"]),
        ("anchor_id", ["anchor_id"]),
    ]
    rows: list[dict[str, Any]] = []
    for name, keys in key_options:
        available = all(key in pilot.columns for key in keys) and all(key in coords.columns for key in keys)
        if name == "anchor_id" and "anchor_id" in pilot.columns and not coords.empty:
            coords = coords.copy()
            coords["anchor_id"] = coords["slice_id"].astype(str) + "::" + coords["anchor_index"].astype(str)
            available = True
        if not available or coords.empty:
            rows.append(
                {
                    "candidate_key": name,
                    "key_available": False,
                    "pilot_unique_rate": 0.0,
                    "coordinate_unique_rate": 0.0,
                    "overlap_rate": 0.0,
                    "duplicate_rate": 1.0,
                    "missing_rate_after_join": 1.0,
                    "status": "FAIL",
                    "reason": "required key columns missing",
                }
            )
            continue
        pilot_keys = pilot[keys].astype(str).agg("||".join, axis=1)
        coord_keys = coords[keys].astype(str).agg("||".join, axis=1)
        overlap = pilot_keys.isin(set(coord_keys))
        duplicate_rate = float(coord_keys.duplicated().sum() / len(coord_keys)) if len(coord_keys) else 1.0
        missing_rate = float(1.0 - overlap.mean()) if len(overlap) else 1.0
        status = "PASS" if missing_rate == 0.0 and duplicate_rate == 0.0 else "WARN"
        rows.append(
            {
                "candidate_key": name,
                "key_available": True,
                "pilot_unique_rate": float(pilot_keys.nunique() / len(pilot_keys)) if len(pilot_keys) else 0.0,
                "coordinate_unique_rate": float(coord_keys.nunique() / len(coord_keys)) if len(coord_keys) else 0.0,
                "overlap_rate": float(overlap.mean()) if len(overlap) else 0.0,
                "duplicate_rate": duplicate_rate,
                "missing_rate_after_join": missing_rate,
                "status": status,
                "reason": "safe exact pilot overlap" if status == "PASS" else "join needs review",
            }
        )
    frame = pd.DataFrame(rows)
    pass_rows = frame[frame["status"] == "PASS"].copy()
    best = None if pass_rows.empty else str(pass_rows.iloc[0]["candidate_key"])
    summary = {
        "generated_at_utc": utc_now(),
        "pilot_anchor_count": int(len(pilot)),
        "coordinate_anchor_count": int(len(coords)),
        "safe_join_identified": best is not None,
        "best_join_key": best,
        "preferred_join_key": "slice_id+anchor_index+anchor_cell_id",
    }
    return frame, summary


def join_key_audit_markdown(frame: pd.DataFrame, summary: dict[str, Any]) -> str:
    return dedent(
        f"""
        # Coordinate Join-Key Audit

        - Pilot anchors: {summary.get("pilot_anchor_count", 0):,}
        - Coordinate anchors loaded from M1: {summary.get("coordinate_anchor_count", 0):,}
        - Safe join identified: {summary.get("safe_join_identified")}
        - Best join key: `{summary.get("best_join_key")}`
        - Preferred join key: `{summary.get("preferred_join_key")}`

        {dataframe_to_markdown(frame)}
        """
    ).strip() + "\n"


def run_coordinate_join_preview(
    output_dir: Path,
    pilot_root: Path = PILOT_OUTPUT_ROOT,
    m1_root: Path = M1_BY_SLICE_ROOT,
    overwrite: bool = False,
    dry_run: bool = True,
) -> dict[str, Any]:
    preview_dir = ensure_dir(output_dir / "coordinate_join_preview")
    pilot = load_pilot_anchor_map(pilot_root)
    if pilot.empty:
        return {
            "generated_at_utc": utc_now(),
            "coordinate_join_run": False,
            "dry_run": dry_run,
            "safe_join_identified": False,
            "reason": "pilot anchor map missing",
        }
    slice_ids = sorted(pilot["slice_id"].astype(str).unique())
    coords = load_m1_coordinates_for_slices(slice_ids, m1_root=m1_root)
    if coords.empty:
        return {
            "generated_at_utc": utc_now(),
            "coordinate_join_run": False,
            "dry_run": dry_run,
            "safe_join_identified": False,
            "reason": "M1 coordinates missing for pilot slices",
        }
    join_keys = ["slice_id", "anchor_index", "anchor_cell_id"]
    missing_keys = [key for key in join_keys if key not in pilot.columns or key not in coords.columns]
    if missing_keys:
        return {
            "generated_at_utc": utc_now(),
            "coordinate_join_run": False,
            "dry_run": dry_run,
            "safe_join_identified": False,
            "reason": f"missing join keys: {missing_keys}",
        }
    coord_cols = [
        "slice_id",
        "anchor_index",
        "anchor_cell_id",
        "x",
        "y",
        "source_m1_path",
    ]
    joined = pilot.merge(
        coords[coord_cols].drop_duplicates(join_keys),
        on=join_keys,
        how="left",
        validate="one_to_one",
    )
    missing_rate = float(joined[["x", "y"]].isna().any(axis=1).mean()) if len(joined) else 1.0
    safe = missing_rate == 0.0
    metaniche_coordinates = pd.DataFrame()
    if safe:
        metaniche_coordinates = (
            joined.groupby("metaniche_id", dropna=False)
            .agg(
                anchor_count=("anchor_id", "size"),
                x_centroid=("x", "mean"),
                y_centroid=("y", "mean"),
                x_var=("x", "var"),
                y_var=("y", "var"),
                dominant_slice_id=("slice_id", lambda s: s.astype(str).value_counts().index[0]),
                dominant_time_day=("time_day", lambda s: s.value_counts().index[0]),
            )
            .reset_index()
        )
    payload = {
        "generated_at_utc": utc_now(),
        "coordinate_join_run": not dry_run,
        "dry_run": dry_run,
        "safe_join_identified": safe,
        "join_key": "slice_id + anchor_index + anchor_cell_id",
        "pilot_anchor_count": int(len(pilot)),
        "joined_anchor_count": int(len(joined)),
        "coordinate_missing_rate": missing_rate,
        "metaniche_coordinate_count": int(len(metaniche_coordinates)),
        "output_files": [
            str(preview_dir / "anchor_coordinates.preview.tsv"),
            str(preview_dir / "metaniche_coordinates.preview.tsv"),
            str(preview_dir / "coordinate_join_qc.json"),
        ],
    }
    if not dry_run and safe:
        atomic_write_tsv(preview_dir / "anchor_coordinates.preview.tsv", joined, overwrite=overwrite)
        atomic_write_tsv(
            preview_dir / "metaniche_coordinates.preview.tsv",
            metaniche_coordinates,
            overwrite=overwrite,
        )
        atomic_write_json(preview_dir / "coordinate_join_qc.json", payload, overwrite=overwrite)
    return payload


def coordinate_join_preview_markdown(payload: dict[str, Any]) -> str:
    return dedent(
        f"""
        # Coordinate Join Preview

        - Coordinate join run: {payload.get("coordinate_join_run")}
        - Dry run: {payload.get("dry_run")}
        - Safe join identified: {payload.get("safe_join_identified")}
        - Join key: `{payload.get("join_key")}`
        - Pilot anchors: {payload.get("pilot_anchor_count", 0):,}
        - Joined anchors: {payload.get("joined_anchor_count", 0):,}
        - Coordinate missing rate: {payload.get("coordinate_missing_rate")}
        - Metaniche coordinate rows: {payload.get("metaniche_coordinate_count", 0):,}
        - Reason/blocker: {payload.get("reason", "none")}
        """
    ).strip() + "\n"


def compute_spatial_compactness_qc(output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = output_dir / "coordinate_join_preview" / "anchor_coordinates.preview.tsv"
    if not path.exists():
        frame = pd.DataFrame(
            [
                {
                    "metaniche_id": "blocked",
                    "anchor_count": 0,
                    "radius_mean": None,
                    "radius_median": None,
                    "radius_p90": None,
                    "radius_max": None,
                    "x_var": None,
                    "y_var": None,
                    "diffuse_flag": True,
                    "status": "BLOCKED",
                }
            ]
        )
        return frame, {
            "generated_at_utc": utc_now(),
            "spatial_compactness_available": False,
            "reason": "coordinate preview output missing",
        }
    anchors = pd.read_csv(path, sep="\t")
    if anchors.empty or not {"metaniche_id", "x", "y"}.issubset(anchors.columns):
        frame = pd.DataFrame()
        return frame, {
            "generated_at_utc": utc_now(),
            "spatial_compactness_available": False,
            "reason": "coordinate columns missing in preview",
        }
    rows: list[dict[str, Any]] = []
    for metaniche_id, group in anchors.groupby("metaniche_id", dropna=False):
        xy = group[["x", "y"]].to_numpy(dtype=float)
        center = xy.mean(axis=0)
        dist = np.linalg.norm(xy - center, axis=1)
        slice_counts = group["slice_id"].astype(str).value_counts() if "slice_id" in group else pd.Series(dtype=int)
        time_counts = group["time_day"].astype(str).value_counts() if "time_day" in group else pd.Series(dtype=int)
        rows.append(
            {
                "metaniche_id": metaniche_id,
                "anchor_count": int(len(group)),
                "radius_mean": float(np.mean(dist)),
                "radius_median": float(np.median(dist)),
                "radius_p90": float(np.quantile(dist, 0.90)),
                "radius_max": float(np.max(dist)),
                "x_var": float(np.var(xy[:, 0])),
                "y_var": float(np.var(xy[:, 1])),
                "dominant_slice_id": slice_counts.index[0] if not slice_counts.empty else None,
                "slice_purity": float(slice_counts.iloc[0] / slice_counts.sum()) if not slice_counts.empty else None,
                "dominant_time_day": time_counts.index[0] if not time_counts.empty else None,
                "time_day_purity": float(time_counts.iloc[0] / time_counts.sum()) if not time_counts.empty else None,
            }
        )
    frame = pd.DataFrame(rows)
    median_p90 = float(frame["radius_p90"].median()) if not frame.empty else 0.0
    global_p90 = float(frame["radius_p90"].quantile(0.90)) if not frame.empty else 0.0
    threshold = max(2.0 * median_p90, global_p90)
    frame["diffuse_flag"] = frame["radius_p90"] > threshold
    frame["status"] = np.where(frame["diffuse_flag"], "WARN", "PASS")
    payload = {
        "generated_at_utc": utc_now(),
        "spatial_compactness_available": True,
        "metaniche_count": int(len(frame)),
        "diffuse_metaniche_count": int(frame["diffuse_flag"].sum()),
        "radius_p90_median": median_p90,
        "diffuse_threshold_radius_p90": threshold,
        "slice_wise_radius_p90_median": frame.groupby("dominant_slice_id")["radius_p90"].median().to_dict(),
        "time_slice_purity_vs_spatial_compactness_note": (
            "Review WARN rows jointly with low time/slice purity before sparse-K."
        ),
    }
    return frame, payload


def spatial_compactness_markdown(frame: pd.DataFrame, payload: dict[str, Any]) -> str:
    if not payload.get("spatial_compactness_available"):
        return dedent(
            f"""
            # Spatial Compactness QC

            Spatial compactness is blocked.

            Reason: {payload.get("reason")}
            """
        ).strip() + "\n"
    preview = frame.sort_values(["diffuse_flag", "radius_p90"], ascending=[False, False]).head(20)
    return dedent(
        f"""
        # Spatial Compactness QC

        - Metaniches: {payload["metaniche_count"]}
        - Diffuse metaniches: {payload["diffuse_metaniche_count"]}
        - Median radius p90: {payload["radius_p90_median"]:.4g}
        - Diffuse threshold radius p90: {payload["diffuse_threshold_radius_p90"]:.4g}

        {dataframe_to_markdown(preview)}
        """
    ).strip() + "\n"


__all__ = [name for name in globals() if not name.startswith("__")]
