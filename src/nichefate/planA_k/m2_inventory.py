"""M2 inventory and feature group helpers for PlanA-K."""

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


def collect_metaniche_preflight_payload() -> tuple[pd.DataFrame, dict[str, Any]]:
    key_files = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "AGENTS.md",
        REPORT_ROOT / "00_PLAN_A_K_GPCCA_REDESIGN_SUMMARY.md",
        REPORT_ROOT / "00_PLAN_A_K_GPCCA_REDESIGN_SUMMARY.json",
        DOC_ROOT / "04_niche_state_coarsening_design.md",
        REPORT_ROOT / "05_niche_state_coarsening_design.md",
        REPORT_ROOT / "05_metaniche_contract.tsv",
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
            "goal": "Conservative M2.5 metaniche coarsening pilot for PlanA-K.",
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


def load_m2_feature_schema(schema_path: Path = M2_SCHEMA_PATH) -> dict[str, Any]:
    if not schema_path.exists():
        return {
            "schema_path": str(schema_path),
            "exists": False,
            "metadata_columns": [],
            "numeric_feature_columns": [],
            "output_columns": [],
            "expected_scales": [],
            "metadata_column_count": 0,
            "numeric_feature_column_count": 0,
            "output_column_count": 0,
        }
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    data["schema_path"] = str(schema_path)
    data["exists"] = True
    return data


def infer_time_day_from_slice_id(slice_id: str | None) -> int | None:
    if not slice_id:
        return None
    match = re.search(r"_D(\d+)_", f"_{slice_id}_")
    if not match:
        return None
    return int(match.group(1))


def first_metadata_row(path: Path, metadata_columns: list[str]) -> dict[str, Any]:
    if not metadata_columns:
        return {}
    try:
        table = pq.read_table(path, columns=metadata_columns).slice(0, 1)
        frame = table.to_pandas()
    except Exception:
        return {}
    if frame.empty:
        return {}
    return {str(key): json_safe(value) for key, value in frame.iloc[0].to_dict().items()}


def discover_m2_inventory(
    m2_root: Path = M2_BY_SLICE_ROOT,
    schema_path: Path = M2_SCHEMA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    schema = load_m2_feature_schema(schema_path)
    metadata_columns = list(schema.get("metadata_columns", []))
    completed_path = m2_root / "completed_slices.csv"
    candidates: list[dict[str, Any]] = []
    if completed_path.exists():
        completed = pd.read_csv(completed_path)
        for row in completed.to_dict(orient="records"):
            output_path = Path(str(row.get("output_path", "")))
            if output_path.exists():
                candidates.append(row)
    else:
        for path in sorted(m2_root.glob("*/m2_representation_*.parquet")):
            candidates.append({"output_path": str(path), "slice_id": path.parent.name})

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        path = Path(str(candidate.get("output_path", "")))
        slice_id = str(candidate.get("slice_id") or path.parent.name)
        row: dict[str, Any] = {
            "path": str(path),
            "file_size_bytes": int(path.stat().st_size) if path.exists() else 0,
            "format": path.suffix.lstrip(".") or "unknown",
            "slice_id": slice_id,
            "timepoint": None,
            "time_day": infer_time_day_from_slice_id(slice_id),
            "rows": None,
            "columns": None,
            "row_groups": None,
            "metadata_columns": "",
            "numeric_feature_columns_count": None,
            "status": str(candidate.get("status", "unknown")),
            "safe_to_sample": False,
            "inspection_note": "",
        }
        if not path.exists():
            row["inspection_note"] = "missing output_path"
            rows.append(row)
            continue
        try:
            parquet = pq.ParquetFile(path)
            names = list(parquet.schema_arrow.names)
            present_metadata = [column for column in metadata_columns if column in names]
            first_row = first_metadata_row(path, present_metadata)
            time_value = first_row.get("time")
            row.update(
                {
                    "timepoint": time_value,
                    "time_day": first_row.get("time_day") or row["time_day"],
                    "mouse_id": first_row.get("mouse_id"),
                    "rows": int(parquet.metadata.num_rows),
                    "columns": int(len(names)),
                    "row_groups": int(parquet.metadata.num_row_groups),
                    "metadata_columns": ";".join(present_metadata),
                    "numeric_feature_columns_count": int(len(names) - len(present_metadata)),
                    "safe_to_sample": True,
                    "inspection_note": "Parquet metadata and one metadata row inspected only",
                }
            )
        except Exception as exc:
            row["inspection_note"] = f"metadata inspection failed: {exc}"
        rows.append(row)

    frame = pd.DataFrame(rows)
    summary = {
        "generated_at_utc": utc_now(),
        "m2_root": str(m2_root),
        "schema_path": str(schema_path),
        "schema_exists": bool(schema.get("exists")),
        "completed_slices_path": str(completed_path),
        "completed_slices_exists": completed_path.exists(),
        "m2_file_count": int(len(frame)),
        "safe_to_sample_count": int(frame["safe_to_sample"].sum()) if not frame.empty else 0,
        "total_rows_from_metadata": int(frame["rows"].dropna().sum()) if not frame.empty else 0,
        "total_bytes": int(frame["file_size_bytes"].sum()) if not frame.empty else 0,
        "metadata_columns": schema.get("metadata_columns", []),
        "numeric_feature_column_count": schema.get("numeric_feature_column_count", 0),
        "expected_scales": schema.get("expected_scales", []),
    }
    return frame, summary


def m2_inventory_markdown(frame: pd.DataFrame, summary: dict[str, Any]) -> str:
    preview_columns = [
        "slice_id",
        "timepoint",
        "time_day",
        "mouse_id",
        "rows",
        "columns",
        "file_size_bytes",
        "safe_to_sample",
    ]
    preview = frame[[column for column in preview_columns if column in frame.columns]].head(12)
    return dedent(
        f"""
        # M2 Output Inventory

        ## Summary

        - M2 root: `{summary["m2_root"]}`
        - Schema path: `{summary["schema_path"]}`
        - Schema exists: {summary["schema_exists"]}
        - M2 files indexed: {summary["m2_file_count"]}
        - Files safe to sample: {summary["safe_to_sample_count"]}
        - Rows from Parquet metadata: {summary["total_rows_from_metadata"]:,}
        - Total bytes from indexed files: {summary["total_bytes"]:,}
        - Numeric feature columns in schema: {summary["numeric_feature_column_count"]}
        - Expected scales: {", ".join(summary["expected_scales"])}

        The inventory uses `completed_slices.csv`, Parquet metadata, and one
        metadata row per file. It does not load the full 4GB+ M2 output set.

        ## Preview

        {dataframe_to_markdown(preview)}
        """
    ).strip() + "\n"


def strip_m2_scale_prefix(column: str) -> tuple[str | None, str]:
    parts = column.split("__", 1)
    if len(parts) == 2 and parts[0].startswith("radius_"):
        return parts[0], parts[1]
    return None, column


def classify_m2_column(column: str, metadata_columns: set[str]) -> str:
    if column in metadata_columns:
        if column in {"slice_id", "slice_file", "time", "time_day", "mouse_id"}:
            return "time_slice_mouse_metadata"
        if column in {"cell_type_l1", "cell_type_l2", "cell_type_l3"}:
            return "cell_type_annotation_metadata"
        return "anchor_identity_metadata"
    _, base = strip_m2_scale_prefix(column)
    if base.startswith(("ct_l1__", "ct_l2__", "ct_l3__")):
        return "niche_composition_features"
    if base.endswith("_entropy"):
        return "entropy_features"
    if base.startswith("emb_mean_pc"):
        return "embedding_mean_features"
    if base.startswith("emb_var_pc"):
        return "embedding_variance_features"
    if base == "n_neighbors":
        return "neighborhood_count_features"
    if any(token in base for token in ["distance", "density", "topology"]):
        return "spatial_topology_density_features"
    if "fate" in base.lower() or "endpoint" in base.lower() or "darlin" in base.lower():
        return "excluded_leakage_or_deferred_features"
    return "technical_or_unknown_features"


def classify_m2_feature_groups(schema: dict[str, Any]) -> pd.DataFrame:
    metadata_columns = set(schema.get("metadata_columns", []))
    output_columns = list(schema.get("output_columns", []))
    grouped: dict[str, list[str]] = {}
    for column in output_columns:
        grouped.setdefault(classify_m2_column(column, metadata_columns), []).append(column)

    decision = {
        "anchor_identity_metadata": (
            "no",
            "yes",
            "no",
            "high",
            "Anchor/cell identifiers are provenance fields, not biological state features.",
        ),
        "time_slice_mouse_metadata": (
            "no",
            "yes",
            "no",
            "high",
            "Time, slice, and mouse labels must be preserved for QC and transition design but not clustered on directly.",
        ),
        "cell_type_annotation_metadata": (
            "no",
            "yes",
            "no",
            "medium",
            "Anchor cell-type labels are useful annotations but can dominate or leak identity if used as clustering inputs.",
        ),
        "niche_composition_features": (
            "yes",
            "yes",
            "yes",
            "low",
            "Neighborhood cell-type composition is a core microenvironment signal.",
        ),
        "entropy_features": (
            "yes",
            "yes",
            "yes",
            "low",
            "Entropy summarizes neighborhood heterogeneity and should help separate niche states.",
        ),
        "embedding_mean_features": (
            "yes",
            "yes",
            "yes",
            "low",
            "Mean expression/latent PCs summarize molecular state of the neighborhood.",
        ),
        "embedding_variance_features": (
            "maybe",
            "yes",
            "yes",
            "medium",
            "Variance PCs may capture heterogeneity but can dominate distances in the first pilot.",
        ),
        "neighborhood_count_features": (
            "maybe",
            "yes",
            "yes",
            "medium",
            "Neighbor count reflects density and scale effects; use only after checking feature domination.",
        ),
        "spatial_topology_density_features": (
            "maybe",
            "yes",
            "yes",
            "medium",
            "Spatial topology is biologically relevant but should be balanced against composition and embedding features.",
        ),
        "excluded_leakage_or_deferred_features": (
            "no",
            "yes",
            "no",
            "high",
            "Endpoint, fate, or DARLIN-derived fields are excluded from this pseudo-only M2.5 pilot.",
        ),
        "technical_or_unknown_features": (
            "no",
            "maybe",
            "maybe",
            "medium",
            "Unknown columns require manual review before clustering.",
        ),
    }

    rows: list[dict[str, Any]] = []
    for group_name in sorted(grouped):
        columns = grouped[group_name]
        use, annotation, scaling, leakage, rationale = decision[group_name]
        scales = sorted({strip_m2_scale_prefix(column)[0] for column in columns if strip_m2_scale_prefix(column)[0]})
        rows.append(
            {
                "feature_group": group_name,
                "column_count": len(columns),
                "example_columns": ";".join(columns[:8]),
                "scales": ";".join(scales),
                "use_for_coarsening": use,
                "use_for_annotation_only": annotation,
                "scaling_required": scaling,
                "leakage_risk": leakage,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def feature_group_audit_markdown(frame: pd.DataFrame, schema: dict[str, Any]) -> str:
    safe_count = int(frame.loc[frame["use_for_coarsening"] == "yes", "column_count"].sum()) if not frame.empty else 0
    maybe_count = int(frame.loc[frame["use_for_coarsening"] == "maybe", "column_count"].sum()) if not frame.empty else 0
    no_count = int(frame.loc[frame["use_for_coarsening"] == "no", "column_count"].sum()) if not frame.empty else 0
    return dedent(
        f"""
        # M2 Feature Group And Metadata Audit

        ## Summary

        - Schema exists: {schema.get("exists", False)}
        - Row granularity: {schema.get("row_granularity", "unknown")}
        - Metadata columns: {schema.get("metadata_column_count", 0)}
        - Numeric feature columns: {schema.get("numeric_feature_column_count", 0)}
        - Default safe coarsening columns: {safe_count}
        - Maybe-later coarsening columns: {maybe_count}
        - Annotation-only / excluded columns: {no_count}

        Default `safe` mode uses neighborhood composition, entropy, and
        embedding-mean features. It preserves time, slice, mouse, anchor, and
        cell-type labels for annotation and QC rather than clustering on them.

        {dataframe_to_markdown(frame)}
        """
    ).strip() + "\n"


def select_m2_feature_columns(
    schema: dict[str, Any],
    feature_mode: str = "safe",
) -> list[str]:
    metadata_columns = set(schema.get("metadata_columns", []))
    columns = list(schema.get("output_columns", []))
    selected: list[str] = []
    for column in columns:
        group = classify_m2_column(column, metadata_columns)
        if feature_mode == "safe" and group in {
            "niche_composition_features",
            "entropy_features",
            "embedding_mean_features",
        }:
            selected.append(column)
        elif feature_mode == "embedding_only" and group in {
            "entropy_features",
            "embedding_mean_features",
        }:
            selected.append(column)
        elif feature_mode == "composition_only" and group in {
            "niche_composition_features",
            "entropy_features",
        }:
            selected.append(column)
        elif feature_mode == "all_safe" and group in {
            "niche_composition_features",
            "entropy_features",
            "embedding_mean_features",
            "embedding_variance_features",
            "neighborhood_count_features",
            "spatial_topology_density_features",
        }:
            selected.append(column)
    return selected


__all__ = [name for name in globals() if not name.startswith("__")]
