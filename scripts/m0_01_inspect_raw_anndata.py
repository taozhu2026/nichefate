#!/usr/bin/env python
"""Inspect raw MERFISH AnnData files for the M0 workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from scipy import sparse

from nichefate.io import (
    expected_raw_files,
    file_size_gb,
    load_config,
    paths_from_config,
    read_h5ad,
)
from nichefate.qc import write_json_report, write_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m0_merfish_colitis.yaml")
    return parser.parse_args()


def value_counts(series, limit: int = 50) -> dict[str, int]:
    counts = series.value_counts(dropna=False).head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def unique_values(series, limit: int = 100) -> list[str]:
    return [str(value) for value in series.drop_duplicates().head(limit).tolist()]


def x_storage_summary(x_obj: Any) -> dict[str, Any]:
    return {
        "type": type(x_obj).__name__,
        "is_sparse": bool(sparse.issparse(x_obj)),
        "dtype": str(getattr(x_obj, "dtype", "unknown")),
        "shape": list(getattr(x_obj, "shape", [])),
    }


def write_partial(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = paths_from_config(config)
    raw_root = paths["raw_dir"]
    reports_dir = paths["output_dir"] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    required, _optional = expected_raw_files(config)
    h5ad_files = [name for name in required if name.endswith(".h5ad")]
    missing = [name for name in h5ad_files if not (raw_root / name).is_file()]
    if missing:
        print(f"Missing raw h5ad files under {raw_root}: {', '.join(missing)}")
        return 0

    required_fields = config["metadata"]["required_obs_fields"]
    optional_fields = config["metadata"]["optional_obs_fields"]
    partial_json = reports_dir / "raw_anndata_inspection.partial.json"
    report = {}
    lines = ["# Raw AnnData Inspection", ""]
    var_names_by_file: dict[str, set[str]] = {}
    for filename in h5ad_files:
        path = raw_root / filename
        print(f"Opening {filename} with anndata backed='r'...", flush=True)
        report[filename] = {"path": str(path), "status": "opening"}
        write_partial(report, partial_json)
        adata = read_h5ad(path, backed="r")
        print(f"Opened {filename}; collecting metadata...", flush=True)
        obs_columns = list(adata.obs.columns)
        var_names_by_file[filename] = set(map(str, adata.var_names))
        x_summary = x_storage_summary(adata.X)
        field_summaries: dict[str, Any] = {}
        for field in ("Sample_type", "Tier1", "Tier2", "Tier3", "Slice_ID"):
            if field in adata.obs:
                field_summaries[field] = {
                    "n_unique": int(adata.obs[field].nunique(dropna=False)),
                    "unique_values": unique_values(adata.obs[field]),
                    "counts": value_counts(adata.obs[field]),
                }
        coordinate_status = {}
        for field in config["spatial"]["coordinate_fields"]:
            coordinate_status[field] = {
                "exists": field in adata.obs,
                "dtype": str(adata.obs[field].dtype) if field in adata.obs else None,
                "numeric": bool(np.issubdtype(adata.obs[field].dtype, np.number))
                if field in adata.obs
                else False,
            }
        item = {
            "path": str(path),
            "size_gb": file_size_gb(path),
            "shape": [int(adata.n_obs), int(adata.n_vars)],
            "obs_columns": obs_columns,
            "var_names_count": int(len(adata.var_names)),
            "layers": list(adata.layers.keys()),
            "obsm_keys": list(adata.obsm.keys()),
            "uns_keys": list(adata.uns.keys()),
            "x": x_summary,
            "required_fields_present": {
                field: field in obs_columns for field in required_fields
            },
            "optional_fields_present": {
                field: field in obs_columns for field in optional_fields
            },
            "field_summaries": field_summaries,
            "coordinate_status": coordinate_status,
            "tier2_exists": "Tier2" in obs_columns,
            "status": "ok",
        }
        report[filename] = item
        write_partial(report, partial_json)
        lines.extend(
            [
                f"## {filename}",
                f"- Path: `{path}`",
                f"- Size GiB: {item['size_gb']:.3f}",
                f"- Shape: `{tuple(item['shape'])}`",
                f"- X: `{item['x']}`",
                f"- Layers: {item['layers']}",
                f"- obsm keys: {item['obsm_keys']}",
                f"- uns keys: {item['uns_keys']}",
                f"- Obs columns: {len(obs_columns)}",
                f"- Tier2 exists: {item['tier2_exists']}",
                f"- Required fields present: {item['required_fields_present']}",
                f"- Coordinate status: {item['coordinate_status']}",
                "",
            ]
        )
        for field, summary in field_summaries.items():
            lines.extend(
                [
                    f"### {filename} {field}",
                    f"- n_unique: {summary['n_unique']}",
                    f"- unique values: {summary['unique_values']}",
                    f"- top counts: {summary['counts']}",
                    "",
                ]
            )
        if hasattr(adata, "file"):
            adata.file.close()

    if len(var_names_by_file) == 2:
        names = list(var_names_by_file)
        intersection = var_names_by_file[names[0]].intersection(var_names_by_file[names[1]])
        report["var_intersection"] = {
            "files": names,
            "intersection_size": len(intersection),
        }
        lines.extend(
            [
                "## Gene Intersection",
                f"- Files: {names}",
                f"- Intersection size: {len(intersection)}",
                "",
            ]
        )

    write_json_report(report, reports_dir / "raw_anndata_inspection.json")
    write_markdown_report(lines, reports_dir / "raw_anndata_inspection.md")
    print(f"Wrote inspection reports to {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
