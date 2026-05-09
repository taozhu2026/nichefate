#!/usr/bin/env python
"""Create M0 QC reports."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from nichefate.download import check_core_files
from nichefate.io import ensure_dirs, load_config, read_h5ad
from nichefate.qc import compute_obs_summary, summarize_anndata, write_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m0_merfish_colitis.yaml")
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _file_count(path: Path, pattern: str) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.glob(pattern))


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _format_gib(size_bytes: int) -> str:
    return f"{size_bytes / (1024**3):.2f} GiB"


def _read_shape(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    adata = read_h5ad(path, backed="r")
    try:
        return int(adata.n_obs), int(adata.n_vars)
    finally:
        if hasattr(adata, "file"):
            adata.file.close()


def _status(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def _append_status(lines: list[str], label: str, ok: bool, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    lines.append(f"- {label}: {_status(ok)}{suffix}")


def _summarize_raw_verification(path: Path) -> str:
    report = _read_json(path)
    if not isinstance(report, dict):
        return "not available"
    rows = report.get("files", [])
    if not isinstance(rows, list):
        return "invalid report format"
    ok = sum(1 for row in rows if isinstance(row, dict) and row.get("ok") is True)
    return f"{ok}/{len(rows)} files passed"


def _summarize_raw_inspection(path: Path) -> str:
    report = _read_json(path)
    if not isinstance(report, dict):
        return "not available"
    return f"{len(report)} raw AnnData entries inspected"


def _append_counts(lines: list[str], title: str, counts: dict[str, int]) -> None:
    lines.extend([f"## {title}", ""])
    if not counts:
        lines.append("Not available.")
    else:
        for key, value in counts.items():
            lines.append(f"- {key}: {value}")
    lines.append("")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dirs = ensure_dirs(config)
    metadata_path = dirs["processed"] / "m0_all_colitis_merfish.metadata_spatial.h5ad"
    final_path = dirs["processed"] / "m0_all_colitis_merfish.final.h5ad"
    embedded_path = dirs["processed"] / "m0_all_colitis_merfish.embedded.h5ad"
    by_slice_dir = dirs["by_slice"]
    by_time_dir = dirs["by_time"]
    graph_summary_path = dirs["reports"] / "graph_degree_summary.csv"
    failed_slices_path = dirs["logs"] / "failed_slices.txt"
    norm_path = dirs["reports"] / "spatial_normalization_params.csv"
    pca_path = dirs["reports"] / "pca_variance_ratio.csv"
    raw_verify_path = dirs["reports"] / "raw_file_verification.json"
    raw_inspection_path = dirs["reports"] / "raw_anndata_inspection.json"
    target = final_path if final_path.is_file() else embedded_path

    metadata_shape = _read_shape(metadata_path)
    embedded_shape = _read_shape(embedded_path)
    final_shape = _read_shape(final_path)
    by_slice_count = _file_count(by_slice_dir, "*.m0.h5ad")
    by_time_files = sorted(by_time_dir.glob("D*.h5ad")) if by_time_dir.is_dir() else []
    by_time_shapes = {path.stem: _read_shape(path) for path in by_time_files}
    by_time_counts = {
        key: shape[0]
        for key, shape in by_time_shapes.items()
        if shape is not None
    }
    failed_text = (
        failed_slices_path.read_text(encoding="utf-8").strip()
        if failed_slices_path.is_file()
        else ""
    )
    pca_rows = len(pd.read_csv(pca_path)) if pca_path.is_file() else 0
    graph_rows = pd.read_csv(graph_summary_path) if graph_summary_path.is_file() else pd.DataFrame()
    graph_methods = sorted(graph_rows["graph_name"].unique().tolist()) if "graph_name" in graph_rows else []
    delaunay_rows = graph_rows[graph_rows["graph_name"] == "delaunay"] if "graph_name" in graph_rows else pd.DataFrame()
    empty_delaunay = (
        delaunay_rows[delaunay_rows["n_edges"] == 0]
        if "n_edges" in delaunay_rows
        else pd.DataFrame()
    )

    lines = [
        "# M0 QC Report",
        "",
        "No lineage barcode is present; analyses downstream of M0 are pseudo-lineage only.",
        "M1 niche construction has not started.",
        "",
        "## Pipeline Status",
        "",
    ]
    _append_status(
        lines,
        "Raw file verification",
        raw_verify_path.is_file(),
        _summarize_raw_verification(raw_verify_path),
    )
    _append_status(
        lines,
        "Raw AnnData inspection",
        raw_inspection_path.is_file(),
        _summarize_raw_inspection(raw_inspection_path),
    )
    _append_status(
        lines,
        "Full metadata/spatial build",
        metadata_shape is not None,
        f"shape {metadata_shape}" if metadata_shape else "",
    )
    _append_status(
        lines,
        "Full PCA",
        embedded_shape is not None and pca_rows > 0,
        f"shape {embedded_shape}; PCA rows {pca_rows}" if embedded_shape else "",
    )
    _append_status(
        lines,
        "Full by-slice graph construction",
        by_slice_count == 58 and graph_summary_path.is_file() and failed_text == "",
        f"{by_slice_count} slice files; graph summary rows {len(graph_rows)}; failed_slices empty={failed_text == ''}",
    )
    _append_status(
        lines,
        "Final export",
        final_shape is not None and len(by_time_counts) == 5,
        f"final shape {final_shape}; by_time files {len(by_time_counts)}",
    )
    lines.append("")

    lines.extend(["## Input File Status", ""])
    for row in check_core_files(config):
        lines.append(f"- {row['filename']}: {row['status']} ({row['path']})")
    lines.append("")

    if target.is_file():
        adata = read_h5ad(target, backed="r")
        try:
            summary = summarize_anndata(adata)
            obs_summary = compute_obs_summary(adata)
            lines.extend(
                [
                    "## AnnData",
                    f"- File: `{target}`",
                    f"- Shape: `{summary['n_obs']} x {summary['n_vars']}`",
                    f"- Gene count after join: {summary['n_vars']}",
                    f"- Slice count: {adata.obs['slice_id'].nunique() if 'slice_id' in adata.obs else 'not available'}",
                    "",
                ]
            )
            _append_counts(lines, "Time-Point Counts", obs_summary.get("time", {}))
            _append_counts(lines, "Dataset Part Counts", obs_summary.get("dataset_part", {}))
            for field in ("cell_type_l1", "cell_type_l2", "cell_type_l3"):
                counts = obs_summary.get(field, {})
                lines.append(f"- {field}: present with {len(counts)} levels")
            lines.append("")
            if "time" in adata.obs and "cell_type_l2" in adata.obs:
                day35 = adata.obs["time"].astype(str) == "D35"
                day35_na = int((adata.obs.loc[day35, "cell_type_l2"].astype(str) == "NA").sum())
                day35_total = int(day35.sum())
                lines.extend(
                    [
                        "## Day35 Tier2 Fallback",
                        "",
                        f"- Day35 cells with `cell_type_l2 == 'NA'`: {day35_na} / {day35_total}",
                        "- `NA` indicates the configured Tier2 fallback for inputs where Tier2 is absent.",
                        "",
                    ]
                )
            optional = config["metadata"]["optional_obs_fields"]
            missing_optional = [field for field in optional if field not in adata.obs]
            lines.extend(["## Missing Optional Fields", ""])
            lines.append(", ".join(missing_optional) if missing_optional else "None")
            lines.append("")
        finally:
            if hasattr(adata, "file"):
                adata.file.close()
    else:
        lines.extend(["## AnnData", "No processed AnnData file found yet.", ""])

    for label, path in (
        ("Spatial normalization summary", norm_path),
        ("PCA summary", pca_path),
        ("Graph summary", graph_summary_path),
    ):
        lines.extend([f"## {label}", ""])
        if path.is_file():
            table = pd.read_csv(path)
            lines.append(f"- File: `{path}`")
            lines.append(f"- Rows: {len(table)}")
        else:
            lines.append("Not available yet.")
        lines.append("")

    lines.extend(["## Graph Methods and Delaunay Status", ""])
    if graph_methods:
        lines.append(f"- Methods: {', '.join(graph_methods)}")
        if "mean_degree" in graph_rows:
            means = graph_rows.groupby("graph_name", observed=True)["mean_degree"].mean()
            for graph_name in sorted(means.index):
                lines.append(f"- {graph_name} average mean degree: {means[graph_name]:.6f}")
        lines.append(f"- Delaunay rows: {len(delaunay_rows)}")
        lines.append(f"- Delaunay zero-edge slices: {len(empty_delaunay)}")
    else:
        lines.append("Graph summary not available.")
    lines.append("")

    lines.extend(["## Export Outputs", ""])
    lines.append(f"- Final h5ad: `{final_path}`")
    for label in ("D0", "D3", "D9", "D21", "D35"):
        count = by_time_counts.get(label, "missing")
        lines.append(f"- {label}: {count}")
    lines.append(f"- by_time total cells: {sum(by_time_counts.values())}")
    lines.append("")

    lines.extend(["## Disk Usage", ""])
    for label, path in (
        ("M0 output", dirs["output_dir"]),
        ("processed", dirs["processed"]),
        ("by_time", dirs["by_time"]),
        ("by_slice", dirs["by_slice"]),
        ("reports", dirs["reports"]),
    ):
        lines.append(f"- {label}: {_format_gib(_path_size_bytes(path))}")
    for path in (Path("/home/zhutao"), Path("/data"), Path("/ssd")):
        if path.exists():
            usage = shutil.disk_usage(path)
            lines.append(f"- {path}: {_format_gib(usage.free)} free")
    lines.append("")

    report_path = dirs["reports"] / "m0_report.md"
    write_markdown_report(lines, report_path)
    print(f"Wrote M0 report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
