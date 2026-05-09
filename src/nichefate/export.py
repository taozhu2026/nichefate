"""Export helpers for M0 artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from nichefate.io import write_h5ad_safely


def export_by_time(adata: Any, output_dir: str | Path) -> list[Path]:
    """Write one h5ad file per M0 time point."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if "time_day" not in adata.obs:
        raise ValueError("Missing obs['time_day']; cannot export by time.")
    for day in sorted(pd.Series(adata.obs["time_day"]).dropna().astype(int).unique()):
        subset = adata[adata.obs["time_day"].astype(int) == day].copy()
        target = output_path / f"D{day}.h5ad"
        write_h5ad_safely(subset, target)
        written.append(target)
    return written


def export_by_slice(adata: Any, output_dir: str | Path) -> list[Path]:
    """Write one h5ad file per slice."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if "slice_id" not in adata.obs:
        raise ValueError("Missing obs['slice_id']; cannot export by slice.")
    written: list[Path] = []
    for slice_id in sorted(adata.obs["slice_id"].astype(str).unique()):
        subset = adata[adata.obs["slice_id"].astype(str) == slice_id].copy()
        safe_slice = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in slice_id)
        target = output_path / f"{safe_slice}.m0.h5ad"
        write_h5ad_safely(subset, target)
        written.append(target)
    return written


def write_sample_tables(adata: Any, reports_dir: str | Path) -> list[Path]:
    """Write small obs summary tables for M0 reports."""

    output_path = Path(reports_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for field in ("time_day", "mouse_id", "slice_id", "cell_type_l1"):
        if field not in adata.obs:
            continue
        table = adata.obs[field].value_counts(dropna=False).rename_axis(field)
        target = output_path / f"{field}_counts.csv"
        table.reset_index(name="n_cells").to_csv(target, index=False)
        written.append(target)
    return written


def export_m0_objects(adata: object, output_dir: str | Path) -> None:
    """Export M0-ready objects to a target directory."""

    write_h5ad_safely(adata, Path(output_dir) / "m0_all_colitis_merfish.final.h5ad")
