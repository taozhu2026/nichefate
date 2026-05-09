#!/usr/bin/env python
"""Build M1 niche features for one full by-slice M0 object."""

from __future__ import annotations

import argparse
import resource
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anndata as ad
import numpy as np
import pandas as pd

from nichefate.io import load_config
from nichefate.niche import (
    build_basic_niche_feature_table,
    compute_neighbor_index,
    load_global_feature_schema,
    write_neighbor_index_npz,
)
from nichefate.niche_qc import (
    composition_sum_qc,
    estimate_full_m1_storage,
    summarize_feature_integrity,
    validate_neighbor_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m1_niche_construction.yaml")
    parser.add_argument("--slice-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--global-schema", type=Path, default=None)
    parser.add_argument("--report-prefix", default="pilot_report")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _paths(config: dict) -> dict[str, Path]:
    return {
        key: Path(value)
        for key, value in config["paths"].items()
        if isinstance(value, str)
    }


def _safe_token(value: object) -> str:
    text = str(value)
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_")
    return token or "slice"


def _format_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    view = frame.head(max_rows).copy()
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for column in view.columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _ensure_can_write(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists; pass --force to replace: {joined}")


def _replace_from_tmp(tmp_path: Path, final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.replace(final_path)
    return final_path


def _cleanup_tmp(path: Path) -> None:
    if path.exists():
        path.unlink()


def _write_feature_table_safely(table: pd.DataFrame, base_path: Path, force: bool) -> Path:
    parquet_path = base_path.with_suffix(".parquet")
    csv_path = base_path.with_suffix(".csv")
    _ensure_can_write([parquet_path, csv_path], force)

    parquet_tmp = parquet_path.with_name(parquet_path.name + ".tmp")
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    _cleanup_tmp(parquet_tmp)
    _cleanup_tmp(csv_tmp)
    try:
        try:
            table.to_parquet(parquet_tmp, index=False)
            return _replace_from_tmp(parquet_tmp, parquet_path)
        except (ImportError, ModuleNotFoundError, ValueError):
            table.to_csv(csv_tmp, index=False)
            return _replace_from_tmp(csv_tmp, csv_path)
    finally:
        _cleanup_tmp(parquet_tmp)
        _cleanup_tmp(csv_tmp)


def _write_neighbor_index_safely(entries: list[dict[str, object]], path: Path, force: bool) -> Path:
    _ensure_can_write([path], force)
    tmp_path = path.with_suffix(".tmp.npz")
    _cleanup_tmp(tmp_path)
    try:
        written = write_neighbor_index_npz(entries, tmp_path)
        return _replace_from_tmp(written, path)
    finally:
        _cleanup_tmp(tmp_path)


def _write_report_safely(lines: list[str], path: Path, force: bool) -> Path:
    _ensure_can_write([path], force)
    tmp_path = path.with_name(path.name + ".tmp")
    _cleanup_tmp(tmp_path)
    try:
        tmp_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return _replace_from_tmp(tmp_path, path)
    finally:
        _cleanup_tmp(tmp_path)


def _load_prototype_context(paths: dict[str, Path]) -> tuple[int | None, float | None]:
    prototype_path = paths["m1_output_dir"] / "prototype" / "niche_features_prototype.csv"
    report_path = paths["reports_dir"] / "m1_prototype_report.md"
    prototype_rows = None
    prototype_wall = None
    if prototype_path.exists():
        prototype_rows = sum(1 for _ in prototype_path.open("r", encoding="utf-8")) - 1
    if report_path.exists():
        match = re.search(
            r"Wall seconds:\s*([0-9.]+)",
            report_path.read_text(encoding="utf-8"),
        )
        prototype_wall = float(match.group(1)) if match else None
    return prototype_rows, prototype_wall


def _build_outputs(
    data: ad.AnnData,
    slice_path: Path,
    config: dict,
    global_schema: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    slice_id = str(data.obs["slice_id"].iloc[0]) if "slice_id" in data.obs else slice_path.stem
    anchors = np.arange(int(data.n_obs), dtype=np.int64)
    feature_tables = []
    neighbor_entries = []

    for scale in config["niche"]["scales"]:
        neighbor_index = compute_neighbor_index(
            data,
            scale,
            anchor_indices=anchors,
            include_anchor=bool(config["niche"]["include_anchor"]),
        )
        feature_tables.append(
            build_basic_niche_feature_table(
                data,
                neighbor_index,
                scale=scale,
                slice_file=slice_path.name,
                cell_type_keys=list(config["input"]["cell_type_keys"]),
                embedding_key=config["input"]["embedding_key"],
                spatial_key=config["input"]["spatial_key"],
                topology_graph_key=config["input"]["graph_key_topology"],
                global_schema=global_schema,
            )
        )
        neighbor_entries.append(
            {
                "slice_id": slice_id,
                "slice_file": slice_path.name,
                "scale": scale,
                "neighbor_index": neighbor_index,
            }
        )
    return pd.concat(feature_tables, ignore_index=True), neighbor_entries


def _pilot_report(
    *,
    slice_path: Path,
    slice_id: str,
    n_cells: int,
    features: pd.DataFrame,
    feature_path: Path,
    neighbor_path: Path,
    report_path: Path,
    elapsed: float,
    max_rss_kb: int,
    integrity: pd.DataFrame,
    composition_qc: pd.DataFrame,
    neighbor_validation: pd.DataFrame,
    prototype_rows: int | None,
    prototype_wall: float | None,
) -> list[str]:
    rows_expected = n_cells * int(features["scale"].nunique())
    anchor_scale_duplicates = int(features.duplicated(["slice_id", "anchor_index", "scale"]).sum())
    exact_scale_rows = int(
        (
            features.groupby(["slice_id", "anchor_index"], observed=True)["scale"].nunique()
            == features["scale"].nunique()
        ).sum()
    )
    n_neighbors = (
        features.groupby("scale", observed=True)["n_neighbors"]
        .agg(["count", "mean", "min", "median", "max"])
        .reset_index()
    )
    output_sizes = pd.DataFrame(
        [
            {
                "path": str(feature_path),
                "bytes": feature_path.stat().st_size,
                "human": _format_bytes(feature_path.stat().st_size),
            },
            {
                "path": str(neighbor_path),
                "bytes": neighbor_path.stat().st_size,
                "human": _format_bytes(neighbor_path.stat().st_size),
            },
            {"path": str(report_path), "bytes": 0, "human": "pending"},
        ]
    )
    scale_names = list(features["scale"].drop_duplicates())
    storage_estimate = estimate_full_m1_storage(
        full_anchors=1_439_542,
        scales=scale_names,
        prototype_rows=len(features),
        prototype_feature_bytes=feature_path.stat().st_size,
        avg_neighbors_by_scale=dict(zip(n_neighbors["scale"], n_neighbors["mean"], strict=False)),
        n_slices=58,
        neighbor_compression_ratio=0.14,
    )
    prototype_scale = None
    if prototype_rows and prototype_wall and prototype_rows > 0:
        prototype_anchors = prototype_rows / len(scale_names)
        prototype_scale = elapsed / max(n_cells, 1), prototype_wall / prototype_anchors

    infinite_total = int(integrity["infinite_values"].sum()) if not integrity.empty else 0
    missing_total = int(integrity["missing_values"].sum()) if not integrity.empty else 0
    composition_ok = bool((composition_qc["rows_not_close_to_one"] == 0).all())
    neighbor_ok = bool(neighbor_validation["ok"].all()) if not neighbor_validation.empty else False
    rows_ok = len(features) == rows_expected
    exact_scales_ok = exact_scale_rows == n_cells
    duplicates_ok = anchor_scale_duplicates == 0
    inf_ok = infinite_total == 0

    lines = [
        "# M1 Full-Slice Pilot Report",
        "",
        "This report covers one full-slice pilot only. Full M1 across all slices was not run.",
        "",
        "## Slice",
        "",
        f"- Slice file: `{slice_path}`",
        f"- Slice ID: `{slice_id}`",
        f"- Cells / anchors: {n_cells}",
        f"- Scales: {', '.join(scale_names)}",
        "",
        "## Outputs",
        "",
        f"- Feature table: `{feature_path}`",
        f"- Neighbor index: `{neighbor_path}`",
        f"- Report: `{report_path}`",
        "",
        "### Output Sizes",
        "",
        *_markdown_table(output_sizes),
        "",
        "## Validation",
        "",
        f"- Feature table shape: {features.shape[0]} rows x {features.shape[1]} columns",
        f"- Expected rows (`n_cells x scales`): {rows_expected}",
        f"- Row count valid: {rows_ok}",
        f"- Neighbor index entries: {len(neighbor_validation)}",
        f"- Every anchor has exactly {len(scale_names)} scale rows: {exact_scales_ok}",
        f"- Duplicated anchor/scale rows: {anchor_scale_duplicates}",
        f"- Infinite values: {infinite_total}",
        f"- Missing values: {missing_total}",
        f"- Composition row sums valid: {composition_ok}",
        f"- Neighbor index QC passed: {neighbor_ok}",
        "",
        "## n_neighbors By Scale",
        "",
        *_markdown_table(n_neighbors),
        "",
        "## Missing And Infinite Values By Feature Group",
        "",
        *_markdown_table(
            integrity[
                [
                    "feature_group",
                    "n_columns",
                    "missing_values",
                    "missing_fraction",
                    "infinite_values",
                    "infinite_fraction",
                ]
            ],
            max_rows=30,
        ),
        "",
        "## Composition Row-Sum QC",
        "",
        *_markdown_table(composition_qc, max_rows=10),
        "",
        "## Neighbor Index QC",
        "",
        *_markdown_table(
            neighbor_validation[
                [
                    "entry",
                    "scale",
                    "n_anchors",
                    "indptr_length",
                    "neighbor_indices_length",
                    "avg_neighbors_npz",
                    "avg_neighbors_feature",
                    "anchor_inclusion_ok",
                    "ok",
                    "errors",
                ]
            ],
            max_rows=10,
        ),
        "",
        "## Runtime",
        "",
        f"- Wall seconds: {elapsed:.3f}",
        f"- Max RSS KB: {max_rss_kb}",
        f"- Max RSS: {_format_bytes(max_rss_kb * 1024)}",
        "",
        "## Prototype Estimate Comparison",
        "",
    ]
    if prototype_scale:
        pilot_seconds_per_anchor, prototype_seconds_per_anchor = prototype_scale
        lines.extend(
            [
                f"- Pilot seconds per anchor: {pilot_seconds_per_anchor:.6g}",
                f"- Prototype seconds per anchor: {prototype_seconds_per_anchor:.6g}",
                f"- Pilot/prototype per-anchor runtime ratio: {pilot_seconds_per_anchor / prototype_seconds_per_anchor:.3f}",
            ]
        )
    else:
        lines.append("- Prototype runtime context was unavailable.")
    lines.extend(
        [
            f"- Pilot-based full feature rows: {storage_estimate['full_feature_rows']}",
            f"- Pilot-based feature parquet estimate: {_format_bytes(storage_estimate['feature_csv_bytes'])}",
            f"- Pilot-based neighbor NPZ estimate: {_format_bytes(storage_estimate['neighbor_npz_bytes'])}",
            f"- Pilot-based total parquet/NPZ-like estimate: {_format_bytes(storage_estimate['total_csv_plus_npz_bytes'])}",
            "",
            "## Strategy Verdict",
            "",
            (
                "- The full-slice pilot validates the M1 execution strategy for this slice."
                if all([rows_ok, exact_scales_ok, duplicates_ok, inf_ok, composition_ok, neighbor_ok])
                else "- The full-slice pilot did not fully validate the strategy; inspect failed checks above."
            ),
        ]
    )
    return lines


def main() -> int:
    args = parse_args()
    start = time.monotonic()
    config = load_config(args.config)
    paths = _paths(config)
    global_schema = load_global_feature_schema(args.global_schema)
    slice_path = args.slice_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = ad.read_h5ad(slice_path)
    try:
        slice_id = str(data.obs["slice_id"].iloc[0]) if "slice_id" in data.obs else slice_path.stem
        slice_token = _safe_token(slice_id)
        n_cells = int(data.n_obs)
        feature_base = output_dir / f"niche_features_{slice_token}"
        neighbor_path = output_dir / f"neighbor_index_{slice_token}.npz"
        report_path = output_dir / f"{args.report_prefix}_{slice_token}.md"
        _ensure_can_write(
            [
                feature_base.with_suffix(".parquet"),
                feature_base.with_suffix(".csv"),
                neighbor_path,
                report_path,
            ],
            args.force,
        )

        features, neighbor_entries = _build_outputs(data, slice_path, config, global_schema)
    finally:
        if hasattr(data, "file"):
            data.file.close()

    feature_path = _write_feature_table_safely(features, feature_base, args.force)
    neighbor_written = _write_neighbor_index_safely(neighbor_entries, neighbor_path, args.force)

    integrity = summarize_feature_integrity(features)
    composition_qc = composition_sum_qc(features)
    neighbor_validation = validate_neighbor_npz(
        neighbor_written,
        feature_table=features,
        slice_n_obs={slice_id: n_cells, slice_path.name: n_cells},
        expected_entries=len(config["niche"]["scales"]),
        include_anchor=bool(config["niche"]["include_anchor"]),
    )
    elapsed = time.monotonic() - start
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    prototype_rows, prototype_wall = _load_prototype_context(paths)
    report_lines = _pilot_report(
        slice_path=slice_path,
        slice_id=slice_id,
        n_cells=n_cells,
        features=features,
        feature_path=feature_path,
        neighbor_path=neighbor_written,
        report_path=report_path,
        elapsed=elapsed,
        max_rss_kb=max_rss_kb,
        integrity=integrity,
        composition_qc=composition_qc,
        neighbor_validation=neighbor_validation,
        prototype_rows=prototype_rows,
        prototype_wall=prototype_wall,
    )
    _write_report_safely(report_lines, report_path, args.force)

    print(f"Wrote feature table: {feature_path}")
    print(f"Wrote neighbor index: {neighbor_written}")
    print(f"Wrote report: {report_path}")
    print(f"SLICE_ID {slice_id}")
    print(f"N_CELLS {n_cells}")
    print(f"FEATURE_SHAPE {features.shape[0]} {features.shape[1]}")
    print(f"NEIGHBOR_ENTRIES {len(neighbor_validation)}")
    print(f"WALL_SECONDS {elapsed:.3f}")
    print(f"MAX_RSS_KB {max_rss_kb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
