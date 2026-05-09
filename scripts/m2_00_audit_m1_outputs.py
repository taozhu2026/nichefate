#!/usr/bin/env python
"""Audit completed M1 by-slice outputs before M2 representation work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m2_niche_representation.yaml")
    return parser.parse_args()


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        key: Path(value)
        for key, value in config["paths"].items()
        if isinstance(value, str) and value.startswith("/")
    }


def _feature_path(slice_dir: Path) -> Path | None:
    paths = sorted(slice_dir.glob("niche_features_*.parquet"))
    if paths:
        return paths[0]
    paths = sorted(slice_dir.glob("niche_features_*.csv"))
    return paths[0] if paths else None


def _read_columns(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).schema_arrow.names
    return list(pd.read_csv(path, nrows=0).columns)


def _row_count(path: Path) -> int:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    return int(sum(1 for _ in path.open("r", encoding="utf-8")) - 1)


def _read_identity(path: Path) -> pd.DataFrame:
    columns = ["slice_id", "anchor_index", "scale"]
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, low_memory=False)


def _audit_slice(
    slice_dir: Path,
    expected_columns: list[str],
    expected_scales: list[str],
) -> dict[str, Any]:
    feature_path = _feature_path(slice_dir)
    neighbor_paths = sorted(slice_dir.glob("neighbor_index_*.npz"))
    report_paths = sorted(slice_dir.glob("m1_report_*.md"))
    row: dict[str, Any] = {
        "slice_id": slice_dir.name,
        "slice_dir": str(slice_dir),
        "feature_path": str(feature_path) if feature_path else "",
        "neighbor_path": str(neighbor_paths[0]) if neighbor_paths else "",
        "report_path": str(report_paths[0]) if report_paths else "",
        "feature_exists": feature_path is not None and feature_path.exists(),
        "neighbor_exists": bool(neighbor_paths),
        "report_exists": bool(report_paths),
        "feature_rows": 0,
        "feature_columns": 0,
        "anchors": 0,
        "schema_aligned": False,
        "duplicate_anchor_scale_rows": 0,
        "anchors_with_expected_scales": 0,
        "anchors_missing_expected_scales": 0,
        "unexpected_scales": "",
        "ok": False,
        "error": "",
    }
    if feature_path is None:
        row["error"] = "missing feature table"
        return row
    if not neighbor_paths:
        row["error"] = "missing neighbor index"
        return row
    if not report_paths:
        row["error"] = "missing slice report"
        return row

    columns = _read_columns(feature_path)
    row["feature_columns"] = len(columns)
    row["schema_aligned"] = columns == expected_columns
    row["feature_rows"] = _row_count(feature_path)
    identity = _read_identity(feature_path)
    row["duplicate_anchor_scale_rows"] = int(
        identity.duplicated(["slice_id", "anchor_index", "scale"]).sum()
    )
    scale_counts = identity.groupby(["slice_id", "anchor_index"], observed=True)[
        "scale"
    ].nunique()
    row["anchors"] = int(len(scale_counts))
    row["anchors_with_expected_scales"] = int((scale_counts == len(expected_scales)).sum())
    row["anchors_missing_expected_scales"] = int(
        (scale_counts != len(expected_scales)).sum()
    )
    unexpected = sorted(set(identity["scale"].dropna().astype(str)) - set(expected_scales))
    row["unexpected_scales"] = ",".join(unexpected)

    checks = [
        row["feature_exists"],
        row["neighbor_exists"],
        row["report_exists"],
        row["schema_aligned"],
        row["duplicate_anchor_scale_rows"] == 0,
        row["anchors_missing_expected_scales"] == 0,
        not unexpected,
        row["feature_rows"] == row["anchors"] * len(expected_scales),
    ]
    row["ok"] = bool(all(checks))
    if not row["ok"]:
        row["error"] = "one or more slice audit checks failed"
    return row


def _write_report(path: Path, rows: list[dict[str, Any]], totals: dict[str, Any]) -> None:
    failed = [row for row in rows if not row["ok"]]
    lines = [
        "# M2 M1 Output Audit",
        "",
        f"- Slice directories found: {totals['slice_dirs_found']}",
        f"- Expected slice directories: {totals['expected_slices']}",
        f"- Slices passing audit: {totals['slices_ok']}",
        f"- Slices failing audit: {len(failed)}",
        f"- Total anchors: {totals['total_anchors']}",
        f"- Total feature rows: {totals['total_feature_rows']}",
        f"- Feature columns: {totals['feature_columns']}",
        f"- Expected scales: {', '.join(totals['expected_scales'])}",
        f"- Global reports present: {totals['global_reports_ok']}",
        f"- Overall status: {'PASS' if totals['ok'] else 'FAIL'}",
    ]
    if failed:
        lines.extend(["", "## Failed Slices", ""])
        lines.extend([f"- `{row['slice_id']}`: {row['error']}" for row in failed[:50]])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = _paths(config)
    expected = config["expected"]
    reports_dir = paths["m2_reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    with paths["m1_global_schema"].open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    expected_columns = list(schema["feature_columns"])
    expected_scales = list(expected["scales"])
    slice_dirs = sorted(path for path in paths["m1_by_slice_dir"].iterdir() if path.is_dir())
    rows = [_audit_slice(path, expected_columns, expected_scales) for path in slice_dirs]

    csv_path = reports_dir / "m2_m1_output_audit.csv"
    md_path = reports_dir / "m2_m1_output_audit.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    global_reports = [
        paths["m1_global_schema"],
        paths["m1_summary_csv"],
        paths["m1_summary_md"],
        paths["m1_celltype_vocabulary"],
    ]
    totals = {
        "slice_dirs_found": len(slice_dirs),
        "expected_slices": int(expected["n_slices"]),
        "slices_ok": sum(1 for row in rows if row["ok"]),
        "total_anchors": sum(int(row["anchors"]) for row in rows),
        "total_feature_rows": sum(int(row["feature_rows"]) for row in rows),
        "feature_columns": len(expected_columns),
        "expected_scales": expected_scales,
        "global_reports_ok": all(path.exists() for path in global_reports),
    }
    totals["ok"] = bool(
        totals["slice_dirs_found"] == totals["expected_slices"]
        and totals["slices_ok"] == totals["expected_slices"]
        and totals["total_anchors"] == int(expected["total_anchors"])
        and totals["total_feature_rows"] == int(expected["total_feature_rows"])
        and totals["feature_columns"] == int(expected["feature_columns"])
        and totals["global_reports_ok"]
    )
    _write_report(md_path, rows, totals)

    print(f"Wrote audit CSV: {csv_path}")
    print(f"Wrote audit report: {md_path}")
    print(f"AUDIT_STATUS {'PASS' if totals['ok'] else 'FAIL'}")
    print(f"SLICES_OK {totals['slices_ok']}")
    print(f"TOTAL_ANCHORS {totals['total_anchors']}")
    print(f"TOTAL_FEATURE_ROWS {totals['total_feature_rows']}")
    return 0 if totals["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
