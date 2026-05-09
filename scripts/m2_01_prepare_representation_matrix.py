#!/usr/bin/env python
"""Prepare a sample M2 niche representation matrix from M1 feature tables."""

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
from nichefate.representation import (
    feature_group_columns,
    finite_value_summary,
    pivot_scale_features,
    select_numeric_feature_columns,
    validate_aligned_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m2_niche_representation.yaml")
    parser.add_argument("--max-slices", type=int, default=2)
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


def _read_feature_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _write_matrix(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table.to_parquet(path, index=False)
        return path
    except Exception:  # noqa: BLE001
        csv_path = path.with_suffix(".csv")
        table.to_csv(csv_path, index=False)
        return csv_path


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# M2 Representation Preflight",
        "",
        f"- Selected slices: {summary['selected_slices']}",
        f"- Matrix path: {summary['matrix_path']}",
        f"- Matrix shape: {summary['matrix_rows']} rows x {summary['matrix_columns']} columns",
        f"- Anchors represented: {summary['anchors_represented']}",
        f"- Metadata columns: {summary['metadata_columns']}",
        f"- Feature columns before pivot: {summary['feature_columns_before_pivot']}",
        f"- Feature columns after scale pivot: {summary['feature_columns_after_pivot']}",
        f"- Missing values: {summary['missing_values']}",
        f"- Infinite values: {summary['infinite_values']}",
        f"- Expected scales: {', '.join(summary['expected_scales'])}",
        f"- Status: {summary['status']}",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = _paths(config)
    representation = config["representation"]
    reports_dir = paths["m2_reports_dir"]
    prototype_dir = paths["m2_prototype_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    prototype_dir.mkdir(parents=True, exist_ok=True)

    with paths["m1_global_schema"].open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    expected_columns = list(schema["feature_columns"])
    expected_scales = list(config["expected"]["scales"])
    slice_dirs = sorted(path for path in paths["m1_by_slice_dir"].iterdir() if path.is_dir())
    selected_dirs = slice_dirs[: max(args.max_slices, 0)]
    if not selected_dirs:
        raise FileNotFoundError("No M1 slice directories selected.")

    tables = []
    selected_slices = []
    for slice_dir in selected_dirs:
        feature_path = _feature_path(slice_dir)
        if feature_path is None:
            raise FileNotFoundError(f"Missing feature table in {slice_dir}")
        table = _read_feature_table(feature_path)
        validate_aligned_schema(table, expected_columns)
        tables.append(table)
        selected_slices.append(slice_dir.name)

    features = pd.concat(tables, ignore_index=True)
    feature_columns = select_numeric_feature_columns(features, config["feature_groups"])
    metadata_columns = list(representation["metadata_columns"])
    matrix = pivot_scale_features(
        features,
        feature_columns=feature_columns,
        expected_scales=expected_scales,
        metadata_columns=metadata_columns,
        anchor_keys=list(representation["anchor_keys"]),
        scale_column=str(representation["scale_column"]),
        separator=str(representation["scale_prefix_separator"]),
    )
    matrix_path = _write_matrix(
        matrix,
        prototype_dir / "m2_representation_sample.parquet",
    )
    value_summary = finite_value_summary(matrix)
    grouped = feature_group_columns(features.columns, config["feature_groups"])
    feature_group_path = reports_dir / "m2_feature_groups.json"
    with feature_group_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "configured_feature_groups": config["feature_groups"],
                "resolved_feature_group_columns": grouped,
                "selected_numeric_feature_columns": feature_columns,
                "feature_columns_before_pivot": len(feature_columns),
                "feature_columns_after_pivot": len(feature_columns) * len(expected_scales),
                "metadata_columns": metadata_columns,
                "expected_scales": expected_scales,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    summary = {
        "selected_slices": ", ".join(selected_slices),
        "matrix_path": str(matrix_path),
        "matrix_rows": int(matrix.shape[0]),
        "matrix_columns": int(matrix.shape[1]),
        "anchors_represented": int(matrix.shape[0]),
        "metadata_columns": len(metadata_columns),
        "feature_columns_before_pivot": len(feature_columns),
        "feature_columns_after_pivot": len(feature_columns) * len(expected_scales),
        "missing_values": value_summary["missing_values"],
        "infinite_values": value_summary["infinite_values"],
        "expected_scales": expected_scales,
        "status": "PASS",
    }
    report_path = reports_dir / "m2_representation_preflight.md"
    _write_report(report_path, summary)

    print(f"Wrote sample matrix: {matrix_path}")
    print(f"Wrote preflight report: {report_path}")
    print(f"Wrote feature groups: {feature_group_path}")
    print(f"MATRIX_SHAPE {matrix.shape[0]} {matrix.shape[1]}")
    print(f"ANCHORS_REPRESENTED {matrix.shape[0]}")
    print(f"FEATURE_COLUMNS_AFTER_PIVOT {summary['feature_columns_after_pivot']}")
    print(f"MISSING_VALUES {summary['missing_values']}")
    print(f"INFINITE_VALUES {summary['infinite_values']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
