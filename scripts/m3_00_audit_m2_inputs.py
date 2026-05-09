#!/usr/bin/env python
"""Audit M2 by-slice inputs and infer adjacent M3 time pairs."""

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
from nichefate.representation import finite_value_summary
from nichefate.transition import infer_adjacent_time_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_transition_kernel.yaml")
    return parser.parse_args()


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        key: Path(value)
        for key, value in config["paths"].items()
        if isinstance(value, str) and value.startswith("/")
    }


def _parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).schema_arrow.names


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    time_pairs: list[dict[str, Any]],
    totals: dict[str, Any],
) -> None:
    lines = [
        "# M3 M2 Input Audit",
        "",
        f"- Slice outputs found: {totals['slice_outputs']}",
        f"- Expected slice outputs: {totals['expected_slices']}",
        f"- Total rows: {totals['total_rows']}",
        f"- Expected total rows: {totals['expected_rows']}",
        f"- Output columns: {totals['output_columns']}",
        f"- Numeric feature columns: {totals['numeric_feature_columns']}",
        f"- Missing values: {totals['missing_values']}",
        f"- Infinite values: {totals['infinite_values']}",
        f"- Schema mismatches: {totals['schema_mismatches']}",
        f"- Total disk usage: {totals['disk_bytes']}",
        f"- Adjacent time pairs: {len(time_pairs)}",
        f"- Expected sampled edge count upper bound: {totals['expected_sampled_edges']}",
        f"- Overall status: {'PASS' if totals['ok'] else 'FAIL'}",
        "",
        "## Time Pairs",
        "",
    ]
    for pair in time_pairs:
        lines.append(
            "- "
            f"{pair['source_time']} -> {pair['target_time']} "
            f"(days {pair['source_day']} -> {pair['target_day']}, "
            f"delta {pair['time_delta']}, rows "
            f"{pair['source_row_count']} -> {pair['target_row_count']})"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = _paths(config)
    reports_dir = paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    with paths["m2_schema"].open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    expected_columns = list(schema["output_columns"])
    numeric_columns = list(schema["numeric_feature_columns"])
    metadata_columns = list(config["input"]["metadata_columns"])
    files = sorted(paths["m2_by_slice_dir"].glob("*/m2_representation_*.parquet"))
    rows: list[dict[str, Any]] = []
    metadata_frames = []
    for path in files:
        columns = _parquet_columns(path)
        row_count = _parquet_rows(path)
        metadata = pd.read_parquet(path, columns=metadata_columns)
        metadata_frames.append(
            metadata.groupby(
                ["slice_id", "slice_file", "time", "time_day", "mouse_id"],
                dropna=False,
                observed=True,
            )
            .size()
            .reset_index(name="rows")
        )
        numeric = pd.read_parquet(path, columns=numeric_columns)
        finite = finite_value_summary(numeric)
        rows.append(
            {
                "slice_id": metadata["slice_id"].iloc[0],
                "path": str(path),
                "rows": row_count,
                "columns": len(columns),
                "numeric_feature_columns": len(numeric_columns),
                "missing_values": finite["missing_values"],
                "infinite_values": finite["infinite_values"],
                "schema_consistent": columns == expected_columns,
                "metadata_complete": all(column in columns for column in metadata_columns),
                "output_bytes": path.stat().st_size,
                "time": metadata["time"].iloc[0],
                "time_day": metadata["time_day"].iloc[0],
                "mouse_id": metadata["mouse_id"].iloc[0],
            }
        )
    audit_csv = reports_dir / "m3_m2_input_audit.csv"
    pd.DataFrame(rows).to_csv(audit_csv, index=False)

    metadata_summary = pd.concat(metadata_frames, ignore_index=True)
    time_pairs = infer_adjacent_time_pairs(
        metadata_summary,
        config["time"]["time_column"],
        config["time"]["time_day_column"],
    )
    pair_count = len(time_pairs)
    edge_cfg = config["candidate_edges"]
    expected_sampled_edges = (
        pair_count
        * int(edge_cfg["max_source_niches_per_pair"])
        * int(edge_cfg["k_candidates"])
    )
    for pair in time_pairs:
        pair["expected_sampled_edge_upper_bound"] = (
            int(edge_cfg["max_source_niches_per_pair"])
            * int(edge_cfg["k_candidates"])
        )
    time_pair_path = reports_dir / "m3_time_pairs.json"
    time_pair_path.write_text(json.dumps(time_pairs, indent=2) + "\n", encoding="utf-8")

    frame = pd.DataFrame(rows)
    totals = {
        "slice_outputs": len(files),
        "expected_slices": 58,
        "total_rows": int(frame["rows"].sum()) if not frame.empty else 0,
        "expected_rows": 1439542,
        "output_columns": int(schema["output_column_count"]),
        "numeric_feature_columns": int(schema["numeric_feature_column_count"]),
        "missing_values": int(frame["missing_values"].sum()) if not frame.empty else 0,
        "infinite_values": int(frame["infinite_values"].sum()) if not frame.empty else 0,
        "schema_mismatches": int((~frame["schema_consistent"]).sum()) if not frame.empty else 0,
        "disk_bytes": int(frame["output_bytes"].sum()) if not frame.empty else 0,
        "expected_sampled_edges": expected_sampled_edges,
    }
    totals["ok"] = bool(
        totals["slice_outputs"] == totals["expected_slices"]
        and totals["total_rows"] == totals["expected_rows"]
        and totals["output_columns"] == 775
        and totals["numeric_feature_columns"] == 765
        and totals["missing_values"] == 0
        and totals["infinite_values"] == 0
        and totals["schema_mismatches"] == 0
    )
    report_path = reports_dir / "m3_m2_input_audit.md"
    _write_report(report_path, rows, time_pairs, totals)
    print(f"Wrote M2 input audit CSV: {audit_csv}")
    print(f"Wrote M2 input audit report: {report_path}")
    print(f"Wrote M3 time pairs: {time_pair_path}")
    print(f"AUDIT_STATUS {'PASS' if totals['ok'] else 'FAIL'}")
    print(f"TIME_PAIRS {pair_count}")
    print(f"EXPECTED_SAMPLED_EDGES {expected_sampled_edges}")
    print(f"TOTAL_ROWS {totals['total_rows']}")
    return 0 if totals["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
