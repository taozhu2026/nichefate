#!/usr/bin/env python
"""Build full M2 representation matrices sequentially by slice."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config
from nichefate.representation import (
    build_m2_representation_table,
    feature_columns_from_schema,
    finite_value_summary,
    m2_output_columns,
    scale_prefixed_feature_columns,
    select_numeric_feature_columns,
    validate_aligned_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m2_niche_representation.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--slice-id", default=None)
    parser.add_argument("--slice-file", type=Path, default=None)
    return parser.parse_args()


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        key: Path(value)
        for key, value in config["paths"].items()
        if isinstance(value, str) and value.startswith("/")
    }


def _format_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


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


def _read_columns(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).schema_arrow.names
    return list(pd.read_csv(path, nrows=0).columns)


def _row_count(path: Path) -> int:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    with path.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _slice_id_from_file(path: Path) -> str:
    name = path.name
    if name.endswith(".m0.h5ad"):
        return name[: -len(".m0.h5ad")]
    if name.startswith("niche_features_"):
        name = name[len("niche_features_") :]
        return Path(name).stem
    return path.stem


def discover_slices(args: argparse.Namespace, paths: dict[str, Path]) -> list[dict[str, Any]]:
    m1_root = paths["m1_by_slice_dir"]
    records = []
    for slice_dir in sorted(path for path in m1_root.iterdir() if path.is_dir()):
        records.append(
            {
                "slice_id": slice_dir.name,
                "m1_slice_dir": slice_dir,
                "feature_path": _feature_path(slice_dir),
            }
        )
    if args.slice_id is not None:
        records = [row for row in records if row["slice_id"] == args.slice_id]
    if args.slice_file is not None:
        selected_id = _slice_id_from_file(args.slice_file)
        records = [row for row in records if row["slice_id"] == selected_id]
        if not records and args.slice_file.is_file():
            records = [
                {
                    "slice_id": selected_id,
                    "m1_slice_dir": args.slice_file.parent,
                    "feature_path": args.slice_file,
                }
            ]
    if args.max_slices is not None:
        records = records[: max(args.max_slices, 0)]
    return records


def build_schema_info(config: dict[str, Any], m1_schema: dict[str, Any]) -> dict[str, Any]:
    representation = config["representation"]
    expected_scales = list(config["expected"]["scales"])
    metadata_columns = list(representation["metadata_columns"])
    source_features = feature_columns_from_schema(
        list(m1_schema["feature_columns"]),
        config["feature_groups"],
    )
    numeric_features = scale_prefixed_feature_columns(
        source_features,
        expected_scales,
        str(representation["scale_prefix_separator"]),
    )
    output_columns = m2_output_columns(
        metadata_columns,
        source_features,
        expected_scales,
        str(representation["scale_prefix_separator"]),
    )
    return {
        "metadata_columns": metadata_columns,
        "source_feature_columns": source_features,
        "numeric_feature_columns": numeric_features,
        "output_columns": output_columns,
        "expected_scales": expected_scales,
        "anchor_keys": list(representation["anchor_keys"]),
        "scale_column": str(representation["scale_column"]),
        "separator": str(representation["scale_prefix_separator"]),
    }


def validate_existing_output(
    output_path: Path,
    expected_columns: list[str],
    expected_rows: int,
    numeric_columns: list[str],
) -> tuple[bool, dict[str, Any], str]:
    if not output_path.exists():
        return False, {}, "missing M2 representation parquet"
    try:
        columns = _read_columns(output_path)
        if columns != expected_columns:
            return False, {"output_columns": len(columns)}, "M2 output schema mismatch"
        rows = _row_count(output_path)
        if rows != expected_rows:
            return False, {"output_rows": rows}, f"M2 output rows {rows} != {expected_rows}"
        numeric = pd.read_parquet(output_path, columns=numeric_columns)
        finite = finite_value_summary(numeric)
        if finite["missing_values"] or finite["infinite_values"]:
            return False, finite, "M2 output has missing or infinite numeric values"
        return (
            True,
            {
                "output_rows": rows,
                "output_columns": len(columns),
                "output_bytes": output_path.stat().st_size,
                "missing_values": finite["missing_values"],
                "infinite_values": finite["infinite_values"],
                "schema_consistent": True,
            },
            "",
        )
    except Exception as exc:  # noqa: BLE001
        return False, {}, str(exc)


def _input_summary(
    record: dict[str, Any],
    expected_m1_columns: list[str],
    expected_scales: list[str],
) -> tuple[bool, dict[str, Any], str]:
    feature_path = record.get("feature_path")
    if feature_path is None or not Path(feature_path).exists():
        return False, {}, "missing M1 feature table"
    feature_path = Path(feature_path)
    columns = _read_columns(feature_path)
    feature_rows = _row_count(feature_path)
    if columns != expected_m1_columns:
        return False, {"input_feature_rows": feature_rows}, "M1 schema mismatch"
    if feature_rows % len(expected_scales) != 0:
        return False, {"input_feature_rows": feature_rows}, "M1 rows not divisible by scales"
    anchors = feature_rows // len(expected_scales)
    return (
        True,
        {
            "input_feature_rows": feature_rows,
            "anchors": anchors,
            "expected_output_rows": anchors,
            "input_feature_columns": len(columns),
            "input_feature_path": str(feature_path),
        },
        "",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({column for row in rows for column in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_failed(path: Path, failed: list[dict[str, Any]]) -> None:
    lines = [f"{row['slice_id']}\t{row['error']}" for row in failed]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_slice_report(path: Path, row: dict[str, Any]) -> None:
    lines = [
        "# M2 Slice Report",
        "",
        f"- Slice ID: {row['slice_id']}",
        f"- Status: {row['status']}",
        f"- Output path: {row.get('output_path', '')}",
        f"- Output rows: {row.get('output_rows', 0)}",
        f"- Output columns: {row.get('output_columns', 0)}",
        f"- Numeric feature columns: {row.get('numeric_feature_columns', 0)}",
        f"- Metadata columns: {row.get('metadata_columns', 0)}",
        f"- Missing values: {row.get('missing_values', 0)}",
        f"- Infinite values: {row.get('infinite_values', 0)}",
        f"- Output bytes: {row.get('output_bytes', 0)}",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_schema(path: Path, config: dict[str, Any], schema_info: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": config["representation"]["version"],
        "row_granularity": config["representation"]["row_granularity"],
        "metadata_columns": schema_info["metadata_columns"],
        "source_feature_columns": schema_info["source_feature_columns"],
        "numeric_feature_columns": schema_info["numeric_feature_columns"],
        "output_columns": schema_info["output_columns"],
        "expected_scales": schema_info["expected_scales"],
        "metadata_column_count": len(schema_info["metadata_columns"]),
        "source_feature_column_count": len(schema_info["source_feature_columns"]),
        "numeric_feature_column_count": len(schema_info["numeric_feature_columns"]),
        "output_column_count": len(schema_info["output_columns"]),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_summary_md(
    path: Path,
    rows: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    schema_info: dict[str, Any],
    dry_run: bool,
    elapsed: float,
    expected_total_rows: int,
) -> None:
    completed = [row for row in rows if row.get("status") in {"completed", "skipped_valid"}]
    planned = [row for row in rows if row.get("status") in {"would_run", "would_rebuild"}]
    represented = completed if not dry_run else rows
    total_rows = sum(int(row.get("output_rows", row.get("expected_output_rows", 0))) for row in represented)
    total_bytes = sum(int(row.get("output_bytes", row.get("estimated_output_bytes", 0))) for row in represented)
    missing = sum(int(row.get("missing_values", 0)) for row in represented)
    infinite = sum(int(row.get("infinite_values", 0)) for row in represented)
    schema_ok = all(bool(row.get("schema_consistent", True)) for row in represented)
    lines = [
        "# M2 Full By-Slice Summary",
        "",
        f"- Dry run: {dry_run}",
        f"- Slices planned: {len(rows)}",
        f"- Slices completed/skipped valid: {len(completed)}",
        f"- Slices planned only: {len(planned)}",
        f"- Failed slices: {len(failed)}",
        f"- Total anchors/output rows: {total_rows}",
        f"- Expected total anchors/output rows: {expected_total_rows}",
        f"- Metadata columns: {len(schema_info['metadata_columns'])}",
        f"- Numeric feature columns: {len(schema_info['numeric_feature_columns'])}",
        f"- Output columns: {len(schema_info['output_columns'])}",
        f"- Output disk usage: {total_bytes} ({_format_bytes(total_bytes)})",
        f"- Missing values: {missing}",
        f"- Infinite values: {infinite}",
        f"- Schema consistency status: {schema_ok}",
        f"- Runtime seconds: {elapsed:.3f}",
    ]
    if failed:
        lines.extend(["", "## Failed Slices", ""])
        lines.extend([f"- `{row['slice_id']}`: {row['error']}" for row in failed[:80]])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _sample_bytes_per_row(paths: dict[str, Path]) -> float:
    sample = paths["m2_prototype_dir"] / "m2_representation_sample.parquet"
    if not sample.exists():
        return 0.0
    rows = _row_count(sample)
    return sample.stat().st_size / rows if rows else 0.0


def process_slice(
    record: dict[str, Any],
    config: dict[str, Any],
    schema_info: dict[str, Any],
    m1_feature_columns: list[str],
    output_root: Path,
    dry_run: bool,
    resume: bool,
    force: bool,
    bytes_per_row: float,
) -> tuple[dict[str, Any], bool]:
    start = time.monotonic()
    slice_id = str(record["slice_id"])
    output_dir = output_root / slice_id
    output_path = output_dir / f"m2_representation_{slice_id}.parquet"
    report_path = output_dir / f"m2_report_{slice_id}.md"
    base = {
        "slice_id": slice_id,
        "m1_slice_dir": str(record["m1_slice_dir"]),
        "output_dir": str(output_dir),
        "output_path": str(output_path),
        "report_path": str(report_path),
        "metadata_columns": len(schema_info["metadata_columns"]),
        "source_feature_columns": len(schema_info["source_feature_columns"]),
        "numeric_feature_columns": len(schema_info["numeric_feature_columns"]),
        "expected_output_columns": len(schema_info["output_columns"]),
    }
    ok, input_row, error = _input_summary(
        record,
        m1_feature_columns,
        schema_info["expected_scales"],
    )
    base.update(input_row)
    if not ok:
        return {**base, "status": "failed", "error": error}, False

    existing_ok, existing_row, existing_error = validate_existing_output(
        output_path,
        schema_info["output_columns"],
        int(base["expected_output_rows"]),
        schema_info["numeric_feature_columns"],
    )
    if dry_run:
        if force:
            status = "would_run"
            error = ""
        elif resume and existing_ok:
            status = "skipped_valid"
            error = ""
        elif resume and output_path.exists() and not existing_ok:
            status = "would_rebuild"
            error = existing_error
        else:
            status = "would_run"
            error = ""
        estimated = int(bytes_per_row * int(base["expected_output_rows"]))
        row = {
            **base,
            **existing_row,
            "status": status,
            "error": error,
            "output_rows": int(base["expected_output_rows"]),
            "output_columns": len(schema_info["output_columns"]),
            "estimated_output_bytes": estimated,
            "schema_consistent": existing_ok if status == "skipped_valid" else True,
            "runtime_seconds": time.monotonic() - start,
        }
        return row, True

    if resume and existing_ok and not force:
        return (
            {
                **base,
                **existing_row,
                "status": "skipped_valid",
                "error": "",
                "runtime_seconds": time.monotonic() - start,
            },
            True,
        )

    try:
        feature_path = Path(str(record["feature_path"]))
        table = _read_feature_table(feature_path)
        validate_aligned_schema(table, m1_feature_columns)
        selected = select_numeric_feature_columns(table, config["feature_groups"])
        if selected != schema_info["source_feature_columns"]:
            raise ValueError("M2 source feature columns differ from schema-derived order.")
        matrix = build_m2_representation_table(
            table,
            feature_columns=schema_info["source_feature_columns"],
            expected_scales=schema_info["expected_scales"],
            metadata_columns=schema_info["metadata_columns"],
            anchor_keys=schema_info["anchor_keys"],
            scale_column=schema_info["scale_column"],
            separator=schema_info["separator"],
        )
        finite = finite_value_summary(matrix[schema_info["numeric_feature_columns"]])
        if finite["missing_values"] or finite["infinite_values"]:
            raise ValueError(
                "M2 matrix has missing or infinite numeric values "
                f"({finite['missing_values']}, {finite['infinite_values']})."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix.to_parquet(output_path, index=False)
        row = {
            **base,
            "status": "completed",
            "error": "",
            "output_rows": int(len(matrix)),
            "output_columns": int(matrix.shape[1]),
            "output_bytes": output_path.stat().st_size,
            "missing_values": finite["missing_values"],
            "infinite_values": finite["infinite_values"],
            "schema_consistent": list(matrix.columns) == schema_info["output_columns"],
            "runtime_seconds": time.monotonic() - start,
        }
        _write_slice_report(report_path, row)
        return row, True
    except Exception as exc:  # noqa: BLE001
        return (
            {
                **base,
                "status": "failed",
                "error": str(exc),
                "runtime_seconds": time.monotonic() - start,
            },
            False,
        )


def main() -> int:
    start = time.monotonic()
    args = parse_args()
    config = load_config(args.config)
    paths = _paths(config)
    output_root = paths["m2_output_dir"] / "by_slice"
    reports_dir = paths["m2_reports_dir"]
    output_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with paths["m1_global_schema"].open("r", encoding="utf-8") as handle:
        m1_schema = json.load(handle)
    schema_info = build_schema_info(config, m1_schema)
    schema_path = reports_dir / "m2_full_feature_schema.json"
    _write_schema(schema_path, config, schema_info)

    records = discover_slices(args, paths)
    if not records:
        raise FileNotFoundError("No M1 slice outputs selected.")

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    bytes_per_row = _sample_bytes_per_row(paths)
    for record in records:
        row, ok = process_slice(
            record,
            config,
            schema_info,
            list(m1_schema["feature_columns"]),
            output_root,
            args.dry_run,
            args.resume,
            args.force,
            bytes_per_row,
        )
        rows.append(row)
        if ok:
            print(f"{row['status'].upper()} {row['slice_id']}")
        else:
            failed.append(row)
            print(f"FAILED {row['slice_id']} {row['error']}")

    elapsed = time.monotonic() - start
    completed_csv = output_root / "completed_slices.csv"
    failed_txt = output_root / "failed_slices.txt"
    summary_csv = reports_dir / "m2_full_by_slice_summary.csv"
    summary_md = reports_dir / "m2_full_by_slice_summary.md"
    _write_csv(completed_csv, rows)
    _write_failed(failed_txt, failed)
    _write_csv(summary_csv, rows)
    _write_summary_md(
        summary_md,
        rows,
        failed,
        schema_info,
        args.dry_run,
        elapsed,
        int(config["expected"]["total_anchors"]),
    )

    represented = rows if args.dry_run else [
        row for row in rows if row.get("status") in {"completed", "skipped_valid"}
    ]
    total_rows = sum(int(row.get("output_rows", row.get("expected_output_rows", 0))) for row in represented)
    total_bytes = sum(int(row.get("output_bytes", row.get("estimated_output_bytes", 0))) for row in represented)
    missing = sum(int(row.get("missing_values", 0)) for row in represented)
    infinite = sum(int(row.get("infinite_values", 0)) for row in represented)
    safe = (
        not failed
        and total_rows == int(config["expected"]["total_anchors"])
        and len(schema_info["numeric_feature_columns"]) == 765
    )

    print(f"Wrote completed slices CSV: {completed_csv}")
    print(f"Wrote failed slices file: {failed_txt}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote summary report: {summary_md}")
    print(f"Wrote feature schema: {schema_path}")
    print(f"DRY_RUN {args.dry_run}")
    print(f"PLANNED_SLICES {len(rows)}")
    print(f"COMPLETED_SLICES {sum(row.get('status') in {'completed', 'skipped_valid'} for row in rows)}")
    print(f"FAILED_SLICES {len(failed)}")
    print(f"TOTAL_ROWS {total_rows}")
    print(f"EXPECTED_ROWS {config['expected']['total_anchors']}")
    print(f"METADATA_COLUMNS {len(schema_info['metadata_columns'])}")
    print(f"NUMERIC_FEATURE_COLUMNS {len(schema_info['numeric_feature_columns'])}")
    print(f"OUTPUT_COLUMNS {len(schema_info['output_columns'])}")
    print(f"OUTPUT_BYTES {total_bytes}")
    print(f"MISSING_VALUES {missing}")
    print(f"INFINITE_VALUES {infinite}")
    print(f"FULL_EXECUTION_SAFE {safe}")
    print(f"WALL_SECONDS {elapsed:.3f}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
