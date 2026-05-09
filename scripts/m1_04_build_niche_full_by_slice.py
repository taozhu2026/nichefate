#!/usr/bin/env python
"""Run full M1 niche construction sequentially over by-slice M0 objects."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anndata as ad
import pandas as pd

from nichefate.io import load_config
from nichefate.niche import load_global_feature_schema
from nichefate.niche_qc import composition_sum_qc, summarize_feature_integrity, validate_neighbor_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m1_niche_construction.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--global-schema", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--slice-file", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _paths(config: dict) -> dict[str, Path]:
    return {
        key: Path(value)
        for key, value in config["paths"].items()
        if isinstance(value, str)
    }


def _format_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _safe_token(value: object) -> str:
    import re

    token = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")
    return token or "slice"


def _slice_record(path: Path) -> dict[str, object]:
    data = ad.read_h5ad(path, backed="r")
    try:
        slice_id = str(data.obs["slice_id"].iloc[0]) if "slice_id" in data.obs else path.stem
        time_label = str(data.obs["time"].iloc[0]) if "time" in data.obs else ""
        return {
            "slice_file": path.name,
            "slice_path": str(path.resolve()),
            "slice_id": slice_id,
            "slice_token": _safe_token(slice_id),
            "time": time_label,
            "n_cells": int(data.n_obs),
        }
    finally:
        if hasattr(data, "file"):
            data.file.close()


def _discover_slices(args: argparse.Namespace, paths: dict[str, Path]) -> list[dict[str, object]]:
    if args.slice_file is not None:
        records = [_slice_record(args.slice_file)]
    else:
        records = [_slice_record(path) for path in sorted(paths["m0_by_slice_dir"].glob("*.m0.h5ad"))]
    if args.max_slices is not None:
        records = records[: args.max_slices]
    return records


def _slice_output_paths(output_dir: Path, record: dict[str, object]) -> dict[str, Path]:
    slice_dir = output_dir / str(record["slice_token"])
    token = str(record["slice_token"])
    parquet_path = slice_dir / f"niche_features_{token}.parquet"
    csv_path = slice_dir / f"niche_features_{token}.csv"
    feature_path = parquet_path if parquet_path.exists() else csv_path
    return {
        "slice_dir": slice_dir,
        "feature_parquet": parquet_path,
        "feature_csv": csv_path,
        "feature": feature_path,
        "neighbor": slice_dir / f"neighbor_index_{token}.npz",
        "report": slice_dir / f"m1_report_{token}.md",
    }


def _read_feature_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _pilot_size_rates(paths: dict[str, Path]) -> tuple[float, float]:
    pilot_dir = paths["m1_output_dir"] / "pilot_full_slice"
    feature_paths = sorted(pilot_dir.glob("niche_features_*.parquet")) + sorted(
        pilot_dir.glob("niche_features_*.csv")
    )
    neighbor_paths = sorted(pilot_dir.glob("neighbor_index_*.npz"))
    feature_bytes_per_row = 0.0
    neighbor_bytes_per_anchor = 0.0
    if feature_paths:
        table = _read_feature_table(feature_paths[0])
        if len(table):
            feature_bytes_per_row = feature_paths[0].stat().st_size / len(table)
            if "anchor_index" in table:
                n_anchors = table["anchor_index"].nunique()
                if neighbor_paths and n_anchors:
                    neighbor_bytes_per_anchor = neighbor_paths[0].stat().st_size / n_anchors
    return feature_bytes_per_row, neighbor_bytes_per_anchor


def _validate_outputs(
    record: dict[str, object],
    output_dir: Path,
    schema: dict[str, object],
) -> tuple[bool, dict[str, object], str]:
    outputs = _slice_output_paths(output_dir, record)
    feature_path = outputs["feature"]
    if not feature_path.exists():
        return False, {}, "missing feature table"
    if not outputs["neighbor"].exists():
        return False, {}, "missing neighbor index"
    if not outputs["report"].exists():
        return False, {}, "missing report"

    features = _read_feature_table(feature_path)
    n_scales = len(schema["scales"])
    expected_rows = int(record["n_cells"]) * n_scales
    if len(features) != expected_rows:
        return False, {}, f"feature rows {len(features)} != expected {expected_rows}"
    expected_columns = list(schema["feature_columns"])
    if list(features.columns) != expected_columns:
        return False, {}, "feature schema is not aligned to global schema"
    if features.duplicated(["slice_id", "anchor_index", "scale"]).any():
        return False, {}, "duplicated anchor/scale rows"
    scale_counts = features.groupby(["slice_id", "anchor_index"], observed=True)["scale"].nunique()
    if not bool((scale_counts == n_scales).all()):
        return False, {}, "not every anchor has one row per scale"
    integrity = summarize_feature_integrity(features)
    if int(integrity["infinite_values"].sum()):
        return False, {}, "infinite values detected"
    composition = composition_sum_qc(features)
    if not bool((composition["rows_not_close_to_one"] == 0).all()):
        return False, {}, "composition row sums failed"
    neighbor_qc = validate_neighbor_npz(
        outputs["neighbor"],
        feature_table=features,
        slice_n_obs={str(record["slice_id"]): int(record["n_cells"]), str(record["slice_file"]): int(record["n_cells"])},
        expected_entries=n_scales,
        include_anchor=True,
    )
    if not bool(neighbor_qc["ok"].all()):
        return False, {}, "neighbor index QC failed"
    n_neighbors = features.groupby("scale", observed=True)["n_neighbors"].mean().to_dict()
    summary = {
        "feature_path": str(feature_path),
        "neighbor_path": str(outputs["neighbor"]),
        "report_path": str(outputs["report"]),
        "feature_rows": len(features),
        "feature_columns": len(features.columns),
        "feature_bytes": feature_path.stat().st_size,
        "neighbor_bytes": outputs["neighbor"].stat().st_size,
        "report_bytes": outputs["report"].stat().st_size,
        "missing_values": int(integrity["missing_values"].sum()),
        "infinite_values": int(integrity["infinite_values"].sum()),
        "composition_valid": True,
        "neighbor_valid": True,
    }
    for scale, value in n_neighbors.items():
        summary[f"mean_neighbors_{scale}"] = float(value)
    return True, summary, ""


def _planned_command(
    config_path: str,
    record: dict[str, object],
    slice_output_dir: Path,
    schema_path: Path,
    force: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "m1_03_build_niche_full.py"),
        "--config",
        config_path,
        "--slice-file",
        str(record["slice_path"]),
        "--output-dir",
        str(slice_output_dir),
        "--global-schema",
        str(schema_path),
        "--report-prefix",
        "m1_report",
    ]
    if force:
        command.append("--force")
    return command


def _human_command(
    config_path: str,
    record: dict[str, object],
    slice_output_dir: Path,
    schema_path: Path,
    force: bool,
) -> str:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "omicverse",
        "python",
        "scripts/m1_03_build_niche_full.py",
        "--config",
        config_path,
        "--slice-file",
        str(record["slice_path"]),
        "--output-dir",
        str(slice_output_dir),
        "--global-schema",
        str(schema_path),
        "--report-prefix",
        "m1_report",
    ]
    if force:
        command.append("--force")
    return " ".join(command)


def _parse_worker_stdout(stdout: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for line in stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "WALL_SECONDS" and len(parts) > 1:
            values["wall_seconds"] = float(parts[1])
        elif parts[0] == "MAX_RSS_KB" and len(parts) > 1:
            values["max_rss_kb"] = int(parts[1])
        elif parts[0] == "FEATURE_SHAPE" and len(parts) > 2:
            values["feature_rows_stdout"] = int(parts[1])
            values["feature_columns_stdout"] = int(parts[2])
        elif parts[0] == "NEIGHBOR_ENTRIES" and len(parts) > 1:
            values["neighbor_entries_stdout"] = int(parts[1])
    return values


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({column for row in rows for column in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_failed(path: Path, failed: list[dict[str, object]]) -> None:
    lines = [f"{row['slice_id']}\t{row['error']}" for row in failed]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_summary_md(
    path: Path,
    rows: list[dict[str, object]],
    failed: list[dict[str, object]],
    dry_run: bool,
    command_list: list[str],
    elapsed: float,
) -> None:
    completed = [row for row in rows if row.get("status") in {"completed", "skipped_valid", "would_run"}]
    built = [row for row in rows if row.get("status") == "completed"]
    total_cells = sum(int(row.get("n_cells", 0)) for row in completed)
    total_rows = sum(int(row.get("feature_rows", 0)) for row in completed if row.get("feature_rows"))
    total_bytes = sum(
        int(row.get("feature_bytes", 0))
        + int(row.get("neighbor_bytes", 0))
        + int(row.get("report_bytes", 0))
        for row in completed
    )
    lines = [
        "# M1 Full By-Slice Summary",
        "",
        f"- Dry run: {dry_run}",
        f"- Slices in plan: {len(rows)}",
        f"- Completed/skipped/would-run slices: {len(completed)}",
        f"- Newly built slices in this invocation: {len(built)}",
        f"- Failed slices: {len(failed)}",
        f"- Total cells represented: {total_cells}",
        f"- Total feature rows represented: {total_rows}",
        f"- Output bytes represented: {total_bytes} ({_format_bytes(total_bytes)})",
        f"- Wall seconds for runner: {elapsed:.3f}",
        "",
        "## Command List",
        "",
    ]
    lines.extend([f"- `{command}`" for command in command_list[:80]])
    if failed:
        lines.extend(["", "## Failed Slices", ""])
        lines.extend([f"- `{row['slice_id']}`: {row['error']}" for row in failed])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    start = time.monotonic()
    args = parse_args()
    config = load_config(args.config)
    paths = _paths(config)
    reports_dir = paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_dir = (args.output_dir or (paths["m1_output_dir"] / "by_slice")).resolve()
    schema_path = (args.global_schema or (reports_dir / "m1_global_schema.json")).resolve()
    schema = load_global_feature_schema(schema_path)
    if schema is None:
        raise FileNotFoundError(f"Missing global schema: {schema_path}")

    records = _discover_slices(args, paths)
    if not records:
        raise FileNotFoundError("No M0 by-slice files selected.")

    rows: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    command_list: list[str] = []
    feature_bytes_per_row, neighbor_bytes_per_anchor = _pilot_size_rates(paths)
    n_scales = len(schema["scales"])

    for record in records:
        outputs = _slice_output_paths(output_dir, record)
        slice_output_dir = outputs["slice_dir"]
        valid_existing, existing_summary, existing_error = _validate_outputs(record, output_dir, schema)
        should_skip = args.resume and valid_existing and not args.force
        force_worker = args.force or (args.resume and not valid_existing)
        human_command = _human_command(
            args.config,
            record,
            slice_output_dir,
            schema_path,
            force_worker,
        )
        command_list.append(human_command)
        base_row = {
            "slice_id": record["slice_id"],
            "slice_file": record["slice_file"],
            "slice_path": record["slice_path"],
            "n_cells": record["n_cells"],
            "feature_rows": int(record["n_cells"]) * n_scales,
            "feature_bytes": int(feature_bytes_per_row * int(record["n_cells"]) * n_scales),
            "neighbor_bytes": int(neighbor_bytes_per_anchor * int(record["n_cells"])),
            "report_bytes": 0,
            "time": record["time"],
            "output_dir": str(slice_output_dir),
        }
        if args.dry_run:
            status = "skipped_valid" if should_skip else "would_run"
            rows.append({**base_row, **existing_summary, "status": status, "error": existing_error})
            print(f"DRY_RUN {status} {record['slice_id']} {human_command}")
            continue
        if should_skip:
            rows.append({**base_row, **existing_summary, "status": "skipped_valid", "error": ""})
            print(f"SKIP_VALID {record['slice_id']}")
            continue

        slice_output_dir.mkdir(parents=True, exist_ok=True)
        command = _planned_command(args.config, record, slice_output_dir, schema_path, force_worker)
        print(f"RUN_SLICE {record['slice_id']}")
        try:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=False,
                text=True,
                capture_output=True,
            )
            parsed = _parse_worker_stdout(result.stdout)
            if result.returncode != 0:
                error = (result.stderr or result.stdout).strip().splitlines()[-1]
                failed.append({**base_row, "status": "failed", "error": error})
                rows.append({**base_row, **parsed, "status": "failed", "error": error})
                print(f"FAILED {record['slice_id']} {error}")
                continue
            valid, summary, error = _validate_outputs(record, output_dir, schema)
            if not valid:
                failed.append({**base_row, "status": "failed", "error": error})
                rows.append({**base_row, **parsed, **summary, "status": "failed", "error": error})
                print(f"FAILED_VALIDATION {record['slice_id']} {error}")
                continue
            rows.append({**base_row, **parsed, **summary, "status": "completed", "error": ""})
            print(f"COMPLETED {record['slice_id']}")
        except Exception as exc:  # noqa: BLE001
            failed.append({**base_row, "status": "failed", "error": str(exc)})
            rows.append({**base_row, "status": "failed", "error": str(exc)})
            print(f"FAILED {record['slice_id']} {exc}")

    elapsed = time.monotonic() - start
    completed_csv = output_dir / "completed_slices.csv"
    failed_txt = output_dir / "failed_slices.txt"
    summary_csv = reports_dir / "m1_full_by_slice_summary.csv"
    summary_md = reports_dir / "m1_full_by_slice_summary.md"
    _write_csv(completed_csv, rows)
    _write_failed(failed_txt, failed)
    _write_csv(summary_csv, rows)
    _write_summary_md(summary_md, rows, failed, args.dry_run, command_list, elapsed)

    print(f"Wrote completed slices CSV: {completed_csv}")
    print(f"Wrote failed slices file: {failed_txt}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote summary report: {summary_md}")
    print(f"N_SLICES {len(records)}")
    print(f"FAILED_SLICES {len(failed)}")
    print(f"WALL_SECONDS {elapsed:.3f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
