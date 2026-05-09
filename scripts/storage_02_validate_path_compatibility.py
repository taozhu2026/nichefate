#!/usr/bin/env python
"""Validate storage path compatibility after audit or approved migration."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.storage_ops import (  # noqa: E402
    DATA_ROOT,
    MANIFESTS_DIR,
    PROJECT_ROOT as DEFAULT_PROJECT_ROOT,
    REPORTS_DIR,
    SCRATCH_ROOT,
    bool_text,
    path_kind,
    read_csv_rows,
    write_csv_rows,
)


COMPAT_FIELDS = [
    "kind",
    "path",
    "new_path",
    "exists",
    "lexists",
    "path_type",
    "is_symlink",
    "symlink_target",
    "symlink_resolved",
    "broken_symlink",
    "classification",
    "read_check",
    "source_file",
    "line",
    "line_text",
]
PATH_PATTERN = re.compile(
    r"(/home/zhutao/scratch/nichefate[^\s'\"`),>]*"
    r"|/data/zhutao/nichefate_data[^\s'\"`),>]*"
    r"|/data/zhutao/nichefate[^\s'\"`),>]*"
    r"|/data/zhutao/merfish_colitis_raw[^\s'\"`),>]*)"
)
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".toml", ".txt", ".sh", ".json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFESTS_DIR / "storage_migration_manifest.csv",
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--scratch-root", type=Path, default=SCRATCH_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    return parser.parse_args()


def lightweight_read_check(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        if path.is_dir():
            entries = [entry.name for entry in sorted(path.iterdir())[:5]]
            return f"dir_list_ok:{entries}"
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                header = handle.readline().strip()
                first = handle.readline().strip()
            return f"text_table_head_ok:header={header[:120]} first={first[:120]}"
        if suffix == ".json" and path.stat().st_size <= 50 * 1024 * 1024:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return f"json_ok:{type(data).__name__}"
        if suffix in {".yaml", ".yml"} and path.stat().st_size <= 10 * 1024 * 1024:
            try:
                import yaml
            except ImportError:
                return "yaml_skipped_missing_dependency"
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            return f"yaml_ok:{type(data).__name__}"
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                return "parquet_skipped_missing_pyarrow"
            parquet_file = pq.ParquetFile(path)
            names = parquet_file.schema_arrow.names
            preview = parquet_file.read_row_group(0, columns=names[: min(5, len(names))])
            return f"parquet_schema_head_ok:cols={names[:10]} rows={preview.num_rows}"
        if suffix in {".h5", ".h5ad", ".hdf5"}:
            try:
                import h5py
            except ImportError:
                return "hdf5_skipped_missing_h5py"
            with h5py.File(path, "r") as handle:
                keys = sorted(list(handle.keys()))
            return f"hdf5_open_ok:keys={keys[:10]}"
        with path.open("rb") as handle:
            handle.read(512)
        return "binary_head_ok"
    except Exception as exc:
        return f"read_check_failed:{type(exc).__name__}:{exc}"


def symlink_details(path: Path) -> dict[str, str]:
    is_link = path.is_symlink()
    return {
        "is_symlink": bool_text(is_link),
        "symlink_target": os.readlink(path) if is_link else "",
        "symlink_resolved": path.resolve(strict=False).as_posix() if is_link else "",
        "broken_symlink": bool_text(is_link and not path.exists()),
    }


def manifest_validation_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        for kind, key in (("manifest_old_path", "old_path"), ("manifest_new_path", "new_path")):
            path = Path(row[key])
            details = symlink_details(path)
            output.append(
                {
                    "kind": kind,
                    "path": str(path),
                    "new_path": row.get("new_path", ""),
                    "exists": bool_text(path.exists()),
                    "lexists": bool_text(os.path.lexists(path)),
                    "path_type": path_kind(path),
                    **details,
                    "classification": row.get("move_status", ""),
                    "read_check": lightweight_read_check(path),
                    "source_file": "",
                    "line": "",
                    "line_text": "",
                }
            )
    return output


def manifest_prefix_map(rows: list[dict[str, str]]) -> list[tuple[Path, dict[str, str]]]:
    pairs = []
    for row in rows:
        old_path = row.get("old_path")
        if old_path:
            pairs.append((Path(old_path), row))
    pairs.sort(key=lambda pair: len(pair[0].as_posix()), reverse=True)
    return pairs


def classify_hard_coded_path(path: Path, prefixes: list[tuple[Path, dict[str, str]]]) -> str:
    if path.is_symlink() and path.exists():
        return "OK_via_symlink"
    if path.exists():
        for prefix, row in prefixes:
            try:
                path.relative_to(prefix)
            except ValueError:
                continue
            if row.get("symlink_required") == "true" and row.get("move_status") in {
                "planned",
                "moved",
                "verified",
            }:
                return "should_update_config"
        return "should_update_config"
    for prefix, row in prefixes:
        try:
            path.relative_to(prefix)
        except ValueError:
            continue
        if row.get("move_status") == "skipped" and row.get("category") == "do_not_move":
            return "obsolete"
    return "unresolved"


def scan_hard_coded_paths(
    project_root: Path,
    prefixes: list[tuple[Path, dict[str, str]]],
) -> list[dict[str, object]]:
    search_roots = [
        project_root / "configs",
        project_root / "scripts",
        project_root / "src",
        project_root / "tests",
        project_root / "README.md",
        project_root / "docs",
    ]
    rows: list[dict[str, object]] = []
    files: list[Path] = []
    for root in search_roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix in TEXT_SUFFIXES
            )
    for file_path in sorted(set(files)):
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(lines, start=1):
            for match in PATH_PATTERN.findall(line):
                if "\\" in match or "[" in match:
                    continue
                path = Path(match.rstrip(".,:;"))
                details = symlink_details(path)
                rows.append(
                    {
                        "kind": "hard_coded_reference",
                        "path": str(path),
                        "new_path": "",
                        "exists": bool_text(path.exists()),
                        "lexists": bool_text(os.path.lexists(path)),
                        "path_type": path_kind(path),
                        **details,
                        "classification": classify_hard_coded_path(path, prefixes),
                        "read_check": lightweight_read_check(path),
                        "source_file": str(file_path.relative_to(project_root)),
                        "line": line_no,
                        "line_text": line.strip(),
                    }
                )
    return rows


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["classification"])
        counts[key] = counts.get(key, 0) + 1
    lines = ["# Storage Path Compatibility", "", "## Classification Counts"]
    for key, value in sorted(counts.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Issues"])
    for row in rows:
        if row["classification"] in {"unresolved", "obsolete"} or row["broken_symlink"] == "true":
            lines.append(
                f"- {row['classification']}: `{row['path']}` "
                f"source={row['source_file']}:{row['line']} read={row['read_check']}"
            )
    lines.extend(["", "## Hard-Coded References"])
    for row in rows:
        if row["kind"] == "hard_coded_reference":
            lines.append(
                f"- {row['classification']}: `{row['path']}` "
                f"in `{row['source_file']}:{row['line']}`"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_rows = read_csv_rows(args.manifest)
    prefixes = manifest_prefix_map(manifest_rows)
    rows = manifest_validation_rows(manifest_rows)
    rows.extend(scan_hard_coded_paths(args.project_root, prefixes))
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.reports_dir / "storage_path_compatibility.csv"
    md_path = args.reports_dir / "storage_path_compatibility.md"
    write_csv_rows(csv_path, rows, COMPAT_FIELDS)
    write_markdown(md_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
