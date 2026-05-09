"""Utilities for conservative nichefate storage audits and migrations."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

SCRATCH_ROOT = Path("/home/zhutao/scratch/nichefate")
DATA_ROOT = Path("/data/zhutao/nichefate")
PROJECT_ROOT = Path("/home/zhutao/projects/nichefate")
LEGACY_DATA_ROOTS = (
    Path("/data/zhutao/nichefate_data"),
    Path("/data/zhutao/merfish_colitis_raw"),
)
STORAGE_ROOT = SCRATCH_ROOT / "storage"
REPORTS_DIR = STORAGE_ROOT / "reports"
MANIFESTS_DIR = STORAGE_ROOT / "manifests"
LOGS_DIR = STORAGE_ROOT / "logs"

DATA_SUBDIRS = (
    "raw",
    "m0/input",
    "m0/intermediate",
    "m0/by_slice",
    "m0/reports",
    "m1/archived_or_heavy",
    "m2/archived_or_heavy",
    "external",
    "manifests",
    "logs",
)

MANIFEST_FIELDS = [
    "old_path",
    "new_path",
    "path_type",
    "category",
    "size_bytes",
    "symlink_required",
    "move_status",
    "reason",
    "downstream_risk",
    "notes",
]

DRY_RUN_SUMMARY_FIELDS = [
    "planned_old_path",
    "planned_new_path",
    "size",
    "category",
    "downstream_risk",
    "symlink_will_be_created",
    "source_already_exists_on_data",
    "target_already_exists",
]

ELIGIBLE_CATEGORIES = {"raw", "m0_input", "m0_intermediate", "archive"}


def ensure_report_dirs(storage_root: Path = STORAGE_ROOT) -> dict[str, Path]:
    paths = {
        "storage": storage_root,
        "reports": storage_root / "reports",
        "manifests": storage_root / "manifests",
        "logs": storage_root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_data_root(data_root: Path = DATA_ROOT) -> None:
    for subdir in DATA_SUBDIRS:
        (data_root / subdir).mkdir(parents=True, exist_ok=True)


def bool_text(value: object) -> str:
    return "true" if bool(value) else "false"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean text, got {value!r}")


def path_kind(path: Path) -> str:
    if path.is_symlink():
        target = path.resolve(strict=False)
        if target.is_dir():
            return "dir"
        if target.is_file():
            return "file"
        return "symlink"
    if path.is_dir():
        return "dir"
    if path.is_file():
        return "file"
    return "missing"


def is_on_data(path: Path) -> bool:
    text = path.as_posix()
    if text.startswith("/data/"):
        return True
    if path.is_symlink():
        return path.resolve(strict=False).as_posix().startswith("/data/")
    return False


def tree_size_bytes(path: Path) -> int:
    if not os.path.lexists(path):
        return 0
    if path.is_symlink():
        return path.lstat().st_size
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in dirs:
            entry = root_path / name
            if entry.is_symlink():
                total += entry.lstat().st_size
        for name in files:
            entry = root_path / name
            try:
                total += entry.lstat().st_size if entry.is_symlink() else entry.stat().st_size
            except OSError:
                continue
    return total


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024 or unit == "PiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def write_csv_rows(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rsync_available() -> bool:
    return shutil.which("rsync") is not None


def copy_path(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if rsync_available():
        if source.is_dir() and not source.is_symlink():
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["rsync", "-aH", "--info=progress2", f"{source}/", f"{target}/"],
                check=True,
            )
        else:
            subprocess.run(
                ["rsync", "-aH", "--info=progress2", str(source), str(target)],
                check=True,
            )
        return
    if source.is_dir() and not source.is_symlink():
        shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target, follow_symlinks=False)


def is_row_eligible(row: dict[str, str], allowed_categories: set[str]) -> bool:
    return (
        row.get("move_status") == "planned"
        and row.get("downstream_risk") == "low"
        and row.get("category") in allowed_categories
        and row.get("category") in ELIGIBLE_CATEGORIES
    )


def dry_run_summary_row(row: dict[str, str]) -> dict[str, object]:
    old_path = Path(row["old_path"])
    new_path = Path(row["new_path"])
    return {
        "planned_old_path": row["old_path"],
        "planned_new_path": row["new_path"],
        "size": row.get("size_bytes", "0"),
        "category": row.get("category", ""),
        "downstream_risk": row.get("downstream_risk", ""),
        "symlink_will_be_created": row.get("symlink_required", "false"),
        "source_already_exists_on_data": bool_text(is_on_data(old_path)),
        "target_already_exists": bool_text(os.path.lexists(new_path)),
    }


def rows_by_status(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get("move_status", "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts
