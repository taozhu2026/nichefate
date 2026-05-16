from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schemas import *


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def atomic_write_text(path: Path, text: str, overwrite: bool = False) -> bool:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        return False
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return True


def atomic_write_json(path: Path, payload: dict[str, Any], overwrite: bool = False) -> bool:
    text = json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n"
    return atomic_write_text(path, text, overwrite=overwrite)


def atomic_write_tsv(path: Path, frame: pd.DataFrame, overwrite: bool = False) -> bool:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        return False
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)
    return True


def atomic_write_csv(path: Path, frame: pd.DataFrame, overwrite: bool = False) -> bool:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        return False
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return True


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def git_branch() -> str:
    return run_git("branch", "--show-current") or "unknown"


def git_root() -> str:
    return run_git("rev-parse", "--show-toplevel") or str(PROJECT_ROOT)


def git_status_short() -> list[str]:
    status = run_git("status", "--short")
    return [line for line in status.splitlines() if line.strip()]


def disk_usage(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "free_gib": round(usage.free / (1024**3), 3),
    }


def file_summary(path: Path) -> dict[str, Any]:
    exists = path.exists()
    if not exists:
        return {
            "path": str(path),
            "exists": False,
            "bytes": 0,
            "mtime_utc": None,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat(),
    }


def read_memory_info() -> dict[str, Any]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {"available": False, "reason": "/proc/meminfo not present"}
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0]) * 1024
        except ValueError:
            continue
    return {
        "available": True,
        "mem_total_gib": round(values.get("MemTotal", 0) / (1024**3), 3),
        "mem_available_gib": round(values.get("MemAvailable", 0) / (1024**3), 3),
        "swap_total_gib": round(values.get("SwapTotal", 0) / (1024**3), 3),
        "swap_free_gib": round(values.get("SwapFree", 0) / (1024**3), 3),
    }


def list_report_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            files.append(str(path.relative_to(PROJECT_ROOT)))
        except ValueError:
            files.append(str(path))
    return files


__all__ = [name for name in globals() if not name.startswith("__")]
