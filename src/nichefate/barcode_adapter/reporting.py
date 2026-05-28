from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


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
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def atomic_write_text(path: str | Path, text: str, overwrite: bool = False) -> Path:
    resolved = Path(path)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {resolved}")
    ensure_dir(resolved.parent)
    tmp = resolved.with_name(resolved.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, resolved)
    return resolved


def atomic_write_json(path: str | Path, payload: dict[str, Any], overwrite: bool = False) -> Path:
    return atomic_write_text(
        path,
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )


def atomic_write_tsv(path: str | Path, frame: pd.DataFrame, overwrite: bool = False) -> Path:
    resolved = Path(path)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {resolved}")
    ensure_dir(resolved.parent)
    tmp = resolved.with_name(resolved.name + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, resolved)
    return resolved


def atomic_write_tsv_gz(path: str | Path, frame: pd.DataFrame, overwrite: bool = False) -> Path:
    resolved = Path(path)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {resolved}")
    ensure_dir(resolved.parent)
    tmp = resolved.with_name(resolved.name + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False, compression="gzip")
    os.replace(tmp, resolved)
    return resolved


def markdown_table(frame: pd.DataFrame, limit: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(limit).copy()
    columns = list(view.columns)
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in view.to_dict(orient="records"):
        values = [
            str(row.get(column, "")).replace("|", "\\|").replace("\n", " ")
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def path_has_ssd(path: str | Path) -> bool:
    return Path(path).expanduser().as_posix().startswith("/ssd/")
