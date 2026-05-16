from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def tsv_column_consistency(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "consistent": False, "bad_rows": []}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {"path": str(path), "exists": True, "consistent": True, "column_count": 0, "bad_rows": []}
    column_count = len(lines[0].split("\t"))
    bad_rows = [idx for idx, line in enumerate(lines[1:], start=2) if len(line.split("\t")) != column_count]
    return {
        "path": str(path),
        "exists": True,
        "consistent": not bad_rows,
        "column_count": column_count,
        "bad_rows": bad_rows,
    }


def table_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        if path.suffix == ".json":
            return None
        sep = "\t" if path.suffix == ".tsv" else ","
        return int(sum(1 for _ in path.open(encoding="utf-8")) - 1)
    except Exception:
        return None


__all__ = [name for name in globals() if not name.startswith("__")]
