"""Download helpers for the limited M0 Dryad file set."""

from __future__ import annotations

import csv
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from nichefate.io import expected_raw_files, file_size_gb, paths_from_config

DRYAD_LISTED_SIZE_GB = {
    "adata.h5ad": 17.96,
    "adata_day35.h5ad": 1.51,
    "README.md": 0.000001,
    "ligand_receptor_pair_masterlist.csv": 0.000001,
}

REQUIRED_CORE_DOWNLOAD_GB = 17.96 + 1.51
FULL_DRYAD_DATASET_GB = 108.70

MINIMUM_BYTES = {
    "adata.h5ad": int(17.96 * 1000**3),
    "adata_day35.h5ad": int(1.51 * 1000**3),
    "README.md": 1024,
    "ligand_receptor_pair_masterlist.csv": 1024,
}


def _required_file_names(config: dict[str, Any]) -> set[str]:
    required, _optional = expected_raw_files(config)
    return set(required)


def download_file(
    url: str,
    dest: str | Path,
    *,
    force: bool = False,
    chunk_size: int = 1024 * 1024,
) -> dict[str, object]:
    """Stream a URL to a destination file."""

    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return {
            "filename": destination.name,
            "url": url,
            "path": str(destination),
            "status": "skipped_existing",
            "size_bytes": destination.stat().st_size,
        }

    temp_path = destination.with_suffix(destination.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with temp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=chunk_size)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if temp_path.exists():
            temp_path.unlink()
        return {
            "filename": destination.name,
            "url": url,
            "path": str(destination),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "size_bytes": destination.stat().st_size if destination.exists() else 0,
        }

    temp_path.replace(destination)
    return {
        "filename": destination.name,
        "url": url,
        "path": str(destination),
        "status": "downloaded",
        "size_bytes": destination.stat().st_size,
    }


def check_core_files(config: dict[str, Any]) -> list[dict[str, object]]:
    """Check existence and minimum-size status for configured core files."""

    raw_dir = paths_from_config(config)["raw_dir"]
    urls = config.get("download", {}).get("files", {})
    if not isinstance(urls, dict):
        raise ValueError("Config section 'download.files' must be a mapping.")

    required_names = _required_file_names(config)
    rows: list[dict[str, object]] = []
    for filename, url in urls.items():
        path = raw_dir / filename
        size_bytes = path.stat().st_size if path.exists() else 0
        minimum_bytes = MINIMUM_BYTES.get(filename, 0)
        rows.append(
            {
                "filename": filename,
                "url": url,
                "path": str(path),
                "required": filename in required_names,
                "exists": path.exists(),
                "size_bytes": size_bytes,
                "size_gb": round(file_size_gb(path), 6),
                "dryad_listed_size_gb": DRYAD_LISTED_SIZE_GB.get(filename, 0),
                "minimum_bytes": minimum_bytes,
                "meets_minimum": size_bytes >= minimum_bytes,
                "status": "ok" if size_bytes >= minimum_bytes else "missing_or_small",
            }
        )
    return rows


def write_download_manifest(
    config: dict[str, Any],
    output_csv: str | Path,
    *,
    rows: list[dict[str, object]] | None = None,
) -> Path:
    """Write a CSV manifest for configured Dryad core files."""

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows = rows if rows is not None else check_core_files(config)
    fieldnames = [
        "filename",
        "url",
        "path",
        "required",
        "exists",
        "size_bytes",
        "size_gb",
        "dryad_listed_size_gb",
        "minimum_bytes",
        "meets_minimum",
        "status",
        "error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest_rows)
    return output_path
