#!/usr/bin/env python
"""Verify local Dryad core files without loading AnnData matrices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import h5py

from nichefate.io import ensure_dirs, expected_raw_files, load_config, paths_from_config
from nichefate.qc import write_json_report, write_markdown_report

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
HTML_PREFIXES = (b"<html", b"<!doctype")
MINIMUM_BYTES = {
    "adata.h5ad": 10 * 1024**3,
    "adata_day35.h5ad": 1 * 1024**3,
    "README.md": 1024,
    "ligand_receptor_pair_masterlist.csv": 1024,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m0_merfish_colitis.yaml")
    return parser.parse_args()


def first_bytes(path: Path, n_bytes: int = 512) -> bytes:
    with path.open("rb") as handle:
        return handle.read(n_bytes)


def describe_h5_node(node: h5py.Dataset | h5py.Group) -> dict[str, Any]:
    if isinstance(node, h5py.Dataset):
        return {"kind": "dataset", "shape": list(node.shape), "dtype": str(node.dtype)}
    return {
        "kind": "group",
        "keys": sorted(list(node.keys())),
        "attrs": {str(key): str(value) for key, value in node.attrs.items()},
    }


def inspect_h5ad_with_h5py(path: Path) -> dict[str, Any]:
    print(f"Opening with h5py: {path}", flush=True)
    with h5py.File(path, "r") as handle:
        details: dict[str, Any] = {
            "top_level_keys": sorted(list(handle.keys())),
            "obs_keys": sorted(list(handle["obs"].keys())) if "obs" in handle else [],
            "var_keys": sorted(list(handle["var"].keys())) if "var" in handle else [],
            "x": describe_h5_node(handle["X"]) if "X" in handle else None,
        }
        for key in ("layers", "obsm", "uns"):
            details[f"{key}_keys"] = (
                sorted(list(handle[key].keys())) if key in handle else []
            )
        return details


def verify_file(filename: str, path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "filename": filename,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "minimum_bytes": MINIMUM_BYTES[filename],
        "meets_minimum": False,
        "html_like": False,
        "hdf5_magic": None,
        "h5py_open_ok": None,
        "ok": False,
        "errors": [],
    }
    if not path.exists():
        row["errors"].append("missing")
        return row

    row["meets_minimum"] = row["size_bytes"] > row["minimum_bytes"]
    if not row["meets_minimum"]:
        row["errors"].append(
            f"too small: {row['size_bytes']} <= {row['minimum_bytes']} bytes"
        )

    prefix = first_bytes(path)
    normalized = prefix.lstrip().lower()
    row["html_like"] = any(normalized.startswith(marker) for marker in HTML_PREFIXES)
    if row["html_like"]:
        row["errors"].append("looks like an HTML error page")

    if filename.endswith(".h5ad"):
        row["hdf5_magic"] = prefix.startswith(HDF5_MAGIC)
        if not row["hdf5_magic"]:
            row["errors"].append("missing HDF5 magic bytes")
        try:
            row["h5ad_structure"] = inspect_h5ad_with_h5py(path)
            row["h5py_open_ok"] = True
        except Exception as exc:
            row["h5py_open_ok"] = False
            row["errors"].append(f"h5py open failed: {type(exc).__name__}: {exc}")

    row["ok"] = not row["errors"]
    return row


def markdown_report(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["# Raw File Verification", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['filename']}",
                f"- Path: `{row['path']}`",
                f"- Exists: {row['exists']}",
                f"- Size bytes: {row['size_bytes']}",
                f"- Minimum bytes: {row['minimum_bytes']}",
                f"- Meets minimum: {row['meets_minimum']}",
                f"- HTML-like: {row['html_like']}",
                f"- HDF5 magic: {row['hdf5_magic']}",
                f"- h5py open OK: {row['h5py_open_ok']}",
                f"- OK: {row['ok']}",
            ]
        )
        if row["errors"]:
            lines.append(f"- Errors: {'; '.join(row['errors'])}")
        structure = row.get("h5ad_structure")
        if structure:
            lines.append(f"- Top-level keys: {structure['top_level_keys']}")
            lines.append(f"- obs keys: {structure['obs_keys']}")
            lines.append(f"- var keys: {structure['var_keys']}")
            lines.append(f"- X: {structure['x']}")
        lines.append("")
    return lines


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dirs = ensure_dirs(config)
    raw_dir = paths_from_config(config)["raw_dir"]
    required, optional = expected_raw_files(config)

    rows = []
    for filename in required + optional:
        print(f"Verifying {filename}", flush=True)
        rows.append(verify_file(filename, raw_dir / filename))

    json_path = dirs["reports"] / "raw_file_verification.json"
    md_path = dirs["reports"] / "raw_file_verification.md"
    write_json_report({"files": rows}, json_path)
    write_markdown_report(markdown_report(rows), md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    failed = [row for row in rows if not row["ok"]]
    if failed:
        print("Raw file verification failed:")
        for row in failed:
            print(f"- {row['filename']}: {'; '.join(row['errors'])}")
        return 1
    print("Raw file verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
