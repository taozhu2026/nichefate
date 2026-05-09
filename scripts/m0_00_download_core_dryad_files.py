#!/usr/bin/env python
"""Download or manifest the limited M0 Dryad core file set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.download import (
    FULL_DRYAD_DATASET_GB,
    REQUIRED_CORE_DOWNLOAD_GB,
    check_core_files,
    download_file,
    write_download_manifest,
)
from nichefate.io import ensure_dirs, load_config, paths_from_config
from nichefate.utils import setup_file_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m0_merfish_colitis.yaml")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _manual_instructions(config: dict[str, object]) -> str:
    raw_dir = paths_from_config(config)["raw_dir"]
    files = config["download"]["files"]
    lines = [
        "Manual download instructions:",
        f"Place files under: {raw_dir}",
    ]
    for filename, url in files.items():
        lines.append(f"- {filename}: {url}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dirs = ensure_dirs(config)
    log_path = dirs["logs"] / "download_core_files.log"
    manifest_path = dirs["reports"] / "download_manifest.csv"
    logger = setup_file_logger("nichefate.download", log_path)

    if args.no_download:
        rows = check_core_files(config)
        write_download_manifest(config, manifest_path, rows=rows)
        logger.info("Wrote dry-run manifest to %s", manifest_path)
        print(f"Download dry-run complete. Manifest: {manifest_path}")
        print(
            "Expected required core download: "
            f"about {REQUIRED_CORE_DOWNLOAD_GB:.1f} GB "
            "(adata.h5ad 17.96 GB + adata_day35.h5ad 1.51 GB, "
            "plus tiny README/LR files)."
        )
        print(
            "Full Dryad dataset warning: "
            f"{FULL_DRYAD_DATASET_GB:.2f} GB; do not download it for M0."
        )
        for row in rows:
            print(
                f"{row['status']}\t{row['filename']}\t"
                f"{row['size_bytes']}/{row['minimum_bytes']} bytes"
            )
        return 0

    raw_dir = paths_from_config(config)["raw_dir"]
    urls = config["download"]["files"]
    rows = []
    failed = False
    for filename, url in urls.items():
        result = download_file(str(url), raw_dir / filename, force=args.force)
        logger.info("%s %s", filename, result)
        rows.append(result)
        if result.get("status") == "failed":
            failed = True

    status_rows = check_core_files(config)
    status_by_name = {row["filename"]: row for row in status_rows}
    for row in rows:
        row.update(status_by_name.get(row["filename"], {}))
    write_download_manifest(config, manifest_path, rows=rows)
    print(f"Manifest: {manifest_path}")
    print(f"Log: {log_path}")

    if failed:
        print("One or more direct downloads failed. No blind retries were attempted.")
        print(_manual_instructions(config))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
