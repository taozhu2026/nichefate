#!/usr/bin/env python
"""Export M0 objects for downstream nichefate development."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.export import export_by_time, write_sample_tables
from nichefate.io import ensure_dirs, load_config, read_h5ad, write_h5ad_safely


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m0_merfish_colitis.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dirs = ensure_dirs(config)
    input_path = dirs["processed"] / "m0_all_colitis_merfish.embedded.h5ad"
    final_path = dirs["processed"] / "m0_all_colitis_merfish.final.h5ad"
    if not input_path.is_file():
        print(f"Missing input h5ad: {input_path}")
        return 0
    adata = read_h5ad(input_path)
    written = export_by_time(adata, dirs["by_time"])
    write_sample_tables(adata, dirs["reports"])
    write_h5ad_safely(adata, final_path)
    print(f"Wrote {len(written)} by-time files and final object: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
