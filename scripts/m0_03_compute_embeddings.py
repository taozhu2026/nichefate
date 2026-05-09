#!/usr/bin/env python
"""Compute or validate M0 embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from nichefate.embedding import compute_pca_m0
from nichefate.io import ensure_dirs, load_config, paths_from_config, read_h5ad, write_h5ad_safely


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m0_merfish_colitis.yaml")
    parser.add_argument("--sample-cells", type=int, default=None)
    return parser.parse_args()


def _sample_adata(adata, n_cells: int | None):
    if n_cells is None or adata.n_obs <= n_cells:
        return adata
    rng = np.random.default_rng(0)
    indices = np.sort(rng.choice(adata.n_obs, size=n_cells, replace=False))
    return adata[indices].copy()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dirs = ensure_dirs(config)
    input_path = dirs["processed"] / "m0_all_colitis_merfish.metadata_spatial.h5ad"
    output_path = dirs["processed"] / "m0_all_colitis_merfish.embedded.h5ad"
    variance_csv = dirs["reports"] / "pca_variance_ratio.csv"
    if not input_path.is_file():
        print(f"Missing input h5ad: {input_path}")
        return 0
    adata = read_h5ad(input_path)
    adata = _sample_adata(adata, args.sample_cells)
    compute_pca_m0(
        adata,
        n_comps=config["preprocessing"]["pca_n_comps"],
        scale=config["preprocessing"]["scale_before_pca"],
        variance_csv=variance_csv,
    )
    write_h5ad_safely(adata, output_path)
    print(f"Wrote embedded AnnData: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
