#!/usr/bin/env python
"""Build M0 spatial graphs."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from nichefate.graph import (
    build_delaunay_graph,
    build_knn_graph,
    build_radius_graph,
    compute_median_nn_distance,
    summarize_sparse_graph,
)
from nichefate.io import ensure_dirs, load_config, read_h5ad, write_h5ad_safely
from nichefate.utils import safe_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m0_merfish_colitis.yaml")
    parser.add_argument("--sample-slices", type=int, default=None)
    parser.add_argument("--max-cells-per-slice", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-failed", action="store_true")
    return parser.parse_args()


def expected_graph_names(spatial_cfg: dict[str, object]) -> list[str]:
    names = [f"radius_x{value}" for value in spatial_cfg["adaptive_radius_multipliers"]]
    names.extend(f"knn_k{value}" for value in spatial_cfg["knn_values"])
    if spatial_cfg.get("build_delaunay", True):
        names.append("delaunay")
    return names


def existing_slice_is_valid(path: Path, expected_graphs: list[str]) -> bool:
    if not path.is_file():
        return False
    try:
        adata = read_h5ad(path, backed="r")
        valid = all(name in adata.obsp for name in expected_graphs)
        if hasattr(adata, "file"):
            adata.file.close()
        return valid
    except Exception:
        return False


def failed_slices(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        line.strip().split("\t", 1)[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dirs = ensure_dirs(config)
    input_path = dirs["processed"] / "m0_all_colitis_merfish.embedded.h5ad"
    if not input_path.is_file():
        print(f"Missing input h5ad: {input_path}")
        return 0

    adata = read_h5ad(input_path)
    if "slice_id" not in adata.obs or "X_spatial_norm" not in adata.obsm:
        print("Input is missing obs['slice_id'] or obsm['X_spatial_norm'].")
        return 1

    spatial_cfg = config["spatial"]
    min_cells = int(spatial_cfg["min_cells_per_slice_for_graph"])
    expected_graphs = expected_graph_names(spatial_cfg)
    slice_ids = sorted(adata.obs["slice_id"].astype(str).unique())
    failed_path = dirs["logs"] / "failed_slices.txt"
    if args.only_failed:
        requested_failed = failed_slices(failed_path)
        slice_ids = [slice_id for slice_id in slice_ids if slice_id in requested_failed]
    if args.sample_slices is not None:
        slice_ids = slice_ids[: args.sample_slices]

    rows = []
    failures = []
    for slice_id in slice_ids:
        target = dirs["by_slice"] / f"{safe_filename(slice_id)}.m0.h5ad"
        slice_log = dirs["logs"] / f"graph_{safe_filename(slice_id)}.log"
        if not args.force and existing_slice_is_valid(target, expected_graphs):
            print(f"Skipping existing valid slice: {slice_id}", flush=True)
            continue
        try:
            slice_log.write_text(f"start\t{slice_id}\n", encoding="utf-8")
            mask = adata.obs["slice_id"].astype(str).to_numpy() == slice_id
            sub = adata[mask].copy()
            if args.max_cells_per_slice is not None and sub.n_obs > args.max_cells_per_slice:
                sub = sub[: args.max_cells_per_slice].copy()
            if sub.n_obs < min_cells:
                slice_log.write_text(
                    f"skipped_small\t{slice_id}\t{sub.n_obs}\n",
                    encoding="utf-8",
                )
                continue
            coords = sub.obsm["X_spatial_norm"]
            median_nn = compute_median_nn_distance(coords)
            for multiplier in spatial_cfg["adaptive_radius_multipliers"]:
                graph_name = f"radius_x{multiplier}"
                matrix = build_radius_graph(coords, radius=median_nn * float(multiplier))
                sub.obsp[graph_name] = matrix
                summary = summarize_sparse_graph(matrix, graph_name, slice_id)
                summary["median_nn_distance"] = median_nn
                rows.append(summary)
            for k in spatial_cfg["knn_values"]:
                graph_name = f"knn_k{k}"
                matrix = build_knn_graph(coords, int(k))
                sub.obsp[graph_name] = matrix
                summary = summarize_sparse_graph(matrix, graph_name, slice_id)
                summary["median_nn_distance"] = median_nn
                rows.append(summary)
            if spatial_cfg.get("build_delaunay", True):
                graph_name = "delaunay"
                matrix = build_delaunay_graph(coords)
                sub.obsp[graph_name] = matrix
                summary = summarize_sparse_graph(matrix, graph_name, slice_id)
                summary["median_nn_distance"] = median_nn
                rows.append(summary)
            write_h5ad_safely(sub, target)
            slice_log.write_text(f"ok\t{slice_id}\t{target}\n", encoding="utf-8")
        except Exception as exc:
            message = f"{slice_id}\t{type(exc).__name__}: {exc}"
            failures.append(message)
            slice_log.write_text(
                message + "\n" + traceback.format_exc(),
                encoding="utf-8",
            )

    summary_path = dirs["reports"] / "graph_degree_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    failed_path.write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")
    print(f"Wrote graph summary: {summary_path}")
    if failures:
        print(f"Graph construction failed for {len(failures)} slices: {failed_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
