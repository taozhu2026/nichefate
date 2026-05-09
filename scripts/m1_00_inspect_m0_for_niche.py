#!/usr/bin/env python
"""Audit M0 by-slice outputs for M1 niche construction."""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from scipy import sparse

from nichefate.io import load_config, read_h5ad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m1_niche_construction.yaml")
    return parser.parse_args()


def _paths(config: dict) -> dict[str, Path]:
    paths = config["paths"]
    return {key: Path(value) for key, value in paths.items() if isinstance(value, str)}


def _first_unique(adata, field: str) -> str:
    if field not in adata.obs:
        return ""
    values = sorted({str(value) for value in adata.obs[field].dropna().unique()})
    return ";".join(values)


def audit_slice(path: Path, config: dict) -> dict[str, object]:
    expected_graphs = (
        config["input"]["graph_keys_main"]
        + [config["input"]["graph_key_topology"]]
        + config["input"]["graph_keys_ablation"]
    )
    required_obs = [
        "slice_id",
        "time",
        "time_day",
        "mouse_id",
        "x",
        "y",
        *config["input"]["cell_type_keys"],
    ]
    required_obsm = [config["input"]["embedding_key"], config["input"]["spatial_key"]]
    adata = read_h5ad(path, backed="r")
    try:
        row: dict[str, object] = {
            "slice_file": path.name,
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "slice_id": _first_unique(adata, "slice_id"),
            "time": _first_unique(adata, "time"),
            "mouse_id": _first_unique(adata, "mouse_id"),
            "missing_obs": ";".join(field for field in required_obs if field not in adata.obs),
            "missing_obsm": ";".join(key for key in required_obsm if key not in adata.obsm),
            "available_obsm": ";".join(sorted(adata.obsm.keys())),
            "available_obsp": ";".join(sorted(adata.obsp.keys())),
        }
        for graph_key in expected_graphs:
            if graph_key not in adata.obsp:
                row[f"graph__{graph_key}__present"] = False
                row[f"graph__{graph_key}__type"] = ""
                row[f"graph__{graph_key}__shape"] = ""
                row[f"graph__{graph_key}__nnz"] = 0
                continue
            graph = adata.obsp[graph_key]
            row[f"graph__{graph_key}__present"] = True
            row[f"graph__{graph_key}__type"] = type(graph).__name__
            row[f"graph__{graph_key}__shape"] = f"{graph.shape[0]}x{graph.shape[1]}"
            row[f"graph__{graph_key}__nnz"] = int(graph.nnz) if sparse.issparse(graph) else -1
        return row
    finally:
        if hasattr(adata, "file"):
            adata.file.close()


def main() -> int:
    args = parse_args()
    start = time.monotonic()
    config = load_config(args.config)
    paths = _paths(config)
    reports_dir = paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    slice_files = sorted(paths["m0_by_slice_dir"].glob("*.m0.h5ad"))
    rows = [audit_slice(path, config) for path in slice_files]
    table = pd.DataFrame(rows)
    csv_path = reports_dir / "m1_m0_input_audit.csv"
    md_path = reports_dir / "m1_m0_input_audit.md"
    table.to_csv(csv_path, index=False)
    elapsed = time.monotonic() - start
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    missing_rows = table[(table["missing_obs"] != "") | (table["missing_obsm"] != "")]
    lines = [
        "# M1 M0 Input Audit",
        "",
        f"- Slice files: {len(table)}",
        f"- Total cells: {int(table['n_obs'].sum()) if len(table) else 0}",
        f"- Missing required rows: {len(missing_rows)}",
        f"- Wall seconds: {elapsed:.3f}",
        f"- Max RSS KB: {max_rss_kb}",
        "",
        "## Graph Keys",
        "",
        f"- Main scales: {', '.join(config['input']['graph_keys_main'])}",
        f"- Topology: {config['input']['graph_key_topology']}",
        f"- Ablation: {', '.join(config['input']['graph_keys_ablation'])}",
    ]
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote M1 M0 audit CSV: {csv_path}")
    print(f"Wrote M1 M0 audit report: {md_path}")
    print(f"WALL_SECONDS {elapsed:.3f}")
    print(f"MAX_RSS_KB {max_rss_kb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
