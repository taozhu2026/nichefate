#!/usr/bin/env python
"""Build a bounded M1 niche prototype from M0 by-slice outputs."""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anndata as ad
import numpy as np
import pandas as pd

from nichefate.io import load_config
from nichefate.niche import (
    build_basic_niche_feature_table,
    compute_neighbor_index,
    write_neighbor_index_npz,
    write_niche_feature_table_parquet_or_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m1_niche_construction.yaml")
    return parser.parse_args()


def _paths(config: dict) -> dict[str, Path]:
    return {
        key: Path(value)
        for key, value in config["paths"].items()
        if isinstance(value, str)
    }


def _slice_record(path: Path) -> dict[str, object]:
    data = ad.read_h5ad(path, backed="r")
    try:
        time_label = str(data.obs["time"].iloc[0]) if "time" in data.obs else ""
        slice_id = str(data.obs["slice_id"].iloc[0]) if "slice_id" in data.obs else path.stem
        return {"path": path, "n_obs": int(data.n_obs), "time": time_label, "slice_id": slice_id}
    finally:
        if hasattr(data, "file"):
            data.file.close()


def select_mixed_timepoint_slices(slice_files: list[Path], max_slices: int) -> list[Path]:
    """Select small deterministic prototype slices across time points when possible."""

    records = sorted((_slice_record(path) for path in slice_files), key=lambda x: (x["n_obs"], x["path"].name))
    if max_slices <= 0 or not records:
        return []
    selected = [records[0]]
    while len(selected) < max_slices:
        selected_times = {record["time"] for record in selected}
        candidate = next((r for r in records if r not in selected and r["time"] not in selected_times), None)
        if candidate is None:
            candidate = next((r for r in records if r not in selected), None)
        if candidate is None:
            break
        selected.append(candidate)
    return [record["path"] for record in selected]


def _anchor_indices(n_obs: int, max_anchors: int, rng: np.random.Generator) -> np.ndarray:
    if n_obs <= max_anchors:
        return np.arange(n_obs, dtype=np.int64)
    return np.sort(rng.choice(n_obs, size=max_anchors, replace=False).astype(np.int64))


def main() -> int:
    args = parse_args()
    start = time.monotonic()
    config = load_config(args.config)
    paths = _paths(config)
    prototype_dir = paths["m1_output_dir"] / "prototype"
    reports_dir = paths["reports_dir"]
    logs_dir = paths["logs_dir"]
    for directory in (prototype_dir, reports_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    slice_files = sorted(paths["m0_by_slice_dir"].glob("*.m0.h5ad"))
    selected = select_mixed_timepoint_slices(slice_files, int(config["prototype"]["max_slices"]))
    rng = np.random.default_rng(int(config["prototype"]["random_seed"]))
    feature_tables = []
    neighbor_entries = []
    selected_rows = []
    day35_l2_notes = []

    for slice_path in selected:
        data = ad.read_h5ad(slice_path)
        try:
            anchors = _anchor_indices(
                int(data.n_obs), int(config["prototype"]["max_anchors_per_slice"]), rng
            )
            slice_id = str(data.obs["slice_id"].iloc[0])
            selected_rows.append(
                {
                    "slice_file": slice_path.name,
                    "slice_id": slice_id,
                    "time": str(data.obs["time"].iloc[0]),
                    "n_obs": int(data.n_obs),
                    "n_anchors": int(len(anchors)),
                }
            )
            if str(data.obs["time"].iloc[0]) == "D35" and "cell_type_l2" in data.obs:
                na_count = int((data.obs["cell_type_l2"].astype(str) == "NA").sum())
                day35_l2_notes.append(f"{slice_path.name}: cell_type_l2 NA {na_count}/{data.n_obs}")
            for scale in config["niche"]["scales"]:
                neighbor_index = compute_neighbor_index(
                    data,
                    scale,
                    anchor_indices=anchors,
                    include_anchor=bool(config["niche"]["include_anchor"]),
                )
                table = build_basic_niche_feature_table(
                    data,
                    neighbor_index,
                    scale=scale,
                    slice_file=slice_path.name,
                    cell_type_keys=list(config["input"]["cell_type_keys"]),
                    embedding_key=config["input"]["embedding_key"],
                    spatial_key=config["input"]["spatial_key"],
                    topology_graph_key=config["input"]["graph_key_topology"],
                )
                feature_tables.append(table)
                neighbor_entries.append(
                    {
                        "slice_id": slice_id,
                        "slice_file": slice_path.name,
                        "scale": scale,
                        "neighbor_index": neighbor_index,
                    }
                )
        finally:
            if hasattr(data, "file"):
                data.file.close()

    features = pd.concat(feature_tables, ignore_index=True) if feature_tables else pd.DataFrame()
    feature_path = write_niche_feature_table_parquet_or_csv(
        features, prototype_dir / "niche_features_prototype.csv"
    )
    neighbor_path = write_neighbor_index_npz(
        neighbor_entries, prototype_dir / "neighbor_index_prototype.npz"
    )
    elapsed = time.monotonic() - start
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report_path = reports_dir / "m1_prototype_report.md"
    ct_columns = [c for c in features.columns if c.startswith(("ct_l1__", "ct_l2__", "ct_l3__"))]
    avg_neighbors = (
        features.groupby("scale", observed=True)["n_neighbors"].mean().sort_index()
        if "scale" in features and "n_neighbors" in features
        else pd.Series(dtype=float)
    )
    lines = [
        "# M1 Niche Prototype Report",
        "",
        "This is a bounded prototype only. Full M1 niche construction has not started.",
        "",
        "## Selected Slices",
        "",
    ]
    for row in selected_rows:
        lines.append(
            f"- {row['slice_file']}: {row['n_anchors']} anchors from {row['n_obs']} cells "
            f"({row['time']}, {row['slice_id']})"
        )
    lines.extend(["", "## Outputs", ""])
    lines.append(f"- Feature table: `{feature_path}`")
    lines.append(f"- Feature table shape: {features.shape[0]} x {features.shape[1]}")
    lines.append(f"- Neighbor index: `{neighbor_path}`")
    lines.append(f"- Scales: {', '.join(config['niche']['scales'])}")
    lines.append("")
    lines.extend(["## Average Neighbors Per Scale", ""])
    for scale, value in avg_neighbors.items():
        lines.append(f"- {scale}: {value:.3f}")
    lines.extend(["", "## Cell-Type Features", ""])
    lines.append(f"- Composition columns: {len(ct_columns)}")
    lines.append("- Biological sanity checks should prioritize `cell_type_l1` and `cell_type_l3`.")
    if day35_l2_notes:
        lines.append("- D35 Tier2 fallback note: " + "; ".join(day35_l2_notes))
    lines.extend(["", "## Runtime", ""])
    lines.append(f"- Wall seconds: {elapsed:.3f}")
    lines.append(f"- Max RSS KB: {max_rss_kb}")
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote prototype feature table: {feature_path}")
    print(f"Wrote prototype neighbor index: {neighbor_path}")
    print(f"Wrote prototype report: {report_path}")
    print(f"WALL_SECONDS {elapsed:.3f}")
    print(f"MAX_RSS_KB {max_rss_kb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
