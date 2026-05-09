#!/usr/bin/env python
"""Prepare a global M1 feature schema from all by-slice M0 objects."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anndata as ad
import pandas as pd

from nichefate.io import load_config
from nichefate.niche import cell_type_composition_prefix, safe_feature_token


METADATA_COLUMNS = [
    "slice_id",
    "slice_file",
    "scale",
    "anchor_index",
    "anchor_cell_id",
    "time",
    "time_day",
    "mouse_id",
    "cell_type_l1",
    "cell_type_l2",
    "cell_type_l3",
    "x",
    "y",
]
SPATIAL_SUMMARY_COLUMNS = [
    "n_neighbors",
    "mean_neighbor_distance",
    "pseudo_local_density",
]
TOPOLOGY_COLUMNS = [
    "local_topology_anchor_degree",
    "local_topology_mean_member_degree",
]


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


def _embedding_columns(n_dims: int) -> tuple[list[str], list[str]]:
    means = [f"emb_mean_pc{idx:03d}" for idx in range(1, n_dims + 1)]
    variances = [f"emb_var_pc{idx:03d}" for idx in range(1, n_dims + 1)]
    return means, variances


def _record_values(
    vocab: dict[str, dict[str, dict[str, object]]],
    key: str,
    values: pd.Series,
    slice_id: str,
) -> None:
    level = vocab.setdefault(key, {})
    for label, count in values.astype(str).value_counts(dropna=False).items():
        token = safe_feature_token(label)
        record = level.setdefault(
            token,
            {
                "cell_type_key": key,
                "label": str(label),
                "token": token,
                "column": f"{cell_type_composition_prefix(key)}__{token}",
                "n_cells": 0,
                "n_slices": 0,
                "slice_ids": [],
            },
        )
        record["n_cells"] = int(record["n_cells"]) + int(count)
        if slice_id not in record["slice_ids"]:
            record["slice_ids"].append(slice_id)
            record["n_slices"] = int(record["n_slices"]) + 1


def _write_markdown(
    path: Path,
    schema: dict[str, object],
    vocabulary: pd.DataFrame,
    elapsed: float,
    max_rss_kb: int,
) -> None:
    lines = [
        "# M1 Global Feature Schema",
        "",
        "This schema aligns per-slice M1 feature tables before full by-slice execution.",
        "",
        "## Inputs",
        "",
        f"- Slice files: {schema['n_slices']}",
        f"- Total cells: {schema['total_cells']}",
        f"- Scales: {', '.join(schema['scales'])}",
        f"- Embedding dimensions: {schema['embedding_dimensions']}",
        "",
        "## Cell-Type Vocabulary Sizes",
        "",
        "| cell_type_key | labels | composition_columns |",
        "| --- | --- | --- |",
    ]
    for key in schema["cell_type_keys"]:
        count = len(schema["composition_columns_by_key"][key])
        lines.append(f"| {key} | {count} | {count} |")
    lines.extend(
        [
            "",
            "## Feature Columns",
            "",
            f"- Metadata columns: {len(schema['metadata_columns'])}",
            f"- Composition columns: {len(schema['composition_columns'])}",
            f"- Entropy columns: {len(schema['entropy_columns'])}",
            f"- Embedding summary columns: {len(schema['embedding_columns'])}",
            f"- Spatial summary columns: {len(schema['spatial_summary_columns'])}",
            f"- Topology columns: {len(schema['topology_columns'])}",
            f"- Total expected feature columns: {len(schema['feature_columns'])}",
            "",
            "## Top Vocabulary Entries",
            "",
            "| cell_type_key | label | token | column | n_cells | n_slices |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in vocabulary.head(30).iterrows():
        lines.append(
            f"| {row['cell_type_key']} | {row['label']} | {row['token']} | "
            f"{row['column']} | {row['n_cells']} | {row['n_slices']} |"
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Wall seconds: {elapsed:.3f}",
            f"- Max RSS KB: {max_rss_kb}",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    start = time.monotonic()
    args = parse_args()
    config = load_config(args.config)
    paths = _paths(config)
    reports_dir = paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    slice_files = sorted(paths["m0_by_slice_dir"].glob("*.m0.h5ad"))
    if not slice_files:
        raise FileNotFoundError(f"No by-slice M0 files found in {paths['m0_by_slice_dir']}")

    cell_type_keys = list(config["input"]["cell_type_keys"])
    vocab: dict[str, dict[str, dict[str, object]]] = {}
    slice_rows = []
    embedding_dimensions: int | None = None

    for path in slice_files:
        data = ad.read_h5ad(path, backed="r")
        try:
            slice_id = str(data.obs["slice_id"].iloc[0]) if "slice_id" in data.obs else path.stem
            time_label = str(data.obs["time"].iloc[0]) if "time" in data.obs else ""
            slice_rows.append(
                {
                    "slice_file": path.name,
                    "slice_id": slice_id,
                    "time": time_label,
                    "n_obs": int(data.n_obs),
                }
            )
            if embedding_dimensions is None:
                embedding_dimensions = int(data.obsm[config["input"]["embedding_key"]].shape[1])
            for key in cell_type_keys:
                if key not in data.obs:
                    raise KeyError(f"{path.name} is missing obs field {key}")
                _record_values(vocab, key, data.obs[key], slice_id)
        finally:
            if hasattr(data, "file"):
                data.file.close()

    if embedding_dimensions is None:
        raise ValueError("Could not infer embedding dimensions from by-slice files.")

    vocabulary_rows = []
    composition_columns_by_key = {}
    for key in cell_type_keys:
        records = sorted(vocab.get(key, {}).values(), key=lambda row: str(row["token"]))
        for record in records:
            row = {
                "cell_type_key": record["cell_type_key"],
                "label": record["label"],
                "token": record["token"],
                "column": record["column"],
                "n_cells": record["n_cells"],
                "n_slices": record["n_slices"],
                "slice_ids": ";".join(sorted(record["slice_ids"])),
            }
            vocabulary_rows.append(row)
        composition_columns_by_key[key] = [str(record["column"]) for record in records]

    vocabulary = pd.DataFrame(vocabulary_rows).sort_values(
        ["cell_type_key", "token"], ignore_index=True
    )
    mean_columns, var_columns = _embedding_columns(embedding_dimensions)
    entropy_columns = [
        f"{cell_type_composition_prefix(key)}_entropy" for key in cell_type_keys
    ]
    composition_columns = [
        column
        for key in cell_type_keys
        for column in composition_columns_by_key[key]
    ]
    feature_columns = (
        METADATA_COLUMNS
        + composition_columns_by_key.get("cell_type_l1", [])
        + [f"{cell_type_composition_prefix('cell_type_l1')}_entropy"]
        + composition_columns_by_key.get("cell_type_l2", [])
        + [f"{cell_type_composition_prefix('cell_type_l2')}_entropy"]
        + composition_columns_by_key.get("cell_type_l3", [])
        + [f"{cell_type_composition_prefix('cell_type_l3')}_entropy"]
        + mean_columns
        + var_columns
        + SPATIAL_SUMMARY_COLUMNS
        + TOPOLOGY_COLUMNS
    )
    schema = {
        "schema_version": 1,
        "created_by": Path(__file__).name,
        "n_slices": len(slice_rows),
        "total_cells": int(sum(row["n_obs"] for row in slice_rows)),
        "scales": list(config["niche"]["scales"]),
        "cell_type_keys": cell_type_keys,
        "embedding_key": config["input"]["embedding_key"],
        "embedding_dimensions": embedding_dimensions,
        "spatial_key": config["input"]["spatial_key"],
        "topology_graph_key": config["input"]["graph_key_topology"],
        "metadata_columns": METADATA_COLUMNS,
        "composition_columns_by_key": composition_columns_by_key,
        "composition_columns": composition_columns,
        "entropy_columns": entropy_columns,
        "embedding_columns": mean_columns + var_columns,
        "spatial_summary_columns": SPATIAL_SUMMARY_COLUMNS,
        "topology_columns": TOPOLOGY_COLUMNS,
        "feature_columns": feature_columns,
        "slices": slice_rows,
    }

    json_path = reports_dir / "m1_global_schema.json"
    vocab_path = reports_dir / "m1_global_celltype_vocabulary.csv"
    md_path = reports_dir / "m1_global_schema.md"
    json_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    vocabulary.to_csv(vocab_path, index=False)
    elapsed = time.monotonic() - start
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _write_markdown(md_path, schema, vocabulary, elapsed, max_rss_kb)

    print(f"Wrote global schema: {json_path}")
    print(f"Wrote vocabulary CSV: {vocab_path}")
    print(f"Wrote schema report: {md_path}")
    for key in cell_type_keys:
        print(f"VOCAB_SIZE {key} {len(composition_columns_by_key[key])}")
    print(f"FEATURE_COLUMNS {len(feature_columns)}")
    print(f"N_SLICES {len(slice_rows)}")
    print(f"TOTAL_CELLS {schema['total_cells']}")
    print(f"WALL_SECONDS {elapsed:.3f}")
    print(f"MAX_RSS_KB {max_rss_kb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
