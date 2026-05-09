#!/usr/bin/env python
"""Build the canonical M0 AnnData object."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from nichefate.io import ensure_dirs, load_config, paths_from_config, read_h5ad, write_h5ad_safely
from nichefate.metadata import standardize_colitis_metadata
from nichefate.spatial import normalize_spatial_by_slice, set_spatial_obsm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m0_merfish_colitis.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-cells", type=int, default=None)
    return parser.parse_args()


def _allocate_proportional_counts(
    counts: dict[tuple[str, str], int],
    total: int,
) -> dict[tuple[str, str], int]:
    """Allocate a global sample budget across file/time strata."""

    available = sum(counts.values())
    if total >= available:
        return dict(counts)

    strata = [key for key, value in counts.items() if value > 0]
    if total < len(strata):
        raise ValueError(
            f"sample-cells={total} is too small to cover {len(strata)} strata."
        )

    exact = {key: counts[key] * total / available for key in strata}
    allocations = {key: min(counts[key], max(1, int(np.floor(exact[key])))) for key in strata}

    while sum(allocations.values()) > total:
        removable = [key for key in strata if allocations[key] > 1]
        key = min(removable, key=lambda item: exact[item] - np.floor(exact[item]))
        allocations[key] -= 1

    while sum(allocations.values()) < total:
        candidates = [key for key in strata if allocations[key] < counts[key]]
        key = max(candidates, key=lambda item: exact[item] - np.floor(exact[item]))
        allocations[key] += 1

    return allocations


def _collect_sample_type_counts(
    raw_dir: Path,
    file_specs: list[tuple[str, str]],
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for filename, dataset_part in file_specs:
        backed = read_h5ad(raw_dir / filename, backed="r")
        try:
            if "Sample_type" in backed.obs:
                labels = backed.obs["Sample_type"].astype(str)
            else:
                labels = pd.Series(dataset_part, index=backed.obs_names)
            for sample_type, value in labels.value_counts(sort=False).items():
                counts[(filename, str(sample_type))] = int(value)
        finally:
            if hasattr(backed, "file"):
                backed.file.close()
    return counts


def _read_h5ad_for_build(
    path: Path,
    filename: str,
    dataset_part: str,
    allocations: dict[tuple[str, str], int] | None,
):
    if allocations is None:
        return read_h5ad(path)

    backed = read_h5ad(path, backed="r")
    try:
        if "Sample_type" in backed.obs:
            labels = backed.obs["Sample_type"].astype(str)
        else:
            labels = pd.Series(dataset_part, index=backed.obs_names)

        rng = np.random.default_rng(0)
        selected: list[np.ndarray] = []
        label_values = labels.to_numpy()
        for sample_type in sorted(labels.unique()):
            requested = allocations.get((filename, str(sample_type)), 0)
            if requested == 0:
                continue
            positions = np.flatnonzero(label_values == str(sample_type))
            selected.append(rng.choice(positions, size=requested, replace=False))

        if not selected:
            raise ValueError(f"No cells selected from {filename}.")

        indices = np.sort(np.concatenate(selected))
        return backed[indices].to_memory()
    finally:
        if hasattr(backed, "file"):
            backed.file.close()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = paths_from_config(config)
    dirs = ensure_dirs(config)
    raw_dir = paths["raw_dir"]
    output_path = dirs["processed"] / "m0_all_colitis_merfish.metadata_spatial.h5ad"
    norm_csv = dirs["reports"] / "spatial_normalization_params.csv"

    raw_files = config["raw_files"]
    file_specs = [
        (raw_files["adata_main"], "day0_21"),
        (raw_files["adata_day35"], "day35"),
    ]
    missing = [filename for filename, _part in file_specs if not (raw_dir / filename).is_file()]
    if missing:
        print(f"Missing raw h5ad files under {raw_dir}: {', '.join(missing)}")
        return 0
    if args.dry_run:
        print(f"Dry run: would write {output_path}")
        return 0

    import anndata as ad

    allocations = None
    if args.sample_cells is not None:
        counts = _collect_sample_type_counts(raw_dir, file_specs)
        allocations = _allocate_proportional_counts(counts, args.sample_cells)
        print("Global stratified sample allocation:", flush=True)
        for (filename, sample_type), count in sorted(allocations.items()):
            print(f"  {filename}\t{sample_type}\t{count}", flush=True)

    adatas = []
    for filename, dataset_part in file_specs:
        adata = _read_h5ad_for_build(
            raw_dir / filename,
            filename,
            dataset_part,
            allocations,
        )
        standardize_colitis_metadata(adata, filename, dataset_part, config)
        if config["preprocessing"]["preserve_input_X"]:
            adata.layers["lognorm_original"] = adata.X.copy()
        adatas.append(adata)

    combined = ad.concat(
        adatas,
        join="inner",
        merge="same",
        label="dataset_part_concat",
        keys=[part for _filename, part in file_specs],
        index_unique="-",
    )
    set_spatial_obsm(combined, *config["spatial"]["coordinate_fields"])
    params = normalize_spatial_by_slice(combined, slice_key="slice_id")
    params.to_csv(norm_csv, index=False)
    write_h5ad_safely(combined, output_path)
    print(f"Wrote M0 metadata/spatial AnnData: {output_path}")
    print(f"Wrote spatial normalization params: {norm_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
