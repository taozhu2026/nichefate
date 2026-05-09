"""Project metadata and M0 metadata standardization."""

from __future__ import annotations

from typing import Any

import pandas as pd

PROJECT_NAME = "nichefate"
DATASET_ID = "merfish_colitis_moffitt_2024"
MILESTONE = "m0"

REQUIRED_RAW_FILES = (
    "adata.h5ad",
    "adata_day35.h5ad",
    "README.md",
)

OPTIONAL_RAW_FILES = (
    "ligand_receptor_pair_masterlist.csv",
)

AVOID_DRYAD_FILES_FOR_M0 = (
    "X.csv",
    "X_raw.csv",
    "transcript-level RNA metadata CSVs",
    "full supplementary CSV sets",
)

M0_OUTPUT_SUBDIRS = (
    "processed",
    "by_time",
    "by_slice",
    "graphs",
    "reports",
    "logs",
)


def build_time_mapping(config: dict[str, Any]) -> dict[str, int]:
    """Build the Sample_type to day mapping from config."""

    mapping = config.get("metadata", {}).get("time_map", {})
    if not isinstance(mapping, dict):
        raise ValueError("Config metadata.time_map must be a mapping.")
    return {str(key): int(value) for key, value in mapping.items()}


def ensure_day35_time_fallback(adata: Any) -> None:
    """Fill missing day-35 sample labels for dedicated day35 inputs."""

    obs = adata.obs
    dataset_part = obs.get("dataset_part")
    source_file = obs.get("source_file")
    is_day35 = pd.Series(False, index=obs.index)
    if dataset_part is not None:
        is_day35 = is_day35 | dataset_part.astype(str).str.lower().str.contains("day35")
    if source_file is not None:
        is_day35 = is_day35 | source_file.astype(str).str.contains("day35", case=False)

    if "Sample_type" not in obs:
        obs["Sample_type"] = pd.Series(pd.NA, index=obs.index, dtype="object")
    missing_day35 = is_day35 & obs["Sample_type"].isna()
    if missing_day35.any():
        obs["Sample_type"] = obs["Sample_type"].astype("object")
        obs.loc[missing_day35, "Sample_type"] = "Day35"


def validate_required_fields(
    adata: Any,
    required_fields: list[str] | tuple[str, ...],
    dataset_name: str,
) -> None:
    """Raise a clear error if required obs fields are missing."""

    missing = [field for field in required_fields if field not in adata.obs.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required obs fields: {', '.join(missing)}"
        )


def standardize_colitis_metadata(
    adata: Any,
    source_file: str,
    dataset_part: str,
    config: dict[str, Any],
) -> Any:
    """Add standard M0 obs fields used by downstream nichefate modules."""

    obs = adata.obs
    obs["source_file"] = source_file
    obs["dataset_part"] = dataset_part
    ensure_day35_time_fallback(adata)

    required = config.get("metadata", {}).get("required_obs_fields", [])
    validate_required_fields(adata, required, source_file)

    time_map = build_time_mapping(config)
    sample_type = obs["Sample_type"].astype(str)
    obs["time_day"] = sample_type.map(time_map)
    if obs["time_day"].isna().any():
        missing_values = sorted(sample_type[obs["time_day"].isna()].unique().tolist())
        raise ValueError(
            f"{source_file} has unmapped Sample_type values: {missing_values}"
        )
    obs["time_day"] = obs["time_day"].astype(int)
    obs["time"] = "D" + obs["time_day"].astype(str)

    obs["mouse_id"] = obs["Mouse_ID"].astype(str)
    obs["slice_id"] = obs["Slice_ID"].astype(str)
    obs["fov_id"] = obs["FOV"].astype(str)
    obs["batch_id"] = (
        obs["Mouse_ID"].astype(str)
        + "_rep"
        + obs["Technical_repeat_number"].astype(str)
    )
    obs["cell_type_l1"] = obs["Tier1"].astype(str)
    obs["cell_type_l2"] = (
        obs["Tier2"].astype(str) if "Tier2" in obs.columns else "NA"
    )
    obs["cell_type_l3"] = obs["Tier3"].astype(str)
    obs["neighborhood_original"] = obs["Leiden_neigh"].astype(str)
    return adata
