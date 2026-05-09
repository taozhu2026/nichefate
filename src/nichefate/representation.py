"""M2 niche representation helpers."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FEATURE_GROUPS = (
    "neighborhood_size",
    "cell_type_composition",
    "entropy",
    "molecular_state",
    "spatial_summary",
    "topology",
)


def expand_column_range(spec: str) -> list[str]:
    """Expand compact names like emb_mean_pc001..pc050."""

    match = re.fullmatch(r"(.+?)(\d+)\.\.(.+?)(\d+)", spec)
    if not match:
        return [spec]
    left_prefix, start_text, right_prefix, end_text = match.groups()
    if left_prefix != right_prefix and not left_prefix.endswith(right_prefix):
        raise ValueError(f"Range prefixes differ: {spec}")
    start = int(start_text)
    end = int(end_text)
    if end < start:
        raise ValueError(f"Range end is before start: {spec}")
    width = max(len(start_text), len(end_text))
    return [f"{left_prefix}{idx:0{width}d}" for idx in range(start, end + 1)]


def feature_group_columns(
    available_columns: Sequence[str],
    group_config: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Resolve configured feature groups against available columns."""

    available = list(available_columns)
    available_set = set(available)
    resolved: dict[str, list[str]] = {}
    for group_name, config in group_config.items():
        if not isinstance(config, Mapping):
            continue
        selected: list[str] = []
        for column in config.get("columns", []) or []:
            if column in available_set:
                selected.append(column)
        for range_spec in config.get("ranges", []) or []:
            for column in expand_column_range(str(range_spec)):
                if column in available_set:
                    selected.append(column)
        for pattern in config.get("patterns", []) or []:
            selected.extend(
                column for column in available if fnmatch.fnmatchcase(column, str(pattern))
            )
        resolved[group_name] = list(dict.fromkeys(selected))
    return resolved


def select_numeric_feature_columns(
    table: pd.DataFrame,
    group_config: Mapping[str, Any],
    group_names: Sequence[str] = DEFAULT_FEATURE_GROUPS,
) -> list[str]:
    """Return numeric feature columns from configured non-metadata groups."""

    grouped = feature_group_columns(table.columns, group_config)
    selected: list[str] = []
    for group_name in group_names:
        selected.extend(grouped.get(group_name, []))
    numeric_columns = set(table.select_dtypes(include=[np.number, "bool"]).columns)
    return [column for column in dict.fromkeys(selected) if column in numeric_columns]


def feature_columns_from_schema(
    available_columns: Sequence[str],
    group_config: Mapping[str, Any],
    group_names: Sequence[str] = DEFAULT_FEATURE_GROUPS,
) -> list[str]:
    """Return configured feature columns from an aligned M1 schema."""

    grouped = feature_group_columns(available_columns, group_config)
    selected: list[str] = []
    for group_name in group_names:
        selected.extend(grouped.get(group_name, []))
    return list(dict.fromkeys(selected))


def scale_prefixed_feature_columns(
    feature_columns: Sequence[str],
    expected_scales: Sequence[str],
    separator: str = "__",
) -> list[str]:
    """Return deterministic scale-prefixed M2 feature columns."""

    return [
        f"{scale}{separator}{column}"
        for scale in expected_scales
        for column in feature_columns
    ]


def m2_output_columns(
    metadata_columns: Sequence[str],
    feature_columns: Sequence[str],
    expected_scales: Sequence[str],
    separator: str = "__",
) -> list[str]:
    """Return deterministic M2 output columns."""

    return list(dict.fromkeys(metadata_columns)) + scale_prefixed_feature_columns(
        feature_columns,
        expected_scales,
        separator,
    )


def validate_aligned_schema(table: pd.DataFrame, expected_columns: Sequence[str]) -> None:
    """Require exact M1 schema alignment before M2 reshaping."""

    actual = list(table.columns)
    expected = list(expected_columns)
    if actual != expected:
        missing = [column for column in expected if column not in actual]
        extra = [column for column in actual if column not in expected]
        raise ValueError(
            "Feature table does not match the aligned M1 schema "
            f"(missing={missing[:5]}, extra={extra[:5]})."
        )


def validate_complete_scales(
    table: pd.DataFrame,
    expected_scales: Sequence[str],
    anchor_keys: Sequence[str],
    scale_column: str = "scale",
) -> None:
    """Require one row for every expected scale per anchor."""

    keys = list(anchor_keys) + [scale_column]
    duplicate_count = int(table.duplicated(keys).sum())
    if duplicate_count:
        raise ValueError(f"Duplicate anchor-scale rows detected: {duplicate_count}")

    expected = set(expected_scales)
    observed = set(table[scale_column].dropna().astype(str).unique())
    unexpected = sorted(observed - expected)
    if unexpected:
        raise ValueError(f"Unexpected scales detected: {unexpected}")

    scale_counts = table.groupby(list(anchor_keys), observed=True)[scale_column].nunique()
    if not bool((scale_counts == len(expected)).all()):
        bad_count = int((scale_counts != len(expected)).sum())
        raise ValueError(f"Missing scale rows for {bad_count} anchors.")


def pivot_scale_features(
    table: pd.DataFrame,
    feature_columns: Sequence[str],
    expected_scales: Sequence[str],
    metadata_columns: Sequence[str],
    anchor_keys: Sequence[str],
    scale_column: str = "scale",
    separator: str = "__",
) -> pd.DataFrame:
    """Pivot M1 scale rows into one M2 row per anchor."""

    validate_complete_scales(table, expected_scales, anchor_keys, scale_column)
    anchor_keys = list(anchor_keys)
    feature_columns = list(feature_columns)
    metadata_columns = list(dict.fromkeys(metadata_columns))
    missing = [
        column
        for column in [*anchor_keys, scale_column, *feature_columns, *metadata_columns]
        if column not in table.columns
    ]
    if missing:
        raise KeyError(f"Missing required columns for scale pivot: {missing}")

    metadata = (
        table.sort_values([*anchor_keys, scale_column])
        .drop_duplicates(anchor_keys)[metadata_columns]
        .set_index(anchor_keys)
    )
    indexed = table.set_index([*anchor_keys, scale_column])
    blocks = []
    for scale in expected_scales:
        block = indexed.xs(scale, level=scale_column)[feature_columns].copy()
        block.columns = [f"{scale}{separator}{column}" for column in feature_columns]
        blocks.append(block)
    pivoted = pd.concat([metadata, *blocks], axis=1).reset_index()
    return pivoted


def build_m2_representation_table(
    table: pd.DataFrame,
    feature_columns: Sequence[str],
    expected_scales: Sequence[str],
    metadata_columns: Sequence[str],
    anchor_keys: Sequence[str],
    scale_column: str = "scale",
    separator: str = "__",
) -> pd.DataFrame:
    """Build an aligned M2 anchor-level representation table from M1 rows."""

    expected_columns = m2_output_columns(
        metadata_columns,
        feature_columns,
        expected_scales,
        separator,
    )
    matrix = pivot_scale_features(
        table,
        feature_columns=feature_columns,
        expected_scales=expected_scales,
        metadata_columns=metadata_columns,
        anchor_keys=anchor_keys,
        scale_column=scale_column,
        separator=separator,
    )
    missing = [column for column in expected_columns if column not in matrix.columns]
    if missing:
        raise KeyError(f"Missing M2 output columns: {missing}")
    return matrix[expected_columns]


def validate_m2_output_table(
    table: pd.DataFrame,
    expected_columns: Sequence[str],
    expected_rows: int | None = None,
    numeric_columns: Sequence[str] | None = None,
) -> dict[str, int | bool]:
    """Validate shape, schema, and finite numeric values for an M2 table."""

    if list(table.columns) != list(expected_columns):
        raise ValueError("M2 output columns do not match the expected schema.")
    if expected_rows is not None and len(table) != expected_rows:
        raise ValueError(f"M2 output rows {len(table)} != expected {expected_rows}.")
    checked = table[list(numeric_columns)] if numeric_columns is not None else table
    summary = finite_value_summary(checked)
    if summary["missing_values"] or summary["infinite_values"]:
        raise ValueError(
            "M2 output contains non-finite values "
            f"(missing={summary['missing_values']}, "
            f"infinite={summary['infinite_values']})."
        )
    return {
        "ok": True,
        "rows": int(len(table)),
        "missing_values": summary["missing_values"],
        "infinite_values": summary["infinite_values"],
    }


def finite_value_summary(table: pd.DataFrame) -> dict[str, int]:
    """Return simple missing and infinite counts for a matrix-like table."""

    numeric = table.select_dtypes(include=[np.number, "bool"])
    infinite = 0
    if not numeric.empty:
        infinite = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    return {
        "missing_values": int(table.isna().sum().sum()),
        "infinite_values": infinite,
    }
