"""QC helpers for M1 niche feature tables and neighbor indices."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


COMPOSITION_LEVELS = ("cell_type_l1", "cell_type_l2", "cell_type_l3")
COMPOSITION_PREFIXES = {
    "cell_type_l1": "ct_l1__",
    "cell_type_l2": "ct_l2__",
    "cell_type_l3": "ct_l3__",
}


def feature_group(column: str) -> str:
    """Classify a niche feature-table column into a broad QC group."""

    if column in {
        "slice_id",
        "slice_file",
        "scale",
        "anchor_index",
        "anchor_cell_id",
        "time",
        "time_day",
        "time_order",
        "sample_id",
        "mouse_id",
    }:
        return "identity"
    if column in {"x", "y"}:
        return "anchor_spatial"
    if column in COMPOSITION_LEVELS:
        return "anchor_cell_type"
    for level, prefix in COMPOSITION_PREFIXES.items():
        if column.startswith(prefix):
            return f"composition_{level[-2:]}"
    if column in {"ct_l1_entropy", "ct_l2_entropy", "ct_l3_entropy"}:
        return f"entropy_{column[3:5]}"
    if column.startswith("emb_mean_"):
        return "embedding_mean"
    if column.startswith("emb_var_"):
        return "embedding_variance"
    if column in {"n_neighbors", "mean_neighbor_distance", "pseudo_local_density"}:
        return "spatial_summary"
    if column.startswith("local_topology_"):
        return "topology_summary"
    return "other"


def summarize_feature_integrity(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing and infinite values by feature group."""

    rows = []
    for group in sorted({feature_group(column) for column in table.columns}):
        columns = [column for column in table.columns if feature_group(column) == group]
        missing_columns = []
        infinite_columns = []
        missing_values = 0
        infinite_values = 0
        for column in columns:
            series = table[column]
            column_missing = int(series.isna().sum())
            missing_values += column_missing
            if column_missing:
                missing_columns.append(column)
            numeric = pd.to_numeric(series, errors="coerce").astype(float)
            column_inf = int(np.isinf(numeric.to_numpy()).sum())
            infinite_values += column_inf
            if column_inf:
                infinite_columns.append(column)
        total_values = int(len(table) * len(columns))
        rows.append(
            {
                "feature_group": group,
                "n_columns": len(columns),
                "n_rows": len(table),
                "total_values": total_values,
                "missing_values": missing_values,
                "missing_fraction": missing_values / total_values if total_values else 0.0,
                "infinite_values": infinite_values,
                "infinite_fraction": infinite_values / total_values if total_values else 0.0,
                "columns_with_missing": ",".join(missing_columns),
                "columns_with_infinite": ",".join(infinite_columns),
            }
        )
    return pd.DataFrame(rows)


def summarize_distribution(
    table: pd.DataFrame,
    value_column: str,
    by: Sequence[str] = ("scale",),
    quantiles: Sequence[float] = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0),
) -> pd.DataFrame:
    """Return compact distribution statistics for a numeric feature column."""

    group_columns = [column for column in by if column in table.columns]
    output_columns = list(group_columns) + [
        "value_column",
        "n_rows",
        "finite_count",
        "missing_count",
        "infinite_count",
        "mean",
        "std",
    ] + [f"q{int(q * 100):03d}" for q in quantiles]
    if value_column not in table.columns:
        return pd.DataFrame(columns=output_columns)

    values = pd.to_numeric(table[value_column], errors="coerce").astype(float)
    frame = table[group_columns].copy() if group_columns else pd.DataFrame(index=table.index)
    frame["__value"] = values

    rows = []
    grouped = (
        frame.groupby(group_columns, dropna=False, observed=True)
        if group_columns
        else [((), frame)]
    )
    for keys, group in grouped:
        if group_columns and not isinstance(keys, tuple):
            keys = (keys,)
        array = group["__value"].to_numpy(dtype=float)
        finite = array[np.isfinite(array)]
        row = {column: key for column, key in zip(group_columns, keys, strict=False)}
        row.update(
            {
                "value_column": value_column,
                "n_rows": int(len(array)),
                "finite_count": int(len(finite)),
                "missing_count": int(np.isnan(array).sum()),
                "infinite_count": int(np.isinf(array).sum()),
                "mean": float(np.mean(finite)) if len(finite) else np.nan,
                "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            }
        )
        for quantile in quantiles:
            row[f"q{int(quantile * 100):03d}"] = (
                float(np.quantile(finite, quantile)) if len(finite) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


def _as_columns(table_or_columns: pd.DataFrame | Iterable[str]) -> list[str]:
    if isinstance(table_or_columns, pd.DataFrame):
        return list(table_or_columns.columns)
    return list(table_or_columns)


def _canonical_level(level: str) -> str:
    aliases = {
        "l1": "cell_type_l1",
        "ct_l1": "cell_type_l1",
        "cell_type_l1": "cell_type_l1",
        "l2": "cell_type_l2",
        "ct_l2": "cell_type_l2",
        "cell_type_l2": "cell_type_l2",
        "l3": "cell_type_l3",
        "ct_l3": "cell_type_l3",
        "cell_type_l3": "cell_type_l3",
    }
    if level not in aliases:
        raise ValueError(f"Unknown composition level: {level}")
    return aliases[level]


def composition_columns(
    table_or_columns: pd.DataFrame | Iterable[str],
    level: str | None = None,
) -> list[str]:
    """Discover normalized cell-type composition columns."""

    columns = _as_columns(table_or_columns)
    if level is not None:
        prefix = COMPOSITION_PREFIXES[_canonical_level(level)]
        return [column for column in columns if column.startswith(prefix)]
    prefixes = tuple(COMPOSITION_PREFIXES.values())
    return [column for column in columns if column.startswith(prefixes)]


def composition_sum_qc(
    table: pd.DataFrame,
    level: str | None = None,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Check row-wise composition sums for one or all standard cell-type levels."""

    levels = [_canonical_level(level)] if level else list(COMPOSITION_LEVELS)
    rows = []
    for current_level in levels:
        columns = composition_columns(table, current_level)
        if columns:
            values = table[columns].apply(pd.to_numeric, errors="coerce").astype(float)
            filled_values = values.fillna(0.0)
            row_sums = filled_values.sum(axis=1).to_numpy(dtype=float)
            finite_sums = row_sums[np.isfinite(row_sums)]
            close = np.isclose(row_sums, 1.0, atol=tolerance, rtol=0.0)
            na_columns = [
                column
                for column in columns
                if column.endswith("__na")
                or column.endswith("__nan")
                or column.endswith("__none")
                or column.endswith("__missing")
            ]
            if na_columns:
                na_fraction = (
                    filled_values[na_columns]
                    .sum(axis=1)
                    .to_numpy(dtype=float)
                )
            else:
                na_fraction = np.zeros(len(table), dtype=float)
            rows_with_missing = int(values.isna().any(axis=1).sum())
        else:
            row_sums = np.full(len(table), np.nan)
            finite_sums = np.array([], dtype=float)
            close = np.zeros(len(table), dtype=bool)
            na_columns = []
            na_fraction = np.zeros(len(table), dtype=float)
            rows_with_missing = 0
        rows.append(
            {
                "composition_level": current_level,
                "n_columns": len(columns),
                "n_rows": len(table),
                "rows_close_to_one": int(close.sum()) if columns else 0,
                "rows_not_close_to_one": int((~close).sum()) if columns else len(table),
                "rows_with_missing_composition": rows_with_missing,
                "min_row_sum": float(np.min(finite_sums)) if len(finite_sums) else np.nan,
                "mean_row_sum": float(np.mean(finite_sums)) if len(finite_sums) else np.nan,
                "max_row_sum": float(np.max(finite_sums)) if len(finite_sums) else np.nan,
                "max_abs_error_from_one": (
                    float(np.max(np.abs(finite_sums - 1.0))) if len(finite_sums) else np.nan
                ),
                "na_composition_columns": ",".join(na_columns),
                "mean_na_composition_fraction": float(np.mean(na_fraction)) if len(table) else 0.0,
                "max_na_composition_fraction": float(np.max(na_fraction)) if len(table) else 0.0,
                "rows_with_nonzero_na_composition": int((na_fraction > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def dominant_composition(
    table: pd.DataFrame,
    level: str,
    by: Sequence[str] = ("scale",),
    top_n: int | None = None,
) -> pd.DataFrame:
    """Summarize dominant composition labels by group."""

    current_level = _canonical_level(level)
    columns = composition_columns(table, current_level)
    group_columns = [column for column in by if column in table.columns]
    output_columns = group_columns + [
        "composition_level",
        "dominant_label",
        "row_count",
        "fraction",
        "mean_dominant_fraction",
        "is_na_label",
    ]
    if not columns:
        return pd.DataFrame(columns=output_columns)

    prefix = COMPOSITION_PREFIXES[current_level]
    values = table[columns].apply(pd.to_numeric, errors="coerce").astype(float)
    filled = values.fillna(-np.inf)
    max_values = filled.max(axis=1)
    dominant_columns = filled.idxmax(axis=1)
    labels = dominant_columns.str[len(prefix) :].where(np.isfinite(max_values), "missing")
    base = table[group_columns].copy() if group_columns else pd.DataFrame(index=table.index)
    base["composition_level"] = current_level
    base["dominant_label"] = labels
    base["dominant_fraction"] = max_values.where(np.isfinite(max_values), np.nan)
    base["is_na_label"] = base["dominant_label"].isin({"na", "nan", "none", "missing"})

    rows = []
    grouped = (
        base.groupby(group_columns + ["composition_level", "dominant_label", "is_na_label"], dropna=False, observed=True)
        if group_columns
        else base.groupby(["composition_level", "dominant_label", "is_na_label"], dropna=False, observed=True)
    )
    totals = (
        base.groupby(group_columns, dropna=False, observed=True).size()
        if group_columns
        else pd.Series({(): len(base)})
    )
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {}
        for column, key in zip(group_columns, keys, strict=False):
            row[column] = key
        label_index = len(group_columns)
        composition_level = keys[label_index]
        dominant_label = keys[label_index + 1]
        is_na_label = bool(keys[label_index + 2])
        if not group_columns:
            total = len(base)
        elif len(group_columns) == 1:
            total = int(totals.loc[keys[0]])
        else:
            total = int(totals.loc[keys[: len(group_columns)]])
        row.update(
            {
                "composition_level": composition_level,
                "dominant_label": dominant_label,
                "row_count": int(len(group)),
                "fraction": len(group) / total if total else 0.0,
                "mean_dominant_fraction": float(group["dominant_fraction"].mean()),
                "is_na_label": is_na_label,
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows, columns=output_columns)
    if result.empty:
        return result
    result = result.sort_values(group_columns + ["row_count"], ascending=[True] * len(group_columns) + [False])
    if top_n is not None and group_columns:
        result = result.groupby(group_columns, dropna=False, observed=True).head(top_n)
    elif top_n is not None:
        result = result.head(top_n)
    return result.reset_index(drop=True)


def load_neighbor_metadata(path: str | Path) -> list[dict[str, object]]:
    """Load metadata records from a compressed neighbor-index NPZ."""

    with np.load(Path(path), allow_pickle=False) as loaded:
        if "metadata_json" not in loaded.files:
            return []
        raw = loaded["metadata_json"]
        if isinstance(raw, np.ndarray):
            raw = raw.item() if raw.shape == () else raw.tolist()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, list):
            raw = raw[0] if raw else "[]"
        metadata = json.loads(str(raw))
    if not isinstance(metadata, list):
        raise ValueError("neighbor metadata_json must decode to a list.")
    return metadata


def neighbor_raw_bytes_from_metadata(
    metadata: Sequence[Mapping[str, object]],
    index_dtype_bytes: int = 8,
) -> pd.DataFrame:
    """Estimate raw uncompressed neighbor-index bytes from metadata."""

    rows = []
    for record in metadata:
        n_anchors = int(record.get("n_anchors", 0))
        n_links = int(record.get("n_neighbor_links", 0))
        raw_bytes = int((n_anchors + n_anchors + 1 + n_links) * index_dtype_bytes)
        rows.append(
            {
                "entry": record.get("entry", ""),
                "slice_id": record.get("slice_id", ""),
                "slice_file": record.get("slice_file", ""),
                "scale": record.get("scale", ""),
                "n_anchors": n_anchors,
                "n_neighbor_links": n_links,
                "raw_bytes": raw_bytes,
            }
        )
    return pd.DataFrame(rows)


def _metadata_from_keys(files: Sequence[str]) -> list[dict[str, object]]:
    prefixes = sorted({name.split("__", 1)[0] for name in files if "__" in name})
    return [{"entry": prefix} for prefix in prefixes]


def _slice_bound(
    record: Mapping[str, object],
    slice_n_obs: Mapping[str, int] | None,
) -> int | None:
    if not slice_n_obs:
        return None
    for key in ("slice_id", "slice_file"):
        value = str(record.get(key, ""))
        if value in slice_n_obs:
            return int(slice_n_obs[value])
    return None


def validate_neighbor_npz(
    path: str | Path,
    feature_table: pd.DataFrame | None = None,
    slice_n_obs: Mapping[str, int] | None = None,
    expected_entries: int | None = None,
    include_anchor: bool | None = None,
    avg_tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Validate ragged neighbor-index arrays stored by M1 prototype code."""

    npz_path = Path(path)
    with np.load(npz_path, allow_pickle=False) as loaded:
        files = list(loaded.files)
        metadata = load_neighbor_metadata(npz_path) or _metadata_from_keys(files)
        rows = []
        entry_count_ok = expected_entries is None or len(metadata) == expected_entries
        for record in metadata:
            prefix = str(record.get("entry", ""))
            required = {
                "anchor_indices": f"{prefix}__anchor_indices",
                "indptr": f"{prefix}__indptr",
                "neighbor_indices": f"{prefix}__neighbor_indices",
            }
            missing_keys = [name for name in required.values() if name not in loaded.files]
            if missing_keys:
                rows.append(
                    {
                        "entry": prefix,
                        "slice_id": record.get("slice_id", ""),
                        "slice_file": record.get("slice_file", ""),
                        "scale": record.get("scale", ""),
                        "ok": False,
                        "errors": "missing keys: " + ",".join(missing_keys),
                        "entry_count": len(metadata),
                        "entry_count_ok": entry_count_ok,
                    }
                )
                continue

            anchors = np.asarray(loaded[required["anchor_indices"]], dtype=np.int64)
            indptr = np.asarray(loaded[required["indptr"]], dtype=np.int64)
            neighbors = np.asarray(loaded[required["neighbor_indices"]], dtype=np.int64)
            errors = []

            indptr_len_ok = len(indptr) == len(anchors) + 1
            if not indptr_len_ok:
                errors.append("indptr length != n_anchors + 1")
            indptr_start_ok = len(indptr) > 0 and int(indptr[0]) == 0
            if not indptr_start_ok:
                errors.append("indptr does not start at 0")
            indptr_monotonic = bool(np.all(np.diff(indptr) >= 0)) if len(indptr) else False
            if not indptr_monotonic:
                errors.append("indptr is not monotonic")
            final_len_ok = bool(len(indptr) and int(indptr[-1]) == len(neighbors))
            if not final_len_ok:
                errors.append("neighbor_indices length != indptr[-1]")

            negative_neighbors = int((neighbors < 0).sum())
            if negative_neighbors:
                errors.append("negative neighbor indices")

            bound = _slice_bound(record, slice_n_obs)
            max_neighbor_index = int(neighbors.max()) if len(neighbors) else -1
            within_slice_bounds = True if bound is None else max_neighbor_index < bound
            if bound is not None and not within_slice_bounds:
                errors.append("neighbor indices outside slice bounds")

            if "n_anchors" in record and int(record["n_anchors"]) != len(anchors):
                errors.append("metadata n_anchors mismatch")
            if "n_neighbor_links" in record and int(record["n_neighbor_links"]) != len(neighbors):
                errors.append("metadata n_neighbor_links mismatch")

            lengths = np.diff(indptr) if indptr_len_ok and indptr_monotonic else np.array([])
            avg_neighbors_npz = float(lengths.mean()) if len(lengths) else np.nan
            feature_rows = 0
            avg_neighbors_feature = np.nan
            avg_neighbors_match = np.nan
            if feature_table is not None and "n_neighbors" in feature_table.columns:
                mask = pd.Series(True, index=feature_table.index)
                for column in ("slice_file", "slice_id", "scale"):
                    value = record.get(column, "")
                    if column in feature_table.columns and value != "":
                        mask &= feature_table[column].astype(str) == str(value)
                selected = feature_table.loc[mask, "n_neighbors"]
                feature_rows = int(len(selected))
                if feature_rows:
                    avg_neighbors_feature = float(
                        pd.to_numeric(selected, errors="coerce").astype(float).mean()
                    )
                    avg_neighbors_match = bool(
                        abs(avg_neighbors_feature - avg_neighbors_npz) <= avg_tolerance
                    )
                    if not avg_neighbors_match:
                        errors.append("average n_neighbors mismatch")
                if feature_rows and feature_rows != len(anchors):
                    errors.append("feature rows != n_anchors")

            anchors_present = 0
            anchors_missing = 0
            anchors_unexpected = 0
            anchor_inclusion_ok = np.nan
            if include_anchor is not None and indptr_len_ok and final_len_ok:
                for row_idx, anchor in enumerate(anchors):
                    row = neighbors[indptr[row_idx] : indptr[row_idx + 1]]
                    present = bool(np.any(row == anchor))
                    anchors_present += int(present)
                    anchors_missing += int(include_anchor and not present)
                    anchors_unexpected += int((not include_anchor) and present)
                anchor_inclusion_ok = anchors_missing == 0 and anchors_unexpected == 0
                if not anchor_inclusion_ok:
                    errors.append("anchor inclusion mismatch")

            ok = (
                entry_count_ok
                and indptr_len_ok
                and indptr_start_ok
                and indptr_monotonic
                and final_len_ok
                and negative_neighbors == 0
                and within_slice_bounds
                and not errors
            )
            rows.append(
                {
                    "entry": prefix,
                    "slice_id": record.get("slice_id", ""),
                    "slice_file": record.get("slice_file", ""),
                    "scale": record.get("scale", ""),
                    "n_anchors": int(len(anchors)),
                    "indptr_length": int(len(indptr)),
                    "neighbor_indices_length": int(len(neighbors)),
                    "indptr_final": int(indptr[-1]) if len(indptr) else -1,
                    "negative_neighbor_count": negative_neighbors,
                    "slice_n_obs": bound,
                    "max_neighbor_index": max_neighbor_index,
                    "within_slice_bounds": within_slice_bounds,
                    "avg_neighbors_npz": avg_neighbors_npz,
                    "feature_rows": feature_rows,
                    "avg_neighbors_feature": avg_neighbors_feature,
                    "avg_neighbors_match": avg_neighbors_match,
                    "anchors_present_count": anchors_present,
                    "anchors_missing_count": anchors_missing,
                    "anchors_unexpected_count": anchors_unexpected,
                    "anchor_inclusion_ok": anchor_inclusion_ok,
                    "entry_count": len(metadata),
                    "entry_count_ok": entry_count_ok,
                    "ok": bool(ok),
                    "errors": "; ".join(errors),
                }
            )
    return pd.DataFrame(rows)


def estimate_full_m1_storage(
    *,
    full_anchors: int,
    scales: int | Sequence[str],
    prototype_rows: int | None = None,
    prototype_feature_bytes: int | None = None,
    avg_neighbors_by_scale: Mapping[str, float] | None = None,
    n_slices: int | None = None,
    parquet_ratio: float = 0.25,
    compressed_csv_ratio: float = 0.25,
    neighbor_compression_ratio: float = 0.35,
) -> dict[str, float | int]:
    """Estimate full M1 feature-table and neighbor-index storage."""

    scale_names = [str(scale) for scale in range(scales)] if isinstance(scales, int) else list(scales)
    full_feature_rows = int(full_anchors * len(scale_names))
    bytes_per_feature_row = (
        prototype_feature_bytes / prototype_rows
        if prototype_rows and prototype_feature_bytes
        else np.nan
    )
    feature_csv_bytes = (
        int(bytes_per_feature_row * full_feature_rows)
        if np.isfinite(bytes_per_feature_row)
        else 0
    )
    neighbor_raw_bytes = 0
    if avg_neighbors_by_scale:
        for scale in scale_names:
            avg_neighbors = float(avg_neighbors_by_scale.get(scale, 0.0))
            indptr_entries = full_anchors + (n_slices or 1)
            neighbor_raw_bytes += int(
                (full_anchors + indptr_entries + full_anchors * avg_neighbors) * 8
            )
    neighbor_npz_bytes = int(neighbor_raw_bytes * neighbor_compression_ratio)
    total_csv_npz_bytes = feature_csv_bytes + neighbor_npz_bytes
    total_parquet_npz_bytes = int(feature_csv_bytes * parquet_ratio) + neighbor_npz_bytes
    return {
        "full_anchors": int(full_anchors),
        "n_scales": int(len(scale_names)),
        "full_feature_rows": full_feature_rows,
        "bytes_per_feature_row_csv": float(bytes_per_feature_row),
        "feature_csv_bytes": int(feature_csv_bytes),
        "feature_parquet_bytes": int(feature_csv_bytes * parquet_ratio),
        "feature_csv_gzip_bytes": int(feature_csv_bytes * compressed_csv_ratio),
        "neighbor_raw_bytes": int(neighbor_raw_bytes),
        "neighbor_npz_bytes": int(neighbor_npz_bytes),
        "total_csv_plus_npz_bytes": int(total_csv_npz_bytes),
        "total_parquet_plus_npz_bytes": int(total_parquet_npz_bytes),
        "per_slice_csv_plus_npz_bytes": int(total_csv_npz_bytes / n_slices)
        if n_slices
        else 0,
        "per_slice_parquet_plus_npz_bytes": int(total_parquet_npz_bytes / n_slices)
        if n_slices
        else 0,
    }
