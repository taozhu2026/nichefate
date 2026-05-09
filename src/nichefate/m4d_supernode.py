"""Shared helpers for M4D supernode macrostate stages."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import softmax
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score


NO_DOWNSTREAM_FLAGS = {
    "no_full_node_dense_gpcca": True,
    "no_true_gpcca_when_backend_unavailable": True,
    "no_node_level_absorption_fate_probability": True,
    "no_branched_nicheflow_training": True,
    "no_branchsbm_training": True,
    "no_m5": True,
    "no_regulator_analysis": True,
}
TRUE_GPCCA_BACKENDS = ("pygpcca", "cellrank_gpcca_if_available")
DIAGNOSTIC_FALLBACK_BACKEND = "scipy_pcca_like_diagnostic_fallback"
NO_TRUE_GPCCA_BACKEND = "none_true_gpcca_available"
TRUE_GPCCA_BACKEND_LABEL = "standard GPCCA backend available"
NO_TRUE_GPCCA_BACKEND_LABEL = "no standard GPCCA backend available in current environment"
FALLBACK_BACKEND_LABEL = "scipy PCCA-like diagnostic fallback only; not true GPCCA"
NODE_COLUMNS = ["global_node_index", "anchor_id", "slice_id", "anchor_index", "anchor_cell_id", "time", "time_day"]
M2_META_COLUMNS = ["slice_id", "anchor_index", "anchor_cell_id", "time", "time_day", "mouse_id"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(json_safe(payload), indent=2) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def assert_no_ssd_path(path: Path, label: str) -> None:
    resolved = str(path.expanduser().resolve())
    if resolved == "/ssd" or resolved.startswith("/ssd/"):
        raise ValueError(f"Refusing to use /ssd for {label}: {path}")


def configured_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    for key, path in paths.items():
        assert_no_ssd_path(path, f"paths.{key}")
    return paths


def m4d_output_paths(paths: dict[str, Path]) -> dict[str, Path]:
    supernodes = paths["output_root"] / "supernodes"
    pcca = paths["output_root"] / "pcca"
    reports = paths["reports_dir"]
    figures = reports / "figures" / "m4d_01"
    return {
        "backend_md": reports / "m4d_pcca_backend_availability.md",
        "backend_csv": reports / "m4d_pcca_backend_availability.csv",
        "standard_gpcca_backend_plan": reports / "m4d_standard_gpcca_backend_plan.md",
        "cellrank_feasibility": reports / "m4d_cellrank_integration_feasibility.md",
        "standard_gpcca_next_step": reports / "m4d_standard_gpcca_next_step_recommendation.md",
        "node_to_supernode": supernodes / "node_to_supernode.parquet",
        "supernode_table": supernodes / "supernode_table.csv",
        "supernode_report": reports / "m4d_supernode_construction_report.md",
        "supernode_qc": reports / "m4d_supernode_qc_summary.csv",
        "p_super_forward": supernodes / "P_supernode_forward.npz",
        "p_super_absorbing": supernodes / "P_supernode_absorbing.npz",
        "supernode_edge_table": supernodes / "supernode_edge_table.csv",
        "markov_report": reports / "m4d_supernode_markov_report.md",
        "markov_qc": reports / "m4d_supernode_markov_qc_summary.csv",
        "macrostate_assignments": pcca / "supernode_macrostate_assignments.csv",
        "macrostate_memberships": pcca / "supernode_macrostate_memberships.csv",
        "terminal_candidates": pcca / "terminal_like_macrostate_candidates.csv",
        "node_projected": pcca / "node_projected_macrostate_summary.parquet",
        "pcca_report": reports / "m4d_supernode_pcca_report.md",
        "pcca_summary": reports / "m4d_supernode_pcca_summary.json",
        "comparison_md": reports / "m4d_pcca_vs_m4c_baseline_comparison.md",
        "comparison_csv": reports / "m4d_pcca_vs_m4c_baseline_comparison.csv",
        "figures_dir": figures,
    }


def infer_final_time(node_table: pd.DataFrame) -> tuple[float, str]:
    max_day = float(node_table["time_day"].astype(float).max())
    labels = sorted(
        node_table.loc[np.isclose(node_table["time_day"].astype(float), max_day), "time"]
        .dropna()
        .astype(str)
        .unique()
    )
    if len(labels) != 1:
        raise ValueError(f"Expected one final time label for max time_day {max_day}, found {labels}.")
    return max_day, labels[0]


def load_node_table(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path, columns=NODE_COLUMNS)
    expected = np.arange(len(table), dtype=np.int64)
    table = table.sort_values("global_node_index", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(table["global_node_index"].to_numpy(dtype=np.int64), expected):
        raise ValueError("M4A node table must be row-aligned by global_node_index.")
    return table


def m2_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*/m2_representation_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No M2 representation parquet files found under {root}")
    return files


def parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(path).names)


def load_feature_group_columns(schema_path: Path, selected_groups: list[str], sample_columns: list[str]) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    resolved = schema["resolved_feature_group_columns"]
    expected_scales = schema.get("expected_scales", ["radius_x2", "radius_x4", "radius_x8"])
    sample_set = set(sample_columns)
    columns: list[str] = []
    for group in selected_groups:
        base_columns = resolved[group]
        for scale in expected_scales:
            for base in base_columns:
                candidate = f"{scale}__{base}"
                if candidate in sample_set:
                    columns.append(candidate)
    columns = sorted(dict.fromkeys(columns))
    if not columns:
        raise ValueError(f"No M2 feature columns resolved for groups {selected_groups}.")
    return columns


def node_lookup_by_slice(node_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lookup: dict[str, pd.DataFrame] = {}
    for slice_id, group in node_table.groupby("slice_id", sort=False, observed=True):
        lookup[str(slice_id)] = group[["anchor_index", "global_node_index", "time", "time_day"]].set_index("anchor_index")
    return lookup


def read_m2_with_global(path: Path, feature_columns: list[str], lookup: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=sorted(set(M2_META_COLUMNS + feature_columns)))
    slice_values = frame["slice_id"].dropna().astype(str).unique()
    if len(slice_values) != 1:
        raise ValueError(f"Expected one slice_id in {path}, found {slice_values.tolist()}.")
    slice_id = slice_values[0]
    if slice_id not in lookup:
        raise KeyError(f"M2 slice {slice_id} is absent from the M4A node table.")
    merged = frame.merge(
        lookup[slice_id].reset_index(),
        on="anchor_index",
        how="left",
        suffixes=("", "_m4a"),
        sort=False,
    )
    if bool(merged["global_node_index"].isna().any()):
        raise ValueError(f"M2 file {path} has anchors missing from M4A node table.")
    if "time_m4a" in merged.columns and not (merged["time"].astype(str) == merged["time_m4a"].astype(str)).all():
        raise ValueError(f"M2 file {path} time labels disagree with M4A node table.")
    return merged.reset_index(drop=True)


def sample_feature_matrix(
    files: list[Path],
    feature_columns: list[str],
    lookup: dict[str, pd.DataFrame],
    max_rows: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    per_file = max(1, int(np.ceil(max_rows / len(files))))
    for path in files:
        frame = read_m2_with_global(path, feature_columns, lookup)
        step = max(1, len(frame) // per_file)
        sampled = frame.iloc[::step].head(per_file)
        rows.append(sampled[feature_columns].to_numpy(dtype=np.float32, copy=True))
    matrix = np.vstack(rows)
    if len(matrix) > max_rows:
        matrix = matrix[:max_rows]
    return matrix


def robust_safe_iqr_stats(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(sample, axis=0).astype(np.float32)
    q25 = np.nanquantile(sample, 0.25, axis=0).astype(np.float32)
    q75 = np.nanquantile(sample, 0.75, axis=0).astype(np.float32)
    iqr = (q75 - q25).astype(np.float32)
    near_constant = (~np.isfinite(iqr)) | (np.abs(iqr) < 1e-8)
    iqr[near_constant] = 1.0
    med[~np.isfinite(med)] = 0.0
    return med, iqr, near_constant


def transform_features(frame: pd.DataFrame, feature_columns: list[str], median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(median, np.where(bad)[1])
    return ((values - median) / scale).astype(np.float32, copy=False)


def supernode_counts_for_times(config: dict[str, Any], times: pd.DataFrame, final_time: str) -> dict[str, int]:
    configured = config["supernode_pcca"]["n_supernodes_per_time"]
    counts: dict[str, int] = {}
    for time_label in times["time"].astype(str).tolist():
        key = "final_time" if time_label == str(final_time) else time_label
        if key not in configured:
            raise KeyError(f"Missing n_supernodes_per_time entry for {time_label} (key {key}).")
        counts[time_label] = int(configured[key])
    return counts


def initialize_time_kmeans(counts: dict[str, int], random_seed: int, batch_size: int) -> dict[str, MiniBatchKMeans]:
    return {
        time_label: MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_seed,
            batch_size=batch_size,
            n_init=1,
            reassignment_ratio=0.01,
        )
        for time_label, n_clusters in counts.items()
    }


def train_kmeans_by_time(
    files: list[Path],
    feature_columns: list[str],
    lookup: dict[str, pd.DataFrame],
    median: np.ndarray,
    scale: np.ndarray,
    models: dict[str, MiniBatchKMeans],
    epochs: int,
) -> None:
    for _ in range(int(epochs)):
        for path in files:
            frame = read_m2_with_global(path, feature_columns, lookup)
            values = transform_features(frame, feature_columns, median, scale)
            for time_label, positions in frame.groupby("time", sort=False, observed=True).groups.items():
                models[str(time_label)].partial_fit(values[np.asarray(positions)])


def supernode_offsets(times: pd.DataFrame, counts: dict[str, int]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for time_label in times["time"].astype(str).tolist():
        offsets[time_label] = cursor
        cursor += counts[time_label]
    return offsets


def assign_nodes_to_supernodes(
    files: list[Path],
    feature_columns: list[str],
    lookup: dict[str, pd.DataFrame],
    median: np.ndarray,
    scale: np.ndarray,
    models: dict[str, MiniBatchKMeans],
    offsets: dict[str, int],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in files:
        frame = read_m2_with_global(path, feature_columns, lookup)
        values = transform_features(frame, feature_columns, median, scale)
        local_clusters = np.zeros(len(frame), dtype=np.int32)
        supernode_ids = np.zeros(len(frame), dtype=np.int32)
        for time_label, positions in frame.groupby("time", sort=False, observed=True).groups.items():
            pos = np.asarray(positions)
            labels = models[str(time_label)].predict(values[pos]).astype(np.int32, copy=False)
            local_clusters[pos] = labels
            supernode_ids[pos] = labels + int(offsets[str(time_label)])
        rows.append(
            pd.DataFrame(
                {
                    "global_node_index": frame["global_node_index"].to_numpy(dtype=np.int64),
                    "supernode_id": supernode_ids,
                    "local_supernode_id": local_clusters,
                    "time": frame["time"].astype(str).to_numpy(),
                    "time_day": frame["time_day"].astype(float).to_numpy(),
                    "slice_id": frame["slice_id"].astype(str).to_numpy(),
                    "anchor_index": frame["anchor_index"].to_numpy(dtype=np.int64),
                }
            )
        )
    result = pd.concat(rows, ignore_index=True).sort_values("global_node_index", kind="mergesort").reset_index(drop=True)
    expected = np.arange(len(result), dtype=np.int64)
    if not np.array_equal(result["global_node_index"].to_numpy(dtype=np.int64), expected):
        raise ValueError("node_to_supernode must contain exactly one row per aligned global_node_index.")
    return result


def build_supernode_table(
    node_to_supernode: pd.DataFrame,
    counts: dict[str, int],
    times: pd.DataFrame,
    final_time: str,
    max_fraction_warning: float,
    max_ratio_warning: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    cursor = 0
    for time_row in times.to_dict("records"):
        time_label = str(time_row["time"])
        group = node_to_supernode.loc[node_to_supernode["time"].astype(str) == time_label]
        sizes = group.groupby(["supernode_id", "local_supernode_id"], observed=True).size().reset_index(name="supernode_size")
        size_by_local = {
            int(row["local_supernode_id"]): int(row["supernode_size"])
            for row in sizes.to_dict("records")
        }
        total = int(len(group))
        median_size = float(sizes["supernode_size"].median()) if len(sizes) else 0.0
        expected_count = int(counts[time_label])
        empty = int(sum(1 for local_id in range(expected_count) if local_id not in size_by_local))
        for local_id in range(expected_count):
            size = int(size_by_local.get(local_id, 0))
            fraction = float(size / total) if total else 0.0
            ratio = float(size / median_size) if median_size else 0.0
            warn = bool(fraction > max_fraction_warning or ratio > max_ratio_warning)
            rows.append(
                {
                    "supernode_id": int(cursor + local_id),
                    "local_supernode_id": int(local_id),
                    "time": time_label,
                    "time_day": float(time_row["time_day"]),
                    "is_final_time": bool(time_label == str(final_time)),
                    "supernode_size": size,
                    "time_total_nodes": total,
                    "supernode_size_fraction_within_time": fraction,
                    "supernode_size_to_time_median_ratio": ratio,
                    "empty_supernode_warning": bool(size == 0),
                    "large_or_imbalanced_warning": warn,
                }
            )
        warning_rows.append(
            {
                "time": time_label,
                "time_day": float(time_row["time_day"]),
                "configured_supernodes": expected_count,
                "nonempty_supernodes": int(len(sizes)),
                "empty_supernodes": int(empty),
                "time_total_nodes": total,
                "max_supernode_size": int(sizes["supernode_size"].max()) if len(sizes) else 0,
                "median_supernode_size": median_size,
                "max_size_fraction": float(sizes["supernode_size"].max() / total) if len(sizes) and total else 0.0,
                "max_size_to_median_ratio": float(sizes["supernode_size"].max() / median_size) if len(sizes) and median_size else 0.0,
                "has_empty_supernode_warning": bool(empty > 0),
                "has_large_or_imbalanced_warning": bool(
                    len(sizes)
                    and (
                        float(sizes["supernode_size"].max() / total) > max_fraction_warning
                        or (median_size > 0 and float(sizes["supernode_size"].max() / median_size) > max_ratio_warning)
                    )
                ),
            }
        )
        cursor += expected_count
    table = pd.DataFrame(rows).sort_values("supernode_id").reset_index(drop=True)
    configured_total = sum(counts.values())
    if len(table) != configured_total:
        raise ValueError(f"Expected {configured_total} configured supernodes, found {len(table)}.")
    if table["supernode_id"].tolist() != list(range(configured_total)):
        raise ValueError("Supernode IDs must be stable, contiguous, and unique.")
    return table, pd.DataFrame(warning_rows)


def module_version(module: Any) -> str:
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else "unknown"


def safe_import_module(module_name: str) -> tuple[Any | None, bool, str, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        return None, False, "import_failed", f"{type(exc).__name__}: {exc}"
    return module, True, "import_ok", ""


def validate_pygpcca_toy_run(module: Any | None) -> dict[str, Any]:
    if module is None:
        return {
            "pygpcca_toy_run_status": "not_run_import_unavailable",
            "pygpcca_toy_true_gpcca_run": False,
            "pygpcca_toy_run_detail": "pygpcca could not be imported",
        }
    gpcca_cls = getattr(module, "GPCCA", None)
    if gpcca_cls is None:
        return {
            "pygpcca_toy_run_status": "skipped_no_gpcca_symbol",
            "pygpcca_toy_true_gpcca_run": False,
            "pygpcca_toy_run_detail": "pygpcca.GPCCA was not found",
        }

    dense = np.array(
        [
            [0.92, 0.08, 0.00, 0.00],
            [0.10, 0.86, 0.04, 0.00],
            [0.00, 0.04, 0.86, 0.10],
            [0.00, 0.00, 0.08, 0.92],
        ],
        dtype=np.float64,
    )
    matrix_candidates: list[Any] = [sp.csr_matrix(dense), dense]
    constructor_kwargs = [{}, {"z": "LM"}, {"z": "LM", "method": "brandts"}]
    method_calls = [
        ("optimize", ((2,), {})),
        ("optimize", ((), {"n_macrostates": 2})),
        ("optimize", ((), {"m": 2})),
        ("fit", ((2,), {})),
        ("fit", ((), {"n_macrostates": 2})),
    ]

    errors: list[str] = []
    for matrix in matrix_candidates:
        for kwargs in constructor_kwargs:
            try:
                gpcca = gpcca_cls(matrix, **kwargs)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"constructor {kwargs}: {type(exc).__name__}: {exc}")
                continue
            for method_name, (args, call_kwargs) in method_calls:
                method = getattr(gpcca, method_name, None)
                if method is None:
                    continue
                try:
                    method(*args, **call_kwargs)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{method_name}{args or call_kwargs}: {type(exc).__name__}: {exc}")
                    continue
                return {
                    "pygpcca_toy_run_status": "succeeded",
                    "pygpcca_toy_true_gpcca_run": True,
                    "pygpcca_toy_run_detail": f"4x4 stochastic matrix validated with {method_name}",
                }
            available_methods = [name for name in ["optimize", "fit"] if getattr(gpcca, name, None) is not None]
            errors.append(f"constructed but no known fit/optimize method; methods={available_methods[:8]}")
    detail = "; ".join(errors[-3:]) if errors else "no constructor or run attempts were possible"
    return {
        "pygpcca_toy_run_status": "failed",
        "pygpcca_toy_true_gpcca_run": False,
        "pygpcca_toy_run_detail": detail,
    }


def inspect_cellrank_integration(cellrank_module: Any | None) -> dict[str, Any]:
    if cellrank_module is None:
        return {
            "cellrank_gpcca_estimator_found": False,
            "cellrank_gpcca_estimator_path": "",
            "cellrank_precomputed_kernel_found": False,
            "cellrank_precomputed_kernel_path": "",
            "cellrank_custom_kernel_base_found": False,
            "cellrank_custom_kernel_base_path": "",
            "cellrank_precomputed_or_custom_kernel_feasible": False,
            "cellrank_integration_detail": "cellrank could not be imported",
        }

    estimator_paths: list[str] = []
    kernel_paths: list[str] = []
    custom_kernel_paths: list[str] = []
    import_errors: list[str] = []
    for module_name in ["cellrank.estimators", "cellrank.tl.estimators"]:
        try:
            submodule = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            import_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        if getattr(submodule, "GPCCA", None) is not None:
            estimator_paths.append(f"{module_name}.GPCCA")
    for module_name in ["cellrank.kernels", "cellrank.tl.kernels"]:
        try:
            submodule = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            import_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        if getattr(submodule, "PrecomputedKernel", None) is not None:
            kernel_paths.append(f"{module_name}.PrecomputedKernel")
        if getattr(submodule, "Kernel", None) is not None:
            custom_kernel_paths.append(f"{module_name}.Kernel")

    feasible = bool(estimator_paths) and bool(kernel_paths or custom_kernel_paths)
    detail = "GPCCA estimator and precomputed/custom kernel route appear present" if feasible else "missing GPCCA estimator or precomputed/custom kernel route"
    if import_errors:
        detail = f"{detail}; import notes: {' | '.join(import_errors[:3])}"
    return {
        "cellrank_gpcca_estimator_found": bool(estimator_paths),
        "cellrank_gpcca_estimator_path": ";".join(estimator_paths),
        "cellrank_precomputed_kernel_found": bool(kernel_paths),
        "cellrank_precomputed_kernel_path": ";".join(kernel_paths),
        "cellrank_custom_kernel_base_found": bool(custom_kernel_paths),
        "cellrank_custom_kernel_base_path": ";".join(custom_kernel_paths),
        "cellrank_precomputed_or_custom_kernel_feasible": feasible,
        "cellrank_integration_detail": detail,
    }


def inspect_backend_availability() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for backend, module_name, role, true_gpcca_backend in [
        ("pygpcca", "pygpcca", "standard_gpcca_backend", True),
        ("cellrank_gpcca_if_available", "cellrank", "standard_gpcca_backend", True),
        (DIAGNOSTIC_FALLBACK_BACKEND, "scipy", "diagnostic_fallback", False),
        ("sklearn_support_dependency", "sklearn", "support_dependency", False),
    ]:
        module_found = importlib.util.find_spec(module_name) is not None
        module, import_ok, import_status, import_error = (None, False, "module_not_found", "")
        if module_found:
            module, import_ok, import_status, import_error = safe_import_module(module_name)
        row = {
            "backend": backend,
            "module": module_name,
            "role": role,
            "module_found": bool(module_found),
            "import_ok": bool(import_ok),
            "available": bool(import_ok),
            "import_status": import_status,
            "import_error": import_error,
            "version": module_version(module) if module is not None else "",
            "selected": False,
            "true_gpcca_backend": bool(true_gpcca_backend),
            "true_gpcca_run": False,
            "result_label": TRUE_GPCCA_BACKEND_LABEL
            if true_gpcca_backend
            else (
                FALLBACK_BACKEND_LABEL
                if backend == DIAGNOSTIC_FALLBACK_BACKEND
                else "support dependency; not a GPCCA backend"
            ),
            "notes": "",
        }
        if backend == "pygpcca":
            row.update(validate_pygpcca_toy_run(module))
            row["true_gpcca_run"] = bool(row["pygpcca_toy_true_gpcca_run"])
            row["notes"] = str(row["pygpcca_toy_run_detail"])
        elif backend == "cellrank_gpcca_if_available":
            cellrank = inspect_cellrank_integration(module)
            row.update(cellrank)
            row["available"] = bool(import_ok and cellrank["cellrank_precomputed_or_custom_kernel_feasible"])
            row["notes"] = str(cellrank["cellrank_integration_detail"])
        elif backend == DIAGNOSTIC_FALLBACK_BACKEND:
            row["notes"] = "available only for bounded diagnostic/emergency fallback; not selected as true GPCCA"
        elif backend == "sklearn_support_dependency":
            row["notes"] = "support dependency for existing coarse spectral diagnostics, not a GPCCA backend"
        rows.append(row)

    frame = pd.DataFrame(rows)
    selected_backend = None
    if bool(frame.loc[frame["backend"] == "pygpcca", "available"].iloc[0]):
        selected_backend = "pygpcca"
    elif bool(frame.loc[frame["backend"] == "cellrank_gpcca_if_available", "available"].iloc[0]):
        cellrank_row = frame.loc[frame["backend"] == "cellrank_gpcca_if_available"].iloc[0]
        if bool(cellrank_row.get("cellrank_precomputed_or_custom_kernel_feasible", False)):
            selected_backend = "cellrank_gpcca_if_available"
    if selected_backend is not None:
        frame.loc[frame["backend"] == selected_backend, "selected"] = True
    return frame


def selected_backend_label(backend_frame: pd.DataFrame) -> tuple[str, str, bool]:
    selected_rows = backend_frame.loc[backend_frame["selected"].astype(bool)]
    if selected_rows.empty:
        return NO_TRUE_GPCCA_BACKEND, NO_TRUE_GPCCA_BACKEND_LABEL, False
    selected = selected_rows.iloc[0]
    backend = str(selected["backend"])
    is_true = backend in TRUE_GPCCA_BACKENDS
    label = str(selected["result_label"]) if is_true else FALLBACK_BACKEND_LABEL
    return backend, label, is_true


def dataframe_markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def backend_availability_markdown(backend: pd.DataFrame) -> str:
    selected_backend, result_label, true_gpcca_backend_available = selected_backend_label(backend)
    true_gpcca_toy_run = bool(backend.get("true_gpcca_run", pd.Series(dtype=bool)).astype(bool).any())
    lines = [
        "# M4D PCCA Backend Availability",
        "",
        f"- Inspection time UTC: {utc_now_iso()}",
        f"- Selected standard backend: `{selected_backend}`",
        f"- Result label: **{result_label}**",
        f"- True GPCCA backend available: `{bool(true_gpcca_backend_available)}`",
        f"- True GPCCA toy run completed: `{true_gpcca_toy_run}`",
        "- No package installation or environment modification was attempted.",
        "- M4D-01a does not run full-node dense GPCCA.",
        "- M4D-01a does not compute node-level absorption fate probabilities.",
        "- scipy is reported only as `scipy_pcca_like_diagnostic_fallback`; it is not true GPCCA.",
        "",
    ]
    if not true_gpcca_backend_available:
        lines.extend(
            [
                "No standard GPCCA backend is validated in the current environment. The main M4D route should remain blocked on standard GPCCA setup rather than silently switching to scipy diagnostics.",
                "",
            ]
        )
    lines.extend(["## Backend Table", "", dataframe_markdown_table(backend), "", "## Scope Safeguards", ""])
    for key, value in NO_DOWNSTREAM_FLAGS.items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def standard_gpcca_backend_plan_markdown(backend: pd.DataFrame) -> str:
    selected_backend, result_label, true_gpcca_backend_available = selected_backend_label(backend)
    total_supernodes = "configured supernode matrix, not the 1.44M-node matrix"
    lines = [
        "# M4D-01a Standard GPCCA Backend Plan",
        "",
        "M4D-01a validates the standard GPCCA integration route only. It does not run production GPCCA, compute fate or absorption probabilities, or start regulator/model-training stages.",
        "",
        "## Backend Priority",
        "- `pygpcca`: preferred standard GPCCA backend for the bounded supernode Markov matrix.",
        "- `cellrank_gpcca_if_available`: acceptable standard route if GPCCA estimator and precomputed/custom kernel integration are present.",
        "- `scipy_pcca_like_diagnostic_fallback`: diagnostic/emergency fallback only; never label its output as true GPCCA.",
        "",
        "## Current Selection",
        f"- selected standard backend: `{selected_backend}`",
        f"- result label: {result_label}",
        f"- true GPCCA backend available: `{bool(true_gpcca_backend_available)}`",
        f"- intended matrix scale: {total_supernodes}",
        "",
        "## Interface Contract",
        "- consume M4D supernode Markov transition matrix after supernode construction",
        "- keep node-level projection, absorption probability, Branched NicheFlow / BranchSBM, M5, and regulator analysis out of this stage",
        "- write only backend availability, CellRank feasibility, plan, and next-step recommendation reports in M4D-01a",
    ]
    return "\n".join(lines) + "\n"


def cellrank_integration_feasibility_markdown(backend: pd.DataFrame) -> str:
    row = backend.loc[backend["backend"] == "cellrank_gpcca_if_available"]
    if row.empty:
        values: dict[str, Any] = {}
    else:
        values = row.iloc[0].to_dict()
    lines = [
        "# M4D CellRank Integration Feasibility",
        "",
        "This report checks whether CellRank appears usable for standard GPCCA on a precomputed/custom supernode Markov kernel. It does not instantiate a full-node kernel or run production GPCCA.",
        "",
        f"- cellrank module found: `{bool(values.get('module_found', False))}`",
        f"- cellrank import available: `{bool(values.get('import_ok', False))}`",
        f"- CellRank GPCCA estimator found: `{bool(values.get('cellrank_gpcca_estimator_found', False))}`",
        f"- estimator path: `{values.get('cellrank_gpcca_estimator_path', '')}`",
        f"- precomputed kernel found: `{bool(values.get('cellrank_precomputed_kernel_found', False))}`",
        f"- precomputed kernel path: `{values.get('cellrank_precomputed_kernel_path', '')}`",
        f"- custom kernel base found: `{bool(values.get('cellrank_custom_kernel_base_found', False))}`",
        f"- custom kernel base path: `{values.get('cellrank_custom_kernel_base_path', '')}`",
        f"- precomputed/custom route appears feasible: `{bool(values.get('cellrank_precomputed_or_custom_kernel_feasible', False))}`",
        f"- detail: {values.get('cellrank_integration_detail', 'cellrank was not inspected')}",
    ]
    return "\n".join(lines) + "\n"


def standard_gpcca_next_step_recommendation_markdown(backend: pd.DataFrame) -> str:
    selected_backend, _, true_gpcca_backend_available = selected_backend_label(backend)
    if selected_backend == "pygpcca":
        immediate = "B. run pyGPCCA on the supernode Markov matrix"
    elif selected_backend == "cellrank_gpcca_if_available":
        immediate = "B. run CellRank GPCCA through the feasible precomputed/custom supernode kernel route"
    else:
        immediate = "A. create isolated nichefate-gpcca environment with pyGPCCA/CellRank"
    lines = [
        "# M4D Standard GPCCA Next-Step Recommendation",
        "",
        f"- immediate recommendation: {immediate}",
        f"- true GPCCA backend currently available: `{bool(true_gpcca_backend_available)}`",
        "",
        "## Ordered Route",
        "A. create isolated `nichefate-gpcca` environment with pyGPCCA/CellRank when the current environment lacks a validated standard backend",
        "B. run pyGPCCA on the supernode Markov matrix after standard backend validation",
        "C. use `scipy_pcca_like_diagnostic_fallback` only if standard backend setup fails, and label it diagnostic-only",
        "",
        "## Explicit Non-Goals",
        "- no full-node GPCCA",
        "- no node-level absorption probability",
        "- no Branched NicheFlow / BranchSBM",
        "- no M5",
        "- no regulator analysis",
    ]
    return "\n".join(lines) + "\n"


def write_standard_gpcca_review_reports(outputs: dict[str, Path], backend: pd.DataFrame) -> None:
    atomic_write_text(outputs["standard_gpcca_backend_plan"], standard_gpcca_backend_plan_markdown(backend))
    atomic_write_text(outputs["cellrank_feasibility"], cellrank_integration_feasibility_markdown(backend))
    atomic_write_text(outputs["standard_gpcca_next_step"], standard_gpcca_next_step_recommendation_markdown(backend))


def aggregate_supernode_transition(
    p_forward: sp.csr_matrix,
    assignment: np.ndarray,
    supernode_table: pd.DataFrame,
    row_sum_tolerance: float,
) -> tuple[sp.csr_matrix, sp.csr_matrix, pd.DataFrame, dict[str, Any]]:
    n_super = int(supernode_table["supernode_id"].max()) + 1
    if p_forward.shape[0] != len(assignment) or p_forward.shape[1] != len(assignment):
        raise ValueError("P_forward shape does not match node_to_supernode assignment length.")
    counts = np.diff(p_forward.indptr)
    source = np.repeat(assignment.astype(np.int32, copy=False), counts)
    target = assignment[p_forward.indices].astype(np.int32, copy=False)
    raw = sp.coo_matrix((p_forward.data.astype(np.float64, copy=False), (source, target)), shape=(n_super, n_super)).tocsr()
    raw.sum_duplicates()
    outgoing = np.asarray(raw.sum(axis=1)).ravel()
    incoming = np.asarray(raw.sum(axis=0)).ravel()
    forward = raw.copy().tolil()
    nonzero_rows = np.where(outgoing > 0)[0]
    for row in nonzero_rows:
        start = raw.indptr[row]
        end = raw.indptr[row + 1]
        if end > start:
            forward.rows[row] = raw.indices[start:end].tolist()
            forward.data[row] = (raw.data[start:end] / outgoing[row]).tolist()
    forward = forward.tocsr()
    final_supernodes = supernode_table.loc[supernode_table["is_final_time"].astype(bool), "supernode_id"].to_numpy(dtype=np.int64)
    final_forward_nnz = int(forward[final_supernodes, :].nnz)
    if final_forward_nnz:
        raise ValueError(f"Final-time supernode rows must have no outgoing forward transitions; found {final_forward_nnz}.")
    absorbing = forward.tolil()
    for row in final_supernodes:
        absorbing[row, row] = 1.0
    absorbing = absorbing.tocsr()
    row_sums = np.asarray(forward.sum(axis=1)).ravel()
    nonfinal_mask = ~supernode_table["is_final_time"].to_numpy(dtype=bool)
    nonfinal_error = np.abs(row_sums[nonfinal_mask] - 1.0)
    if len(nonfinal_error) and float(nonfinal_error.max()) > row_sum_tolerance:
        raise ValueError(f"Non-final supernode row sum max error exceeds tolerance: {float(nonfinal_error.max())}")
    absorbing_row_sums = np.asarray(absorbing.sum(axis=1)).ravel()
    absorbing_error = np.abs(absorbing_row_sums - 1.0)
    if float(absorbing_error.max()) > row_sum_tolerance:
        raise ValueError(f"Absorbing supernode row sum max error exceeds tolerance: {float(absorbing_error.max())}")
    meta = supernode_table.set_index("supernode_id")
    coo = raw.tocoo()
    prob_dense = forward.toarray()
    edge_rows = []
    for src, dst, mass in zip(coo.row, coo.col, coo.data, strict=True):
        edge_rows.append(
            {
                "source_supernode_id": int(src),
                "target_supernode_id": int(dst),
                "source_time": str(meta.loc[int(src), "time"]),
                "target_time": str(meta.loc[int(dst), "time"]),
                "source_time_day": float(meta.loc[int(src), "time_day"]),
                "target_time_day": float(meta.loc[int(dst), "time_day"]),
                "source_supernode_size": int(meta.loc[int(src), "supernode_size"]),
                "target_supernode_size": int(meta.loc[int(dst), "supernode_size"]),
                "transition_mass": float(mass),
                "outgoing_mass_before_normalization": float(outgoing[int(src)]),
                "incoming_mass": float(incoming[int(dst)]),
                "row_normalized_probability": float(prob_dense[int(src), int(dst)]),
            }
        )
    edge_table = pd.DataFrame(edge_rows)
    qc = {
        "n_supernodes": n_super,
        "forward_nnz": int(forward.nnz),
        "absorbing_nnz": int(absorbing.nnz),
        "nonfinal_row_sum_error_max": float(nonfinal_error.max()) if len(nonfinal_error) else 0.0,
        "absorbing_row_sum_error_max": float(absorbing_error.max()),
        "final_forward_outgoing_nnz": final_forward_nnz,
        "final_absorbing_self_loop_min": float(absorbing.diagonal()[final_supernodes].min()) if len(final_supernodes) else 0.0,
        "negative_values": int((forward.data < 0).sum() + (absorbing.data < 0).sum()),
        "nan_values": int(np.isnan(forward.data).sum() + np.isnan(absorbing.data).sum()),
    }
    return forward, absorbing, edge_table, qc


def spectral_embedding(p_absorbing: sp.csr_matrix, n_components: int) -> np.ndarray:
    dense = p_absorbing.T.toarray()
    eigvals, eigvecs = np.linalg.eig(dense)
    order = np.argsort(-eigvals.real)
    coords = eigvecs[:, order[:n_components]].real.astype(np.float64, copy=False)
    coords = coords / np.maximum(np.linalg.norm(coords, axis=1, keepdims=True), 1e-12)
    return coords


def cluster_spectral_macrostates(
    p_absorbing: sp.csr_matrix,
    n_macrostates: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = spectral_embedding(p_absorbing, max(2, int(n_macrostates)))
    kmeans = KMeans(n_clusters=int(n_macrostates), random_state=random_seed, n_init=20)
    labels = kmeans.fit_predict(coords).astype(np.int32)
    distances = kmeans.transform(coords)
    memberships = softmax(-distances, axis=1).astype(np.float32)
    top_membership = memberships[np.arange(len(labels)), labels]
    return labels, memberships, top_membership


def project_nodes_to_macrostates(
    node_to_supernode: pd.DataFrame,
    assignments: pd.DataFrame,
    memberships: np.ndarray,
    result_label: str,
    true_gpcca_run: bool,
) -> pd.DataFrame:
    required = {"global_node_index", "supernode_id"}
    missing = required.difference(node_to_supernode.columns)
    if missing:
        raise ValueError(f"node_to_supernode is missing columns: {sorted(missing)}")
    assign = assignments[["supernode_id", "macrostate_id", "macrostate_label", "membership_probability"]].copy()
    assign["supernode_id"] = assign["supernode_id"].astype(int)
    assign = assign.sort_values("supernode_id", kind="mergesort").reset_index(drop=True)
    expected = np.arange(len(assign), dtype=np.int64)
    if not np.array_equal(assign["supernode_id"].to_numpy(dtype=np.int64), expected):
        raise ValueError("Supernode macrostate assignments must be row-aligned by supernode_id.")
    super_ids = node_to_supernode["supernode_id"].to_numpy(dtype=np.int64)
    if super_ids.min(initial=0) < 0 or super_ids.max(initial=-1) >= memberships.shape[0]:
        raise ValueError("node_to_supernode contains supernode IDs outside membership matrix bounds.")
    projected = node_to_supernode[["global_node_index", "supernode_id"]].copy()
    projected["projected_macrostate_id"] = assign.loc[super_ids, "macrostate_id"].to_numpy(dtype=np.int32)
    projected["projected_macrostate_label"] = assign.loc[super_ids, "macrostate_label"].astype(str).to_numpy()
    projected["projected_macrostate_membership_probability"] = assign.loc[
        super_ids, "membership_probability"
    ].to_numpy(dtype=np.float32)
    projected["macrostate_result_label"] = result_label
    projected["true_gpcca_run"] = bool(true_gpcca_run)
    projected["directionality_evidence_source"] = "pseudo_lineage_time_coupled_transition"
    projected["barcode_compatible_contract"] = True
    node_memberships = memberships[super_ids]
    for macro_id in range(node_memberships.shape[1]):
        projected[f"membership_macrostate_{macro_id:02d}"] = node_memberships[:, macro_id].astype(np.float32, copy=False)
    projected = projected.sort_values("global_node_index", kind="mergesort").reset_index(drop=True)
    expected_nodes = np.arange(len(projected), dtype=np.int64)
    if not np.array_equal(projected["global_node_index"].to_numpy(dtype=np.int64), expected_nodes):
        raise ValueError("Projected macrostate table must preserve global_node_index identity and alignment.")
    return projected


def normalized_entropy(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    probs = counts.astype(float) / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum() / np.log(len(counts)))


def terminal_like_candidates(
    assignments: pd.DataFrame,
    memberships: pd.DataFrame,
    supernode_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    p_absorbing: sp.csr_matrix,
) -> pd.DataFrame:
    table = assignments.merge(supernode_table, on="supernode_id", how="left")
    final_time = str(supernode_table.loc[supernode_table["is_final_time"].astype(bool), "time"].iloc[0])
    global_final_fraction = float(
        supernode_table.loc[supernode_table["is_final_time"].astype(bool), "supernode_size"].sum()
        / supernode_table["supernode_size"].sum()
    )
    incoming = edge_table.groupby("target_supernode_id", observed=True)["transition_mass"].sum().to_dict() if len(edge_table) else {}
    diag = p_absorbing.diagonal()
    rows: list[dict[str, Any]] = []
    for macro_id, group in table.groupby("macrostate_id", sort=True, observed=True):
        size = int(group["supernode_size"].sum())
        final_size = int(group.loc[group["is_final_time"].astype(bool), "supernode_size"].sum())
        final_fraction = float(final_size / size) if size else 0.0
        enrichment = float(final_fraction / global_final_fraction) if global_final_fraction else 0.0
        absorbing_mass = float(
            np.average(diag[group["supernode_id"].to_numpy(dtype=int)], weights=group["supernode_size"].to_numpy(dtype=float))
            if size
            else 0.0
        )
        incoming_mass = float(sum(incoming.get(int(sid), 0.0) for sid in group["supernode_id"].to_numpy(dtype=int)))
        time_counts = group.groupby("time", observed=True)["supernode_size"].sum()
        dominant_time = str(time_counts.idxmax()) if len(time_counts) else ""
        entropy = normalized_entropy(time_counts.to_numpy(dtype=float))
        score = float(
            0.40 * final_fraction
            + 0.25 * min(enrichment / 5.0, 1.0)
            + 0.20 * absorbing_mass
            + 0.15 * (1.0 - entropy)
        )
        if score >= 0.65 and final_fraction >= 0.50:
            label = "high_terminal_like_candidate"
        elif score >= 0.40 and final_fraction >= 0.25:
            label = "moderate_terminal_like_candidate"
        else:
            label = "not_terminal_like"
        rows.append(
            {
                "macrostate_id": int(macro_id),
                "macrostate_label": f"pcca_like_macrostate_{int(macro_id):02d}",
                "size": size,
                "final_time_fraction": final_fraction,
                "final_time_enrichment": enrichment,
                "absorbing_mass": absorbing_mass,
                "incoming_mass": incoming_mass,
                "dominant_time": dominant_time,
                "time_entropy": entropy,
                "terminal_like_score": score,
                "terminal_like_label": label,
                "final_time_label": final_time,
            }
        )
    return pd.DataFrame(rows).sort_values("terminal_like_score", ascending=False).reset_index(drop=True)


def purity_score(labels_pred: np.ndarray, labels_true: np.ndarray) -> float:
    frame = pd.DataFrame({"pred": labels_pred, "true": labels_true})
    counts = frame.groupby(["pred", "true"], observed=True).size().reset_index(name="n")
    return float(counts.groupby("pred", observed=True)["n"].max().sum() / len(frame)) if len(frame) else 0.0


def comparison_tables(projected: pd.DataFrame, m4c: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    merged = projected[["global_node_index", "supernode_id", "projected_macrostate_id"]].merge(
        m4c[["global_node_index", "dominant_fate"]],
        on="global_node_index",
        how="left",
        sort=False,
    )
    if bool(merged["dominant_fate"].isna().any()):
        raise ValueError("M4C dominant fate comparison has missing rows after merge.")
    node_counts = (
        merged.groupby(["projected_macrostate_id", "dominant_fate"], observed=True)
        .size()
        .reset_index(name="count")
    )
    node_counts["comparison_level"] = "node_projected"
    super_major = (
        merged.groupby(["supernode_id", "dominant_fate"], observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(["supernode_id", "n"], ascending=[True, False])
        .drop_duplicates("supernode_id")
    )
    super_macro = merged[["supernode_id", "projected_macrostate_id"]].drop_duplicates()
    super_comp = super_macro.merge(super_major[["supernode_id", "dominant_fate"]], on="supernode_id", how="left")
    super_counts = (
        super_comp.groupby(["projected_macrostate_id", "dominant_fate"], observed=True)
        .size()
        .reset_index(name="count")
    )
    super_counts["comparison_level"] = "supernode_majority"
    output = pd.concat([node_counts, super_counts], ignore_index=True)
    output["fraction_of_level"] = output["count"] / output.groupby("comparison_level", observed=True)["count"].transform("sum")
    true = merged["dominant_fate"].to_numpy(dtype=int)
    pred = merged["projected_macrostate_id"].to_numpy(dtype=int)
    metrics = {
        "node_adjusted_mutual_info": float(adjusted_mutual_info_score(true, pred)),
        "node_normalized_mutual_info": float(normalized_mutual_info_score(true, pred)),
        "node_purity_like_score": purity_score(pred, true),
        "supernode_adjusted_mutual_info": float(
            adjusted_mutual_info_score(
                super_comp["dominant_fate"].to_numpy(dtype=int),
                super_comp["projected_macrostate_id"].to_numpy(dtype=int),
            )
        ),
        "supernode_normalized_mutual_info": float(
            normalized_mutual_info_score(
                super_comp["dominant_fate"].to_numpy(dtype=int),
                super_comp["projected_macrostate_id"].to_numpy(dtype=int),
            )
        ),
        "supernode_purity_like_score": purity_score(
            super_comp["projected_macrostate_id"].to_numpy(dtype=int),
            super_comp["dominant_fate"].to_numpy(dtype=int),
        ),
    }
    return output, metrics


def make_m4d01_figures(
    figures_dir: Path,
    supernode_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    assignments: pd.DataFrame,
    terminal_candidates: pd.DataFrame,
    comparison: pd.DataFrame,
    warning_only: bool = True,
) -> list[str]:
    warnings: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        data = [g["supernode_size"].to_numpy() for _, g in supernode_table.groupby("time", sort=True, observed=True)]
        labels = [str(k) for k, _ in supernode_table.groupby("time", sort=True, observed=True)]
        ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_title("Supernode size distribution by time")
        fig.tight_layout()
        fig.savefig(figures_dir / "m4d_supernode_size_by_time.png", dpi=140)
        plt.close(fig)

        if len(edge_table):
            n_super = int(max(edge_table["source_supernode_id"].max(), edge_table["target_supernode_id"].max())) + 1
            dense = np.zeros((n_super, n_super), dtype=float)
            dense[
                edge_table["source_supernode_id"].to_numpy(dtype=int),
                edge_table["target_supernode_id"].to_numpy(dtype=int),
            ] = edge_table["row_normalized_probability"].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(dense, aspect="auto", interpolation="nearest", cmap="magma")
            ax.set_title("Supernode transition probabilities")
            fig.colorbar(im, ax=ax, fraction=0.04)
            fig.tight_layout()
            fig.savefig(figures_dir / "m4d_supernode_transition_heatmap.png", dpi=140)
            plt.close(fig)

        macro_sizes = assignments.groupby("macrostate_id", observed=True)["supernode_size"].sum()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(macro_sizes.index.astype(str), macro_sizes.to_numpy(dtype=int))
        ax.set_title("PCCA-like macrostate node-size distribution")
        fig.tight_layout()
        fig.savefig(figures_dir / "m4d_pcca_macrostate_size_distribution.png", dpi=140)
        plt.close(fig)

        comp = assignments.pivot_table(
            index="macrostate_id",
            columns="time",
            values="supernode_size",
            aggfunc="sum",
            fill_value=0,
        )
        comp = comp.div(comp.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(comp.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title("PCCA-like macrostate time composition")
        ax.set_xticks(np.arange(len(comp.columns)))
        ax.set_xticklabels(comp.columns.astype(str), rotation=30)
        ax.set_yticks(np.arange(len(comp.index)))
        ax.set_yticklabels(comp.index.astype(str))
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(figures_dir / "m4d_pcca_macrostate_time_composition.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(terminal_candidates["macrostate_label"].astype(str), terminal_candidates["terminal_like_score"].astype(float))
        ax.tick_params(axis="x", rotation=45)
        ax.set_title("Terminal-like macrostate candidate scores")
        fig.tight_layout()
        fig.savefig(figures_dir / "m4d_terminal_like_macrostate_candidates.png", dpi=140)
        plt.close(fig)

        node_comp = comparison.loc[comparison["comparison_level"] == "node_projected"]
        heat = node_comp.pivot_table(
            index="projected_macrostate_id",
            columns="dominant_fate",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="cividis")
        ax.set_title("PCCA-like macrostate vs M4C dominant fate")
        ax.set_xticks(np.arange(len(heat.columns)))
        ax.set_xticklabels(heat.columns.astype(str), rotation=45)
        ax.set_yticks(np.arange(len(heat.index)))
        ax.set_yticklabels(heat.index.astype(str))
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(figures_dir / "m4d_pcca_vs_m4c_overlap_heatmap.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].bar(macro_sizes.index.astype(str), macro_sizes.to_numpy(dtype=int))
        axes[0, 0].set_title("Macrostate sizes")
        axes[0, 1].bar(terminal_candidates["macrostate_id"].astype(str), terminal_candidates["terminal_like_score"])
        axes[0, 1].set_title("Terminal-like scores")
        axes[1, 0].hist(supernode_table["supernode_size"], bins=30)
        axes[1, 0].set_title("Supernode sizes")
        axes[1, 1].imshow(heat.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="cividis")
        axes[1, 1].set_title("PCCA-like vs M4C")
        fig.tight_layout()
        fig.savefig(figures_dir / "m4d_supernode_pcca_dashboard.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        if not warning_only:
            raise
        warnings.append(f"M4D-01 figure generation failed after core outputs passed: {exc}")
    return warnings
