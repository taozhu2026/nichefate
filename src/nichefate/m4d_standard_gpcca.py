"""M4D-01c standard pyGPCCA execution on a supernode Markov chain."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans


NODE_COLUMNS = [
    "global_node_index",
    "anchor_id",
    "slice_id",
    "anchor_index",
    "anchor_cell_id",
    "time",
    "time_day",
    "mouse_id",
    "is_final_time",
]
M2_META_COLUMNS = ["slice_id", "anchor_index", "anchor_cell_id", "time", "time_day", "mouse_id"]
SELECTED_GROUPS = ["molecular_state", "cell_type_composition", "entropy", "spatial_summary", "topology"]
TEMP_ENV_KEYS = [
    "TMPDIR",
    "TEMP",
    "TMP",
    "OMPI_MCA_orte_tmpdir_base",
    "OMPI_MCA_prte_tmpdir_base",
    "PRTE_MCA_prte_tmpdir_base",
    "PMIX_MCA_pmix_tmpdir_base",
    "PETSC_TMP",
]


@dataclass(frozen=True)
class M4DPaths:
    node_table: Path
    p_forward: Path
    m2_by_slice_root: Path
    m2_feature_groups: Path
    reports_dir: Path
    supernode_root: Path
    gpcca_root: Path
    assignments: Path
    sizes: Path
    p_super: Path
    edge_table: Path
    component_md: Path
    component_csv: Path
    component_excluded: Path
    selected_k: Path
    memberships: Path
    macrostate_assignments: Path
    coarse_transition: Path
    gpcca_report: Path
    gpcca_summary: Path
    node_projection: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def atomic_write_text(path: Path, text: str, overwrite: bool = True) -> None:
    check_writable(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any], overwrite: bool = True) -> None:
    atomic_write_text(path, json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", overwrite=overwrite)


def atomic_write_csv(path: Path, frame: pd.DataFrame, overwrite: bool = True) -> None:
    check_writable(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame, overwrite: bool = True) -> None:
    check_writable(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_save_np(path: Path, array: np.ndarray, overwrite: bool = True) -> None:
    check_writable(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)


def atomic_save_sparse_npz(path: Path, matrix: sp.spmatrix, overwrite: bool = True) -> None:
    check_writable(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    sp.save_npz(tmp, matrix)
    os.replace(tmp, path)


def assert_no_ssd_path(path: Path, label: str) -> None:
    resolved = str(path.expanduser().resolve())
    if resolved == "/ssd" or resolved.startswith("/ssd/"):
        raise ValueError(f"Refusing to use /ssd for {label}: {path}")


def check_writable(path: Path, overwrite: bool) -> None:
    assert_no_ssd_path(path, "output")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output without --overwrite: {path}")


def require_parquet_engine() -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    ok = False
    for name in ["pyarrow", "fastparquet"]:
        try:
            module = __import__(name)
        except Exception as exc:  # noqa: BLE001
            statuses[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}", "version": ""}
            continue
        statuses[name] = {"available": True, "error": "", "version": str(getattr(module, "__version__", "unknown"))}
        ok = True
    if not ok:
        raise RuntimeError("No pandas parquet engine is available. Install pyarrow only in nichefate-gpcca before running.")
    return statuses


def parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(path).names)


def m4d_paths(config: dict[str, Any]) -> M4DPaths:
    paths = config["paths"]
    gpcca = config["standard_gpcca"]
    reports = Path(paths["reports_dir"])
    super_root = Path(gpcca["output_supernode_root"])
    gpcca_root = Path(gpcca["output_gpcca_root"])
    for label, raw in {
        "m4a_node_table": paths["m4a_node_table"],
        "p_forward": paths["p_forward"],
        "m2_by_slice_root": paths["m2_by_slice_root"],
        "reports_dir": reports,
        "output_supernode_root": super_root,
        "output_gpcca_root": gpcca_root,
    }.items():
        assert_no_ssd_path(Path(raw), label)
    return M4DPaths(
        node_table=Path(paths["m4a_node_table"]),
        p_forward=Path(paths["p_forward"]),
        m2_by_slice_root=Path(paths["m2_by_slice_root"]),
        m2_feature_groups=Path(paths["m2_feature_groups"]),
        reports_dir=reports,
        supernode_root=super_root,
        gpcca_root=gpcca_root,
        assignments=super_root / "supernode_assignments.parquet",
        sizes=super_root / "supernode_sizes.csv",
        p_super=super_root / "P_super_csr.npz",
        edge_table=super_root / "supernode_edge_table.csv",
        component_md=reports / "m4d_supernode_component_report.md",
        component_csv=reports / "m4d_supernode_component_summary.csv",
        component_excluded=reports / "m4d_supernode_component_excluded_supernodes.csv",
        selected_k=gpcca_root / "gpcca_selected_k.json",
        memberships=gpcca_root / "gpcca_memberships.npy",
        macrostate_assignments=gpcca_root / "gpcca_macrostate_assignments.csv",
        coarse_transition=gpcca_root / "gpcca_coarse_transition_matrix.npy",
        gpcca_report=reports / "m4d_gpcca_report.md",
        gpcca_summary=reports / "m4d_gpcca_summary.json",
        node_projection=gpcca_root / "node_gpcca_macrostate_membership.parquet",
    )


def production_outputs(paths: M4DPaths) -> list[Path]:
    return [
        paths.assignments,
        paths.sizes,
        paths.p_super,
        paths.edge_table,
        paths.component_md,
        paths.component_csv,
        paths.selected_k,
        paths.memberships,
        paths.macrostate_assignments,
        paths.coarse_transition,
        paths.gpcca_report,
        paths.gpcca_summary,
        paths.node_projection,
    ]


def validate_existing_outputs(paths: M4DPaths, expected_nodes: int | None = None) -> tuple[bool, str]:
    missing = [str(path) for path in production_outputs(paths) if not path.exists()]
    if missing:
        return False, "missing outputs: " + "; ".join(missing[:5])
    try:
        selected = json.loads(paths.selected_k.read_text(encoding="utf-8"))
        memberships = np.load(paths.memberships)
        assignments = pd.read_csv(paths.macrostate_assignments)
        coarse = np.load(paths.coarse_transition)
        projected = pd.read_parquet(paths.node_projection, columns=["global_node_index", "gpcca_macrostate_id"])
    except Exception as exc:  # noqa: BLE001
        return False, f"output QC read failed: {type(exc).__name__}: {exc}"
    if "selected_k" not in selected or int(selected["selected_k"]) <= 0:
        return False, "selected_k JSON is invalid"
    if memberships.ndim != 2 or coarse.ndim != 2:
        return False, "memberships or coarse transition matrix has invalid shape"
    if assignments.empty:
        return False, "macrostate assignment table is empty"
    if expected_nodes is not None and len(projected) != int(expected_nodes):
        return False, f"node projection row count {len(projected)} != expected {expected_nodes}"
    expected = np.arange(len(projected), dtype=np.int64)
    if not np.array_equal(projected["global_node_index"].to_numpy(dtype=np.int64), expected):
        return False, "node projection global_node_index is not contiguous and aligned"
    return True, "existing M4D-01c outputs passed QC"


def prepare_output_dirs(paths: M4DPaths) -> None:
    for path in [paths.supernode_root, paths.gpcca_root, paths.reports_dir]:
        assert_no_ssd_path(path, "output directory")
        path.mkdir(parents=True, exist_ok=True)


def load_node_table(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path, columns=NODE_COLUMNS)
    table = table.sort_values("global_node_index", kind="mergesort").reset_index(drop=True)
    expected = np.arange(len(table), dtype=np.int64)
    if not np.array_equal(table["global_node_index"].to_numpy(dtype=np.int64), expected):
        raise ValueError("M4A node table must be sorted and contiguous by global_node_index.")
    table["time_label"] = table["time"].astype(str)
    return table


def time_summary(node_table: pd.DataFrame) -> pd.DataFrame:
    summary = (
        node_table.groupby(["time_label", "time_day"], sort=True, observed=True)
        .size()
        .reset_index(name="node_count")
        .sort_values(["time_day", "time_label"])
        .reset_index(drop=True)
    )
    return summary


def allocate_supernodes_largest_remainder(
    time_counts: pd.DataFrame,
    target_supernodes: int,
    allowed_range: tuple[int, int] = (5000, 20000),
) -> dict[str, int]:
    low, high = allowed_range
    if not (int(low) <= int(target_supernodes) <= int(high)):
        raise ValueError(f"target_supernodes={target_supernodes} outside allowed range {allowed_range}")
    counts = time_counts["node_count"].to_numpy(dtype=np.int64)
    labels = time_counts["time_label"].astype(str).tolist()
    if int(counts.sum()) <= 0:
        raise ValueError("Cannot allocate supernodes without nodes.")
    raw = counts.astype(np.float64) * float(target_supernodes) / float(counts.sum())
    floors = np.floor(raw).astype(int)
    alloc = np.maximum(floors, 1)
    alloc = np.minimum(alloc, counts.astype(int))
    while int(alloc.sum()) < int(target_supernodes):
        capacity = counts.astype(int) - alloc
        if int(capacity.max()) <= 0:
            raise ValueError("Insufficient node capacity for requested supernode count.")
        remainders = raw - np.floor(raw)
        score = np.where(capacity > 0, remainders, -1.0)
        idx = int(np.argmax(score))
        alloc[idx] += 1
    while int(alloc.sum()) > int(target_supernodes):
        removable = alloc > 1
        score = np.where(removable, raw - alloc, np.inf)
        idx = int(np.argmin(score))
        alloc[idx] -= 1
    return {label: int(value) for label, value in zip(labels, alloc, strict=True)}


def discover_m2_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*/m2_representation_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No M2 representation parquet files found under {root}")
    return files


def load_feature_columns(
    m2_feature_groups_path: Path,
    m3_feature_groups_path: Path,
    m2_schema_path: Path,
    selected_groups: list[str],
    sample_columns: list[str],
) -> tuple[list[str], str]:
    sample_set = set(sample_columns)
    if m2_feature_groups_path.exists():
        schema = json.loads(m2_feature_groups_path.read_text(encoding="utf-8"))
        resolved = schema["resolved_feature_group_columns"]
        scales = schema.get("expected_scales", ["radius_x2", "radius_x4", "radius_x8"])
        columns: list[str] = []
        for group in selected_groups:
            for base in resolved[group]:
                for scale in scales:
                    candidate = f"{scale}__{base}"
                    if candidate in sample_set:
                        columns.append(candidate)
        columns = sorted(dict.fromkeys(columns))
        if columns:
            return columns, str(m2_feature_groups_path)
    if not m3_feature_groups_path.exists() or not m2_schema_path.exists():
        raise FileNotFoundError("M2 feature groups missing and alternate M3/M2 feature schema files are unavailable.")
    m3 = json.loads(m3_feature_groups_path.read_text(encoding="utf-8"))
    m2_schema = json.loads(m2_schema_path.read_text(encoding="utf-8"))
    allowed = set(m2_schema.get("numeric_feature_columns", m2_schema.get("output_columns", [])))
    columns = []
    for group in selected_groups:
        columns.extend(m3["feature_groups"].get(group, []))
    columns = sorted(dict.fromkeys(column for column in columns if column in allowed and column in sample_set))
    if not columns:
        raise ValueError("No feature columns resolved from M3/M2 fallback schemas.")
    return columns, f"{m3_feature_groups_path};{m2_schema_path}"


def node_lookup_by_slice(node_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lookup: dict[str, pd.DataFrame] = {}
    for slice_id, group in node_table.groupby("slice_id", sort=False, observed=True):
        lookup[str(slice_id)] = group[["anchor_index", "global_node_index", "time_label", "time_day"]].set_index("anchor_index")
    return lookup


def read_m2_with_global(path: Path, feature_columns: list[str], lookup: dict[str, pd.DataFrame]) -> pd.DataFrame:
    read_columns = sorted(set(M2_META_COLUMNS + feature_columns))
    frame = pd.read_parquet(path, columns=read_columns)
    slice_values = frame["slice_id"].dropna().astype(str).unique()
    if len(slice_values) != 1:
        raise ValueError(f"Expected one slice_id in {path}, found {slice_values.tolist()}.")
    slice_id = slice_values[0]
    if slice_id not in lookup:
        raise KeyError(f"M2 slice {slice_id} is absent from M4A node table.")
    merged = frame.merge(
        lookup[slice_id].reset_index(),
        on="anchor_index",
        how="left",
        sort=False,
        suffixes=("", "_m4a"),
    )
    if bool(merged["global_node_index"].isna().any()):
        raise ValueError(f"M2 file {path} has anchors missing from M4A node table.")
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


def robust_safe_iqr_stats(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    median = np.nanmedian(sample, axis=0).astype(np.float32)
    q25 = np.nanquantile(sample, 0.25, axis=0).astype(np.float32)
    q75 = np.nanquantile(sample, 0.75, axis=0).astype(np.float32)
    scale = (q75 - q25).astype(np.float32)
    near_constant = (~np.isfinite(scale)) | (np.abs(scale) < 1e-8)
    scale[near_constant] = 1.0
    median[~np.isfinite(median)] = 0.0
    return median, scale, int(near_constant.sum())


def transform_features(frame: pd.DataFrame, feature_columns: list[str], median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(median, np.where(bad)[1])
    values = (values - median) / scale
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def initialize_models(
    files: list[Path],
    feature_columns: list[str],
    lookup: dict[str, pd.DataFrame],
    median: np.ndarray,
    scale: np.ndarray,
    allocations: dict[str, int],
    random_seed: int,
    batch_size: int,
) -> dict[str, MiniBatchKMeans]:
    models = {
        label: MiniBatchKMeans(
            n_clusters=count,
            random_state=random_seed,
            batch_size=batch_size,
            init="random",
            n_init=1,
            reassignment_ratio=0.01,
            max_no_improvement=20,
        )
        for label, count in allocations.items()
    }
    buffers: dict[str, list[np.ndarray]] = {label: [] for label in allocations}
    initialized: set[str] = set()
    for path in files:
        frame = read_m2_with_global(path, feature_columns, lookup)
        values = transform_features(frame, feature_columns, median, scale)
        for label, positions in frame.groupby("time_label", sort=False, observed=True).groups.items():
            label = str(label)
            if label in initialized:
                continue
            buffers[label].append(values[np.asarray(positions)])
            total = sum(len(part) for part in buffers[label])
            if total >= allocations[label]:
                batch = np.vstack(buffers[label])
                print(
                    f"[M4D-01c] initializing {label} MiniBatchKMeans with "
                    f"{len(batch)} rows and {allocations[label]} clusters",
                    flush=True,
                )
                models[label].partial_fit(batch)
                initialized.add(label)
        if len(initialized) == len(allocations):
            break
    missing = sorted(set(allocations).difference(initialized))
    if missing:
        raise ValueError(f"Could not initialize MiniBatchKMeans for time layers: {missing}")
    return models


def train_models(
    files: list[Path],
    feature_columns: list[str],
    lookup: dict[str, pd.DataFrame],
    median: np.ndarray,
    scale: np.ndarray,
    models: dict[str, MiniBatchKMeans],
    epochs: int,
) -> None:
    for epoch in range(int(epochs)):
        print(f"[M4D-01c] training MiniBatchKMeans epoch {epoch + 1}/{int(epochs)}", flush=True)
        for path in files:
            frame = read_m2_with_global(path, feature_columns, lookup)
            values = transform_features(frame, feature_columns, median, scale)
            for label, positions in frame.groupby("time_label", sort=False, observed=True).groups.items():
                models[str(label)].partial_fit(values[np.asarray(positions)])


def supernode_offsets(time_counts: pd.DataFrame, allocations: dict[str, int]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for label in time_counts["time_label"].astype(str).tolist():
        offsets[label] = cursor
        cursor += int(allocations[label])
    return offsets


def assign_supernodes(
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
        local = np.zeros(len(frame), dtype=np.int32)
        super_ids = np.zeros(len(frame), dtype=np.int32)
        for label, positions in frame.groupby("time_label", sort=False, observed=True).groups.items():
            pos = np.asarray(positions)
            labels = models[str(label)].predict(values[pos]).astype(np.int32, copy=False)
            local[pos] = labels
            super_ids[pos] = labels + int(offsets[str(label)])
        rows.append(
            pd.DataFrame(
                {
                    "global_node_index": frame["global_node_index"].to_numpy(dtype=np.int64),
                    "time_label": frame["time_label"].astype(str).to_numpy(),
                    "time_day": frame["time_day_m4a"].astype(float).to_numpy()
                    if "time_day_m4a" in frame.columns
                    else frame["time_day"].astype(float).to_numpy(),
                    "slice_id": frame["slice_id"].astype(str).to_numpy(),
                    "anchor_index": frame["anchor_index"].to_numpy(dtype=np.int64),
                    "supernode_id": super_ids,
                    "local_supernode_id": local,
                }
            )
        )
    out = pd.concat(rows, ignore_index=True).sort_values("global_node_index", kind="mergesort").reset_index(drop=True)
    expected = np.arange(len(out), dtype=np.int64)
    if not np.array_equal(out["global_node_index"].to_numpy(dtype=np.int64), expected):
        raise ValueError("Supernode assignments must contain exactly one aligned row per global_node_index.")
    return out


def supernode_size_table(assignments: pd.DataFrame, time_counts: pd.DataFrame, allocations: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    offsets = supernode_offsets(time_counts, allocations)
    final_day = float(time_counts["time_day"].max())
    for record in time_counts.to_dict("records"):
        label = str(record["time_label"])
        total = int(record["node_count"])
        group = assignments.loc[assignments["time_label"].astype(str) == label]
        sizes = group.groupby("supernode_id", observed=True).size().to_dict()
        local_sizes = []
        for local_id in range(int(allocations[label])):
            sid = int(offsets[label] + local_id)
            size = int(sizes.get(sid, 0))
            local_sizes.append(size)
        median_size = float(np.median([value for value in local_sizes if value > 0])) if any(local_sizes) else 0.0
        for local_id, size in enumerate(local_sizes):
            sid = int(offsets[label] + local_id)
            fraction = float(size / total) if total else 0.0
            rows.append(
                {
                    "supernode_id": sid,
                    "local_supernode_id": int(local_id),
                    "time_label": label,
                    "time_day": float(record["time_day"]),
                    "is_final_time": bool(np.isclose(float(record["time_day"]), final_day)),
                    "supernode_size": int(size),
                    "time_total_nodes": total,
                    "supernode_size_fraction_within_time": fraction,
                    "supernode_size_to_time_median_ratio": float(size / median_size) if median_size else 0.0,
                    "empty_supernode_warning": bool(size == 0),
                    "severe_imbalance_warning": bool(fraction > 0.15 or (median_size and size / median_size > 5.0)),
                }
            )
    table = pd.DataFrame(rows).sort_values("supernode_id").reset_index(drop=True)
    expected = list(range(int(sum(allocations.values()))))
    if table["supernode_id"].tolist() != expected:
        raise ValueError("Supernode IDs must be contiguous.")
    if bool((table["supernode_size"] <= 0).any()):
        empty = table.loc[table["supernode_size"] <= 0, "supernode_id"].head(10).tolist()
        raise ValueError(f"Empty supernodes detected after assignment; first IDs: {empty}")
    return table


def build_supernodes(
    config: dict[str, Any],
    paths: M4DPaths,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    gpcca = config["standard_gpcca"]
    print("[M4D-01c] loading M4A node table", flush=True)
    node_table = load_node_table(paths.node_table)
    counts = time_summary(node_table)
    allocations = allocate_supernodes_largest_remainder(
        counts,
        int(gpcca["target_supernodes"]),
        tuple(int(x) for x in gpcca["allowed_supernode_range"]),
    )
    files = discover_m2_files(paths.m2_by_slice_root)
    print(f"[M4D-01c] discovered {len(files)} M2 representation files", flush=True)
    sample_cols = parquet_columns(files[0])
    feature_cols, feature_source = load_feature_columns(
        paths.m2_feature_groups,
        Path("/home/zhutao/scratch/nichefate/m3/reports/m3_feature_groups.json"),
        Path("/home/zhutao/scratch/nichefate/m2/reports/m2_full_feature_schema.json"),
        SELECTED_GROUPS,
        sample_cols,
    )
    lookup = node_lookup_by_slice(node_table)
    print(f"[M4D-01c] sampling feature matrix with {len(feature_cols)} columns", flush=True)
    sample = sample_feature_matrix(files, feature_cols, lookup, int(config.get("supernode_pcca", {}).get("scaling_sample_rows", 200000)))
    median, scale, near_constant = robust_safe_iqr_stats(sample)
    print("[M4D-01c] initializing time-layer MiniBatchKMeans models", flush=True)
    models = initialize_models(
        files,
        feature_cols,
        lookup,
        median,
        scale,
        allocations,
        int(gpcca.get("random_seed", 1)),
        int(config.get("supernode_pcca", {}).get("minibatch_size", 8192)),
    )
    train_models(
        files,
        feature_cols,
        lookup,
        median,
        scale,
        models,
        int(config.get("supernode_pcca", {}).get("training_epochs", 2)),
    )
    print("[M4D-01c] assigning nodes to supernodes", flush=True)
    assignments = assign_supernodes(files, feature_cols, lookup, median, scale, models, supernode_offsets(counts, allocations))
    sizes = supernode_size_table(assignments, counts, allocations)
    if len(assignments) != len(node_table):
        raise ValueError(f"Assignment row count {len(assignments)} != node table row count {len(node_table)}")
    if assignments["global_node_index"].nunique() != len(node_table):
        raise ValueError("global_node_index is not unique in supernode assignments.")
    atomic_write_parquet(paths.assignments, assignments[["global_node_index", "time_label", "time_day", "slice_id", "anchor_index", "supernode_id"]], overwrite)
    atomic_write_csv(paths.sizes, sizes, overwrite)
    meta = {
        "feature_group_source": feature_source,
        "feature_column_count": len(feature_cols),
        "near_constant_feature_count": near_constant,
        "allocation": allocations,
        "n_nodes": len(assignments),
        "n_supernodes": int(len(sizes)),
        "supernode_size_min": int(sizes["supernode_size"].min()),
        "supernode_size_median": float(sizes["supernode_size"].median()),
        "supernode_size_max": int(sizes["supernode_size"].max()),
        "severe_imbalance_warnings": int(sizes["severe_imbalance_warning"].sum()),
    }
    return assignments, sizes, meta


def load_or_build_supernodes(config: dict[str, Any], paths: M4DPaths, resume: bool, overwrite: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if resume and paths.assignments.exists() and paths.sizes.exists():
        assignments = pd.read_parquet(paths.assignments)
        sizes = pd.read_csv(paths.sizes)
        if len(sizes) == int(config["standard_gpcca"]["target_supernodes"]) and assignments["global_node_index"].is_unique:
            meta = {
                "resumed_supernodes": True,
                "n_nodes": int(len(assignments)),
                "n_supernodes": int(len(sizes)),
                "supernode_size_min": int(sizes["supernode_size"].min()),
                "supernode_size_median": float(sizes["supernode_size"].median()),
                "supernode_size_max": int(sizes["supernode_size"].max()),
                "severe_imbalance_warnings": int(sizes.get("severe_imbalance_warning", pd.Series(dtype=bool)).sum()),
                "allocation": {
                    str(label): int(count)
                    for label, count in sizes.groupby("time_label", sort=True, observed=True).size().items()
                },
            }
            try:
                files = discover_m2_files(paths.m2_by_slice_root)
                sample_cols = parquet_columns(files[0])
                feature_cols, feature_source = load_feature_columns(
                    paths.m2_feature_groups,
                    Path("/home/zhutao/scratch/nichefate/m3/reports/m3_feature_groups.json"),
                    Path("/home/zhutao/scratch/nichefate/m2/reports/m2_full_feature_schema.json"),
                    SELECTED_GROUPS,
                    sample_cols,
                )
                meta["feature_group_source"] = feature_source
                meta["feature_column_count"] = int(len(feature_cols))
            except Exception as exc:  # noqa: BLE001
                meta["feature_group_source"] = f"resumed; feature schema recheck unavailable: {type(exc).__name__}: {exc}"
            return assignments, sizes, meta
    return build_supernodes(config, paths, overwrite=overwrite)


def aggregate_p_super(
    p_forward_path: Path,
    assignments: pd.DataFrame,
    sizes: pd.DataFrame,
    row_sum_tolerance: float,
) -> tuple[sp.csr_matrix, pd.DataFrame, dict[str, Any]]:
    print("[M4D-01c] loading and aggregating M4A P_forward to P_super", flush=True)
    p_forward = sp.load_npz(p_forward_path).tocsr()
    assignment = assignments.sort_values("global_node_index", kind="mergesort")["supernode_id"].to_numpy(dtype=np.int32)
    n_super = int(sizes["supernode_id"].max()) + 1
    if p_forward.shape != (len(assignment), len(assignment)):
        raise ValueError("P_forward shape does not match supernode assignment length.")
    counts = np.diff(p_forward.indptr)
    source = np.repeat(assignment, counts)
    target = assignment[p_forward.indices]
    raw = sp.coo_matrix((p_forward.data.astype(np.float64, copy=False), (source, target)), shape=(n_super, n_super)).tocsr()
    raw.sum_duplicates()
    row_mass = np.asarray(raw.sum(axis=1)).ravel()
    normalized = raw.copy().astype(np.float64)
    repeats = np.diff(normalized.indptr)
    inv = np.zeros_like(row_mass, dtype=np.float64)
    nonzero = row_mass > 0
    inv[nonzero] = 1.0 / row_mass[nonzero]
    normalized.data *= np.repeat(inv, repeats)
    zero_rows = np.where(row_mass <= 0)[0]
    final_mask = sizes.sort_values("supernode_id")["is_final_time"].to_numpy(dtype=bool)
    final_zero = [int(row) for row in zero_rows if final_mask[int(row)]]
    nonfinal_zero = [int(row) for row in zero_rows if not final_mask[int(row)]]
    if len(zero_rows):
        closure = sp.csr_matrix((np.ones(len(zero_rows), dtype=np.float64), (zero_rows, zero_rows)), shape=normalized.shape)
        normalized = (normalized + closure).tocsr()
    row_sums = np.asarray(normalized.sum(axis=1)).ravel()
    max_error = float(np.max(np.abs(row_sums - 1.0))) if len(row_sums) else 0.0
    if max_error > float(row_sum_tolerance):
        raise ValueError(f"P_super row sum max error {max_error} exceeds tolerance {row_sum_tolerance}")
    if bool(np.isnan(normalized.data).any()) or bool((normalized.data < 0).any()):
        raise ValueError("P_super contains NaN or negative values.")
    coo = normalized.tocoo()
    edge_table = pd.DataFrame(
        {
            "source_supernode_id": coo.row.astype(np.int32),
            "target_supernode_id": coo.col.astype(np.int32),
            "row_normalized_probability": coo.data.astype(np.float64),
        }
    )
    meta = sizes[["supernode_id", "time_label", "time_day", "supernode_size", "is_final_time"]].copy()
    edge_table = edge_table.merge(meta.add_prefix("source_"), on="source_supernode_id", how="left")
    edge_table = edge_table.merge(meta.add_prefix("target_"), on="target_supernode_id", how="left")
    qc = {
        "p_forward_shape": list(p_forward.shape),
        "p_forward_nnz": int(p_forward.nnz),
        "p_super_shape": list(normalized.shape),
        "p_super_nnz": int(normalized.nnz),
        "p_super_dtype": str(normalized.dtype),
        "row_sum_error_max": max_error,
        "zero_outgoing_supernode_count": int(len(zero_rows)),
        "zero_outgoing_supernode_fraction": float(len(zero_rows) / n_super) if n_super else 0.0,
        "final_time_zero_outgoing_supernode_count": int(len(final_zero)),
        "nonfinal_zero_outgoing_supernode_count": int(len(nonfinal_zero)),
        "final_time_zero_outgoing_supernodes": final_zero[:100],
        "nonfinal_zero_outgoing_supernodes": nonfinal_zero[:100],
        "structural_closure_note": "Self-loops were added only to zero-outgoing supernode rows for row-stochastic pyGPCCA input; this is not terminal-state inference and not absorption probability.",
    }
    return normalized.tocsr(), edge_table, qc


def load_or_build_p_super(
    config: dict[str, Any],
    paths: M4DPaths,
    assignments: pd.DataFrame,
    sizes: pd.DataFrame,
    resume: bool,
    overwrite: bool,
) -> tuple[sp.csr_matrix, pd.DataFrame, dict[str, Any]]:
    if resume and paths.p_super.exists() and paths.edge_table.exists():
        p_super = sp.load_npz(paths.p_super).tocsr()
        edge_table = pd.read_csv(paths.edge_table)
        row_sums = np.asarray(p_super.sum(axis=1)).ravel()
        if p_super.shape == (len(sizes), len(sizes)) and float(np.max(np.abs(row_sums - 1.0))) <= float(config["standard_gpcca"]["row_sum_tolerance"]):
            row_error = float(np.max(np.abs(row_sums - 1.0))) if len(row_sums) else 0.0
            row_nnz = np.diff(p_super.indptr)
            diag = p_super.diagonal()
            closure_rows = np.where((row_nnz == 1) & np.isclose(diag, 1.0))[0]
            final_mask = sizes.sort_values("supernode_id")["is_final_time"].to_numpy(dtype=bool)
            final_zero = [int(row) for row in closure_rows if final_mask[int(row)]]
            nonfinal_zero = [int(row) for row in closure_rows if not final_mask[int(row)]]
            return (
                p_super,
                edge_table,
                {
                    "resumed_p_super": True,
                    "p_super_shape": list(p_super.shape),
                    "p_super_nnz": int(p_super.nnz),
                    "p_super_dtype": str(p_super.dtype),
                    "row_sum_error_max": row_error,
                    "zero_outgoing_supernode_count": int(len(closure_rows)),
                    "zero_outgoing_supernode_fraction": float(len(closure_rows) / p_super.shape[0]) if p_super.shape[0] else 0.0,
                    "final_time_zero_outgoing_supernode_count": int(len(final_zero)),
                    "nonfinal_zero_outgoing_supernode_count": int(len(nonfinal_zero)),
                    "final_time_zero_outgoing_supernodes": final_zero[:100],
                    "nonfinal_zero_outgoing_supernodes": nonfinal_zero[:100],
                    "zero_row_counts_inferred_from_existing_structural_self_loops": True,
                    "structural_closure_note": "Self-loops were added only to zero-outgoing supernode rows for row-stochastic pyGPCCA input; this is not terminal-state inference and not absorption probability.",
                },
            )
    p_super, edge_table, qc = aggregate_p_super(paths.p_forward, assignments, sizes, float(config["standard_gpcca"]["row_sum_tolerance"]))
    atomic_save_sparse_npz(paths.p_super, p_super, overwrite=overwrite)
    atomic_write_csv(paths.edge_table, edge_table, overwrite=overwrite)
    return p_super, edge_table, qc


def component_review(p_super: sp.csr_matrix, sizes: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    graph = ((p_super != 0) + (p_super.T != 0)).astype(bool).astype(np.int8)
    n_components, labels = connected_components(graph, directed=False, return_labels=True)
    counts = np.bincount(labels, minlength=n_components)
    largest = int(counts.argmax()) if len(counts) else 0
    summary = pd.DataFrame(
        {
            "component_id": np.arange(n_components, dtype=np.int32),
            "component_size": counts.astype(np.int64),
            "is_largest_component": np.arange(n_components) == largest,
        }
    ).sort_values(["component_size", "component_id"], ascending=[False, True])
    largest_nodes = np.where(labels == largest)[0].astype(np.int64)
    excluded = sizes.loc[~sizes["supernode_id"].isin(largest_nodes)].copy()
    excluded["component_id"] = labels[excluded["supernode_id"].to_numpy(dtype=int)] if len(excluded) else []
    qc = {
        "n_components": int(n_components),
        "largest_component_id": largest,
        "largest_component_size": int(counts[largest]) if len(counts) else 0,
        "largest_component_fraction": float(counts[largest] / p_super.shape[0]) if p_super.shape[0] else 0.0,
        "small_component_count": int((counts < 10).sum()),
        "run_scope": "full_p_super" if int(n_components) == 1 else "largest_component",
    }
    return largest_nodes, summary, excluded, qc


def write_component_reports(paths: M4DPaths, summary: pd.DataFrame, excluded: pd.DataFrame, qc: dict[str, Any], overwrite: bool) -> None:
    atomic_write_csv(paths.component_csv, summary, overwrite=overwrite)
    if len(excluded):
        atomic_write_csv(paths.component_excluded, excluded, overwrite=overwrite)
    lines = [
        "# M4D Supernode Component Report",
        "",
        f"- generated at UTC: {utc_now_iso()}",
        f"- components: `{qc['n_components']}`",
        f"- largest component size: `{qc['largest_component_size']}`",
        f"- largest component fraction: `{qc['largest_component_fraction']:.6f}`",
        f"- small component count: `{qc['small_component_count']}`",
        f"- pyGPCCA run scope: `{qc['run_scope']}`",
        "",
        "If the largest component is not effectively all supernodes, pyGPCCA is run on the largest component and excluded supernodes are recorded separately.",
    ]
    atomic_write_text(paths.component_md, "\n".join(lines) + "\n", overwrite=overwrite)


def restrict_to_component(p_super: sp.csr_matrix, component_indices: np.ndarray) -> sp.csr_matrix:
    sub = p_super[component_indices, :][:, component_indices].tocsr().astype(np.float64)
    row_sums = np.asarray(sub.sum(axis=1)).ravel()
    nonzero = row_sums > 0
    if nonzero.any():
        sub.data *= np.repeat(np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=nonzero), np.diff(sub.indptr))
    zero = np.where(row_sums <= 0)[0]
    if len(zero):
        sub = (sub + sp.csr_matrix((np.ones(len(zero)), (zero, zero)), shape=sub.shape)).tocsr()
    return sub


def set_gpcca_tmp_env() -> None:
    tmp = Path("/tmp/nichefate_m4d01c_gpcca_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for key in TEMP_ENV_KEYS:
        os.environ[key] = str(tmp)


def run_pygpcca_candidate(matrix: sp.csr_matrix, k: int, method: str) -> dict[str, Any]:
    set_gpcca_tmp_env()
    import pygpcca

    result: dict[str, Any] = {"k": int(k), "success": False, "error": ""}
    try:
        gpcca = pygpcca.GPCCA(matrix, z="LM", method=method)
        gpcca.optimize(int(k))
        memberships = np.asarray(getattr(gpcca, "memberships"), dtype=np.float64)
        assignments = np.asarray(getattr(gpcca, "macrostate_assignment"), dtype=np.int32)
        coarse = np.asarray(getattr(gpcca, "coarse_grained_transition_matrix"), dtype=np.float64)
        eigenvalues = getattr(gpcca, "eigenvalues", None)
        if memberships.ndim != 2 or assignments.ndim != 1 or coarse.ndim != 2:
            raise ValueError("pyGPCCA returned invalid memberships, assignments, or coarse matrix shape.")
        if memberships.shape[0] != matrix.shape[0] or assignments.shape[0] != matrix.shape[0]:
            raise ValueError("pyGPCCA output row count does not match input matrix.")
        unique = sorted(int(x) for x in np.unique(assignments))
        remap = {old: new for new, old in enumerate(unique)}
        dense_assign = np.asarray([remap[int(x)] for x in assignments], dtype=np.int32)
        counts = np.bincount(dense_assign, minlength=len(unique))
        metastability = float(np.trace(coarse) / coarse.shape[0]) if coarse.size else 0.0
        result.update(
            {
                "success": True,
                "memberships": memberships,
                "assignments": dense_assign,
                "coarse": coarse,
                "eigenvalues": np.asarray(eigenvalues).tolist() if eigenvalues is not None else [],
                "observed_k": int(memberships.shape[1]),
                "macrostate_counts": counts.astype(int).tolist(),
                "min_macrostate_size": int(counts.min()) if len(counts) else 0,
                "max_macrostate_size": int(counts.max()) if len(counts) else 0,
                "min_macrostate_fraction": float(counts.min() / matrix.shape[0]) if len(counts) else 0.0,
                "metastability": metastability,
                "crispness_mean_max_membership": float(np.max(memberships, axis=1).mean()) if len(memberships) else 0.0,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def gpcca_candidate_table(results: list[dict[str, Any]], n_states: int) -> tuple[pd.DataFrame, int]:
    rows = []
    min_size = max(10, int(np.ceil(0.001 * n_states)))
    for result in results:
        success = bool(result.get("success", False))
        nondegenerate = bool(success and result.get("min_macrostate_size", 0) >= min_size)
        rows.append(
            {
                "k": int(result["k"]),
                "success": success,
                "error": str(result.get("error", "")),
                "observed_k": int(result.get("observed_k", 0)),
                "min_macrostate_size": int(result.get("min_macrostate_size", 0)),
                "max_macrostate_size": int(result.get("max_macrostate_size", 0)),
                "min_macrostate_fraction": float(result.get("min_macrostate_fraction", 0.0)),
                "metastability": float(result.get("metastability", 0.0)),
                "crispness_mean_max_membership": float(result.get("crispness_mean_max_membership", 0.0)),
                "nondegenerate": nondegenerate,
            }
        )
    table = pd.DataFrame(rows)
    return table, min_size


def select_best_candidate(results: list[dict[str, Any]], n_states: int) -> dict[str, Any]:
    table, min_size = gpcca_candidate_table(results, n_states)
    passing = table.loc[table["nondegenerate"].astype(bool)].copy()
    if passing.empty:
        raise RuntimeError("No non-degenerate pyGPCCA candidate k succeeded.")
    selected_row = passing.sort_values(["metastability", "min_macrostate_fraction", "k"], ascending=[False, False, True]).iloc[0]
    selected_k = int(selected_row["k"])
    for result in results:
        if int(result["k"]) == selected_k:
            return {"selected": result, "table": table, "min_size_threshold": min_size}
    raise AssertionError("selected candidate disappeared")


def write_gpcca_failure_reports(
    paths: M4DPaths,
    results: list[dict[str, Any]],
    n_states: int,
    component_qc: dict[str, Any],
    error: Exception,
    overwrite: bool,
) -> None:
    table, min_size = gpcca_candidate_table(results, n_states)
    summary = {
        "generated_at_utc": utc_now_iso(),
        "status": "failed",
        "backend": "pygpcca",
        "configured_method": "krylov",
        "input_shape": [int(n_states), int(n_states)],
        "component_qc": component_qc,
        "min_size_threshold": int(min_size),
        "candidate_table": table.to_dict("records"),
        "failure_error": f"{type(error).__name__}: {error}",
        "failure_interpretation": "standard pyGPCCA krylov failed before producing a non-degenerate macrostate membership solution; no M4D GPCCA macrostate projection was written.",
        "no_absorption_probability": True,
        "no_fate_probability": True,
        "no_full_node_gpcca": True,
    }
    atomic_write_json(paths.gpcca_summary, summary, overwrite=overwrite)
    lines = [
        "# M4D-01c Standard pyGPCCA Report",
        "",
        f"- generated at UTC: {summary['generated_at_utc']}",
        "- status: `failed`",
        "- backend: `pygpcca`",
        "- configured method: `krylov`",
        f"- input shape: `{summary['input_shape']}`",
        f"- failure: {summary['failure_error']}",
        "- no M4D GPCCA membership matrix was accepted.",
        "- no node-level GPCCA macrostate projection was written.",
        "- no absorption probability or GPCCA-derived fate probability was computed.",
        "",
        "## Candidate k Table",
        "",
        "| k | success | nondegenerate | metastability | min size | error |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary["candidate_table"]:
        lines.append(
            f"| {row['k']} | {row['success']} | {row['nondegenerate']} | "
            f"{float(row['metastability']):.6f} | {row['min_macrostate_size']} | {str(row['error']).replace('|', ';')} |"
        )
    atomic_write_text(paths.gpcca_report, "\n".join(lines) + "\n", overwrite=overwrite)


def run_gpcca(
    config: dict[str, Any],
    p_super: sp.csr_matrix,
    sizes: pd.DataFrame,
    paths: M4DPaths,
    resume: bool,
    overwrite: bool,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    existing_ok, existing_msg = validate_existing_outputs(paths, expected_nodes=None)
    if resume and existing_ok:
        return np.load(paths.memberships), pd.read_csv(paths.macrostate_assignments), {"resumed_gpcca": True, "resume_message": existing_msg}
    print("[M4D-01c] reviewing P_super weak connected components", flush=True)
    largest_nodes, comp_summary, excluded, comp_qc = component_review(p_super, sizes)
    write_component_reports(paths, comp_summary, excluded, comp_qc, overwrite=overwrite)
    run_indices = np.arange(p_super.shape[0], dtype=np.int64) if comp_qc["run_scope"] == "full_p_super" else largest_nodes
    matrix = p_super if comp_qc["run_scope"] == "full_p_super" else restrict_to_component(p_super, run_indices)
    print(f"[M4D-01c] running pyGPCCA on {matrix.shape[0]} supernodes", flush=True)
    results = [
        run_pygpcca_candidate(matrix, int(k), str(config["standard_gpcca"].get("pygpcca_method", "krylov")))
        for k in config["standard_gpcca"]["candidate_n_macrostates"]
    ]
    try:
        selected_bundle = select_best_candidate(results, matrix.shape[0])
    except Exception as exc:
        write_gpcca_failure_reports(paths, results, matrix.shape[0], comp_qc, exc, overwrite=overwrite)
        raise
    selected = selected_bundle["selected"]
    selected_memberships = selected["memberships"]
    selected_assignments = selected["assignments"]
    selected_k = int(selected["observed_k"])
    full_memberships = np.zeros((p_super.shape[0], selected_k), dtype=np.float32)
    full_assignments = np.full(p_super.shape[0], -1, dtype=np.int32)
    full_memberships[run_indices] = selected_memberships.astype(np.float32, copy=False)
    full_assignments[run_indices] = selected_assignments.astype(np.int32, copy=False)
    excluded_component_column = False
    if len(run_indices) != p_super.shape[0]:
        excluded_component_column = True
        sentinel = np.zeros((p_super.shape[0], 1), dtype=np.float32)
        excluded_mask = full_assignments < 0
        sentinel[excluded_mask, 0] = 1.0
        full_memberships = np.hstack([full_memberships, sentinel])
    macro_rows = []
    size_by_super = sizes.set_index("supernode_id")["supernode_size"]
    for sid in range(p_super.shape[0]):
        probs = full_memberships[sid]
        macro_id = int(full_assignments[sid])
        if macro_id < 0 and excluded_component_column:
            top_probability = 1.0
            component_status = "excluded_component"
        else:
            top_probability = float(probs[macro_id]) if macro_id >= 0 else 0.0
            component_status = "gpcca_component"
        macro_rows.append(
            {
                "supernode_id": sid,
                "gpcca_macrostate_id": macro_id,
                "gpcca_macrostate_probability": top_probability,
                "supernode_size": int(size_by_super.loc[sid]),
                "component_status": component_status,
            }
        )
    macro = pd.DataFrame(macro_rows).merge(sizes, on=["supernode_id", "supernode_size"], how="left")
    atomic_write_json(
        paths.selected_k,
        {
            "selected_k": selected_k,
            "candidate_table": selected_bundle["table"].to_dict("records"),
            "selection_rationale": "ranked by metastability, then minimum macrostate fraction, then smaller k",
            "excluded_component_column": excluded_component_column,
            "terminology": "M4D-01c outputs are GPCCA macrostate memberships and macrostate projections, not fate probabilities.",
        },
        overwrite=overwrite,
    )
    atomic_save_np(paths.memberships, full_memberships, overwrite=overwrite)
    atomic_write_csv(paths.macrostate_assignments, macro, overwrite=overwrite)
    atomic_save_np(paths.coarse_transition, selected["coarse"].astype(np.float64, copy=False), overwrite=overwrite)
    summary = {
        "generated_at_utc": utc_now_iso(),
        "backend": "pygpcca",
        "input_shape": list(matrix.shape),
        "run_scope": comp_qc["run_scope"],
        "component_qc": comp_qc,
        "selected_k": selected_k,
        "candidate_table": selected_bundle["table"].to_dict("records"),
        "selected_metastability": float(selected["metastability"]),
        "selected_min_macrostate_size": int(selected["min_macrostate_size"]),
        "selected_min_macrostate_fraction": float(selected["min_macrostate_fraction"]),
        "excluded_component_column": excluded_component_column,
        "no_absorption_probability": True,
        "no_fate_probability": True,
        "no_full_node_gpcca": True,
    }
    atomic_write_json(paths.gpcca_summary, summary, overwrite=overwrite)
    atomic_write_text(paths.gpcca_report, gpcca_report_markdown(summary), overwrite=overwrite)
    return full_memberships, macro, summary


def membership_entropy(matrix: np.ndarray) -> np.ndarray:
    clipped = np.clip(matrix.astype(np.float64, copy=False), 1e-300, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    denom = np.log(matrix.shape[1]) if matrix.shape[1] > 1 else 1.0
    return (entropy / denom).astype(np.float32, copy=False)


def project_gpcca_to_nodes(
    assignments: pd.DataFrame,
    memberships: np.ndarray,
    macro: pd.DataFrame,
    paths: M4DPaths,
    overwrite: bool,
) -> pd.DataFrame:
    table = assignments[["global_node_index", "supernode_id"]].copy().sort_values("global_node_index", kind="mergesort")
    expected = np.arange(len(table), dtype=np.int64)
    if not np.array_equal(table["global_node_index"].to_numpy(dtype=np.int64), expected):
        raise ValueError("Cannot project GPCCA memberships because global_node_index identity is not aligned.")
    macro = macro.sort_values("supernode_id", kind="mergesort").reset_index(drop=True)
    super_ids = table["supernode_id"].to_numpy(dtype=np.int64)
    node_probs = memberships[super_ids]
    dominant = macro.loc[super_ids, "gpcca_macrostate_id"].to_numpy(dtype=np.int32)
    top_prob = macro.loc[super_ids, "gpcca_macrostate_probability"].to_numpy(dtype=np.float32)
    projected = table.copy()
    projected["gpcca_macrostate_id"] = dominant
    projected["gpcca_macrostate_probability"] = top_prob
    projected["gpcca_macrostate_probabilities"] = [row.astype(np.float32).tolist() for row in node_probs]
    projected["gpcca_membership_entropy"] = membership_entropy(node_probs)
    for col in range(node_probs.shape[1]):
        projected[f"gpcca_prob_{col:02d}"] = node_probs[:, col].astype(np.float32, copy=False)
    if bool(projected["gpcca_macrostate_id"].isna().any()):
        raise ValueError("Projected GPCCA macrostate has missing values.")
    if not np.isfinite(node_probs).all():
        raise ValueError("Projected GPCCA probabilities contain non-finite values.")
    row_sums = node_probs.sum(axis=1)
    if float(np.max(np.abs(row_sums - 1.0))) > 1e-4:
        raise ValueError("Projected GPCCA membership rows do not sum to 1.")
    atomic_write_parquet(paths.node_projection, projected, overwrite=overwrite)
    return projected


def gpcca_report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# M4D-01c Standard pyGPCCA Report",
        "",
        f"- generated at UTC: {summary['generated_at_utc']}",
        "- terminology: M4D-01c outputs are GPCCA macrostate memberships / macrostate projections.",
        "- M4D-01c outputs are not fate probabilities and not absorption probabilities.",
        f"- backend: `{summary['backend']}`",
        f"- input shape: `{summary['input_shape']}`",
        f"- run scope: `{summary['run_scope']}`",
        f"- selected k: `{summary['selected_k']}`",
        f"- selected metastability: `{summary['selected_metastability']:.6f}`",
        f"- selected minimum macrostate size: `{summary['selected_min_macrostate_size']}`",
        f"- no full-node GPCCA: `{summary['no_full_node_gpcca']}`",
        f"- no GPCCA-derived absorption probability: `{summary['no_absorption_probability']}`",
        f"- no GPCCA-derived fate probability: `{summary['no_fate_probability']}`",
        "",
        "## Candidate k Table",
        "",
        "| k | success | nondegenerate | metastability | min size | error |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary["candidate_table"]:
        lines.append(
            f"| {row['k']} | {row['success']} | {row['nondegenerate']} | "
            f"{float(row['metastability']):.6f} | {row['min_macrostate_size']} | {str(row['error']).replace('|', ';')} |"
        )
    return "\n".join(lines) + "\n"


def write_supernode_report(paths: M4DPaths, super_meta: dict[str, Any], p_qc: dict[str, Any], overwrite: bool) -> None:
    report = {
        "schema_version": "m4d_01c_supernode_qc_v1",
        "generated_at_utc": utc_now_iso(),
        **super_meta,
        **p_qc,
        "zero_row_closure_policy": "Final-time zero rows are expected; non-final zero rows are warnings. Any zero row receives a structural self-loop only to make P_super row-stochastic for pyGPCCA.",
        "not_absorption_probability": True,
    }
    atomic_write_json(paths.reports_dir / "m4d_supernode_qc_summary.json", report, overwrite=overwrite)
    lines = [
        "# M4D-01c Supernode Markov QC",
        "",
        f"- generated at UTC: {report['generated_at_utc']}",
        f"- supernodes: `{report.get('n_supernodes')}`",
        f"- assigned nodes: `{report.get('n_nodes')}`",
        f"- feature-group source: `{report.get('feature_group_source', 'resumed or unavailable')}`",
        f"- P_super shape: `{report.get('p_super_shape')}`",
        f"- P_super nnz: `{report.get('p_super_nnz')}`",
        f"- P_super dtype: `{report.get('p_super_dtype')}`",
        f"- row-sum max error: `{report.get('row_sum_error_max')}`",
        f"- final-time zero-outgoing supernodes: `{report.get('final_time_zero_outgoing_supernode_count')}`",
        f"- non-final zero-outgoing supernodes: `{report.get('nonfinal_zero_outgoing_supernode_count')}`",
        "",
        "Structural self-loop closure is row-stochastic input preparation for pyGPCCA. It is not terminal-state inference and not absorption probability.",
    ]
    atomic_write_text(paths.reports_dir / "m4d_supernode_markov_report.md", "\n".join(lines) + "\n", overwrite=overwrite)


def run_m4d_01c(config: dict[str, Any], resume: bool = False, overwrite: bool = False) -> dict[str, Any]:
    if not config.get("standard_gpcca", {}).get("enabled", False):
        raise ValueError("standard_gpcca.enabled must be true to run M4D-01c.")
    if str(config["standard_gpcca"].get("backend", "")) != "pygpcca":
        raise ValueError("M4D-01c requires standard_gpcca.backend: pygpcca.")
    parquet_status = require_parquet_engine()
    paths = m4d_paths(config)
    prepare_output_dirs(paths)
    expected_nodes = int(config.get("coordinates", {}).get("expected_global_nodes", 0)) or None
    existing_ok, existing_msg = validate_existing_outputs(paths, expected_nodes=expected_nodes)
    if resume and existing_ok:
        return {
            "status": "resumed_existing_outputs",
            "message": existing_msg,
            "paths": json_safe(paths.__dict__),
            "parquet_status": parquet_status,
        }
    assignments, sizes, super_meta = load_or_build_supernodes(config, paths, resume=resume, overwrite=overwrite)
    p_super, edge_table, p_qc = load_or_build_p_super(config, paths, assignments, sizes, resume=resume, overwrite=overwrite)
    write_supernode_report(paths, super_meta, p_qc, overwrite=overwrite)
    memberships, macro, gpcca_summary = run_gpcca(config, p_super, sizes, paths, resume=resume, overwrite=overwrite)
    projected = project_gpcca_to_nodes(assignments, memberships, macro, paths, overwrite=overwrite)
    result = {
        "status": "completed",
        "parquet_status": parquet_status,
        "supernode_count": int(len(sizes)),
        "node_projection_rows": int(len(projected)),
        "p_super_shape": list(p_super.shape),
        "p_super_nnz": int(p_super.nnz),
        "p_super_dtype": str(p_super.dtype),
        "p_super_row_sum_error_max": float(np.max(np.abs(np.asarray(p_super.sum(axis=1)).ravel() - 1.0))),
        "component_review": gpcca_summary.get("component_qc", {}),
        "selected_k": int(gpcca_summary.get("selected_k", 0)),
        "output_paths": json_safe(paths.__dict__),
        "explicit_confirmations": {
            "no_full_node_gpcca": True,
            "no_gpcca_derived_absorption_fate_probability": True,
            "no_m4c_fate_recomputation": True,
            "no_branched_nicheflow_or_branchsbm": True,
            "no_m5_or_regulator": True,
            "no_barcode_preprocessing": True,
            "no_ssd": True,
        },
    }
    atomic_write_json(paths.reports_dir / "m4d_01c_run_summary.json", result, overwrite=overwrite)
    return result
