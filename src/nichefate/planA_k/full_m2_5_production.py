"""Full M2.5 metaniche production runner helpers for PlanA-K.

The default execution path is dry-run/report-only. Full production requires an
explicit ``--no-dry-run`` from the CLI wrapper and writes only to the approved
PlanA-K scratch root.
"""

from __future__ import annotations

import getpass
import json
import math
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from .io import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_tsv,
    disk_usage,
    ensure_dir,
    git_status_short,
    json_safe,
    read_memory_info,
    utc_now,
)
from .reporting import dataframe_to_markdown
from .schemas import (
    M1_BY_SLICE_ROOT,
    M2_BY_SLICE_ROOT,
    PLAN_A_K_PRODUCTION_SCRATCH_ROOT,
    PROJECT_ROOT,
)


FULL_M2_5_REPORT_ROOT = PROJECT_ROOT / "reports" / "planA_k_full_m2_5_implementation"
FULL_M2_5_PRODUCTION_ROOT = PLAN_A_K_PRODUCTION_SCRATCH_ROOT / "full_m2_5"
DEFAULT_TMP_ROOT = Path("/home/zhutao/tmp/nichefate_planA_k")
JOIN_KEYS = ["slice_id", "anchor_index", "anchor_cell_id"]
COORDINATE_COLUMNS = [*JOIN_KEYS, "x", "y"]
REQUIRED_OUTPUTS = [
    "run_manifest.json",
    "feature_lock.used.json",
    "feature_columns.txt",
    "scaler.joblib",
    "pca.joblib",
    "pca_components.npy",
    "training_sample_manifest.tsv",
    "anchor_to_metaniche.parquet",
    "metaniche_table.parquet",
    "metaniche_feature_centroids.parquet",
    "metaniche_coordinates.tsv",
    "metaniche_composition.tsv",
    "metaniche_qc.json",
    "full_m2_5_summary.md",
    "full_m2_5_summary.json",
]


@dataclass(frozen=True)
class FullM25Params:
    feature_lock: Path
    output_root: Path
    seed: int
    m1_root: Path = M1_BY_SLICE_ROOT
    m2_root: Path = M2_BY_SLICE_ROOT
    dry_run: bool = True
    smoke_test: bool = False
    max_slices: int | None = None
    max_anchors_per_slice: int | None = None
    overwrite: bool = False
    resume: bool = False
    n_pca_components: int = 30
    target_mode: str = "adaptive"
    min_metaniches_per_slice: int = 50
    max_metaniches_per_slice: int = 150
    base_metaniches_per_slice: int = 100
    tmp_dir: Path = DEFAULT_TMP_ROOT


def path_has_ssd(path: Path) -> bool:
    return any(part == "ssd" for part in path.resolve().parts)


def validate_no_ssd_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if path_has_ssd(resolved):
        raise ValueError(f"Refusing /ssd path for {label}: {resolved}")
    return resolved


def load_feature_lock(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Feature lock not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    feature_columns = list(data.get("feature_columns", []))
    if int(data.get("feature_column_count", len(feature_columns))) != len(feature_columns):
        raise ValueError("Feature lock feature_column_count does not match feature_columns length.")
    if len(feature_columns) != 600 and "full_m2_5_feature_lock" in path.name:
        raise ValueError(f"Expected 600 locked features for full production, found {len(feature_columns)}.")
    if any(column in set(data.get("metadata_columns", [])) for column in feature_columns):
        raise ValueError("Feature lock includes metadata columns as features.")
    if "/ssd" in json.dumps(data):
        raise ValueError("Feature lock contains a /ssd path.")
    return data


def discover_slice_inputs(m1_root: Path, m2_root: Path) -> pd.DataFrame:
    m2_files = {
        path.parent.name: path
        for path in sorted(m2_root.glob("*/m2_representation_*.parquet"))
    }
    rows = []
    for slice_id, m2_path in m2_files.items():
        m1_path = m1_root / slice_id / f"niche_features_{slice_id}.parquet"
        m2_meta = pq.ParquetFile(m2_path).metadata
        rows.append(
            {
                "slice_id": slice_id,
                "m2_path": str(m2_path),
                "m1_path": str(m1_path),
                "m1_exists": m1_path.exists(),
                "m2_rows": int(m2_meta.num_rows),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("slice_id").reset_index(drop=True)
    return frame


def select_slices_for_run(inventory: pd.DataFrame, params: FullM25Params) -> pd.DataFrame:
    selected = inventory.copy()
    if params.smoke_test:
        selected = selected.head(min(params.max_slices or 2, 2))
    elif params.max_slices is not None:
        selected = selected.head(params.max_slices)
    return selected.reset_index(drop=True)


def validate_output_root(params: FullM25Params) -> Path:
    root = validate_no_ssd_path(params.output_root, "output-root")
    validate_no_ssd_path(params.tmp_dir, "tmp-dir")
    if params.resume and params.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")
    if not params.dry_run and not params.smoke_test:
        expected = FULL_M2_5_PRODUCTION_ROOT.resolve()
        if root != expected:
            raise ValueError(f"Full production output root must be {expected}, got {root}.")
    if params.smoke_test and root == FULL_M2_5_PRODUCTION_ROOT.resolve():
        raise ValueError("Smoke-test must not write to the production output root.")
    if root.exists() and any(root.iterdir()) and not params.overwrite and not params.resume and not params.dry_run:
        raise FileExistsError(f"{root} is non-empty; pass --overwrite or --resume.")
    return root


def validate_feature_schema(path: Path, required_columns: list[str]) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    missing = [column for column in required_columns if column not in names]
    return {
        "path": str(path),
        "row_count": int(parquet.metadata.num_rows),
        "column_count": len(names),
        "missing_columns": missing,
        "valid": not missing,
    }


def read_parquet_sample(path: Path, columns: list[str], max_rows: int | None) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    available = [column for column in columns if column in parquet.schema_arrow.names]
    if max_rows is None:
        return pq.read_table(path, columns=available).to_pandas()
    chunks = []
    remaining = max_rows
    for batch in parquet.iter_batches(columns=available, batch_size=min(8192, max(1, remaining))):
        frame = batch.to_pandas()
        if len(frame) > remaining:
            frame = frame.head(remaining)
        chunks.append(frame)
        remaining -= len(frame)
        if remaining <= 0:
            break
    if not chunks:
        return pd.DataFrame(columns=available)
    return pd.concat(chunks, ignore_index=True)


def read_m1_coordinates(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    columns = [*JOIN_KEYS, "scale", "x", "y"]
    frame = read_parquet_sample(path, columns, None if max_rows is None else max_rows * 4)
    if "scale" in frame.columns and "radius_x2" in set(frame["scale"].astype(str)):
        frame = frame[frame["scale"].astype(str) == "radius_x2"].copy()
    if max_rows is not None:
        frame = frame.head(max_rows).copy()
    return frame[[column for column in [*JOIN_KEYS, "x", "y"] if column in frame.columns]]


def validate_coordinate_join_for_slice(
    m2_path: Path,
    m1_path: Path,
    max_rows: int | None = None,
) -> dict[str, Any]:
    if not m1_path.exists():
        return {
            "valid": False,
            "join_coverage": 0.0,
            "duplicate_join_key_rows": 0,
            "reason": f"M1 coordinate file missing: {m1_path}",
        }
    m2_keys = read_parquet_sample(m2_path, JOIN_KEYS, max_rows).dropna(subset=JOIN_KEYS)
    coords = read_m1_coordinates(m1_path, max_rows=max_rows).dropna(subset=JOIN_KEYS)
    m2_dupes = int(m2_keys.duplicated(JOIN_KEYS).sum())
    coord_dupes = int(coords.duplicated(JOIN_KEYS).sum())
    if m2_dupes or coord_dupes:
        return {
            "valid": False,
            "join_coverage": 0.0,
            "duplicate_join_key_rows": m2_dupes + coord_dupes,
            "reason": "duplicate join keys",
        }
    joined = m2_keys.merge(coords, on=JOIN_KEYS, how="left", indicator=True)
    coverage = float((joined["_merge"] == "both").mean()) if len(joined) else 0.0
    return {
        "valid": coverage >= 0.999,
        "m2_rows_checked": int(len(m2_keys)),
        "coordinate_rows_checked": int(len(coords)),
        "join_coverage": coverage,
        "duplicate_join_key_rows": 0,
        "missing_coordinate_rows": int((joined["_merge"] != "both").sum()) if len(joined) else 0,
        "reason": "" if coverage >= 0.999 else "join coverage below 99.9%",
    }


def validate_coordinate_joins(
    selected: pd.DataFrame,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for item in selected.to_dict(orient="records"):
        result = validate_coordinate_join_for_slice(
            Path(str(item["m2_path"])),
            Path(str(item["m1_path"])),
            max_rows=max_rows,
        )
        result["slice_id"] = item["slice_id"]
        rows.append(result)
    frame = pd.DataFrame(rows)
    summary = {
        "slice_count": int(len(frame)),
        "all_valid": bool(frame["valid"].all()) if not frame.empty else False,
        "min_join_coverage": float(frame["join_coverage"].min()) if not frame.empty else 0.0,
        "duplicate_join_key_rows": int(frame["duplicate_join_key_rows"].sum()) if not frame.empty else 0,
        "blockers": frame.loc[~frame["valid"], "reason"].dropna().astype(str).unique().tolist()
        if not frame.empty
        else ["no slices selected"],
    }
    return frame, summary


def adaptive_metaniche_count(
    slice_anchor_count: int,
    mean_slice_anchor_count: float,
    base: int = 100,
    min_count: int = 50,
    max_count: int = 150,
    target_mode: str = "adaptive",
) -> int:
    if target_mode == "fixed":
        target = base
    elif target_mode == "adaptive":
        ratio = slice_anchor_count / max(mean_slice_anchor_count, 1.0)
        target = int(round(base * math.sqrt(ratio)))
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")
    return int(max(min_count, min(max_count, target)))


def planned_slice_counts(selected: pd.DataFrame, params: FullM25Params) -> pd.DataFrame:
    mean_rows = float(selected["m2_rows"].mean()) if not selected.empty else 0.0
    rows = []
    for row in selected.to_dict(orient="records"):
        source_rows = int(row["m2_rows"])
        rows.append(
            {
                "slice_id": row["slice_id"],
                "source_anchor_count": source_rows,
                "planned_anchor_count": min(source_rows, params.max_anchors_per_slice)
                if params.max_anchors_per_slice is not None
                else source_rows,
                "planned_metaniche_count": adaptive_metaniche_count(
                    slice_anchor_count=source_rows,
                    mean_slice_anchor_count=mean_rows,
                    base=params.base_metaniches_per_slice,
                    min_count=params.min_metaniches_per_slice,
                    max_count=params.max_metaniches_per_slice,
                    target_mode=params.target_mode,
                ),
            }
        )
    return pd.DataFrame(rows)


def prepare_numeric(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.fillna(numeric.median(axis=0)).fillna(0.0)


def sample_training_rows(
    selected: pd.DataFrame,
    feature_columns: list[str],
    max_rows_per_slice: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunks = []
    manifest = []
    for order, row in enumerate(selected.to_dict(orient="records")):
        sample_n = max_rows_per_slice or min(int(row["m2_rows"]), 5000)
        frame = read_parquet_sample(Path(str(row["m2_path"])), feature_columns, sample_n)
        chunks.append(prepare_numeric(frame, feature_columns))
        manifest.append(
            {
                "slice_id": row["slice_id"],
                "source_m2_path": row["m2_path"],
                "sampled_rows": int(len(frame)),
                "seed": seed + order,
            }
        )
    if not chunks:
        return pd.DataFrame(columns=feature_columns), pd.DataFrame(manifest)
    return pd.concat(chunks, ignore_index=True), pd.DataFrame(manifest)


def fit_scaler_pca(
    training: pd.DataFrame,
    n_components: int,
    batch_size: int = 4096,
) -> tuple[StandardScaler, IncrementalPCA, np.ndarray]:
    matrix = training.to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.partial_fit(matrix)
    scaled = scaler.transform(matrix)
    effective = min(n_components, scaled.shape[1], max(1, scaled.shape[0] - 1))
    pca = IncrementalPCA(n_components=effective, batch_size=min(batch_size, max(1, len(scaled))))
    pca.fit(scaled)
    return scaler, pca, pca.transform(scaled)


def load_slice_frame(
    row: dict[str, Any],
    feature_columns: list[str],
    metadata_columns: list[str],
    max_rows: int | None,
) -> pd.DataFrame:
    columns = [column for column in [*metadata_columns, *feature_columns] if column]
    frame = read_parquet_sample(Path(str(row["m2_path"])), columns, max_rows)
    if "anchor_id" not in frame.columns and {"slice_id", "anchor_index"}.issubset(frame.columns):
        frame["anchor_id"] = frame["slice_id"].astype(str) + "::" + frame["anchor_index"].astype(str)
    return frame


def coarsen_slice(
    frame: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler,
    pca: IncrementalPCA,
    n_clusters: int,
    seed: int,
    slice_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = prepare_numeric(frame, feature_columns)
    reduced = pca.transform(scaler.transform(numeric.to_numpy(dtype=np.float32)))
    k = min(max(1, n_clusters), max(1, len(frame)))
    labels = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed + slice_index,
        batch_size=min(4096, max(1, len(frame))),
        n_init=5,
    ).fit_predict(reduced)
    assigned = frame.copy()
    assigned["metaniche_id"] = [
        f"{assigned['slice_id'].iloc[0]}::MN{int(label):04d}" for label in labels
    ]
    mapping_cols = [
        column
        for column in [
            "anchor_id",
            "metaniche_id",
            "slice_id",
            "time",
            "time_day",
            "mouse_id",
            "anchor_index",
            "anchor_cell_id",
            "cell_type_l1",
            "cell_type_l2",
            "cell_type_l3",
        ]
        if column in assigned.columns
    ]
    anchor_map = assigned[mapping_cols].copy()
    table_rows = []
    for metaniche_id, group in assigned.groupby("metaniche_id", sort=True):
        out = {"metaniche_id": metaniche_id, "anchor_count": int(len(group))}
        for column in ["slice_id", "time", "time_day", "mouse_id", "cell_type_l1", "cell_type_l2", "cell_type_l3"]:
            if column in group.columns:
                counts = group[column].astype(str).value_counts(dropna=False)
                out[f"dominant_{column}"] = counts.index[0]
                out[f"{column}_purity"] = float(counts.iloc[0] / counts.sum())
        table_rows.append(out)
    metaniche_table = pd.DataFrame(table_rows)
    centroids = assigned[[*feature_columns, "metaniche_id"]].copy()
    centroids[feature_columns] = prepare_numeric(centroids, feature_columns)
    centroids = centroids.groupby("metaniche_id", sort=True)[feature_columns].mean().reset_index()
    composition_rows = []
    for label_column in ["cell_type_l1", "cell_type_l2", "cell_type_l3"]:
        if label_column not in assigned.columns:
            continue
        total_by_mn = assigned.groupby("metaniche_id").size()
        counts = assigned.groupby(["metaniche_id", label_column]).size().reset_index(name="count")
        for item in counts.to_dict(orient="records"):
            total = int(total_by_mn.loc[item["metaniche_id"]])
            composition_rows.append(
                {
                    "metaniche_id": item["metaniche_id"],
                    "label_column": label_column,
                    "label_value": item[label_column],
                    "count": int(item["count"]),
                    "fraction": float(item["count"] / total) if total else 0.0,
                }
            )
    return anchor_map, metaniche_table, centroids, pd.DataFrame(composition_rows)


def attach_coordinates(anchor_map: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    chunks = []
    for row in selected.to_dict(orient="records"):
        coords = read_m1_coordinates(Path(str(row["m1_path"])), max_rows=None)
        subset = anchor_map[anchor_map["slice_id"].astype(str) == str(row["slice_id"])]
        if subset.empty:
            continue
        chunks.append(subset.merge(coords, on=JOIN_KEYS, how="left"))
    joined = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if joined.empty or not {"x", "y", "metaniche_id"}.issubset(joined.columns):
        return joined, pd.DataFrame(), {"coordinate_missing_rate": 1.0, "coordinate_join_complete": False}
    missing = joined[["x", "y"]].isna().any(axis=1)
    coords = (
        joined.dropna(subset=["x", "y"])
        .groupby("metaniche_id", sort=True)
        .agg(
            anchor_count=("anchor_id", "size"),
            x_centroid=("x", "mean"),
            y_centroid=("y", "mean"),
            x_var=("x", "var"),
            y_var=("y", "var"),
            dominant_slice_id=("slice_id", lambda s: s.astype(str).value_counts().index[0]),
            dominant_time_day=("time_day", lambda s: s.astype(str).value_counts().index[0])
            if "time_day" in joined.columns
            else ("slice_id", "size"),
        )
        .reset_index()
    )
    return joined, coords, {
        "coordinate_missing_rate": float(missing.mean()) if len(joined) else 1.0,
        "coordinate_join_complete": bool(not missing.any()),
    }


def rare_state_summary(anchor_map: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for column in ["cell_type_l1", "cell_type_l2", "cell_type_l3"]:
        if column not in anchor_map.columns:
            continue
        counts = anchor_map[column].astype(str).value_counts(dropna=False)
        total = int(counts.sum())
        for value, count in counts.items():
            if count <= max(10, total * 0.001):
                subset = anchor_map[anchor_map[column].astype(str) == str(value)]
                rows.append(
                    {
                        "label_column": column,
                        "label_value": str(value),
                        "anchor_count": int(count),
                        "metaniche_count": int(subset["metaniche_id"].nunique()),
                        "flag": "rare_label_tracked",
                    }
                )
    return {"rare_state_warning_count": len(rows), "rows": rows[:100]}


def ensure_write_policy(params: FullM25Params) -> Path:
    root = validate_output_root(params)
    if params.dry_run:
        return root
    if root.exists() and params.overwrite:
        shutil.rmtree(root)
    ensure_dir(root)
    ensure_dir(root / "logs")
    return root


def build_dry_run_payload(params: FullM25Params) -> dict[str, Any]:
    feature_lock = load_feature_lock(params.feature_lock)
    output_root = validate_output_root(params)
    inventory = discover_slice_inputs(params.m1_root, params.m2_root)
    selected = select_slices_for_run(inventory, params)
    required = [*feature_lock["metadata_columns"], *feature_lock["feature_columns"]]
    schema_rows = [
        validate_feature_schema(Path(str(row["m2_path"])), required)
        for row in selected.to_dict(orient="records")
    ]
    join_frame, join_summary = validate_coordinate_joins(
        selected,
        max_rows=params.max_anchors_per_slice if params.smoke_test else 256,
    )
    planned = planned_slice_counts(selected, params)
    blockers = []
    if inventory.empty:
        blockers.append("No M2 inputs discovered.")
    if any(row["missing_columns"] for row in schema_rows):
        blockers.append("Required feature-lock columns are missing from at least one selected M2 file.")
    if not join_summary["all_valid"]:
        blockers.append("Coordinate join validation did not pass.")
    return {
        "generated_at_utc": utc_now(),
        "mode": "smoke-test" if params.smoke_test else "dry-run",
        "dry_run": params.dry_run,
        "production_will_run": False,
        "feature_lock": {
            "path": str(params.feature_lock),
            "feature_column_count": len(feature_lock["feature_columns"]),
        },
        "output_root": str(output_root),
        "output_root_exists": output_root.exists(),
        "tmp_dir": str(params.tmp_dir),
        "input_inventory": {
            "m2_file_count": int(len(inventory)),
            "selected_slice_count": int(len(selected)),
            "total_selected_anchors": int(selected["m2_rows"].sum()) if not selected.empty else 0,
        },
        "planned_slices": planned.to_dict(orient="records"),
        "schema_validation": schema_rows,
        "coordinate_join_summary": join_summary,
        "coordinate_join_rows": join_frame.to_dict(orient="records"),
        "blockers": blockers,
        "ready_for_no_dry_run": not blockers,
        "environment": implementation_environment_payload(params),
    }


def implementation_environment_payload(params: FullM25Params) -> dict[str, Any]:
    return {
        "hostname": platform.node() or "unknown",
        "date_utc": utc_now(),
        "user": getpass.getuser(),
        "pwd": str(Path.cwd()),
        "git_status_short": git_status_short(),
        "disk_usage": {
            "scratch": disk_usage(Path("/home/zhutao/scratch")),
            "output_root_parent": disk_usage(params.output_root.parent),
        },
        "memory": read_memory_info(),
        "output_policy": {
            "production_output_root": str(FULL_M2_5_PRODUCTION_ROOT),
            "repo_report_root": str(FULL_M2_5_REPORT_ROOT),
            "ssd_outputs_allowed": False,
        },
    }


def dry_run_markdown(payload: dict[str, Any]) -> str:
    planned = pd.DataFrame(payload.get("planned_slices", []))
    join_rows = pd.DataFrame(payload.get("coordinate_join_rows", []))
    return dedent(
        f"""
        # Full M2.5 Dry-Run Contract

        - Mode: `{payload["mode"]}`
        - Production will run: {payload["production_will_run"]}
        - Feature columns: {payload["feature_lock"]["feature_column_count"]}
        - Selected slices: {payload["input_inventory"]["selected_slice_count"]}
        - Selected anchors: {payload["input_inventory"]["total_selected_anchors"]:,}
        - Ready for `--no-dry-run`: {payload["ready_for_no_dry_run"]}
        - Blockers: {", ".join(payload["blockers"]) if payload["blockers"] else "none"}

        ## Planned Slices

        {dataframe_to_markdown(planned)}

        ## Coordinate Join Check

        {dataframe_to_markdown(join_rows)}
        """
    ).strip() + "\n"


def write_implementation_preflight(params: FullM25Params, overwrite: bool = True) -> dict[str, Any]:
    report_root = ensure_dir(FULL_M2_5_REPORT_ROOT)
    payload = {
        "generated_at_utc": utc_now(),
        "phase": "implementation_preflight",
        "environment": implementation_environment_payload(params),
        "target_production_output_exists": FULL_M2_5_PRODUCTION_ROOT.exists(),
        "feature_lock_path": str(params.feature_lock),
        "production_script_target": "scripts/planA_k_23_run_full_m2_5_production.py",
        "full_production_executed": False,
    }
    atomic_write_json(report_root / "00_implementation_preflight.json", payload, overwrite=overwrite)
    atomic_write_text(
        report_root / "00_implementation_preflight.md",
        dedent(
            f"""
            # Full M2.5 Implementation Preflight

            - Generated at: {payload["generated_at_utc"]}
            - Hostname: {payload["environment"]["hostname"]}
            - User: {payload["environment"]["user"]}
            - PWD: `{payload["environment"]["pwd"]}`
            - Production output root: `{FULL_M2_5_PRODUCTION_ROOT}`
            - Target production output exists: {payload["target_production_output_exists"]}
            - Full production executed in this task: False
            """
        ).strip()
        + "\n",
        overwrite=overwrite,
    )
    return payload


def write_dry_run_reports(payload: dict[str, Any], overwrite: bool = True) -> None:
    report_root = ensure_dir(FULL_M2_5_REPORT_ROOT)
    atomic_write_json(report_root / "01_dry_run_contract.json", payload, overwrite=overwrite)
    atomic_write_text(report_root / "01_dry_run_contract.md", dry_run_markdown(payload), overwrite=overwrite)


def write_smoke_test_reports(payload: dict[str, Any], overwrite: bool = True) -> None:
    report_root = ensure_dir(FULL_M2_5_REPORT_ROOT)
    atomic_write_json(report_root / "02_smoke_test_summary.json", payload, overwrite=overwrite)
    atomic_write_text(
        report_root / "02_smoke_test_summary.md",
        full_m2_5_summary_markdown(payload),
        overwrite=overwrite,
    )


def run_full_m2_5(params: FullM25Params) -> dict[str, Any]:
    if params.dry_run and not params.smoke_test:
        payload = build_dry_run_payload(params)
        write_implementation_preflight(params)
        write_dry_run_reports(payload)
        return payload
    if params.smoke_test:
        params = FullM25Params(**{**params.__dict__, "dry_run": False, "max_slices": min(params.max_slices or 2, 2), "max_anchors_per_slice": min(params.max_anchors_per_slice or 1000, 1000)})
    root = ensure_write_policy(params)
    feature_lock = load_feature_lock(params.feature_lock)
    inventory = discover_slice_inputs(params.m1_root, params.m2_root)
    selected = select_slices_for_run(inventory, params)
    join_frame, join_summary = validate_coordinate_joins(selected, max_rows=params.max_anchors_per_slice)
    if not join_summary["all_valid"]:
        raise RuntimeError(f"Coordinate join validation failed: {join_summary['blockers']}")
    plan = planned_slice_counts(selected, params)
    training, training_manifest = sample_training_rows(
        selected,
        feature_lock["feature_columns"],
        params.max_anchors_per_slice,
        params.seed,
    )
    scaler, pca, _ = fit_scaler_pca(training, params.n_pca_components)
    all_maps = []
    all_tables = []
    all_centroids = []
    all_composition = []
    for idx, row in enumerate(selected.to_dict(orient="records")):
        n_clusters = int(plan.loc[plan["slice_id"] == row["slice_id"], "planned_metaniche_count"].iloc[0])
        frame = load_slice_frame(
            row,
            feature_lock["feature_columns"],
            feature_lock["metadata_columns"],
            params.max_anchors_per_slice,
        )
        anchor_map, table, centroids, composition = coarsen_slice(
            frame,
            feature_lock["feature_columns"],
            scaler,
            pca,
            n_clusters,
            params.seed,
            idx,
        )
        all_maps.append(anchor_map)
        all_tables.append(table)
        all_centroids.append(centroids)
        all_composition.append(composition)
    anchor_map = pd.concat(all_maps, ignore_index=True)
    metaniche_table = pd.concat(all_tables, ignore_index=True)
    centroids = pd.concat(all_centroids, ignore_index=True)
    composition = pd.concat(all_composition, ignore_index=True) if all_composition else pd.DataFrame()
    joined, coordinates, coord_payload = attach_coordinates(anchor_map, selected)
    rare_payload = rare_state_summary(anchor_map)
    qc = {
        "generated_at_utc": utc_now(),
        "smoke_test": params.smoke_test,
        "full_production_run": not params.smoke_test,
        "anchors_processed": int(len(anchor_map)),
        "metaniche_count": int(metaniche_table["metaniche_id"].nunique()) if not metaniche_table.empty else 0,
        "slice_count": int(anchor_map["slice_id"].nunique()) if "slice_id" in anchor_map else 0,
        "timepoints": sorted(anchor_map["time"].dropna().astype(str).unique().tolist()) if "time" in anchor_map else [],
        "coordinate_join": coord_payload,
        "rare_state_summary": rare_payload,
        "join_validation": join_summary,
        "planned_slices": plan.to_dict(orient="records"),
    }
    write_outputs(
        root=root,
        params=params,
        feature_lock=feature_lock,
        training_manifest=training_manifest,
        scaler=scaler,
        pca=pca,
        anchor_map=anchor_map,
        metaniche_table=metaniche_table,
        centroids=centroids,
        coordinates=coordinates,
        composition=composition,
        qc=qc,
    )
    payload = {
        "generated_at_utc": utc_now(),
        "status": "completed",
        "smoke_test": params.smoke_test,
        "full_production_run": not params.smoke_test,
        "output_root": str(root),
        "qc": qc,
        "output_files": [str(root / name) for name in REQUIRED_OUTPUTS if (root / name).exists()],
        "production_executed": not params.smoke_test,
    }
    if params.smoke_test:
        write_smoke_test_reports(payload)
    return payload


def write_outputs(
    root: Path,
    params: FullM25Params,
    feature_lock: dict[str, Any],
    training_manifest: pd.DataFrame,
    scaler: StandardScaler,
    pca: IncrementalPCA,
    anchor_map: pd.DataFrame,
    metaniche_table: pd.DataFrame,
    centroids: pd.DataFrame,
    coordinates: pd.DataFrame,
    composition: pd.DataFrame,
    qc: dict[str, Any],
) -> None:
    ensure_dir(root)
    ensure_dir(root / "logs")
    atomic_write_json(root / "run_manifest.json", run_manifest(params, qc), overwrite=True)
    atomic_write_json(root / "feature_lock.used.json", feature_lock, overwrite=True)
    atomic_write_text(root / "feature_columns.txt", "\n".join(feature_lock["feature_columns"]) + "\n", overwrite=True)
    joblib.dump(scaler, root / "scaler.joblib")
    joblib.dump(pca, root / "pca.joblib")
    np.save(root / "pca_components.npy", pca.components_)
    atomic_write_tsv(root / "training_sample_manifest.tsv", training_manifest, overwrite=True)
    anchor_map.to_parquet(root / "anchor_to_metaniche.parquet", index=False)
    metaniche_table.to_parquet(root / "metaniche_table.parquet", index=False)
    centroids.to_parquet(root / "metaniche_feature_centroids.parquet", index=False)
    atomic_write_tsv(root / "metaniche_coordinates.tsv", coordinates, overwrite=True)
    atomic_write_tsv(root / "metaniche_composition.tsv", composition, overwrite=True)
    atomic_write_json(root / "metaniche_qc.json", qc, overwrite=True)
    summary_payload = {
        "generated_at_utc": utc_now(),
        "full_production_run": not params.smoke_test,
        "smoke_test": params.smoke_test,
        "anchors_processed": qc["anchors_processed"],
        "metaniche_count": qc["metaniche_count"],
        "slice_count": qc["slice_count"],
        "timepoints": qc["timepoints"],
        "coordinate_join_complete": qc["coordinate_join"]["coordinate_join_complete"],
    }
    atomic_write_json(root / "full_m2_5_summary.json", summary_payload, overwrite=True)
    atomic_write_text(root / "full_m2_5_summary.md", full_m2_5_summary_markdown({"qc": qc, **summary_payload}), overwrite=True)


def run_manifest(params: FullM25Params, qc: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now(),
        "command_scope": "smoke_test" if params.smoke_test else "full_production",
        "seed": params.seed,
        "feature_lock": str(params.feature_lock),
        "output_root": str(params.output_root),
        "m1_root": str(params.m1_root),
        "m2_root": str(params.m2_root),
        "n_pca_components": params.n_pca_components,
        "target_mode": params.target_mode,
        "metaniche_count_bounds": [
            params.min_metaniches_per_slice,
            params.base_metaniches_per_slice,
            params.max_metaniches_per_slice,
        ],
        "qc": qc,
        "guardrails": {
            "darlin_processed": False,
            "raw_data_modified": False,
            "frozen_p_fate_modified": False,
            "ssd_outputs_used": False,
            "slurm_submitted": False,
            "full_kmix_a_run": False,
            "gpcca_run": False,
            "branchsbm_trained": False,
        },
    }


def full_m2_5_summary_markdown(payload: dict[str, Any]) -> str:
    qc = payload.get("qc", payload)
    return dedent(
        f"""
        # Full M2.5 Output Summary

        - Full production run: {payload.get("full_production_run", qc.get("full_production_run"))}
        - Smoke test: {payload.get("smoke_test", qc.get("smoke_test"))}
        - Anchors processed: {qc.get("anchors_processed", 0):,}
        - Metaniches produced: {qc.get("metaniche_count", 0):,}
        - Slices represented: {qc.get("slice_count", 0)}
        - Timepoints: {", ".join(qc.get("timepoints", []))}
        - Coordinate join complete: {qc.get("coordinate_join", {}).get("coordinate_join_complete")}

        This output does not run Kmix_A, GPCCA, BranchSBM, Slurm, DARLIN, or
        macrostate annotation.
        """
    ).strip() + "\n"


def implementation_summary_payload(
    validations: dict[str, Any],
    output_files: list[str],
) -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now(),
        "script_implemented": True,
        "supported_cli": [
            "--feature-lock",
            "--output-root",
            "--seed",
            "--m1-root",
            "--m2-root",
            "--dry-run/--no-dry-run",
            "--smoke-test",
            "--max-slices",
            "--max-anchors-per-slice",
            "--overwrite",
            "--resume",
            "--n-pca-components",
            "--target-mode",
            "--min-metaniches-per-slice",
            "--max-metaniches-per-slice",
            "--base-metaniches-per-slice",
            "--tmp-dir",
        ],
        "validations": validations,
        "production_run_executed": False,
        "ready_for_full_production_guard_rerun": bool(
            validations.get("py_compile_passed")
            and validations.get("synthetic_tests_passed")
            and validations.get("dry_run_passed")
            and validations.get("smoke_test_passed")
        ),
        "next_safe_command": (
            "conda run --no-capture-output -n omicverse python "
            "scripts/planA_k_23_run_full_m2_5_production.py "
            "--feature-lock configs/planA_k/full_m2_5_feature_lock.draft.json "
            "--output-root /home/zhutao/scratch/nichefate/planA_k_production/full_m2_5 "
            "--seed 271828 --dry-run"
        ),
        "should_not_be_claimed_yet": [
            "full M2.5 production completed",
            "all 1,439,542 anchors processed by production",
            "production metaniche count",
            "full Kmix_A readiness",
            "GPCCA readiness beyond resource-cautious preflight",
        ],
        "files_created": output_files,
        "files_not_touched": [
            "DARLIN data",
            "raw data",
            "frozen P_fate outputs",
            "/ssd",
            "git index or remote",
        ],
        "guardrails": {
            "no_darlin_processing": True,
            "no_raw_data_modification": True,
            "no_frozen_p_fate_modification": True,
            "no_ssd_output": True,
            "no_slurm": True,
            "no_full_production_m2_5": True,
            "no_full_kmix_a": True,
            "no_gpcca": True,
            "no_branchsbm_training": True,
            "no_git_add_commit_push": True,
            "git_status_before_after_recorded": True,
        },
    }


def write_implementation_summary(payload: dict[str, Any], overwrite: bool = True) -> None:
    report_root = ensure_dir(FULL_M2_5_REPORT_ROOT)
    atomic_write_json(report_root / "00_FULL_M2_5_IMPLEMENTATION_SUMMARY.json", payload, overwrite=overwrite)
    atomic_write_text(
        report_root / "00_FULL_M2_5_IMPLEMENTATION_SUMMARY.md",
        dedent(
            f"""
            # Full M2.5 Implementation Summary

            - Script implemented: {payload["script_implemented"]}
            - Py compile passed: {payload["validations"].get("py_compile_passed")}
            - Synthetic tests passed: {payload["validations"].get("synthetic_tests_passed")}
            - Dry-run passed: {payload["validations"].get("dry_run_passed")}
            - Bounded smoke-test passed: {payload["validations"].get("smoke_test_passed")}
            - Production run executed: {payload["production_run_executed"]}
            - Ready for guard rerun: {payload["ready_for_full_production_guard_rerun"]}
            - Next safe command: `{payload["next_safe_command"]}`

            ## Do Not Claim Yet

            {chr(10).join(f"- {item}" for item in payload["should_not_be_claimed_yet"])}

            ## Files Created

            {chr(10).join(f"- `{item}`" for item in payload["files_created"])}
            """
        ).strip()
        + "\n",
        overwrite=overwrite,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
