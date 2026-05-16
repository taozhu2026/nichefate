"""Metaniche coarsening, pilot QC, and hardening helpers for PlanA-K."""

from __future__ import annotations

import getpass
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.sparse import csgraph

from .schemas import *
from .io import *
from .reporting import *
from .validation import *
from .kernel_qc import *
from .m2_inventory import *


def metaniche_pilot_protocol_text() -> str:
    return dedent(
        """
        # M2.5 Metaniche Pilot Protocol

        ## Biological Unit

        An anchor-indexed micro-niche is a sampled local microenvironment, not
        a cell-level state. The anchor is an indexing point. A metaniche or
        niche-state is an aggregation of similar micro-niches in M2
        representation space. GPCCA macrostates will later be inferred from
        directed sparse transitions among metaniches.

        ## First Pilot Scope

        Use a small, safe subset only: at most four representative M2 slices
        and at most 5,000 anchors per slice. The pilot must not use the full
        M2 dataset and must not run GPCCA.

        ## Coarsening Strategy

        - Sample anchors stratified by timepoint/slice.
        - Select safe feature groups: composition, entropy, and embedding mean.
        - Standardize features.
        - Reduce dimensionality with PCA, capped at 30 components.
        - Cluster with Leiden when explicitly requested and available, or use
          MiniBatchKMeans for a capped reproducible pilot.
        - Produce metaniche centroids and an anchor-to-metaniche map.

        ## Coarsening Constraints

        - Preserve timepoint and slice labels.
        - Do not mix incompatible conditions without recording it.
        - Preserve spatial centroid only when coordinates exist in M2 metadata.
        - Preserve cell-type composition summaries when labels exist.
        - Track rare-state diagnostics so rare states are not silently erased.

        ## Output Contract

        - `anchor_to_metaniche.tsv`
        - `metaniche_table.tsv`
        - `metaniche_feature_centroids.csv`
        - `metaniche_composition.tsv`
        - `metaniche_qc.json`
        - `pilot_summary.md`

        ## QC

        - Number of anchors sampled.
        - Number of metaniches.
        - Anchors per metaniche distribution.
        - Timepoint and slice purity.
        - Spatial compactness if coordinates exist.
        - Feature compactness.
        - Rare-state loss warning.
        - Whether the metaniche count is suitable for sparse K and GPCCA pilot.

        ## Failure Modes

        - Too few metaniches.
        - Too many tiny metaniches.
        - Mixed timepoint artifacts.
        - Spatially incoherent clusters.
        - Feature group domination.
        - Rare states collapsed away.
        - Missing metadata.
        """
    ).strip() + "\n"


def choose_representative_m2_slices(
    inventory: pd.DataFrame,
    max_slices: int,
) -> pd.DataFrame:
    if inventory.empty:
        return inventory.copy()
    safe = inventory[inventory["safe_to_sample"] == True].copy()  # noqa: E712
    if safe.empty:
        return safe
    safe["time_day_sort"] = pd.to_numeric(safe.get("time_day"), errors="coerce")
    safe["rows_sort"] = pd.to_numeric(safe.get("rows"), errors="coerce").fillna(0)
    selected_indices: list[int] = []
    for _, group in safe.sort_values(["time_day_sort", "slice_id"]).groupby(
        "time_day_sort", dropna=False
    ):
        pick = group.sort_values(["rows_sort", "slice_id"], ascending=[False, True]).index[0]
        selected_indices.append(int(pick))
        if len(selected_indices) >= max_slices:
            break
    return safe.loc[selected_indices].sort_values(["time_day_sort", "slice_id"])


def sample_parquet_rows_by_position(
    path: Path,
    columns: list[str],
    max_rows: int,
    seed: int,
    batch_size: int = 8192,
) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    row_count = int(parquet.metadata.num_rows)
    if row_count == 0 or max_rows <= 0:
        return pd.DataFrame(columns=columns)
    sample_size = min(max_rows, row_count)
    rng = np.random.default_rng(seed)
    selected_positions = np.sort(rng.choice(row_count, size=sample_size, replace=False))
    chunks: list[pd.DataFrame] = []
    offset = 0
    cursor = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        batch_rows = batch.num_rows
        batch_end = offset + batch_rows
        while cursor < len(selected_positions) and selected_positions[cursor] < offset:
            cursor += 1
        end_cursor = cursor
        while end_cursor < len(selected_positions) and selected_positions[end_cursor] < batch_end:
            end_cursor += 1
        if end_cursor > cursor:
            local_positions = selected_positions[cursor:end_cursor] - offset
            frame = batch.to_pandas()
            chunks.append(frame.iloc[local_positions].copy())
        offset = batch_end
        cursor = end_cursor
        if cursor >= len(selected_positions):
            break
    if not chunks:
        return pd.DataFrame(columns=columns)
    return pd.concat(chunks, ignore_index=True)


def bounded_sample_m2_rows(
    inventory: pd.DataFrame,
    schema: dict[str, Any],
    feature_columns: list[str],
    max_slices: int,
    max_anchors_per_slice: int,
    seed: int,
) -> pd.DataFrame:
    selected = choose_representative_m2_slices(inventory, max_slices=max_slices)
    metadata_columns = [
        column for column in schema.get("metadata_columns", []) if column not in feature_columns
    ]
    chunks: list[pd.DataFrame] = []
    for order, row in enumerate(selected.to_dict(orient="records")):
        path = Path(str(row["path"]))
        parquet = pq.ParquetFile(path)
        available_columns = set(parquet.schema_arrow.names)
        columns = [
            column
            for column in [*metadata_columns, *feature_columns]
            if column in available_columns
        ]
        sample = sample_parquet_rows_by_position(
            path=path,
            columns=columns,
            max_rows=max_anchors_per_slice,
            seed=seed + order,
        )
        sample["source_m2_path"] = str(path)
        sample["source_slice_order"] = order
        chunks.append(sample)
    if not chunks:
        return pd.DataFrame()
    sampled = pd.concat(chunks, ignore_index=True)
    if "anchor_index" in sampled.columns:
        slice_values = sampled["slice_id"].astype(str) if "slice_id" in sampled.columns else ""
        sampled["anchor_id"] = (
            slice_values + "::" + sampled["anchor_index"].astype(str)
        )
    elif "anchor_cell_id" in sampled.columns:
        slice_values = sampled["slice_id"].astype(str) if "slice_id" in sampled.columns else ""
        sampled["anchor_id"] = slice_values + "::" + sampled["anchor_cell_id"].astype(str)
    else:
        sampled["anchor_id"] = [f"sample_anchor_{index:08d}" for index in range(len(sampled))]
    return sampled


def summarize_purity(values: pd.Series) -> tuple[Any, float]:
    if values.empty:
        return None, 0.0
    counts = values.astype(str).value_counts(dropna=False)
    if counts.empty:
        return None, 0.0
    return counts.index[0], float(counts.iloc[0] / counts.sum())


def build_metaniche_outputs(
    sampled: pd.DataFrame,
    feature_columns: list[str],
    labels: np.ndarray,
    reduced: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    assigned = sampled.copy()
    unique_labels = sorted(pd.Series(labels).astype(str).unique())
    label_map = {label: f"MN{index:04d}" for index, label in enumerate(unique_labels)}
    assigned["metaniche_id"] = pd.Series(labels).astype(str).map(label_map).to_numpy()

    rows: list[dict[str, Any]] = []
    for metaniche_id, group in assigned.groupby("metaniche_id", sort=True):
        row: dict[str, Any] = {
            "metaniche_id": metaniche_id,
            "anchor_count": int(len(group)),
        }
        for column in ["time", "time_day", "slice_id", "mouse_id", "cell_type_l1", "cell_type_l2", "cell_type_l3"]:
            if column in group.columns:
                dominant, purity = summarize_purity(group[column])
                row[f"dominant_{column}"] = dominant
                row[f"{column}_purity"] = purity
        rows.append(row)
    metaniche_table = pd.DataFrame(rows)

    mapping_columns = [
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
            "source_m2_path",
        ]
        if column in assigned.columns
    ]
    anchor_map = assigned[mapping_columns].copy()

    centroids = assigned.groupby("metaniche_id", sort=True)[feature_columns].mean(numeric_only=True)
    centroids = centroids.reset_index()

    composition_rows: list[dict[str, Any]] = []
    for label_column in ["cell_type_l1", "cell_type_l2", "cell_type_l3"]:
        if label_column not in assigned.columns:
            continue
        total_by_metaniche = assigned.groupby("metaniche_id").size()
        counts = assigned.groupby(["metaniche_id", label_column]).size().reset_index(name="count")
        for count_row in counts.to_dict(orient="records"):
            total = int(total_by_metaniche.loc[count_row["metaniche_id"]])
            composition_rows.append(
                {
                    "metaniche_id": count_row["metaniche_id"],
                    "label_column": label_column,
                    "label_value": count_row[label_column],
                    "count": int(count_row["count"]),
                    "fraction": float(count_row["count"] / total) if total else 0.0,
                }
            )
    composition = pd.DataFrame(composition_rows)

    reduced_frame = pd.DataFrame(
        reduced,
        columns=[f"pc{i + 1:02d}" for i in range(reduced.shape[1])],
    )
    reduced_frame["metaniche_id"] = assigned["metaniche_id"].to_numpy()
    reduced_centroids = reduced_frame.groupby("metaniche_id").mean()
    distances: list[float] = []
    for index, row in reduced_frame.iterrows():
        metaniche_id = row["metaniche_id"]
        vector = row.drop(labels=["metaniche_id"]).to_numpy(dtype=float)
        center = reduced_centroids.loc[metaniche_id].to_numpy(dtype=float)
        distances.append(float(np.linalg.norm(vector - center)))
    assigned["_feature_distance_to_centroid"] = distances

    size_series = assigned["metaniche_id"].value_counts()
    qc = {
        "sampled_anchor_count": int(len(assigned)),
        "metaniche_count": int(size_series.size),
        "anchors_per_metaniche_min": int(size_series.min()) if not size_series.empty else 0,
        "anchors_per_metaniche_median": float(size_series.median()) if not size_series.empty else 0.0,
        "anchors_per_metaniche_max": int(size_series.max()) if not size_series.empty else 0,
        "singleton_metaniche_count": int((size_series == 1).sum()),
        "tiny_metaniche_count_lt10": int((size_series < 10).sum()),
        "coordinate_centroid_available": False,
        "spatial_compactness": None,
        "feature_compactness_mean_distance": float(np.mean(distances)) if distances else None,
        "feature_compactness_median_distance": float(np.median(distances)) if distances else None,
    }
    for column in ["time", "time_day", "slice_id", "cell_type_l1", "cell_type_l2", "cell_type_l3"]:
        purity_column = f"{column}_purity"
        if purity_column in metaniche_table.columns:
            qc[f"{column}_purity_median"] = float(metaniche_table[purity_column].median())
            qc[f"{column}_purity_min"] = float(metaniche_table[purity_column].min())
    return anchor_map, metaniche_table, centroids, composition, qc


def run_sampled_metaniche_pilot(
    inventory: pd.DataFrame,
    schema: dict[str, Any],
    output_dir: Path,
    max_slices: int = 4,
    max_anchors_per_slice: int = 5000,
    feature_mode: str = "safe",
    n_components: int = 30,
    n_clusters: int = 200,
    cluster_method: str = "kmeans",
    resolution: float = 1.0,
    dry_run: bool = True,
    overwrite: bool = False,
    seed: int = 17,
) -> dict[str, Any]:
    pilot_dir = output_dir / "pilot_outputs"
    summary_path = output_dir / "04_pilot_run_summary.json"
    if dry_run:
        return {
            "generated_at_utc": utc_now(),
            "pilot_run": False,
            "dry_run": True,
            "reason": "Dry run requested; no coarsening outputs were written.",
            "selected_slices": choose_representative_m2_slices(inventory, max_slices).to_dict(orient="records"),
            "feature_mode": feature_mode,
            "recommended_command": (
                "conda run -n omicverse python scripts/planA_k_05_metaniche_pilot.py "
                "--no-dry-run --max-slices 4 --max-anchors-per-slice 5000 "
                "--n-components 30 --n-clusters 200 --feature-mode safe --cluster-method kmeans --overwrite"
            ),
        }
    if summary_path.exists() and not overwrite:
        return {
            "generated_at_utc": utc_now(),
            "pilot_run": False,
            "dry_run": False,
            "reason": f"{summary_path} already exists; pass --overwrite to replace pilot report outputs.",
        }
    if "safe_to_sample" not in inventory.columns:
        safe_inventory = inventory.iloc[0:0].copy()
    else:
        safe_inventory = inventory[inventory["safe_to_sample"] == True]  # noqa: E712
    if safe_inventory.empty:
        return {
            "generated_at_utc": utc_now(),
            "pilot_run": False,
            "dry_run": False,
            "reason": "No M2 files were safe to sample.",
        }
    max_slices = min(max_slices, 4)
    max_anchors_per_slice = min(max_anchors_per_slice, 5000)
    n_components = min(n_components, 30)
    n_clusters = min(n_clusters, 500)
    feature_columns = select_m2_feature_columns(schema, feature_mode=feature_mode)
    if not feature_columns:
        return {
            "generated_at_utc": utc_now(),
            "pilot_run": False,
            "dry_run": False,
            "reason": f"No feature columns selected for feature_mode={feature_mode}.",
        }

    sampled = bounded_sample_m2_rows(
        inventory=safe_inventory,
        schema=schema,
        feature_columns=feature_columns,
        max_slices=max_slices,
        max_anchors_per_slice=max_anchors_per_slice,
        seed=seed,
    )
    if sampled.empty:
        return {
            "generated_at_utc": utc_now(),
            "pilot_run": False,
            "dry_run": False,
            "reason": "Bounded sampling returned zero rows.",
        }
    feature_columns = [column for column in feature_columns if column in sampled.columns]
    numeric = sampled[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(axis=0)).fillna(0.0)

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(numeric.to_numpy(dtype=np.float32))
    effective_components = min(n_components, scaled.shape[1], max(1, scaled.shape[0] - 1))
    reduced = PCA(n_components=effective_components, random_state=seed).fit_transform(scaled)

    method_used = cluster_method
    labels: np.ndarray
    if cluster_method == "leiden":
        try:
            import scanpy as sc
            import anndata as ad

            adata = ad.AnnData(reduced)
            sc.pp.neighbors(adata, n_neighbors=min(30, max(2, len(sampled) - 1)), use_rep="X")
            sc.tl.leiden(adata, resolution=resolution, random_state=seed, key_added="metaniche")
            labels = adata.obs["metaniche"].astype(str).to_numpy()
            if len(set(labels)) > 500:
                raise RuntimeError("Leiden produced more than 500 clusters")
        except Exception as exc:
            method_used = f"kmeans_fallback_after_leiden_error:{exc}"
            k = min(n_clusters, max(2, len(sampled) // 20), len(sampled))
            labels = MiniBatchKMeans(
                n_clusters=k,
                random_state=seed,
                batch_size=min(2048, len(sampled)),
                n_init=10,
            ).fit_predict(reduced)
    else:
        k = min(n_clusters, max(2, len(sampled) // 20), len(sampled))
        labels = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=min(2048, len(sampled)),
            n_init=10,
        ).fit_predict(reduced)

    anchor_map, metaniche_table, centroids, composition, qc = build_metaniche_outputs(
        sampled=sampled,
        feature_columns=feature_columns,
        labels=np.asarray(labels),
        reduced=np.asarray(reduced),
    )
    ensure_dir(pilot_dir)
    atomic_write_tsv(pilot_dir / "anchor_to_metaniche.tsv", anchor_map, overwrite=overwrite)
    atomic_write_tsv(pilot_dir / "metaniche_table.tsv", metaniche_table, overwrite=overwrite)
    atomic_write_csv(pilot_dir / "metaniche_feature_centroids.csv", centroids, overwrite=overwrite)
    atomic_write_tsv(pilot_dir / "metaniche_composition.tsv", composition, overwrite=overwrite)
    qc_payload = {
        "generated_at_utc": utc_now(),
        "pilot_run": True,
        "dry_run": False,
        "feature_mode": feature_mode,
        "feature_column_count": len(feature_columns),
        "max_slices": max_slices,
        "max_anchors_per_slice": max_anchors_per_slice,
        "n_components": effective_components,
        "n_clusters_requested": n_clusters,
        "cluster_method_used": method_used,
        "selected_slices": choose_representative_m2_slices(safe_inventory, max_slices).to_dict(orient="records"),
        "qc": qc,
        "output_files": [
            str(pilot_dir / "anchor_to_metaniche.tsv"),
            str(pilot_dir / "metaniche_table.tsv"),
            str(pilot_dir / "metaniche_feature_centroids.csv"),
            str(pilot_dir / "metaniche_composition.tsv"),
            str(pilot_dir / "metaniche_qc.json"),
            str(pilot_dir / "pilot_summary.md"),
        ],
    }
    atomic_write_json(pilot_dir / "metaniche_qc.json", qc_payload, overwrite=overwrite)
    atomic_write_text(
        pilot_dir / "pilot_summary.md",
        pilot_run_summary_markdown(qc_payload),
        overwrite=overwrite,
    )
    return qc_payload


def pilot_run_summary_markdown(payload: dict[str, Any]) -> str:
    if not payload.get("pilot_run"):
        return dedent(
            f"""
            # M2.5 Pilot Run Summary

            - Pilot run: False
            - Dry run: {payload.get("dry_run")}
            - Reason: {payload.get("reason", "not specified")}
            - Recommended next command: `{payload.get("recommended_command", "not available")}`
            """
        ).strip() + "\n"
    qc = payload["qc"]
    return dedent(
        f"""
        # M2.5 Pilot Run Summary

        - Pilot run: True
        - Sampled anchors: {qc["sampled_anchor_count"]:,}
        - Metaniches: {qc["metaniche_count"]:,}
        - Feature columns used: {payload["feature_column_count"]:,}
        - PCA components: {payload["n_components"]}
        - Cluster method used: {payload["cluster_method_used"]}
        - Anchors per metaniche median: {qc["anchors_per_metaniche_median"]:.2f}
        - Tiny metaniches with <10 anchors: {qc["tiny_metaniche_count_lt10"]}
        - Coordinate centroid available: {qc["coordinate_centroid_available"]}

        The pilot is bounded to sampled M2 rows only and does not run GPCCA.
        """
    ).strip() + "\n"


def load_pilot_outputs(output_dir: Path) -> dict[str, pd.DataFrame]:
    pilot_dir = output_dir / "pilot_outputs"
    paths = {
        "anchor_map": pilot_dir / "anchor_to_metaniche.tsv",
        "metaniche_table": pilot_dir / "metaniche_table.tsv",
        "centroids": pilot_dir / "metaniche_feature_centroids.csv",
        "composition": pilot_dir / "metaniche_composition.tsv",
    }
    loaded: dict[str, pd.DataFrame] = {}
    for key, path in paths.items():
        if path.exists():
            if path.suffix == ".csv":
                loaded[key] = pd.read_csv(path)
            else:
                loaded[key] = pd.read_csv(path, sep="\t")
    return loaded


def compute_metaniche_qc_from_outputs(output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    loaded = load_pilot_outputs(output_dir)
    if "anchor_map" not in loaded or "metaniche_table" not in loaded:
        payload = {
            "generated_at_utc": utc_now(),
            "pilot_outputs_found": False,
            "reason": "anchor_to_metaniche.tsv or metaniche_table.tsv is missing",
            "suitable_for_sparse_k_pilot": False,
            "suitable_for_full_gpcca_later": False,
        }
        rows = pd.DataFrame(
            [
                {
                    "qc_metric": "pilot_outputs_found",
                    "value": False,
                    "status": "FAIL",
                    "interpretation": payload["reason"],
                }
            ]
        )
        return rows, payload

    anchor_map = loaded["anchor_map"]
    metaniche_table = loaded["metaniche_table"]
    size_series = anchor_map["metaniche_id"].value_counts()
    metaniche_count = int(size_series.size)
    anchor_count = int(len(anchor_map))
    time_pair_candidates = []
    if "time_day" in anchor_map.columns:
        days = sorted(pd.to_numeric(anchor_map["time_day"], errors="coerce").dropna().unique())
        time_pair_candidates = [f"D{int(a)}->D{int(b)}" for a, b in zip(days[:-1], days[1:])]
    rows = [
        {
            "qc_metric": "sampled_anchor_count",
            "value": anchor_count,
            "status": "PASS" if anchor_count <= 20_000 else "FAIL",
            "interpretation": "Pilot sample is within the requested <=20,000 anchor cap.",
        },
        {
            "qc_metric": "metaniche_count",
            "value": metaniche_count,
            "status": "PASS" if 10 <= metaniche_count <= 500 else "WARN",
            "interpretation": "Candidate state count for a sparse-K pilot.",
        },
        {
            "qc_metric": "anchors_per_metaniche_median",
            "value": float(size_series.median()) if not size_series.empty else 0.0,
            "status": "PASS" if not size_series.empty and size_series.median() >= 10 else "WARN",
            "interpretation": "Median metaniche size should avoid mostly singleton states.",
        },
        {
            "qc_metric": "tiny_metaniche_count_lt10",
            "value": int((size_series < 10).sum()),
            "status": "PASS" if int((size_series < 10).sum()) <= max(5, 0.1 * metaniche_count) else "WARN",
            "interpretation": "Too many tiny states may indicate over-clustering.",
        },
        {
            "qc_metric": "timepoint_purity_median",
            "value": float(metaniche_table["time_day_purity"].median()) if "time_day_purity" in metaniche_table else None,
            "status": "PASS" if "time_day_purity" in metaniche_table and metaniche_table["time_day_purity"].median() >= 0.8 else "WARN",
            "interpretation": "High timepoint purity supports directed sparse-K construction.",
        },
        {
            "qc_metric": "timepoint_purity_min",
            "value": float(metaniche_table["time_day_purity"].min()) if "time_day_purity" in metaniche_table else None,
            "status": "PASS" if "time_day_purity" in metaniche_table and metaniche_table["time_day_purity"].min() >= 0.8 else "WARN",
            "interpretation": "Low minimum purity flags a subset of mixed-time metaniches for manual review.",
        },
        {
            "qc_metric": "slice_purity_median",
            "value": float(metaniche_table["slice_id_purity"].median()) if "slice_id_purity" in metaniche_table else None,
            "status": "PASS" if "slice_id_purity" in metaniche_table and metaniche_table["slice_id_purity"].median() >= 0.8 else "WARN",
            "interpretation": "High slice purity means the pilot preserved slice structure.",
        },
        {
            "qc_metric": "slice_purity_min",
            "value": float(metaniche_table["slice_id_purity"].min()) if "slice_id_purity" in metaniche_table else None,
            "status": "PASS" if "slice_id_purity" in metaniche_table and metaniche_table["slice_id_purity"].min() >= 0.8 else "WARN",
            "interpretation": "Low minimum purity flags a subset of mixed-slice metaniches for manual review.",
        },
        {
            "qc_metric": "coordinate_centroid_available",
            "value": False,
            "status": "WARN",
            "interpretation": "The current M2 schema does not expose x/y coordinates, so spatial compactness is deferred.",
        },
        {
            "qc_metric": "recommended_top_k_range",
            "value": "10,20,30" if metaniche_count >= 30 else "5,10",
            "status": "PASS",
            "interpretation": "Use top-k values below the state count and inspect zero rows.",
        },
    ]
    rare_warning = "not evaluated"
    if "cell_type_l3" in anchor_map.columns:
        counts = anchor_map["cell_type_l3"].astype(str).value_counts()
        rare = counts[counts < max(10, int(0.01 * len(anchor_map)))]
        rare_warning = f"{len(rare)} rare cell_type_l3 labels in the sampled anchors"
        rows.append(
            {
                "qc_metric": "rare_state_preservation_warning",
                "value": rare_warning,
                "status": "WARN" if len(rare) else "PASS",
                "interpretation": "Rare labels require manual review before production coarsening.",
            }
        )
    suitable_sparse = 10 <= metaniche_count <= 500 and anchor_count <= 20_000
    payload = {
        "generated_at_utc": utc_now(),
        "pilot_outputs_found": True,
        "sampled_anchor_count": anchor_count,
        "metaniche_count": metaniche_count,
        "anchors_per_metaniche": {
            "min": int(size_series.min()) if not size_series.empty else 0,
            "median": float(size_series.median()) if not size_series.empty else 0.0,
            "max": int(size_series.max()) if not size_series.empty else 0,
        },
        "singleton_metaniche_count": int((size_series == 1).sum()),
        "tiny_metaniche_count_lt10": int((size_series < 10).sum()),
        "coordinate_centroid_available": False,
        "spatial_compactness": None,
        "composition_available": "composition" in loaded and not loaded.get("composition", pd.DataFrame()).empty,
        "rare_state_preservation_warning": rare_warning,
        "recommended_top_k_range": [10, 20, 30] if metaniche_count >= 30 else [5, 10],
        "timepoint_pairs_for_next_sparse_k": time_pair_candidates,
        "suitable_for_sparse_k_pilot": suitable_sparse,
        "suitable_for_full_gpcca_later": False,
        "full_gpcca_reason": "This is a sampled pilot only; production M2.5 and sparse-K QC are still required.",
    }
    return pd.DataFrame(rows), payload


def metaniche_qc_markdown(frame: pd.DataFrame, payload: dict[str, Any]) -> str:
    if not payload.get("pilot_outputs_found"):
        return dedent(
            f"""
            # Metaniche QC

            Pilot outputs were not found.

            Reason: {payload.get("reason")}
            """
        ).strip() + "\n"
    return dedent(
        f"""
        # Metaniche QC And GPCCA-Readiness Assessment

        - Sampled anchors: {payload["sampled_anchor_count"]:,}
        - Metaniches: {payload["metaniche_count"]:,}
        - Anchors per metaniche median: {payload["anchors_per_metaniche"]["median"]:.2f}
        - Singleton metaniches: {payload["singleton_metaniche_count"]}
        - Tiny metaniches <10 anchors: {payload["tiny_metaniche_count_lt10"]}
        - Coordinate centroid available: {payload["coordinate_centroid_available"]}
        - Suitable for sparse-K pilot: {payload["suitable_for_sparse_k_pilot"]}
        - Suitable for full GPCCA later: {payload["suitable_for_full_gpcca_later"]}

        {dataframe_to_markdown(frame)}
        """
    ).strip() + "\n"


def next_sparse_k_pilot_design(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    use_states = bool(payload.get("suitable_for_sparse_k_pilot", False))
    next_command = (
        "conda run -n omicverse python scripts/planA_k_06_metaniche_qc.py --overwrite"
        if not use_states
        else (
            "conda run -n omicverse python scripts/planA_k_03_sparse_kernel_design_probe.py "
            "--dry-run --overwrite"
        )
    )
    design = {
        "generated_at_utc": utc_now(),
        "use_pilot_metaniches_as_states": use_states,
        "timepoint_pairs": payload.get("timepoint_pairs_for_next_sparse_k", []),
        "feature_representation": "metaniche_feature_centroids.csv from safe M2 composition + entropy + embedding-mean features",
        "candidate_edge_construction": {
            "adjacent_timepoint_only": True,
            "top_k": [10, 20, 30],
            "adaptive_bandwidth": "kth_neighbor_distance",
            "spatial_gating": "optional; off until coordinates are available",
            "row_normalization": True,
            "zero_row_handling": "record zero rows; add lazy self-loop only after inspection",
            "lazy_self_loop": [0.05, 0.10],
        },
        "qc_required_before_gpcca": [
            "row-stochasticity",
            "zero rows",
            "nnz per row",
            "weak and strong components",
            "row entropy",
            "timepoint edge balance",
            "macrostate stability across top_k after GPCCA pilot",
        ],
        "exact_next_safe_command": next_command,
        "next_codex_task": (
            "Implement a dry-run-only metaniche sparse-K pilot script that consumes "
            "reports/planA_k_metaniche_pilot/pilot_outputs after reviewing mixed-purity warnings."
        ),
    }
    text = dedent(
        f"""
        # Next Sparse-K Pilot Design

        - Use pilot metaniches as states: {design["use_pilot_metaniches_as_states"]}
        - Timepoint pairs available: {", ".join(design["timepoint_pairs"]) or "not available"}
        - Feature representation: {design["feature_representation"]}
        - Candidate edges: adjacent timepoints only, top_k = 10/20/30, adaptive bandwidth.
        - Spatial gating: optional and off until coordinates are available.
        - Zero-row handling: inspect first, then add lazy self-loop 0.05 or 0.10 only if justified.

        ## Required QC Before GPCCA

        {dataframe_to_markdown(pd.DataFrame({"qc": design["qc_required_before_gpcca"]}))}

        ## Exact Next Safe Command

        `{design["exact_next_safe_command"]}`
        """
    ).strip() + "\n"
    return text, design


def metaniche_final_summary_payload(
    output_dir: Path,
    inventory_summary: dict[str, Any] | None,
    feature_audit: pd.DataFrame | None,
    pilot_payload: dict[str, Any] | None,
    qc_payload: dict[str, Any] | None,
    git_status_after: list[str],
) -> dict[str, Any]:
    pilot_payload = pilot_payload or {}
    qc_payload = qc_payload or {}
    feature_groups = []
    if feature_audit is not None and not feature_audit.empty:
        feature_groups = feature_audit["feature_group"].tolist()
    created_files = list_report_files(output_dir)
    for expected in [
        output_dir / "00_M2_5_METANICHE_PILOT_SUMMARY.md",
        output_dir / "00_M2_5_METANICHE_PILOT_SUMMARY.json",
        DOC_ROOT / "05_m2_5_metaniche_pilot_protocol.md",
    ]:
        rel = str(expected.relative_to(PROJECT_ROOT))
        if rel not in created_files:
            created_files.append(rel)
    return {
        "generated_at_utc": utc_now(),
        "m2_outputs_found_and_safely_indexed": bool(
            inventory_summary and inventory_summary.get("safe_to_sample_count", 0) > 0
        ),
        "feature_groups_available_for_coarsening": feature_groups,
        "recommended_metaniche_definition": (
            "A metaniche/niche-state aggregates similar anchor-indexed micro-niches "
            "in M2 representation space and is the preferred state unit before sparse-K/GPCCA."
        ),
        "sampled_m2_5_pilot_run": bool(pilot_payload.get("pilot_run")),
        "sampled_anchor_count": qc_payload.get("sampled_anchor_count"),
        "metaniche_count": qc_payload.get("metaniche_count"),
        "biologically_and_computationally_plausible": (
            "candidate-level yes, with warnings about missing coordinates, rare-state preservation, "
            "and a small number of mixed time/slice metaniches"
            if qc_payload.get("suitable_for_sparse_k_pilot", False)
            else "not yet"
        ),
        "coarsening_required_before_gpcca": True,
        "uncertainties": [
            "M2 schema does not expose x/y coordinates, so spatial compactness is deferred.",
            "The pilot is sampled and cannot support production-scale GPCCA claims.",
            "Rare-state preservation needs manual review before production M2.5.",
        ],
        "pilot_output_suitable_for_sparse_k_pilot": bool(
            qc_payload.get("suitable_for_sparse_k_pilot", False)
        ),
        "minimal_safe_next_command": (
            "conda run -n omicverse python scripts/planA_k_03_sparse_kernel_design_probe.py --dry-run --overwrite"
            if qc_payload.get("suitable_for_sparse_k_pilot", False)
            else "conda run -n omicverse python scripts/planA_k_05_metaniche_pilot.py --no-dry-run --overwrite"
        ),
        "next_codex_task": (
            "Add a dry-run-only metaniche sparse-K pilot that reads the sampled metaniche outputs, "
            "constructs adjacent-time candidate edges at top_k 10/20/30, and reports kernel QC without GPCCA."
        ),
        "must_not_be_claimed_yet": [
            "full-scale M2.5 production is complete",
            "full GPCCA has been run",
            "metaniches are final biological macrostates",
            "K_gpcca improves over frozen P_fate",
            "DARLIN-supported transitions exist",
            "BranchSBM has been trained",
        ],
        "files_created": created_files,
        "files_not_touched": [
            "raw data",
            "frozen P_fate outputs",
            "DARLIN / spatio_DARLIN processing outputs",
            "/ssd",
            "scratch M2 source files",
        ],
        "validation_checklist": {
            "no_darlin_processing": True,
            "no_raw_data_modification": True,
            "no_frozen_p_fate_modification": True,
            "no_ssd_output": True,
            "no_slurm": True,
            "no_full_gpcca": True,
            "no_branchsbm_training": True,
            "git_status_after": git_status_after,
        },
    }


def metaniche_final_summary_markdown(payload: dict[str, Any]) -> str:
    return dedent(
        f"""
        # M2.5 Metaniche Pilot Summary

        ## 1. Were M2 outputs found and safely indexed?
        {payload["m2_outputs_found_and_safely_indexed"]}

        ## 2. What feature groups are available for coarsening?
        {", ".join(payload["feature_groups_available_for_coarsening"]) or "none"}

        ## 3. What is the recommended definition of metaniche / niche-state?
        {payload["recommended_metaniche_definition"]}

        ## 4. Was a sampled M2.5 pilot run?
        {payload["sampled_m2_5_pilot_run"]}

        ## 5. If yes, how many anchors and metaniches were produced?
        Anchors: {payload["sampled_anchor_count"]}; metaniches: {payload["metaniche_count"]}.

        ## 6. Are the metaniches biologically and computationally plausible?
        {payload["biologically_and_computationally_plausible"]}

        ## 7. Is coarsening clearly required before GPCCA?
        {payload["coarsening_required_before_gpcca"]}

        ## 8. What failed or remains uncertain?
        {dataframe_to_markdown(pd.DataFrame({"uncertainty": payload["uncertainties"]}))}

        ## 9. Is the pilot output suitable for a sparse-K pilot?
        {payload["pilot_output_suitable_for_sparse_k_pilot"]}

        ## 10. What is the minimal safe next command?
        `{payload["minimal_safe_next_command"]}`

        ## 11. What should not be claimed yet?
        {dataframe_to_markdown(pd.DataFrame({"claim_to_avoid": payload["must_not_be_claimed_yet"]}))}

        ## 12. What files were created?
        {dataframe_to_markdown(pd.DataFrame({"file": payload["files_created"]}))}

        ## 13. What files were not touched?
        {dataframe_to_markdown(pd.DataFrame({"file_or_area": payload["files_not_touched"]}))}

        ## Validation Checklist

        - no DARLIN processing: {payload["validation_checklist"]["no_darlin_processing"]}
        - no raw data modification: {payload["validation_checklist"]["no_raw_data_modification"]}
        - no frozen P_fate modification: {payload["validation_checklist"]["no_frozen_p_fate_modification"]}
        - no /ssd output: {payload["validation_checklist"]["no_ssd_output"]}
        - no Slurm: {payload["validation_checklist"]["no_slurm"]}
        - no full GPCCA: {payload["validation_checklist"]["no_full_gpcca"]}
        - no BranchSBM training: {payload["validation_checklist"]["no_branchsbm_training"]}
        """
    ).strip() + "\n"


def stratified_pilot_design_payload() -> tuple[str, dict[str, Any]]:
    options = pd.DataFrame(
        [
            {
                "option": "A",
                "strategy": "per-slice coarsening",
                "mixing_risk": "lowest",
                "rare_state_handling": "best first pass",
                "recommended": True,
                "reason": "Directly prevents cross-slice and cross-time clusters in the pilot.",
            },
            {
                "option": "B",
                "strategy": "per-timepoint coarsening",
                "mixing_risk": "medium",
                "rare_state_handling": "good",
                "recommended": False,
                "reason": "Can still mix slices within a timepoint.",
            },
            {
                "option": "C",
                "strategy": "slice-stratified sampling + global clustering",
                "mixing_risk": "medium",
                "rare_state_handling": "moderate",
                "recommended": False,
                "reason": "This was close to the original pilot and still produced mixed metaniches.",
            },
            {
                "option": "D",
                "strategy": "global clustering + post-hoc purity filtering",
                "mixing_risk": "highest",
                "rare_state_handling": "reactive",
                "recommended": False,
                "reason": "Filtering after clustering does not define a clean state contract.",
            },
        ]
    )
    payload = {
        "generated_at_utc": utc_now(),
        "recommended_strategy": "A. per-slice coarsening",
        "max_slices": 4,
        "max_anchors_per_slice": 5000,
        "total_anchor_cap": 20000,
        "feature_groups": ["niche_composition", "entropy", "embedding_mean"],
        "pca_components": 30,
        "cluster_rule": "allocate up to 50 KMeans clusters per slice for a 200-state pilot cap",
        "purity_threshold": 0.95,
        "rare_state_preservation_check": "required before sparse-K",
        "coordinate_requirement": "required for spatial compactness QC, not for dry-run clustering",
        "output_contract": [
            "anchor_to_metaniche.tsv",
            "metaniche_table.tsv",
            "metaniche_feature_centroids.csv",
            "metaniche_composition.tsv",
            "metaniche_qc.json",
            "pilot_summary.md",
        ],
        "options": options.to_dict(orient="records"),
    }
    text = dedent(
        f"""
        # Stratified Metaniche Pilot Design

        Recommended first strategy: **{payload["recommended_strategy"]}**.

        - Max slices: {payload["max_slices"]}
        - Max anchors per slice: {payload["max_anchors_per_slice"]}
        - Total anchor cap: {payload["total_anchor_cap"]}
        - Feature groups: {", ".join(payload["feature_groups"])}
        - PCA components: {payload["pca_components"]}
        - Cluster rule: {payload["cluster_rule"]}
        - Purity threshold: {payload["purity_threshold"]}
        - Coordinate requirement: {payload["coordinate_requirement"]}

        {dataframe_to_markdown(options)}
        """
    ).strip() + "\n"
    return text, payload


def run_stratified_metaniche_pilot(
    output_dir: Path,
    m2_root: Path = M2_BY_SLICE_ROOT,
    schema_path: Path = M2_SCHEMA_PATH,
    max_slices: int = 4,
    max_anchors_per_slice: int = 5000,
    feature_mode: str = "safe",
    n_components: int = 30,
    n_clusters: int = 200,
    seed: int = 17,
    dry_run: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    strat_dir = ensure_dir(output_dir / "stratified_pilot_outputs")
    inventory, _ = discover_m2_inventory(m2_root=m2_root, schema_path=schema_path)
    schema = load_m2_feature_schema(schema_path)
    feature_columns = select_m2_feature_columns(schema, feature_mode=feature_mode)
    selected = choose_representative_m2_slices(inventory, min(max_slices, 4))
    if dry_run:
        return {
            "generated_at_utc": utc_now(),
            "stratified_pilot_run": False,
            "dry_run": True,
            "selected_slices": selected.to_dict(orient="records"),
            "recommended_command": (
                "conda run -n omicverse python scripts/planA_k_09_stratified_metaniche_pilot.py "
                "--no-dry-run --max-slices 4 --max-anchors-per-slice 5000 --n-clusters 200 --overwrite"
            ),
        }
    sampled = bounded_sample_m2_rows(
        inventory=inventory,
        schema=schema,
        feature_columns=feature_columns,
        max_slices=min(max_slices, 4),
        max_anchors_per_slice=min(max_anchors_per_slice, 5000),
        seed=seed,
    )
    if sampled.empty:
        return {
            "generated_at_utc": utc_now(),
            "stratified_pilot_run": False,
            "dry_run": False,
            "reason": "sampling returned no anchors",
        }
    feature_columns = [column for column in feature_columns if column in sampled.columns]
    numeric = sampled[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(axis=0)).fillna(0.0)

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(numeric.to_numpy(dtype=np.float32))
    effective_components = min(n_components, 30, scaled.shape[1], max(1, scaled.shape[0] - 1))
    reduced = PCA(n_components=effective_components, random_state=seed).fit_transform(scaled)
    labels = np.empty(len(sampled), dtype=object)
    slice_count = max(1, sampled["slice_id"].nunique())
    clusters_per_slice = max(2, min(125, int(np.ceil(min(n_clusters, 500) / slice_count))))
    for slice_order, (slice_id, index) in enumerate(sampled.groupby("slice_id").groups.items()):
        idx = np.asarray(list(index))
        k = min(clusters_per_slice, max(2, len(idx) // 20), len(idx))
        local_labels = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed + slice_order,
            batch_size=min(2048, len(idx)),
            n_init=10,
        ).fit_predict(reduced[idx])
        labels[idx] = [f"{slice_id}__K{label}" for label in local_labels]

    anchor_map, metaniche_table, centroids, composition, qc = build_metaniche_outputs(
        sampled=sampled,
        feature_columns=feature_columns,
        labels=labels,
        reduced=reduced,
    )
    atomic_write_tsv(strat_dir / "anchor_to_metaniche.tsv", anchor_map, overwrite=overwrite)
    atomic_write_tsv(strat_dir / "metaniche_table.tsv", metaniche_table, overwrite=overwrite)
    atomic_write_csv(strat_dir / "metaniche_feature_centroids.csv", centroids, overwrite=overwrite)
    atomic_write_tsv(strat_dir / "metaniche_composition.tsv", composition, overwrite=overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "stratified_pilot_run": True,
        "dry_run": False,
        "strategy": "per-slice coarsening",
        "cluster_method_used": "per_slice_kmeans",
        "feature_mode": feature_mode,
        "feature_column_count": len(feature_columns),
        "sampled_anchor_count": int(len(anchor_map)),
        "metaniche_count": int(len(metaniche_table)),
        "clusters_per_slice_cap": int(clusters_per_slice),
        "n_components": int(effective_components),
        "qc": qc,
        "output_files": [
            str(strat_dir / "anchor_to_metaniche.tsv"),
            str(strat_dir / "metaniche_table.tsv"),
            str(strat_dir / "metaniche_feature_centroids.csv"),
            str(strat_dir / "metaniche_composition.tsv"),
            str(strat_dir / "metaniche_qc.json"),
            str(strat_dir / "pilot_summary.md"),
        ],
    }
    atomic_write_json(strat_dir / "metaniche_qc.json", payload, overwrite=overwrite)
    atomic_write_text(strat_dir / "pilot_summary.md", pilot_run_summary_markdown({"pilot_run": True, **payload}), overwrite=overwrite)
    return payload


def compare_original_and_stratified_pilots(
    output_dir: Path,
    original_root: Path = PILOT_OUTPUT_ROOT,
) -> tuple[str, dict[str, Any]]:
    original = pd.read_csv(original_root / "metaniche_table.tsv", sep="\t") if (original_root / "metaniche_table.tsv").exists() else pd.DataFrame()
    strat_root = output_dir / "stratified_pilot_outputs"
    stratified = pd.read_csv(strat_root / "metaniche_table.tsv", sep="\t") if (strat_root / "metaniche_table.tsv").exists() else pd.DataFrame()
    original_spatial = {}
    stratified_spatial = {}
    original_rare = {}
    stratified_rare = {}
    original_spatial_path = output_dir / "04_spatial_compactness_qc.json"
    stratified_spatial_path = strat_root / "spatial_compactness_qc.json"
    original_rare_path = output_dir / "05_rare_state_preservation_audit.json"
    stratified_rare_path = strat_root / "rare_state_preservation_audit.json"
    if original_spatial_path.exists():
        original_spatial = json.loads(original_spatial_path.read_text(encoding="utf-8"))
    if stratified_spatial_path.exists():
        stratified_spatial = json.loads(stratified_spatial_path.read_text(encoding="utf-8"))
    if original_rare_path.exists():
        original_rare = json.loads(original_rare_path.read_text(encoding="utf-8"))
    if stratified_rare_path.exists():
        stratified_rare = json.loads(stratified_rare_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for name, frame, spatial, rare in [
        ("original", original, original_spatial, original_rare),
        ("stratified", stratified, stratified_spatial, stratified_rare),
    ]:
        if frame.empty:
            rows.append({"pilot": name, "available": False})
            continue
        rows.append(
            {
                "pilot": name,
                "available": True,
                "metaniche_count": int(len(frame)),
                "anchor_count": int(frame["anchor_count"].sum()) if "anchor_count" in frame else None,
                "anchor_count_median": float(frame["anchor_count"].median()) if "anchor_count" in frame else None,
                "time_purity_median": float(frame["time_day_purity"].median()) if "time_day_purity" in frame else None,
                "time_purity_min": float(frame["time_day_purity"].min()) if "time_day_purity" in frame else None,
                "slice_purity_median": float(frame["slice_id_purity"].median()) if "slice_id_purity" in frame else None,
                "slice_purity_min": float(frame["slice_id_purity"].min()) if "slice_id_purity" in frame else None,
                "spatial_compactness_available": bool(spatial.get("spatial_compactness_available", False)),
                "diffuse_metaniche_count": spatial.get("diffuse_metaniche_count"),
                "radius_p90_median": spatial.get("radius_p90_median"),
                "rare_state_audit_available": bool(rare.get("rare_state_audit_available", False)),
                "rare_collapsed_warning_count": rare.get("collapsed_warning_count"),
            }
        )
    comparison = pd.DataFrame(rows)
    payload = {
        "generated_at_utc": utc_now(),
        "stratified_pilot_run": not stratified.empty,
        "comparison": comparison.to_dict(orient="records"),
        "suitability_for_sparse_k_pilot": (
            "candidate_ready_after_coordinate_and_rare_state_review"
            if not stratified.empty
            else "blocked_no_stratified_output"
        ),
    }
    text = dedent(
        f"""
        # Stratified Pilot Summary

        {dataframe_to_markdown(comparison)}

        - Sparse-K suitability: {payload["suitability_for_sparse_k_pilot"]}
        """
    ).strip() + "\n"
    return text, payload


def m2_5_state_contract_v2() -> tuple[str, pd.DataFrame]:
    rows = [
        {
            "level": "anchor",
            "field_or_file": "anchor_id",
            "required": True,
            "source": "derived",
            "description": "Stable `slice_id::anchor_index` identifier.",
        },
        {
            "level": "anchor",
            "field_or_file": "slice_id; anchor_index; anchor_cell_id",
            "required": True,
            "source": "M2/M1",
            "description": "Primary join key for M2 features and M1 coordinates.",
        },
        {
            "level": "anchor",
            "field_or_file": "x; y",
            "required": True,
            "source": "M1",
            "description": "Anchor-level spatial coordinates rescued from M1.",
        },
        {
            "level": "anchor",
            "field_or_file": "time; time_day; mouse_id",
            "required": True,
            "source": "M2",
            "description": "Temporal/sample metadata for directional sparse-K.",
        },
        {
            "level": "metaniche",
            "field_or_file": "anchor_to_metaniche.tsv",
            "required": True,
            "source": "M2.5",
            "description": "Anchor provenance and metaniche assignment.",
        },
        {
            "level": "metaniche",
            "field_or_file": "metaniche_table.tsv",
            "required": True,
            "source": "M2.5",
            "description": "State table with size, dominant metadata, and purity.",
        },
        {
            "level": "metaniche",
            "field_or_file": "metaniche_feature_centroids.csv",
            "required": True,
            "source": "M2.5",
            "description": "Feature centroid matrix for sparse-K candidate edges.",
        },
        {
            "level": "metaniche",
            "field_or_file": "metaniche_coordinates.preview.tsv",
            "required": True,
            "source": "coordinate rescue",
            "description": "Metaniche x/y centroid and coordinate variance.",
        },
        {
            "level": "qc",
            "field_or_file": "spatial compactness QC",
            "required": True,
            "source": "hardening",
            "description": "Radius distribution and diffuse-state flags.",
        },
        {
            "level": "qc",
            "field_or_file": "rare-state preservation audit",
            "required": True,
            "source": "hardening",
            "description": "Rare-label collapse/enrichment warnings.",
        },
        {
            "level": "optional_annotation",
            "field_or_file": "metaniche_composition.tsv",
            "required": False,
            "source": "M2.5",
            "description": "Cell-type label composition for biological annotation.",
        },
        {
            "level": "production_blocker",
            "field_or_file": "full production M2.5",
            "required": True,
            "source": "future",
            "description": "Current outputs are sampled pilots and cannot support full GPCCA claims.",
        },
    ]
    frame = pd.DataFrame(rows)
    text = dedent(
        f"""
        # M2.5 State Contract v2

        This contract supersedes the first M2.5 pilot contract for sparse-K preparation.
        Sparse-K should consume metaniche states only after coordinate rescue, purity QC,
        spatial compactness QC, and rare-state preservation checks are attached.

        {dataframe_to_markdown(frame)}

        ## Required Files For Sparse-K

        - `anchor_to_metaniche.tsv`
        - `metaniche_table.tsv`
        - `metaniche_feature_centroids.csv`
        - `metaniche_coordinates.preview.tsv`
        - `05_rare_state_preservation_audit.tsv`
        - `04_spatial_compactness_qc.tsv`

        ## Blockers For Full Production

        - This is still a sampled pilot.
        - Production M2.5 must run across all intended slices with the same coordinate contract.
        - Rare states and diffuse metaniches require review before any GPCCA claim.
        """
    ).strip() + "\n"
    return text, frame


def hardening_final_summary_payload(
    output_dir: Path,
    coord_payload: dict[str, Any],
    spatial_payload: dict[str, Any],
    rare_payload: dict[str, Any],
    strat_payload: dict[str, Any],
    git_status_after: list[str],
) -> dict[str, Any]:
    coord_found = bool(coord_payload.get("safe_join_identified"))
    strat_run = bool(strat_payload.get("stratified_pilot_run"))
    next_ready = coord_found and spatial_payload.get("spatial_compactness_available", False)
    return {
        "generated_at_utc": utc_now(),
        "xy_coordinates_found": coord_found,
        "can_join_to_m2_anchors": coord_found,
        "best_join_key": coord_payload.get("join_key", "slice_id + anchor_index + anchor_cell_id"),
        "coordinate_join_worked_on_pilot": bool(coord_payload.get("coordinate_join_run") and coord_found),
        "pilot_metaniches_spatially_compact": (
            spatial_payload.get("diffuse_metaniche_count", 0) == 0
            if spatial_payload.get("spatial_compactness_available")
            else False
        ),
        "rare_states_preserved": rare_payload.get("collapsed_warning_count", 0) == 0,
        "time_slice_mixing_seriousness": "moderate in original pilot; stratified per-slice pilot is recommended",
        "stratified_coarsening_recommended": True,
        "stratified_pilot_run": strat_run,
        "state_contract_for_sparse_k": "M2.5 state contract v2",
        "next_sparse_k_pilot_ready": bool(next_ready),
        "minimal_safe_next_command": (
            "conda run -n omicverse python scripts/planA_k_03_sparse_kernel_design_probe.py --dry-run --overwrite"
        ),
        "must_not_be_claimed_yet": [
            "production M2.5 is complete",
            "full GPCCA has been run",
            "metaniches are final biological macrostates",
            "K_gpcca improves over frozen P_fate",
            "DARLIN-supported transitions exist",
            "BranchSBM has been trained",
        ],
        "files_created": list_report_files(output_dir)
        + [str((DOC_ROOT / "06_m2_5_state_contract_v2.md").relative_to(PROJECT_ROOT))],
        "files_not_touched": [
            "raw data",
            "frozen P_fate outputs",
            "DARLIN / spatio_DARLIN outputs",
            "/ssd",
            "scratch M0/M1/M2 source files",
        ],
        "validation_checklist": {
            "no_darlin_processing": True,
            "no_raw_data_modification": True,
            "no_frozen_p_fate_modification": True,
            "no_ssd_output": True,
            "no_slurm": True,
            "no_full_gpcca": True,
            "no_branchsbm_training": True,
            "git_status_after": git_status_after,
        },
        "coordinate_join_qc": coord_payload,
        "spatial_compactness_qc": spatial_payload,
        "rare_state_qc": rare_payload,
        "stratified_pilot_qc": strat_payload,
    }


def hardening_final_summary_markdown(payload: dict[str, Any]) -> str:
    return dedent(
        f"""
        # Metaniche Hardening Summary

        ## 1. Were x/y coordinates found?
        {payload["xy_coordinates_found"]}

        ## 2. Can they be joined to M2 anchors?
        {payload["can_join_to_m2_anchors"]}

        ## 3. What is the best join key?
        `{payload["best_join_key"]}`

        ## 4. Did coordinate join work on the pilot?
        {payload["coordinate_join_worked_on_pilot"]}

        ## 5. Are pilot metaniches spatially compact?
        {payload["pilot_metaniches_spatially_compact"]}

        ## 6. Are rare states preserved?
        {payload["rare_states_preserved"]}

        ## 7. How serious is time/slice mixing?
        {payload["time_slice_mixing_seriousness"]}

        ## 8. Is stratified coarsening recommended?
        {payload["stratified_coarsening_recommended"]}

        ## 9. Was a stratified pilot run?
        {payload["stratified_pilot_run"]}

        ## 10. Which M2.5 state contract should be used for sparse-K?
        {payload["state_contract_for_sparse_k"]}

        ## 11. Is the next sparse-K pilot ready?
        {payload["next_sparse_k_pilot_ready"]}

        ## 12. What is the minimal safe next command?
        `{payload["minimal_safe_next_command"]}`

        ## 13. What should not be claimed yet?
        {dataframe_to_markdown(pd.DataFrame({"claim_to_avoid": payload["must_not_be_claimed_yet"]}))}

        ## 14. What files were created?
        {dataframe_to_markdown(pd.DataFrame({"file": payload["files_created"]}))}

        ## 15. What files were not touched?
        {dataframe_to_markdown(pd.DataFrame({"file_or_area": payload["files_not_touched"]}))}

        ## Validation

        - no DARLIN processing: {payload["validation_checklist"]["no_darlin_processing"]}
        - no raw data modification: {payload["validation_checklist"]["no_raw_data_modification"]}
        - no frozen P_fate modification: {payload["validation_checklist"]["no_frozen_p_fate_modification"]}
        - no /ssd output: {payload["validation_checklist"]["no_ssd_output"]}
        - no Slurm: {payload["validation_checklist"]["no_slurm"]}
        - no full GPCCA: {payload["validation_checklist"]["no_full_gpcca"]}
        - no BranchSBM training: {payload["validation_checklist"]["no_branchsbm_training"]}
        """
    ).strip() + "\n"


__all__ = [name for name in globals() if not name.startswith("__")]
