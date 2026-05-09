#!/usr/bin/env python
"""Build one M3 transition pilot shard, or dry-run the selected shard."""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any

# Keep exact-neighbor pilot execution from inheriting very high node CPU counts
# that can exceed the OpenBLAS build's supported thread metadata limit.
for _thread_var in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_thread_var, "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config
from nichefate.transition import (
    build_candidate_neighbors,
    CandidateNeighborBackendStatus,
    categorical_target_diagnostics,
    combine_scaled_evidence,
    full_transition_schema_columns,
    pair_adaptive_temperature,
    pairwise_row_distance,
    row_normalize_weights,
    safe_scale_vector,
    standardize_feature_matrices,
    transition_probability_diagnostics,
)


GROUP_TO_EVIDENCE = {
    "molecular_state": "molecular",
    "cell_type_composition": "composition",
    "entropy": "entropy",
    "spatial_summary": "spatial_summary",
    "topology": "topology",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_transition_kernel.yaml")
    parser.add_argument("--source-slice-id", required=True)
    parser.add_argument("--source-slice-file", required=True)
    parser.add_argument("--source-time", required=True)
    parser.add_argument("--target-time", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        key: Path(value)
        for key, value in config["paths"].items()
        if isinstance(value, str) and value.startswith("/")
    }


def _safe_token(value: object) -> str:
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")
    return token or "value"


def _slice_path(root: Path, slice_id: str) -> Path:
    return root / slice_id / f"m2_representation_{slice_id}.parquet"


def output_paths(output_dir: Path, source_time: str, target_time: str, slice_id: str) -> dict[str, Path]:
    stem = f"{_safe_token(source_time)}_to_{_safe_token(target_time)}__{_safe_token(slice_id)}"
    return {
        "edges": output_dir / f"candidate_edges_{stem}.parquet",
        "report": output_dir / f"pilot_report_{stem}.md",
    }


def dry_run_summary(shard: dict[str, Any], edge_path: Path) -> dict[str, Any]:
    """Return dry-run details without writing any pilot outputs."""

    return {
        "selected_shard": f"{shard['source_time']}->{shard['target_time']} {shard['source_slice_id']}",
        "source_rows": int(shard["source_rows"]),
        "target_rows": int(shard["target_time_rows"]),
        "expected_edge_rows": int(shard["expected_edge_rows"]),
        "output_path": str(edge_path),
    }


def select_shard(shards: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    selected = shards[
        (shards["source_time"].astype(str) == str(args.source_time))
        & (shards["target_time"].astype(str) == str(args.target_time))
        & (shards["source_slice_id"].astype(str) == str(args.source_slice_id))
        & (shards["source_slice_file"].astype(str) == str(args.source_slice_file))
    ]
    if len(selected) != 1:
        raise ValueError(f"Selected pilot shard must match exactly one row, found {len(selected)}.")
    return selected.iloc[0].to_dict()


def _load_target_time(root: Path, target_slices: list[str], columns: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(_slice_path(root, slice_id), columns=columns) for slice_id in target_slices]
    return pd.concat(frames, ignore_index=True)


def _edge_metadata(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    shard: dict[str, Any],
) -> pd.DataFrame:
    src = source.iloc[source_idx]
    tgt = target.iloc[target_idx]
    return pd.DataFrame(
        {
            "source_anchor_id": src["slice_id"].astype(str).to_numpy()
            + "::"
            + src["anchor_index"].astype(str).to_numpy(),
            "target_anchor_id": tgt["slice_id"].astype(str).to_numpy()
            + "::"
            + tgt["anchor_index"].astype(str).to_numpy(),
            "source_anchor_index": src["anchor_index"].to_numpy(),
            "target_anchor_index": tgt["anchor_index"].to_numpy(),
            "source_time": src["time"].astype(str).to_numpy(),
            "target_time": tgt["time"].astype(str).to_numpy(),
            "source_day": src["time_day"].to_numpy(),
            "target_day": tgt["time_day"].to_numpy(),
            "time_delta": float(shard["time_delta"]),
            "source_slice_id": src["slice_id"].astype(str).to_numpy(),
            "target_slice_id": tgt["slice_id"].astype(str).to_numpy(),
            "source_slice_file": src["slice_file"].astype(str).to_numpy(),
            "target_slice_file": tgt["slice_file"].astype(str).to_numpy(),
            "source_mouse_id": src["mouse_id"].astype(str).to_numpy(),
            "target_mouse_id": tgt["mouse_id"].astype(str).to_numpy(),
            "evidence_mode": "pseudo_lineage",
        }
    )


def build_pilot_edges(
    source: pd.DataFrame,
    target: pd.DataFrame,
    shard: dict[str, Any],
    config: dict[str, Any],
    feature_groups: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    groups = feature_groups["feature_groups"]
    full = config["full_m3"]
    cost_cfg = config["cost"]
    retrieval_groups = list(full["retrieval_feature_groups"])
    rerank_groups = list(full["rerank_feature_groups"])
    retrieval_columns = list(dict.fromkeys(column for group in retrieval_groups for column in groups[group]))
    source_retrieval, target_retrieval, standardize_stats = standardize_feature_matrices(
        source[retrieval_columns].to_numpy(dtype=float),
        target[retrieval_columns].to_numpy(dtype=float),
        float(cost_cfg["min_scale"]),
    )
    neighbors = build_candidate_neighbors(
        source_retrieval,
        target_retrieval,
        int(full["candidate_k"]),
        backend=full["neighbor_backend"],
        metric=config["candidate_edges"].get("retrieval_metric", "euclidean"),
        chunk_size=int(config["candidate_edges"].get("numpy_chunk_size", 512)),
    )
    if isinstance(neighbors, CandidateNeighborBackendStatus):
        raise RuntimeError(f"Candidate-neighbor backend unavailable: {neighbors.reason}")
    source_idx = np.repeat(np.arange(len(source)), neighbors.indices.shape[1])
    target_idx = neighbors.indices.reshape(-1)
    frame = _edge_metadata(source, target, source_idx, target_idx, shard)
    scaled_column_by_group = {}
    scaling_stats: dict[str, dict[str, Any]] = {}
    for group in rerank_groups:
        evidence = GROUP_TO_EVIDENCE[group]
        raw_col = f"raw_{evidence}_distance"
        scaled_col = f"scaled_{evidence}_distance"
        metric = "l1" if group == "cell_type_composition" else "euclidean"
        frame[raw_col] = pairwise_row_distance(source, target, source_idx, target_idx, groups[group], metric=metric)
        frame[scaled_col], stats = safe_scale_vector(frame[raw_col], float(cost_cfg["min_scale"]))
        frame[f"scaling_method_{evidence}"] = stats["scaling_method_used"]
        frame[f"zero_variance_{evidence}"] = stats["zero_variance"]
        scaled_column_by_group[group] = scaled_col
        scaling_stats[group] = stats
    frame["raw_pseudotime_score"] = 0.0
    frame["raw_barcode_score"] = 0.0
    frame["scaled_pseudotime_score"] = 0.0
    frame["scaled_barcode_score"] = 0.0
    frame["source_mass"] = 1.0
    frame["target_mass"] = 1.0
    frame["growth_prior"] = 1.0
    frame["unbalanced_weight"] = 1.0
    frame["mass_adjusted_weight"] = 0.0
    frame["combined_cost"] = combine_scaled_evidence(frame, cost_cfg["evidence_weights"], scaled_column_by_group)
    tau = pair_adaptive_temperature(frame["combined_cost"], float(cost_cfg["min_temperature"]))
    frame["tau_pair"] = tau
    exponent = np.clip(-frame["combined_cost"].to_numpy(dtype=float) / tau, -700, 700)
    frame["raw_edge_weight"] = np.exp(exponent)
    frame["mass_adjusted_weight"] = frame["raw_edge_weight"]
    frame["row_normalized_transition_prob"] = row_normalize_weights(frame)
    frame = frame[full_transition_schema_columns()]
    metadata = {
        "backend": neighbors.backend,
        "tau_pair": tau,
        "retrieval_feature_columns": len(retrieval_columns),
        "zero_variance_retrieval_columns": standardize_stats["zero_variance_columns"],
        "scaling_stats": scaling_stats,
    }
    return frame, metadata


def target_distribution_diagnostics(frame: pd.DataFrame) -> dict[str, float]:
    diagnostics = {}
    diagnostics.update(categorical_target_diagnostics(frame, "source_slice_id", "target_slice_id"))
    diagnostics.update(categorical_target_diagnostics(frame, "source_slice_id", "target_mouse_id"))
    return diagnostics


def validate_pilot_edges(frame: pd.DataFrame, shard: dict[str, Any], schema_columns: list[str]) -> dict[str, Any]:
    if list(frame.columns) != schema_columns:
        raise ValueError("Pilot edge schema does not match full transition schema.")
    expected_rows = int(shard["expected_edge_rows"])
    if len(frame) != expected_rows:
        raise ValueError(f"Pilot edge rows {len(frame)} != expected {expected_rows}.")
    counts = frame.groupby("source_anchor_id", observed=True).size()
    if not bool((counts == int(shard["candidate_k"])).all()):
        raise ValueError("Not every source anchor has exactly K candidates.")
    row_sums = frame.groupby("source_anchor_id", observed=True)["row_normalized_transition_prob"].sum()
    if not bool(np.allclose(row_sums.to_numpy(dtype=float), 1.0)):
        raise ValueError("Local transition probabilities do not sum to 1.")
    metadata_cols = [column for column in frame.columns if column.startswith(("source_", "target_"))]
    if int(frame[metadata_cols].isna().sum().sum()):
        raise ValueError("Pilot edge table has missing source/target metadata.")
    numeric_cols = frame.select_dtypes(include=[np.number, "bool"]).columns
    if int((~np.isfinite(frame[numeric_cols].to_numpy(dtype=float))).sum()):
        raise ValueError("Pilot edge table has NaN or infinite numeric values.")
    if not bool((frame["source_time"].astype(str) == str(shard["source_time"])).all()):
        raise ValueError("Source time mismatch.")
    if not bool((frame["target_time"].astype(str) == str(shard["target_time"])).all()):
        raise ValueError("Target time mismatch.")
    if not bool(np.allclose(frame["time_delta"], float(shard["time_delta"]))):
        raise ValueError("Time delta mismatch.")
    diagnostics = transition_probability_diagnostics(frame)
    diagnostics.update(target_distribution_diagnostics(frame))
    diagnostics["candidate_count_min"] = int(counts.min())
    diagnostics["candidate_count_max"] = int(counts.max())
    diagnostics["row_count"] = int(len(frame))
    return diagnostics


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# M3 Pilot Shard Report",
        "",
        "This report covers one real source-slice transition pilot shard only.",
        "It does not build full M3, a global Markov transition matrix P, GPCCA, fate probabilities, downstream model training artifacts, or regulator analysis.",
        "`row_normalized_transition_prob` remains local to each source niche candidate set.",
        "",
        f"- Status: {summary['status']}",
        f"- Source time: {summary['source_time']}",
        f"- Target time: {summary['target_time']}",
        f"- Source slice: {summary['source_slice_id']}",
        f"- Source rows: {summary['source_rows']}",
        f"- Target rows: {summary['target_rows']}",
        f"- Edge rows: {summary['edge_rows']}",
        f"- KNN backend: {summary['backend']}",
        f"- Tau pair: {summary['tau_pair']}",
        f"- Runtime seconds: {summary['runtime_seconds']:.3f}",
        f"- Max RSS KB: {summary['max_rss_kb']}",
        f"- Output bytes: {summary['output_bytes']}",
        f"- Row sum range: {summary['row_sum_min']:.6g} - {summary['row_sum_max']:.6g}",
        f"- Row entropy mean: {summary['row_entropy_mean']:.6g}",
        f"- Top1 probability mean: {summary['top1_probability_mean']:.6g}",
        f"- Target slice entropy mean: {summary['target_slice_id_entropy_mean']:.6g}",
        f"- Top target slice fraction mean: {summary['top_target_slice_id_fraction_mean']:.6g}",
        f"- Target mouse entropy mean: {summary['target_mouse_id_entropy_mean']:.6g}",
        f"- Top target mouse fraction mean: {summary['top_target_mouse_id_fraction_mean']:.6g}",
        "",
        "Raw edge weights and mass-adjusted weights are preserved for future unbalanced transport and downstream pseudo-pair supervision.",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    start = time.monotonic()
    args = parse_args()
    config = load_config(args.config)
    if config["full_m3"]["enabled"]:
        raise RuntimeError("Refusing to run while full_m3.enabled is true.")
    if config["full_m3"].get("write_global_kernel"):
        raise RuntimeError("Refusing to build or configure a global Markov kernel.")
    dry_run = bool(args.dry_run or not args.force)
    paths = _paths(config)
    reports_dir = paths["reports_dir"]
    shards = pd.read_csv(reports_dir / "m3_full_transition_shards.csv")
    shard = select_shard(shards, args)
    output = output_paths(args.output_dir, args.source_time, args.target_time, args.source_slice_id)
    schema_columns = list(json.loads((reports_dir / "m3_full_transition_schema.json").read_text())["columns"])
    if dry_run:
        summary = dry_run_summary(shard, output["edges"])
        print("DRY_RUN True")
        print(f"SELECTED_SHARD {summary['selected_shard']}")
        print(f"SOURCE_ROWS {summary['source_rows']}")
        print(f"TARGET_ROWS {summary['target_rows']}")
        print(f"EXPECTED_EDGE_ROWS {summary['expected_edge_rows']}")
        print(f"OUTPUT_PATH {summary['output_path']}")
        return 0
    if output["edges"].exists() and not args.force:
        raise FileExistsError(f"Output exists; use --force to replace: {output['edges']}")

    with (reports_dir / "m3_time_pairs.json").open("r", encoding="utf-8") as handle:
        pairs = json.load(handle)
    pair = next(
        item
        for item in pairs
        if str(item["source_time"]) == str(args.source_time)
        and str(item["target_time"]) == str(args.target_time)
    )
    feature_groups = json.loads((reports_dir / "m3_feature_groups.json").read_text())
    retrieval = [column for group in config["full_m3"]["retrieval_feature_groups"] for column in feature_groups["feature_groups"][group]]
    rerank = [column for group in config["full_m3"]["rerank_feature_groups"] for column in feature_groups["feature_groups"][group]]
    read_columns = list(dict.fromkeys(config["input"]["metadata_columns"] + retrieval + rerank))
    source = pd.read_parquet(_slice_path(paths["m2_by_slice_dir"], args.source_slice_id), columns=read_columns)
    target = _load_target_time(paths["m2_by_slice_dir"], pair["target_slices"], read_columns)
    frame, metadata = build_pilot_edges(source, target, shard, config, feature_groups)
    diagnostics = validate_pilot_edges(frame, shard, schema_columns)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output["edges"], index=False)
    runtime = time.monotonic() - start
    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    summary = {
        "status": "PASS",
        "source_time": args.source_time,
        "target_time": args.target_time,
        "source_slice_id": args.source_slice_id,
        "source_rows": int(shard["source_rows"]),
        "target_rows": int(shard["target_time_rows"]),
        "edge_rows": len(frame),
        "backend": metadata["backend"],
        "tau_pair": metadata["tau_pair"],
        "runtime_seconds": runtime,
        "max_rss_kb": max_rss,
        "output_bytes": output["edges"].stat().st_size,
        **diagnostics,
    }
    _write_report(output["report"], summary)
    print(f"Wrote pilot edge table: {output['edges']}")
    print(f"Wrote pilot report: {output['report']}")
    print(f"STATUS {summary['status']}")
    print(f"EDGE_ROWS {summary['edge_rows']}")
    print(f"RUNTIME_SECONDS {summary['runtime_seconds']:.3f}")
    print(f"MAX_RSS_KB {summary['max_rss_kb']}")
    print(f"OUTPUT_BYTES {summary['output_bytes']}")
    print(f"TAU_PAIR {summary['tau_pair']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
