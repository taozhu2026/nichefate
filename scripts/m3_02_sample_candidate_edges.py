#!/usr/bin/env python
"""Sample adjacent-time M3 candidate edges and local transition probabilities."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config
from nichefate.transition import (
    build_candidate_neighbors,
    combine_scaled_evidence,
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
    return parser.parse_args()


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        key: Path(value)
        for key, value in config["paths"].items()
        if isinstance(value, str) and value.startswith("/")
    }


def _safe_token(value: object) -> str:
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")
    return token or "time"


def _slice_file(root: Path, slice_id: str) -> Path:
    return root / slice_id / f"m2_representation_{slice_id}.parquet"


def _row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _allocate_sample_counts(
    paths: list[Path],
    max_rows: int,
    rng: np.random.Generator,
) -> list[int]:
    counts = np.array([_row_count(path) for path in paths], dtype=int)
    total = int(counts.sum())
    if total <= max_rows:
        return counts.tolist()
    probs = counts / total
    sampled = rng.multinomial(max_rows, probs)
    sampled = np.minimum(sampled, counts)
    remaining = max_rows - int(sampled.sum())
    while remaining > 0:
        capacity = counts - sampled
        available = np.flatnonzero(capacity > 0)
        if len(available) == 0:
            break
        choice = int(rng.choice(available))
        sampled[choice] += 1
        remaining -= 1
    return sampled.astype(int).tolist()


def _load_sample(
    root: Path,
    slice_ids: list[str],
    columns: list[str],
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    paths = [_slice_file(root, slice_id) for slice_id in slice_ids]
    counts = _allocate_sample_counts(paths, max_rows, rng)
    frames = []
    for path, count in zip(paths, counts, strict=False):
        if count <= 0:
            continue
        frame = pd.read_parquet(path, columns=columns)
        if count < len(frame):
            frame = frame.sample(n=count, random_state=int(rng.integers(0, 2**31 - 1)))
        frames.append(frame)
    if not frames:
        raise ValueError("Sampling produced no rows.")
    return pd.concat(frames, ignore_index=True)


def _write_edges(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return path
    except Exception:  # noqa: BLE001
        csv_path = path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        return csv_path


def _edge_metadata(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    pair: dict[str, Any],
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
            "time_delta": float(pair["time_delta"]),
            "source_slice_id": src["slice_id"].astype(str).to_numpy(),
            "target_slice_id": tgt["slice_id"].astype(str).to_numpy(),
            "source_slice_file": src["slice_file"].astype(str).to_numpy(),
            "target_slice_file": tgt["slice_file"].astype(str).to_numpy(),
            "source_mouse_id": src["mouse_id"].astype(str).to_numpy(),
            "target_mouse_id": tgt["mouse_id"].astype(str).to_numpy(),
            "evidence_mode": "pseudo_lineage",
        }
    )


def _summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    expected_edges = int(rows[0]["expected_edge_count"]) if rows else 0
    actual_edges = sum(int(row["edge_count"]) for row in rows)
    lines = [
        "# M3 Sample Candidate Edges Summary",
        "",
        f"- Expected sampled edge count upper bound: {expected_edges}",
        f"- Actual sampled edge count: {actual_edges}",
        "- Feature matrices are standardized before KNN retrieval.",
        "- `combined_cost` is computed only from scaled evidence columns.",
        "- `row_normalized_transition_prob` is a local candidate-set transition probability, not the full global Markov transition matrix P.",
        "- `raw_edge_weight` and `mass_adjusted_weight` are preserved for future unbalanced transport and Branched NicheFlow pseudo-pair supervision.",
        "- Candidate construction is sample-aware and metadata-preserving, not longitudinal sample-paired in v1.",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"{row['source_time']} -> {row['target_time']}: "
            f"{row['edge_count']} edges, tau={row['tau_pair']:.6g}, "
            f"backend={row['neighbor_backend']}, runtime={row['runtime_seconds']:.3f}s"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    start_all = time.monotonic()
    args = parse_args()
    config = load_config(args.config)
    paths = _paths(config)
    reports_dir = paths["reports_dir"]
    prototype_dir = paths["prototype_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    prototype_dir.mkdir(parents=True, exist_ok=True)

    with (reports_dir / "m3_time_pairs.json").open("r", encoding="utf-8") as handle:
        time_pairs = json.load(handle)
    with (reports_dir / "m3_feature_groups.json").open("r", encoding="utf-8") as handle:
        feature_groups = json.load(handle)
    groups = feature_groups["feature_groups"]
    retrieval_groups = list(config["candidate_edges"]["retrieval_feature_groups"])
    rerank_groups = list(config["cost"]["rerank_feature_groups"])
    retrieval_columns = list(dict.fromkeys(
        column for group in retrieval_groups for column in groups[group]
    ))
    rerank_columns = list(dict.fromkeys(
        column for group in rerank_groups for column in groups[group]
    ))
    metadata_columns = list(config["input"]["metadata_columns"])
    read_columns = list(dict.fromkeys(metadata_columns + retrieval_columns + rerank_columns))
    edge_cfg = config["candidate_edges"]
    cost_cfg = config["cost"]
    summary_rows = []
    scaling_rows = []

    for pair_index, pair in enumerate(time_pairs):
        start_pair = time.monotonic()
        source = _load_sample(
            paths["m2_by_slice_dir"],
            pair["source_slices"],
            read_columns,
            int(edge_cfg["max_source_niches_per_pair"]),
            int(edge_cfg["random_seed"]) + pair_index * 2,
        )
        target = _load_sample(
            paths["m2_by_slice_dir"],
            pair["target_slices"],
            read_columns,
            int(edge_cfg["max_target_niches_per_pair"]),
            int(edge_cfg["random_seed"]) + pair_index * 2 + 1,
        )
        source_retrieval, target_retrieval, standardize_stats = standardize_feature_matrices(
            source[retrieval_columns].to_numpy(dtype=float),
            target[retrieval_columns].to_numpy(dtype=float),
            float(cost_cfg["min_scale"]),
        )
        neighbors = build_candidate_neighbors(
            source_retrieval,
            target_retrieval,
            int(edge_cfg["k_candidates"]),
            backend=edge_cfg["neighbor_backend"],
            metric=edge_cfg.get("retrieval_metric", "euclidean"),
            chunk_size=int(edge_cfg.get("numpy_chunk_size", 512)),
            random_seed=int(edge_cfg["random_seed"]),
        )
        source_idx = np.repeat(np.arange(len(source)), neighbors.indices.shape[1])
        target_idx = neighbors.indices.reshape(-1)
        frame = _edge_metadata(source, target, source_idx, target_idx, pair)
        frame["candidate_rank"] = np.tile(np.arange(neighbors.indices.shape[1]), len(source))
        frame["retrieval_distance"] = neighbors.distances.reshape(-1)

        scaled_column_by_group = {}
        for group in rerank_groups:
            evidence_name = GROUP_TO_EVIDENCE[group]
            raw_col = f"raw_{evidence_name}_distance"
            scaled_col = f"scaled_{evidence_name}_distance"
            metric = "l1" if group == "cell_type_composition" else "euclidean"
            frame[raw_col] = pairwise_row_distance(
                source,
                target,
                source_idx,
                target_idx,
                groups[group],
                metric=metric,
            )
            frame[scaled_col], stats = safe_scale_vector(
                frame[raw_col],
                min_scale=float(cost_cfg["min_scale"]),
            )
            scaled_column_by_group[group] = scaled_col
            scaling_rows.append(
                {
                    "source_time": pair["source_time"],
                    "target_time": pair["target_time"],
                    "evidence_group": group,
                    **stats,
                }
            )

        frame["raw_pseudotime_score"] = 0.0
        frame["raw_barcode_score"] = 0.0
        frame["scaled_pseudotime_score"] = 0.0
        frame["scaled_barcode_score"] = 0.0
        frame["combined_cost"] = combine_scaled_evidence(
            frame,
            cost_cfg["evidence_weights"],
            scaled_column_by_group,
        )
        tau = pair_adaptive_temperature(
            frame["combined_cost"],
            float(cost_cfg["min_temperature"]),
        )
        frame["tau_pair"] = tau
        exponent = np.clip(-frame["combined_cost"].to_numpy(dtype=float) / tau, -700, 700)
        frame["raw_edge_weight"] = np.exp(exponent)
        frame["source_mass"] = 1.0
        frame["target_mass"] = 1.0
        frame["growth_prior"] = 1.0
        frame["unbalanced_weight"] = 1.0
        frame["mass_adjusted_weight"] = frame["raw_edge_weight"]
        frame["row_normalized_transition_prob"] = row_normalize_weights(frame)
        diagnostics = transition_probability_diagnostics(frame)

        output_name = (
            f"candidate_edges_{_safe_token(pair['source_time'])}_"
            f"to_{_safe_token(pair['target_time'])}.parquet"
        )
        output_path = _write_edges(frame, prototype_dir / output_name)
        runtime = time.monotonic() - start_pair
        summary_rows.append(
            {
                "source_time": pair["source_time"],
                "target_time": pair["target_time"],
                "source_day": pair["source_day"],
                "target_day": pair["target_day"],
                "time_delta": pair["time_delta"],
                "source_sampled_rows": len(source),
                "target_sampled_rows": len(target),
                "source_matrix_shape": f"{source_retrieval.shape[0]}x{source_retrieval.shape[1]}",
                "target_matrix_shape": f"{target_retrieval.shape[0]}x{target_retrieval.shape[1]}",
                "retrieval_feature_groups": ",".join(retrieval_groups),
                "rerank_feature_groups": ",".join(rerank_groups),
                "neighbor_backend": neighbors.backend,
                "candidate_k": neighbors.indices.shape[1],
                "expected_edge_count": len(time_pairs)
                * int(edge_cfg["max_source_niches_per_pair"])
                * int(edge_cfg["k_candidates"]),
                "pair_expected_edge_upper_bound": int(edge_cfg["max_source_niches_per_pair"])
                * int(edge_cfg["k_candidates"]),
                "edge_count": len(frame),
                "tau_pair": tau,
                "combined_cost_min": float(frame["combined_cost"].min()),
                "combined_cost_median": float(frame["combined_cost"].median()),
                "combined_cost_max": float(frame["combined_cost"].max()),
                "raw_edge_weight_min": float(frame["raw_edge_weight"].min()),
                "raw_edge_weight_median": float(frame["raw_edge_weight"].median()),
                "raw_edge_weight_max": float(frame["raw_edge_weight"].max()),
                "zero_variance_retrieval_columns": standardize_stats["zero_variance_columns"],
                "output_path": str(output_path),
                "runtime_seconds": runtime,
                **diagnostics,
            }
        )
        print(
            f"WROTE_EDGES {pair['source_time']}->{pair['target_time']} "
            f"{len(frame)} {output_path}"
        )

    summary_csv = reports_dir / "m3_sample_candidate_edges_summary.csv"
    summary_md = reports_dir / "m3_sample_candidate_edges_summary.md"
    scaling_csv = reports_dir / "m3_sample_candidate_edges_scaling.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    pd.DataFrame(scaling_rows).to_csv(scaling_csv, index=False)
    _summary_md(summary_md, summary_rows)
    print(f"Wrote candidate edge summary CSV: {summary_csv}")
    print(f"Wrote candidate edge summary report: {summary_md}")
    print(f"Wrote scaling summary CSV: {scaling_csv}")
    print(f"TIME_PAIRS {len(summary_rows)}")
    print(f"TOTAL_EDGES {sum(int(row['edge_count']) for row in summary_rows)}")
    print(f"WALL_SECONDS {time.monotonic() - start_all:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
