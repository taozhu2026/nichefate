#!/usr/bin/env python
"""Design full M3 sharded transition construction without executing it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config
from nichefate.transition import (
    build_full_transition_shards,
    edge_density_metrics,
    estimate_time_pair_memory,
    full_transition_schema_columns,
)


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


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sample_bytes_per_edge(prototype_dir: Path) -> float:
    import pyarrow.parquet as pq

    files = sorted(prototype_dir.glob("candidate_edges_*.parquet"))
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    bytes_ = sum(path.stat().st_size for path in files)
    return float(bytes_ / rows) if rows else 0.0


def _recommended_first_shard(shards: pd.DataFrame) -> dict[str, Any]:
    earliest_day = shards["source_day"].min()
    candidates = shards[shards["source_day"] == earliest_day]
    row = candidates.sort_values(["source_rows", "source_slice_id"]).iloc[0]
    return row.to_dict()


def _recommended_pair(time_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        time_pairs,
        key=lambda item: (int(item["source_row_count"]), int(item["target_row_count"])),
    )[0]


def _runtime_estimates(
    time_pairs: list[dict[str, Any]],
    sampled_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    sample = sampled_summary.set_index(["source_time", "target_time"])
    for pair in time_pairs:
        key = (pair["source_time"], pair["target_time"])
        sampled = sample.loc[key]
        source_scale = float(pair["source_row_count"]) / float(sampled["source_sampled_rows"])
        target_scale = float(pair["target_row_count"]) / float(sampled["target_sampled_rows"])
        estimated_seconds = float(sampled["runtime_seconds"]) * source_scale * target_scale
        rows.append(
            {
                "source_time": pair["source_time"],
                "target_time": pair["target_time"],
                "sampled_runtime_seconds": float(sampled["runtime_seconds"]),
                "estimated_exact_runtime_seconds": estimated_seconds,
                "estimated_exact_runtime_hours": estimated_seconds / 3600.0,
            }
        )
    return pd.DataFrame(rows)


def _write_design_report(
    path: Path,
    config: dict[str, Any],
    shards: pd.DataFrame,
    density: pd.DataFrame,
    storage: dict[str, float],
    first_shard: dict[str, Any],
    first_pair: dict[str, Any],
) -> None:
    lines = [
        "# M3 Full Transition Design",
        "",
        "This report designs full M3 edge-shard construction only. It does not run",
        "full M3 edge construction, assemble a global Markov transition matrix P,",
        "run GPCCA, compute fate probabilities, train downstream models, or run",
        "regulator analysis.",
        "",
        "## Edge Shards Versus Global Kernel",
        "",
        "- Full M3 edge-shard construction writes source-target candidate edge shards.",
        "- Later global Markov P assembly will consume completed edge shards.",
        "- `row_normalized_transition_prob` is local to a source niche candidate set, not the full global Markov transition matrix P.",
        "- `raw_edge_weight` and `mass_adjusted_weight` are preserved for future unbalanced transport and downstream pseudo-pair supervision.",
        "- Full global P assembly and global row-stochastic validation are later-stage work.",
        "",
        "## Shard Plan",
        "",
        f"- Execution mode: {config['full_m3']['execution_mode']}",
        f"- Full execution enabled: {config['full_m3']['enabled']}",
        f"- Sharding strategy: {config['full_m3']['sharding_strategy']}",
        f"- Shards planned: {len(shards)}",
        f"- Expected edge rows: {int(shards['expected_edge_rows'].sum())}",
        f"- Candidate K: {config['full_m3']['candidate_k']}",
        f"- Candidate K mode: {config['full_m3']['candidate_k_mode']}",
        f"- Estimated parquet bytes per edge: {storage['bytes_per_edge']:.3f}",
        f"- Estimated total parquet bytes: {storage['estimated_total_bytes']:.0f}",
        "",
        "## Recommended Pilots",
        "",
        "These are recommendations only; no pilot shard or full time-pair pilot is executed in this stage.",
        "",
        "- First pilot shard:",
        f"  - source_time: {first_shard['source_time']}",
        f"  - target_time: {first_shard['target_time']}",
        f"  - source_slice_id: {first_shard['source_slice_id']}",
        f"  - source_slice_file: {first_shard['source_slice_file']}",
        f"  - source_rows: {int(first_shard['source_rows'])}",
        f"  - expected_edge_rows: {int(first_shard['expected_edge_rows'])}",
        "- First full time-pair pilot:",
        f"  - source_time: {first_pair['source_time']}",
        f"  - target_time: {first_pair['target_time']}",
        f"  - source_rows: {int(first_pair['source_row_count'])}",
        f"  - target_rows: {int(first_pair['target_row_count'])}",
        "",
        "## Fixed-K Density",
        "",
    ]
    for row in density.to_dict("records"):
        lines.append(
            "- "
            f"{row['source_time']} -> {row['target_time']}: "
            f"target_pool={int(row['target_pool_size'])}, "
            f"K/target={row['k_over_target_pool']:.6g}"
        )
    lines.extend(
        [
            "",
            "Fixed K is kept for v1 comparability. Future pilots should inspect KNN",
            "kth-distance distributions and target-slice entropy to detect density",
            "or batch effects before considering adaptive K.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_risk_report(
    path: Path,
    config: dict[str, Any],
    memory: pd.DataFrame,
    runtime: pd.DataFrame,
    density: pd.DataFrame,
) -> None:
    lines = [
        "# M3 Full Transition Risk Assessment",
        "",
        "Runtime estimates are approximate sampled-preflight extrapolations and may underestimate exact KNN runtime at full target-pool scale.",
        "",
        "## Compute Risk",
        "",
        "- Exact KNN complexity: `O(N_source x N_target x D_retrieval)`.",
        "- `sklearn_exact` is acceptable for sampled preflight and pilot shards.",
        "- `sklearn_exact` may be a bottleneck for the largest full adjacent-pair searches.",
        "- FAISS, hnswlib, and pynndescent remain future ANN backend options.",
        "- No new dependencies should be added in this design stage.",
        "",
        "## Runtime Estimates",
        "",
    ]
    for row in runtime.to_dict("records"):
        lines.append(
            "- "
            f"{row['source_time']} -> {row['target_time']}: "
            f"sampled={row['sampled_runtime_seconds']:.3f}s, "
            f"approx_full_exact={row['estimated_exact_runtime_hours']:.2f}h"
        )
    lines.extend(["", "## Parallel Memory Risk", ""])
    for row in memory.to_dict("records"):
        lines.append(
            "- "
            f"{row['source_time']} -> {row['target_time']}: "
            f"target_retrieval={row['target_retrieval_matrix_gb']:.3f} GiB, "
            f"target_rerank={row['target_rerank_matrix_gb']:.3f} GiB, "
            f"source_shard={row['source_shard_matrix_gb']:.3f} GiB, "
            f"per_worker={row['per_worker_memory_gb']:.3f} GiB, "
            f"safe_concurrency={int(row['safe_single_node_concurrency'])}"
        )
    lines.extend(
        [
            "",
            f"- Memory warning threshold: {config['full_m3']['max_memory_gb_warning']} GiB.",
            "- Current recommended execution mode is sequential for the first pilot.",
            "- Python multiprocessing is not recommended initially because target matrices may be duplicated per worker.",
            "- Future full execution should prefer Slurm or array-job style execution with a concurrency cap.",
            "- Future optimization may use memmap or shared target matrices.",
            "",
            "## Fixed-K Risk",
            "",
        ]
    )
    for row in density.to_dict("records"):
        lines.append(
            "- "
            f"{row['source_time']} -> {row['target_time']}: "
            f"target_pool_size={int(row['target_pool_size'])}, "
            f"K/target_pool={row['k_over_target_pool']:.6g}, "
            f"expected_candidate_edge_density={row['expected_candidate_edge_density']:.6g}"
        )
    lines.extend(
        [
            "",
            "- Fixed K can introduce density bias across target pools.",
            "- KNN kth-distance distribution should be checked in a future pilot.",
            "- Target-slice entropy should be checked as a batch-effect diagnostic.",
            "- Candidate target collapse to one target slice or one target sample should be a warning, not an automatic failure.",
            "",
            "## Execution Strategy",
            "",
            "- First: one serial pilot shard.",
            "- Second: one full time-pair pilot using the smallest adjacent-pair workload.",
            "- Later full M3: Slurm array or job-array style execution with explicit concurrency cap.",
            "- Avoid unrestricted multiprocessing on a single node.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_schema(path: Path, config: dict[str, Any]) -> None:
    payload = {
        "schema_version": "m3_full_transition_design_v1",
        "scope": "edge_shard_schema_only",
        "write_global_kernel": bool(config["full_m3"]["write_global_kernel"]),
        "row_normalization_scope": config["full_m3"]["row_normalization_scope"],
        "columns": full_transition_schema_columns(),
        "notes": [
            "row_normalized_transition_prob is local to each source candidate set",
            "global Markov P assembly is later-stage work",
            "raw_edge_weight and mass_adjusted_weight are preserved",
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_validation_plan(path: Path, shards: pd.DataFrame) -> None:
    lines = [
        "# Full M3 Validation Plan",
        "",
        "## Per-Shard Checks",
        "",
        "- output exists",
        "- row count equals `source_rows x K`",
        "- every source anchor has exactly K candidates",
        "- no missing source or target metadata",
        "- no NaN or infinite values in evidence, cost, weights, or probability",
        "- local transition probabilities sum to 1 per source anchor",
        "- time direction and time delta are correct",
        "- source and target slice/sample metadata are preserved",
        "- raw and mass-adjusted weights are present",
        "- barcode and pseudotime placeholders are present",
        "- scaling diagnostics are present",
        "",
        "## Per-Time-Pair Checks",
        "",
        "- all source slices completed",
        "- total edge rows equal source-time rows x K",
        "- no failed shards",
        "- row-sum, probability entropy, and top1 probability summaries",
        "- target slice and target sample selection summaries",
        "- evidence component distributions",
        "",
        "## Diagnostic Warnings",
        "",
        "- target_slice_entropy per source slice",
        "- target_mouse_entropy per source slice",
        "- top_target_slice_fraction",
        "- top_target_mouse_fraction",
        "- warn if candidate targets collapse to one target slice or one target sample",
        "",
        "## Global Checks",
        "",
        "- all adjacent pairs completed",
        "- final time point appears only as target",
        f"- expected shard count: {len(shards)}",
        f"- expected full edge rows: {int(shards['expected_edge_rows'].sum())}",
        "- failed shard file is empty",
        "- completed shard table is complete",
        "- no schema mismatches",
        "- no M3-core dataset-specific logic",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    full = config["full_m3"]
    if full["enabled"] or full["execution_mode"] != "design_only":
        raise RuntimeError("This script is design-only and refuses full M3 execution.")

    paths = _paths(config)
    reports_dir = Path(full["summary_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    time_pairs = _load_json(reports_dir / "m3_time_pairs.json")
    m2_summary = pd.read_csv(paths["m2_summary"])
    m2_schema = _load_json(paths["m2_schema"])
    sampled_summary = pd.read_csv(reports_dir / "m3_sample_candidate_edges_summary.csv")
    sampled_qc = pd.read_csv(reports_dir / "m3_sampled_transition_preflight_qc.csv")
    if not bool(sampled_qc["ok"].all()):
        raise RuntimeError("Sampled transition preflight QC must pass before full design.")

    k = int(full["candidate_k"])
    shards = build_full_transition_shards(time_pairs, m2_summary, k)
    density = edge_density_metrics(time_pairs, k)
    feature_groups = _load_json(reports_dir / "m3_feature_groups.json")
    retrieval_dims = len(feature_groups["retrieval_feature_columns"])
    rerank_dims = len(
        {
            column
            for columns in feature_groups["rerank_feature_columns"].values()
            for column in columns
        }
    )
    memory = estimate_time_pair_memory(
        time_pairs,
        shards,
        retrieval_dims,
        rerank_dims,
        float(full["max_memory_gb_warning"]),
    )
    runtime = _runtime_estimates(time_pairs, sampled_summary)
    bytes_per_edge = _sample_bytes_per_edge(paths["prototype_dir"])
    storage = {
        "bytes_per_edge": bytes_per_edge,
        "expected_total_edge_rows": float(shards["expected_edge_rows"].sum()),
        "estimated_total_bytes": bytes_per_edge * float(shards["expected_edge_rows"].sum()),
    }
    first_shard = _recommended_first_shard(shards)
    first_pair = _recommended_pair(time_pairs)

    shard_path = reports_dir / "m3_full_transition_shards.csv"
    shards.to_csv(shard_path, index=False)
    _write_schema(reports_dir / "m3_full_transition_schema.json", config)
    _write_design_report(
        reports_dir / "m3_full_transition_design.md",
        config,
        shards,
        density,
        storage,
        first_shard,
        first_pair,
    )
    _write_risk_report(
        reports_dir / "m3_full_transition_risk_assessment.md",
        config,
        memory,
        runtime,
        density,
    )
    _write_validation_plan(reports_dir / "m3_full_transition_validation_plan.md", shards)
    print(f"Wrote full transition shard plan: {shard_path}")
    print(f"SHARDS {len(shards)}")
    print(f"EXPECTED_EDGE_ROWS {int(shards['expected_edge_rows'].sum())}")
    print(f"FIRST_PILOT_SHARD {first_shard['source_time']}->{first_shard['target_time']} {first_shard['source_slice_id']}")
    print(f"FIRST_TIME_PAIR_PILOT {first_pair['source_time']}->{first_pair['target_time']}")
    print("DESIGN_ONLY True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
