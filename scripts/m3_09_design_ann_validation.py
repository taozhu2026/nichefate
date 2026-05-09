#!/usr/bin/env python
"""Design exact-vs-ANN validation for M3 without running full construction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config


VALIDATION_METRICS = [
    "recall@K",
    "top1 agreement",
    "mean Jaccard overlap of candidate sets",
    "distance rank correlation",
    "raw_edge_weight drift",
    "row_normalized_transition_prob drift",
    "row entropy delta",
    "top1 probability delta",
    "target slice entropy delta",
    "target mouse entropy delta",
    "runtime ratio",
    "memory ratio",
]

VALIDATION_COLUMNS = [
    "recommended",
    "validation_rank",
    "source_time",
    "target_time",
    "source_slice_id",
    "source_slice_file",
    "source_rows",
    "target_rows",
    "candidate_k",
    "expected_edge_rows",
    "validation_sample_size",
    "sample_required",
    "reviewed_time_pair",
    "conservative_projection_hours",
    "why_suitable",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_transition_kernel.yaml")
    parser.add_argument(
        "--shards-csv",
        type=Path,
        default=Path("/home/zhutao/scratch/nichefate/m3/reports/m3_full_transition_shards.csv"),
    )
    parser.add_argument(
        "--pilot-metrics-csv",
        type=Path,
        default=Path(
            "/home/zhutao/scratch/nichefate/m3/reports/"
            "m3_D21_to_D35_pilot_review_metrics.csv"
        ),
    )
    parser.add_argument(
        "--runtime-projection-csv",
        type=Path,
        default=Path("/home/zhutao/scratch/nichefate/m3/reports/m3_backend_runtime_projection.csv"),
    )
    return parser.parse_args()


def _reports_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["reports_dir"])


def _assert_no_ssd(config: dict[str, Any]) -> None:
    if bool(config["paths"].get("use_ssd", False)):
        raise RuntimeError("Refusing M3-09 design while paths.use_ssd is true.")
    for value in config.get("paths", {}).values():
        if isinstance(value, str) and value.startswith("/ssd"):
            raise RuntimeError(f"Refusing to use /ssd path in M3-09 design: {value}")


def _reviewed_pair_keys(pilot_metrics: pd.DataFrame) -> set[tuple[str, str]]:
    if not {"source_time", "target_time"} <= set(pilot_metrics.columns):
        return set()
    return {
        (str(row["source_time"]), str(row["target_time"]))
        for row in pilot_metrics.to_dict("records")
    }


def design_validation_shards(
    shards: pd.DataFrame,
    pilot_metrics: pd.DataFrame,
    runtime_projection: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    required = {
        "source_time",
        "target_time",
        "source_slice_id",
        "source_slice_file",
        "source_rows",
        "target_time_rows",
        "candidate_k",
        "expected_edge_rows",
    }
    missing = sorted(required - set(shards.columns))
    if missing:
        raise KeyError(f"Shard table is missing required columns: {missing}")
    reviewed_pairs = _reviewed_pair_keys(pilot_metrics)
    work = shards.copy()
    work["source_time"] = work["source_time"].astype(str)
    work["target_time"] = work["target_time"].astype(str)
    work["target_rows"] = pd.to_numeric(work["target_time_rows"], errors="raise").astype(int)
    work["source_rows"] = pd.to_numeric(work["source_rows"], errors="raise").astype(int)
    work["candidate_k"] = pd.to_numeric(work["candidate_k"], errors="raise").astype(int)
    work["expected_edge_rows"] = pd.to_numeric(work["expected_edge_rows"], errors="raise").astype(int)
    work["reviewed_time_pair"] = [
        (source, target) in reviewed_pairs
        for source, target in zip(work["source_time"], work["target_time"], strict=True)
    ]
    if bool(work["reviewed_time_pair"].any()):
        work = work[work["reviewed_time_pair"]].copy()
    runtime_cols = ["source_time", "target_time", "conservative_projection_hours"]
    if set(runtime_cols) <= set(runtime_projection.columns):
        runtime = runtime_projection[runtime_cols].copy()
        runtime["source_time"] = runtime["source_time"].astype(str)
        runtime["target_time"] = runtime["target_time"].astype(str)
        runtime["conservative_projection_hours"] = pd.to_numeric(
            runtime["conservative_projection_hours"],
            errors="coerce",
        )
        work = work.merge(runtime, on=["source_time", "target_time"], how="left")
    else:
        work["conservative_projection_hours"] = pd.NA
    sample_cap = int(config["candidate_edges"].get("max_source_niches_per_pair", 5000))
    work["validation_sample_size"] = work["source_rows"].clip(upper=sample_cap).astype(int)
    work["sample_required"] = work["source_rows"] > work["validation_sample_size"]
    work = work.sort_values(
        ["source_rows", "target_rows", "source_slice_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    work["validation_rank"] = range(1, len(work) + 1)
    work["recommended"] = work["validation_rank"] == 1
    work["why_suitable"] = work.apply(_why_suitable, axis=1)
    return work[VALIDATION_COLUMNS]


def _why_suitable(row: pd.Series) -> str:
    pieces = [
        "reviewed pilot time pair" if bool(row["reviewed_time_pair"]) else "smallest planned shard",
        "small source slice bounds exact validation cost",
        "fixed K preserves candidate-set comparability",
    ]
    if bool(row["sample_required"]):
        pieces.append("source sampling keeps validation bounded")
    else:
        pieces.append("full source slice can be validated")
    return "; ".join(pieces)


def write_validation_plan(
    path: Path,
    validation_shards: pd.DataFrame,
    pilot_metrics: pd.DataFrame,
) -> None:
    recommended = validation_shards[validation_shards["recommended"]].iloc[0].to_dict()
    metric_row = pilot_metrics.iloc[0].to_dict() if not pilot_metrics.empty else {}
    lines = [
        "# M3 Exact-vs-ANN Validation Plan",
        "",
        "This is a design report only. It does not run full M3, build new edge shards, assemble global Markov P, run GPCCA, compute fate probabilities, run Branched NicheFlow, M5, or regulator analysis.",
        "",
        "## Recommended Validation Shard",
        f"- source_time: {recommended['source_time']}",
        f"- target_time: {recommended['target_time']}",
        f"- source_slice_id: {recommended['source_slice_id']}",
        f"- source_slice_file: {recommended['source_slice_file']}",
        f"- source_rows: {int(recommended['source_rows'])}",
        f"- target_rows: {int(recommended['target_rows'])}",
        f"- candidate_k: {int(recommended['candidate_k'])}",
        f"- expected edge rows: {int(recommended['expected_edge_rows'])}",
        f"- validation sample size: {int(recommended['validation_sample_size'])}",
        f"- sample required: {bool(recommended['sample_required'])}",
        f"- why suitable: {recommended['why_suitable']}",
        "",
        "## Pilot Review Context",
        f"- completed_shards: {metric_row.get('completed_shards', 'unknown')}",
        f"- failed_shards: {metric_row.get('failed_shards', 'unknown')}",
        f"- row_sum_abs_error_max: {metric_row.get('row_sum_abs_error_max', 'unknown')}",
        f"- certainty_classification: {metric_row.get('certainty_classification', 'unknown')}",
        "",
        "## Validation Metrics",
    ]
    lines.extend([f"- {metric}" for metric in VALIDATION_METRICS])
    lines.extend(
        [
            "",
            "## Diagnostic Acceptance Targets",
            "- recall@30 >= 0.8 is a soft target.",
            "- top1 agreement >= 0.8 is a soft target.",
            "- No severe shift in entropy, top1 probability, target-slice collapse, or target-mouse collapse diagnostics.",
            "- Thresholds are review gates for backend readiness, not hard biological claims.",
            "",
            "## Required Comparison Runs",
            "- Run `sklearn_exact` on the recommended shard or bounded validation sample.",
            "- Run one usable ANN backend on the same standardized retrieval matrix and candidate K.",
            "- Compare candidate sets before approving ANN for larger time pairs.",
            "- Do not run D3->D9 or D9->D21 full construction until backend validation is reviewed.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    _assert_no_ssd(config)
    reports_dir = _reports_dir(config)
    reports_dir.mkdir(parents=True, exist_ok=True)
    shards = pd.read_csv(args.shards_csv)
    pilot_metrics = pd.read_csv(args.pilot_metrics_csv)
    runtime_projection = pd.read_csv(args.runtime_projection_csv)
    validation_shards = design_validation_shards(
        shards,
        pilot_metrics,
        runtime_projection,
        config,
    )
    csv_path = reports_dir / "m3_ann_validation_shards.csv"
    md_path = reports_dir / "m3_ann_validation_plan.md"
    validation_shards.to_csv(csv_path, index=False)
    write_validation_plan(md_path, validation_shards, pilot_metrics)
    rec = validation_shards[validation_shards["recommended"]].iloc[0]
    print(f"Wrote ANN validation shard table: {csv_path}")
    print(f"Wrote ANN validation plan: {md_path}")
    print(
        "RECOMMENDED_VALIDATION_SHARD "
        f"{rec['source_time']}->{rec['target_time']} {rec['source_slice_id']}"
    )
    print("DESIGN_ONLY True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
