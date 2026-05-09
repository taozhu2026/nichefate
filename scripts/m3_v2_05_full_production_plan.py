#!/usr/bin/env python
"""Write the M3-v2 full production plan and versioned output contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


OUTPUT_ROOT = Path("/home/zhutao/scratch/nichefate/m3_v2_production_plan")
M3_V1_EDGE_ROOT = Path("/home/zhutao/scratch/nichefate/m3/full_by_shard")
M3_V2_PACKAGE_REVIEW = Path("/home/zhutao/scratch/nichefate/m3_v2_package_review")
MODE_SCHEMA = M3_V2_PACKAGE_REVIEW / "m3_v2_constrained_mode_schema.json"

MODE_NAME = "constrained_v1prior_sharpening"
VARIANT = "v1prior_1.0_tau_0.5_top10"
EXPECTED_CANDIDATE_K = 30
RETAINED_TOP_K = 10
PILOT_RUNTIME_SECONDS = 78.85
PILOT_EDGE_COUNT = 1_500_000
PILOT_PEAK_RSS_GIB = 3.41


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    return parser.parse_args()


def validate_output_root(path: Path) -> None:
    resolved = path.resolve()
    protected = [
        Path("/home/zhutao/scratch/nichefate/m3").resolve(),
        Path("/home/zhutao/scratch/nichefate/m4a").resolve(),
        Path("/home/zhutao/scratch/nichefate/m4b").resolve(),
        Path("/home/zhutao/scratch/nichefate/m4c").resolve(),
        Path("/home/zhutao/scratch/nichefate/m3_v2_pilot").resolve(),
        Path("/home/zhutao/scratch/nichefate/m3_v2_pilot_tuning").resolve(),
        Path("/home/zhutao/scratch/nichefate/m3_v2_pilot_confirmatory").resolve(),
    ]
    for root in protected:
        if resolved == root or root in resolved.parents:
            raise ValueError(f"Refusing to write M3-v2 production plan under protected path: {resolved}")


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    validate_output_root(output_root)
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return {"root": output_root, "reports": reports}


def read_mode_schema() -> dict[str, Any]:
    if not MODE_SCHEMA.is_file():
        raise FileNotFoundError(f"Missing M3-v2 constrained mode schema: {MODE_SCHEMA}")
    schema = json.loads(MODE_SCHEMA.read_text())
    if schema.get("mode_name") != MODE_NAME:
        raise ValueError(f"Unexpected mode name in schema: {schema.get('mode_name')}")
    return schema


def collect_shard_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(M3_V1_EDGE_ROOT.glob("*/*/candidate_edges_*.parquet")):
        parquet = pq.ParquetFile(path)
        time_pair = path.parent.parent.name
        source_time, target_time = time_pair.split("_to_")
        rows.append(
            {
                "time_pair": time_pair,
                "source_time": source_time,
                "target_time": target_time,
                "source_slice_id": path.parent.name,
                "path": str(path),
                "row_count": int(parquet.metadata.num_rows),
                "estimated_source_count": int(round(parquet.metadata.num_rows / EXPECTED_CANDIDATE_K)),
                "size_bytes": int(path.stat().st_size),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No M3-v1 candidate edge shards found under {M3_V1_EDGE_ROOT}")
    return pd.DataFrame(rows)


def time_pair_summary(shards: pd.DataFrame) -> pd.DataFrame:
    summary = (
        shards.groupby(["time_pair", "source_time", "target_time"], sort=True)
        .agg(
            shard_count=("path", "count"),
            m3_v1_edge_count=("row_count", "sum"),
            estimated_source_count=("estimated_source_count", "sum"),
            m3_v1_size_bytes=("size_bytes", "sum"),
        )
        .reset_index()
    )
    summary["m3_v2_retained_edge_count_top10"] = summary["estimated_source_count"] * RETAINED_TOP_K
    summary["estimated_m3_v2_size_bytes"] = (
        summary["m3_v1_size_bytes"] * (RETAINED_TOP_K / EXPECTED_CANDIDATE_K) * 1.6
    ).round().astype(int)
    return summary


def resource_estimate(summary: pd.DataFrame) -> dict[str, Any]:
    total_v1_edges = int(summary["m3_v1_edge_count"].sum())
    total_sources = int(summary["estimated_source_count"].sum())
    retained_edges = int(summary["m3_v2_retained_edge_count_top10"].sum())
    seconds_per_million = PILOT_RUNTIME_SECONDS / (PILOT_EDGE_COUNT / 1_000_000)
    estimated_runtime = seconds_per_million * (total_v1_edges / 1_000_000)
    conservative_runtime = estimated_runtime * 1.35
    return {
        "expected_time_pairs": summary["time_pair"].tolist(),
        "m3_v1_shard_count": int(summary["shard_count"].sum()),
        "estimated_full_source_count": total_sources,
        "m3_v1_candidate_edge_count": total_v1_edges,
        "m3_v2_retained_edge_count_after_top10": retained_edges,
        "pilot_seconds_per_1m_edges": seconds_per_million,
        "estimated_total_runtime_seconds": estimated_runtime,
        "conservative_total_runtime_seconds": conservative_runtime,
        "estimated_peak_rss_gib": max(PILOT_PEAK_RSS_GIB, 4.0),
        "recommended_memory_gib": 8,
        "m3_v1_total_size_gib": float(summary["m3_v1_size_bytes"].sum() / 1024**3),
        "estimated_m3_v2_output_size_gib": float(summary["estimated_m3_v2_size_bytes"].sum() / 1024**3),
        "slurm_recommended": True,
    }


def input_contract_rows() -> list[dict[str, Any]]:
    return [
        {
            "input": "full M3-v1 edge shards",
            "expected_path": "/home/zhutao/scratch/nichefate/m3/full_by_shard/{time_pair}/{source_slice}/candidate_edges_*.parquet",
            "required": True,
            "join_keys": "source_anchor_id,target_anchor_id",
            "required_columns": "source_anchor_id,target_anchor_id,source_slice_id,target_slice_id,source_mouse_id,target_mouse_id,row_normalized_transition_prob",
            "expected_count": "all frozen M3-v1 shards; count computed from Parquet metadata",
        },
        {
            "input": "M2 anchor-level niche representation",
            "expected_path": "/home/zhutao/scratch/nichefate/m2/by_slice/{slice_id}/m2_representation_{slice_id}.parquet",
            "required": True,
            "join_keys": "slice_id,anchor_index -> anchor_id",
            "required_columns": "selected molecular_state, cell_type_composition, spatial_summary, topology features",
            "expected_count": "one row per M2 anchor in source/target slices",
        },
        {
            "input": "M4A global node table",
            "expected_path": "/home/zhutao/scratch/nichefate/m4a/node_table/global_node_table.parquet",
            "required": True,
            "join_keys": "anchor_id",
            "required_columns": "anchor_id,global_node_index,slice_id,anchor_index,time,mouse_id",
            "expected_count": "one row per global anchor node",
        },
        {
            "input": "M4C-v1 fate node summary",
            "expected_path": "/home/zhutao/scratch/nichefate/m4c/fate_probabilities/fate_probability_node_summary.parquet",
            "required": True,
            "join_keys": "anchor_id",
            "required_columns": "anchor_id,dominant_fate,dominant_fate_probability,normalized_plasticity_entropy",
            "expected_count": "one row per node with M4C-v1 summary",
        },
        {
            "input": "M4E node neighborhood annotation",
            "expected_path": "/home/zhutao/scratch/nichefate/m4e/neighborhood_annotation/node_neighborhood_annotation.parquet",
            "required": True,
            "join_keys": "anchor_id",
            "required_columns": "anchor_id,time_label,slice_id,mouse_id,leiden_neigh,cell_type_l1,cell_type_l3,x,y",
            "expected_count": "one row per annotated node",
        },
        {
            "input": "M4E refined endpoint mapping",
            "expected_path": "/home/zhutao/scratch/nichefate/m4e/endpoint_refinement/refined_endpoint_mapping.csv",
            "required": True,
            "join_keys": "dominant_fate/raw_terminal_macrostate",
            "required_columns": "raw_terminal_macrostate,refined_endpoint_id,refined_endpoint_label,confidence_tier_after_refinement",
            "expected_count": "one row per raw endpoint",
        },
        {
            "input": "M3-v2 constrained mode schema",
            "expected_path": "/home/zhutao/scratch/nichefate/m3_v2_package_review/m3_v2_constrained_mode_schema.json",
            "required": True,
            "join_keys": "mode_name",
            "required_columns": "mode_name,validated_pseudo_only_parameters,versioned_output_roots_if_production_runs",
            "expected_count": "one JSON object",
        },
    ]


def checklist_rows() -> list[dict[str, Any]]:
    items = [
        ("scope", "Use frozen M3-v1 candidate edges as base input", True),
        ("scope", "Do not regenerate source-target candidates", True),
        ("versioning", "Write M3-v2 shards under /home/zhutao/scratch/nichefate/m3_v2/full_by_shard/", True),
        ("versioning", "Do not overwrite M3-v1/M4A-v1/M4B-v1/M4C-v1", True),
        ("algorithm", "Apply constrained_v1prior_sharpening with lambda=1 tau_scale=0.5 top_k=10", True),
        ("algorithm", "Use G_barcode=1.0 in pseudo-only production", True),
        ("qc", "Validate finite nonnegative weights and per-source row sums", True),
        ("qc", "Write shard-level QC for all shards", True),
        ("qc", "Write global v1-v2 benchmark before adoption", False),
        ("resume", "Maintain completed_shards.csv and failed_shards.txt", True),
        ("resume", "Skip existing passing shards in resume mode", True),
        ("handoff", "Keep M4A-v2 output under /home/zhutao/scratch/nichefate/m4a_v2/", False),
        ("handoff", "Keep M4C-v2 output under /home/zhutao/scratch/nichefate/m4c_v2/", False),
    ]
    return [
        {
            "category": category,
            "check_item": item,
            "required_before_full_run": required,
            "status_for_planning": "specified" if required else "deferred_until_follow_on_stage",
        }
        for category, item, required in items
    ]


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    work = frame[columns].copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: f"{float(value):.4g}" if pd.notna(value) else "NA")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in work.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_reports(paths: dict[str, Path], summary: pd.DataFrame, estimate: dict[str, Any], schema: dict[str, Any]) -> None:
    reports = paths["reports"]
    input_contract = pd.DataFrame(input_contract_rows())
    retained_edges = estimate["m3_v2_retained_edge_count_after_top10"]
    reports.joinpath("m3_v2_full_production_plan.md").write_text(
        f"""# M3-v2 Full Production Plan

## Scope

Full M3-v2 production should reweight the full existing M3-v1 candidate edge set using `{MODE_NAME}`. It must not regenerate source-target candidates unless explicitly approved later. This isolates the effect of the constrained v1-prior sharpening mode while preserving M3-v1 as the frozen pseudo-only baseline.

Expected time pairs:

- D0 -> D3
- D3 -> D9
- D9 -> D21
- D21 -> D35

## Algorithm

`P_v2(i -> j) proportional to P_v1(i -> j)^1.0 * exp(-d_state(i,j) / (tau_i * 0.5)) * G_composition(i,j) * G_spatial_topology(i,j) * G_slice_mouse(i,j) * G_barcode(i,j)`

Then keep `top_k = 10` target edges per source, renormalize per source anchor, and validate row sums. `G_barcode = 1.0` for pseudo-only production.

## Full M3-v1 Shard Summary

{markdown_table(summary, ['time_pair', 'shard_count', 'estimated_source_count', 'm3_v1_edge_count', 'm3_v2_retained_edge_count_top10'])}
"""
    )

    reports.joinpath("m3_v2_full_output_contract.md").write_text(
        f"""# M3-v2 Full Output Contract

## Versioned Output Roots

- M3-v2 edge shards: `/home/zhutao/scratch/nichefate/m3_v2/full_by_shard/`
- M3-v2 reports: `/home/zhutao/scratch/nichefate/m3_v2/reports/`
- M3-v2 figures: `/home/zhutao/scratch/nichefate/m3_v2/reports/figures/`
- M3-v2 logs: `/home/zhutao/scratch/nichefate/m3_v2/logs/`
- M4A-v2 handoff root: `/home/zhutao/scratch/nichefate/m4a_v2/`
- M4C-v2 handoff root: `/home/zhutao/scratch/nichefate/m4c_v2/`

No M3-v1, M4A-v1, M4B-v1, or M4C-v1 outputs may be overwritten. V1 and V2 must remain directly comparable.

## Shard Output Columns

Each retained M3-v2 edge shard should include source/target anchor IDs, source/target slice and mouse IDs, M3-v1 probability, M3-v2 probability, state distance, tau, composition gate, spatial/topology gate, slice/mouse gate, barcode gate, unnormalized v2 weight, and rank within source before top-k truncation.

## Production Input Contract

{markdown_table(input_contract, ['input', 'expected_path', 'required', 'join_keys', 'required_columns'])}
"""
    )

    reports.joinpath("m3_v2_full_runtime_resource_estimate.md").write_text(
        f"""# M3-v2 Full Runtime and Resource Estimate

Pilot reference: 50,000 sources, 1,500,000 edges, 78.85 seconds, peak RSS about 3.41 GiB.

## Estimated Full Run

- M3-v1 shards: {estimate['m3_v1_shard_count']}
- Estimated full source count: {estimate['estimated_full_source_count']:,}
- M3-v1 candidate edges read: {estimate['m3_v1_candidate_edge_count']:,}
- M3-v2 retained edges after top10: {retained_edges:,}
- Estimated runtime per 1M candidate edges: {estimate['pilot_seconds_per_1m_edges']:.2f} seconds
- Estimated total runtime: {estimate['estimated_total_runtime_seconds'] / 60:.1f} minutes
- Conservative total runtime: {estimate['conservative_total_runtime_seconds'] / 60:.1f} minutes
- Estimated peak RSS: {estimate['estimated_peak_rss_gib']:.1f} GiB
- Recommended memory request: {estimate['recommended_memory_gib']} GiB
- Existing M3-v1 shard size: {estimate['m3_v1_total_size_gib']:.2f} GiB
- Estimated retained M3-v2 output size: {estimate['estimated_m3_v2_output_size_gib']:.2f} GiB

Slurm is recommended for full production, preferably one array job per source-slice shard with resume-aware outputs.
"""
    )

    reports.joinpath("m3_v2_full_qc_and_validation_plan.md").write_text(
        """# M3-v2 Full QC and Validation Plan

## Shard-Level QC

- source count
- edge count before top-k
- retained edge count after top-k
- row-sum max error
- no NaN weights or probabilities
- no negative weights or probabilities
- number of sources with fewer than top_k retained targets
- v1-v2 entropy delta
- top1 probability delta
- target slice/mouse concentration
- target Leiden_neigh consistency
- target fine cluster consistency
- endpoint plausibility

## Global Summary QC

- total sources and edges
- per-time-pair source/edge counts
- entropy and top1 distributions
- slice/mouse collapse diagnostics
- endpoint plausibility by time pair
- Leiden/fine cluster consistency by time pair
- comparison against M3-v1
"""
    )

    reports.joinpath("m3_v2_full_resume_and_failure_recovery_plan.md").write_text(
        """# M3-v2 Full Resume and Failure Recovery Plan

- Process shards independently.
- Write `completed_shards.csv` with shard path, output path, row counts, QC status, and timestamp.
- Write `failed_shards.txt` with failed shard IDs and error summaries.
- Default mode is no overwrite.
- Resume mode skips an existing shard output only if its QC report passes.
- Existing shard outputs that fail QC are marked invalid and are not rerun unless an explicit overwrite/rerun flag is provided.
- Stop-on-error mode should be available for interactive validation; Slurm array mode should continue independent shards and summarize failures.
"""
    )

    reports.joinpath("m3_v2_full_v1_comparison_plan.md").write_text(
        """# M3-v2 Full V1 Comparison Plan

After full production, compare M3-v1 and M3-v2 per time pair, slice/mouse, Leiden_neigh, and refined endpoint.

Metrics:

- endpoint plausibility
- Leiden consistency
- fine cluster consistency
- entropy/top1 sharpening
- slice/mouse collapse
- target diversity
- spatial coherence if coordinates are available
- downstream M4C-v1 vs M4C-v2 after M4C-v2 exists

Post-benchmark decision categories:

- keep_v1_as_main_baseline
- keep_v1_and_v2_as_complementary
- adopt_v2_as_default_pseudo_mode
- defer_v2_until_barcode
"""
    )

    reports.joinpath("m4a_v2_handoff_contract.md").write_text(
        """# M4A-v2 Handoff Contract

M4A-v2 should consume:

- M3-v2 full edge shards
- M4A-v1 node table or equivalent global node table
- M3-v2 manifest and shard QC summary

M4A-v2 should produce:

- `P_forward_v2`
- `P_absorbing_v2` if still needed
- `W_raw_v2`
- `W_mass_adjusted_v2` if applicable
- M4A-v2 schema, report, and QC outputs

M4A-v2 must not overwrite M4A-v1. Row-sum QC must match or exceed M4A-v1 standards.
"""
    )

    reports.joinpath("m4c_v2_handoff_contract.md").write_text(
        """# M4C-v2 Handoff Contract

M4C-v2 should consume:

- `P_fate_v2` from M4A-v2
- refined endpoint mapping from M4E
- existing M4C-v1 and M4E metadata for comparison

M4C-v2 should produce:

- fate probability matrix v2
- node summary v2
- time/slice/mouse summaries
- tissue maps
- M4C-v1 vs M4C-v2 comparison report

M4C-v2 should be interpreted as a pseudo-only v2 endpoint-attraction map, not lineage-validated fate and not GPCCA-derived fate.
"""
    )

    reports.joinpath("m3_v2_safety_and_non_goals.md").write_text(
        """# M3-v2 Safety and Non-Goals

- No pyGPCCA in M3-v2 production.
- No K_gpcca.
- No BranchSBM / Branched NicheFlow.
- No barcode preprocessing.
- No M5/regulator.
- No upstream v1 overwrite.
- No data movement.
- All outputs must be versioned.
"""
    )


def write_checklist(paths: dict[str, Path]) -> pd.DataFrame:
    checklist = pd.DataFrame(checklist_rows())
    checklist.to_csv(paths["root"] / "m3_v2_full_production_checklist.csv", index=False)
    return checklist


def write_summary(paths: dict[str, Path], summary: pd.DataFrame, estimate: dict[str, Any], schema: dict[str, Any]) -> None:
    payload = {
        "mode_name": MODE_NAME,
        "interpretation": "calibrated v1-prior sharpening mode; complementary to M3-v1",
        "locked_parameters": schema["validated_pseudo_only_parameters"],
        "production_scope": "reweight full frozen M3-v1 candidate edge set without candidate regeneration",
        "time_pairs": summary["time_pair"].tolist(),
        "output_roots": {
            "m3_v2_full_by_shard": "/home/zhutao/scratch/nichefate/m3_v2/full_by_shard/",
            "m3_v2_reports": "/home/zhutao/scratch/nichefate/m3_v2/reports/",
            "m3_v2_figures": "/home/zhutao/scratch/nichefate/m3_v2/reports/figures/",
            "m3_v2_logs": "/home/zhutao/scratch/nichefate/m3_v2/logs/",
            "m4a_v2": "/home/zhutao/scratch/nichefate/m4a_v2/",
            "m4c_v2": "/home/zhutao/scratch/nichefate/m4c_v2/",
        },
        "resource_estimate": estimate,
        "full_production_execution_recommended_next": False,
        "next_engineering_step": "implement the full M3-v2 shard runner and dry-run/preflight validation, without running full production yet",
    }
    (paths["root"] / "m3_v2_full_production_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def run(output_root: Path) -> dict[str, Any]:
    paths = ensure_dirs(output_root)
    schema = read_mode_schema()
    shards = collect_shard_inventory()
    summary = time_pair_summary(shards)
    estimate = resource_estimate(summary)
    write_reports(paths, summary, estimate, schema)
    write_checklist(paths)
    write_summary(paths, summary, estimate, schema)
    return {
        "output_root": str(output_root),
        "mode_name": MODE_NAME,
        "time_pairs": summary["time_pair"].tolist(),
        "m3_v1_shards": int(summary["shard_count"].sum()),
        "estimated_sources": int(summary["estimated_source_count"].sum()),
        "m3_v1_candidate_edges": int(summary["m3_v1_edge_count"].sum()),
        "m3_v2_retained_edges_top10": int(summary["m3_v2_retained_edge_count_top10"].sum()),
        "estimated_runtime_minutes": estimate["estimated_total_runtime_seconds"] / 60,
    }


def main() -> None:
    args = parse_args()
    payload = run(Path(args.output_root))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
