#!/usr/bin/env python
"""Generate M4A-v2 assembly planning and input/output contracts.

This script is planning-only. It inspects existing M4A-v1 and M3-v2
artifacts read-only, then writes reports and inventories under the M4A-v2
planning root. It does not assemble sparse matrices.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ROOT = Path("/home/zhutao/scratch/nichefate")
DEFAULT_OUTPUT_ROOT = ROOT / "m4a_v2_plan"
REPORT_NAMES = [
    "m4a_v2_assembly_plan.md",
    "m4a_v2_input_contract.md",
    "m4a_v2_output_contract.md",
    "m4a_v2_schema_mapping_from_m3_v2.md",
    "m4a_v2_qc_and_validation_plan.md",
    "m4a_v2_resume_and_failure_recovery_plan.md",
    "m4a_v2_v1_comparison_plan.md",
    "m4a_v2_m4c_v2_handoff_plan.md",
]
EXPECTED_M3_V2_SHARDS = 52
EXPECTED_M3_V2_SOURCES = 1_348_582
EXPECTED_M3_V2_EDGES = 13_485_820
EXPECTED_M3_V1_EDGES = 40_457_460
EXPECTED_NODE_COUNT = 1_439_542
EXPECTED_FINAL_NODES = 90_960
ROW_SUM_TOLERANCE = 1e-5
REQUIRED_M3_V2_COLUMNS = [
    "source_anchor_id",
    "target_anchor_id",
    "source_time",
    "target_time",
    "source_slice_id",
    "target_slice_id",
    "source_mouse_id",
    "target_mouse_id",
    "v2_row_normalized_transition_prob",
    "v2_unnormalized_weight",
    "v2_rank_within_source",
    "v2_mode_name",
    "v2_lambda",
    "v2_tau_scale",
    "v2_top_k",
    "v2_g_barcode",
]
PROTECTED_ROOTS = [
    ROOT / "m3",
    ROOT / "m3_v2",
    ROOT / "m4a",
    ROOT / "m4b",
    ROOT / "m4c",
]
FORBIDDEN_DOWNSTREAM_LABELS = [
    "M4A-v2 assembly",
    "M4C-v2",
    "pyGPCCA",
    "K_gpcca",
    "M4D diagnostics",
    "barcode preprocessing",
    "M5/regulator",
    "BranchSBM / Branched NicheFlow",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-shards", type=int, default=5)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolved(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = resolved(left)
    right_resolved = resolved(right)
    return is_relative_to(left_resolved, right_resolved) or is_relative_to(right_resolved, left_resolved)


def reject_ssd(path: Path, label: str) -> None:
    path = resolved(path)
    if path == Path("/ssd") or Path("/ssd") in path.parents:
        raise ValueError(f"Refusing /ssd path for {label}: {path}")


def validate_output_root(output_root: Path) -> Path:
    root = resolved(output_root)
    reject_ssd(root, "M4A-v2 planning output root")
    for protected in PROTECTED_ROOTS:
        if paths_overlap(root, protected):
            raise ValueError(f"Planning output root overlaps protected production root {protected}: {root}")
    return root


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
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def snapshot(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for root in paths:
        if not root.exists():
            out[str(root)] = {"exists": False, "size": -1, "mtime_ns": -1, "is_dir": False}
            continue
        entries = [root, *sorted(path for path in root.rglob("*") if path.exists())] if root.is_dir() else [root]
        for path in entries:
            stat = path.stat()
            out[str(path)] = {
                "exists": True,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "is_dir": path.is_dir(),
            }
    return out


def diff_snapshot(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[str]:
    diffs = []
    for path in sorted(set(before) | set(after)):
        if path not in before:
            diffs.append(f"ADDED\t{path}")
        elif path not in after:
            diffs.append(f"REMOVED\t{path}")
        elif before[path] != after[path]:
            diffs.append(f"CHANGED\t{path}\tbefore={before[path]}\tafter={after[path]}")
    return diffs


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def choose_sample(paths: list[Path], sample_count: int) -> list[Path]:
    if len(paths) <= sample_count:
        return paths
    indices = np.linspace(0, len(paths) - 1, sample_count, dtype=int)
    return [paths[int(index)] for index in indices]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def inspect_m4a_v1() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "m4a_markov_assembly.yaml"
    schema_path = ROOT / "m4a" / "reports" / "m4a_transition_object_schema.json"
    report_path = ROOT / "m4a" / "reports" / "m4a_assembly_report.md"
    node_table_path = ROOT / "m4a" / "node_table" / "global_node_table.parquet"
    config = read_yaml(config_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    node_pf = pq.ParquetFile(node_table_path)
    node_columns = node_pf.schema_arrow.names
    node_sample = pd.read_parquet(
        node_table_path,
        columns=["anchor_id", "global_node_index", "time", "is_final_time"],
    )
    final_nodes = int(node_sample["is_final_time"].sum())
    return {
        "config_path": config_path,
        "schema_path": schema_path,
        "report_path": report_path,
        "node_table_path": node_table_path,
        "config": config,
        "schema": schema,
        "node_columns": node_columns,
        "node_count": int(node_pf.metadata.num_rows),
        "final_nodes": final_nodes,
        "anchor_id_unique": bool(node_sample["anchor_id"].is_unique),
        "global_node_index_unique": bool(node_sample["global_node_index"].is_unique),
        "time_counts": node_sample.groupby("time", observed=True).size().to_dict(),
    }


def inspect_m3_v2(sample_shards: int) -> dict[str, Any]:
    qc_path = ROOT / "m3_v2" / "reports" / "m3_v2_full_qc_summary.csv"
    completed_path = ROOT / "m3_v2" / "reports" / "completed_shards.csv"
    benchmark_path = ROOT / "m3_v2_benchmark" / "m3_v1_vs_v2_edge_benchmark_summary.json"
    qc = pd.read_csv(qc_path)
    completed = pd.read_csv(completed_path)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    required_qc = {
        "shard_id",
        "time_pair",
        "m3_v2_output_parquet",
        "source_count",
        "v1_candidate_edges",
        "retained_v2_edges",
        "row_sum_max_abs_error",
        "probability_nonfinite_count",
        "probability_negative_count",
        "weight_nonfinite_count",
        "weight_negative_count",
        "targets_per_source_max",
    }
    missing_qc = sorted(required_qc - set(qc.columns))
    if missing_qc:
        raise ValueError(f"M3-v2 QC summary missing required columns: {missing_qc}")
    checks = {
        "shard_count": int(len(qc)) == EXPECTED_M3_V2_SHARDS,
        "source_count": int(qc["source_count"].sum()) == EXPECTED_M3_V2_SOURCES,
        "v1_candidate_edges": int(qc["v1_candidate_edges"].sum()) == EXPECTED_M3_V1_EDGES,
        "retained_v2_edges": int(qc["retained_v2_edges"].sum()) == EXPECTED_M3_V2_EDGES,
        "row_sum_max_abs_error": float(qc["row_sum_max_abs_error"].max()) <= ROW_SUM_TOLERANCE,
        "targets_per_source_max": int(qc["targets_per_source_max"].max()) <= 10,
        "probability_nonfinite_count": int(qc["probability_nonfinite_count"].sum()) == 0,
        "probability_negative_count": int(qc["probability_negative_count"].sum()) == 0,
        "weight_nonfinite_count": int(qc["weight_nonfinite_count"].sum()) == 0,
        "weight_negative_count": int(qc["weight_negative_count"].sum()) == 0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"M3-v2 full QC checks failed for planning: {failed}")
    shard_paths = [resolved(value) for value in qc["m3_v2_output_parquet"].astype(str)]
    missing_paths = [str(path) for path in shard_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing M3-v2 shard parquets: {missing_paths[:5]}")
    for path in shard_paths:
        if not is_relative_to(path, resolved(ROOT / "m3_v2" / "full_by_shard")):
            raise ValueError(f"M3-v2 shard path outside full_by_shard root: {path}")
        reject_ssd(path, "M3-v2 input shard")
    samples = choose_sample(shard_paths, sample_shards)
    sample_rows = []
    sample_required_missing: dict[str, list[str]] = {}
    sample_duplicate_total = 0
    for path in samples:
        columns = parquet_columns(path)
        missing = sorted(set(REQUIRED_M3_V2_COLUMNS) - set(columns))
        if missing:
            sample_required_missing[str(path)] = missing
        pf = pq.ParquetFile(path)
        sample_frame = pd.read_parquet(
            path,
            columns=["source_anchor_id", "target_anchor_id", "v2_row_normalized_transition_prob"],
        )
        duplicate_count = int(sample_frame[["source_anchor_id", "target_anchor_id"]].duplicated().sum())
        nonfinite_prob = int((~np.isfinite(sample_frame["v2_row_normalized_transition_prob"].to_numpy(float))).sum())
        negative_prob = int((sample_frame["v2_row_normalized_transition_prob"].to_numpy(float) < 0).sum())
        sample_duplicate_total += duplicate_count
        sample_rows.append(
            {
                "sample_path": str(path),
                "rows": int(pf.metadata.num_rows),
                "column_count": int(len(columns)),
                "required_columns_present": not missing,
                "duplicate_source_target_pairs": duplicate_count,
                "probability_nonfinite_count": nonfinite_prob,
                "probability_negative_count": negative_prob,
            }
        )
    if sample_required_missing:
        raise ValueError(f"Sampled M3-v2 shards missing required columns: {sample_required_missing}")
    if sample_duplicate_total:
        raise ValueError(f"Sampled M3-v2 shards contain duplicate source-target pairs: {sample_duplicate_total}")
    return {
        "qc_path": qc_path,
        "completed_path": completed_path,
        "benchmark_path": benchmark_path,
        "qc": qc,
        "completed": completed,
        "benchmark": benchmark,
        "checks": checks,
        "sample_rows": sample_rows,
        "sample_paths": samples,
        "sample_schema_columns": parquet_columns(samples[0]),
        "time_pair_counts": qc.groupby("time_pair", observed=True)
        .agg(shards=("shard_id", "size"), sources=("source_count", "sum"), retained_edges=("retained_v2_edges", "sum"))
        .reset_index(),
    }


def validate_anchor_mapping(m4a: dict[str, Any], m3_v2: dict[str, Any]) -> dict[str, Any]:
    node_table_path = Path(m4a["node_table_path"])
    anchor_index = pd.Index(
        pd.read_parquet(node_table_path, columns=["anchor_id"])["anchor_id"].astype(str),
    )
    rows = []
    missing_source_total = 0
    missing_target_total = 0
    for path in m3_v2["sample_paths"]:
        frame = pd.read_parquet(path, columns=["source_anchor_id", "target_anchor_id"])
        source_missing = int((anchor_index.get_indexer(frame["source_anchor_id"].astype(str)) < 0).sum())
        target_missing = int((anchor_index.get_indexer(frame["target_anchor_id"].astype(str)) < 0).sum())
        missing_source_total += source_missing
        missing_target_total += target_missing
        rows.append(
            {
                "sample_path": str(path),
                "sample_rows": int(len(frame)),
                "source_anchor_missing_from_node_table": source_missing,
                "target_anchor_missing_from_node_table": target_missing,
            }
        )
    if missing_source_total or missing_target_total:
        raise ValueError(
            "Sampled M3-v2 anchors do not map to M4A-v1 node table: "
            f"source_missing={missing_source_total}, target_missing={missing_target_total}"
        )
    return {
        "join_key": "source_anchor_id/target_anchor_id -> M4A node_table.anchor_id",
        "sample_mapping_rows": rows,
        "sample_source_missing": missing_source_total,
        "sample_target_missing": missing_target_total,
    }


def required_input_inventory(m4a: dict[str, Any], m3_v2: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "input_name": "M3-v2 full edge shards",
            "path": str(ROOT / "m3_v2" / "full_by_shard"),
            "required": True,
            "read_only": True,
            "expected_count": EXPECTED_M3_V2_SHARDS,
            "required_columns": ";".join(REQUIRED_M3_V2_COLUMNS),
            "join_keys": "source_anchor_id,target_anchor_id",
            "failure_behavior": "fail planning/preflight if missing schema, shard path, probabilities, or mappable anchor keys",
        },
        {
            "input_name": "M3-v2 full QC summary",
            "path": str(m3_v2["qc_path"]),
            "required": True,
            "read_only": True,
            "expected_count": EXPECTED_M3_V2_SHARDS,
            "required_columns": "shard_id;time_pair;m3_v2_output_parquet;source_count;retained_v2_edges;row_sum_max_abs_error",
            "join_keys": "shard_id,m3_v2_output_parquet",
            "failure_behavior": "fail if totals or QC do not match M3-v2-07 production",
        },
        {
            "input_name": "M3-v2 benchmark summary",
            "path": str(m3_v2["benchmark_path"]),
            "required": True,
            "read_only": True,
            "expected_count": 1,
            "required_columns": "json:status;decision_category;v1_probability_column;v2_probability_column;global_summary",
            "join_keys": "not applicable",
            "failure_behavior": "fail if not parseable or does not confirm complementary M3-v1/M3-v2 interpretation",
        },
        {
            "input_name": "M4A-v1 global node table",
            "path": str(m4a["node_table_path"]),
            "required": True,
            "read_only": True,
            "expected_count": EXPECTED_NODE_COUNT,
            "required_columns": "global_node_index;anchor_id;time;time_day;is_final_time",
            "join_keys": "anchor_id",
            "failure_behavior": "fail if anchor IDs are not unique or M3-v2 edge endpoints cannot map",
        },
        {
            "input_name": "M2 representation metadata",
            "path": str(ROOT / "m2" / "by_slice"),
            "required": False,
            "read_only": True,
            "expected_count": "available by slice",
            "required_columns": "slice_id;anchor_index;time;time_day",
            "join_keys": "slice_id,anchor_index",
            "failure_behavior": "optional validation fallback only; do not rebuild node table unless explicitly approved",
        },
        {
            "input_name": "M4E annotations",
            "path": str(ROOT / "m4e"),
            "required": False,
            "read_only": True,
            "expected_count": "annotation-dependent",
            "required_columns": "anchor_id;endpoint/neighborhood annotation columns",
            "join_keys": "anchor_id",
            "failure_behavior": "optional interpretability/handoff check only; not used for matrix assembly",
        },
    ]
    return pd.DataFrame(rows)


def planned_output_inventory() -> pd.DataFrame:
    root = ROOT / "m4a_v2"
    rows = [
        {
            "output_name": "P_forward_no_terminal_selfloops_v2",
            "planned_path": str(root / "transition_objects" / "P_forward_no_terminal_selfloops_v2.npz"),
            "output_type": "scipy sparse CSR npz",
            "created_in_planning": False,
            "expected_shape": f"{EXPECTED_NODE_COUNT}x{EXPECTED_NODE_COUNT}",
            "expected_nnz": EXPECTED_M3_V2_EDGES,
            "description": "Forward pseudo-only M3-v2 row-normalized transition matrix without D35 terminal self-loops.",
        },
        {
            "output_name": "P_absorbing_terminal_selfloops_v2",
            "planned_path": str(root / "transition_objects" / "P_absorbing_terminal_selfloops_v2.npz"),
            "output_type": "scipy sparse CSR npz",
            "created_in_planning": False,
            "expected_shape": f"{EXPECTED_NODE_COUNT}x{EXPECTED_NODE_COUNT}",
            "expected_nnz": EXPECTED_M3_V2_EDGES + EXPECTED_FINAL_NODES,
            "description": "Markov-ready structural variant with D35 self-loops only; no fate probabilities.",
        },
        {
            "output_name": "W_v2_unnormalized_weight",
            "planned_path": str(root / "transition_objects" / "W_v2_unnormalized_weight.npz"),
            "output_type": "scipy sparse CSR npz",
            "created_in_planning": False,
            "expected_shape": f"{EXPECTED_NODE_COUNT}x{EXPECTED_NODE_COUNT}",
            "expected_nnz": EXPECTED_M3_V2_EDGES,
            "description": "Sparse diagnostic matrix from M3-v2 unnormalized constrained sharpening weights.",
        },
        {
            "output_name": "global_node_table_v2",
            "planned_path": str(root / "node_table" / "global_node_table.parquet"),
            "output_type": "parquet",
            "created_in_planning": False,
            "expected_shape": f"{EXPECTED_NODE_COUNT} rows",
            "expected_nnz": "",
            "description": "Versioned copy of M4A-v1-compatible node table; no v1 overwrite.",
        },
        {
            "output_name": "assembly reports",
            "planned_path": str(root / "reports"),
            "output_type": "markdown/csv/json",
            "created_in_planning": False,
            "expected_shape": "report set",
            "expected_nnz": "",
            "description": "Assembly report, QC summary, v1-v2 matrix comparison, output inventory, and schema JSON.",
        },
    ]
    return pd.DataFrame(rows)


def checklist() -> pd.DataFrame:
    rows = [
        ("inputs", "M3-v2 shard count equals 52", True, "passed_in_planning"),
        ("inputs", "M3-v2 retained edge total equals 13,485,820", True, "passed_in_planning"),
        ("schema", "v2_row_normalized_transition_prob present in sampled shards", True, "passed_in_planning"),
        ("schema", "source/target anchor IDs map to M4A node table in sampled shards", True, "passed_in_planning"),
        ("safety", "M4A-v2 production output root is separate from M4A-v1", True, "planned"),
        ("safety", "No /ssd output paths", True, "passed_in_planning"),
        ("execution", "M4A-v2 assembly not run in M4A-v2-00", True, "not_run"),
        ("execution", "M4C-v2, GPCCA, barcode, M5, BranchSBM not run", True, "not_run"),
        ("future_preflight", "Full duplicate edge-key check before matrix assembly", True, "required_before_full_run"),
        ("future_preflight", "No overwrite unless explicit --overwrite", True, "required_before_full_run"),
    ]
    return pd.DataFrame(rows, columns=["category", "check_item", "required_before_full_run", "status_for_planning"])


def md_table(frame: pd.DataFrame) -> str:
    work = frame.copy()
    for col in work.columns:
        work[col] = work[col].map(lambda value: str(value).replace("\n", " "))
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for row in work.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def report_texts(
    m4a: dict[str, Any],
    m3_v2: dict[str, Any],
    mapping: dict[str, Any],
    inputs: pd.DataFrame,
    outputs: pd.DataFrame,
) -> dict[str, str]:
    time_pair_table = md_table(m3_v2["time_pair_counts"])
    sample_table = md_table(pd.DataFrame(m3_v2["sample_rows"]))
    output_table = md_table(outputs[["output_name", "planned_path", "expected_shape", "expected_nnz"]])
    safety_lines = "\n".join(f"- {label}: not run" for label in FORBIDDEN_DOWNSTREAM_LABELS)
    mode = m3_v2["benchmark"].get("decision_category", "unknown")
    return {
        "m4a_v2_assembly_plan.md": f"""# M4A-v2-00 Assembly Plan

## Planning Status
- planning_only: true
- M4A-v2 assembly run: no
- M4C-v2 execution run: no
- upstream production outputs modified: no
- output root for planning artifacts: `{DEFAULT_OUTPUT_ROOT}`

## Scope
M4A-v2 will be a versioned pseudo-only sparse Markov transition assembly derived from full M3-v2 `constrained_v1prior_sharpening` edge shards. It remains complementary to M4A-v1 and must not overwrite `/home/zhutao/scratch/nichefate/m4a`.

## M4A-v1 Pattern Reused
- Node table: `{m4a['node_table_path']}`
- Node count: {m4a['node_count']:,}
- Final-time D35 nodes: {m4a['final_nodes']:,}
- Existing matrix objects: `P_forward_no_terminal_selfloops`, `P_absorbing_terminal_selfloops`, `W_raw_edge_weight`, `W_mass_adjusted_weight`
- Existing terminal policy: final-time rows have no forward outgoing mass; absorbing variant adds terminal self-loops.

## M3-v2 Inputs Confirmed
- M3-v2 shards: {EXPECTED_M3_V2_SHARDS}
- M3-v2 sources: {EXPECTED_M3_V2_SOURCES:,}
- M3-v2 retained edges: {EXPECTED_M3_V2_EDGES:,}
- M3-v2 probability column: `v2_row_normalized_transition_prob`
- M3-v2 benchmark decision: `{mode}`

## Time-Pair Totals
{time_pair_table}

## Planned Matrix Assembly Logic
- Map `source_anchor_id` and `target_anchor_id` to `global_node_index` using M4A node-table `anchor_id`.
- Matrix rows and columns cover all global nodes.
- `P_forward_no_terminal_selfloops_v2[i, j] = v2_row_normalized_transition_prob`.
- `W_v2_unnormalized_weight[i, j] = v2_unnormalized_weight`.
- D35/final-time rows remain zero in the forward matrix.
- The absorbing structural variant adds D35 self-loops with weight 1.0 only.
- No GPCCA, fate probability, barcode, or lineage-validation logic belongs in M4A-v2.
""",
        "m4a_v2_input_contract.md": f"""# M4A-v2 Input Contract

All inputs are read-only. Missing required inputs or schema mismatches must fail preflight before any matrix assembly.

{md_table(inputs)}

## Mapping Contract
- Primary node join: `{mapping['join_key']}`
- Sample source anchor missing count: {mapping['sample_source_missing']}
- Sample target anchor missing count: {mapping['sample_target_missing']}
- M4E annotations are not matrix assembly inputs; they are downstream interpretability inputs only.
""",
        "m4a_v2_output_contract.md": f"""# M4A-v2 Output Contract

Production outputs must be written only under `/home/zhutao/scratch/nichefate/m4a_v2/`.

{output_table}

## Planned Report Outputs
- `/home/zhutao/scratch/nichefate/m4a_v2/reports/m4a_v2_assembly_report.md`
- `/home/zhutao/scratch/nichefate/m4a_v2/reports/m4a_v2_qc_summary.csv`
- `/home/zhutao/scratch/nichefate/m4a_v2/reports/m4a_v1_vs_v2_matrix_comparison.csv`
- `/home/zhutao/scratch/nichefate/m4a_v2/reports/m4a_v2_output_inventory.csv`
- `/home/zhutao/scratch/nichefate/m4a_v2/reports/m4a_v2_transition_object_schema.json`

These production outputs were not created by M4A-v2-00.
""",
        "m4a_v2_schema_mapping_from_m3_v2.md": f"""# M4A-v2 Schema Mapping From M3-v2

## Required M3-v2 Edge Columns
{md_table(pd.DataFrame({'column': REQUIRED_M3_V2_COLUMNS}))}

## Sampled Shard Schema Validation
{sample_table}

## Assembly Mapping
| M3-v2 field | M4A-v2 use |
| --- | --- |
| `source_anchor_id` | map to matrix row global node index |
| `target_anchor_id` | map to matrix column global node index |
| `v2_row_normalized_transition_prob` | forward transition probability entry |
| `v2_unnormalized_weight` | diagnostic weight matrix entry |
| `source_time`, `target_time` | time-pair QC and mass summaries |
| `source_slice_id`, `target_slice_id`, `source_mouse_id`, `target_mouse_id` | coverage and concentration QC |
| `v2_mode_name`, `v2_lambda`, `v2_tau_scale`, `v2_top_k`, `v2_g_barcode` | provenance and locked-parameter validation |

M3-v2 includes v1 probability columns for comparison, but M4A-v2 transition entries must use `v2_row_normalized_transition_prob`.
""",
        "m4a_v2_qc_and_validation_plan.md": f"""# M4A-v2 QC And Validation Plan

## Required QC Gates
- Node count equals {EXPECTED_NODE_COUNT:,}.
- Final-time node count equals {EXPECTED_FINAL_NODES:,}.
- Source rows with outgoing transitions equal {EXPECTED_M3_V2_SOURCES:,}.
- Forward matrix nnz equals {EXPECTED_M3_V2_EDGES:,}.
- Absorbing matrix nnz equals {EXPECTED_M3_V2_EDGES + EXPECTED_FINAL_NODES:,}.
- Non-final row-sum max error <= {ROW_SUM_TOLERANCE}.
- Forward final-time row sums are zero.
- Absorbing all-row max error <= {ROW_SUM_TOLERANCE}.
- No NaN, infinite, or negative entries.
- No duplicate source-target matrix coordinates.
- No non-final zero-outgoing rows.
- Source and edge counts match M3-v2 QC by time pair.

## Validation Commands For Future Assembler
- dry-run/preflight must validate schemas, anchor mapping, duplicate coordinates, output-root separation, and no-overwrite behavior.
- full assembly may run only after separate approval.
""",
        "m4a_v2_resume_and_failure_recovery_plan.md": """# M4A-v2 Resume And Failure Recovery Plan

Future M4A-v2 assembler requirements:
- Support `--preflight-only`, `--resume`, `--stop-on-error`, and `--overwrite` with overwrite default false.
- Write completed-step manifest under `/home/zhutao/scratch/nichefate/m4a_v2/reports/`.
- Write failures to `/home/zhutao/scratch/nichefate/m4a_v2/logs/failed_steps.txt`.
- Use atomic writes for reports, manifests, node table, and sparse matrix outputs.
- Use temporary paths only under `/home/zhutao/scratch/nichefate/m4a_v2/tmp/`.
- On failure, leave existing valid outputs untouched and report partial temp files for cleanup.
- Existing invalid outputs must be reported and not replaced unless `--overwrite` is explicitly passed.
""",
        "m4a_v2_v1_comparison_plan.md": f"""# M4A-v1 vs M4A-v2 Matrix Comparison Plan

Compare the versioned M4A-v2 matrix objects against frozen M4A-v1 without modifying either root.

## Metrics
- Node count and node-table identity.
- Source row count: M4A-v1 expected {EXPECTED_M3_V1_EDGES // 30:,}; M4A-v2 expected {EXPECTED_M3_V2_SOURCES:,}.
- Edge count / nnz: M4A-v1 {EXPECTED_M3_V1_EDGES:,}; M4A-v2 {EXPECTED_M3_V2_EDGES:,}.
- Matrix sparsity and memory/disk footprint.
- Row-sum max error and rows exceeding tolerance.
- Source coverage and edge counts by time pair.
- Target distribution by time pair, slice, and mouse.
- Top1 transition concentration if retained from source-level sparse rows.
- Change in final-time and non-final zero-outgoing rows.

## Interpretation
M4A-v2 is expected to be sharper and sparser because M3-v2 retained top-10 edges per source. It does not replace M4A-v1; it provides a constrained sharpened pseudo-transition assembly for downstream comparison.
""",
        "m4a_v2_m4c_v2_handoff_plan.md": """# M4A-v2 To M4C-v2 Handoff Plan

M4C-v2 is downstream and was not planned beyond this handoff contract.

## M4C-v2 Inputs Needed Later
- `P_forward_no_terminal_selfloops_v2.npz`
- `P_absorbing_terminal_selfloops_v2.npz`
- M4A-v2 node table
- M4E endpoint mapping and annotations
- M4C-v1 baseline summaries for comparison
- M3-v2 benchmark summary for interpretation

## Expected M4C-v2 Outputs Later
- Fate probability matrix v2
- Node-level dominant endpoint v2
- Plasticity v2
- Tissue maps v2
- M4C-v1 vs M4C-v2 benchmark

No M4C-v2 execution, GPCCA, K_gpcca, barcode preprocessing, M5, or BranchSBM is authorized by M4A-v2-00.
""",
    }


def write_outputs(
    output_root: Path,
    m4a: dict[str, Any],
    m3_v2: dict[str, Any],
    mapping: dict[str, Any],
    before: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reports = output_root / "reports"
    inputs = required_input_inventory(m4a, m3_v2)
    outputs = planned_output_inventory()
    checks = checklist()
    report_payload = report_texts(m4a, m3_v2, mapping, inputs, outputs)
    for name, text in report_payload.items():
        write_text(reports / name, text)
    write_csv(output_root / "m4a_v2_required_input_inventory.csv", inputs)
    write_csv(output_root / "m4a_v2_planned_output_inventory.csv", outputs)
    write_csv(output_root / "m4a_v2_assembly_checklist.csv", checks)
    after = snapshot(PROTECTED_ROOTS)
    diffs = diff_snapshot(before, after)
    summary = {
        "stage": "M4A-v2-00",
        "planning_only": True,
        "generated_at_utc": utc_now(),
        "output_root": output_root,
        "reports_dir": reports,
        "m4a_v1_inspection_read_only": True,
        "m4a_v2_execution_run": False,
        "m4c_v2_execution_run": False,
        "forbidden_downstream_run": False,
        "no_ssd_outputs": True,
        "protected_metadata_diff_count": len(diffs),
        "protected_metadata_diffs": diffs,
        "confirmed_m3_v2_probability_column": "v2_row_normalized_transition_prob",
        "confirmed_m3_v2_weight_column": "v2_unnormalized_weight",
        "confirmed_join_key": mapping["join_key"],
        "sample_shard_schema_columns": m3_v2["sample_schema_columns"],
        "sample_shard_checks": m3_v2["sample_rows"],
        "node_count": m4a["node_count"],
        "final_time_node_count": m4a["final_nodes"],
        "expected_source_rows": EXPECTED_M3_V2_SOURCES,
        "expected_v1_candidate_edges": EXPECTED_M3_V1_EDGES,
        "expected_retained_v2_edges": EXPECTED_M3_V2_EDGES,
        "planned_m4a_v2_output_root": ROOT / "m4a_v2",
        "planned_matrix_objects": outputs["output_name"].tolist(),
        "qc_criteria": {
            "row_sum_tolerance": ROW_SUM_TOLERANCE,
            "no_nan_inf_negative": True,
            "no_duplicate_source_target_entries": True,
            "non_final_zero_outgoing_rows": 0,
            "final_time_zero_outgoing_rows": EXPECTED_FINAL_NODES,
        },
        "m3_v2_full_qc_checks": m3_v2["checks"],
        "m3_v2_time_pair_counts": m3_v2["time_pair_counts"].to_dict(orient="records"),
        "decision": "planning_passed" if not diffs else "planning_failed_metadata_diff",
        "exact_next_recommended_step": (
            "Implement M4A-v2 assembler with dry-run/preflight only; do not execute M4A-v2 assembly until separately approved."
        ),
    }
    write_json(output_root / "m4a_v2_plan_summary.json", summary)
    if diffs:
        raise RuntimeError("Protected upstream metadata changed during planning.")
    return summary


def validate_required_outputs(output_root: Path) -> None:
    required = [output_root / "reports" / name for name in REPORT_NAMES]
    required.extend(
        [
            output_root / "m4a_v2_required_input_inventory.csv",
            output_root / "m4a_v2_planned_output_inventory.csv",
            output_root / "m4a_v2_assembly_checklist.csv",
            output_root / "m4a_v2_plan_summary.json",
        ]
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty required M4A-v2 planning outputs: {missing}")
    with (output_root / "m4a_v2_plan_summary.json").open("r", encoding="utf-8") as handle:
        json.load(handle)
    input_cols = set(pd.read_csv(output_root / "m4a_v2_required_input_inventory.csv", nrows=1).columns)
    output_cols = set(pd.read_csv(output_root / "m4a_v2_planned_output_inventory.csv", nrows=1).columns)
    checklist_cols = set(pd.read_csv(output_root / "m4a_v2_assembly_checklist.csv", nrows=1).columns)
    if {"input_name", "path", "required", "read_only", "required_columns", "join_keys"} - input_cols:
        raise ValueError("Input inventory CSV schema check failed.")
    if {"output_name", "planned_path", "output_type", "created_in_planning", "expected_shape"} - output_cols:
        raise ValueError("Output inventory CSV schema check failed.")
    if {"category", "check_item", "required_before_full_run", "status_for_planning"} - checklist_cols:
        raise ValueError("Checklist CSV schema check failed.")


def main() -> None:
    args = parse_args()
    output_root = validate_output_root(args.output_root)
    before = snapshot(PROTECTED_ROOTS)
    m4a = inspect_m4a_v1()
    if m4a["node_count"] != EXPECTED_NODE_COUNT or m4a["final_nodes"] != EXPECTED_FINAL_NODES:
        raise RuntimeError(
            f"M4A-v1 node table count mismatch: nodes={m4a['node_count']} final={m4a['final_nodes']}"
        )
    if not m4a["anchor_id_unique"] or not m4a["global_node_index_unique"]:
        raise RuntimeError("M4A-v1 node table does not have unique anchor/global node keys.")
    m3_v2 = inspect_m3_v2(args.sample_shards)
    mapping = validate_anchor_mapping(m4a, m3_v2)
    summary = write_outputs(output_root, m4a, m3_v2, mapping, before)
    validate_required_outputs(output_root)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
