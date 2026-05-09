#!/usr/bin/env python
"""Generate the M4C-v2 planning and handoff contract.

This script is intentionally planning-only. It inspects existing M4C-v1,
M4A-v2, and M4E artifacts read-only, then writes contract documents under
``/home/zhutao/scratch/nichefate/m4c_v2_plan``. It does not create M4C-v2
production outputs or run fate propagation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for thread_var in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(thread_var, "1")

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal envs
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ROOT = Path("/home/zhutao/scratch/nichefate")
DEFAULT_OUTPUT_ROOT = ROOT / "m4c_v2_plan"
M4C_V2_PRODUCTION_ROOT = ROOT / "m4c_v2"

EXPECTED_NODES = 1_439_542
EXPECTED_FINAL_NODES = 90_960
EXPECTED_SOURCE_ROWS = 1_348_582
EXPECTED_V2_FORWARD_NNZ = 13_485_820
EXPECTED_V2_ABSORBING_NNZ = 13_576_780
EXPECTED_FINAL_TIME = "D35"
ROW_SUM_TOLERANCE = 1e-5

PROTECTED_ROOTS = [
    ROOT / "m3",
    ROOT / "m3_v2",
    ROOT / "m4a",
    ROOT / "m4a_v2",
    ROOT / "m4b",
    ROOT / "m4c",
]

FORBIDDEN_DOWNSTREAM_ROOTS = [
    ROOT / "m4c_v2",
    ROOT / "m4c_v2" / "gpcca",
    ROOT / "m4c_v2" / "pygpcca",
    ROOT / "m4c_v2" / "k_gpcca",
    ROOT / "m4c_v2" / "barcode",
    ROOT / "m4c_v2" / "m5",
    ROOT / "m4c_v2" / "branchsbm",
    ROOT / "m4a_v2" / "gpcca",
    ROOT / "m4a_v2" / "pygpcca",
    ROOT / "m4a_v2" / "k_gpcca",
    ROOT / "m4a_v2" / "barcode",
    ROOT / "m4a_v2" / "m5",
    ROOT / "m4a_v2" / "branchsbm",
]

REPORT_FILES = [
    "m4c_v2_fate_propagation_plan.md",
    "m4c_v2_input_contract.md",
    "m4c_v2_output_contract.md",
    "m4c_v2_endpoint_taxonomy_reuse_plan.md",
    "m4c_v2_qc_and_validation_plan.md",
    "m4c_v2_v1_comparison_plan.md",
    "m4c_v2_visualization_plan.md",
    "m4c_v2_resume_and_failure_recovery_plan.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolved(path: str | Path) -> Path:
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
    return is_relative_to(left_resolved, right_resolved) or is_relative_to(
        right_resolved,
        left_resolved,
    )


def reject_ssd(path: Path) -> None:
    path = resolved(path)
    if path == Path("/ssd") or Path("/ssd") in path.parents:
        raise ValueError(f"Refusing /ssd path: {path}")


def validate_output_root(output_root: Path) -> Path:
    output_root = resolved(output_root)
    reject_ssd(output_root)
    for protected in [*PROTECTED_ROOTS, *FORBIDDEN_DOWNSTREAM_ROOTS]:
        if paths_overlap(output_root, protected):
            raise ValueError(
                f"Output root overlaps protected or forbidden root "
                f"{protected}: {output_root}"
            )
    return output_root


def output_paths(output_root: Path) -> dict[str, Path]:
    root = validate_output_root(output_root)
    reports = root / "reports"
    paths = {
        "root": root,
        "reports": reports,
        "input_inventory": root / "m4c_v2_required_input_inventory.csv",
        "planned_output_inventory": root / "m4c_v2_planned_output_inventory.csv",
        "planning_checklist": root / "m4c_v2_planning_checklist.csv",
        "summary": root / "m4c_v2_plan_summary.json",
    }
    for name in REPORT_FILES:
        paths[name] = reports / name
    return paths


def ensure_output_dirs(paths: dict[str, Path]) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)


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


def atomic_write_text(path: Path, text: str) -> None:
    reject_ssd(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    reject_ssd(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def snapshot(paths: list[Path]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for root in paths:
        if not root.exists():
            metadata[str(root)] = {
                "exists": False,
                "size": -1,
                "mtime_ns": -1,
                "is_dir": False,
            }
            continue
        entries = [root, *sorted(path for path in root.rglob("*") if path.exists())]
        for path in entries:
            stat = path.stat()
            metadata[str(path)] = {
                "exists": True,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "is_dir": path.is_dir(),
            }
    return metadata


def diff_snapshot(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> list[str]:
    diffs: list[str] = []
    for path in sorted(set(before) | set(after)):
        if path not in before:
            diffs.append(f"ADDED\t{path}")
        elif path not in after:
            diffs.append(f"REMOVED\t{path}")
        elif before[path] != after[path]:
            diffs.append(
                f"CHANGED\t{path}\tbefore={before[path]}\tafter={after[path]}"
            )
    return diffs


def file_status(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "exists": bool(exists),
        "bytes": int(path.stat().st_size) if exists else 0,
        "mtime_ns": int(path.stat().st_mtime_ns) if exists else -1,
    }


def load_yaml_file(path: Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def parquet_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"rows": None, "columns": []}
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        return {
            "rows": int(parquet_file.metadata.num_rows),
            "columns": list(parquet_file.schema.names),
        }
    except Exception:  # noqa: BLE001
        return {"rows": None, "columns": []}


def required_input_specs(root: Path = ROOT) -> list[dict[str, Any]]:
    return [
        {
            "input_name": "m4a_v2_p_absorbing_terminal_selfloops",
            "path": root
            / "m4a_v2"
            / "transition_objects"
            / "P_absorbing_terminal_selfloops_v2.npz",
            "object_type": "sparse_matrix_npz",
            "required": True,
            "required_columns_or_keys": "scipy sparse npz arrays",
            "expected_shape_or_rows": (
                f"{EXPECTED_NODES}x{EXPECTED_NODES}; "
                f"nnz={EXPECTED_V2_ABSORBING_NNZ}"
            ),
            "join_keys": "row/column global_node_index",
            "failure_behavior": "fail preflight before fate propagation",
            "notes": "Canonical row-stochastic absorbing transition object.",
        },
        {
            "input_name": "m4a_v2_p_forward_no_terminal_selfloops",
            "path": root
            / "m4a_v2"
            / "transition_objects"
            / "P_forward_no_terminal_selfloops_v2.npz",
            "object_type": "sparse_matrix_npz",
            "required": True,
            "required_columns_or_keys": "scipy sparse npz arrays",
            "expected_shape_or_rows": (
                f"{EXPECTED_NODES}x{EXPECTED_NODES}; "
                f"nnz={EXPECTED_V2_FORWARD_NNZ}"
            ),
            "join_keys": "row/column global_node_index",
            "failure_behavior": "fail preflight before fate propagation",
            "notes": "Time-layered DAG propagation object matching M4C-v1 semantics.",
        },
        {
            "input_name": "m4a_v2_global_node_table",
            "path": root / "m4a_v2" / "node_table" / "global_node_table.parquet",
            "object_type": "parquet_table",
            "required": True,
            "required_columns_or_keys": (
                "global_node_index,anchor_id,time,time_day,is_final_time,"
                "slice_id,anchor_index"
            ),
            "expected_shape_or_rows": f"{EXPECTED_NODES} rows",
            "join_keys": "global_node_index,anchor_id",
            "failure_behavior": "fail preflight if row order or keys are invalid",
            "notes": "Row i must correspond to global_node_index i.",
        },
        {
            "input_name": "m4a_v2_qc_summary",
            "path": root / "m4a_v2" / "reports" / "m4a_v2_02_qc_summary.csv",
            "object_type": "csv_report",
            "required": True,
            "required_columns_or_keys": "status,matrix_shape,forward_nnz,absorbing_nnz",
            "expected_shape_or_rows": "one completed full-production row",
            "join_keys": "none",
            "failure_behavior": "fail preflight if M4A-v2 QC did not complete",
            "notes": "Confirms full sparse assembly status.",
        },
        {
            "input_name": "m4a_v2_benchmark_summary",
            "path": root / "m4a_v2_benchmark" / "m4a_v2_benchmark_summary.json",
            "object_type": "json_report",
            "required": True,
            "required_columns_or_keys": "full_qc_status,m4c_v2_readiness_status",
            "expected_shape_or_rows": "decision_category=proceed_to_m4c_v2_planning",
            "join_keys": "none",
            "failure_behavior": "fail planning if benchmark did not pass",
            "notes": "Carries v1-v2 matrix comparison decision.",
        },
        {
            "input_name": "m4a_v2_m4c_v2_readiness_checklist",
            "path": root
            / "m4a_v2_benchmark"
            / "m4a_v2_m4c_v2_required_inputs_checklist.csv",
            "object_type": "csv_report",
            "required": True,
            "required_columns_or_keys": "item,status,details",
            "expected_shape_or_rows": "all rows PASS",
            "join_keys": "none",
            "failure_behavior": "fail planning if any required item fails",
            "notes": "Readiness gate from M4A-v2-03.",
        },
        {
            "input_name": "m4e_refined_endpoint_mapping",
            "path": root / "m4e" / "endpoint_refinement" / "refined_endpoint_mapping.csv",
            "object_type": "csv_table",
            "required": True,
            "required_columns_or_keys": (
                "raw_terminal_macrostate,refined_endpoint_id,"
                "refined_endpoint_label,confidence_tier_after_refinement"
            ),
            "expected_shape_or_rows": "12 refined endpoints",
            "join_keys": "raw_terminal_macrostate",
            "failure_behavior": "fail preflight unless schema is repaired",
            "notes": "Endpoint taxonomy must be reused, not redefined.",
        },
        {
            "input_name": "m4e_endpoint_node_annotation",
            "path": root
            / "m4e"
            / "endpoint_annotation"
            / "endpoint_node_annotation.parquet",
            "object_type": "parquet_table",
            "required": True,
            "required_columns_or_keys": "global_node_index,anchor_id,terminal/refined endpoint fields",
            "expected_shape_or_rows": f"{EXPECTED_FINAL_NODES} endpoint rows",
            "join_keys": "global_node_index,anchor_id",
            "failure_behavior": "fail preflight if D35 endpoint rows cannot be mapped",
            "notes": "D35 terminal candidate rows remain endpoint-annotatable.",
        },
        {
            "input_name": "m4e_neighborhood_annotation",
            "path": root
            / "m4e"
            / "neighborhood_annotation"
            / "node_neighborhood_annotation.parquet",
            "object_type": "parquet_table",
            "required": True,
            "required_columns_or_keys": "global_node_index,neighborhood/leiden metadata",
            "expected_shape_or_rows": f"{EXPECTED_NODES} rows",
            "join_keys": "global_node_index,anchor_id",
            "failure_behavior": "warn for optional biological summaries; fail if required metadata missing",
            "notes": "Used for interpretation summaries, not fate propagation itself.",
        },
        {
            "input_name": "m4c_v1_fate_probability_matrix",
            "path": root / "m4c" / "fate_probabilities" / "fate_probability_matrix.npz",
            "object_type": "numpy_npz",
            "required": True,
            "required_columns_or_keys": "probabilities,global_node_index,terminal_macrostate_ids",
            "expected_shape_or_rows": f"{EXPECTED_NODES}x12",
            "join_keys": "global_node_index",
            "failure_behavior": "skip v1-v2 benchmark only if explicitly approved",
            "notes": "Comparison-only input; read-only.",
        },
        {
            "input_name": "m4c_v1_node_summary",
            "path": root
            / "m4c"
            / "fate_probabilities"
            / "fate_probability_node_summary.parquet",
            "object_type": "parquet_table",
            "required": True,
            "required_columns_or_keys": (
                "global_node_index,dominant_fate,dominant_fate_probability,"
                "normalized_plasticity_entropy"
            ),
            "expected_shape_or_rows": f"{EXPECTED_NODES} rows",
            "join_keys": "global_node_index,anchor_id",
            "failure_behavior": "skip v1-v2 benchmark only if explicitly approved",
            "notes": "Comparison-only input; read-only.",
        },
        {
            "input_name": "m4c_v1_schema",
            "path": root / "m4c" / "reports" / "m4c_fate_probability_schema.json",
            "object_type": "json_report",
            "required": True,
            "required_columns_or_keys": "method,fate_probability_matrix,qc",
            "expected_shape_or_rows": "schema_version=m4c_fate_probability_schema_v1",
            "join_keys": "none",
            "failure_behavior": "fail planning if M4C-v1 semantics cannot be inspected",
            "notes": "Documents v1 output contract and propagation method.",
        },
        {
            "input_name": "m3_v2_benchmark_summary",
            "path": root
            / "m3_v2_benchmark"
            / "m3_v1_vs_v2_edge_benchmark_summary.json",
            "object_type": "json_report",
            "required": True,
            "required_columns_or_keys": "mode/top-k interpretation fields",
            "expected_shape_or_rows": "M3-v2 constrained_v1prior_sharpening context",
            "join_keys": "none",
            "failure_behavior": "warn for interpretation context only",
            "notes": "Explains why M4A-v2 is top10-constrained and sharper.",
        },
        {
            "input_name": "m3_v2_full_qc_summary",
            "path": root / "m3_v2" / "reports" / "m3_v2_full_qc_summary.csv",
            "object_type": "csv_report",
            "required": True,
            "required_columns_or_keys": "full production QC fields",
            "expected_shape_or_rows": "M3-v2 full production summary",
            "join_keys": "none",
            "failure_behavior": "warn for interpretation context only",
            "notes": "Read-only upstream context.",
        },
    ]


def status_for_spec(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec["path"])
    status = file_status(path)
    passed = bool(status["exists"] and status["bytes"] > 0)
    parquet = parquet_metadata(path) if spec["object_type"] == "parquet_table" else {}
    return {
        **{key: value for key, value in spec.items() if key != "path"},
        "path": str(path),
        "read_only": True,
        "exists": bool(status["exists"]),
        "bytes": int(status["bytes"]),
        "observed_rows": parquet.get("rows"),
        "observed_columns": ",".join(parquet.get("columns", [])),
        "status": "PASS" if passed else ("FAIL" if spec["required"] else "WARN"),
    }


def build_required_input_inventory(root: Path = ROOT) -> pd.DataFrame:
    return pd.DataFrame([status_for_spec(spec) for spec in required_input_specs(root)])


def planned_fate_shape(endpoint_count: int, node_count: int = EXPECTED_NODES) -> str:
    return f"{int(node_count)}x{int(endpoint_count)}"


def build_planned_output_inventory(
    production_root: Path = M4C_V2_PRODUCTION_ROOT,
    endpoint_count: int = 12,
) -> pd.DataFrame:
    fate_shape = planned_fate_shape(endpoint_count)
    rows = [
        (
            "fate_probability_matrix_v2",
            "fate_probabilities/fate_probability_matrix_v2.npz",
            "numpy_npz",
            fate_shape,
            "Dense float32 endpoint probabilities keyed by global_node_index.",
        ),
        (
            "fate_probability_node_summary_v2",
            "fate_probabilities/fate_probability_node_summary_v2.parquet",
            "parquet_table",
            f"{EXPECTED_NODES} rows",
            "Node-level dominant endpoint and plasticity summary.",
        ),
        (
            "dominant_endpoint_assignment_v2",
            "fate_probabilities/dominant_endpoint_assignment_v2.parquet",
            "parquet_table",
            f"{EXPECTED_NODES} rows",
            "Minimal dominant endpoint assignment table.",
        ),
        (
            "plasticity_score_v2",
            "fate_probabilities/plasticity_score_v2.parquet",
            "parquet_table",
            f"{EXPECTED_NODES} rows",
            "Entropy and top1-margin plasticity metrics.",
        ),
        (
            "fate_probability_by_time_summary_v2",
            "fate_probabilities/fate_probability_by_time_summary_v2.csv",
            "csv_table",
            "time x endpoint",
            "Endpoint mass and dominant endpoint composition by time.",
        ),
        (
            "fate_probability_by_slice_summary_v2",
            "fate_probabilities/fate_probability_by_slice_summary_v2.csv",
            "csv_table",
            "slice x endpoint",
            "Endpoint mass and dominant endpoint composition by slice.",
        ),
        (
            "fate_probability_by_mouse_summary_v2",
            "fate_probabilities/fate_probability_by_mouse_summary_v2.csv",
            "csv_table",
            "mouse x endpoint",
            "Endpoint mass and dominant endpoint composition by mouse.",
        ),
        (
            "endpoint_composition_summary_v2",
            "fate_probabilities/endpoint_composition_summary_v2.csv",
            "csv_table",
            "endpoint summary rows",
            "Refined endpoint composition and confidence-tier summary.",
        ),
        (
            "m4c_v1_vs_v2_node_comparison",
            "comparison/m4c_v1_vs_v2_node_comparison.parquet",
            "parquet_table",
            f"{EXPECTED_NODES} rows",
            "Per-node v1/v2 fate agreement and probability deltas.",
        ),
        (
            "m4c_v1_vs_v2_global_summary",
            "comparison/m4c_v1_vs_v2_global_summary.csv",
            "csv_table",
            "global metric rows",
            "Global v1/v2 fate probability comparison.",
        ),
        (
            "m4c_v2_fate_probability_qc_summary",
            "reports/m4c_v2_fate_probability_qc_summary.csv",
            "csv_report",
            "one QC row",
            "Fate matrix row-sum, invalid entry, and endpoint mapping QC.",
        ),
        (
            "m4c_v2_fate_propagation_report",
            "reports/m4c_v2_fate_propagation_report.md",
            "markdown_report",
            "one report",
            "Execution report for a future approved M4C-v2 run.",
        ),
        (
            "m4c_v2_output_inventory",
            "reports/m4c_v2_output_inventory.csv",
            "csv_report",
            "all created outputs",
            "Manifest of completed M4C-v2 production outputs.",
        ),
        (
            "visualization_ready_node_table_v2",
            "visualization_tables/m4c_v2_visualization_node_table.parquet",
            "parquet_table",
            f"{EXPECTED_NODES} rows",
            "Visualization-ready table; no fate maps created by this plan.",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "output_name": name,
                "planned_path": str(production_root / relative_path),
                "object_type": object_type,
                "expected_shape_or_rows": expected_shape,
                "required_for_m4c_v2": True,
                "write_stage": "future_m4c_v2_execution",
                "overwrite_policy": "refuse existing output unless --overwrite is explicit",
                "production_created_in_this_task": False,
                "description": description,
            }
            for name, relative_path, object_type, expected_shape, description in rows
        ]
    )


def normalize_endpoint_tier(value: Any) -> str:
    text = str(value).lower()
    if "high_confidence" in text:
        return "high_confidence"
    if "slice" in text or "mouse" in text:
        return "slice_or_mouse_associated"
    if "mixed" in text or "unresolved" in text:
        return "plausible_but_mixed"
    if "rare" in text or "low_size" in text or "low_mass" in text:
        return "low_size_or_low_mass"
    return "needs_review"


def summarize_endpoint_taxonomy(endpoint_mapping: pd.DataFrame) -> pd.DataFrame:
    if endpoint_mapping.empty:
        return pd.DataFrame(
            columns=["reuse_category", "n_endpoints", "refined_endpoint_ids"]
        )
    mapping = endpoint_mapping.copy()
    count_column = (
        "raw_terminal_macrostate"
        if "raw_terminal_macrostate" in mapping.columns
        else "refined_endpoint_id"
    )
    mapping["reuse_category"] = mapping["confidence_tier_after_refinement"].map(
        normalize_endpoint_tier
    )
    grouped = (
        mapping.groupby("reuse_category", sort=True, observed=True)
        .agg(
            n_endpoints=(count_column, "nunique"),
            refined_endpoint_ids=(
                "refined_endpoint_id",
                lambda values: ",".join(sorted(set(map(str, values)))),
            ),
        )
        .reset_index()
    )
    return grouped


def endpoint_count_from_mapping(endpoint_mapping: pd.DataFrame) -> int:
    """Return the planned M4C endpoint column count.

    M4E can flag candidate endpoint merges by assigning the same refined label to
    multiple raw terminal macrostates. M4C-v2 planning does not approve such
    merges, so the executable contract preserves the raw M4C-v1 terminal
    macrostate cardinality and carries refined labels as annotations.
    """

    if endpoint_mapping.empty:
        return 0
    if "raw_terminal_macrostate" in endpoint_mapping.columns:
        return int(endpoint_mapping["raw_terminal_macrostate"].nunique())
    return int(endpoint_mapping["refined_endpoint_id"].nunique())


def inspect_m4c_v1(root: Path = ROOT) -> dict[str, Any]:
    script = PROJECT_ROOT / "scripts" / "m4c_01_compute_markov_fate_probabilities.py"
    config = PROJECT_ROOT / "configs" / "m4c_fate_probability.yaml"
    schema_path = root / "m4c" / "reports" / "m4c_fate_probability_schema.json"
    qc_path = root / "m4c" / "reports" / "m4c_fate_probability_qc_summary.csv"
    script_text = script.read_text(encoding="utf-8") if script.is_file() else ""
    config_payload = load_yaml_file(config)
    schema = read_json(schema_path)
    qc = read_csv(qc_path)
    qc_row = qc.iloc[0].to_dict() if not qc.empty else {}
    return {
        "read_only_inspection": True,
        "script": str(script),
        "config": str(config),
        "script_exists": bool(script.is_file()),
        "config_exists": bool(config.is_file()),
        "method_from_config": config_payload.get("fate", {}).get("method"),
        "method_from_schema": schema.get("method"),
        "uses_time_layered_backward_propagation": (
            "compute_fate_probabilities" in script_text
            and "time_layered_backward_propagation"
            in json.dumps(config_payload, sort_keys=True)
        ),
        "validates_fate_matrix": "validate_fate_matrix" in script_text,
        "builds_node_summary": "build_node_summary" in script_text,
        "fate_matrix_shape": schema.get("fate_probability_matrix", {}).get("shape"),
        "terminal_macrostate_count": len(schema.get("terminal_macrostate_ids", [])),
        "row_sum_tolerance": qc_row.get("row_sum_tolerance"),
        "nonfinal_row_sum_error_max": qc_row.get("nonfinal_row_sum_error_max"),
        "nan_values": qc_row.get("nan_values"),
        "negative_values": qc_row.get("negative_values"),
    }


def collect_state() -> dict[str, Any]:
    input_inventory = build_required_input_inventory(ROOT)
    endpoint_mapping = read_csv(
        ROOT / "m4e" / "endpoint_refinement" / "refined_endpoint_mapping.csv"
    )
    endpoint_tiers = summarize_endpoint_taxonomy(endpoint_mapping)
    m4a_v2_summary = read_json(
        ROOT / "m4a_v2_benchmark" / "m4a_v2_benchmark_summary.json"
    )
    m4a_v2_readiness = read_csv(
        ROOT
        / "m4a_v2_benchmark"
        / "m4a_v2_m4c_v2_required_inputs_checklist.csv"
    )
    m4a_v2_qc = read_csv(ROOT / "m4a_v2" / "reports" / "m4a_v2_02_qc_summary.csv")
    v1_v2_global = read_csv(
        ROOT
        / "m4a_v2_benchmark"
        / "m4a_v1_vs_v2_matrix_comparison_global.csv"
    )
    v1_v2_by_time = read_csv(
        ROOT
        / "m4a_v2_benchmark"
        / "m4a_v1_vs_v2_matrix_comparison_by_time_pair.csv"
    )
    endpoint_count = endpoint_count_from_mapping(endpoint_mapping)
    unique_refined_endpoint_count = (
        int(endpoint_mapping["refined_endpoint_id"].nunique())
        if not endpoint_mapping.empty
        else 0
    )
    merge_candidate_count = int(
        endpoint_mapping["refined_endpoint_id"].duplicated(keep=False).sum()
    ) if not endpoint_mapping.empty else 0
    planned_outputs = build_planned_output_inventory(
        M4C_V2_PRODUCTION_ROOT,
        endpoint_count,
    )
    return {
        "generated_at_utc": utc_now(),
        "input_inventory": input_inventory,
        "planned_outputs": planned_outputs,
        "endpoint_mapping": endpoint_mapping,
        "endpoint_tiers": endpoint_tiers,
        "m4a_v2_summary": m4a_v2_summary,
        "m4a_v2_readiness": m4a_v2_readiness,
        "m4a_v2_qc": m4a_v2_qc,
        "v1_v2_global": v1_v2_global,
        "v1_v2_by_time": v1_v2_by_time,
        "m4c_v1": inspect_m4c_v1(ROOT),
        "endpoint_count": endpoint_count,
        "unique_refined_endpoint_count": unique_refined_endpoint_count,
        "merge_candidate_mapping_rows": merge_candidate_count,
        "planned_fate_matrix_shape": planned_fate_shape(endpoint_count),
    }


def choose_execution_recommendation(state: dict[str, Any]) -> tuple[str, str]:
    inventory = state["input_inventory"]
    required_pass = bool(
        (inventory.loc[inventory["required"].astype(bool), "status"] == "PASS").all()
    )
    summary = state["m4a_v2_summary"]
    readiness = state["m4a_v2_readiness"]
    readiness_pass = bool(
        not readiness.empty and (readiness["status"].astype(str) == "PASS").all()
    )
    m4a_qc_pass = summary.get("full_qc_status") == "PASS"
    m4c_ready = summary.get("m4c_v2_readiness_status") == "PASS"
    if required_pass and readiness_pass and m4a_qc_pass and m4c_ready:
        return (
            "implement_m4c_v2_runner_dryrun_preflight_only",
            "Planning inputs and readiness gates pass; execution still requires "
            "a separate dry-run/preflight runner task.",
        )
    return (
        "repair_planning_inputs_before_m4c_v2_runner",
        "One or more required planning inputs or readiness checks failed.",
    )


def build_planning_checklist(
    state: dict[str, Any],
    safety: dict[str, Any] | None = None,
) -> pd.DataFrame:
    safety = safety or {}
    inventory = state["input_inventory"]
    required_failures = int(
        (inventory.loc[inventory["required"].astype(bool), "status"] != "PASS").sum()
    )
    readiness = state["m4a_v2_readiness"]
    readiness_failures = (
        int((readiness["status"].astype(str) != "PASS").sum())
        if not readiness.empty
        else 1
    )
    summary = state["m4a_v2_summary"]
    recommendation, _ = choose_execution_recommendation(state)
    rows = [
        (
            "scope",
            "planning_only_no_m4c_v2_execution",
            "PASS",
            "No fate matrix generation or M4C-v2 production root creation is performed.",
        ),
        (
            "inspection",
            "m4c_v1_inspected_read_only",
            "PASS" if state["m4c_v1"]["uses_time_layered_backward_propagation"] else "FAIL",
            "M4C-v1 script/config/schema inspected without modification.",
        ),
        (
            "inputs",
            "required_input_inventory",
            "PASS" if required_failures == 0 else "FAIL",
            f"required input failures: {required_failures}",
        ),
        (
            "inputs",
            "m4a_v2_full_qc_pass",
            "PASS" if summary.get("full_qc_status") == "PASS" else "FAIL",
            str(summary.get("full_qc_status")),
        ),
        (
            "inputs",
            "m4a_v2_m4c_v2_readiness_pass",
            "PASS" if readiness_failures == 0 else "FAIL",
            f"readiness failures: {readiness_failures}",
        ),
        (
            "endpoint_taxonomy",
            "reuse_m4e_refined_endpoint_taxonomy",
            "PASS" if state["endpoint_count"] > 0 else "FAIL",
            f"refined endpoint count: {state['endpoint_count']}",
        ),
        (
            "output_contract",
            "production_outputs_are_planned_not_created",
            "PASS",
            "Planned output inventory targets m4c_v2; this task writes only m4c_v2_plan.",
        ),
        (
            "safety",
            "upstream_metadata_diff_zero",
            "PASS" if safety.get("upstream_metadata_diff_count", 0) == 0 else "FAIL",
            f"diff count: {safety.get('upstream_metadata_diff_count', 0)}",
        ),
        (
            "safety",
            "forbidden_downstream_diff_zero",
            "PASS"
            if safety.get("forbidden_downstream_diff_count", 0) == 0
            else "FAIL",
            f"diff count: {safety.get('forbidden_downstream_diff_count', 0)}",
        ),
        (
            "safety",
            "ssd_output_check_zero",
            "PASS" if safety.get("ssd_output_count", 0) == 0 else "FAIL",
            f"/ssd output count: {safety.get('ssd_output_count', 0)}",
        ),
        (
            "recommendation",
            recommendation,
            "PASS"
            if recommendation == "implement_m4c_v2_runner_dryrun_preflight_only"
            else "FAIL",
            "Next step is implementation of the dry-run/preflight runner only.",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "category": category,
                "check": check,
                "status": status,
                "details": details,
            }
            for category, check, status, details in rows
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
    if frame.empty:
        return "_No rows available._"
    clipped = frame.loc[:, [column for column in columns if column in frame.columns]]
    clipped = clipped.head(max_rows).copy()
    clipped = clipped.fillna("")
    header = "| " + " | ".join(clipped.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(clipped.columns)) + " |"
    rows = [
        "| "
        + " | ".join(str(value).replace("\n", " ") for value in row)
        + " |"
        for row in clipped.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def safety_note() -> str:
    return "\n".join(
        [
            "- No M4C-v2 fate propagation was executed.",
            "- No M4C-v2 fate matrix was generated.",
            "- No pyGPCCA, K_gpcca, M4D diagnostics, barcode preprocessing, M5, "
            "BranchSBM, or Branched NicheFlow was run.",
            "- No upstream M3/M3-v2/M4A-v1/M4A-v2/M4B-v1/M4C-v1 outputs were modified.",
            "- No `/ssd` path was used.",
        ]
    )


def report_fate_plan(state: dict[str, Any]) -> str:
    summary = state["m4a_v2_summary"]
    return f"""# M4C-v2 Fate Propagation Plan

## Scope
M4C-v2 is the endpoint-anchored Markov propagation branch for M4A-v2. It is
pseudo-only, uses M3-v2 constrained_v1prior_sharpening evidence through M4A-v2,
and is not barcode-aware or GPCCA-derived.

## Planned Inputs
- M4A-v2 absorbing matrix: `P_absorbing_terminal_selfloops_v2`.
- M4A-v2 forward matrix: `P_forward_no_terminal_selfloops_v2`.
- M4A-v2 `global_node_table` preserving `row i == global_node_index i`.
- M4E refined endpoint mapping and D35 endpoint annotations.
- M4C-v1 outputs for comparison only.

## Planned Propagation Semantics
1. Load M4A-v2 node ordering and validate `{EXPECTED_NODES}` rows.
2. Load the M4E refined endpoint taxonomy without redefining terminal states.
3. Initialize a `{state['planned_fate_matrix_shape']}` fate matrix where D35
   endpoint rows are one-hot over refined endpoints.
4. Use the M4A-v2 forward DAG blocks for time-layered backward propagation,
   matching M4C-v1 semantics and avoiding dense absorbing-chain solves.
5. Require `P_absorbing_terminal_selfloops_v2` as the canonical structural
   transition object and validate that D35 self-loops are present.
6. Write node-level dominant endpoint, plasticity, group summaries, QC reports,
   and v1-v2 comparison tables only during a separately approved execution task.

## Expected M4A-v2 Matrix State
- matrix shape: `{summary.get('matrix_shape', f'{EXPECTED_NODES}x{EXPECTED_NODES}')}`
- forward nnz: `{summary.get('forward_nnz', EXPECTED_V2_FORWARD_NNZ)}`
- absorbing nnz: `{summary.get('absorbing_nnz', EXPECTED_V2_ABSORBING_NNZ)}`
- source rows: `{summary.get('source_rows', EXPECTED_SOURCE_ROWS)}`
- final-time D35 rows: `{summary.get('d35_self_loop_count', EXPECTED_FINAL_NODES)}`
- row-sum max error: `{summary.get('forward_row_sum_max_error')}`

## Safety
{safety_note()}
"""


def report_input_contract(state: dict[str, Any]) -> str:
    columns = [
        "input_name",
        "object_type",
        "required",
        "status",
        "expected_shape_or_rows",
        "join_keys",
        "path",
    ]
    return f"""# M4C-v2 Input Contract

All inputs are read-only. Missing required inputs must fail future preflight
before any production write.

{markdown_table(state['input_inventory'], columns)}

## Failure Behavior
- Required M4A-v2 matrix, node-table, QC, M4E endpoint, and M4C-v1 comparison
  inputs must pass before execution.
- Optional biological interpretation can degrade to warnings only if explicitly
  documented in the future runner.
- M4C-v2 must not repair or rewrite upstream artifacts.
"""


def report_output_contract(state: dict[str, Any]) -> str:
    columns = [
        "output_name",
        "object_type",
        "expected_shape_or_rows",
        "production_created_in_this_task",
        "planned_path",
    ]
    return f"""# M4C-v2 Output Contract

Planned production root: `{M4C_V2_PRODUCTION_ROOT}`.

No planned production output is created in this planning task. Future execution
must refuse existing outputs unless `--overwrite` is explicit.

{markdown_table(state['planned_outputs'], columns)}
"""


def report_endpoint_taxonomy(state: dict[str, Any]) -> str:
    endpoint_columns = [
        "raw_terminal_macrostate",
        "refined_endpoint_id",
        "refined_endpoint_label",
        "confidence_tier_after_refinement",
        "refined_endpoint_category",
    ]
    return f"""# M4C-v2 Endpoint Taxonomy Reuse Plan

## Decision
Reuse the existing M4E refined endpoint taxonomy. M4C-v2 must not create new
terminal states unless a later task separately approves a taxonomy revision.
The planned fate matrix preserves `{state['endpoint_count']}` raw M4C-v1
terminal macrostate columns and carries refined endpoint labels as annotations.
Unique refined endpoint IDs currently count `{state['unique_refined_endpoint_count']}`;
duplicate refined IDs are treated as merge candidates, not approved merges.

## Confidence Tier Summary
{markdown_table(state['endpoint_tiers'], ['reuse_category', 'n_endpoints', 'refined_endpoint_ids'])}

## Endpoint Mapping
{markdown_table(state['endpoint_mapping'], endpoint_columns, max_rows=20)}

## Interpretation Rules
- High-confidence endpoints can be interpreted as primary biological endpoint
  labels.
- Plausible-but-mixed endpoints must retain mixed-label caveats.
- Low-size/low-mass endpoints must be carried forward with rare-endpoint caveats.
- Slice/mouse-associated endpoints must be flagged in M4C-v2 summaries and
  v1-v2 comparisons.
"""


def report_qc_plan(state: dict[str, Any]) -> str:
    return f"""# M4C-v2 QC And Validation Plan

## Matrix And Node QC
- Fate matrix shape must equal `{state['planned_fate_matrix_shape']}`.
- M4A-v2 node order must remain `row i == global_node_index i`.
- D35 endpoint rows must be one-hot over refined endpoints.
- Non-final fate rows must sum to 1 within `{ROW_SUM_TOLERANCE:g}`.
- NaN, inf, and negative fate probabilities must be zero.
- Endpoint mapping missing count must be zero for D35 terminal candidates.

## Transition QC
- `P_absorbing_terminal_selfloops_v2` must exist and have D35 self-loops.
- `P_forward_no_terminal_selfloops_v2` must have zero D35 outgoing rows.
- M4A-v2 row-sum, duplicate-coordinate, and mapping QC must be imported from
  the completed M4A-v2 reports.

## Safety QC
- Upstream metadata diff must be zero.
- Forbidden downstream diff must be zero before execution begins.
- `/ssd` output count must be zero.
- No GPCCA, K_gpcca, barcode, M5, BranchSBM, or M4D artifacts may be created.
"""


def report_v1_comparison_plan(state: dict[str, Any]) -> str:
    global_columns = [
        "comparison_scope",
        "source_rows_v1",
        "source_rows_v2",
        "forward_nnz_v1",
        "forward_nnz_v2",
        "top1_delta_v2_minus_v1",
        "entropy_delta_v2_minus_v1",
    ]
    by_time_columns = [
        "time_pair",
        "source_rows_v1",
        "source_rows_v2",
        "edge_count_v1",
        "edge_count_v2",
        "source_coverage_preserved",
        "v2_sparser",
    ]
    return f"""# M4C-v2 V1 Comparison Plan

## Metrics
- Fate row-sum QC and invalid-entry counts.
- Dominant endpoint agreement and transition to refined endpoint labels.
- Endpoint probability entropy, top1 probability, and top1 margin.
- Plasticity score distribution and v2-v1 deltas.
- Endpoint mass by time point, slice, mouse, and Leiden neighborhood.
- JS divergence and correlation between v1 and v2 endpoint probabilities.
- Artifact checks for slice/mouse concentration and non-final zero-outgoing rows.

## Expected Behavior
- M4A-v2 is sparser and should produce sharper probabilities than M4A-v1.
- Source row coverage should remain preserved.
- M4C-v2 remains complementary to M4C-v1 unless benchmark evidence supports
  replacement in a later decision.

## Existing M4A-v1 vs M4A-v2 Matrix Context
{markdown_table(state['v1_v2_global'], global_columns)}

## By-Time-Pair Context
{markdown_table(state['v1_v2_by_time'], by_time_columns)}
"""


def report_visualization_plan() -> str:
    return """# M4C-v2 Visualization Plan

## Lightweight Tables And Figures
- Endpoint probability heatmaps by time, slice, mouse, and neighborhood.
- Dominant endpoint composition bar plots and heatmaps.
- Plasticity and top1-margin distributions by time and metadata group.
- M4C-v1 vs M4C-v2 agreement maps and summary plots.
- V1-v2 plasticity delta maps generated only in a future visualization task.

## Non-Goals
- Do not generate M4C-v2 fate maps in this planning task.
- Do not generate GPCCA, K_gpcca, or M4D plots.
- Do not compute fate probabilities as part of visualization planning.
"""


def report_resume_plan() -> str:
    return """# M4C-v2 Resume And Failure Recovery Plan

## Future Runner Requirements
- Provide dry-run/preflight mode before production execution.
- Refuse existing outputs by default; require explicit `--overwrite`.
- Use atomic writes where feasible and write large temporaries only under
  `/home/zhutao/scratch/nichefate/m4c_v2/tmp`.
- Maintain completed and failed step manifests.
- Support safe resume after interrupted fate matrix generation.
- Validate all required inputs before creating production outputs.

## Failure Handling
- If input schema or endpoint mapping fails, stop before matrix allocation.
- If row-sum or invalid-entry QC fails, write a failed-step report and stop.
- If comparison inputs are missing, stop unless a later task explicitly allows
  execution without v1-v2 benchmark outputs.

## Safety
""" + safety_note() + "\n"


def build_summary(
    state: dict[str, Any],
    checklist: pd.DataFrame,
    safety: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    recommendation, recommendation_reason = choose_execution_recommendation(state)
    checklist_pass = bool((checklist["status"].astype(str) == "PASS").all())
    return {
        "stage": "M4C-v2-00",
        "status": "PASSED" if checklist_pass else "FAILED",
        "generated_at_utc": state["generated_at_utc"],
        "runtime_seconds": runtime_seconds,
        "planning_only": True,
        "m4c_v2_execution_run": False,
        "fate_matrix_generated": False,
        "m4c_v1_inspection_read_only": True,
        "m4a_v2_inputs_confirmed": (
            state["m4a_v2_summary"].get("full_qc_status") == "PASS"
            and state["m4a_v2_summary"].get("m4c_v2_readiness_status") == "PASS"
        ),
        "endpoint_taxonomy_reuse_decision": "reuse_m4e_refined_endpoint_taxonomy",
        "planned_output_root": str(M4C_V2_PRODUCTION_ROOT),
        "planned_fate_matrix_shape": state["planned_fate_matrix_shape"],
        "planned_node_count": EXPECTED_NODES,
        "planned_endpoint_count": state["endpoint_count"],
        "unique_refined_endpoint_count": state["unique_refined_endpoint_count"],
        "merge_candidate_mapping_rows": state["merge_candidate_mapping_rows"],
        "planned_final_time": EXPECTED_FINAL_TIME,
        "planned_qc_criteria": {
            "row_sum_tolerance": ROW_SUM_TOLERANCE,
            "invalid_entries": 0,
            "missing_endpoint_mapping": 0,
            "non_final_zero_outgoing_rows": 0,
            "upstream_metadata_diff_count": 0,
            "forbidden_downstream_diff_count": 0,
            "ssd_output_count": 0,
        },
        "planned_v1_v2_comparison_metrics": [
            "dominant_endpoint_agreement",
            "endpoint_probability_entropy",
            "plasticity_score_distribution",
            "endpoint_mass_by_time",
            "endpoint_mass_by_slice_mouse",
            "endpoint_mass_by_neighborhood",
            "js_divergence",
            "correlation",
            "spatial_coherence_diagnostics",
        ],
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "exact_next_recommended_step": (
            "Implement the M4C-v2 runner with dry-run/preflight only; do not "
            "execute M4C-v2 fate propagation until separately approved."
        ),
        **safety,
    }


def write_artifacts(
    paths: dict[str, Path],
    state: dict[str, Any],
    safety: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    checklist = build_planning_checklist(state, safety)
    summary = build_summary(state, checklist, safety, runtime_seconds)

    atomic_write_csv(paths["input_inventory"], state["input_inventory"])
    atomic_write_csv(paths["planned_output_inventory"], state["planned_outputs"])
    atomic_write_csv(paths["planning_checklist"], checklist)
    atomic_write_json(paths["summary"], summary)

    atomic_write_text(paths["m4c_v2_fate_propagation_plan.md"], report_fate_plan(state))
    atomic_write_text(paths["m4c_v2_input_contract.md"], report_input_contract(state))
    atomic_write_text(paths["m4c_v2_output_contract.md"], report_output_contract(state))
    atomic_write_text(
        paths["m4c_v2_endpoint_taxonomy_reuse_plan.md"],
        report_endpoint_taxonomy(state),
    )
    atomic_write_text(paths["m4c_v2_qc_and_validation_plan.md"], report_qc_plan(state))
    atomic_write_text(paths["m4c_v2_v1_comparison_plan.md"], report_v1_comparison_plan(state))
    atomic_write_text(paths["m4c_v2_visualization_plan.md"], report_visualization_plan())
    atomic_write_text(
        paths["m4c_v2_resume_and_failure_recovery_plan.md"],
        report_resume_plan(),
    )
    return summary


def validate_generated_outputs(paths: dict[str, Path]) -> None:
    required = [
        paths["summary"],
        paths["input_inventory"],
        paths["planned_output_inventory"],
        paths["planning_checklist"],
        *[paths[name] for name in REPORT_FILES],
    ]
    missing_or_empty = [
        str(path)
        for path in required
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing_or_empty:
        raise RuntimeError(f"Missing or empty planning outputs: {missing_or_empty}")
    with paths["summary"].open("r", encoding="utf-8") as handle:
        json.load(handle)
    expected_csv_columns = {
        "input_inventory": {"input_name", "path", "status", "read_only"},
        "planned_output_inventory": {
            "output_name",
            "planned_path",
            "production_created_in_this_task",
        },
        "planning_checklist": {"category", "check", "status", "details"},
    }
    for name, columns in expected_csv_columns.items():
        frame = pd.read_csv(paths[name])
        missing = sorted(columns - set(frame.columns))
        if missing:
            raise RuntimeError(f"{paths[name]} is missing columns: {missing}")


def count_ssd_outputs(paths: dict[str, Path]) -> int:
    planned = [path for key, path in paths.items() if key != "reports"]
    return int(sum(str(resolved(path)).startswith("/ssd/") for path in planned))


def run(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    start = time.monotonic()
    paths = output_paths(output_root)
    before_upstream = snapshot(PROTECTED_ROOTS)
    before_forbidden = snapshot(FORBIDDEN_DOWNSTREAM_ROOTS)
    ensure_output_dirs(paths)
    state = collect_state()
    provisional_safety = {
        "upstream_metadata_diff_count": 0,
        "upstream_metadata_diffs": [],
        "forbidden_downstream_diff_count": 0,
        "forbidden_downstream_diffs": [],
        "ssd_output_count": count_ssd_outputs(paths),
    }
    write_artifacts(paths, state, provisional_safety, time.monotonic() - start)

    after_upstream = snapshot(PROTECTED_ROOTS)
    after_forbidden = snapshot(FORBIDDEN_DOWNSTREAM_ROOTS)
    upstream_diffs = diff_snapshot(before_upstream, after_upstream)
    forbidden_diffs = diff_snapshot(before_forbidden, after_forbidden)
    safety = {
        "upstream_metadata_diff_count": len(upstream_diffs),
        "upstream_metadata_diffs": upstream_diffs,
        "forbidden_downstream_diff_count": len(forbidden_diffs),
        "forbidden_downstream_diffs": forbidden_diffs,
        "ssd_output_count": count_ssd_outputs(paths),
    }
    summary = write_artifacts(paths, state, safety, time.monotonic() - start)
    validate_generated_outputs(paths)
    return summary


def main() -> int:
    args = parse_args()
    summary = run(args.output_root)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
