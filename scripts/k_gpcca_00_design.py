#!/usr/bin/env python
"""Generate the K_gpcca-00 standard GPCCA-compatible kernel design package.

This script is design-only. It writes reports and design inventories under
``/home/zhutao/scratch/nichefate/k_gpcca_design`` and never constructs a
K_gpcca matrix, runs pyGPCCA/CellRank, or modifies production artifacts.
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

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ROOT = Path("/home/zhutao/scratch/nichefate")
DEFAULT_OUTPUT_ROOT = ROOT / "k_gpcca_design"
FUTURE_K_ROOT = ROOT / "k_gpcca"

PROTECTED_ROOTS = [
    ROOT / "m3",
    ROOT / "m3_v2",
    ROOT / "m4a",
    ROOT / "m4a_v2",
    ROOT / "m4b",
    ROOT / "m4c",
    ROOT / "m4c_v2",
]

FORBIDDEN_EXECUTION_ROOTS = [
    ROOT / "m4d",
    ROOT / "gpcca",
    FUTURE_K_ROOT,
    ROOT / "m5",
    ROOT / "branchsbm",
    ROOT / "barcode",
    ROOT / "darlin",
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

REPORT_NAMES = [
    "k_gpcca_design_overview.md",
    "p_fate_vs_k_gpcca_design_distinction.md",
    "k_gpcca_kernel_mathematical_spec.md",
    "k_gpcca_input_contract.md",
    "k_gpcca_output_contract.md",
    "k_gpcca_pilot_protocol.md",
    "k_gpcca_supernode_strategy.md",
    "k_gpcca_pyGPCCA_execution_policy.md",
    "k_gpcca_acceptance_criteria.md",
    "k_gpcca_barcode_extension_contract.md",
    "k_gpcca_risk_register.md",
]

CSV_NAMES = [
    "k_gpcca_design_checklist.csv",
    "k_gpcca_candidate_parameter_grid.csv",
    "k_gpcca_planned_output_inventory.csv",
]

SUMMARY_NAME = "k_gpcca_design_summary.json"
NEXT_STEP = (
    "K_gpcca-01 pilot kernel constructor dry-run/preflight only, or revise "
    "K_gpcca design if major issues are found; do not construct K_gpcca in this task."
)


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
    for protected in PROTECTED_ROOTS:
        if paths_overlap(output_root, protected):
            raise ValueError(
                f"Output root overlaps protected production root {protected}: {output_root}"
            )
    for forbidden in FORBIDDEN_EXECUTION_ROOTS:
        if paths_overlap(output_root, forbidden):
            raise ValueError(
                f"Output root overlaps forbidden execution root {forbidden}: {output_root}"
            )
    return output_root


def output_paths(output_root: Path) -> dict[str, Path]:
    root = validate_output_root(output_root)
    reports = root / "reports"
    paths = {
        "root": root,
        "reports": reports,
        "summary": root / SUMMARY_NAME,
    }
    for name in REPORT_NAMES:
        paths[name] = reports / name
    for name in CSV_NAMES:
        paths[name] = root / name
    return paths


def ensure_dirs(paths: dict[str, Path]) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
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


def count_ssd_outputs(output_root: Path) -> int:
    output_root = resolved(output_root)
    if output_root == Path("/ssd") or Path("/ssd") in output_root.parents:
        return 1
    if not output_root.exists():
        return 0
    return int(
        sum(
            path.resolve() == Path("/ssd") or Path("/ssd") in path.resolve().parents
            for path in output_root.rglob("*")
        )
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_context() -> dict[str, Any]:
    m2_schema = load_json(ROOT / "m2" / "reports" / "m2_full_feature_schema.json")
    plan_a = load_json(ROOT / "planA_freeze" / "planA_freeze_summary.json")
    m4a_v2 = load_json(ROOT / "m4a_v2_benchmark" / "m4a_v2_benchmark_summary.json")
    m4c_v2 = load_json(ROOT / "m4c_v2_benchmark" / "m4c_v2_benchmark_summary.json")
    return {
        "m2_schema": m2_schema,
        "plan_a": plan_a,
        "m4a_v2": m4a_v2,
        "m4c_v2": m4c_v2,
    }


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for record in frame.astype(str).to_dict(orient="records"):
        values = [
            record[column].replace("|", "\\|").replace("\n", " ")
            for column in columns
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def build_candidate_parameter_grid() -> pd.DataFrame:
    rows = [
        ("pilot_v1_balanced", "full_resolution_subset", "M3-v1", 0.60, 0.35, 0.05, 0.0, 30, "cosine", "D9_D21_D35_or_D21_D35_subset", "default", "Balanced conservative cross-time pilot."),
        ("pilot_v2_balanced", "full_resolution_subset", "M3-v2", 0.60, 0.35, 0.05, 0.0, 30, "cosine", "D9_D21_D35_or_D21_D35_subset", "default", "Balanced sharpened cross-time pilot."),
        ("pilot_v1_directional", "full_resolution_subset", "M3-v1", 0.45, 0.50, 0.05, 0.0, 30, "cosine", "D9_D21_D35_or_D21_D35_subset", "sensitivity", "Tests stronger conservative directionality."),
        ("pilot_v2_directional", "full_resolution_subset", "M3-v2", 0.45, 0.50, 0.05, 0.0, 30, "cosine", "D9_D21_D35_or_D21_D35_subset", "sensitivity", "Tests stronger sharpened directionality."),
        ("pilot_v1_within_heavy", "full_resolution_subset", "M3-v1", 0.75, 0.22, 0.03, 0.0, 50, "cosine", "D9_D21_D35_or_D21_D35_subset", "sensitivity", "Tests stronger within-time manifold retention."),
        ("pilot_v2_within_heavy", "full_resolution_subset", "M3-v2", 0.75, 0.22, 0.03, 0.0, 50, "cosine", "D9_D21_D35_or_D21_D35_subset", "sensitivity", "Tests stronger within-time manifold retention."),
        ("pilot_v1_selfloop_001", "full_resolution_subset", "M3-v1", 0.62, 0.37, 0.01, 0.0, 30, "cosine", "D9_D21_D35_or_D21_D35_subset", "selfloop_sensitivity", "Low self-loop stabilization."),
        ("pilot_v2_selfloop_001", "full_resolution_subset", "M3-v2", 0.62, 0.37, 0.01, 0.0, 30, "cosine", "D9_D21_D35_or_D21_D35_subset", "selfloop_sensitivity", "Low self-loop stabilization."),
        ("pilot_v1_selfloop_003", "full_resolution_subset", "M3-v1", 0.60, 0.37, 0.03, 0.0, 30, "cosine", "D9_D21_D35_or_D21_D35_subset", "selfloop_sensitivity", "Moderate self-loop stabilization."),
        ("pilot_v2_selfloop_003", "full_resolution_subset", "M3-v2", 0.60, 0.37, 0.03, 0.0, 30, "cosine", "D9_D21_D35_or_D21_D35_subset", "selfloop_sensitivity", "Moderate self-loop stabilization."),
        ("pilot_v1_selfloop_010", "full_resolution_subset", "M3-v1", 0.55, 0.35, 0.10, 0.0, 30, "cosine", "D9_D21_D35_or_D21_D35_subset", "selfloop_sensitivity", "High self-loop stress test; reject if diagonal dominates."),
        ("pilot_v2_selfloop_010", "full_resolution_subset", "M3-v2", 0.55, 0.35, 0.10, 0.0, 30, "cosine", "D9_D21_D35_or_D21_D35_subset", "selfloop_sensitivity", "High self-loop stress test; reject if diagonal dominates."),
        ("pilot_mixed_cross_time_review", "full_resolution_subset", "M3-v1_v2_mixed", 0.60, 0.35, 0.05, 0.0, 30, "cosine", "D9_D21_D35_or_D21_D35_subset", "review_only", "Mixed cross-time evidence only if v1/v2 comparison justifies it."),
        ("supernode_v1_balanced", "supernode", "M3-v1", 0.60, 0.35, 0.05, 0.0, 30, "cosine", "coarse_grained_D9_D21_D35_or_D21_D35", "fallback_if_full_resolution_infeasible", "Computational fallback using standard pyGPCCA on coarse kernel."),
        ("supernode_v2_balanced", "supernode", "M3-v2", 0.60, 0.35, 0.05, 0.0, 30, "cosine", "coarse_grained_D9_D21_D35_or_D21_D35", "fallback_if_full_resolution_infeasible", "Computational fallback using standard pyGPCCA on coarse kernel."),
        ("future_barcode_placeholder", "future_barcode", "M3-v2_plus_barcode", 0.50, 0.30, 0.05, 0.15, 30, "cosine", "future_after_darlin_processed_clone_tables", "future_after_darlin", "Placeholder for barcode-aware K_gpcca extension; not executable now."),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "grid_id",
            "route",
            "cross_time_source",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "within_time_k",
            "similarity_metric",
            "scope",
            "priority",
            "rationale",
        ],
    )


def build_design_checklist() -> pd.DataFrame:
    rows = [
        ("p_fate_frozen", "PASS", "Plan A freeze decision retained; P_fate remains endpoint-anchored Markov propagation.", "Stop if Plan A freeze is missing or failed."),
        ("k_gpcca_separate_branch", "PASS", "K_gpcca is specified as a separate standard GPCCA-compatible kernel.", "Reject designs that reuse P_fate as formal GPCCA output."),
        ("base_kernel_formula_defined", "PASS", "row_normalize(alpha*K_within_time + beta*P_cross_time + gamma*I_self)", "Require mathematical spec before constructor work."),
        ("barcode_extension_formula_defined", "PASS", "row_normalize(alpha*K_within_time + beta*P_cross_time + delta*P_barcode_or_G_barcode + gamma*I_self)", "Keep barcode rows future-only until processed DARLIN tables exist."),
        ("within_time_contract_defined", "PASS", "Same-time kNN from M2 niche representation.", "Do not connect across time in K_within_time."),
        ("cross_time_contract_defined", "PASS", "M3-v1 and M3-v2 are cross-time evidence candidates.", "Do not label M3-v2 as GPCCA."),
        ("self_loop_contract_defined", "PASS", "gamma candidates 0.01/0.03/0.05/0.10.", "Reject settings where diagonal mass dominates."),
        ("full_resolution_and_supernode_routes_defined", "PASS", "Full subset first; supernode route only if infeasible.", "Report supernode sensitivity."),
        ("pygpcca_only_policy_defined", "PASS", "Formal results require pyGPCCA or CellRank-compatible GPCCA.", "Report failure; no heuristic final fallback."),
        ("acceptance_criteria_defined", "PASS", "Kernel, standard GPCCA, biological, artifact, and sensitivity criteria defined.", "Do not proceed without acceptance gate."),
        ("design_only_scope", "PASS", "No K_gpcca matrix construction or pyGPCCA execution in K_gpcca-00.", "Stop on any production output attempt."),
    ]
    return pd.DataFrame(
        rows,
        columns=["check", "status", "evidence", "failure_behavior"],
    )


def build_planned_output_inventory() -> pd.DataFrame:
    output_root = FUTURE_K_ROOT
    rows = [
        ("k_gpcca_sparse_matrix", output_root / "kernel_objects" / "K_gpcca.npz", "future_kernel", "sparse square Markov kernel", False, "K_gpcca-01_or_later"),
        ("k_gpcca_kernel_qc_report", output_root / "reports" / "k_gpcca_kernel_qc_report.md", "future_qc", "row-stochastic and connectivity QC", False, "K_gpcca-01_or_later"),
        ("k_gpcca_connectivity_diagnostics", output_root / "reports" / "k_gpcca_connectivity_diagnostics.csv", "future_qc", "weak components and zero-outgoing diagnostics", False, "K_gpcca-01_or_later"),
        ("pygpcca_macrostates", output_root / "gpcca" / "pygpcca_macrostates.parquet", "future_gpcca", "standard pyGPCCA macrostate assignments", False, "K_gpcca-02_or_later"),
        ("pygpcca_memberships", output_root / "gpcca" / "pygpcca_memberships.npz", "future_gpcca", "standard pyGPCCA membership matrix", False, "K_gpcca-02_or_later"),
        ("gpcca_fate_probabilities", output_root / "gpcca" / "gpcca_fate_probabilities.npz", "future_gpcca", "standard GPCCA fate probabilities", False, "K_gpcca-02_or_later"),
        ("macrostate_annotation", output_root / "reports" / "k_gpcca_macrostate_annotation.md", "future_annotation", "M4E/M2 macrostate annotation", False, "K_gpcca-02_or_later"),
        ("p_fate_comparison", output_root / "reports" / "k_gpcca_vs_p_fate_comparison.csv", "future_comparison", "comparison to M4C-v1/v2 P_fate outputs", False, "K_gpcca-03_or_later"),
        ("barcode_aware_kernel", output_root / "barcode_extension" / "K_gpcca_barcode.npz", "future_after_darlin", "barcode-aware kernel extension", False, "future_after_darlin"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "output_name",
            "planned_path",
            "category",
            "expected_content",
            "created_in_this_task",
            "earliest_stage",
        ],
    ).assign(planned_path=lambda frame: frame["planned_path"].astype(str))


def build_input_contract_rows() -> pd.DataFrame:
    rows = [
        ("m2_niche_representations", ROOT / "m2" / "by_slice", "required", "slice_id,time,time_day,anchor identifiers,numeric niche features", "anchor/cell keys plus time", "read_only", "fail preflight"),
        ("m2_feature_schema", ROOT / "m2" / "reports" / "m2_full_feature_schema.json", "required", "metadata_columns,numeric_feature_columns", "feature names", "read_only", "fail preflight"),
        ("m3_v1_cross_time_edges", ROOT / "m3" / "full_by_shard", "required_for_v1_pilot", "source_anchor_id,target_anchor_id,row-normalized transition probability", "anchor ids", "read_only", "skip v1 pilot if absent"),
        ("m3_v2_cross_time_edges", ROOT / "m3_v2" / "full_by_shard", "required_for_v2_pilot", "source_anchor_id,target_anchor_id,v2_row_normalized_transition_prob", "anchor ids", "read_only", "skip v2 pilot if absent"),
        ("m4a_v2_node_table", ROOT / "m4a_v2" / "node_table" / "global_node_table.parquet", "required", "global_node_index,anchor_id,time,time_day,slice_id,mouse_id", "anchor_id/global_node_index", "read_only", "fail preflight"),
        ("m4e_endpoint_taxonomy", ROOT / "m4e" / "endpoint_refinement" / "refined_endpoint_mapping.csv", "required_for_annotation", "raw_terminal_macrostate,refined_endpoint_id,confidence tier", "terminal macrostate id", "read_only", "warn for kernel, fail annotation"),
        ("m4e_neighborhood_annotation", ROOT / "m4e" / "neighborhood_annotation" / "node_neighborhood_annotation.parquet", "required_for_artifact_qc", "global_node_index or anchor keys,neighborhood labels", "node keys", "read_only", "warn for kernel, fail artifact QC"),
        ("p_fate_v1_v2_outputs", ROOT / "m4c_v2_benchmark", "required_for_comparison", "P_fate benchmark tables and summaries", "global_node_index/endpoint ids", "read_only", "skip comparison if absent"),
        ("darlin_processed_clone_tables", ROOT / "darlin" / "processed_clone_tables", "future_optional", "clone_id,barcode_id,cell_id,time,support/confidence", "cell/anchor keys", "read_only_future", "not used before DARLIN onboarding"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "input_name",
            "path",
            "required_status",
            "required_columns_or_objects",
            "join_keys",
            "access_mode",
            "failure_behavior",
        ],
    ).assign(path=lambda frame: frame["path"].astype(str))


def build_risk_register() -> pd.DataFrame:
    rows = [
        ("pygpcca_failure", "standard pyGPCCA still fails", "high", "Run kernel diagnostics first; report failure honestly; revise K construction.", "pyGPCCA exception or no convergence."),
        ("within_time_overwhelms_direction", "K_within_time overwhelms cross-time directionality", "medium", "Lower alpha, compare terminal classification and directional flow.", "Macrostates are same-time manifold clusters only."),
        ("cross_time_overwhelms_manifold", "P_cross_time overwhelms within-time manifold", "medium", "Lower beta, inspect spatial coherence and neighborhood preservation.", "Macrostates mirror P_fate edges without manifold structure."),
        ("self_loop_too_large", "Self-loop mass dominates the kernel", "medium", "Run gamma sensitivity and cap diagonal mass.", "High diagonal mass or poor mixing."),
        ("supernode_distortion", "Supernode clustering distorts branch structure", "medium", "Compare clustering methods and supernode resolutions.", "Macrostate assignments unstable across supernode settings."),
        ("slice_mouse_artifact", "Slice/mouse artifacts dominate macrostate structure", "high", "Stratified artifact QC and alpha/beta sensitivity.", "Large slice or mouse-associated macrostate shifts."),
        ("endpoint_label_collapse", "Macrostates become endpoint labels only", "medium", "Compare to P_fate; require independent macrostate structure.", "Dominant terminal agreement is trivial and membership entropy collapses."),
        ("fate_interpretability_failure", "GPCCA fate probabilities are not biologically interpretable", "medium", "Annotate with M4E/M2 metadata; require biological review.", "Terminal states lack coherent annotations."),
        ("runtime_memory_infeasible", "Full-resolution K_gpcca is infeasible", "high", "Bounded pilot first; use supernode route only as computational fallback.", "Memory/runtime exceeds pilot limits."),
    ]
    return pd.DataFrame(
        rows,
        columns=["risk_id", "risk", "severity", "mitigation", "trigger"],
    )


def context_metric(context: dict[str, Any], group: str, key: str, default: Any = "NA") -> Any:
    value = context.get(group, {}).get(key, default)
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def build_design_overview(context: dict[str, Any]) -> str:
    return f"""# K_gpcca-00 Design Overview

Generated: {utc_now()}

## Goal

Design a standard GPCCA-compatible K_gpcca branch for complete niche-level CellRank-like Plan A.

## Current Plan A State

- P_fate freeze decision: `{context_metric(context, "plan_a", "p_fate_freeze_decision", "keep_v1_and_v2_as_complementary_p_fate_branch")}`
- P_fate status: `{context_metric(context, "plan_a", "p_fate_status", "implemented_benchmarked_frozen")}`
- K_gpcca status: `{context_metric(context, "plan_a", "k_gpcca_status", "not_implemented_needs_design_and_pilot")}`
- M4A-v2 shape: `{context_metric(context, "m4a_v2", "matrix_shape", "1439542x1439542")}`
- M4C-v2 fate shape: `{context_metric(context, "m4c_v2", "fate_matrix_shape", "1439542x12")}`

## Design Scope

This task writes design reports only. It does not construct K_gpcca matrices, run pyGPCCA, run CellRank, run M4D diagnostics, or modify production outputs.

## Proposed Branch

K_gpcca is a separate Markov kernel for standard pyGPCCA / CellRank-compatible macrostate discovery. It combines within-time niche manifold connectivity, cross-time directional evidence, and self-loop stabilization.

## Next Step

{NEXT_STEP}
"""


def build_distinction_report() -> str:
    return """# P_fate vs K_gpcca Design Distinction

## P_fate

- Status: implemented, benchmarked, and frozen.
- Input: M4A-v1/v2 sparse transition matrices and M4E endpoint taxonomy.
- Output: endpoint fate probabilities, dominant endpoint assignment, and plasticity.
- Role: endpoint-anchored Markov propagation.
- Interpretation: pseudo-only fate map with conservative v1 and sharpened complementary v2 modes.

## K_gpcca

- Status: design-only in K_gpcca-00.
- Input: a future GPCCA-compatible kernel combining within-time manifold connectivity, cross-time evidence, and stabilization.
- Output: standard pyGPCCA macrostates, terminal/initial/intermediate classifications, and GPCCA fate probabilities.
- Role: CellRank-like macrostate discovery.

## Non-Equivalence

- K_gpcca is not the same as P_fate.
- Strictly time-forward P_fate should not be forced into pyGPCCA.
- M3-v2 can be cross-time evidence but is not a GPCCA solution.
- Custom GPCCA-like code must not be used as formal GPCCA output.
- P_fate remains a baseline/control even if K_gpcca succeeds.
"""


def build_math_spec_report(grid: pd.DataFrame) -> str:
    gamma_values = ", ".join(str(value) for value in sorted(grid["gamma"].unique()))
    return f"""# K_gpcca Kernel Mathematical Spec

## Base Kernel

`K_gpcca = row_normalize(alpha * K_within_time + beta * P_cross_time + gamma * I_self)`

Where:

- `K_within_time`: within-time niche manifold connectivity.
- `P_cross_time`: cross-time directional evidence from M3-v1 and/or M3-v2.
- `I_self`: self-loop / numerical stabilization / aperiodicity.
- `alpha`, `beta`, and `gamma`: tunable non-negative weights.

K_gpcca must be sparse, non-negative, row-stochastic, and suitable for standard pyGPCCA / CellRank-compatible GPCCA.

## Future Barcode Extension

`K_gpcca_barcode = row_normalize(alpha * K_within_time + beta * P_cross_time + delta * P_barcode_or_G_barcode + gamma * I_self)`

`delta` is a future tunable barcode-evidence weight and remains inactive until processed DARLIN clone/barcode tables are available.

## K_within_time

- Construct from M2 niche representations using same-time kNN only.
- Candidate features: molecular state embedding, cell-type composition features, spatial topology features, and niche representation features.
- Neighborhood labels may support QC or optional stratification, but should not force endpoint labels into the kernel.
- Candidate k values: 15, 30, 50.
- Candidate metrics: cosine and euclidean.
- Scaling: standardize numeric feature blocks; optionally L2-normalize cosine feature vectors.
- Row normalization: normalize same-time rows after pruning, then normalize the final weighted sum.
- QC: same-time-only edge check, row-sum error, degree distribution, slice/mouse mixing, component structure.

## P_cross_time

- M3-v1: conservative broad pseudo-transition component.
- M3-v2: sharpened constrained_v1prior_sharpening component.
- Mixed v1/v2: review-only pilot option if v1/v2 comparison justifies it.

## Self-Loop Stabilization

- Purpose: aperiodicity, local retention, numerical stabilization, and prevention of strictly layered DAG behavior.
- Candidate gamma values in the design grid: {gamma_values}.
- Self-loop mass must not dominate the kernel and must pass sensitivity analysis.
"""


def build_input_contract_report(input_contract: pd.DataFrame) -> str:
    return f"""# K_gpcca Input Contract

## Required Inputs

{dataframe_to_markdown(input_contract)}

## Contract Rules

- All P_fate and M3/M4A/M4C/M4E inputs are read-only.
- K_within_time uses M2 representations and must only connect nodes within the same time point.
- P_cross_time uses M3-v1 and/or M3-v2 evidence and must preserve explicit source/target anchor mapping.
- M4E annotations are used for interpretation and artifact QC, not for redefining K_gpcca terminal states.
- DARLIN barcode inputs are future-only and must come from official/lab-standard processed clone tables.
"""


def build_output_contract_report(outputs: pd.DataFrame) -> str:
    return f"""# K_gpcca Output Contract

## Planned Future Outputs

{dataframe_to_markdown(outputs)}

## K_gpcca-00 Rule

None of these future K_gpcca production outputs are created in this design task. K_gpcca-00 writes design reports and checklists only.
"""


def build_pilot_protocol_report(grid: pd.DataFrame) -> str:
    pilot_rows = grid[grid["route"] == "full_resolution_subset"][
        ["grid_id", "cross_time_source", "alpha", "beta", "gamma", "within_time_k", "priority"]
    ]
    return f"""# K_gpcca Pilot Protocol

## Pilot Scope

- Start small.
- Preferred pilot: D9/D21/D35 or D21/D35 plus within-time layers.
- Select a deterministic 20k–100k node subset or one bounded time-window subset.
- Include within-time edges, cross-time edges, and self-loops.
- Run standard pyGPCCA only after kernel QC passes.

## Candidate Full-Resolution Subset Settings

{dataframe_to_markdown(pilot_rows)}

## Pilot Gates

1. Preflight input schema and path safety.
2. Construct K_within_time, P_cross_time, and I_self only in K_gpcca-01 or later.
3. Validate row-stochasticity, invalid entries, duplicate coordinates, connectivity, and non-final zero-outgoing rows.
4. Run standard pyGPCCA only after QC passes.
5. Annotate macrostates with M4E/M2 metadata and compare to P_fate-v1/v2.

## Outputs Later

Future pilots should produce K_gpcca sparse matrix, kernel QC report, connectivity diagnostics, pyGPCCA macrostates, terminal/initial/intermediate classification, GPCCA fate probabilities, macrostate annotation, and P_fate comparison reports.
"""


def build_supernode_strategy_report() -> str:
    return """# K_gpcca Supernode Strategy

## Position

The supernode route is a computational fallback, not the biological definition of K_gpcca.

## When To Use

- Use only if full-resolution pilot construction or standard pyGPCCA is infeasible.
- Use after a bounded full-resolution subset has established the kernel design is coherent.

## Requirements

- Preserve time point.
- Preserve niche state.
- Preserve neighborhood structure.
- Preserve cross-time directional evidence.
- Compare multiple clustering/coarse-graining methods if feasible.
- Run standard pyGPCCA on the coarse-grained K_gpcca.
- Report supernode sensitivity across resolutions and clustering methods.

## Rejection Criteria

Reject supernode outputs if coarse-graining collapses slice/mouse groups, erases plausible branch structure, or produces macrostates inconsistent with full-resolution subset behavior.
"""


def build_pygpcca_policy_report() -> str:
    return """# K_gpcca pyGPCCA Execution Policy

## Formal GPCCA Backend Policy

Formal GPCCA outputs must use one of:

- standard pyGPCCA
- CellRank-compatible GPCCA estimator

## Disallowed As Formal Results

- custom Schur decomposition replacement
- custom GPCCA-like macrostate code
- heuristic macrostate fallback as a final result

## Failure Policy

If pyGPCCA fails:

- report the failure directly
- inspect K_gpcca diagnostics
- revise kernel construction or scope
- do not silently replace the failed run with a heuristic result

## Execution Gate

K_gpcca-00 does not execute pyGPCCA or CellRank. Standard pyGPCCA execution may only occur in a later approved stage after kernel QC passes.
"""


def build_acceptance_report() -> str:
    return """# K_gpcca Acceptance Criteria

## Kernel QC

- Row-sum max error <= 1e-5.
- NaN, inf, and negative entries all zero.
- Duplicate coordinates zero after sparse assembly.
- Acceptable weak-component/connectivity structure.
- No non-final zero-outgoing artifacts.
- Self-loop mass does not dominate.

## Standard GPCCA QC

- Standard pyGPCCA or CellRank-compatible GPCCA runs without non-standard fallback.
- Macrostate membership entropy is reasonable.
- Terminal, initial, and intermediate classifications are biologically interpretable.
- GPCCA fate probabilities are finite and row-normalized.

## Biological and Artifact QC

- No strong slice/mouse collapse.
- Macrostates map to cell type, neighborhood, and endpoint annotations.
- Spatial coherence is plausible.
- Results are stable across a small alpha/beta/gamma sensitivity grid.
- Results are not merely identical to P_fate endpoint propagation.

## P_fate Comparison

Compare K_gpcca terminal macrostates to M4E endpoints; compare GPCCA fate probabilities to M4C-v1/v2 fate probabilities; compute dominant terminal agreement, fate entropy/plasticity differences, spatial coherence, endpoint plausibility, and neighborhood/slice/mouse artifact checks.
"""


def build_barcode_report() -> str:
    return """# K_gpcca Barcode Extension Contract

## Position

Barcode evidence enters K_gpcca later, after official/lab-standard DARLIN preprocessing. K_gpcca-00 does not preprocess barcode raw reads and does not construct barcode-aware kernels.

## Required Future Barcode Inputs

- `clone_id`
- `barcode_id`
- `cell_id` or anchor key compatible with M1/M2/M4A node mapping
- `time`
- `slice_id`
- optional `mouse_id`
- optional read/support/confidence columns
- optional clone membership weights

## Derived Future Tables

- clone-by-niche composition table
- clone-supported transition evidence
- lineage consistency metrics
- pseudo-only versus barcode-aware comparison tables

## Kernel Extension

Barcode evidence can enter as `P_barcode` or `G_barcode`, yielding:

`K_gpcca_barcode = row_normalize(alpha * K_within_time + beta * P_cross_time + delta * P_barcode_or_G_barcode + gamma * I_self)`

Barcode-aware K_gpcca must be benchmarked against pseudo-v1 and pseudo-v2. If barcode/hybrid only improves over v1 but not v2, barcode contribution is weaker; if it improves over v2 on clone consistency and biological fate metrics, barcode adds evidence beyond generic pseudo-transition sharpening.
"""


def build_risk_report(risks: pd.DataFrame) -> str:
    return f"""# K_gpcca Risk Register

{dataframe_to_markdown(risks)}

## Risk Policy

Any high-severity risk triggered during pilot execution blocks full-scale K_gpcca execution until mitigation and revalidation are complete.
"""


def build_summary(
    context: dict[str, Any],
    grid: pd.DataFrame,
    checklist: pd.DataFrame,
    outputs: pd.DataFrame,
    safety: dict[str, Any],
    output_root: Path,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "stage": "K_gpcca-00",
        "status": "PASSED"
        if safety["upstream_metadata_diff_count"] == 0
        and safety["forbidden_downstream_diff_count"] == 0
        and safety["ssd_output_count"] == 0
        else "REVIEW",
        "generated_at_utc": utc_now(),
        "runtime_seconds": runtime_seconds,
        "output_root": output_root,
        "reports_dir": output_root / "reports",
        "design_only": True,
        "k_gpcca_constructed": False,
        "pygpcca_executed": False,
        "cellrank_executed": False,
        "p_fate_freeze_decision": context.get("plan_a", {}).get(
            "p_fate_freeze_decision",
            "keep_v1_and_v2_as_complementary_p_fate_branch",
        ),
        "proposed_kernel": "K_gpcca = row_normalize(alpha*K_within_time + beta*P_cross_time + gamma*I_self)",
        "proposed_barcode_kernel": "K_gpcca_barcode = row_normalize(alpha*K_within_time + beta*P_cross_time + delta*P_barcode_or_G_barcode + gamma*I_self)",
        "candidate_parameter_grid_rows": int(len(grid)),
        "candidate_gamma_values": sorted(float(value) for value in grid["gamma"].unique()),
        "candidate_cross_time_sources": sorted(str(value) for value in grid["cross_time_source"].unique()),
        "planned_future_output_rows": int(len(outputs)),
        "design_check_count": int(len(checklist)),
        "design_check_fail_count": int((checklist["status"] != "PASS").sum()),
        "pilot_scope": "D9/D21/D35 or D21/D35 plus within-time layers; deterministic 20k-100k node subset.",
        "full_vs_supernode_strategy": "Full-resolution subset first; supernode route only if full-resolution is infeasible.",
        "pygpcca_only_policy": "Formal GPCCA outputs require standard pyGPCCA or CellRank-compatible GPCCA; no heuristic fallback as final result.",
        "barcode_extension_positioning": "Future-only after official/lab-standard DARLIN preprocessing; consume processed clone/barcode tables as P_barcode or G_barcode.",
        "upstream_metadata_diff_count": safety["upstream_metadata_diff_count"],
        "upstream_metadata_diffs": safety["upstream_metadata_diffs"],
        "forbidden_downstream_diff_count": safety["forbidden_downstream_diff_count"],
        "forbidden_downstream_diffs": safety["forbidden_downstream_diffs"],
        "ssd_output_count": safety["ssd_output_count"],
        "next_recommended_step": NEXT_STEP,
    }


def validate_generated_outputs(paths: dict[str, Path]) -> dict[str, Any]:
    required = [paths[name] for name in REPORT_NAMES + CSV_NAMES] + [paths["summary"]]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.stat().st_size == 0]
    return {
        "required_output_count": len(required),
        "missing_required_outputs": missing,
        "empty_required_outputs": empty,
    }


def run(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    start = time.perf_counter()
    paths = output_paths(output_root)
    output_root = paths["root"]

    protected_before = snapshot(PROTECTED_ROOTS)
    forbidden_before = snapshot(FORBIDDEN_EXECUTION_ROOTS)

    ensure_dirs(paths)
    context = load_context()
    grid = build_candidate_parameter_grid()
    checklist = build_design_checklist()
    outputs = build_planned_output_inventory()
    input_contract = build_input_contract_rows()
    risks = build_risk_register()

    protected_after = snapshot(PROTECTED_ROOTS)
    forbidden_after = snapshot(FORBIDDEN_EXECUTION_ROOTS)
    safety = {
        "upstream_metadata_diffs": diff_snapshot(protected_before, protected_after),
        "forbidden_downstream_diffs": diff_snapshot(forbidden_before, forbidden_after),
        "ssd_output_count": count_ssd_outputs(output_root),
    }
    safety["upstream_metadata_diff_count"] = len(safety["upstream_metadata_diffs"])
    safety["forbidden_downstream_diff_count"] = len(safety["forbidden_downstream_diffs"])

    reports = {
        "k_gpcca_design_overview.md": build_design_overview(context),
        "p_fate_vs_k_gpcca_design_distinction.md": build_distinction_report(),
        "k_gpcca_kernel_mathematical_spec.md": build_math_spec_report(grid),
        "k_gpcca_input_contract.md": build_input_contract_report(input_contract),
        "k_gpcca_output_contract.md": build_output_contract_report(outputs),
        "k_gpcca_pilot_protocol.md": build_pilot_protocol_report(grid),
        "k_gpcca_supernode_strategy.md": build_supernode_strategy_report(),
        "k_gpcca_pyGPCCA_execution_policy.md": build_pygpcca_policy_report(),
        "k_gpcca_acceptance_criteria.md": build_acceptance_report(),
        "k_gpcca_barcode_extension_contract.md": build_barcode_report(),
        "k_gpcca_risk_register.md": build_risk_report(risks),
    }
    for name, body in reports.items():
        atomic_write_text(paths[name], body)

    atomic_write_csv(paths["k_gpcca_design_checklist.csv"], checklist)
    atomic_write_csv(paths["k_gpcca_candidate_parameter_grid.csv"], grid)
    atomic_write_csv(paths["k_gpcca_planned_output_inventory.csv"], outputs)

    runtime_seconds = time.perf_counter() - start
    summary = build_summary(
        context,
        grid,
        checklist,
        outputs,
        safety,
        output_root,
        runtime_seconds,
    )
    atomic_write_json(paths["summary"], summary)
    summary["output_validation"] = validate_generated_outputs(paths)
    atomic_write_json(paths["summary"], summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args.output_root)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_root": summary["output_root"],
                "candidate_parameter_grid_rows": summary["candidate_parameter_grid_rows"],
                "upstream_metadata_diff_count": summary["upstream_metadata_diff_count"],
                "forbidden_downstream_diff_count": summary["forbidden_downstream_diff_count"],
                "ssd_output_count": summary["ssd_output_count"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
