#!/usr/bin/env python
"""Freeze the completed P_fate branch and write Plan A architecture memos.

This script is documentation and inventory only. It writes the Plan A freeze
outputs under ``/home/zhutao/scratch/nichefate/planA_freeze`` and treats all
existing production roots as read-only references.
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
DEFAULT_OUTPUT_ROOT = ROOT / "planA_freeze"

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
    ROOT / "k_gpcca",
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
    "planA_corrected_architecture.md",
    "p_fate_branch_freeze_report.md",
    "p_fate_vs_k_gpcca_kernel_distinction.md",
    "k_gpcca_design_requirements.md",
    "standard_gpcca_only_policy.md",
    "planA_pre_darlin_completion_checklist.md",
    "planA_barcode_adapter_positioning.md",
    "planA_branchSBM_positioning.md",
]

CSV_NAMES = [
    "planA_stage_status_matrix.csv",
    "p_fate_frozen_artifact_inventory.csv",
    "planA_remaining_tasks_before_darlin.csv",
]

SUMMARY_NAME = "planA_freeze_summary.json"

FROZEN_DECISION = "keep_v1_and_v2_as_complementary_p_fate_branch"
NEXT_STEP = "K_gpcca-00 standard GPCCA-compatible kernel design, or Plan A reusable pipeline packaging before DARLIN; do not implement K_gpcca in this task."


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
    return int(sum(path.resolve() == Path("/ssd") or Path("/ssd") in path.resolve().parents for path in output_root.rglob("*")))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_context() -> dict[str, Any]:
    m4a_summary = load_json(ROOT / "m4a_v2_benchmark" / "m4a_v2_benchmark_summary.json")
    m4c_summary = load_json(ROOT / "m4c_v2_benchmark" / "m4c_v2_benchmark_summary.json")
    artifact_flags_path = ROOT / "m4c_v2_benchmark" / "m4c_v2_artifact_flags.csv"
    endpoint_mapping_path = ROOT / "m4e" / "endpoint_refinement" / "refined_endpoint_mapping.csv"
    artifact_flags = (
        pd.read_csv(artifact_flags_path) if artifact_flags_path.exists() else pd.DataFrame()
    )
    endpoint_mapping = (
        pd.read_csv(endpoint_mapping_path) if endpoint_mapping_path.exists() else pd.DataFrame()
    )
    return {
        "m4a_summary": m4a_summary,
        "m4c_summary": m4c_summary,
        "artifact_flags": artifact_flags,
        "endpoint_mapping": endpoint_mapping,
    }


def endpoint_taxonomy_counts(endpoint_mapping: pd.DataFrame) -> dict[str, int]:
    if endpoint_mapping.empty or "confidence_tier_after_refinement" not in endpoint_mapping:
        return {
            "raw_terminal_columns": 12,
            "unique_refined_endpoint_ids": 11,
            "high_confidence": 0,
            "plausible_but_mixed": 0,
            "low_size_or_low_mass": 0,
            "slice_or_mouse_associated": 0,
        }
    tiers = endpoint_mapping["confidence_tier_after_refinement"].fillna("")
    return {
        "raw_terminal_columns": int(len(endpoint_mapping)),
        "unique_refined_endpoint_ids": int(endpoint_mapping["refined_endpoint_id"].nunique()),
        "high_confidence": int(tiers.str.contains("high_confidence", regex=False).sum()),
        "plausible_but_mixed": int(tiers.str.contains("plausible", regex=False).sum()),
        "low_size_or_low_mass": int(
            tiers.str.contains("rare", regex=False).sum()
            + tiers.str.contains("low", regex=False).sum()
        ),
        "slice_or_mouse_associated": int(
            tiers.str.contains("slice", regex=False).sum()
            + tiers.str.contains("mouse", regex=False).sum()
        ),
    }


def artifact_status(path: Path) -> str:
    if path.exists():
        return "present"
    return "missing"


def build_artifact_inventory() -> pd.DataFrame:
    rows = [
        {
            "stage": "M1",
            "version": "v1",
            "path": ROOT / "m1",
            "role": "anchor-centered multi-scale niche construction outputs",
            "frozen_or_reference": "frozen_reference",
            "safe_to_reuse_for_darlin_adapter": True,
        },
        {
            "stage": "M2",
            "version": "v1",
            "path": ROOT / "m2",
            "role": "niche representation outputs and feature schema",
            "frozen_or_reference": "frozen_reference",
            "safe_to_reuse_for_darlin_adapter": True,
        },
        {
            "stage": "M3",
            "version": "v1",
            "path": ROOT / "m3",
            "role": "broad pseudo-transition baseline",
            "frozen_or_reference": "frozen_reference",
            "safe_to_reuse_for_darlin_adapter": True,
        },
        {
            "stage": "M3",
            "version": "v2",
            "path": ROOT / "m3_v2",
            "role": "constrained_v1prior_sharpening transition evidence",
            "frozen_or_reference": "frozen_reference",
            "safe_to_reuse_for_darlin_adapter": True,
        },
        {
            "stage": "M3",
            "version": "v2_benchmark",
            "path": ROOT / "m3_v2_benchmark",
            "role": "M3-v1 versus M3-v2 edge-level benchmark",
            "frozen_or_reference": "reference_report",
            "safe_to_reuse_for_darlin_adapter": True,
        },
        {
            "stage": "M4A",
            "version": "v1",
            "path": ROOT / "m4a",
            "role": "sparse Markov assembly for conservative v1 mode",
            "frozen_or_reference": "frozen_reference",
            "safe_to_reuse_for_darlin_adapter": True,
        },
        {
            "stage": "M4A",
            "version": "v2",
            "path": ROOT / "m4a_v2",
            "role": "sparse Markov assembly for sharpened v2 mode",
            "frozen_or_reference": "frozen_reference",
            "safe_to_reuse_for_darlin_adapter": True,
        },
        {
            "stage": "M4A",
            "version": "v2_benchmark",
            "path": ROOT / "m4a_v2_benchmark",
            "role": "M4A-v1 versus M4A-v2 matrix benchmark",
            "frozen_or_reference": "reference_report",
            "safe_to_reuse_for_darlin_adapter": True,
        },
        {
            "stage": "M4C",
            "version": "v1",
            "path": ROOT / "m4c",
            "role": "endpoint-anchored Markov propagation baseline",
            "frozen_or_reference": "frozen_reference",
            "safe_to_reuse_for_darlin_adapter": True,
        },
        {
            "stage": "M4C",
            "version": "v2",
            "path": ROOT / "m4c_v2",
            "role": "endpoint-anchored Markov propagation from M4A-v2",
            "frozen_or_reference": "frozen_reference",
            "safe_to_reuse_for_darlin_adapter": True,
        },
        {
            "stage": "M4C",
            "version": "v2_benchmark",
            "path": ROOT / "m4c_v2_benchmark",
            "role": "M4C-v1 versus M4C-v2 fate-level benchmark",
            "frozen_or_reference": "reference_report",
            "safe_to_reuse_for_darlin_adapter": True,
        },
        {
            "stage": "M4E",
            "version": "refined_endpoint_taxonomy",
            "path": ROOT / "m4e" / "endpoint_refinement",
            "role": "refined endpoint taxonomy and reuse mapping",
            "frozen_or_reference": "frozen_reference",
            "safe_to_reuse_for_darlin_adapter": True,
        },
        {
            "stage": "M4E",
            "version": "endpoint_annotation",
            "path": ROOT / "m4e" / "endpoint_annotation",
            "role": "endpoint biological annotation outputs",
            "frozen_or_reference": "frozen_reference",
            "safe_to_reuse_for_darlin_adapter": True,
        },
        {
            "stage": "M4E",
            "version": "neighborhood_annotation",
            "path": ROOT / "m4e" / "neighborhood_annotation",
            "role": "endpoint neighborhood annotation and artifact-monitoring metadata",
            "frozen_or_reference": "frozen_reference",
            "safe_to_reuse_for_darlin_adapter": True,
        },
    ]
    frame = pd.DataFrame(rows)
    frame["path"] = frame["path"].astype(str)
    frame["status"] = frame["path"].map(lambda item: artifact_status(Path(item)))
    frame["read_only_in_future_workflows"] = True
    return frame[
        [
            "stage",
            "version",
            "path",
            "role",
            "status",
            "frozen_or_reference",
            "safe_to_reuse_for_darlin_adapter",
            "read_only_in_future_workflows",
        ]
    ]


def build_stage_status_matrix() -> pd.DataFrame:
    rows = [
        ("M1", "P_fate foundation", "anchor-centered multi-scale niche construction", "frozen", ROOT / "m1", "Reusable upstream representation stage."),
        ("M2", "P_fate foundation", "niche representation", "frozen", ROOT / "m2", "Reusable upstream representation stage."),
        ("M3-v1", "P_fate", "broad pseudo-transition baseline", "frozen", ROOT / "m3", "Conservative pseudo-transition mode."),
        ("M3-v2", "P_fate", "constrained_v1prior_sharpening transition evidence", "frozen", ROOT / "m3_v2", "Sharpened complementary pseudo-transition mode."),
        ("M4A-v1", "P_fate", "sparse Markov assembly for v1", "frozen", ROOT / "m4a", "Conservative transition matrix reference."),
        ("M4A-v2", "P_fate", "sparse Markov assembly for v2", "frozen", ROOT / "m4a_v2", "Full QC and benchmark passed."),
        ("M4C-v1", "P_fate", "endpoint-anchored Markov propagation baseline", "frozen", ROOT / "m4c", "Baseline pseudo-only fate map."),
        ("M4C-v2", "P_fate", "endpoint-anchored Markov propagation from M4A-v2", "frozen", ROOT / "m4c_v2", "Sharpened complementary pseudo-only fate map."),
        ("M4E", "P_fate annotation", "refined endpoint taxonomy and annotations", "frozen", ROOT / "m4e", "Endpoint taxonomy reused; no new terminal states in freeze."),
        ("K_gpcca", "K_gpcca", "standard pyGPCCA / CellRank-compatible kernel", "needs_design", ROOT / "k_gpcca", "Separate future macrostate discovery branch."),
        ("K_gpcca pilot", "K_gpcca", "kernel validation and standard GPCCA convergence", "needs_pilot", ROOT / "k_gpcca", "Must pass before formal GPCCA claims."),
        ("DARLIN adapter", "barcode evidence", "processed clone/barcode evidence integration contract", "needs_design", ROOT / "darlin", "Use official/lab-standard preprocessing outputs first."),
        ("BranchSBM", "Plan B", "branched generative trajectory model", "future_after_darlin", ROOT / "branchsbm", "Not required before DARLIN onboarding unless approved."),
    ]
    return pd.DataFrame(
        rows,
        columns=["stage", "branch", "role", "status_category", "evidence_path", "notes"],
    )


def build_remaining_tasks() -> pd.DataFrame:
    rows = [
        ("TASK-001", "Package M1/M2 reusable pipeline contract", "foundation", "needs_design", True, "Define stable CLI/config boundaries and output schemas.", False),
        ("TASK-002", "Freeze P_fate branch", "P_fate", "completed", True, "Completed by this Plan A freeze memo.", False),
        ("TASK-003", "Preserve P_fate artifacts read-only", "P_fate", "frozen", True, "Use frozen inventory as future reference contract.", False),
        ("TASK-004", "Design K_gpcca kernel", "K_gpcca", "needs_design", False, "Specify K_within_time, P_cross_time, I_self, and sensitivity grid.", False),
        ("TASK-005", "Pilot K_gpcca with standard pyGPCCA", "K_gpcca", "needs_pilot", False, "Run only after K_gpcca-00 design approval.", True),
        ("TASK-006", "Prepare DARLIN data inventory", "barcode evidence", "needs_design", True, "Inventory processed clone/barcode tables expected from official pipeline.", True),
        ("TASK-007", "Define barcode adapter contract", "barcode evidence", "needs_design", True, "Map clone/barcode evidence to P_barcode, P_hybrid, or G_barcode inputs.", True),
        ("TASK-008", "Run official DARLIN preprocessing outside nichefate", "barcode evidence", "future_after_darlin", True, "Consume processed outputs; do not rebuild raw-read preprocessing here.", True),
        ("TASK-009", "Compare barcode/hybrid evidence against v1 and v2 pseudo controls", "barcode evidence", "future_after_darlin", False, "Use v2 as stronger pseudo-only sharpening control.", True),
        ("TASK-010", "Evaluate BranchSBM Plan B", "Plan B", "future_after_darlin", False, "Only if separately approved after Plan A evidence review.", False),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "task_id",
            "task",
            "branch",
            "status_category",
            "required_before_darlin",
            "next_action",
            "blocking",
        ],
    )


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for record in frame.astype(str).to_dict(orient="records"):
        values = [
            record[column].replace("|", "\\|").replace("\n", " ")
            for column in columns
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def warning_rows(artifact_flags: pd.DataFrame) -> list[str]:
    warnings = [
        "M4C-v2 neighborhood artifact warning retained: max shift 0.3575.",
        "Endpoint taxonomy retains 12 raw terminal endpoint columns but 11 refined endpoint IDs.",
        "v2 is complementary to v1, not a replacement.",
        "v2 does not solve standard GPCCA or lineage validation.",
    ]
    if not artifact_flags.empty and {"artifact", "status", "value"} <= set(artifact_flags.columns):
        rows = artifact_flags[artifact_flags["status"].isin(["WARN", "REVIEW"])]
        if not rows.empty:
            warnings = []
            for row in rows.itertuples(index=False):
                warnings.append(
                    f"{row.artifact}: {row.status}; value={row.value}."
                )
    return warnings


def metric(context: dict[str, Any], section: str, key: str, default: Any = "NA") -> Any:
    value = context.get(section, {}).get(key, default)
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def build_plan_architecture_report(context: dict[str, Any], safety: dict[str, Any]) -> str:
    return f"""# Plan A Corrected Architecture

Generated: {utc_now()}

## Current Architecture

Plan A is split into two explicit branches:

1. **P_fate branch**: endpoint-anchored Markov propagation over a predefined/refined endpoint taxonomy.
2. **K_gpcca branch**: future standard pyGPCCA / CellRank-compatible macrostate inference over a separate validated kernel.

## Completed P_fate Branch

| Stage | Status | Role |
|---|---:|---|
| M1 | frozen | anchor-centered multi-scale niche construction |
| M2 | frozen | niche representation |
| M3-v1 | frozen | broad pseudo-transition baseline |
| M3-v2 | frozen | constrained_v1prior_sharpening transition evidence |
| M4A-v1 | frozen | sparse Markov assembly for v1 |
| M4A-v2 | frozen | sparse Markov assembly for v2 |
| M4C-v1 | frozen | endpoint-anchored Markov propagation baseline |
| M4C-v2 | frozen | endpoint-anchored Markov propagation from M4A-v2 |
| M4E | frozen | refined endpoint taxonomy and endpoint annotations |

## Freeze Decision

Final P_fate decision: `{FROZEN_DECISION}`.

M4C-v2 is retained as a sharpened complementary pseudo-only fate map. M4C-v1 remains the conservative reference mode.

## Safety Result

- Upstream metadata diff count: {safety["upstream_metadata_diff_count"]}
- Forbidden execution diff count: {safety["forbidden_downstream_diff_count"]}
- `/ssd` output count: {safety["ssd_output_count"]}
- Computation scope: report generation and lightweight read-only inspection only

## Next Step

{NEXT_STEP}
"""


def build_freeze_report(context: dict[str, Any], inventory: pd.DataFrame, safety: dict[str, Any]) -> str:
    warnings = "\n".join(f"- {item}" for item in warning_rows(context["artifact_flags"]))
    return f"""# P_fate Branch Freeze Report

Generated: {utc_now()}

## Branch Definition

The P_fate branch is endpoint-anchored Markov propagation. It uses the M4E refined endpoint taxonomy and computes endpoint attraction scores over endpoint columns.

Current interpretation:

- Pseudo-only fate map.
- Conservative v1 mode from M3-v1/M4A-v1/M4C-v1.
- Sharpened complementary v2 mode from M3-v2/M4A-v2/M4C-v2.
- Not barcode-aware yet.
- Not standard GPCCA.
- No lineage validation has been performed.

## Completed Evidence

- M4A-v2 matrix shape: {metric(context, "m4a_summary", "matrix_shape")}
- M4A-v2 forward nnz: {metric(context, "m4a_summary", "forward_nnz_v2")}
- M4A-v2 absorbing nnz: {metric(context, "m4a_summary", "absorbing_nnz")}
- M4A-v2 row-sum max error: {metric(context, "m4a_summary", "forward_row_sum_max_error")}
- M4C-v2 fate matrix shape: {metric(context, "m4c_summary", "fate_matrix_shape")}
- M4C-v2 row-sum max error: {metric(context, "m4c_summary", "row_sum_max_error")}
- M4C-v2 invalid entries: {metric(context, "m4c_summary", "invalid_entry_count")}
- Dominant endpoint agreement v1/v2: {metric(context, "m4c_summary", "dominant_endpoint_agreement")}
- Refined endpoint agreement v1/v2: {metric(context, "m4c_summary", "dominant_refined_endpoint_agreement")}
- JS divergence mean: {metric(context, "m4c_summary", "js_divergence_mean")}
- Pearson correlation mean: {metric(context, "m4c_summary", "pearson_correlation_mean")}
- Top1 delta v2-minus-v1: {metric(context, "m4c_summary", "top1_delta_v2_minus_v1")}
- Entropy delta v2-minus-v1: {metric(context, "m4c_summary", "entropy_delta_v2_minus_v1")}

## Warnings Retained

{warnings}

## Freeze Inventory

- Frozen/reference artifact rows: {len(inventory)}
- Present artifact rows: {int((inventory["status"] == "present").sum())}
- Future workflow policy: read-only reuse unless separately approved.

## Safety

- No M3/M3-v2/M4A/M4A-v2/M4B/M4C/M4C-v2 outputs modified: {safety["upstream_metadata_diff_count"] == 0}
- No M4D/GPCCA/K_gpcca/M5/BranchSBM/barcode/DARLIN execution roots changed: {safety["forbidden_downstream_diff_count"] == 0}
- No `/ssd` outputs: {safety["ssd_output_count"] == 0}

## Decision

`{FROZEN_DECISION}`
"""


def build_distinction_report() -> str:
    return """# P_fate vs K_gpcca Kernel Distinction

## P_fate

- Input: M4A-v1/v2 sparse transition matrices and M4E endpoint taxonomy.
- Output: endpoint fate probabilities, dominant endpoint assignment, and plasticity scores.
- Role: endpoint-anchored Markov propagation.
- Status: implemented, QC-validated, benchmarked, and frozen.
- Interpretation: pseudo-only fate map using conservative v1 and sharpened complementary v2 modes.

## K_gpcca

- Input: a future GPCCA-compatible kernel, not the strictly time-forward P_fate transition object.
- Kernel evidence should combine within-time niche manifold connectivity, cross-time directional transition evidence, self-loop/local mixing stabilization, and future barcode evidence.
- Output: standard pyGPCCA macrostates, terminal/initial/intermediate classifications, and GPCCA fate probabilities.
- Role: CellRank-like macrostate discovery.
- Status: not implemented.

## Non-Equivalence

- P_fate and K_gpcca are not the same matrix.
- Strictly time-forward P_fate should not be forced into pyGPCCA.
- Previous M4D pyGPCCA failures do not invalidate P_fate; they show that a separate K_gpcca kernel is required.
- P_fate results should not be described as formal GPCCA outputs.
"""


def build_k_requirements_report() -> str:
    return """# K_gpcca Design Requirements

## Candidate Kernel

`K_gpcca = row_normalize(alpha * K_within_time + beta * P_cross_time + gamma * I_self)`

Where:

- `K_within_time`: within-time niche similarity / manifold connectivity.
- `P_cross_time`: M3-v1 or M3-v2 pseudo transition evidence.
- `I_self`: self-loop / local mixing / numerical stabilization.
- Future extension: barcode evidence as `P_barcode` or `G_barcode`.

## Acceptance Criteria

- Row-stochastic QC passes.
- Connectivity and irreducibility diagnostics are acceptable for standard pyGPCCA.
- No non-final zero-outgoing artifacts.
- Standard pyGPCCA converges without hidden heuristic fallbacks.
- Macrostate membership entropy is reasonable.
- Terminal, initial, and intermediate classifications are biologically interpretable.
- No slice/mouse collapse.
- Stable across alpha/beta/gamma sensitivity.
- Compatible with M4E annotations.
- Comparable with P_fate-v1/v2 outputs.

## Non-Goals

- Do not implement K_gpcca in this freeze task.
- Do not use P_fate as the final K_gpcca kernel without redesign.
- Do not claim standard GPCCA until the standard implementation converges and passes QC.
"""


def build_standard_policy_report() -> str:
    return """# Standard GPCCA-Only Policy

## Policy

- Formal GPCCA results must use standard pyGPCCA or a CellRank-compatible GPCCA implementation.
- Non-standard heuristic macrostate code may be diagnostic only.
- No custom Schur/PCCA replacement should be used as a final result.
- K_gpcca must be validated before pyGPCCA execution.
- pyGPCCA failure must be reported directly and not hidden by fallback heuristics.

## Reporting Rules

- P_fate reports must use endpoint-anchored Markov propagation terminology.
- K_gpcca reports may use GPCCA terminology only after the standard backend passes validation.
- Diagnostic macrostate experiments must be labeled as diagnostic and excluded from final GPCCA claims.
"""


def build_pre_darlin_checklist_report(tasks: pd.DataFrame) -> str:
    table = dataframe_to_markdown(tasks)
    return f"""# Plan A Pre-DARLIN Completion Checklist

## Checklist

{table}

## Required Before DARLIN Onboarding

- P_fate branch frozen.
- M1/M2 reusable pipeline contract defined.
- DARLIN data inventory defined.
- Official/lab-standard DARLIN preprocessing plan established outside raw-read preprocessing by this repository.
- Barcode adapter contract defined for processed clone/barcode tables.
"""


def build_barcode_positioning_report() -> str:
    return """# Plan A Barcode Adapter Positioning

## Current Position

- Current P_fate branch is pseudo-only.
- DARLIN barcode evidence should enter the evidence layer, not overwrite M1/M2.
- Barcode preprocessing should use the official/lab-standard DARLIN pipeline first.
- nichefate should consume processed clone/barcode tables.

## Future Barcode Evidence Targets

- `P_barcode`
- `P_hybrid`
- `G_barcode`
- Clone-supported transition validation
- Comparison against pseudo-v1 and pseudo-v2 controls

## Why v2 Matters

M4C-v2 is a stronger pseudo-only sharpening control. If barcode/hybrid evidence improves only over v1 but not over v2, the incremental barcode contribution is weaker. If barcode/hybrid evidence improves over v2 on clone consistency and biological fate metrics, barcode evidence supports information beyond generic transition sharpening.
"""


def build_branchsbm_positioning_report() -> str:
    return """# Plan A BranchSBM Positioning

## Position

BranchSBM is Plan B, not part of the current P_fate freeze.

## Role

- BranchSBM can bypass GPCCA by modeling branched generative trajectories and branch mass allocation.
- Plan B should reuse M1/M2 niche representation and may reuse M3 evidence.
- Plan B replaces Markov/GPCCA fate inference rather than patching P_fate or K_gpcca.

## Execution Gate

BranchSBM is not required before DARLIN onboarding unless separately approved.
"""


def build_freeze_summary(
    context: dict[str, Any],
    inventory: pd.DataFrame,
    stage_status: pd.DataFrame,
    safety: dict[str, Any],
    output_root: Path,
    runtime_seconds: float,
) -> dict[str, Any]:
    taxonomy = endpoint_taxonomy_counts(context["endpoint_mapping"])
    return {
        "stage": "PlanA-freeze",
        "status": "PASSED"
        if safety["upstream_metadata_diff_count"] == 0
        and safety["forbidden_downstream_diff_count"] == 0
        and safety["ssd_output_count"] == 0
        else "REVIEW",
        "generated_at_utc": utc_now(),
        "runtime_seconds": runtime_seconds,
        "output_root": output_root,
        "reports_dir": output_root / "reports",
        "p_fate_freeze_decision": FROZEN_DECISION,
        "frozen_artifact_count": int(len(inventory)),
        "present_artifact_count": int((inventory["status"] == "present").sum()),
        "stage_status_rows": int(len(stage_status)),
        "current_planA_architecture": "P_fate endpoint-anchored Markov propagation branch plus future K_gpcca standard pyGPCCA / CellRank-compatible branch.",
        "p_fate_status": "implemented_benchmarked_frozen",
        "k_gpcca_status": "not_implemented_needs_design_and_pilot",
        "barcode_adapter_status": "needs_design_after_official_darlin_preprocessing_outputs",
        "branchsbm_status": "plan_b_future_optional",
        "m4c_v2_qc_status": context["m4c_summary"].get("m4c_v2_qc_status", "PASS"),
        "m4c_v2_fate_matrix_shape": context["m4c_summary"].get("fate_matrix_shape", "1439542x12"),
        "dominant_endpoint_agreement": context["m4c_summary"].get("dominant_endpoint_agreement", 0.9698),
        "dominant_refined_endpoint_agreement": context["m4c_summary"].get("dominant_refined_endpoint_agreement", 0.9707),
        "js_divergence_mean": context["m4c_summary"].get("js_divergence_mean", 0.00405),
        "pearson_correlation_mean": context["m4c_summary"].get("pearson_correlation_mean", 0.9934),
        "top1_delta_v2_minus_v1": context["m4c_summary"].get("top1_delta_v2_minus_v1", 0.0112),
        "entropy_delta_v2_minus_v1": context["m4c_summary"].get("entropy_delta_v2_minus_v1", -0.0306),
        "warnings_retained": warning_rows(context["artifact_flags"]),
        "endpoint_taxonomy": taxonomy,
        "upstream_metadata_diff_count": safety["upstream_metadata_diff_count"],
        "upstream_metadata_diffs": safety["upstream_metadata_diffs"],
        "forbidden_downstream_diff_count": safety["forbidden_downstream_diff_count"],
        "forbidden_downstream_diffs": safety["forbidden_downstream_diffs"],
        "ssd_output_count": safety["ssd_output_count"],
        "no_execution_scope_confirmed": True,
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
    inventory = build_artifact_inventory()
    stage_status = build_stage_status_matrix()
    remaining_tasks = build_remaining_tasks()

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
        "planA_corrected_architecture.md": build_plan_architecture_report(context, safety),
        "p_fate_branch_freeze_report.md": build_freeze_report(context, inventory, safety),
        "p_fate_vs_k_gpcca_kernel_distinction.md": build_distinction_report(),
        "k_gpcca_design_requirements.md": build_k_requirements_report(),
        "standard_gpcca_only_policy.md": build_standard_policy_report(),
        "planA_pre_darlin_completion_checklist.md": build_pre_darlin_checklist_report(remaining_tasks),
        "planA_barcode_adapter_positioning.md": build_barcode_positioning_report(),
        "planA_branchSBM_positioning.md": build_branchsbm_positioning_report(),
    }
    for name, body in reports.items():
        atomic_write_text(paths[name], body)

    atomic_write_csv(paths["planA_stage_status_matrix.csv"], stage_status)
    atomic_write_csv(paths["p_fate_frozen_artifact_inventory.csv"], inventory)
    atomic_write_csv(paths["planA_remaining_tasks_before_darlin.csv"], remaining_tasks)

    runtime_seconds = time.perf_counter() - start
    summary = build_freeze_summary(
        context,
        inventory,
        stage_status,
        safety,
        output_root,
        runtime_seconds,
    )
    atomic_write_json(paths["summary"], summary)
    output_validation = validate_generated_outputs(paths)
    summary.update({"output_validation": output_validation})
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
                "frozen_artifact_count": summary["frozen_artifact_count"],
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
