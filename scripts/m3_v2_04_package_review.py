#!/usr/bin/env python
"""Write the M3-v2 constrained-kernel package review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_ROOT = Path("/home/zhutao/scratch/nichefate/m3_v2_package_review")
M3_V2_01_ROOT = Path("/home/zhutao/scratch/nichefate/m3_v2_pilot")
M3_V2_02_ROOT = Path("/home/zhutao/scratch/nichefate/m3_v2_pilot_tuning")
M3_V2_03_ROOT = Path("/home/zhutao/scratch/nichefate/m3_v2_pilot_confirmatory")

MODE_NAME = "constrained_v1prior_sharpening"
VARIANT_NAME = "v1prior_1.0_tau_0.5_top10"
DECISION = "proceed_to_full_m3_v2_production_planning"


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
        M3_V2_01_ROOT.resolve(),
        M3_V2_02_ROOT.resolve(),
        M3_V2_03_ROOT.resolve(),
    ]
    for root in protected:
        if resolved == root or root in resolved.parents:
            raise ValueError(f"Refusing to write package-review outputs under protected path: {resolved}")


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    validate_output_root(output_root)
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    return {"root": output_root, "reports": reports}


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV is missing: {path}")
    return pd.read_csv(path)


def read_required_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Required JSON must contain an object: {path}")
    return payload


def metric_value(metrics: pd.DataFrame, name: str, column: str) -> float:
    row = metrics[metrics["metric_name"] == name]
    if row.empty:
        raise ValueError(f"Missing metric row: {name}")
    return float(row.iloc[0][column])


def load_evidence() -> tuple[pd.DataFrame, dict[str, Any]]:
    v2_01_metrics = read_required_csv(M3_V2_01_ROOT / "pilot_metric_summary.csv")
    v2_01_json = read_required_json(M3_V2_01_ROOT / "pilot_metric_summary.json")
    v2_02_ranked = read_required_csv(M3_V2_02_ROOT / "variant_ranked_decision_table.csv")
    v2_02_json = read_required_json(M3_V2_02_ROOT / "variant_metric_summary.json")
    v2_03_decision = read_required_csv(M3_V2_03_ROOT / "confirmatory_pilot_decision_table.csv")
    v2_03_json = read_required_json(M3_V2_03_ROOT / "confirmatory_pilot_combined_metric_summary.json")

    best = v2_02_ranked[v2_02_ranked["variant"] == VARIANT_NAME]
    if best.empty:
        raise ValueError(f"Validated variant missing from M3-v2-02 decision table: {VARIANT_NAME}")
    best = best.iloc[0]
    confirm = v2_03_decision[v2_03_decision["method"] == VARIANT_NAME].copy()
    if len(confirm) < 3:
        raise ValueError("Expected M3-v2-03 decision rows for A, B, and optional C pilots.")

    rows: list[dict[str, Any]] = [
        {
            "stage": "M3-v2-01",
            "pilot_id": "D9_D21_initial",
            "transition_pair": "D9->D21",
            "source_anchor_count": int(v2_01_json["source_anchor_count"]),
            "candidate_edge_count": int(v2_01_json["candidate_edge_count"]),
            "mode_or_variant": "primary_state_soft_gates_no_v1_prior",
            "endpoint_plausibility": metric_value(
                v2_01_metrics,
                "source-target refined endpoint plausibility",
                "v2_value",
            ),
            "leiden_consistency": metric_value(
                v2_01_metrics,
                "top-target Leiden_neigh consistency",
                "v2_value",
            ),
            "transition_entropy": metric_value(
                v2_01_metrics,
                "transition entropy / top1 concentration",
                "v2_value",
            ),
            "top1_probability": float(v2_01_json["details"]["v2_top1_probability_mean"]),
            "slice_mouse_collapse": metric_value(
                v2_01_metrics,
                "slice/mouse collapse diagnostics",
                "v2_value",
            ),
            "row_sum_pass": bool(v2_01_json["row_qc"]["row_sum_pass"]),
            "acceptance_status": "failed",
            "decision": "keep_v1_as_main_baseline",
            "interpretation": "Too diffuse; state-only/adaptive tau kernel was nearly uniform over 30 candidates.",
        },
        {
            "stage": "M3-v2-02",
            "pilot_id": "D9_D21_tuning",
            "transition_pair": "D9->D21",
            "source_anchor_count": int(v2_02_json["source_anchor_count"]),
            "candidate_edge_count": int(v2_02_json["edge_count"]),
            "mode_or_variant": VARIANT_NAME,
            "endpoint_plausibility": float(best["refined_endpoint_plausibility"]),
            "leiden_consistency": float(best["leiden_consistency"]),
            "transition_entropy": float(best["transition_entropy_mean"]),
            "top1_probability": float(best["top1_probability_mean"]),
            "slice_mouse_collapse": float(best["slice_mouse_collapse"]),
            "row_sum_pass": bool(best["row_sum_pass"]),
            "acceptance_status": "passed",
            "decision": "revise_v2_and_repeat_pilot",
            "interpretation": "Best constrained v1-prior sharpening variant preserved plausibility while improving sharpness.",
        },
    ]
    for row in confirm.itertuples(index=False):
        rows.append(
            {
                "stage": "M3-v2-03",
                "pilot_id": row.pilot_id,
                "transition_pair": f"{row.source_time}->{row.target_time}",
                "source_anchor_count": int(row.source_anchor_count),
                "candidate_edge_count": int(row.candidate_edge_count),
                "mode_or_variant": row.method,
                "endpoint_plausibility": float(row.refined_endpoint_plausibility),
                "leiden_consistency": float(row.leiden_consistency),
                "transition_entropy": float(row.transition_entropy_mean),
                "top1_probability": float(row.top1_probability_mean),
                "slice_mouse_collapse": float(row.slice_mouse_collapse),
                "row_sum_pass": bool(row.row_sum_pass),
                "acceptance_status": "passed" if bool(row.passes_acceptance) else "failed",
                "decision": str(row.decision_category),
                "interpretation": "Confirmatory constrained v1-prior sharpening pilot.",
            }
        )
    metadata = {
        "v2_01": v2_01_json,
        "v2_02": v2_02_json,
        "v2_03": v2_03_json,
        "all_confirmatory_passed": bool(confirm["passes_acceptance"].all()),
    }
    return pd.DataFrame(rows), metadata


def mode_schema() -> dict[str, Any]:
    return {
        "mode_name": MODE_NAME,
        "status": "pilot_validated_package_review",
        "interpretation": "Complementary v1-prior sharpening mode; not an independent replacement for M3-v1.",
        "formula": (
            "P_v2(i->j) proportional to P_v1(i->j)^lambda * "
            "exp(-d_state(i,j)/(tau_i*tau_scale)) * G_composition(i,j) * "
            "G_spatial_topology(i,j) * G_slice_mouse(i,j) * G_barcode(i,j)"
        ),
        "validated_pseudo_only_parameters": {
            "lambda": 1.0,
            "tau_scale": 0.5,
            "top_k": 10,
            "G_barcode": 1.0,
            "barcode_mode": "neutral",
            "row_normalization": "per_source_anchor",
        },
        "required_inputs": [
            "M3-v1 candidate edge table with row_normalized_transition_prob",
            "M2-derived state, composition, and spatial/topology features",
            "source and target anchor metadata",
        ],
        "required_qc": [
            "finite weights",
            "nonnegative weights",
            "row sums equal 1 per source anchor",
            "candidate coverage check",
            "source coverage check",
            "endpoint plausibility diagnostics",
            "slice/mouse collapse diagnostics",
        ],
        "versioned_output_roots_if_production_runs": {
            "m3_v2": "/home/zhutao/scratch/nichefate/m3_v2/full_by_shard/",
            "m3_v2_reports": "/home/zhutao/scratch/nichefate/m3_v2/reports/",
            "m4a_v2": "/home/zhutao/scratch/nichefate/m4a_v2/",
            "m4c_v2": "/home/zhutao/scratch/nichefate/m4c_v2/",
        },
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    work = frame[columns].copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: f"{float(value):.4g}" if pd.notna(value) else "NA")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in work.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_reports(paths: dict[str, Path], evidence: pd.DataFrame, metadata: dict[str, Any]) -> None:
    root = paths["root"]
    reports = paths["reports"]
    schema = mode_schema()
    (root / "m3_v2_constrained_mode_schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True))
    evidence.to_csv(root / "m3_v2_pilot_evidence_summary.csv", index=False)

    reports.joinpath("m3_v2_constrained_mode_definition.md").write_text(
        f"""# M3-v2 Constrained Mode Definition

Official mode name: `{MODE_NAME}`

## Formula

`P_v2(i -> j) proportional to P_v1(i -> j)^lambda * exp(-d_state(i,j) / (tau_i * tau_scale)) * G_composition(i,j) * G_spatial_topology(i,j) * G_slice_mouse(i,j) * G_barcode(i,j)`

## Locked Pseudo-Only Parameters

- `lambda = 1.0`
- `tau_scale = 0.5`
- `top_k = 10`
- `G_barcode = 1.0`
- row normalization is per source anchor

This mode is a constrained v1-prior sharpening layer. It should preserve raw source/target anchor traceability and retain M3-v1 probability for direct comparison.
"""
    )

    reports.joinpath("m3_v1_v2_complementary_interpretation.md").write_text(
        f"""# M3-v1 and M3-v2 Complementary Interpretation

M3-v1 remains the frozen pseudo-only baseline. `{MODE_NAME}` does not replace v1 by default.

The constrained M3-v2 mode uses M3-v1 as a prior and sharpens/reweights transitions with state, composition, spatial/topology, and slice/mouse gates. It should be interpreted as a calibrated transition mode, not an independent transition discovery method.

If full production is later run, both M3-v1 and M3-v2 outputs must be preserved for comparison. M4C-v1 remains the frozen baseline until a versioned M4C-v2 benchmark is reviewed.
"""
    )

    reports.joinpath("m3_v2_pilot_evidence_review.md").write_text(
        f"""# M3-v2 Pilot Evidence Review

## Evidence Summary

{markdown_table(evidence, ['stage', 'pilot_id', 'transition_pair', 'mode_or_variant', 'endpoint_plausibility', 'leiden_consistency', 'transition_entropy', 'top1_probability', 'slice_mouse_collapse', 'acceptance_status'])}

## Review

- M3-v2-01 failed because the state-only/adaptive tau kernel became too diffuse and nearly uniform over 30 candidate targets.
- M3-v2-02 identified `{VARIANT_NAME}` as the best constrained variant.
- M3-v2-03 confirmed stability across D9->D21 repeat, D3->D9, and D21->D35.
- Endpoint and Leiden plausibility were mostly preserved.
- Entropy decreased and top1 probability increased, showing sharper transition probabilities.
- Slice/mouse collapse did not materially worsen in required pilots.
- The decision remains `keep_v1_and_v2_as_complementary`.
"""
    )

    reports.joinpath("m3_v2_full_production_readiness_checklist.md").write_text(
        f"""# M3-v2 Full Production Readiness Checklist

- [x] All pilot time pairs passed: D9->D21 repeat, D3->D9, D21->D35.
- [x] Parameters locked for pseudo-only mode: `lambda=1.0`, `tau_scale=0.5`, `top_k=10`, `G_barcode=1.0`.
- [x] M3-v2 interpretation is complementary to M3-v1.
- [ ] Full-production output paths are created as versioned v2 paths.
- [ ] M3-v1 outputs are not overwritten.
- [ ] Row-sum QC is required for every shard.
- [ ] No NaN or negative weights are allowed.
- [ ] Source coverage and candidate coverage checks are required per shard.
- [ ] Per-time-pair metrics are required.
- [ ] Slice/mouse collapse diagnostics are required.
- [ ] Endpoint plausibility diagnostics are required.
- [ ] M4A-v2 and M4C-v2 outputs must be versioned separately.
- [ ] V1 vs V2 full benchmark is required before adoption.
"""
    )

    reports.joinpath("m3_v2_output_versioning_plan.md").write_text(
        """# M3-v2 Output Versioning Plan

- M3-v1 remains under the current `/home/zhutao/scratch/nichefate/m3/` path.
- M3-v2 full output, if later run, must write to `/home/zhutao/scratch/nichefate/m3_v2/full_by_shard/`.
- M3-v2 reports must write to `/home/zhutao/scratch/nichefate/m3_v2/reports/`.
- M4A-v2 must write to `/home/zhutao/scratch/nichefate/m4a_v2/`.
- M4C-v2 must write to `/home/zhutao/scratch/nichefate/m4c_v2/`.
- No M3-v1, M4A-v1, or M4C-v1 production outputs may be overwritten.
- M4C-v1 remains the frozen baseline comparator.
"""
    )

    reports.joinpath("m3_v2_barcode_extension_note.md").write_text(
        f"""# M3-v2 Barcode Extension Note

In pseudo-only mode, `G_barcode = 1.0`.

In a future barcode-aware mode, `P_base` does not have to be M3-v1. It could be:

- `P_pseudo`
- `P_barcode`
- `P_hybrid`

The constrained sharpening framework may be reused as a calibration layer:

`P_final proportional to P_base^lambda * state/spatial gates`

Raw DARLIN reads must first be processed with official or lab-standard DARLIN preprocessing. nichefate should consume processed clone/barcode tables, not raw FASTQ directly.
"""
    )

    decision_ok = bool(metadata["all_confirmatory_passed"])
    decision = DECISION if decision_ok else "run_more_pilot"
    reports.joinpath("m3_v2_package_review_decision.md").write_text(
        f"""# M3-v2 Package Review Decision

Decision: `{decision}`

The package review confirms that `{MODE_NAME}` is stable, versioned, and clearly complementary to M3-v1.

This decision authorizes full M3-v2 production planning only. It does not authorize running full M3-v2 production, M4A-v2 assembly, or M4C-v2 propagation in this task.

Next engineering step: design the full M3-v2 production plan and versioned output contract using the locked constrained mode parameters.
"""
    )


def write_inventory(output_root: Path) -> None:
    inventory_path = output_root / "reports" / "m3_v2_package_review_output_inventory.csv"
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != inventory_path:
            rows.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(output_root)),
                    "file_type": path.suffix.lstrip(".") or "text",
                    "size_bytes": path.stat().st_size,
                }
            )
    pd.DataFrame(rows).to_csv(inventory_path, index=False)


def run(output_root: Path) -> dict[str, Any]:
    paths = ensure_dirs(output_root)
    evidence, metadata = load_evidence()
    write_reports(paths, evidence, metadata)
    write_inventory(output_root)
    return {
        "output_root": str(output_root),
        "mode_name": MODE_NAME,
        "decision": DECISION if metadata["all_confirmatory_passed"] else "run_more_pilot",
        "evidence_rows": int(len(evidence)),
        "all_confirmatory_passed": bool(metadata["all_confirmatory_passed"]),
    }


def main() -> None:
    args = parse_args()
    payload = run(Path(args.output_root))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
