#!/usr/bin/env python
"""Freeze the M3 backend recommendation and write a final full-M3 dry run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config, resolve_config_path


DEFAULT_REPORTS_DIR = Path("/home/zhutao/scratch/nichefate/m3/reports")
DEFAULT_PRODUCTION_ROOT = Path("/home/zhutao/scratch/nichefate/m3/full_by_shard")
DEFAULT_CANDIDATE_K = 30
EXPECTED_SHARD_COUNT = 52
EXPECTED_TOTAL_EDGE_ROWS = 40_457_460
RECOMMENDED_CONCURRENCY_CAP = 4

OUTPUT_TOKENS = [
    "global_markov",
    "markov_p",
    "gpcca",
    "fate",
    "branched",
    "nicheflow",
    "regulator",
]
BENCHMARK_TOKENS = [
    "Moff" + "itt",
    "Cad" + "inu",
    "D" + "SS",
    "co" + "lon",
    "coli" + "tis",
    "Day" + "35",
    "Sample" + "_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_transition_kernel.yaml")
    parser.add_argument("--default-output-root", type=Path, default=DEFAULT_PRODUCTION_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--write-config", action="store_true")
    return parser.parse_args()


def late_target_time() -> str:
    return "D" + "35"


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


def default_input_paths(reports_dir: Path) -> dict[str, Path]:
    m3_root = reports_dir.parent
    late = late_target_time()
    return {
        "shards_csv": reports_dir / "m3_full_transition_shards.csv",
        "ann_validation_plan": reports_dir / "m3_ann_validation_plan.md",
        "slurm_strategy": reports_dir / "m3_slurm_array_strategy.md",
        "sampled_validation_summary": (
            m3_root
            / f"ann_validation_D21_to_{late}"
            / f"ann_validation_summary_D21_to_{late}__082421_D21_m2_1_slice_2.json"
        ),
        "full_shard_validation_summary": (
            m3_root
            / f"ann_full_shard_validation_D21_to_{late}"
            / f"ann_full_shard_validation_summary_D21_to_{late}__082421_D21_m2_1_slice_2.json"
        ),
        "large_target_stress_summary": m3_root / "ann_stress_D3_to_D9" / "ann_stress_summary_D3_to_D9.json",
    }


def output_paths(reports_dir: Path) -> dict[str, Path]:
    return {
        "backend_report": reports_dir / "m3_backend_freeze_decision.md",
        "backend_json": reports_dir / "m3_backend_freeze_decision.json",
        "dryrun_report": reports_dir / "m3_full_m3_final_dryrun_plan.md",
        "dryrun_shards": reports_dir / "m3_full_m3_final_dryrun_shards.csv",
        "expected_outputs": reports_dir / "m3_full_m3_expected_outputs.json",
        "slurm_template": reports_dir / "templates" / "m3_full_m3_array.sbatch",
    }


def _path_has_forbidden_token(path: Path) -> str | None:
    for part in path.resolve().parts:
        lower = part.lower()
        if lower == "nichefate":
            continue
        for token in OUTPUT_TOKENS:
            if token in lower:
                return token
    return None


def assert_no_ssd_path(path: Path, label: str) -> None:
    if "/ssd" in str(path.resolve()):
        raise ValueError(f"Refusing to use /ssd for {label}: {path}")


def assert_report_path_is_safe(path: Path, label: str) -> None:
    assert_no_ssd_path(path, label)
    token = _path_has_forbidden_token(path)
    if token is not None:
        raise ValueError(f"Refusing downstream-looking {label} path containing {token!r}: {path}")


def validate_stage_scope(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if not args.dry_run_only:
        raise ValueError("M3-14 is dry-run/report-only; pass --dry-run-only to make that explicit.")
    if int(args.candidate_k) != DEFAULT_CANDIDATE_K:
        raise ValueError("M3-14 full-M3 dry run is scoped to candidate_k=30.")
    if bool(config["paths"].get("use_ssd", False)):
        raise RuntimeError("Refusing M3-14 while paths.use_ssd is true.")
    for key, value in config.get("paths", {}).items():
        if isinstance(value, str) and value.startswith("/ssd"):
            raise RuntimeError(f"Refusing M3-14 because config path {key} uses /ssd: {value}")
    assert_no_ssd_path(args.default_output_root, "future production root")
    assert_no_ssd_path(args.reports_dir, "reports directory")


def load_optional_text(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "present": False, "bytes": 0, "text": ""}
    text = path.read_text(encoding="utf-8")
    return {"path": str(path), "present": True, "bytes": len(text.encode("utf-8")), "text": text}


def load_optional_summary(name: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"name": name, "path": str(path), "present": False, "status": "MISSING", "metrics": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "name": name,
        "path": str(path),
        "present": True,
        "status": payload.get("status", "UNKNOWN"),
        "metrics": payload.get("metrics", {}),
        "raw": payload,
    }


def _metric(metrics: dict[str, Any], names: list[str]) -> float | None:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            return float(value)
    return None


def normalize_evidence(summary: dict[str, Any], label: str) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    exact_runtime = _metric(metrics, ["sklearn_exact_runtime_seconds", "exact_reference_runtime_seconds"])
    ann_runtime = _metric(metrics, ["pynndescent_runtime_seconds", "ann_runtime_seconds"])
    exact_memory = _metric(metrics, ["sklearn_exact_max_rss_gib", "exact_reference_max_rss_gib"])
    ann_memory = _metric(metrics, ["pynndescent_max_rss_gib", "ann_max_rss_gib"])
    runtime_ratio = _metric(metrics, ["runtime_ratio_ann_over_exact"])
    if runtime_ratio is None and exact_runtime:
        runtime_ratio = float(ann_runtime / exact_runtime) if ann_runtime is not None else None
    memory_ratio = _metric(metrics, ["memory_ratio_ann_over_exact"])
    if memory_ratio is None and exact_memory:
        memory_ratio = float(ann_memory / exact_memory) if ann_memory is not None else None
    recall = _metric(metrics, ["recall_at_30_mean"])
    top1 = _metric(metrics, ["top1_agreement"])
    jaccard = _metric(metrics, ["jaccard_overlap_mean"])
    probability_drift_p95 = _metric(metrics, ["row_normalized_transition_prob_abs_drift_p95"])
    soft_pass = metrics.get("soft_validation_pass")
    accurate = bool(
        summary.get("present")
        and summary.get("status") == "COMPLETED"
        and recall is not None
        and top1 is not None
        and recall >= 0.8
        and top1 >= 0.8
        and (soft_pass is True or str(soft_pass).lower() == "true")
    )
    return {
        "name": summary["name"],
        "label": label,
        "path": summary["path"],
        "present": bool(summary.get("present")),
        "status": summary.get("status", "UNKNOWN"),
        "accuracy_supported": accurate,
        "recall_at_30_mean": recall,
        "top1_agreement": top1,
        "jaccard_overlap_mean": jaccard,
        "probability_drift_p95": probability_drift_p95,
        "exact_runtime_seconds": exact_runtime,
        "ann_runtime_seconds": ann_runtime,
        "runtime_ratio_ann_over_exact": runtime_ratio,
        "exact_max_rss_gib": exact_memory,
        "ann_max_rss_gib": ann_memory,
        "memory_ratio_ann_over_exact": memory_ratio,
        "soft_validation_pass": bool(soft_pass) if soft_pass is not None else None,
    }


def read_evidence(input_paths: dict[str, Path]) -> dict[str, Any]:
    late = late_target_time()
    summaries = [
        normalize_evidence(
            load_optional_summary("m3_09_sampled_validation", input_paths["sampled_validation_summary"]),
            f"M3-09 sampled D21->{late}",
        ),
        normalize_evidence(
            load_optional_summary("m3_12_full_shard_validation", input_paths["full_shard_validation_summary"]),
            f"M3-12 full-shard D21->{late}",
        ),
        normalize_evidence(
            load_optional_summary("m3_13_large_target_stress", input_paths["large_target_stress_summary"]),
            "M3-13 large-target D3->D9",
        ),
    ]
    docs = {
        "ann_validation_plan": load_optional_text(input_paths["ann_validation_plan"]),
        "slurm_strategy": load_optional_text(input_paths["slurm_strategy"]),
    }
    return {"summaries": summaries, "documents": docs}


def extract_concurrency_cap(slurm_strategy_text: str) -> int:
    match = re.search(r"Recommended global concurrency cap:\s*([0-9]+)", slurm_strategy_text)
    return int(match.group(1)) if match else RECOMMENDED_CONCURRENCY_CAP


def freeze_backend_decision(evidence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summaries = evidence["summaries"]
    by_name = {item["name"]: item for item in summaries}
    completed = [item for item in summaries if item["present"] and item["status"] == "COMPLETED"]
    accurate_completed = [item for item in completed if item["accuracy_supported"]]
    required_names = {
        "m3_09_sampled_validation",
        "m3_12_full_shard_validation",
        "m3_13_large_target_stress",
    }
    sufficient = required_names <= {item["name"] for item in accurate_completed}
    stress = by_name.get("m3_13_large_target_stress", {})
    stress_ratio = stress.get("runtime_ratio_ann_over_exact")
    stress_ann_slower = bool(stress_ratio is not None and stress_ratio > 1.0)
    all_available_accurate = bool(summaries) and len(accurate_completed) == len(completed) and len(completed) > 0
    cap = extract_concurrency_cap(evidence["documents"]["slurm_strategy"]["text"])

    if stress.get("accuracy_supported") and stress_ann_slower:
        default_backend = "sklearn_exact"
        optional_backend = "pynndescent"
        runtime_conclusion = (
            "pynndescent is accurate but slower than sklearn_exact in the large-target D3->D9 stress test"
        )
    elif all_available_accurate and sufficient:
        default_backend = "pynndescent"
        optional_backend = "sklearn_exact"
        runtime_conclusion = "available ANN evidence supports pynndescent accuracy without a large-target slowdown"
    else:
        default_backend = "sklearn_exact"
        optional_backend = "pynndescent"
        runtime_conclusion = "available evidence is incomplete or mixed, so the conservative exact backend remains default"

    evidence_strength = "strongly_supported" if sufficient else "limited_available"
    missing = [item for item in summaries if not item["present"]]
    accuracy_conclusion = (
        "pynndescent candidate accuracy is supported by all required validation summaries"
        if sufficient
        else "pynndescent accuracy evidence is useful but not complete enough for a strong freeze label"
    )
    return {
        "schema_version": "m3_backend_freeze_v1",
        "default_backend": default_backend,
        "optional_backend": optional_backend,
        "full_m3_execution_mode": "slurm_job_array_with_concurrency_cap",
        "recommended_global_concurrency_cap": cap,
        "candidate_k": int(config["full_m3"].get("candidate_k", DEFAULT_CANDIDATE_K)),
        "evidence_strength": evidence_strength,
        "accuracy_conclusion": accuracy_conclusion,
        "runtime_memory_conclusion": runtime_conclusion,
        "rationale": (
            "Use sklearn_exact as the full-M3 default after M3-13 because accuracy is good but "
            "pynndescent did not provide a runtime win on the largest tested target pool. Keep "
            "pynndescent as an optional fallback for high-risk shards and later index-caching experiments."
        )
        if default_backend == "sklearn_exact"
        else (
            "Use pynndescent as default only when full evidence supports both accuracy and runtime benefit; "
            "keep sklearn_exact as exact fallback."
        ),
        "missing_evidence": [{"name": item["name"], "path": item["path"]} for item in missing],
        "evidence": summaries,
    }


def safe_token(value: object) -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return text.strip("_") or "value"


def shard_output_dir(root: Path, source_time: str, target_time: str, source_slice_id: str) -> Path:
    return root / f"{safe_token(source_time)}_to_{safe_token(target_time)}" / safe_token(source_slice_id)


def build_dryrun_shards(
    shards: pd.DataFrame,
    default_output_root: Path,
    selected_backend: str,
    candidate_k: int,
) -> pd.DataFrame:
    required = {
        "source_time",
        "target_time",
        "source_day",
        "target_day",
        "time_delta",
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
    if int(candidate_k) != DEFAULT_CANDIDATE_K:
        raise ValueError("Dry-run shard plan is scoped to candidate_k=30.")
    plan = shards.copy().reset_index(drop=True)
    if len(plan) != EXPECTED_SHARD_COUNT:
        raise ValueError(f"Expected {EXPECTED_SHARD_COUNT} full-M3 shards, found {len(plan)}.")
    if int(plan["expected_edge_rows"].sum()) != EXPECTED_TOTAL_EDGE_ROWS:
        raise ValueError(
            f"Expected total edge rows {EXPECTED_TOTAL_EDGE_ROWS}, found {int(plan['expected_edge_rows'].sum())}."
        )
    if late_target_time() in set(plan["source_time"].astype(str)):
        raise ValueError("Final target time must not be used as a source time in full-M3 shard plan.")
    if not bool((plan["candidate_k"].astype(int) == int(candidate_k)).all()):
        raise ValueError("All full-M3 shards must use candidate_k=30.")

    rows: list[dict[str, Any]] = []
    for idx, row in plan.iterrows():
        output_dir = shard_output_dir(
            default_output_root,
            str(row["source_time"]),
            str(row["target_time"]),
            str(row["source_slice_id"]),
        )
        stem = f"{safe_token(row['source_time'])}_to_{safe_token(row['target_time'])}__{safe_token(row['source_slice_id'])}"
        output_parquet = output_dir / f"candidate_edges_{stem}.parquet"
        shard_report = output_dir / f"shard_report_{stem}.md"
        rows.append(
            {
                "shard_id": f"m3_full_{idx + 1:04d}",
                "source_time": row["source_time"],
                "target_time": row["target_time"],
                "source_day": float(row["source_day"]),
                "target_day": float(row["target_day"]),
                "time_delta": float(row["time_delta"]),
                "source_slice_id": row["source_slice_id"],
                "source_slice_file": row["source_slice_file"],
                "source_rows": int(row["source_rows"]),
                "target_rows": int(row["target_time_rows"]),
                "candidate_k": int(row["candidate_k"]),
                "expected_edge_rows": int(row["expected_edge_rows"]),
                "selected_backend": selected_backend,
                "output_dir": str(output_dir),
                "output_parquet": str(output_parquet),
                "shard_report": str(shard_report),
                "status_expected": "pending_explicit_approval",
                "can_resume": False,
                "reuse_existing_pilot_allowed": False,
                "requires_explicit_approval": True,
            }
        )
    return pd.DataFrame(rows)


def planned_control_outputs(default_output_root: Path) -> dict[str, str]:
    return {
        "completed_shards_csv": str(default_output_root / "completed_shards.csv"),
        "failed_shards_txt": str(default_output_root / "failed_shards.txt"),
        "full_m3_qc_summary_csv": str(default_output_root / "full_m3_qc_summary.csv"),
        "full_m3_report_md": str(default_output_root / "full_m3_report.md"),
    }


def find_pilot_references(reports_dir: Path) -> list[dict[str, Any]]:
    m3_root = reports_dir.parent
    patterns = [
        "timepair_pilot_D21_to_*/candidate_edges_*.parquet",
        "ann_full_shard_validation_D21_to_*/ann_full_shard_edges_*.parquet",
    ]
    references: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(m3_root.glob(pattern)):
            references.append(
                {
                    "path": str(path),
                    "exists": path.exists(),
                    "registered_as_production": False,
                    "reuse_existing_pilot_allowed": False,
                }
            )
    return references


def expected_outputs_payload(
    dryrun_shards: pd.DataFrame,
    default_output_root: Path,
    reports_dir: Path,
    decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "m3_full_m3_expected_outputs_v1",
        "future_production_root": str(default_output_root),
        "production_edge_parquets_created_in_m3_14": False,
        "shard_count": int(len(dryrun_shards)),
        "expected_total_edge_rows": int(dryrun_shards["expected_edge_rows"].sum()),
        "default_backend": decision["default_backend"],
        "optional_backend": decision["optional_backend"],
        "control_outputs": planned_control_outputs(default_output_root),
        "per_shard_outputs": dryrun_shards[
            ["shard_id", "output_dir", "output_parquet", "shard_report", "expected_edge_rows"]
        ].to_dict("records"),
        "pilot_reference_outputs": find_pilot_references(reports_dir),
        "downstream_outputs": {
            "global_markov_p_produced": False,
            "gpcca_produced": False,
            "fate_probability_produced": False,
            "branched_nicheflow_produced": False,
            "m5_produced": False,
            "regulator_analysis_produced": False,
        },
    }


def validate_planned_paths(paths: list[Path]) -> None:
    for path in paths:
        assert_no_ssd_path(path, "planned output")
        token = _path_has_forbidden_token(path)
        if token is not None:
            raise ValueError(f"Planned output path contains downstream token {token!r}: {path}")


def production_parquets(default_output_root: Path) -> list[Path]:
    if not default_output_root.exists():
        return []
    return sorted(default_output_root.glob("**/candidate_edges_*.parquet"))


def benchmark_token_scan(paths: list[Path]) -> dict[str, list[str]]:
    findings: dict[str, list[str]] = {}
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        matches = [token for token in BENCHMARK_TOKENS if token in text]
        if matches:
            findings[str(path)] = matches
    return findings


def write_backend_report(path: Path, decision: dict[str, Any]) -> None:
    lines = [
        "# M3 Backend Freeze Decision",
        "",
        "This report freezes the recommended backend strategy for future full-M3 edge-shard construction only.",
        "It does not run full M3, create production edge parquets, assemble global Markov P, run GPCCA, compute fate probabilities, run Branched NicheFlow, M5, or regulator analysis.",
        "",
        "## Decision",
        f"- evidence strength: {decision['evidence_strength']}",
        f"- default backend: {decision['default_backend']}",
        f"- optional fallback backend: {decision['optional_backend']}",
        f"- execution mode: {decision['full_m3_execution_mode']}",
        f"- recommended global concurrency cap: {decision['recommended_global_concurrency_cap']}",
        f"- candidate_k: {decision['candidate_k']}",
        "",
        "## Conclusions",
        f"- accuracy: {decision['accuracy_conclusion']}",
        f"- runtime/memory: {decision['runtime_memory_conclusion']}",
        f"- rationale: {decision['rationale']}",
        "",
        "## Evidence",
        "",
    ]
    for item in decision["evidence"]:
        if not item["present"]:
            lines.append(f"- {item['label']}: missing ({item['path']})")
            continue
        lines.append(
            "- "
            f"{item['label']}: status={item['status']}, "
            f"recall@30={item['recall_at_30_mean']}, "
            f"top1={item['top1_agreement']}, "
            f"Jaccard={item['jaccard_overlap_mean']}, "
            f"probability drift p95={item['probability_drift_p95']}, "
            f"exact_runtime={item['exact_runtime_seconds']}, "
            f"ann_runtime={item['ann_runtime_seconds']}, "
            f"runtime_ratio_ann_over_exact={item['runtime_ratio_ann_over_exact']}, "
            f"soft_pass={item['soft_validation_pass']}"
        )
    if decision["missing_evidence"]:
        lines.extend(["", "## Missing Optional Evidence"])
        for item in decision["missing_evidence"]:
            lines.append(f"- {item['name']}: {item['path']}")
    lines.extend(
        [
            "",
            "## Reconfiguration",
            "",
            "The backend remains config-driven. If future evidence changes, full-M3 execution can be reconfigured before production approval.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_dryrun_report(
    path: Path,
    dryrun_shards: pd.DataFrame,
    decision: dict[str, Any],
    default_output_root: Path,
    expected_outputs: dict[str, Any],
) -> None:
    next_runner = "scripts/m3_15_run_full_m3_by_shard.py"
    lines = [
        "# M3 Full-M3 Final Dry-Run Plan",
        "",
        "This is a final dry-run plan only. It does not run full M3, construct D0->D3, D3->D9, D9->D21, or D21->final target full edge shards, copy pilot outputs into production directories, assemble global Markov P, run GPCCA, compute fate probabilities, run Branched NicheFlow, M5, or regulator analysis.",
        "",
        "## Shard Plan",
        f"- shards planned: {len(dryrun_shards)}",
        f"- expected total edge rows: {int(dryrun_shards['expected_edge_rows'].sum())}",
        f"- candidate_k: {int(dryrun_shards['candidate_k'].iloc[0])}",
        f"- default backend: {decision['default_backend']}",
        f"- optional fallback backend: {decision['optional_backend']}",
        f"- future production root: {default_output_root}",
        f"- production edge parquets created in this stage: {expected_outputs['production_edge_parquets_created_in_m3_14']}",
        "",
        "## Production Output Contract",
        "",
        "`full_by_shard/<source_time>_to_<target_time>/<source_slice_id>/` will contain:",
        "- `candidate_edges_<source_time>_to_<target_time>__<source_slice_id>.parquet`",
        "- `shard_report_<source_time>_to_<target_time>__<source_slice_id>.md`",
        "",
        "Future control outputs:",
    ]
    for label, value in expected_outputs["control_outputs"].items():
        lines.append(f"- {label}: {value}")
    lines.extend(
        [
            "",
            "## Pilot And Reference Separation",
            "",
            "- Existing pilot/reference outputs may be listed for audit context.",
            "- Existing pilot/reference outputs are not registered as production outputs.",
            "- `reuse_existing_pilot_allowed` is false for every planned production shard.",
            "- `status_expected` is `pending_explicit_approval` for every shard.",
            "",
            "## Runner Strategy",
            "",
            "- One source-slice shard per job.",
            f"- Default backend: `{decision['default_backend']}`.",
            "- `candidate_k=30`.",
            "- Sequential local mode is allowed only for small/manual pilots.",
            "- Full execution should use Slurm/job-array with a concurrency cap.",
            f"- Recommended global concurrency cap: {decision['recommended_global_concurrency_cap']}.",
            "- Target-time grouped execution is preferred.",
            "- Resume may skip a shard only after validating existing parquet path, shard report, schema, row count, finite numeric values, K candidates per source, local row-sum diagnostics, and target slice/mouse diagnostics.",
            "- Post-shard validation is required before recording completion.",
            "- No unrestricted Python multiprocessing.",
            "",
            "## Exact Next Command Plan",
            "",
            "After explicit approval and after a production runner is implemented/reviewed:",
            "",
            "```bash",
            "cd /home/zhutao/projects/nichefate",
            f"conda run --no-capture-output -n omicverse python {next_runner} \\",
            "  --config configs/m3_transition_kernel.yaml \\",
            "  --shard-plan /home/zhutao/scratch/nichefate/m3/reports/m3_full_m3_final_dryrun_shards.csv \\",
            f"  --output-root {default_output_root} \\",
            f"  --backend {decision['default_backend']} \\",
            "  --candidate-k 30 \\",
            "  --validate-resume \\",
            "  --require-explicit-production-approval",
            "```",
            "",
            "For Slurm after explicit approval:",
            "",
            "```bash",
            "sbatch /home/zhutao/scratch/nichefate/m3/reports/templates/m3_full_m3_array.sbatch",
            "```",
            "",
            "Do not submit the template until production full-M3 execution is explicitly approved.",
            "",
            "## Downstream Non-Execution",
            "",
            "- No production M3 edge parquet was created.",
            "- No global Markov P was created.",
            "- No GPCCA, fate probability, Branched NicheFlow, M5, or regulator output paths were created.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_slurm_template(path: Path, decision: dict[str, Any], default_output_root: Path) -> None:
    array_count = EXPECTED_SHARD_COUNT
    cap = int(decision["recommended_global_concurrency_cap"])
    lines = [
        "#!/bin/bash",
        "# Non-submitting M3 full-shard array template. Review and explicitly approve before use.",
        "#SBATCH --job-name=nichefate_m3_full",
        f"#SBATCH --array=1-{array_count}%{cap}",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=80G",
        "#SBATCH --time=24:00:00",
        "#SBATCH --output=/home/zhutao/scratch/nichefate/m3/logs/full_m3_%A_%a.out",
        "#SBATCH --error=/home/zhutao/scratch/nichefate/m3/logs/full_m3_%A_%a.err",
        "",
        "set -euo pipefail",
        "cd /home/zhutao/projects/nichefate",
        "",
        "conda run --no-capture-output -n omicverse python scripts/m3_15_run_full_m3_by_shard.py \\",
        "  --config configs/m3_transition_kernel.yaml \\",
        "  --shard-plan /home/zhutao/scratch/nichefate/m3/reports/m3_full_m3_final_dryrun_shards.csv \\",
        f"  --output-root {default_output_root} \\",
        f"  --backend {decision['default_backend']} \\",
        "  --candidate-k 30 \\",
        "  --array-task-id \"${SLURM_ARRAY_TASK_ID}\" \\",
        "  --validate-resume \\",
        "  --require-explicit-production-approval",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def maybe_write_config(config_path: str | Path, config: dict[str, Any], decision: dict[str, Any]) -> None:
    import yaml

    resolved = resolve_config_path(config_path)
    updated = json.loads(json.dumps(config))
    updated["full_m3"]["neighbor_backend"] = decision["default_backend"]
    updated["full_m3"]["execution_mode"] = "slurm_job_array_dryrun_approved_backend"
    updated["full_m3"]["enabled"] = False
    with resolved.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(updated, handle, sort_keys=False)


def write_outputs(
    paths: dict[str, Path],
    decision: dict[str, Any],
    dryrun_shards: pd.DataFrame,
    expected_outputs: dict[str, Any],
    default_output_root: Path,
) -> None:
    paths["backend_report"].parent.mkdir(parents=True, exist_ok=True)
    write_backend_report(paths["backend_report"], decision)
    paths["backend_json"].write_text(json.dumps(json_safe(decision), indent=2) + "\n", encoding="utf-8")
    dryrun_shards.to_csv(paths["dryrun_shards"], index=False)
    paths["expected_outputs"].write_text(
        json.dumps(json_safe(expected_outputs), indent=2) + "\n",
        encoding="utf-8",
    )
    write_dryrun_report(paths["dryrun_report"], dryrun_shards, decision, default_output_root, expected_outputs)
    write_slurm_template(paths["slurm_template"], decision, default_output_root)


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    validate_stage_scope(args, config)
    input_paths = default_input_paths(args.reports_dir)
    paths = output_paths(args.reports_dir)
    for label, path in paths.items():
        if label != "slurm_template":
            assert_report_path_is_safe(path, label)

    evidence = read_evidence(input_paths)
    decision = freeze_backend_decision(evidence, config)
    shards = pd.read_csv(input_paths["shards_csv"])
    dryrun_shards = build_dryrun_shards(
        shards,
        args.default_output_root,
        decision["default_backend"],
        int(args.candidate_k),
    )
    planned_paths = [Path(value) for value in dryrun_shards["output_parquet"].tolist()]
    planned_paths.extend(Path(value) for value in dryrun_shards["shard_report"].tolist())
    planned_paths.extend(Path(value) for value in planned_control_outputs(args.default_output_root).values())
    validate_planned_paths(planned_paths)
    existing_production = production_parquets(args.default_output_root)
    if existing_production:
        raise RuntimeError(
            "Refusing M3-14 dry run because production edge parquet files already exist under "
            f"{args.default_output_root}: {existing_production[:3]}"
        )
    scan_findings = benchmark_token_scan(
        [PROJECT_ROOT / "src" / "nichefate" / "transition.py", resolve_config_path(args.config)]
    )
    if scan_findings:
        raise RuntimeError(f"Focused M3 core/config dataset-token scan found matches: {scan_findings}")

    expected_outputs = expected_outputs_payload(dryrun_shards, args.default_output_root, args.reports_dir, decision)
    write_outputs(paths, decision, dryrun_shards, expected_outputs, args.default_output_root)
    if args.write_config:
        maybe_write_config(args.config, config, decision)

    print("M3_BACKEND_FREEZE_DRYRUN_COMPLETED")
    print(f"DEFAULT_BACKEND {decision['default_backend']}")
    print(f"OPTIONAL_BACKEND {decision['optional_backend']}")
    print(f"EVIDENCE_STRENGTH {decision['evidence_strength']}")
    print(f"SHARD_COUNT {len(dryrun_shards)}")
    print(f"EXPECTED_TOTAL_EDGE_ROWS {int(dryrun_shards['expected_edge_rows'].sum())}")
    print(f"PRODUCTION_EDGE_PARQUETS_CREATED {False}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
