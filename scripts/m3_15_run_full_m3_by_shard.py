#!/usr/bin/env python
"""Run full M3 source-slice edge-shard construction with controlled execution."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

for _thread_var in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_thread_var, "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config
from nichefate.transition import full_transition_schema_columns


DEFAULT_PLAN_CSV = Path("/home/zhutao/scratch/nichefate/m3/reports/m3_full_m3_final_dryrun_shards.csv")
DEFAULT_EXPECTED_OUTPUTS = Path("/home/zhutao/scratch/nichefate/m3/reports/m3_full_m3_expected_outputs.json")
DEFAULT_OUTPUT_ROOT = Path("/home/zhutao/scratch/nichefate/m3/full_by_shard")
DEFAULT_REPORTS_DIR = Path("/home/zhutao/scratch/nichefate/m3/reports")
DEFAULT_BACKEND = "sklearn_exact"
DEFAULT_CANDIDATE_K = 30
EXPECTED_SHARD_COUNT = 52
EXPECTED_TOTAL_EDGE_ROWS = 40_457_460
ROW_SUM_ATOL = 1e-8

NO_DOWNSTREAM_FLAGS = {
    "no_global_markov_p": True,
    "no_gpcca": True,
    "no_fate_probability": True,
    "no_branched_nicheflow": True,
    "no_m5": True,
    "no_regulator_analysis": True,
}


def _load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M3_05 = _load_script_module(
    "m3_05_build_transition_pilot_shard",
    PROJECT_ROOT / "scripts" / "m3_05_build_transition_pilot_shard.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_transition_kernel.yaml")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--expected-outputs-json", type=Path, default=DEFAULT_EXPECTED_OUTPUTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--source-time", default=None)
    parser.add_argument("--target-time", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--execution-mode", default="sequential")
    return parser.parse_args()


def late_target_time() -> str:
    return "D" + "35"


def max_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / float(1024**2)


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


def load_expected_outputs(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected outputs contract: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_scope(args: argparse.Namespace, config: dict[str, Any], expected_outputs: dict[str, Any]) -> None:
    if args.execution_mode != "sequential":
        raise ValueError("M3-15 runner supports only --execution-mode sequential.")
    if args.backend != DEFAULT_BACKEND:
        raise ValueError("M3-15 production execution must use frozen backend sklearn_exact.")
    if int(args.candidate_k) != DEFAULT_CANDIDATE_K:
        raise ValueError("M3-15 production execution is scoped to candidate_k=30.")
    if int(args.blas_threads) <= 0:
        raise ValueError("--blas-threads must be positive.")
    if bool(config["paths"].get("use_ssd", False)):
        raise RuntimeError("Refusing M3-15 while paths.use_ssd is true.")
    for key, value in config.get("paths", {}).items():
        if isinstance(value, str) and value.startswith("/ssd"):
            raise RuntimeError(f"Refusing M3-15 because config path {key} uses /ssd: {value}")
    if "/ssd" in str(args.output_root.resolve()):
        raise ValueError(f"Refusing to write M3-15 outputs under /ssd: {args.output_root}")
    if expected_outputs.get("default_backend") != args.backend:
        raise ValueError("Expected-output contract backend does not match requested backend.")
    if Path(expected_outputs["future_production_root"]).resolve() != args.output_root.resolve():
        raise ValueError("Expected-output contract production root does not match --output-root.")
    if config["full_m3"].get("write_global_kernel"):
        raise RuntimeError("Refusing to run while full_m3.write_global_kernel is true.")


def load_plan(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing full-M3 shard plan: {path}")
    plan = pd.read_csv(path)
    required = {
        "shard_id",
        "source_time",
        "target_time",
        "source_day",
        "target_day",
        "time_delta",
        "source_slice_id",
        "source_slice_file",
        "source_rows",
        "target_rows",
        "candidate_k",
        "expected_edge_rows",
        "selected_backend",
        "output_dir",
        "output_parquet",
        "shard_report",
        "reuse_existing_pilot_allowed",
        "requires_explicit_approval",
    }
    missing = sorted(required - set(plan.columns))
    if missing:
        raise KeyError(f"Shard plan is missing required columns: {missing}")
    if len(plan) != EXPECTED_SHARD_COUNT:
        raise ValueError(f"Expected {EXPECTED_SHARD_COUNT} shards, found {len(plan)}.")
    if int(plan["expected_edge_rows"].sum()) != EXPECTED_TOTAL_EDGE_ROWS:
        raise ValueError("Shard plan expected edge row sum does not match M3-14 contract.")
    if late_target_time() in set(plan["source_time"].astype(str)):
        raise ValueError("Final time point must not be used as source.")
    if not bool((plan["candidate_k"].astype(int) == DEFAULT_CANDIDATE_K).all()):
        raise ValueError("All shards must use candidate_k=30.")
    if bool(plan["reuse_existing_pilot_allowed"].astype(bool).any()):
        raise ValueError("Pilot outputs must not be marked reusable as production outputs.")
    return plan


def filter_plan(plan: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = plan.copy()
    if args.source_time is not None:
        selected = selected[selected["source_time"].astype(str) == str(args.source_time)]
    if args.target_time is not None:
        selected = selected[selected["target_time"].astype(str) == str(args.target_time)]
    selected = selected.reset_index(drop=True)
    if args.max_shards is not None:
        if int(args.max_shards) <= 0:
            raise ValueError("--max-shards must be positive when provided.")
        selected = selected.head(int(args.max_shards)).reset_index(drop=True)
    return selected


def apply_runtime_threads(blas_threads: int) -> None:
    for key in [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[key] = str(int(blas_threads))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def slice_path(root: Path, slice_id: str) -> Path:
    return root / slice_id / f"m2_representation_{slice_id}.parquet"


def target_slices_for_pair(config: dict[str, Any], source_time: str, target_time: str) -> list[str]:
    pairs = load_json(Path(config["paths"]["reports_dir"]) / "m3_time_pairs.json")
    for pair in pairs:
        if str(pair["source_time"]) == str(source_time) and str(pair["target_time"]) == str(target_time):
            return [str(value) for value in pair["target_slices"]]
    raise ValueError(f"Missing time-pair metadata for {source_time}->{target_time}.")


def feature_columns(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    payload = load_json(Path(config["paths"]["reports_dir"]) / "m3_feature_groups.json")
    groups = payload["feature_groups"]
    retrieval = [
        column
        for group in config["full_m3"]["retrieval_feature_groups"]
        for column in groups[group]
    ]
    rerank = [
        column
        for group in config["full_m3"]["rerank_feature_groups"]
        for column in groups[group]
    ]
    read_columns = list(dict.fromkeys(config["input"]["metadata_columns"] + retrieval + rerank))
    return payload, read_columns


def load_shard_data(
    config: dict[str, Any],
    shard: pd.Series,
    feature_read_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(config["paths"]["m2_by_slice_dir"])
    source = pd.read_parquet(slice_path(root, str(shard["source_slice_id"])), columns=feature_read_columns)
    target_frames = [
        pd.read_parquet(slice_path(root, slice_id), columns=feature_read_columns)
        for slice_id in target_slices_for_pair(config, str(shard["source_time"]), str(shard["target_time"]))
    ]
    target = pd.concat(target_frames, ignore_index=True)
    if len(source) != int(shard["source_rows"]):
        raise ValueError(f"Source rows {len(source)} != expected {shard['source_rows']}.")
    if len(target) != int(shard["target_rows"]):
        raise ValueError(f"Target rows {len(target)} != expected {shard['target_rows']}.")
    if not bool((source["time"].astype(str) == str(shard["source_time"])).all()):
        raise ValueError("Source data time does not match shard plan.")
    if not bool((target["time"].astype(str) == str(shard["target_time"])).all()):
        raise ValueError("Target data time does not match shard plan.")
    return source, target


def configure_full_m3(config: dict[str, Any], backend: str, candidate_k: int) -> dict[str, Any]:
    configured = json.loads(json.dumps(config))
    configured["full_m3"]["neighbor_backend"] = backend
    configured["full_m3"]["candidate_k"] = int(candidate_k)
    configured["full_m3"]["write_global_kernel"] = False
    configured["full_m3"]["enabled"] = False
    return configured


def row_entropy(values: pd.Series) -> float:
    probs = values.astype(float).to_numpy()
    return float(-(probs * np.log(np.clip(probs, 1e-300, None))).sum())


def categorical_entropy_top_fraction(frame: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    entropies = {}
    top_fractions = {}
    for source_id, values in frame.groupby("source_anchor_id", observed=True)[column]:
        probs = values.astype(str).value_counts(normalize=True).to_numpy(dtype=float)
        entropies[source_id] = float(-(probs * np.log(np.clip(probs, 1e-300, None))).sum()) if len(probs) else 0.0
        top_fractions[source_id] = float(probs.max()) if len(probs) else 0.0
    return pd.Series(entropies), pd.Series(top_fractions)


def collapse_warnings(metrics: dict[str, Any]) -> list[str]:
    warnings = []
    if metrics["top_target_slice_id_fraction_p95"] >= 0.95:
        warnings.append("top target-slice fraction p95 >= 0.95")
    if metrics["top_target_mouse_id_fraction_p95"] >= 0.95:
        warnings.append("top target-mouse fraction p95 >= 0.95")
    return warnings


def validate_edge_frame(frame: pd.DataFrame, shard: pd.Series) -> dict[str, Any]:
    schema = full_transition_schema_columns()
    missing = sorted(set(schema) - set(frame.columns))
    if missing:
        raise ValueError(f"Shard is missing required columns: {missing}")
    expected_rows = int(shard["expected_edge_rows"])
    if len(frame) != expected_rows:
        raise ValueError(f"Observed rows {len(frame)} != expected rows {expected_rows}.")
    source_count = int(frame["source_anchor_id"].nunique())
    if source_count != int(shard["source_rows"]):
        raise ValueError(f"Source anchors represented {source_count} != expected {shard['source_rows']}.")
    counts = frame.groupby("source_anchor_id", observed=True).size()
    if int(counts.min()) != int(shard["candidate_k"]) or int(counts.max()) != int(shard["candidate_k"]):
        raise ValueError("Not every source anchor has exactly candidate_k targets.")
    numeric_check_cols = [
        "combined_cost",
        "tau_pair",
        "raw_edge_weight",
        "mass_adjusted_weight",
        "row_normalized_transition_prob",
    ]
    values = frame[numeric_check_cols].to_numpy(dtype=float)
    if int((~np.isfinite(values)).sum()):
        raise ValueError("Shard has NaN or infinite costs, weights, or probabilities.")
    if bool((frame["row_normalized_transition_prob"] < 0).any()):
        raise ValueError("Shard has negative probabilities.")
    if bool((frame["raw_edge_weight"] < 0).any()) or bool((frame["mass_adjusted_weight"] < 0).any()):
        raise ValueError("Shard has negative weights.")
    row_sums = frame.groupby("source_anchor_id", observed=True)["row_normalized_transition_prob"].sum()
    max_abs_row_sum_error = float((row_sums - 1.0).abs().max())
    if max_abs_row_sum_error > ROW_SUM_ATOL:
        raise ValueError(f"Row-sum max error {max_abs_row_sum_error} exceeds tolerance {ROW_SUM_ATOL}.")
    metadata_cols = [column for column in frame.columns if column.startswith(("source_", "target_"))]
    if int(frame[metadata_cols].isna().sum().sum()):
        raise ValueError("Shard has missing source/target metadata.")
    if not bool((frame["source_time"].astype(str) == str(shard["source_time"])).all()):
        raise ValueError("Source time mismatch.")
    if not bool((frame["target_time"].astype(str) == str(shard["target_time"])).all()):
        raise ValueError("Target time mismatch.")
    if not bool(np.allclose(frame["time_delta"].astype(float), float(shard["time_delta"]))):
        raise ValueError("Time delta mismatch.")
    probs = frame.groupby("source_anchor_id", observed=True)["row_normalized_transition_prob"]
    entropy = probs.apply(row_entropy)
    top1 = probs.max()
    slice_entropy, slice_top = categorical_entropy_top_fraction(frame, "target_slice_id")
    mouse_entropy, mouse_top = categorical_entropy_top_fraction(frame, "target_mouse_id")
    metrics = {
        "observed_edge_rows": int(len(frame)),
        "source_anchors_represented": source_count,
        "candidate_count_min": int(counts.min()),
        "candidate_count_max": int(counts.max()),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "row_sum_abs_error_max": max_abs_row_sum_error,
        "row_entropy_mean": float(entropy.mean()),
        "row_entropy_median": float(entropy.median()),
        "top1_probability_mean": float(top1.mean()),
        "top1_probability_median": float(top1.median()),
        "target_slice_id_entropy_mean": float(slice_entropy.mean()),
        "target_mouse_id_entropy_mean": float(mouse_entropy.mean()),
        "top_target_slice_id_fraction_p95": float(slice_top.quantile(0.95)),
        "top_target_mouse_id_fraction_p95": float(mouse_top.quantile(0.95)),
        "negative_probability_count": int((frame["row_normalized_transition_prob"] < 0).sum()),
        "nonfinite_numeric_count": 0,
        "schema_version": "m3_full_transition_schema_v1",
    }
    metrics["collapse_warnings"] = collapse_warnings(metrics)
    return metrics


def validate_existing_outputs(shard: pd.Series) -> tuple[bool, dict[str, Any], str]:
    edge_path = Path(shard["output_parquet"])
    report_path = Path(shard["shard_report"])
    if not edge_path.exists() or not report_path.exists():
        return False, {}, "missing parquet or shard report"
    try:
        frame = pd.read_parquet(edge_path)
        metrics = validate_edge_frame(frame, shard)
        metrics["output_bytes"] = int(edge_path.stat().st_size)
        return True, metrics, "valid existing shard"
    except Exception as exc:  # noqa: BLE001
        return False, {}, f"existing shard validation failed: {exc}"


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def shard_report_text(summary: dict[str, Any]) -> str:
    warnings = summary.get("collapse_warnings", [])
    lines = [
        "# M3 Full Edge Shard Report",
        "",
        "This report covers one production full-M3 local transition edge shard.",
        "`row_normalized_transition_prob` is local to each source candidate set and is not a global Markov P row.",
        "No global Markov P, GPCCA, fate probability, Branched NicheFlow, M5, or regulator analysis was run.",
        "",
        f"- shard_id: {summary['shard_id']}",
        f"- status: {summary['status']}",
        f"- source_time: {summary['source_time']}",
        f"- target_time: {summary['target_time']}",
        f"- source_slice_id: {summary['source_slice_id']}",
        f"- source_rows: {summary['source_rows']}",
        f"- target_rows: {summary['target_rows']}",
        f"- candidate_k: {summary['candidate_k']}",
        f"- observed_edge_rows: {summary['observed_edge_rows']}",
        f"- backend: {summary['backend']}",
        f"- runtime_seconds: {summary['runtime_seconds']:.3f}",
        f"- max_rss_gib: {summary['max_rss_gib']:.4f}",
        f"- output_bytes: {summary['output_bytes']}",
        f"- row_sum_abs_error_max: {summary['row_sum_abs_error_max']:.6g}",
        f"- row_entropy_mean/median: {summary['row_entropy_mean']:.6g} / {summary['row_entropy_median']:.6g}",
        f"- top1_probability_mean/median: {summary['top1_probability_mean']:.6g} / {summary['top1_probability_median']:.6g}",
        f"- target_slice_id_entropy_mean: {summary['target_slice_id_entropy_mean']:.6g}",
        f"- target_mouse_id_entropy_mean: {summary['target_mouse_id_entropy_mean']:.6g}",
        f"- top_target_slice_id_fraction_p95: {summary['top_target_slice_id_fraction_p95']:.6g}",
        f"- top_target_mouse_id_fraction_p95: {summary['top_target_mouse_id_fraction_p95']:.6g}",
    ]
    if warnings:
        lines.extend(["", "## Warning-Only Collapse Diagnostics", *[f"- {warning}" for warning in warnings]])
    return "\n".join(lines).rstrip() + "\n"


def build_shard(
    shard: pd.Series,
    config: dict[str, Any],
    feature_groups: dict[str, Any],
    read_columns: list[str],
    backend: str,
    candidate_k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source, target = load_shard_data(config, shard, read_columns)
    run_config = configure_full_m3(config, backend, candidate_k)
    builder_shard = {
        "source_time": shard["source_time"],
        "target_time": shard["target_time"],
        "source_day": float(shard["source_day"]),
        "target_day": float(shard["target_day"]),
        "time_delta": float(shard["time_delta"]),
        "source_slice_id": shard["source_slice_id"],
        "source_slice_file": shard["source_slice_file"],
        "source_rows": int(shard["source_rows"]),
        "target_time_rows": int(shard["target_rows"]),
        "candidate_k": int(candidate_k),
        "expected_edge_rows": int(shard["expected_edge_rows"]),
    }
    return M3_05.build_pilot_edges(source, target, builder_shard, run_config, feature_groups)


def completed_record(shard: pd.Series, metrics: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "shard_id": shard["shard_id"],
        "source_time": shard["source_time"],
        "target_time": shard["target_time"],
        "source_slice_id": shard["source_slice_id"],
        "source_rows": int(shard["source_rows"]),
        "target_rows": int(shard["target_rows"]),
        "candidate_k": int(shard["candidate_k"]),
        "expected_edge_rows": int(shard["expected_edge_rows"]),
        "output_parquet": shard["output_parquet"],
        "shard_report": shard["shard_report"],
        "status": status,
        **metrics,
    }


def run_one_shard(
    shard: pd.Series,
    config: dict[str, Any],
    feature_groups: dict[str, Any],
    read_columns: list[str],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any] | None, str | None]:
    edge_path = Path(shard["output_parquet"])
    report_path = Path(shard["shard_report"])
    if args.resume and not args.overwrite:
        valid, metrics, reason = validate_existing_outputs(shard)
        if valid:
            record = completed_record(shard, metrics, "SKIPPED_VALID_EXISTING")
            return "skipped", record, None
        if edge_path.exists() or report_path.exists():
            print(f"RESUME_RERUN {shard['shard_id']} {reason}")
    elif (edge_path.exists() or report_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Output exists for {shard['shard_id']}; use --resume or --overwrite.")

    start = time.monotonic()
    frame, metadata = build_shard(shard, config, feature_groups, read_columns, args.backend, int(args.candidate_k))
    metrics = validate_edge_frame(frame, shard)
    atomic_write_parquet(frame, edge_path)
    metrics["output_bytes"] = int(edge_path.stat().st_size)
    metrics["runtime_seconds"] = float(time.monotonic() - start)
    metrics["max_rss_gib"] = max_rss_gib()
    metrics["backend"] = metadata["backend"]
    metrics["tau_pair"] = float(metadata["tau_pair"])
    record = completed_record(shard, metrics, "COMPLETED")
    atomic_write_text(report_path, shard_report_text(record))
    return "completed", record, None


def write_completed(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        pd.DataFrame(records).to_csv(path, index=False)
    else:
        pd.DataFrame().to_csv(path, index=False)


def write_failed(path: Path, failures: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(failures).rstrip() + ("\n" if failures else ""))


def aggregate_summary(records: list[dict[str, Any]], failures: list[str], plan: pd.DataFrame) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    grouped = (
        frame.groupby(["source_time", "target_time"], observed=True)
        .agg(
            shards=("shard_id", "count"),
            observed_edge_rows=("observed_edge_rows", "sum"),
            expected_edge_rows=("expected_edge_rows", "sum"),
            runtime_seconds=("runtime_seconds", "sum"),
            max_rss_gib=("max_rss_gib", "max"),
            output_bytes=("output_bytes", "sum"),
            row_sum_abs_error_max=("row_sum_abs_error_max", "max"),
            row_entropy_mean=("row_entropy_mean", "mean"),
            row_entropy_median=("row_entropy_median", "median"),
            top1_probability_mean=("top1_probability_mean", "mean"),
            top1_probability_median=("top1_probability_median", "median"),
            target_slice_id_entropy_mean=("target_slice_id_entropy_mean", "mean"),
            target_mouse_id_entropy_mean=("target_mouse_id_entropy_mean", "mean"),
            top_target_slice_id_fraction_p95=("top_target_slice_id_fraction_p95", "max"),
            top_target_mouse_id_fraction_p95=("top_target_mouse_id_fraction_p95", "max"),
        )
        .reset_index()
    )
    grouped["failed_shards"] = len(failures)
    grouped["planned_shards_total"] = len(plan)
    return grouped


def full_manifest(records: list[dict[str, Any]], failures: list[str], plan: pd.DataFrame) -> pd.DataFrame:
    if records:
        completed = pd.DataFrame(records)
        cols = [
            "shard_id",
            "source_time",
            "target_time",
            "source_slice_id",
            "status",
            "observed_edge_rows",
            "expected_edge_rows",
            "runtime_seconds",
            "max_rss_gib",
            "output_bytes",
            "output_parquet",
            "shard_report",
        ]
        manifest = completed[cols].copy()
    else:
        manifest = pd.DataFrame()
    if failures:
        failed_ids = [failure.split("\t", 1)[0] for failure in failures]
        failed_plan = plan[plan["shard_id"].isin(failed_ids)].copy()
        failed_plan["status"] = "FAILED"
        for column in [
            "observed_edge_rows",
            "runtime_seconds",
            "max_rss_gib",
            "output_bytes",
        ]:
            failed_plan[column] = np.nan
        manifest = pd.concat(
            [
                manifest,
                failed_plan[
                    [
                        "shard_id",
                        "source_time",
                        "target_time",
                        "source_slice_id",
                        "status",
                        "observed_edge_rows",
                        "expected_edge_rows",
                        "runtime_seconds",
                        "max_rss_gib",
                        "output_bytes",
                        "output_parquet",
                        "shard_report",
                    ]
                ],
            ],
            ignore_index=True,
        )
    return manifest.sort_values("shard_id").reset_index(drop=True) if len(manifest) else manifest


def generate_figures(figures_dir: Path, records: list[dict[str, Any]], summary: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if not records:
        return warnings
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(records)
        pair = frame["source_time"].astype(str) + "->" + frame["target_time"].astype(str)
        frame = frame.assign(time_pair=pair)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(frame["shard_id"], frame["runtime_seconds"])
        axes[0].tick_params(axis="x", rotation=90, labelsize=6)
        axes[0].set_title("Runtime by shard")
        axes[1].bar(frame["shard_id"], frame["max_rss_gib"])
        axes[1].tick_params(axis="x", rotation=90, labelsize=6)
        axes[1].set_title("Max RSS GiB by shard")
        fig.tight_layout()
        fig.savefig(figures_dir / "m3_full_runtime_memory_by_shard.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(summary["source_time"].astype(str) + "->" + summary["target_time"].astype(str), summary["observed_edge_rows"])
        ax.tick_params(axis="x", rotation=25)
        ax.set_title("Edge rows by time pair")
        fig.tight_layout()
        fig.savefig(figures_dir / "m3_full_edge_rows_by_time_pair.png", dpi=140)
        plt.close(fig)

        figure_specs = [
            ("row_entropy_mean", "m3_full_row_entropy_by_time_pair.png", "Row entropy mean"),
            ("top1_probability_mean", "m3_full_top1_probability_by_time_pair.png", "Top1 probability mean"),
            (
                "top_target_slice_id_fraction_p95",
                "m3_full_target_slice_collapse_by_time_pair.png",
                "Target slice top fraction p95",
            ),
            (
                "top_target_mouse_id_fraction_p95",
                "m3_full_target_mouse_collapse_by_time_pair.png",
                "Target mouse top fraction p95",
            ),
        ]
        labels = summary["source_time"].astype(str) + "->" + summary["target_time"].astype(str)
        for column, name, title in figure_specs:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(labels, summary[column])
            ax.tick_params(axis="x", rotation=25)
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(figures_dir / name, dpi=140)
            plt.close(fig)

        heat_cols = [
            "row_sum_abs_error_max",
            "row_entropy_mean",
            "top1_probability_mean",
            "top_target_slice_id_fraction_p95",
            "top_target_mouse_id_fraction_p95",
        ]
        fig, ax = plt.subplots(figsize=(8, max(5, len(frame) * 0.15)))
        values = frame[heat_cols].to_numpy(dtype=float)
        ax.imshow(values, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(frame)))
        ax.set_yticklabels(frame["source_slice_id"], fontsize=5)
        ax.set_xticks(range(len(heat_cols)))
        ax.set_xticklabels(heat_cols, rotation=35, ha="right")
        ax.set_title("Source-slice QC heatmap")
        fig.tight_layout()
        fig.savefig(figures_dir / "m3_full_source_slice_qc_heatmap.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].bar(labels, summary["observed_edge_rows"])
        axes[0, 0].tick_params(axis="x", rotation=25)
        axes[0, 0].set_title("Edge rows")
        axes[0, 1].bar(labels, summary["runtime_seconds"])
        axes[0, 1].tick_params(axis="x", rotation=25)
        axes[0, 1].set_title("Runtime seconds")
        axes[1, 0].bar(labels, summary["row_entropy_mean"])
        axes[1, 0].tick_params(axis="x", rotation=25)
        axes[1, 0].set_title("Row entropy")
        axes[1, 1].bar(labels, summary["top1_probability_mean"])
        axes[1, 1].tick_params(axis="x", rotation=25)
        axes[1, 1].set_title("Top1 probability")
        fig.tight_layout()
        fig.savefig(figures_dir / "m3_full_summary_dashboard.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Figure generation failed but edge construction/QC completed: {exc}")
    return warnings


def run_summary_report(
    records: list[dict[str, Any]],
    failures: list[str],
    summary: pd.DataFrame,
    figure_warnings: list[str],
    total_runtime: float,
) -> str:
    completed = [record for record in records if record["status"] in {"COMPLETED", "SKIPPED_VALID_EXISTING"}]
    total_rows = int(sum(record["observed_edge_rows"] for record in completed)) if completed else 0
    total_bytes = int(sum(record["output_bytes"] for record in completed)) if completed else 0
    peak_rss = max((float(record["max_rss_gib"]) for record in completed), default=0.0)
    lines = [
        "# M3 Full-M3 Edge-Shard Run Summary",
        "",
        "This run created local transition edge shards only. `row_normalized_transition_prob` is local to each source candidate set, not a global Markov P row.",
        "No global Markov P, GPCCA, fate probability, Branched NicheFlow, M5, or regulator analysis was run.",
        "",
        f"- completed/skipped valid shards: {len(completed)}",
        f"- failed shards: {len(failures)}",
        f"- total observed edge rows: {total_rows}",
        f"- total runtime seconds: {total_runtime:.3f}",
        f"- peak max RSS GiB: {peak_rss:.4f}",
        f"- total output bytes: {total_bytes}",
        "",
        "## Per-Time-Pair Rows",
    ]
    if len(summary):
        for row in summary.to_dict("records"):
            lines.append(
                "- "
                f"{row['source_time']}->{row['target_time']}: "
                f"shards={int(row['shards'])}, rows={int(row['observed_edge_rows'])}, "
                f"runtime_seconds={row['runtime_seconds']:.3f}, "
                f"row_entropy_mean={row['row_entropy_mean']:.6g}, "
                f"top1_probability_mean={row['top1_probability_mean']:.6g}, "
                f"slice_top_fraction_p95={row['top_target_slice_id_fraction_p95']:.6g}, "
                f"mouse_top_fraction_p95={row['top_target_mouse_id_fraction_p95']:.6g}"
            )
    collapse = []
    for record in completed:
        for warning in record.get("collapse_warnings", []):
            collapse.append(f"{record['shard_id']}: {warning}")
    if collapse:
        lines.extend(["", "## Warning-Only Collapse Diagnostics", *[f"- {item}" for item in collapse]])
    if figure_warnings:
        lines.extend(["", "## Figure Warnings", *[f"- {item}" for item in figure_warnings]])
    if failures:
        lines.extend(["", "## Failures", *[f"- {failure}" for failure in failures]])
    return "\n".join(lines).rstrip() + "\n"


def write_run_outputs(
    output_root: Path,
    reports_dir: Path,
    records: list[dict[str, Any]],
    failures: list[str],
    plan: pd.DataFrame,
    total_runtime: float,
) -> dict[str, Path]:
    completed_csv = output_root / "completed_shards.csv"
    failed_txt = output_root / "failed_shards.txt"
    manifest_csv = output_root / "full_m3_manifest.csv"
    manifest_json = output_root / "full_m3_manifest.json"
    summary_csv = reports_dir / "m3_full_m3_run_summary.csv"
    summary_md = reports_dir / "m3_full_m3_run_summary.md"
    figures_dir = reports_dir / "figures" / "full_m3"

    write_completed(completed_csv, records)
    write_failed(failed_txt, failures)
    manifest = full_manifest(records, failures, plan)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_csv, index=False)
    manifest_json.write_text(
        json.dumps(
            json_safe(
                {
                    "records": manifest.to_dict("records"),
                    "failures": failures,
                    **NO_DOWNSTREAM_FLAGS,
                }
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = aggregate_summary(records, failures, plan)
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    figure_warnings = generate_figures(figures_dir, records, summary)
    atomic_write_text(summary_md, run_summary_report(records, failures, summary, figure_warnings, total_runtime))
    return {
        "completed_shards": completed_csv,
        "failed_shards": failed_txt,
        "manifest_csv": manifest_csv,
        "manifest_json": manifest_json,
        "summary_csv": summary_csv,
        "summary_md": summary_md,
        "figures_dir": figures_dir,
    }


def dry_run_report(plan: pd.DataFrame, selected: pd.DataFrame, args: argparse.Namespace) -> None:
    print("M3_FULL_RUNNER_DRY_RUN")
    print(f"PLAN_SHARDS {len(plan)}")
    print(f"SELECTED_SHARDS {len(selected)}")
    print(f"EXPECTED_TOTAL_EDGE_ROWS {int(plan['expected_edge_rows'].sum())}")
    print(f"SELECTED_EXPECTED_EDGE_ROWS {int(selected['expected_edge_rows'].sum()) if len(selected) else 0}")
    print(f"OUTPUT_ROOT {args.output_root}")
    print(f"BACKEND {args.backend}")
    print(f"CANDIDATE_K {args.candidate_k}")
    print("NO_GLOBAL_MARKOV_P True")


def run(args: argparse.Namespace) -> int:
    apply_runtime_threads(int(args.blas_threads))
    start = time.monotonic()
    config = load_config(args.config)
    expected_outputs = load_expected_outputs(args.expected_outputs_json)
    validate_scope(args, config, expected_outputs)
    plan = load_plan(args.plan_csv)
    selected = filter_plan(plan, args)
    if args.dry_run:
        dry_run_report(plan, selected, args)
        return 0

    feature_groups, read_columns = feature_columns(config)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for _, shard in selected.iterrows():
        print(f"M3_FULL_SHARD_START {shard['shard_id']} {shard['source_time']}->{shard['target_time']} {shard['source_slice_id']}")
        try:
            status, record, failure = run_one_shard(shard, config, feature_groups, read_columns, args)
            if record is not None:
                records.append(record)
                print(
                    f"M3_FULL_SHARD_{status.upper()} {shard['shard_id']} "
                    f"ROWS {record['observed_edge_rows']} RUNTIME {record.get('runtime_seconds', 0):.3f}"
                )
            if failure:
                failures.append(f"{shard['shard_id']}\t{failure}")
        except Exception as exc:  # noqa: BLE001
            message = f"{shard['shard_id']}\t{type(exc).__name__}: {exc}"
            failures.append(message)
            print(f"M3_FULL_SHARD_FAILED {message}")
            if args.stop_on_error:
                break

    total_runtime = time.monotonic() - start
    reports_dir = Path(config["paths"]["reports_dir"])
    outputs = write_run_outputs(args.output_root, reports_dir, records, failures, plan, total_runtime)
    completed_count = len([r for r in records if r["status"] in {"COMPLETED", "SKIPPED_VALID_EXISTING"}])
    observed_rows = int(sum(r["observed_edge_rows"] for r in records))
    print("M3_FULL_RUN_COMPLETED" if not failures else "M3_FULL_RUN_COMPLETED_WITH_FAILURES")
    print(f"COMPLETED_OR_VALID_EXISTING_SHARDS {completed_count}")
    print(f"FAILED_SHARDS {len(failures)}")
    print(f"OBSERVED_EDGE_ROWS {observed_rows}")
    print(f"EXPECTED_TOTAL_EDGE_ROWS {EXPECTED_TOTAL_EDGE_ROWS}")
    print(f"TOTAL_RUNTIME_SECONDS {total_runtime:.3f}")
    print(f"PEAK_MAX_RSS_GIB {max((float(r['max_rss_gib']) for r in records), default=0.0):.4f}")
    print(f"SUMMARY {outputs['summary_md']}")
    if failures:
        return 1
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
