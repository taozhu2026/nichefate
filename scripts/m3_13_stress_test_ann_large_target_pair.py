#!/usr/bin/env python
"""Run the M3-13 D3->D9 large-target ANN stress validation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
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
from nichefate.transition import CandidateNeighborBackendStatus, inspect_candidate_neighbor_backend


DEFAULT_SOURCE_TIME = "D3"
DEFAULT_TARGET_TIME = "D9"
DEFAULT_SOURCE_SLICE_ID = "092421_D3_m3_1_slice_3"
DEFAULT_SOURCE_ROWS = 21_962
DEFAULT_TARGET_ROWS = 660_977
DEFAULT_EXPECTED_FULL_ROWS = 658_860
DEFAULT_SAMPLE_SIZE = 3_000
MAX_SAMPLE_SIZE_WITHOUT_OVERRIDE = 5_000
DEFAULT_CANDIDATE_K = 30
DEFAULT_RANDOM_SEED = 1
DEFAULT_PLAN_CSV = Path("/home/zhutao/scratch/nichefate/m3/reports/m3_full_transition_shards.csv")
DEFAULT_OUTPUT_DIR = Path("/home/zhutao/scratch/nichefate/m3/ann_stress_D3_to_D9")
DEFAULT_MAX_EXACT_DENSE_MEMORY_GIB = 80.0

OUTPUT_TOKENS = [
    "global_markov",
    "markov_p",
    "gpcca",
    "fate",
    "branched",
    "nicheflow",
    "m5",
    "regulator",
]

NO_DOWNSTREAM_FLAGS = {
    "no_production_m3_edges": True,
    "no_full_m3": True,
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


M3_11 = _load_script_module(
    "m3_11_validate_ann_backend_on_sampled_shard",
    PROJECT_ROOT / "scripts" / "m3_11_validate_ann_backend_on_sampled_shard.py",
)


@dataclass(frozen=True)
class ExactReferenceSafety:
    """Decision from the sklearn_exact reference safety guard."""

    should_run: bool
    planned_source_sample_size: int
    actual_source_sample_size: int
    target_rows: int
    pairwise_distance_evaluations: int
    estimated_dense_distance_gib: float
    max_dense_distance_gib: float
    fallback_sample_used: bool
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_transition_kernel.yaml")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--source-time", default=DEFAULT_SOURCE_TIME)
    parser.add_argument("--target-time", default=DEFAULT_TARGET_TIME)
    parser.add_argument("--source-slice-id", default=DEFAULT_SOURCE_SLICE_ID)
    parser.add_argument("--sample-source-anchors", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--exact-backend", default="sklearn_exact")
    parser.add_argument("--ann-backend", default="pynndescent")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--allow-larger-sample", action="store_true")
    parser.add_argument("--allow-non-default-pair", action="store_true")
    parser.add_argument("--allow-non-default-shard", action="store_true")
    parser.add_argument("--allow-non-default-output-dir", action="store_true")
    parser.add_argument("--fallback-source-anchors", type=int, default=None)
    parser.add_argument("--max-exact-dense-memory-gib", type=float, default=None)
    parser.add_argument("--write-candidate-tables", action="store_true")
    return parser.parse_args()


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "report": output_dir / "ann_stress_report_D3_to_D9.md",
        "metrics": output_dir / "ann_stress_metrics_D3_to_D9.csv",
        "summary": output_dir / "ann_stress_summary_D3_to_D9.json",
        "overlap": output_dir / "ann_stress_candidate_overlap_D3_to_D9.csv",
        "exact_candidates": output_dir / "validation_only_sklearn_exact_candidates_D3_to_D9.parquet",
        "ann_candidates": output_dir / "ann_stress_pynndescent_candidates_D3_to_D9.parquet",
        "figures_dir": output_dir / "figures",
    }


def _assert_no_ssd(config: dict[str, Any]) -> None:
    if bool(config["paths"].get("use_ssd", False)):
        raise RuntimeError("Refusing M3-13 ANN stress validation while paths.use_ssd is true.")
    for value in config.get("paths", {}).values():
        if isinstance(value, str) and value.startswith("/ssd"):
            raise RuntimeError(f"Refusing to use /ssd path in M3-13 ANN stress validation: {value}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def ensure_validation_output_dir(output_dir: Path, config: dict[str, Any]) -> None:
    resolved = output_dir.resolve()
    if "/ssd" in str(resolved):
        raise ValueError(f"Refusing to write M3-13 stress outputs under /ssd: {resolved}")
    production_roots = [
        Path(config["full_m3"]["output_root"]).resolve(),
        Path(config["paths"]["m3_output_dir"]).resolve() / "by_pair",
        Path(config["paths"]["m3_output_dir"]).resolve() / "full",
    ]
    for root in production_roots:
        if _is_relative_to(resolved, root):
            raise ValueError(f"Refusing to write M3-13 stress outputs under production M3 directory: {root}")
    lower = " ".join(part.lower() for part in resolved.parts if part.lower() != "nichefate")
    for token in OUTPUT_TOKENS:
        if token in lower:
            raise ValueError(f"Refusing downstream-looking M3-13 stress output path containing {token!r}.")


def validate_requested_scope(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if (
        not args.allow_non_default_pair
        and (str(args.source_time) != DEFAULT_SOURCE_TIME or str(args.target_time) != DEFAULT_TARGET_TIME)
    ):
        raise ValueError("This stress stage is scoped to D3->D9; pass --allow-non-default-pair to override.")
    if not args.allow_non_default_shard and str(args.source_slice_id) != DEFAULT_SOURCE_SLICE_ID:
        raise ValueError(
            "This stress stage is scoped to source slice "
            f"{DEFAULT_SOURCE_SLICE_ID}; pass --allow-non-default-shard to override."
        )
    if int(args.sample_source_anchors) > MAX_SAMPLE_SIZE_WITHOUT_OVERRIDE and not args.allow_larger_sample:
        raise ValueError("Refusing source sample size >5000 without --allow-larger-sample.")
    if int(args.sample_source_anchors) <= 0:
        raise ValueError("--sample-source-anchors must be positive.")
    if args.fallback_source_anchors is not None:
        if int(args.fallback_source_anchors) <= 0:
            raise ValueError("--fallback-source-anchors must be positive when provided.")
        if int(args.fallback_source_anchors) >= int(args.sample_source_anchors):
            raise ValueError("--fallback-source-anchors must be smaller than --sample-source-anchors.")
        if int(args.fallback_source_anchors) > MAX_SAMPLE_SIZE_WITHOUT_OVERRIDE and not args.allow_larger_sample:
            raise ValueError("Refusing fallback source sample size >5000 without --allow-larger-sample.")
    if int(args.candidate_k) != DEFAULT_CANDIDATE_K:
        raise ValueError("This stress stage is scoped to candidate_k=30.")
    if args.exact_backend != "sklearn_exact":
        raise ValueError("This stress stage uses sklearn_exact as the bounded exact reference.")
    if args.ann_backend != "pynndescent":
        raise ValueError("This stress stage is scoped to pynndescent as the ANN backend.")
    if (
        not args.allow_non_default_output_dir
        and args.output_dir.resolve() != DEFAULT_OUTPUT_DIR.resolve()
    ):
        raise ValueError(f"M3-13 stress outputs must stay under {DEFAULT_OUTPUT_DIR}.")
    _assert_no_ssd(config)
    ensure_validation_output_dir(args.output_dir, config)


def select_stress_shard(
    shards: pd.DataFrame,
    source_time: str,
    target_time: str,
    source_slice_id: str,
) -> dict[str, Any]:
    shard = M3_11.select_validation_shard(shards, source_time, target_time, source_slice_id)
    if str(source_time) == DEFAULT_SOURCE_TIME and str(target_time) == DEFAULT_TARGET_TIME:
        checks = {
            "source_rows": DEFAULT_SOURCE_ROWS,
            "target_time_rows": DEFAULT_TARGET_ROWS,
            "expected_edge_rows": DEFAULT_EXPECTED_FULL_ROWS,
            "candidate_k": DEFAULT_CANDIDATE_K,
        }
        for key, expected in checks.items():
            if int(shard[key]) != int(expected):
                raise ValueError(f"D3->D9 stress shard {key} {shard[key]} != expected {expected}.")
    return shard


def dense_distance_gib(source_rows: int, target_rows: int) -> float:
    return float(int(source_rows) * int(target_rows) * 8) / float(1024**3)


def exact_reference_safety_guard(
    planned_source_sample_size: int,
    target_rows: int,
    max_dense_distance_gib: float,
    fallback_source_anchors: int | None = None,
) -> ExactReferenceSafety:
    planned_pairs = int(planned_source_sample_size) * int(target_rows)
    planned_gib = dense_distance_gib(planned_source_sample_size, target_rows)
    if planned_gib <= float(max_dense_distance_gib):
        return ExactReferenceSafety(
            should_run=True,
            planned_source_sample_size=int(planned_source_sample_size),
            actual_source_sample_size=int(planned_source_sample_size),
            target_rows=int(target_rows),
            pairwise_distance_evaluations=planned_pairs,
            estimated_dense_distance_gib=planned_gib,
            max_dense_distance_gib=float(max_dense_distance_gib),
            fallback_sample_used=False,
            reason="planned sklearn_exact reference is within the configured dense-memory guard",
        )
    if fallback_source_anchors is not None:
        fallback_gib = dense_distance_gib(int(fallback_source_anchors), target_rows)
        if fallback_gib <= float(max_dense_distance_gib):
            return ExactReferenceSafety(
                should_run=True,
                planned_source_sample_size=int(planned_source_sample_size),
                actual_source_sample_size=int(fallback_source_anchors),
                target_rows=int(target_rows),
                pairwise_distance_evaluations=int(fallback_source_anchors) * int(target_rows),
                estimated_dense_distance_gib=fallback_gib,
                max_dense_distance_gib=float(max_dense_distance_gib),
                fallback_sample_used=True,
                reason="explicit smaller fallback sample is within the configured dense-memory guard",
            )
    return ExactReferenceSafety(
        should_run=False,
        planned_source_sample_size=int(planned_source_sample_size),
        actual_source_sample_size=0,
        target_rows=int(target_rows),
        pairwise_distance_evaluations=planned_pairs,
        estimated_dense_distance_gib=planned_gib,
        max_dense_distance_gib=float(max_dense_distance_gib),
        fallback_sample_used=False,
        reason=(
            "sklearn_exact reference would exceed the configured dense-memory guard; "
            "resources were not increased"
        ),
    )


def max_exact_dense_memory_gib(config: dict[str, Any], override: float | None) -> float:
    if override is not None:
        return float(override)
    return float(config.get("full_m3", {}).get("max_memory_gb_warning", DEFAULT_MAX_EXACT_DENSE_MEMORY_GIB))


def base_context(
    args: argparse.Namespace,
    shard: dict[str, Any] | None,
    safety: ExactReferenceSafety | None,
) -> dict[str, Any]:
    context = {
        "source_time": args.source_time,
        "target_time": args.target_time,
        "source_slice_id": args.source_slice_id,
        "source_slice_rationale": "median-sized D3->D9 source shard among 13 shards",
        "planned_source_sample_size": int(args.sample_source_anchors),
        "actual_source_sample_size": int(safety.actual_source_sample_size) if safety else 0,
        "fallback_sample_used": bool(safety.fallback_sample_used) if safety else False,
        "candidate_k": int(args.candidate_k),
        "random_seed": int(args.random_seed),
        "exact_backend": args.exact_backend,
        "ann_backend": args.ann_backend,
    }
    if shard is not None:
        context.update(
            {
                "source_slice_file": shard["source_slice_file"],
                "source_rows_full_shard": int(shard["source_rows"]),
                "target_rows": int(shard["target_time_rows"]),
                "expected_full_shard_rows": int(shard["expected_edge_rows"]),
            }
        )
    if safety is not None:
        context.update(asdict(safety))
    return context


def write_stop_outputs(
    paths: dict[str, Path],
    status: str,
    reason: str,
    context: dict[str, Any],
    backend_status: CandidateNeighborBackendStatus | None = None,
    figure_warnings: list[str] | None = None,
) -> None:
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([{**context, "status": status, "reason": reason}])
    metrics.to_csv(paths["metrics"], index=False)
    pd.DataFrame(columns=["source_anchor_id", "recall_at_k", "jaccard_overlap", "top1_agreement"]).to_csv(
        paths["overlap"],
        index=False,
    )
    summary = {
        "status": status,
        "reason": reason,
        "metrics": M3_11.json_safe(metrics.iloc[0].to_dict()),
        "backend_status": asdict(backend_status) if backend_status else None,
        "figure_warnings": figure_warnings or [],
        "outputs": {key: str(value) for key, value in paths.items()},
        **NO_DOWNSTREAM_FLAGS,
    }
    paths["summary"].write_text(json.dumps(M3_11.json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["report"].write_text(
        "\n".join(
            [
                f"# M3-13 D3->D9 Large-Target ANN Stress Test {status.title()}",
                "",
                f"- Reason: {reason}",
                f"- planned source sample size: {context.get('planned_source_sample_size')}",
                f"- actual source sample size: {context.get('actual_source_sample_size')}",
                f"- fallback_sample_used: {context.get('fallback_sample_used')}",
                f"- target rows: {context.get('target_rows')}",
                f"- candidate_k: {context.get('candidate_k')}",
                "- No package installation or conda modification was attempted.",
                "- No production M3 edges, full M3, global Markov P, GPCCA, fate probability, Branched NicheFlow, M5, or regulator analysis was run.",
            ]
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )


def add_stress_soft_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    row = result.iloc[0]
    result.loc[0, "soft_jaccard_pass"] = bool(row["jaccard_overlap_mean"] >= 0.7)
    result.loc[0, "soft_probability_drift_ok"] = bool(
        row["row_normalized_transition_prob_abs_drift_p95"] <= 0.05
    )
    result.loc[0, "soft_validation_pass"] = bool(
        row["soft_recall_pass"]
        and row["soft_top1_pass"]
        and result.loc[0, "soft_jaccard_pass"]
        and result.loc[0, "soft_probability_drift_ok"]
        and row["soft_row_entropy_shift_ok"]
        and row["soft_top1_probability_shift_ok"]
        and row["soft_target_collapse_shift_ok"]
    )
    return result


def generate_figures(
    figures_dir: Path,
    overlap: pd.DataFrame,
    drift: pd.DataFrame,
    row_diag: pd.DataFrame,
    metrics: pd.DataFrame,
    force_failure: bool = False,
) -> list[str]:
    warnings: list[str] = []
    try:
        if force_failure:
            raise RuntimeError("forced figure failure")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir.mkdir(parents=True, exist_ok=True)
        row = metrics.iloc[0]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(overlap["recall_at_k"].dropna(), bins=30)
        ax.axvline(0.8, color="red", linestyle="--")
        ax.set_title("M3-13 D3->D9 Recall@30")
        ax.set_xlabel("recall@30")
        ax.set_ylabel("source anchors")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_stress_recall_distribution.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(drift["row_normalized_transition_prob_abs_drift"].dropna(), bins=40)
        ax.axvline(0.05, color="red", linestyle="--")
        ax.set_title("M3-13 Probability Drift")
        ax.set_xlabel("absolute drift")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_stress_probability_drift.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(row_diag["row_entropy_delta"].dropna(), bins=40)
        ax.set_title("M3-13 Row Entropy Delta")
        ax.set_xlabel("pynndescent - sklearn_exact")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_stress_entropy_delta.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].bar(
            ["sklearn_exact", "pynndescent"],
            [row["sklearn_exact_runtime_seconds"], row["pynndescent_runtime_seconds"]],
        )
        axes[0].set_title("Runtime seconds")
        axes[1].bar(
            ["sklearn_exact", "pynndescent"],
            [row["sklearn_exact_max_rss_gib"], row["pynndescent_max_rss_gib"]],
        )
        axes[1].set_title("Max RSS GiB")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_stress_runtime_memory.png", dpi=140)
        plt.close(fig)

        labels = ["slice entropy", "mouse entropy", "top slice", "top mouse"]
        values = [
            row["target_slice_id_entropy_delta_mean"],
            row["target_mouse_id_entropy_delta_mean"],
            row["top_target_slice_id_fraction_delta_mean"],
            row["top_target_mouse_id_fraction_delta_mean"],
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=25)
        ax.set_title("M3-13 Target Slice/Mouse Delta")
        fig.tight_layout()
        fig.savefig(figures_dir / "ann_stress_target_slice_mouse_delta.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Figure generation failed but stress reports/tables were written: {exc}")
    return warnings


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    figure_warnings: list[str],
    exact_metadata: dict[str, Any],
    ann_metadata: dict[str, Any],
) -> None:
    row = metrics.iloc[0].to_dict()
    lines = [
        "# M3-13 D3->D9 Large-Target ANN Stress Test",
        "",
        "This is a validation-only stress output. It is not a production M3 edge shard and does not assemble global Markov P, run GPCCA, compute fate probabilities, run Branched NicheFlow, M5, or regulator analysis.",
        "",
        "## Selected Shard",
        f"- source_time: {row['source_time']}",
        f"- target_time: {row['target_time']}",
        f"- source_slice_id: {row['source_slice_id']}",
        f"- rationale: {row['source_slice_rationale']}",
        f"- full source rows: {int(row['source_rows_full_shard'])}",
        f"- D9 target rows: {int(row['target_rows'])}",
        f"- expected full-shard rows: {int(row['expected_full_shard_rows'])}",
        f"- planned source sample size: {int(row['planned_source_sample_size'])}",
        f"- actual source sample size: {int(row['actual_source_sample_size'])}",
        f"- fallback_sample_used: {bool(row['fallback_sample_used'])}",
        f"- candidate_k: {int(row['candidate_k'])}",
        "",
        "## Candidate Retrieval",
        f"- recall@30 mean/median/p05/p95: {row['recall_at_30_mean']:.6g} / {row['recall_at_30_median']:.6g} / {row['recall_at_30_p05']:.6g} / {row['recall_at_30_p95']:.6g}",
        f"- top1 agreement: {row['top1_agreement']:.6g}",
        f"- mean Jaccard overlap: {row['jaccard_overlap_mean']:.6g}",
        f"- median Jaccard overlap: {row['jaccard_overlap_median']:.6g}",
        f"- mean distance-rank correlation: {row['distance_rank_correlation_mean']:.6g}",
        "",
        "## Probability And Diagnostic Drift",
        f"- row_normalized_transition_prob abs drift mean/median/p95: {row['row_normalized_transition_prob_abs_drift_mean']:.6g} / {row['row_normalized_transition_prob_abs_drift_median']:.6g} / {row['row_normalized_transition_prob_abs_drift_p95']:.6g}",
        f"- row entropy exact/ANN/delta mean: {row['row_entropy_exact_mean']:.6g} / {row['row_entropy_ann_mean']:.6g} / {row['row_entropy_delta_mean']:.6g}",
        f"- top1 probability exact/ANN/delta mean: {row['top1_probability_exact_mean']:.6g} / {row['top1_probability_ann_mean']:.6g} / {row['top1_probability_delta_mean']:.6g}",
        f"- target slice entropy delta mean: {row['target_slice_id_entropy_delta_mean']:.6g}",
        f"- target mouse entropy delta mean: {row['target_mouse_id_entropy_delta_mean']:.6g}",
        f"- top target slice fraction delta mean: {row['top_target_slice_id_fraction_delta_mean']:.6g}",
        f"- top target mouse fraction delta mean: {row['top_target_mouse_id_fraction_delta_mean']:.6g}",
        "",
        "## Runtime And Memory",
        f"- sklearn_exact runtime seconds: {row['sklearn_exact_runtime_seconds']:.3f}",
        f"- pynndescent runtime seconds: {row['pynndescent_runtime_seconds']:.3f}",
        f"- runtime ratio ANN/exact: {row['runtime_ratio_ann_over_exact']:.6g}",
        f"- sklearn_exact max RSS GiB: {row['sklearn_exact_max_rss_gib']:.4f}",
        f"- pynndescent max RSS GiB: {row['pynndescent_max_rss_gib']:.4f}",
        f"- memory ratio ANN/exact: {row['memory_ratio_ann_over_exact']:.6g}",
        "",
        "## Soft Diagnostic Thresholds",
        f"- recall@30 >= 0.8: {bool(row['soft_recall_pass'])}",
        f"- top1 agreement >= 0.8: {bool(row['soft_top1_pass'])}",
        f"- mean Jaccard overlap >= 0.7: {bool(row['soft_jaccard_pass'])}",
        f"- probability drift p95 <= 0.05: {bool(row['soft_probability_drift_ok'])}",
        f"- row entropy shift ok: {bool(row['soft_row_entropy_shift_ok'])}",
        f"- top1 probability shift ok: {bool(row['soft_top1_probability_shift_ok'])}",
        f"- target collapse shift ok: {bool(row['soft_target_collapse_shift_ok'])}",
        f"- overall soft validation pass: {bool(row['soft_validation_pass'])}",
        "",
        "## Backend Metadata",
        f"- exact backend tau_pair: {exact_metadata.get('tau_pair')}",
        f"- ANN backend tau_pair: {ann_metadata.get('tau_pair')}",
        "",
        "## Explicit Non-Execution Confirmation",
        "- No production M3 edges were created.",
        "- Full M3 was not run.",
        "- Global Markov P was not assembled.",
        "- GPCCA, fate probability, Branched NicheFlow, M5, and regulator analysis were not run.",
    ]
    if figure_warnings:
        lines.extend(["", "## Figure Warnings", *[f"- {warning}" for warning in figure_warnings]])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_validation(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    validate_requested_scope(args, config)
    paths = output_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ann_status = inspect_candidate_neighbor_backend(args.ann_backend, run_toy_check=True)
    if not ann_status.available:
        context = base_context(args, None, None)
        reason = f"{args.ann_backend} is not usable: {ann_status.reason}"
        write_stop_outputs(paths, "SKIPPED", reason, context, ann_status)
        print(f"ANN_STRESS_SKIPPED {reason}")
        return 0

    shard = select_stress_shard(
        pd.read_csv(args.plan_csv),
        args.source_time,
        args.target_time,
        args.source_slice_id,
    )
    max_gib = max_exact_dense_memory_gib(config, args.max_exact_dense_memory_gib)
    safety = exact_reference_safety_guard(
        int(args.sample_source_anchors),
        int(shard["target_time_rows"]),
        max_gib,
        args.fallback_source_anchors,
    )
    if not safety.should_run:
        context = base_context(args, shard, safety)
        write_stop_outputs(paths, "SKIPPED", safety.reason, context, ann_status)
        print(f"ANN_STRESS_SKIPPED {safety.reason}")
        return 0

    source, target, feature_groups, retrieval_columns, _ = M3_11.load_validation_data(
        config,
        shard,
        args.source_time,
        args.target_time,
        int(safety.actual_source_sample_size),
        int(args.random_seed),
    )
    if len(source) != int(safety.actual_source_sample_size):
        raise ValueError(f"Expected {safety.actual_source_sample_size} sampled source anchors, found {len(source)}.")
    if len(target) != int(shard["target_time_rows"]):
        raise ValueError(f"Expected {shard['target_time_rows']} target rows, found {len(target)}.")

    source_retrieval, target_retrieval, standardize_stats = M3_11.standardize_feature_matrices(
        source[retrieval_columns].to_numpy(dtype=float),
        target[retrieval_columns].to_numpy(dtype=float),
        float(config["cost"]["min_scale"]),
    )
    metric = config["candidate_edges"].get("retrieval_metric", "euclidean")
    chunk_size = int(config["candidate_edges"].get("numpy_chunk_size", 512))
    context = base_context(args, shard, safety)
    context.update(
        {
            "source_sample_size": len(source),
            "target_rows": len(target),
            "expected_validation_edge_rows": len(source) * int(args.candidate_k),
            "retrieval_feature_columns": len(retrieval_columns),
            "zero_variance_retrieval_columns": standardize_stats["zero_variance_columns"],
        }
    )

    try:
        exact_neighbors, exact_timing = M3_11.run_backend(
            source_retrieval,
            target_retrieval,
            args.exact_backend,
            metric,
            int(args.candidate_k),
            chunk_size,
            int(args.random_seed),
        )
    except MemoryError as exc:
        reason = f"sklearn_exact failed with MemoryError: {exc}"
        write_stop_outputs(paths, "FAILED", reason, context, ann_status)
        print(f"ANN_STRESS_FAILED {reason}")
        return 1
    if isinstance(exact_neighbors, CandidateNeighborBackendStatus):
        reason = f"Exact backend unavailable: {exact_neighbors.reason}"
        write_stop_outputs(paths, "FAILED", reason, context, exact_neighbors)
        print(f"ANN_STRESS_FAILED {reason}")
        return 1

    ann_neighbors, ann_timing = M3_11.run_backend(
        source_retrieval,
        target_retrieval,
        args.ann_backend,
        metric,
        int(args.candidate_k),
        chunk_size,
        int(args.random_seed),
    )
    if isinstance(ann_neighbors, CandidateNeighborBackendStatus):
        write_stop_outputs(paths, "SKIPPED", ann_neighbors.reason, context, ann_neighbors)
        print(f"ANN_STRESS_SKIPPED {ann_neighbors.reason}")
        return 0

    validation_shard = dict(shard)
    validation_shard["source_rows"] = len(source)
    validation_shard["candidate_k"] = int(args.candidate_k)
    validation_shard["expected_edge_rows"] = len(source) * int(args.candidate_k)
    exact_edges, exact_metadata = M3_11.build_validation_edges(
        source,
        target,
        validation_shard,
        config,
        feature_groups,
        exact_neighbors,
    )
    ann_edges, ann_metadata = M3_11.build_validation_edges(
        source,
        target,
        validation_shard,
        config,
        feature_groups,
        ann_neighbors,
    )
    overlap = M3_11.compare_candidate_sets(exact_neighbors, ann_neighbors, source)
    drift = M3_11.drift_metrics(exact_edges, ann_edges)
    row_diag = M3_11.compare_row_diagnostics(exact_edges, ann_edges)
    metrics = M3_11.build_metrics(
        overlap,
        exact_neighbors,
        ann_neighbors,
        drift,
        row_diag,
        exact_timing,
        ann_timing,
        context,
    )
    metrics = add_stress_soft_metrics(metrics)

    metrics.to_csv(paths["metrics"], index=False)
    overlap.to_csv(paths["overlap"], index=False)
    if args.write_candidate_tables:
        M3_11.candidate_table(source, target, exact_neighbors).to_parquet(paths["exact_candidates"], index=False)
        M3_11.candidate_table(source, target, ann_neighbors).to_parquet(paths["ann_candidates"], index=False)
    figure_warnings = generate_figures(paths["figures_dir"], overlap, drift, row_diag, metrics)
    write_report(paths["report"], metrics, figure_warnings, exact_metadata, ann_metadata)
    summary = {
        "status": "COMPLETED",
        "metrics": metrics.iloc[0].to_dict(),
        "ann_backend_status": asdict(ann_status),
        "figure_warnings": figure_warnings,
        "outputs": {key: str(value) for key, value in paths.items()},
        **NO_DOWNSTREAM_FLAGS,
    }
    paths["summary"].write_text(json.dumps(M3_11.json_safe(summary), indent=2) + "\n", encoding="utf-8")
    row = metrics.iloc[0]
    print(f"ANN_STRESS_COMPLETED {args.source_time}->{args.target_time} {args.source_slice_id}")
    print(f"SOURCE_SAMPLE_SIZE {len(source)}")
    print(f"TARGET_ROWS {len(target)}")
    print(f"CANDIDATE_K {int(args.candidate_k)}")
    print(f"RECALL_AT_30_MEAN {row['recall_at_30_mean']:.6g}")
    print(f"TOP1_AGREEMENT {row['top1_agreement']:.6g}")
    print(f"JACCARD_OVERLAP_MEAN {row['jaccard_overlap_mean']:.6g}")
    print(f"SOFT_VALIDATION_PASS {bool(row['soft_validation_pass'])}")
    return 0


def main() -> int:
    return run_validation(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
