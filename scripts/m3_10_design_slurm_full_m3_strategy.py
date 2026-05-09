#!/usr/bin/env python
"""Design bounded Slurm/job-array execution for full M3 shards without submitting jobs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config
from nichefate.transition import matrix_memory_gb


STRATEGY_COLUMNS = [
    "target_time_group",
    "array_task_index",
    "array_range",
    "recommended_concurrency_cap",
    "safe_single_node_concurrency",
    "source_time",
    "target_time",
    "source_slice_id",
    "source_slice_file",
    "source_rows",
    "target_rows",
    "candidate_k",
    "expected_edge_rows",
    "estimated_per_worker_memory_gb",
    "output_dir",
    "logs_dir",
    "completed_shards_csv",
    "failed_shards_txt",
    "resume_policy",
    "retry_policy",
    "validation_policy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_transition_kernel.yaml")
    parser.add_argument(
        "--shards-csv",
        type=Path,
        default=Path("/home/zhutao/scratch/nichefate/m3/reports/m3_full_transition_shards.csv"),
    )
    parser.add_argument("--write-template", action="store_true", default=True)
    return parser.parse_args()


def _safe_token(value: object) -> str:
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")
    return token or "value"


def _reports_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["reports_dir"])


def _assert_design_only(config: dict[str, Any]) -> None:
    if bool(config["paths"].get("use_ssd", False)):
        raise RuntimeError("Refusing M3-10 strategy while paths.use_ssd is true.")
    for value in config.get("paths", {}).values():
        if isinstance(value, str) and value.startswith("/ssd"):
            raise RuntimeError(f"Refusing to use /ssd path in M3-10 strategy: {value}")
    full = config["full_m3"]
    if bool(full.get("enabled")):
        raise RuntimeError("This strategy script refuses to run while full_m3.enabled is true.")
    if full.get("execution_mode") != "design_only":
        raise RuntimeError("This strategy script requires full_m3.execution_mode=design_only.")
    if bool(full.get("write_global_kernel")):
        raise RuntimeError("This strategy script refuses global Markov P configuration.")


def _feature_dimensions(config: dict[str, Any]) -> tuple[int, int, str]:
    feature_path = _reports_dir(config) / "m3_feature_groups.json"
    if not feature_path.exists():
        return 0, 0, "m3_feature_groups.json missing; memory estimates omit feature matrices"
    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    groups = payload["feature_groups"]
    retrieval = list(
        dict.fromkeys(
            column
            for group in config["full_m3"]["retrieval_feature_groups"]
            for column in groups[group]
        )
    )
    rerank = list(
        dict.fromkeys(
            column
            for group in config["full_m3"]["rerank_feature_groups"]
            for column in groups[group]
        )
    )
    return len(retrieval), len(rerank), "feature dimensions loaded from m3_feature_groups.json"


def estimate_target_time_memory(
    shards: pd.DataFrame,
    config: dict[str, Any],
    retrieval_dimensions: int,
    rerank_dimensions: int,
) -> pd.DataFrame:
    rows = []
    max_memory = float(config["full_m3"].get("max_memory_gb_warning", 80))
    for target_time, group in shards.groupby("target_time", observed=True):
        target_rows = int(group["target_rows"].max())
        max_source_rows = int(group["source_rows"].max())
        target_retrieval = matrix_memory_gb(target_rows, retrieval_dimensions)
        target_rerank = matrix_memory_gb(target_rows, rerank_dimensions)
        source_retrieval = matrix_memory_gb(max_source_rows, retrieval_dimensions)
        source_rerank = matrix_memory_gb(max_source_rows, rerank_dimensions)
        per_worker = target_retrieval + target_rerank + source_retrieval + source_rerank
        safe = max(1, int(max_memory // per_worker)) if per_worker else 1
        cap = max(1, min(4, safe))
        rows.append(
            {
                "target_time": str(target_time),
                "target_rows": target_rows,
                "max_source_rows": max_source_rows,
                "target_retrieval_matrix_gb": target_retrieval,
                "target_rerank_matrix_gb": target_rerank,
                "source_shard_matrix_gb": source_retrieval + source_rerank,
                "estimated_per_worker_memory_gb": per_worker,
                "safe_single_node_concurrency": safe,
                "recommended_concurrency_cap": cap,
            }
        )
    return pd.DataFrame(rows)


def build_strategy_table(
    shards: pd.DataFrame,
    config: dict[str, Any],
    memory: pd.DataFrame,
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
    work = shards.copy()
    work["source_time"] = work["source_time"].astype(str)
    work["target_time"] = work["target_time"].astype(str)
    work["source_slice_id"] = work["source_slice_id"].astype(str)
    work["source_slice_file"] = work["source_slice_file"].astype(str)
    work["source_rows"] = pd.to_numeric(work["source_rows"], errors="raise").astype(int)
    work["target_rows"] = pd.to_numeric(work["target_time_rows"], errors="raise").astype(int)
    work["candidate_k"] = pd.to_numeric(work["candidate_k"], errors="raise").astype(int)
    work["expected_edge_rows"] = pd.to_numeric(work["expected_edge_rows"], errors="raise").astype(int)
    work = work.sort_values(["target_time", "source_time", "source_slice_id"]).reset_index(drop=True)
    work["target_time_group"] = work["target_time"].map(lambda value: f"target_{_safe_token(value)}")
    work["array_task_index"] = work.groupby("target_time_group", observed=True).cumcount() + 1
    group_sizes = work.groupby("target_time_group", observed=True)["source_slice_id"].transform("count")
    work = work.merge(memory, on=["target_time", "target_rows"], how="left")
    work["array_range"] = [
        f"1-{int(size)}%{int(cap)}"
        for size, cap in zip(group_sizes, work["recommended_concurrency_cap"], strict=True)
    ]
    output_root = Path(config["full_m3"]["output_root"])
    logs_dir = Path(config["paths"]["logs_dir"]) / "m3_full_array"
    reports_dir = _reports_dir(config)
    work["output_dir"] = [
        str(output_root / f"{_safe_token(src)}_to_{_safe_token(tgt)}")
        for src, tgt in zip(work["source_time"], work["target_time"], strict=True)
    ]
    work["logs_dir"] = str(logs_dir)
    work["completed_shards_csv"] = str(reports_dir / "completed_shards.csv")
    work["failed_shards_txt"] = str(reports_dir / "failed_shards.txt")
    work["resume_policy"] = "validate existing edge parquet and shard report before skip"
    work["retry_policy"] = "retry failed source-slice shard individually after reviewing failed_shards.txt"
    work["validation_policy"] = "validate schema, row counts, finite values, local row sums after each shard"
    return work[STRATEGY_COLUMNS]


def write_strategy_report(
    path: Path,
    strategy: pd.DataFrame,
    memory: pd.DataFrame,
    dimensions_note: str,
) -> None:
    global_cap = int(strategy["recommended_concurrency_cap"].min()) if not strategy.empty else 1
    lines = [
        "# M3 Bounded Slurm Array Strategy",
        "",
        "This report designs execution only. It does not submit jobs, build full M3, create edge shards, assemble global Markov P, run GPCCA, compute fate probabilities, run Branched NicheFlow, M5, or regulator analysis.",
        "",
        "## Summary",
        f"- Planned source-slice jobs: {len(strategy)}",
        "- Shard unit: one source-slice shard per job.",
        "- Array grouping: target-time grouped arrays.",
        f"- Recommended global concurrency cap: {global_cap}",
        "- No unrestricted local multiprocessing.",
        f"- Memory estimate note: {dimensions_note}",
        "",
        "## Target-Time Memory And Concurrency",
        "",
        "| target_time | target_rows | per_worker_gb | safe_single_node_concurrency | recommended_cap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in memory.to_dict("records"):
        lines.append(
            "| "
            f"{row['target_time']} | "
            f"{int(row['target_rows'])} | "
            f"{float(row['estimated_per_worker_memory_gb']):.4f} | "
            f"{int(row['safe_single_node_concurrency'])} | "
            f"{int(row['recommended_concurrency_cap'])} |"
        )
    lines.extend(
        [
            "",
            "## Resume And Retry",
            "- `completed_shards.csv` records validated completed shards.",
            "- `failed_shards.txt` records failed shard identifiers and reasons.",
            "- Resume may skip a shard only after validating the existing parquet path, shard report, schema, row count, finite numeric values, and local row-sum diagnostics.",
            "- Invalid existing output is treated as failed and retried explicitly, not silently skipped.",
            "- Failed shards are retried individually after reviewing logs and failure reason.",
            "",
            "## Per-Shard Validation",
            "- Validate before skip during resume.",
            "- Validate immediately after each shard finishes.",
            "- Required checks: expected row count, exactly K candidates per source, finite values, metadata completeness, local transition probabilities summing to one, and target slice/mouse entropy diagnostics.",
            "",
            "## Later Global Markov P Stage",
            "- Edge shards are local candidate sets and do not constitute a global Markov transition matrix P.",
            "- Global P assembly requires a separate contract for indexing, global row-stochastic checks, disconnected components, memory layout, and downstream interpretation.",
            "- GPCCA, fate probabilities, Branched NicheFlow, M5, and regulator analysis remain out of scope.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_slurm_template(path: Path, config: dict[str, Any], strategy_csv: Path) -> None:
    logs_dir = Path(config["paths"]["logs_dir"]) / "m3_full_array"
    text = f"""#!/usr/bin/env bash
#SBATCH --job-name=m3_full_shard
#SBATCH --output={logs_dir}/m3_full_shard_%A_%a.out
#SBATCH --error={logs_dir}/m3_full_shard_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd {PROJECT_ROOT}

CONFIG="${{CONFIG:-configs/m3_transition_kernel.yaml}}"
SHARD_TABLE="${{SHARD_TABLE:-{strategy_csv}}}"
TARGET_TIME_GROUP="${{TARGET_TIME_GROUP:?set TARGET_TIME_GROUP from m3_slurm_array_shards.csv}}"
TASK_ID="${{SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}}"

ROW_FIELDS="$(conda run --no-capture-output -n omicverse python - "$SHARD_TABLE" "$TARGET_TIME_GROUP" "$TASK_ID" <<'PY'
import pandas as pd
import sys

table = pd.read_csv(sys.argv[1])
group = sys.argv[2]
task_id = int(sys.argv[3])
row = table[
    (table["target_time_group"].astype(str) == group)
    & (table["array_task_index"].astype(int) == task_id)
].iloc[0]
print(row["source_time"], row["target_time"], row["source_slice_id"], row["source_slice_file"], row["output_dir"])
PY
)"
read -r SOURCE_TIME TARGET_TIME SOURCE_SLICE_ID SOURCE_SLICE_FILE OUTPUT_DIR <<< "$ROW_FIELDS"

conda run --no-capture-output -n omicverse python scripts/m3_05_build_transition_pilot_shard.py \\
  --config "$CONFIG" \\
  --source-time "$SOURCE_TIME" \\
  --target-time "$TARGET_TIME" \\
  --source-slice-id "$SOURCE_SLICE_ID" \\
  --source-slice-file "$SOURCE_SLICE_FILE" \\
  --output-dir "$OUTPUT_DIR"
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    _assert_design_only(config)
    reports_dir = _reports_dir(config)
    reports_dir.mkdir(parents=True, exist_ok=True)
    shards = pd.read_csv(args.shards_csv)
    retrieval_dims, rerank_dims, dimensions_note = _feature_dimensions(config)
    shards_for_memory = shards.copy()
    shards_for_memory["target_time"] = shards_for_memory["target_time"].astype(str)
    shards_for_memory["target_rows"] = pd.to_numeric(
        shards_for_memory["target_time_rows"],
        errors="raise",
    ).astype(int)
    shards_for_memory["source_rows"] = pd.to_numeric(
        shards_for_memory["source_rows"],
        errors="raise",
    ).astype(int)
    memory = estimate_target_time_memory(
        shards_for_memory,
        config,
        retrieval_dims,
        rerank_dims,
    )
    strategy = build_strategy_table(shards, config, memory)
    csv_path = reports_dir / "m3_slurm_array_shards.csv"
    md_path = reports_dir / "m3_slurm_array_strategy.md"
    template_dir = reports_dir / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    template_path = template_dir / "m3_full_shard_array.sbatch"
    strategy.to_csv(csv_path, index=False)
    write_strategy_report(md_path, strategy, memory, dimensions_note)
    if args.write_template:
        write_slurm_template(template_path, config, csv_path)
    print(f"Wrote Slurm strategy shard table: {csv_path}")
    print(f"Wrote Slurm strategy report: {md_path}")
    if args.write_template:
        print(f"Wrote Slurm template: {template_path}")
    print(f"RECOMMENDED_GLOBAL_CONCURRENCY_CAP {int(strategy['recommended_concurrency_cap'].min())}")
    print("SUBMITTED_JOBS False")
    print("DESIGN_ONLY True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
