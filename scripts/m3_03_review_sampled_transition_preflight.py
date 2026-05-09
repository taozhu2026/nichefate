#!/usr/bin/env python
"""Review sampled M3 transition evidence preflight outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config
from nichefate.transition import transition_probability_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_transition_kernel.yaml")
    return parser.parse_args()


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        key: Path(value)
        for key, value in config["paths"].items()
        if isinstance(value, str) and value.startswith("/")
    }


def _read_edges(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _finite_count(frame: pd.DataFrame, columns: list[str]) -> int:
    if not columns:
        return 0
    values = frame[columns].to_numpy(dtype=float)
    return int((~np.isfinite(values)).sum())


def _write_md(path: Path, rows: list[dict[str, Any]], expected_edges: int) -> None:
    actual_edges = sum(int(row["edge_count"]) for row in rows)
    lines = [
        "# M3 Sampled Transition Preflight QC",
        "",
        f"- Expected sampled edge count upper bound: {expected_edges}",
        f"- Actual sampled edge count: {actual_edges}",
        "- Feature matrices were standardized before KNN retrieval.",
        "- `combined_cost` was computed only from scaled evidence columns.",
        "- `row_normalized_transition_prob` is a local candidate-set transition probability, not the full global Markov transition matrix P.",
        "- `raw_edge_weight` and `mass_adjusted_weight` are preserved for future unbalanced transport and Branched NicheFlow pseudo-pair supervision.",
        "- Candidate construction is sample-aware and metadata-preserving, not longitudinal sample-paired in v1.",
        "",
    ]
    for row in rows:
        status = "PASS" if row["ok"] else "FAIL"
        lines.append(
            "- "
            f"{row['source_time']} -> {row['target_time']}: {status}, "
            f"edges={row['edge_count']}, row_sum_range="
            f"{row['row_sum_min']:.6g}-{row['row_sum_max']:.6g}"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_plots(reports_dir: Path, rows: list[dict[str, Any]], edge_paths: list[Path]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for path in edge_paths:
        frame = _read_edges(path)
        label = path.stem.replace("candidate_edges_", "")
        for column in ["combined_cost", "row_normalized_transition_prob"]:
            plt.figure(figsize=(5, 3))
            frame[column].hist(bins=50)
            plt.title(f"{column} {label}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"{column}_{label}.png", dpi=120)
            plt.close()
    del rows


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = _paths(config)
    reports_dir = paths["reports_dir"]
    with (reports_dir / "m3_time_pairs.json").open("r", encoding="utf-8") as handle:
        time_pairs = json.load(handle)

    edge_paths = sorted(paths["prototype_dir"].glob("candidate_edges_*"))
    raw_cols = [
        "raw_molecular_distance",
        "raw_composition_distance",
        "raw_entropy_distance",
        "raw_spatial_summary_distance",
        "raw_topology_distance",
        "raw_pseudotime_score",
        "raw_barcode_score",
    ]
    scaled_cols = [
        "scaled_molecular_distance",
        "scaled_composition_distance",
        "scaled_entropy_distance",
        "scaled_spatial_summary_distance",
        "scaled_topology_distance",
        "scaled_pseudotime_score",
        "scaled_barcode_score",
    ]
    metadata_cols = [
        "source_slice_id",
        "target_slice_id",
        "source_slice_file",
        "target_slice_file",
        "source_mouse_id",
        "target_mouse_id",
        "source_time",
        "target_time",
    ]
    rows = []
    for path in edge_paths:
        frame = _read_edges(path)
        source_time = str(frame["source_time"].iloc[0])
        target_time = str(frame["target_time"].iloc[0])
        pair = next(
            (
                item
                for item in time_pairs
                if str(item["source_time"]) == source_time
                and str(item["target_time"]) == target_time
            ),
            None,
        )
        diagnostics = transition_probability_diagnostics(frame)
        row_sums = frame.groupby("source_anchor_id", observed=True)[
            "row_normalized_transition_prob"
        ].sum()
        candidate_counts = frame.groupby("source_anchor_id", observed=True).size()
        finite_cols = raw_cols + scaled_cols + [
            "combined_cost",
            "tau_pair",
            "raw_edge_weight",
            "mass_adjusted_weight",
            "row_normalized_transition_prob",
        ]
        nonfinite = _finite_count(frame, finite_cols)
        metadata_missing = int(frame[metadata_cols].isna().sum().sum())
        time_direction_ok = bool((frame["target_day"] > frame["source_day"]).all())
        time_delta_ok = True
        if pair is not None:
            time_delta_ok = bool(np.allclose(frame["time_delta"], float(pair["time_delta"])))
        rows.append(
            {
                "source_time": source_time,
                "target_time": target_time,
                "edge_count": len(frame),
                "source_count": int(frame["source_anchor_id"].nunique()),
                "candidate_count_min": int(candidate_counts.min()),
                "candidate_count_max": int(candidate_counts.max()),
                "candidate_count_mean": float(candidate_counts.mean()),
                "combined_cost_min": float(frame["combined_cost"].min()),
                "combined_cost_median": float(frame["combined_cost"].median()),
                "combined_cost_max": float(frame["combined_cost"].max()),
                "transition_prob_min": float(frame["row_normalized_transition_prob"].min()),
                "transition_prob_median": float(frame["row_normalized_transition_prob"].median()),
                "transition_prob_max": float(frame["row_normalized_transition_prob"].max()),
                "nonfinite_values": nonfinite,
                "metadata_missing": metadata_missing,
                "time_direction_ok": time_direction_ok,
                "time_delta_ok": time_delta_ok,
                "sample_metadata_preserved": metadata_missing == 0,
                "row_sums_close_to_one": bool(np.allclose(row_sums, 1.0)),
                **diagnostics,
            }
        )
    for row in rows:
        row["ok"] = bool(
            row["nonfinite_values"] == 0
            and row["metadata_missing"] == 0
            and row["time_direction_ok"]
            and row["time_delta_ok"]
            and row["row_sums_close_to_one"]
        )
    qc_csv = reports_dir / "m3_sampled_transition_preflight_qc.csv"
    qc_md = reports_dir / "m3_sampled_transition_preflight_qc.md"
    pd.DataFrame(rows).to_csv(qc_csv, index=False)
    expected_edges = (
        len(time_pairs)
        * int(config["candidate_edges"]["max_source_niches_per_pair"])
        * int(config["candidate_edges"]["k_candidates"])
    )
    _write_md(qc_md, rows, expected_edges)
    _write_plots(reports_dir, rows, edge_paths)
    print(f"Wrote sampled transition QC CSV: {qc_csv}")
    print(f"Wrote sampled transition QC report: {qc_md}")
    print(f"EXPECTED_TIME_PAIRS {len(time_pairs)}")
    print(f"EDGE_FILES {len(edge_paths)}")
    print(f"FAILED_QC {sum(not row['ok'] for row in rows)}")
    return 0 if len(edge_paths) == len(time_pairs) and all(row["ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
