#!/usr/bin/env python
"""Review and freeze M4C Markov fate probability results."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config


DEFAULT_CONFIG = "configs/m4c_fate_probability.yaml"
FATE_PROBABILITIES_RECOMPUTED = False
FATE_MATRIX_LOADED = False
NO_DOWNSTREAM_FLAGS = {
    "no_gpcca": True,
    "no_terminal_state_inference": True,
    "no_branched_nicheflow_training": True,
    "no_branchsbm_training": True,
    "no_m5": True,
    "no_regulator_analysis": True,
}
BARCODE_COMPATIBLE_INTERPRETATION_NOTE = (
    "Future barcode-aware M3 transition evidence can replace or supplement the current "
    "pseudo-lineage/time-coupled transition evidence while preserving the M4C fate interface."
)
M4C_SEMANTIC_NOTE = (
    "M4C is Markov baseline v1 using final-time clustering targets and "
    "pseudo-lineage/time-coupled Markov fate probabilities. It is not the "
    "standard GPCCA/CellRank-inspired M4D route, and its targets are not "
    "proven biological terminal fates."
)
FINAL_REVIEW_FIGURES = [
    "m4c_final_entropy_plasticity_by_time.png",
    "m4c_final_dominant_fate_by_time.png",
    "m4c_final_terminal_probability_heatmap_by_time.png",
    "m4c_final_fate_mass_by_terminal_macrostate.png",
    "m4c_final_slice_fate_variability_heatmap.png",
    "m4c_final_mouse_fate_variability_heatmap.png",
    "m4c_final_terminal_confidence_tiers.png",
    "m4c_final_markov_fate_dashboard.png",
]
NODE_REVIEW_COLUMNS = [
    "global_node_index",
    "time_day",
    "time",
    "slice_id",
    "mouse_id",
    "plasticity_entropy",
    "normalized_plasticity_entropy",
    "dominant_fate",
    "dominant_fate_label",
    "dominant_fate_probability",
    "fate_margin_top1_minus_top2",
    "directionality_evidence_source",
    "barcode_compatible_contract",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(json_safe(payload), indent=2) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def assert_no_ssd_path(path: Path, label: str) -> None:
    resolved = str(path.expanduser().resolve())
    if resolved == "/ssd" or resolved.startswith("/ssd/"):
        raise ValueError(f"Refusing to use /ssd for {label}: {path}")


def configured_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    for key, path in paths.items():
        assert_no_ssd_path(path, f"paths.{key}")
    return paths


def review_input_paths(paths: dict[str, Path]) -> dict[str, Path]:
    fate_dir = paths["output_root"] / "fate_probabilities"
    reports_dir = paths["reports_dir"]
    m4b_terminal_dir = paths["terminal_summary"].parent
    m4b_reports_dir = m4b_terminal_dir.parent / "reports"
    return {
        "fate_matrix": fate_dir / "fate_probability_matrix.npz",
        "node_summary": fate_dir / "fate_probability_node_summary.parquet",
        "by_time": fate_dir / "fate_probability_by_time_summary.csv",
        "by_slice": fate_dir / "fate_probability_by_slice_summary.csv",
        "by_mouse": fate_dir / "fate_probability_by_mouse_summary.csv",
        "qc": reports_dir / "m4c_fate_probability_qc_summary.csv",
        "m4c_report": reports_dir / "m4c_markov_fate_probability_report.md",
        "m4c_schema": reports_dir / "m4c_fate_probability_schema.json",
        "m4c_figures_dir": paths["figures_dir"],
        "terminal_assignments": paths["terminal_assignments"],
        "terminal_summary": paths["terminal_summary"],
        "terminal_feature_summary": m4b_terminal_dir / "terminal_macrostate_feature_summary.csv",
        "m4b_design_report": m4b_reports_dir / "m4b_terminal_macrostate_design_report.md",
        "m4b_gpcca_feasibility_report": m4b_reports_dir / "m4b_markov_gpcca_feasibility_report.md",
    }


def review_output_paths(paths: dict[str, Path]) -> dict[str, Path]:
    reports_dir = paths["reports_dir"]
    final_figures_dir = paths["figures_dir"] / "final_review"
    outputs = {
        "final_review_report": reports_dir / "m4c_markov_fate_final_review.md",
        "freeze_summary": reports_dir / "m4c_markov_fate_final_freeze_summary.json",
        "confidence_tiers": reports_dir / "m4c_terminal_macrostate_confidence_tiers.csv",
        "interpretation_cautions": reports_dir / "m4c_fate_result_interpretation_cautions.md",
        "result_inventory": reports_dir / "m4c_markov_fate_result_inventory.csv",
        "final_figures_dir": final_figures_dir,
    }
    forbidden = ["gpcca", "branched_nicheflow", "branchsbm", "m5", "regulator"]
    offenders = [str(path) for key, path in outputs.items() if key != "final_figures_dir" and any(t in str(path).lower() for t in forbidden)]
    if offenders:
        raise ValueError(f"M4C-02 output paths include forbidden downstream targets: {offenders}")
    return outputs


def validate_required_inputs(inputs: dict[str, Path]) -> None:
    required = [
        "node_summary",
        "by_time",
        "by_slice",
        "by_mouse",
        "qc",
        "terminal_assignments",
        "terminal_summary",
        "terminal_feature_summary",
        "m4b_design_report",
        "m4b_gpcca_feasibility_report",
    ]
    missing = [f"{key}: {inputs[key]}" for key in required if not inputs[key].exists()]
    if missing:
        raise FileNotFoundError("Missing required M4C-02 review inputs:\n" + "\n".join(missing))


def read_node_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=NODE_REVIEW_COLUMNS)
    if not frame["global_node_index"].is_monotonic_increasing:
        raise ValueError("M4C node summary must preserve global_node_index ordering.")
    expected = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["global_node_index"].to_numpy(dtype=np.int64), expected):
        raise ValueError("M4C node summary row i must match global_node_index i.")
    return frame


def load_review_inputs(inputs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    validate_required_inputs(inputs)
    return {
        "node_summary": read_node_summary(inputs["node_summary"]),
        "by_time": pd.read_csv(inputs["by_time"]),
        "by_slice": pd.read_csv(inputs["by_slice"]),
        "by_mouse": pd.read_csv(inputs["by_mouse"]),
        "qc": pd.read_csv(inputs["qc"]),
        "terminal_summary": pd.read_csv(inputs["terminal_summary"]),
        "terminal_feature_summary": pd.read_csv(inputs["terminal_feature_summary"]),
    }


def scalar_qc(qc: pd.DataFrame) -> dict[str, Any]:
    if len(qc) != 1:
        raise ValueError(f"Expected one M4C QC row, found {len(qc)}.")
    row = qc.iloc[0].to_dict()
    error = max(
        float(row.get("nonfinal_row_sum_error_max", 0.0)),
        float(row.get("final_row_sum_error_max", 0.0)),
    )
    row["row_sum_min_qc_bound"] = 1.0 - error
    row["row_sum_max_qc_bound"] = 1.0 + error
    row["fate_matrix_loaded_for_review"] = FATE_MATRIX_LOADED
    row["fate_probabilities_recomputed"] = FATE_PROBABILITIES_RECOMPUTED
    return row


def metric_stats(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = frame[metric].to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "p25": float(np.quantile(values, 0.25)),
                "p50": float(np.quantile(values, 0.50)),
                "p75": float(np.quantile(values, 0.75)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def metrics_by_time(node_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "plasticity_entropy",
        "normalized_plasticity_entropy",
        "dominant_fate_probability",
        "fate_margin_top1_minus_top2",
    ]
    rows: list[dict[str, Any]] = []
    for (time_day, time_label), group in node_summary.groupby(["time_day", "time"], sort=True, observed=True):
        row: dict[str, Any] = {
            "time_day": float(time_day),
            "time": str(time_label),
            "n_nodes": int(len(group)),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_p25"] = float(np.quantile(values, 0.25))
            row[f"{metric}_p50"] = float(np.quantile(values, 0.50))
            row[f"{metric}_p75"] = float(np.quantile(values, 0.75))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["time_day", "time"]).reset_index(drop=True)


def terminal_fate_mass_summary(by_time: pd.DataFrame, node_summary: pd.DataFrame) -> pd.DataFrame:
    total_nodes = float(len(node_summary))
    dominant_counts = node_summary["dominant_fate"].value_counts().to_dict()
    grouped = (
        by_time.groupby(["terminal_macrostate", "terminal_macrostate_label"], observed=True)
        .agg(
            total_fate_mass=("sum_probability", "sum"),
            mean_probability_over_time=("mean_probability", "mean"),
            mean_normalized_mass_fraction=("normalized_mass_fraction", "mean"),
            max_dominant_fate_fraction=("dominant_fate_fraction", "max"),
            mean_dominant_fate_fraction=("dominant_fate_fraction", "mean"),
        )
        .reset_index()
        .sort_values("terminal_macrostate")
    )
    grouped["total_fate_mass_fraction"] = grouped["total_fate_mass"] / total_nodes if total_nodes else 0.0
    grouped["dominant_fate_node_count"] = grouped["terminal_macrostate"].map(lambda value: int(dominant_counts.get(int(value), 0)))
    grouped["dominant_fate_node_fraction"] = grouped["dominant_fate_node_count"] / total_nodes if total_nodes else 0.0
    return grouped


def dominant_fate_by_time(by_time: pd.DataFrame) -> pd.DataFrame:
    return (
        by_time.sort_values(["time_day", "time", "dominant_fate_fraction"], ascending=[True, True, False])
        .groupby(["time_day", "time"], sort=True, observed=True)
        .head(1)
        .reset_index(drop=True)
    )


def detect_dominant_fate_collapse(by_time: pd.DataFrame, dominance_threshold: float = 0.50) -> pd.DataFrame:
    top = dominant_fate_by_time(by_time)
    n_times = int(top[["time_day", "time"]].drop_duplicates().shape[0])
    rows: list[dict[str, Any]] = []
    for (macro_id, label), group in top.groupby(["terminal_macrostate", "terminal_macrostate_label"], observed=True):
        top_all_times = len(group) == n_times
        threshold_all_times = bool((group["dominant_fate_fraction"].to_numpy(dtype=float) >= dominance_threshold).all())
        if top_all_times or threshold_all_times:
            rows.append(
                {
                    "terminal_macrostate": int(macro_id),
                    "terminal_macrostate_label": str(label),
                    "top_time_points": int(len(group)),
                    "n_time_points": n_times,
                    "mean_top_dominant_fate_fraction": float(group["dominant_fate_fraction"].mean()),
                    "dominates_all_time_points": bool(top_all_times),
                    "exceeds_threshold_all_top_times": bool(threshold_all_times),
                    "warning": bool(top_all_times and threshold_all_times),
                }
            )
    return pd.DataFrame(rows)


def flag_timepoint_structure(time_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = time_metrics.copy()
    plasticity = frame["normalized_plasticity_entropy_mean"].to_numpy(dtype=float)
    top1 = frame["dominant_fate_probability_mean"].to_numpy(dtype=float)
    margin = frame["fate_margin_top1_minus_top2_mean"].to_numpy(dtype=float)
    high_cut = max(float(np.quantile(plasticity, 0.75)), 0.35)
    low_cut = min(float(np.quantile(plasticity, 0.25)), 0.05)
    frame["high_plasticity_warning"] = plasticity >= high_cut
    frame["low_plasticity_warning"] = plasticity <= low_cut
    frame["overly_flat_warning"] = (plasticity >= 0.35) | (top1 <= 0.65) | (margin <= 0.30)
    frame["winner_take_all_warning"] = (plasticity <= 0.05) | (top1 >= 0.90) | (margin >= 0.80)
    labels: list[str] = []
    for row in frame.to_dict("records"):
        active = [
            name
            for name in [
                "high_plasticity_warning",
                "low_plasticity_warning",
                "overly_flat_warning",
                "winner_take_all_warning",
            ]
            if bool(row[name])
        ]
        labels.append(";".join(active) if active else "none")
    frame["timepoint_review_label"] = labels
    return frame


def group_variability_summary(summary: pd.DataFrame, group_column: str) -> pd.DataFrame:
    grouped = (
        summary.groupby([group_column, "terminal_macrostate", "terminal_macrostate_label"], dropna=False, observed=True)
        .agg(
            dominant_fate_fraction=("dominant_fate_fraction", "mean"),
            normalized_mass_fraction=("normalized_mass_fraction", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for (macro_id, label), group in grouped.groupby(["terminal_macrostate", "terminal_macrostate_label"], observed=True):
        values = group["dominant_fate_fraction"].to_numpy(dtype=float)
        max_idx = int(np.argmax(values)) if len(values) else 0
        mean_value = float(np.mean(values)) if len(values) else 0.0
        max_value = float(values[max_idx]) if len(values) else 0.0
        rows.append(
            {
                "terminal_macrostate": int(macro_id),
                "terminal_macrostate_label": str(label),
                "group_column": group_column,
                "n_groups": int(len(group)),
                "mean_dominant_fate_fraction": mean_value,
                "std_dominant_fate_fraction": float(np.std(values)) if len(values) else 0.0,
                "max_dominant_fate_fraction": max_value,
                "min_dominant_fate_fraction": float(np.min(values)) if len(values) else 0.0,
                "max_to_mean_ratio": float(max_value / mean_value) if mean_value > 0 else 0.0,
                "max_association_group": str(group.iloc[max_idx][group_column]) if len(group) else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["terminal_macrostate", "group_column"]).reset_index(drop=True)


def association_warnings(
    variability: pd.DataFrame,
    min_fraction: float = 0.45,
    ratio_threshold: float = 2.50,
) -> pd.DataFrame:
    frame = variability.copy()
    frame["association_warning"] = (frame["max_dominant_fate_fraction"] >= min_fraction) & (
        frame["max_to_mean_ratio"] >= ratio_threshold
    )
    frame["confidence_caution_label"] = np.where(
        frame["association_warning"],
        "potential_group_association_warning",
        "no_strong_group_association_detected",
    )
    return frame


def top_feature_summary(feature_summary: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    if feature_summary.empty:
        return pd.DataFrame(columns=["terminal_macrostate_id", "top_feature_summary"])
    frame = feature_summary.copy()
    frame["abs_mean"] = frame["mean"].astype(float).abs()
    rows: list[dict[str, Any]] = []
    for macro_id, group in frame.groupby("terminal_macrostate_id", observed=True):
        top = group.sort_values("abs_mean", ascending=False).head(top_n)
        rows.append(
            {
                "terminal_macrostate_id": int(macro_id),
                "top_feature_summary": "; ".join(
                    f"{row.feature}={float(row.mean):.3g}" for row in top.itertuples(index=False)
                ),
            }
        )
    return pd.DataFrame(rows)


def assign_confidence_tiers(
    terminal_summary: pd.DataFrame,
    fate_mass: pd.DataFrame,
    slice_warnings: pd.DataFrame,
    mouse_warnings: pd.DataFrame,
    feature_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    terminal = terminal_summary.sort_values("terminal_macrostate_id").copy()
    terminal = terminal.rename(columns={"terminal_macrostate_id": "terminal_macrostate"})
    merged = terminal.merge(fate_mass, on=["terminal_macrostate", "terminal_macrostate_label"], how="left")
    for column in [
        "total_fate_mass",
        "total_fate_mass_fraction",
        "dominant_fate_node_count",
        "dominant_fate_node_fraction",
        "mean_dominant_fate_fraction",
        "max_dominant_fate_fraction",
    ]:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = merged[column].fillna(0.0)

    slice_flags = slice_warnings.loc[slice_warnings["association_warning"], ["terminal_macrostate", "max_association_group"]]
    mouse_flags = mouse_warnings.loc[mouse_warnings["association_warning"], ["terminal_macrostate", "max_association_group"]]
    slice_map = dict(zip(slice_flags["terminal_macrostate"], slice_flags["max_association_group"], strict=False))
    mouse_map = dict(zip(mouse_flags["terminal_macrostate"], mouse_flags["max_association_group"], strict=False))
    merged["slice_association_warning"] = merged["terminal_macrostate"].map(lambda value: int(value) in slice_map)
    merged["mouse_association_warning"] = merged["terminal_macrostate"].map(lambda value: int(value) in mouse_map)
    merged["slice_association_group"] = merged["terminal_macrostate"].map(lambda value: slice_map.get(int(value), ""))
    merged["mouse_association_group"] = merged["terminal_macrostate"].map(lambda value: mouse_map.get(int(value), ""))

    if feature_summary is not None:
        merged = merged.merge(top_feature_summary(feature_summary), left_on="terminal_macrostate", right_on="terminal_macrostate_id", how="left")
        merged = merged.drop(columns=[column for column in ["terminal_macrostate_id"] if column in merged.columns])
    if "top_feature_summary" not in merged.columns:
        merged["top_feature_summary"] = ""
    merged["top_feature_summary"] = merged["top_feature_summary"].fillna("")

    size_fraction = merged.get("fraction_final_nodes", pd.Series(0.0, index=merged.index)).astype(float)
    mass_fraction = merged["total_fate_mass_fraction"].astype(float)
    incoming_mass = merged.get("incoming_mass_sum_structural", pd.Series(0.0, index=merged.index)).astype(float)
    dominant_l1 = merged.get("dominant_cell_type_l1_fraction", pd.Series(np.nan, index=merged.index)).astype(float)
    l1_entropy = merged.get("cell_type_l1_entropy", pd.Series(np.nan, index=merged.index)).astype(float)

    size_low = max(0.03, float(size_fraction.quantile(0.10)))
    mass_low = max(0.03, float(mass_fraction.quantile(0.10)))
    incoming_low = float(incoming_mass.quantile(0.10)) if len(incoming_mass) else 0.0
    size_high = float(size_fraction.median()) if len(size_fraction) else 0.0
    mass_high = float(mass_fraction.median()) if len(mass_fraction) else 0.0
    incoming_high = float(incoming_mass.median()) if len(incoming_mass) else 0.0
    entropy_high = float(l1_entropy.quantile(0.75)) if l1_entropy.notna().any() else np.inf

    tiers: list[str] = []
    reasons: list[str] = []
    for idx, row in merged.iterrows():
        assoc = bool(row["slice_association_warning"]) or bool(row["mouse_association_warning"])
        low = (
            float(size_fraction.loc[idx]) <= size_low
            or float(mass_fraction.loc[idx]) <= mass_low
            or (float(incoming_mass.loc[idx]) <= incoming_low and float(size_fraction.loc[idx]) < size_high)
        )
        mixed = (
            (pd.notna(dominant_l1.loc[idx]) and float(dominant_l1.loc[idx]) < 0.50)
            or (pd.notna(l1_entropy.loc[idx]) and float(l1_entropy.loc[idx]) >= entropy_high)
        )
        high = (
            float(size_fraction.loc[idx]) >= size_high
            and float(mass_fraction.loc[idx]) >= mass_high
            and float(incoming_mass.loc[idx]) >= incoming_high
            and (pd.isna(dominant_l1.loc[idx]) or float(dominant_l1.loc[idx]) >= 0.60)
            and not assoc
        )
        if assoc:
            tiers.append("potential_slice_or_mouse_associated_endpoint")
            reasons.append("dominant fate fraction is concentrated in one slice or mouse group")
        elif low:
            tiers.append("low_size_or_low_mass_endpoint")
            reasons.append("small final-time macrostate, low total fate mass, or low incoming mass")
        elif high:
            tiers.append("high_confidence_terminal_like")
            reasons.append("large enough endpoint with strong fate mass, incoming support, and clearer cell-type composition")
        elif mixed:
            tiers.append("mixed_or_intermediate_final_time_state")
            reasons.append("mixed final-time cell-type composition or high cell-type entropy")
        else:
            tiers.append("unclear")
            reasons.append("does not clearly meet high-confidence, low-mass, mixed, or batch-associated criteria")
    merged["confidence_tier"] = tiers
    merged["confidence_reason"] = reasons
    return merged.sort_values("terminal_macrostate").reset_index(drop=True)


def terminal_tier_counts(confidence: pd.DataFrame) -> dict[str, int]:
    counts = confidence["confidence_tier"].value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def final_time_label(by_time: pd.DataFrame) -> str:
    final_day = float(by_time["time_day"].astype(float).max())
    labels = sorted(
        by_time.loc[np.isclose(by_time["time_day"].astype(float), final_day), "time"]
        .dropna()
        .astype(str)
        .unique()
    )
    if len(labels) != 1:
        raise ValueError(f"Expected one final observed time label for max time_day {final_day}, found {labels}.")
    return labels[0]


def make_heatmap_frame(summary: pd.DataFrame, group_column: str, value_column: str = "dominant_fate_fraction") -> pd.DataFrame:
    return summary.pivot_table(
        index=group_column,
        columns="terminal_macrostate_label",
        values=value_column,
        aggfunc="mean",
        fill_value=0.0,
    ).sort_index()


def generate_figures(
    final_figures_dir: Path,
    by_time: pd.DataFrame,
    by_slice: pd.DataFrame,
    by_mouse: pd.DataFrame,
    time_metrics: pd.DataFrame,
    fate_mass: pd.DataFrame,
    confidence: pd.DataFrame,
    warning_only: bool = True,
) -> tuple[list[str], list[Path]]:
    warnings: list[str] = []
    generated = [final_figures_dir / name for name in FINAL_REVIEW_FIGURES]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        final_figures_dir.mkdir(parents=True, exist_ok=True)

        x = np.arange(len(time_metrics))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, time_metrics["plasticity_entropy_mean"], marker="o", label="entropy")
        ax.plot(x, time_metrics["normalized_plasticity_entropy_mean"], marker="o", label="normalized")
        ax.set_xticks(x)
        ax.set_xticklabels(time_metrics["time"].astype(str), rotation=30)
        ax.set_title("Entropy and normalized plasticity by time")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(final_figures_dir / FINAL_REVIEW_FIGURES[0], dpi=140)
        plt.close(fig)

        dominant = by_time.pivot(index="time", columns="terminal_macrostate_label", values="dominant_fate_fraction").fillna(0.0)
        order = by_time[["time_day", "time"]].drop_duplicates().sort_values(["time_day", "time"])["time"].astype(str)
        dominant = dominant.loc[order]
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = np.zeros(len(dominant), dtype=float)
        for label in dominant.columns:
            values = dominant[label].to_numpy(dtype=float)
            ax.bar(dominant.index.astype(str), values, bottom=bottom, label=str(label))
            bottom += values
        ax.set_title("Dominant fate composition by time")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=6, ncol=3)
        fig.tight_layout()
        fig.savefig(final_figures_dir / FINAL_REVIEW_FIGURES[1], dpi=140)
        plt.close(fig)

        probability = by_time.pivot(index="time", columns="terminal_macrostate_label", values="mean_probability").fillna(0.0)
        probability = probability.loc[order]
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(probability.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="magma")
        ax.set_title("Mean terminal macrostate probability by time")
        ax.set_xticks(np.arange(len(probability.columns)))
        ax.set_xticklabels(probability.columns.astype(str), rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(probability.index)))
        ax.set_yticklabels(probability.index.astype(str))
        fig.colorbar(im, ax=ax, fraction=0.04, label="mean probability")
        fig.tight_layout()
        fig.savefig(final_figures_dir / FINAL_REVIEW_FIGURES[2], dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(fate_mass["terminal_macrostate_label"].astype(str), fate_mass["total_fate_mass_fraction"].astype(float))
        ax.set_title("Total fate mass fraction by terminal macrostate")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(final_figures_dir / FINAL_REVIEW_FIGURES[3], dpi=140)
        plt.close(fig)

        for filename, summary, group_column, title in [
            (FINAL_REVIEW_FIGURES[4], by_slice, "slice_id", "Dominant fate variability by slice"),
            (FINAL_REVIEW_FIGURES[5], by_mouse, "mouse_id", "Dominant fate variability by mouse"),
        ]:
            heat = make_heatmap_frame(summary, group_column)
            fig, ax = plt.subplots(figsize=(10, max(4, min(14, 0.18 * len(heat)))))
            im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="viridis")
            ax.set_title(title)
            ax.set_xticks(np.arange(len(heat.columns)))
            ax.set_xticklabels(heat.columns.astype(str), rotation=45, ha="right", fontsize=7)
            y_step = max(1, len(heat) // 30)
            ax.set_yticks(np.arange(0, len(heat), y_step))
            ax.set_yticklabels(heat.index.astype(str)[::y_step], fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.03, label="dominant fate fraction")
            fig.tight_layout()
            fig.savefig(final_figures_dir / filename, dpi=140)
            plt.close(fig)

        tier_counts = confidence["confidence_tier"].value_counts().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(tier_counts.index.astype(str), tier_counts.to_numpy(dtype=int))
        ax.set_title("Terminal macrostate confidence tiers")
        fig.tight_layout()
        fig.savefig(final_figures_dir / FINAL_REVIEW_FIGURES[6], dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].plot(time_metrics["time"].astype(str), time_metrics["normalized_plasticity_entropy_mean"], marker="o")
        axes[0, 0].tick_params(axis="x", rotation=30)
        axes[0, 0].set_title("Mean normalized plasticity")
        axes[0, 1].bar(fate_mass["terminal_macrostate_label"].astype(str), fate_mass["total_fate_mass_fraction"])
        axes[0, 1].tick_params(axis="x", rotation=60, labelsize=6)
        axes[0, 1].set_title("Fate mass fraction")
        axes[1, 0].barh(tier_counts.index.astype(str), tier_counts.to_numpy(dtype=int))
        axes[1, 0].set_title("Confidence tiers")
        axes[1, 1].plot(time_metrics["time"].astype(str), time_metrics["dominant_fate_probability_mean"], marker="o", label="top1")
        axes[1, 1].plot(time_metrics["time"].astype(str), time_metrics["fate_margin_top1_minus_top2_mean"], marker="o", label="margin")
        axes[1, 1].tick_params(axis="x", rotation=30)
        axes[1, 1].set_title("Top1 probability and margin")
        axes[1, 1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(final_figures_dir / FINAL_REVIEW_FIGURES[7], dpi=140)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        if not warning_only:
            raise
        warnings.append(f"Final review figure generation failed after core review passed: {exc}")
    return warnings, generated


def inventory_rows(inputs: dict[str, Path], outputs: dict[str, Path], final_figures: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(category: str, name: str, path: Path, status: str, frozen: bool, notes: str = "") -> None:
        rows.append(
            {
                "artifact_category": category,
                "artifact_name": name,
                "path": str(path),
                "exists": bool(path.exists()),
                "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
                "status": status,
                "frozen": bool(frozen),
                "notes": notes,
            }
        )

    for key in ["fate_matrix", "node_summary", "by_time", "by_slice", "by_mouse", "qc", "m4c_report", "m4c_schema"]:
        add("m4c_input", key, inputs[key], "reused_read_only", True, "M4C-01 output reused; not recomputed")
    for key in ["terminal_assignments", "terminal_summary", "terminal_feature_summary", "m4b_design_report"]:
        add("m4b_input", key, inputs[key], "reused_read_only", True, "M4B interpretability input reused")
    for key in ["final_review_report", "freeze_summary", "confidence_tiers", "interpretation_cautions", "result_inventory"]:
        add("m4c_final_review_output", key, outputs[key], "generated_by_m4c_02", True)
    for path in final_figures:
        add("m4c_final_review_figure", path.name, path, "generated_by_m4c_02", True)
    return pd.DataFrame(rows)


def interpretation_cautions_text(
    confidence: pd.DataFrame,
    slice_warnings: pd.DataFrame,
    mouse_warnings: pd.DataFrame,
    final_observed_time: str = "the final observed time point",
) -> str:
    slice_count = int(slice_warnings["association_warning"].sum()) if "association_warning" in slice_warnings else 0
    mouse_count = int(mouse_warnings["association_warning"].sum()) if "association_warning" in mouse_warnings else 0
    tier_counts = terminal_tier_counts(confidence)
    lines = [
        "# M4C Markov Fate Interpretation Cautions",
        "",
        M4C_SEMANTIC_NOTE,
        f"{final_observed_time} is the final observed time point in this dataset, not necessarily a true biological endpoint.",
        BARCODE_COMPATIBLE_INTERPRETATION_NOTE,
        "",
        "## Scope Boundaries",
        "- M4C did not run GPCCA.",
        "- M4C did not run Branched NicheFlow / BranchSBM.",
        "- M4C did not run M5 or regulator analysis.",
        "- M4C-02 did not merge, remove, or rediscover terminal macrostates.",
        "- M4C-02 did not recompute fate probabilities.",
        "",
        "## Confidence Tier Counts",
        *[f"- {tier}: {count}" for tier, count in tier_counts.items()],
        "",
        "## Stability Cautions",
        f"- terminal macrostates with slice association warnings: {slice_count}",
        f"- terminal macrostates with mouse association warnings: {mouse_count}",
        "These warning-only diagnostics do not invalidate the Markov fate baseline.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def final_review_report_text(
    qc: dict[str, Any],
    metric_summary: pd.DataFrame,
    time_flags: pd.DataFrame,
    dominant_top: pd.DataFrame,
    collapse: pd.DataFrame,
    confidence: pd.DataFrame,
    slice_warnings: pd.DataFrame,
    mouse_warnings: pd.DataFrame,
    final_figures: list[Path],
    figure_warnings: list[str],
    outputs: dict[str, Path],
    final_observed_time: str,
) -> str:
    entropy = metric_summary.set_index("metric")
    tier_counts = terminal_tier_counts(confidence)
    top_conf = confidence[["terminal_macrostate_label", "confidence_tier", "confidence_reason"]].to_dict("records")
    lines = [
        "# M4C-02 Markov Fate Final Review And Freeze",
        "",
        M4C_SEMANTIC_NOTE,
        "Terminal macrostates are M4B candidate final-time niche macrostates, not proven biological terminal fates.",
        f"{final_observed_time} is the final observed time point, not necessarily a true biological endpoint.",
        BARCODE_COMPATIBLE_INTERPRETATION_NOTE,
        "",
        "## Numerical QC",
        f"- fate matrix shape: {qc.get('fate_matrix_shape')}",
        f"- row-sum lower QC bound: {float(qc.get('row_sum_min_qc_bound', np.nan)):.9g}",
        f"- row-sum upper QC bound: {float(qc.get('row_sum_max_qc_bound', np.nan)):.9g}",
        f"- non-final row-sum max error: {float(qc.get('nonfinal_row_sum_error_max', np.nan)):.6g}",
        f"- final one-hot error max: {float(qc.get('final_onehot_error_max', np.nan)):.6g}",
        f"- NaN values: {int(qc.get('nan_values', 0))}",
        f"- negative values: {int(qc.get('negative_values', 0))}",
        f"- fate probabilities recomputed in M4C-02: {FATE_PROBABILITIES_RECOMPUTED}",
        f"- fate matrix loaded in M4C-02: {FATE_MATRIX_LOADED}",
        "",
        "## Entropy And Confidence Summary",
        f"- mean entropy: {entropy.loc['plasticity_entropy', 'mean']:.6g}",
        f"- mean normalized plasticity: {entropy.loc['normalized_plasticity_entropy', 'mean']:.6g}",
        f"- mean top1 probability: {entropy.loc['dominant_fate_probability', 'mean']:.6g}",
        f"- mean top1-minus-top2 margin: {entropy.loc['fate_margin_top1_minus_top2', 'mean']:.6g}",
        "",
        "## Dominant Fate Composition By Time",
    ]
    for row in dominant_top.to_dict("records"):
        lines.append(
            f"- {row['time']}: {row['terminal_macrostate_label']} "
            f"dominant_fate_fraction={float(row['dominant_fate_fraction']):.6g}, "
            f"mean_probability={float(row['mean_probability']):.6g}"
        )
    lines.extend(["", "## Timepoint Structure Flags"])
    for row in time_flags.to_dict("records"):
        lines.append(
            f"- {row['time']}: normalized_plasticity_mean={float(row['normalized_plasticity_entropy_mean']):.6g}, "
            f"top1_mean={float(row['dominant_fate_probability_mean']):.6g}, "
            f"margin_mean={float(row['fate_margin_top1_minus_top2_mean']):.6g}, "
            f"flags={row['timepoint_review_label']}"
        )
    lines.extend(["", "## Dominant Fate Collapse Review"])
    if collapse.empty:
        lines.append("- No terminal macrostate dominated every reviewed time point.")
    else:
        for row in collapse.to_dict("records"):
            lines.append(
                f"- {row['terminal_macrostate_label']}: top_time_points={row['top_time_points']}/"
                f"{row['n_time_points']}, warning={row['warning']}"
            )
    lines.extend(["", "## Terminal Macrostate Confidence Tiers"])
    lines.extend([f"- {tier}: {count}" for tier, count in tier_counts.items()])
    for row in top_conf:
        lines.append(f"- {row['terminal_macrostate_label']}: {row['confidence_tier']} ({row['confidence_reason']})")
    lines.extend(
        [
            "",
            "## Slice And Mouse Stability Warnings",
            f"- slice association warnings: {int(slice_warnings['association_warning'].sum())}",
            f"- mouse association warnings: {int(mouse_warnings['association_warning'].sum())}",
            "",
            "## Figures",
            *[f"- {path}" for path in final_figures],
            "",
            "## Freeze Artifacts",
            f"- final review report: {outputs['final_review_report']}",
            f"- freeze summary JSON: {outputs['freeze_summary']}",
            f"- confidence tiers CSV: {outputs['confidence_tiers']}",
            f"- interpretation cautions: {outputs['interpretation_cautions']}",
            f"- result inventory: {outputs['result_inventory']}",
        ]
    )
    if figure_warnings:
        lines.extend(["", "## Figure Warnings", *[f"- {warning}" for warning in figure_warnings]])
    lines.extend(
        [
            "",
            "## Not Run",
            "- GPCCA was not run.",
            "- Branched NicheFlow / BranchSBM training was not run.",
            "- M5 was not run.",
            "- Regulator analysis was not run.",
            "",
            "## Next Recommendation",
            "Review these generated figures manually with the user, then choose whether to accept the Markov fate baseline, "
            "refine terminal macrostate interpretation only, design a coarse-grained GPCCA consistency review, or begin planning "
            "the Branched NicheFlow / BranchSBM route. No next-stage implementation should start without explicit approval.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def freeze_summary_payload(
    qc: dict[str, Any],
    metric_summary: pd.DataFrame,
    time_flags: pd.DataFrame,
    collapse: pd.DataFrame,
    confidence: pd.DataFrame,
    slice_warnings: pd.DataFrame,
    mouse_warnings: pd.DataFrame,
    outputs: dict[str, Path],
    final_figures: list[Path],
    figure_warnings: list[str],
    runtime_seconds: float,
    final_observed_time: str,
) -> dict[str, Any]:
    return {
        "schema_version": "m4c_markov_fate_final_freeze_v1",
        "generated_at_utc": utc_now_iso(),
        "freeze_status": "frozen_for_manual_visual_review",
        "m4c_semantic_note": M4C_SEMANTIC_NOTE,
        "final_observed_time": final_observed_time,
        "barcode_compatible_interpretation_note": BARCODE_COMPATIBLE_INTERPRETATION_NOTE,
        "fate_probabilities_recomputed": FATE_PROBABILITIES_RECOMPUTED,
        "fate_matrix_loaded_for_review": FATE_MATRIX_LOADED,
        "numerical_qc": qc,
        "metric_summary": metric_summary.to_dict("records"),
        "timepoint_flags": time_flags.to_dict("records"),
        "dominant_fate_collapse_warnings": collapse.to_dict("records"),
        "confidence_tier_counts": terminal_tier_counts(confidence),
        "slice_association_warning_count": int(slice_warnings["association_warning"].sum()),
        "mouse_association_warning_count": int(mouse_warnings["association_warning"].sum()),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "figures": [str(path) for path in final_figures],
        "figure_warnings": figure_warnings,
        "runtime_seconds": float(runtime_seconds),
        **NO_DOWNSTREAM_FLAGS,
    }


def run(args: argparse.Namespace) -> int:
    start = time.monotonic()
    config = load_config(args.config)
    paths = configured_paths(config)
    inputs = review_input_paths(paths)
    outputs = review_output_paths(paths)
    frames = load_review_inputs(inputs)

    qc = scalar_qc(frames["qc"])
    node_summary = frames["node_summary"]
    by_time = frames["by_time"]
    by_slice = frames["by_slice"]
    by_mouse = frames["by_mouse"]
    terminal_summary = frames["terminal_summary"]
    feature_summary = frames["terminal_feature_summary"]

    if int(qc.get("terminal_macrostates", 0)) != int(len(terminal_summary)):
        raise ValueError("Terminal macrostate count in M4C QC does not match M4B terminal summary.")
    if set(node_summary["directionality_evidence_source"].astype(str).unique()) != {"pseudo_lineage_time_coupled_transition"}:
        raise ValueError("Node summary directionality evidence metadata does not match M4C contract.")
    if set(node_summary["barcode_compatible_contract"].astype(bool).unique()) != {True}:
        raise ValueError("Node summary barcode-compatible contract metadata is not true for all nodes.")

    metric_summary = metric_stats(
        node_summary,
        [
            "plasticity_entropy",
            "normalized_plasticity_entropy",
            "dominant_fate_probability",
            "fate_margin_top1_minus_top2",
        ],
    )
    time_metrics = metrics_by_time(node_summary)
    final_observed_time = final_time_label(by_time)
    time_flags = flag_timepoint_structure(time_metrics)
    fate_mass = terminal_fate_mass_summary(by_time, node_summary)
    collapse = detect_dominant_fate_collapse(by_time)
    slice_variability = group_variability_summary(by_slice, "slice_id")
    mouse_variability = group_variability_summary(by_mouse, "mouse_id")
    slice_warnings = association_warnings(slice_variability)
    mouse_warnings = association_warnings(mouse_variability)
    confidence = assign_confidence_tiers(terminal_summary, fate_mass, slice_warnings, mouse_warnings, feature_summary)

    if len(confidence) != int(qc.get("terminal_macrostates", len(confidence))):
        raise ValueError("M4C-02 confidence review changed the terminal macrostate count.")

    figure_warnings, final_figures = generate_figures(
        outputs["final_figures_dir"],
        by_time,
        by_slice,
        by_mouse,
        time_flags,
        fate_mass,
        confidence,
        bool(config.get("visualization", {}).get("figure_failure_is_warning", True)),
    )
    runtime = time.monotonic() - start

    atomic_write_csv(outputs["confidence_tiers"], confidence)
    atomic_write_text(
        outputs["interpretation_cautions"],
        interpretation_cautions_text(confidence, slice_warnings, mouse_warnings, final_observed_time),
    )
    atomic_write_text(
        outputs["final_review_report"],
        final_review_report_text(
            qc,
            metric_summary,
            time_flags,
            dominant_fate_by_time(by_time),
            collapse,
            confidence,
            slice_warnings,
            mouse_warnings,
            final_figures,
            figure_warnings,
            outputs,
            final_observed_time,
        ),
    )
    atomic_write_json(
        outputs["freeze_summary"],
        freeze_summary_payload(
            qc,
            metric_summary,
            time_flags,
            collapse,
            confidence,
            slice_warnings,
            mouse_warnings,
            outputs,
            final_figures,
            figure_warnings,
            runtime,
            final_observed_time,
        ),
    )
    inventory = inventory_rows(inputs, outputs, final_figures)
    atomic_write_csv(outputs["result_inventory"], inventory)
    inventory = inventory_rows(inputs, outputs, final_figures)
    atomic_write_csv(outputs["result_inventory"], inventory)

    print("M4C_02_MARKOV_FATE_FINAL_REVIEW_COMPLETE")
    print(f"GLOBAL_NODES {qc.get('global_nodes')}")
    print(f"TERMINAL_MACROSTATES {qc.get('terminal_macrostates')}")
    print(f"FATE_MATRIX_SHAPE {qc.get('fate_matrix_shape')}")
    print(f"FATE_PROBABILITIES_RECOMPUTED {FATE_PROBABILITIES_RECOMPUTED}")
    print(f"FATE_MATRIX_LOADED {FATE_MATRIX_LOADED}")
    print(f"CONFIDENCE_TIERS {terminal_tier_counts(confidence)}")
    print(f"SLICE_ASSOCIATION_WARNINGS {int(slice_warnings['association_warning'].sum())}")
    print(f"MOUSE_ASSOCIATION_WARNINGS {int(mouse_warnings['association_warning'].sum())}")
    print(f"FINAL_REVIEW_REPORT {outputs['final_review_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
