"""M4V-01 visualizations for M4C baseline and M4D GPCCA outputs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA

from nichefate.m4d_standard_gpcca import (
    M2_META_COLUMNS,
    SELECTED_GROUPS,
    assert_no_ssd_path,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    discover_m2_files,
    json_safe,
    load_feature_columns,
    m4d_paths,
    node_lookup_by_slice,
    parquet_columns,
    read_m2_with_global,
    require_parquet_engine,
    validate_existing_outputs,
)


@dataclass(frozen=True)
class M4VPaths:
    m4c_summary: Path
    m4d_projection: Path
    coordinate_cache: Path
    node_table: Path
    m2_root: Path
    visualization_table: Path
    reports_dir: Path
    figures_root: Path
    m4c_figures: Path
    m4d_figures: Path
    comparison_figures: Path
    state_space_figures: Path
    report_md: Path
    summary_json: Path
    figure_inventory: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def m4v_paths(config: dict[str, Any]) -> M4VPaths:
    paths = config["paths"]
    viz = config["m4_visualization"]
    reports = Path(paths["reports_dir"])
    figures = reports / "figures" / "m4v_01"
    out = M4VPaths(
        m4c_summary=Path(viz["m4c_baseline_node_summary"]),
        m4d_projection=Path(config["standard_gpcca"]["output_gpcca_root"]) / "node_gpcca_macrostate_membership.parquet",
        coordinate_cache=Path(viz["m4d_coordinate_cache"]),
        node_table=Path(paths["m4a_node_table"]),
        m2_root=Path(paths["m2_by_slice_root"]),
        visualization_table=Path(paths["visualization_dir"]) / "m4c_m4d_visualization_node_table.parquet",
        reports_dir=reports,
        figures_root=figures,
        m4c_figures=figures / "m4c_baseline",
        m4d_figures=figures / "m4d_gpcca",
        comparison_figures=figures / "comparison",
        state_space_figures=figures / "state_space",
        report_md=reports / "m4v_01_visualization_report.md",
        summary_json=reports / "m4v_01_visualization_summary.json",
        figure_inventory=reports / "m4v_01_figure_inventory.csv",
    )
    for label, path in out.__dict__.items():
        if isinstance(path, Path):
            assert_no_ssd_path(path, label)
    return out


def valid_m4d_outputs_for_visualization(config: dict[str, Any]) -> tuple[bool, str]:
    expected_nodes = int(config.get("coordinates", {}).get("expected_global_nodes", 0)) or None
    return validate_existing_outputs(m4d_paths(config), expected_nodes=expected_nodes)


def prepare_dirs(paths: M4VPaths) -> None:
    for path in [paths.reports_dir, paths.m4c_figures, paths.m4d_figures, paths.comparison_figures, paths.state_space_figures]:
        path.mkdir(parents=True, exist_ok=True)


def representative_slices(table: pd.DataFrame, preferred_times: list[str] | None = None) -> pd.DataFrame:
    preferred_times = preferred_times or ["D0", "D3", "D9", "D21", "D35"]
    counts = (
        table.groupby(["time_day", "time_label", "slice_id"], sort=True, observed=True)
        .size()
        .reset_index(name="n_nodes")
    )
    rows = []
    for time_label in preferred_times:
        group = counts.loc[counts["time_label"].astype(str) == time_label].copy()
        if group.empty:
            continue
        median = float(group["n_nodes"].median())
        group["distance_to_median"] = (group["n_nodes"] - median).abs()
        rows.append(group.sort_values(["distance_to_median", "slice_id"]).iloc[0].to_dict())
    if not rows:
        for _, group in counts.groupby("time_label", sort=True, observed=True):
            median = float(group["n_nodes"].median())
            local = group.copy()
            local["distance_to_median"] = (local["n_nodes"] - median).abs()
            rows.append(local.sort_values(["distance_to_median", "slice_id"]).iloc[0].to_dict())
    return pd.DataFrame(rows)


def build_visualization_table(paths: M4VPaths, overwrite: bool) -> pd.DataFrame:
    m4c_cols = [
        "global_node_index",
        "dominant_fate",
        "dominant_fate_label",
        "dominant_fate_probability",
        "plasticity_entropy",
        "normalized_plasticity_entropy",
        "fate_margin_top1_minus_top2",
    ]
    coord_cols = [
        "global_node_index",
        "x_raw",
        "y_raw",
        "x_scaled_by_slice",
        "y_scaled_by_slice",
    ]
    node_cols = [
        "global_node_index",
        "time",
        "time_day",
        "slice_id",
        "mouse_id",
        "cell_type_l1",
        "cell_type_l2",
        "cell_type_l3",
    ]
    m4d = pd.read_parquet(paths.m4d_projection)
    m4c = pd.read_parquet(paths.m4c_summary, columns=m4c_cols)
    coords = pd.read_parquet(paths.coordinate_cache, columns=coord_cols)
    nodes = pd.read_parquet(paths.node_table, columns=node_cols).rename(columns={"time": "time_label"})
    table = nodes.merge(coords, on="global_node_index", how="left", sort=False)
    table = table.merge(m4c, on="global_node_index", how="left", sort=False)
    table = table.merge(
        m4d[
            [
                "global_node_index",
                "gpcca_macrostate_id",
                "gpcca_macrostate_probability",
                "gpcca_membership_entropy",
            ]
            + [col for col in m4d.columns if col.startswith("gpcca_prob_")]
        ],
        on="global_node_index",
        how="left",
        sort=False,
    )
    table = table.sort_values("global_node_index", kind="mergesort").reset_index(drop=True)
    expected = np.arange(len(table), dtype=np.int64)
    if not np.array_equal(table["global_node_index"].to_numpy(dtype=np.int64), expected):
        raise ValueError("Visualization table must preserve global_node_index identity.")
    required = ["x_scaled_by_slice", "dominant_fate", "gpcca_macrostate_id"]
    missing = [col for col in required if bool(table[col].isna().any())]
    if missing:
        raise ValueError(f"Visualization table has missing required columns: {missing}")
    atomic_write_parquet(paths.visualization_table, table, overwrite=overwrite)
    return table


def figure_record(path: Path, category: str, title: str, status: str = "generated", notes: str = "") -> dict[str, Any]:
    return {
        "category": category,
        "title": title,
        "path": str(path),
        "status": status,
        "notes": notes,
    }


def with_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_bar_by_time(table: pd.DataFrame, out: Path, value: str, title: str, ylabel: str) -> None:
    plt = with_matplotlib()
    summary = table.groupby(["time_day", "time_label"], sort=True, observed=True)[value].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(summary["time_label"].astype(str), summary[value], marker="o")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def save_composition(table: pd.DataFrame, out: Path, label_col: str, title: str) -> None:
    plt = with_matplotlib()
    comp = (
        table.groupby(["time_day", "time_label", label_col], sort=True, observed=True)
        .size()
        .reset_index(name="n")
    )
    comp["fraction"] = comp["n"] / comp.groupby(["time_day", "time_label"], observed=True)["n"].transform("sum")
    pivot = comp.pivot_table(index="time_label", columns=label_col, values="fraction", fill_value=0.0, aggfunc="sum")
    order = comp[["time_day", "time_label"]].drop_duplicates().sort_values(["time_day", "time_label"])["time_label"].astype(str)
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(pivot), dtype=float)
    for column in pivot.columns:
        values = pivot[column].to_numpy(dtype=float)
        ax.bar(pivot.index.astype(str), values, bottom=bottom, label=str(column))
        bottom += values
    ax.set_title(title)
    ax.set_ylabel("fraction")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=6, ncol=4)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def save_tissue_maps(
    table: pd.DataFrame,
    slices: pd.DataFrame,
    out_dir: Path,
    color_col: str,
    title_prefix: str,
    cmap: str,
    category: str,
    inventory: list[dict[str, Any]],
) -> None:
    plt = with_matplotlib()
    for row in slices.to_dict("records"):
        subset = table.loc[table["slice_id"].astype(str) == str(row["slice_id"])].copy()
        if subset.empty:
            continue
        path = out_dir / f"{title_prefix}_{row['time_label']}_{row['slice_id']}.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(
            subset["x_scaled_by_slice"],
            subset["y_scaled_by_slice"],
            c=subset[color_col],
            s=2,
            cmap=cmap,
            linewidths=0,
        )
        ax.set_title(f"{title_prefix} {row['time_label']} {row['slice_id']}")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        inventory.append(figure_record(path, category, f"{title_prefix} {row['time_label']} {row['slice_id']}"))


def make_m4c_figures(table: pd.DataFrame, paths: M4VPaths, inventory: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    try:
        p = paths.m4c_figures
        save_composition(table, p / "m4c_dominant_fate_composition_by_time.png", "dominant_fate_label", "M4C dominant fate composition by time")
        inventory.append(figure_record(p / "m4c_dominant_fate_composition_by_time.png", "m4c_baseline", "M4C dominant fate composition by time"))
        for col, title, name in [
            ("normalized_plasticity_entropy", "M4C normalized plasticity by time", "m4c_normalized_plasticity_by_time.png"),
            ("plasticity_entropy", "M4C entropy by time", "m4c_entropy_by_time.png"),
            ("dominant_fate_probability", "M4C top1 probability by time", "m4c_top1_probability_by_time.png"),
            ("fate_margin_top1_minus_top2", "M4C top1 margin by time", "m4c_margin_by_time.png"),
        ]:
            save_bar_by_time(table, p / name, col, title, col)
            inventory.append(figure_record(p / name, "m4c_baseline", title))
        slices = representative_slices(table)
        for color_col, prefix, cmap in [
            ("dominant_fate", "m4c_dominant_fate", "tab20"),
            ("normalized_plasticity_entropy", "m4c_normalized_plasticity", "viridis"),
            ("dominant_fate_probability", "m4c_top1_probability", "magma"),
        ]:
            save_tissue_maps(table, slices, p, color_col, prefix, cmap, "m4c_baseline", inventory)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"M4C figure generation warning: {type(exc).__name__}: {exc}")
    return warnings


def make_m4d_figures(table: pd.DataFrame, paths: M4VPaths, inventory: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    try:
        p = paths.m4d_figures
        plt = with_matplotlib()
        sizes = table.groupby("gpcca_macrostate_id", observed=True).size().reset_index(name="n")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(sizes["gpcca_macrostate_id"].astype(str), sizes["n"])
        ax.set_title("M4D GPCCA macrostate size distribution")
        fig.tight_layout()
        path = p / "m4d_gpcca_macrostate_size_distribution.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        inventory.append(figure_record(path, "m4d_gpcca", "M4D GPCCA macrostate size distribution"))
        save_composition(table, p / "m4d_gpcca_macrostate_composition_by_time.png", "gpcca_macrostate_id", "M4D GPCCA macrostate composition by time")
        inventory.append(figure_record(p / "m4d_gpcca_macrostate_composition_by_time.png", "m4d_gpcca", "M4D GPCCA macrostate composition by time"))
        for col, title, name in [
            ("gpcca_macrostate_probability", "M4D GPCCA top membership probability by time", "m4d_gpcca_top_probability_by_time.png"),
            ("gpcca_membership_entropy", "M4D GPCCA membership entropy by time", "m4d_gpcca_entropy_by_time.png"),
        ]:
            save_bar_by_time(table, p / name, col, title, col)
            inventory.append(figure_record(p / name, "m4d_gpcca", title))
        slices = representative_slices(table)
        for color_col, prefix, cmap in [
            ("gpcca_macrostate_id", "m4d_gpcca_dominant_macrostate", "tab20"),
            ("gpcca_macrostate_probability", "m4d_gpcca_macrostate_probability", "magma"),
            ("gpcca_membership_entropy", "m4d_gpcca_membership_entropy", "viridis"),
        ]:
            save_tissue_maps(table, slices, p, color_col, prefix, cmap, "m4d_gpcca", inventory)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"M4D GPCCA figure generation warning: {type(exc).__name__}: {exc}")
    return warnings


def make_comparison_figures(table: pd.DataFrame, paths: M4VPaths, inventory: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    try:
        p = paths.comparison_figures
        plt = with_matplotlib()
        overlap = table.pivot_table(
            index="gpcca_macrostate_id",
            columns="dominant_fate_label",
            values="global_node_index",
            aggfunc="count",
            fill_value=0,
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(overlap.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="cividis")
        ax.set_title("M4C dominant fate vs M4D GPCCA macrostate")
        ax.set_xticks(np.arange(len(overlap.columns)))
        ax.set_xticklabels(overlap.columns.astype(str), rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(overlap.index)))
        ax.set_yticklabels(overlap.index.astype(str))
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        path = p / "m4c_vs_m4d_overlap_heatmap.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        inventory.append(figure_record(path, "comparison", "M4C dominant fate vs M4D GPCCA macrostate overlap heatmap"))

        agreement = (
            table.groupby(["time_day", "time_label", "dominant_fate_label", "gpcca_macrostate_id"], observed=True)
            .size()
            .reset_index(name="n")
        )
        major = agreement.sort_values(["time_day", "time_label", "n"], ascending=[True, True, False]).drop_duplicates(["time_day", "time_label"])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(major["time_label"].astype(str), major["n"])
        ax.set_title("Time-stratified strongest M4C/M4D overlap count")
        fig.tight_layout()
        path = p / "time_stratified_m4c_m4d_agreement_summary.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        inventory.append(figure_record(path, "comparison", "Time-stratified M4C vs M4D agreement summary"))

        slice_major = (
            table.groupby(["slice_id", "dominant_fate_label", "gpcca_macrostate_id"], observed=True)
            .size()
            .reset_index(name="n")
        )
        heat = slice_major.pivot_table(index="slice_id", columns="gpcca_macrostate_id", values="n", aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(10, max(5, min(14, 0.18 * len(heat)))))
        im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title("Slice-level M4D GPCCA macrostate counts")
        y_step = max(1, len(heat) // 25)
        ax.set_yticks(np.arange(0, len(heat), y_step))
        ax.set_yticklabels(heat.index.astype(str)[::y_step], fontsize=6)
        ax.set_xticks(np.arange(len(heat.columns)))
        ax.set_xticklabels(heat.columns.astype(str))
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        path = p / "slice_level_agreement_heatmap.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        inventory.append(figure_record(path, "comparison", "Slice-level agreement heatmap"))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(table["normalized_plasticity_entropy"], table["gpcca_membership_entropy"], s=1, alpha=0.15)
        ax.set_xlabel("M4C normalized plasticity")
        ax.set_ylabel("M4D GPCCA membership entropy")
        ax.set_title("M4C vs M4D entropy comparison")
        fig.tight_layout()
        path = p / "macrostate_fate_entropy_comparison.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        inventory.append(figure_record(path, "comparison", "Macrostate/fate entropy comparison"))

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].imshow(overlap.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="cividis")
        axes[0, 0].set_title("Overlap")
        axes[0, 1].hist(table["dominant_fate_probability"], bins=40)
        axes[0, 1].set_title("M4C top1 probability")
        axes[1, 0].hist(table["gpcca_macrostate_probability"], bins=40)
        axes[1, 0].set_title("M4D GPCCA top membership")
        axes[1, 1].scatter(table["normalized_plasticity_entropy"], table["gpcca_membership_entropy"], s=1, alpha=0.15)
        axes[1, 1].set_title("Entropy relation")
        fig.tight_layout()
        path = p / "m4c_m4d_summary_dashboard.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        inventory.append(figure_record(path, "comparison", "Summary dashboard"))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Comparison figure generation warning: {type(exc).__name__}: {exc}")
    return warnings


def state_space_sample(config: dict[str, Any], table: pd.DataFrame, max_rows: int = 80000) -> tuple[np.ndarray, pd.DataFrame, str]:
    m2_root = Path(config["paths"]["m2_by_slice_root"])
    files = discover_m2_files(m2_root)
    sample_cols = parquet_columns(files[0])
    feature_cols, source = load_feature_columns(
        Path(config["paths"]["m2_feature_groups"]),
        Path("/home/zhutao/scratch/nichefate/m3/reports/m3_feature_groups.json"),
        Path("/home/zhutao/scratch/nichefate/m2/reports/m2_full_feature_schema.json"),
        SELECTED_GROUPS,
        sample_cols,
    )
    nodes = pd.read_parquet(config["paths"]["m4a_node_table"], columns=["global_node_index", "slice_id", "anchor_index", "time", "time_day"])
    nodes["time_label"] = nodes["time"].astype(str)
    lookup = node_lookup_by_slice(nodes)
    rows: list[pd.DataFrame] = []
    per_file = max(1, int(np.ceil(max_rows / len(files))))
    for path in files:
        frame = read_m2_with_global(path, feature_cols, lookup)
        step = max(1, len(frame) // per_file)
        sampled = frame.iloc[::step].head(per_file).copy()
        rows.append(sampled[["global_node_index"] + feature_cols])
    features = pd.concat(rows, ignore_index=True).head(max_rows)
    meta_cols = [
        "global_node_index",
        "dominant_fate",
        "dominant_fate_label",
        "normalized_plasticity_entropy",
        "gpcca_macrostate_id",
        "gpcca_membership_entropy",
    ]
    meta = features[["global_node_index"]].merge(table[meta_cols], on="global_node_index", how="left", sort=False)
    matrix = features[feature_cols].to_numpy(dtype=np.float32, copy=True)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, meta, source


def make_state_space_figures(config: dict[str, Any], table: pd.DataFrame, paths: M4VPaths, inventory: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    try:
        matrix, meta, source = state_space_sample(config, table)
        if len(matrix) > 100000:
            pca = IncrementalPCA(n_components=2, batch_size=20000)
            coords = pca.fit_transform(matrix)
        else:
            coords = PCA(n_components=2, random_state=1).fit_transform(matrix)
        meta = meta.copy()
        meta["pca1"] = coords[:, 0]
        meta["pca2"] = coords[:, 1]
        plt = with_matplotlib()
        for color_col, title, name, cmap in [
            ("dominant_fate", "PCA colored by M4C dominant fate", "pca_m4c_dominant_fate.png", "tab20"),
            ("normalized_plasticity_entropy", "PCA colored by M4C plasticity", "pca_m4c_plasticity.png", "viridis"),
            ("gpcca_macrostate_id", "PCA colored by M4D GPCCA macrostate", "pca_m4d_gpcca_macrostate.png", "tab20"),
            ("gpcca_membership_entropy", "PCA colored by M4D GPCCA entropy", "pca_m4d_gpcca_entropy.png", "magma"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 5))
            sc = ax.scatter(meta["pca1"], meta["pca2"], c=meta[color_col], s=2, cmap=cmap, linewidths=0)
            ax.set_title(title)
            fig.colorbar(sc, ax=ax, fraction=0.04)
            fig.tight_layout()
            path = paths.state_space_figures / name
            fig.savefig(path, dpi=140)
            plt.close(fig)
            inventory.append(figure_record(path, "state_space", title, notes=f"feature_source={source}"))
        warnings.append("State-space transition vector field skipped in this first pass; no tissue-space arrows were drawn.")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"State-space figure generation warning: {type(exc).__name__}: {exc}")
    return warnings


def build_m4c_only_table(paths: M4VPaths) -> pd.DataFrame:
    node_cols = ["global_node_index", "time", "time_day", "slice_id", "mouse_id"]
    m4c_cols = [
        "global_node_index",
        "dominant_fate",
        "dominant_fate_label",
        "dominant_fate_probability",
        "plasticity_entropy",
        "normalized_plasticity_entropy",
        "fate_margin_top1_minus_top2",
    ]
    coord_cols = ["global_node_index", "x_scaled_by_slice", "y_scaled_by_slice"]
    nodes = pd.read_parquet(paths.node_table, columns=node_cols).rename(columns={"time": "time_label"})
    return (
        nodes.merge(pd.read_parquet(paths.coordinate_cache, columns=coord_cols), on="global_node_index", how="left", sort=False)
        .merge(pd.read_parquet(paths.m4c_summary, columns=m4c_cols), on="global_node_index", how="left", sort=False)
        .sort_values("global_node_index", kind="mergesort")
        .reset_index(drop=True)
    )


def visualization_report(summary: dict[str, Any]) -> str:
    lines = [
        "# M4V-01 Visualization Report",
        "",
        f"- generated at UTC: {summary['generated_at_utc']}",
        f"- status: `{summary['status']}`",
        f"- M4D gate passed: `{summary['m4d_gate_passed']}`",
        f"- M4D gate message: {summary['m4d_gate_message']}",
        "- M4C outputs are Markov baseline fate probabilities.",
        "- M4D outputs are GPCCA macrostate memberships / macrostate projections.",
        "- M4D outputs are not fate probabilities and no absorption probability was computed.",
        "- cross-time physical tissue arrows drawn: `False`",
        f"- figures generated: `{summary['figures_generated']}`",
        "",
        "## Warnings",
    ]
    warnings = summary.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    return "\n".join(lines) + "\n"


def run_m4v_01(config: dict[str, Any], resume: bool = False, overwrite: bool = False) -> dict[str, Any]:
    if not config.get("m4_visualization", {}).get("enabled", False):
        raise ValueError("m4_visualization.enabled must be true.")
    if bool(config["m4_visualization"].get("cross_time_physical_arrows_allowed", False)):
        raise ValueError("Cross-time physical arrows are forbidden for M4V-01.")
    parquet_status = require_parquet_engine()
    paths = m4v_paths(config)
    prepare_dirs(paths)
    gate_ok, gate_msg = valid_m4d_outputs_for_visualization(config)
    inventory: list[dict[str, Any]] = []
    warnings: list[str] = []
    if resume and paths.summary_json.exists() and paths.figure_inventory.exists() and gate_ok:
        summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
        summary["status"] = "resumed_existing_outputs"
        return summary
    if not gate_ok:
        table = build_m4c_only_table(paths)
        warnings.extend(make_m4c_figures(table, paths, inventory))
        status = "m4c_only_m4d_skipped"
        warnings.append("Full M4D visualization skipped because M4D-01c outputs did not pass the gate.")
    else:
        table = build_visualization_table(paths, overwrite=overwrite)
        warnings.extend(make_m4c_figures(table, paths, inventory))
        warnings.extend(make_m4d_figures(table, paths, inventory))
        warnings.extend(make_comparison_figures(table, paths, inventory))
        warnings.extend(make_state_space_figures(config, table, paths, inventory))
        status = "completed"
    atomic_write_csv(paths.figure_inventory, pd.DataFrame(inventory), overwrite=overwrite)
    summary = {
        "schema_version": "m4v_01_visualization_summary_v1",
        "generated_at_utc": utc_now_iso(),
        "status": status,
        "parquet_status": parquet_status,
        "m4d_gate_passed": bool(gate_ok),
        "m4d_gate_message": gate_msg,
        "visualization_table": str(paths.visualization_table) if gate_ok else "",
        "figures_generated": int(sum(1 for item in inventory if item["status"] == "generated")),
        "warnings": warnings,
        "tissue_maps_enabled": bool(config["m4_visualization"].get("tissue_maps_enabled", True)),
        "cross_time_physical_arrows_drawn": False,
        "state_space_arrows_status": "skipped_with_warning",
        "terminology": {
            "m4c": "Markov baseline fate probabilities",
            "m4d": "GPCCA macrostate memberships / macrostate projections",
            "no_absorption_probability": True,
        },
    }
    atomic_write_json(paths.summary_json, summary, overwrite=overwrite)
    atomic_write_text(paths.report_md, visualization_report(summary), overwrite=overwrite)
    return summary
