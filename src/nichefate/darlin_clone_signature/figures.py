from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .reporting import ensure_dir


def _save(path: Path) -> Path:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def make_round2_figures(
    report_root: Path,
    evidence: pd.DataFrame,
    feature_reference: pd.DataFrame,
    edges: pd.DataFrame,
    signatures: pd.DataFrame,
    assignment: pd.DataFrame,
    membership: pd.DataFrame,
    tile_summary: pd.DataFrame,
    null_comparison: pd.DataFrame,
    bridge_sensitivity: pd.DataFrame,
) -> tuple[list[Path], dict[str, Any]]:
    figure_dir = ensure_dir(report_root / "figures")
    paths: list[Path] = []

    plt.figure(figsize=(7, 4))
    feature_reference["cellbin_fraction"].clip(lower=1e-8).hist(bins=60)
    plt.xscale("log")
    plt.xlabel("Cellbin fraction")
    plt.ylabel("Feature count")
    plt.title("Feature rarity distribution")
    paths.append(_save(figure_dir / "feature_rarity_distribution.png"))

    plt.figure(figsize=(6, 4))
    class_counts = signatures["clone_class"].value_counts() if not signatures.empty else pd.Series(dtype=int)
    class_counts.plot(kind="bar", color=["#4464ad", "#4c956c", "#d58936"][: max(1, len(class_counts))])
    plt.ylabel("Clone count")
    plt.title("Clone class distribution")
    paths.append(_save(figure_dir / "clone_class_distribution.png"))

    plt.figure(figsize=(6, 4))
    if not edges.empty:
        edges["observed_shared_cellbins"].hist(bins=40)
    plt.xlabel("Observed shared cellbins")
    plt.ylabel("Edge count")
    plt.title("Feature compatibility graph summary")
    paths.append(_save(figure_dir / "feature_compatibility_graph_summary.png"))

    plt.figure(figsize=(6, 4))
    status_counts = assignment["assignment_status"].value_counts() if not assignment.empty else pd.Series(dtype=int)
    status_counts.plot(kind="bar", color="#2a9d8f")
    plt.ylabel("Cellbins")
    plt.title("Clone assignment status")
    paths.append(_save(figure_dir / "clone_assignment_status_distribution.png"))

    plt.figure(figsize=(6, 4))
    assigned = assignment.loc[assignment["assignment_score"].gt(0), "assignment_score"]
    if not assigned.empty:
        assigned.hist(bins=50)
    plt.xlabel("Assignment score")
    plt.ylabel("Cellbins")
    plt.title("Assignment score distribution")
    paths.append(_save(figure_dir / "assignment_score_distribution.png"))

    plt.figure(figsize=(6, 4))
    margins = assignment.loc[assignment["assignment_score"].gt(0), "score_margin"]
    if not margins.empty:
        margins.hist(bins=50)
    plt.xlabel("Score margin")
    plt.ylabel("Cellbins")
    plt.title("Score margin distribution")
    paths.append(_save(figure_dir / "score_margin_distribution.png"))

    if not membership.empty:
        clone_sizes = membership.groupby("clone_id")["membership_weight"].sum().sort_values(ascending=False)
    else:
        clone_sizes = pd.Series(dtype=float)
    plt.figure(figsize=(6, 4))
    if not clone_sizes.empty:
        clone_sizes.reset_index(drop=True).plot()
    plt.xlabel("Clone rank")
    plt.ylabel("Weighted cellbins")
    plt.title("Clone size distribution")
    paths.append(_save(figure_dir / "clone_size_distribution.png"))

    for metric in ["fraction_clone_assigned", "clone_entropy", "dominant_clone_fraction", "clone_richness"]:
        if metric in tile_summary:
            plt.figure(figsize=(6, 4))
            values = tile_summary[metric].astype(float)
            values.hist(bins=40)
            plt.xlabel(metric)
            plt.ylabel("Tiles")
            plt.title(f"Tile {metric}")
            paths.append(_save(figure_dir / f"tile_{metric}.png"))

    plt.figure(figsize=(6, 4))
    if not null_comparison.empty:
        null_comparison.set_index(["clone_set", "null_control"])["n_clones"].plot(kind="bar")
    plt.ylabel("Clone count")
    plt.title("Real model vs null controls")
    paths.append(_save(figure_dir / "real_vs_null_clone_counts.png"))

    plt.figure(figsize=(6, 4))
    if not bridge_sensitivity.empty:
        bridge_sensitivity.groupby("bridge_filtering")["n_high_confidence_clones"].median().plot(kind="bar")
    plt.ylabel("Median high-confidence clones")
    plt.title("Bridge filtering sensitivity")
    paths.append(_save(figure_dir / "bridge_filtering_sensitivity.png"))

    key_dir = ensure_dir(report_root / "key_figure_candidates")
    key_paths = []
    for path in paths[:8]:
        target = key_dir / path.name
        shutil.copy2(path, target)
        key_paths.append(target)
    payload = {
        "figure_count": len(paths),
        "key_figure_count": len(key_paths),
        "figures": [str(path) for path in paths],
        "key_figure_candidates": [str(path) for path in key_paths],
    }
    return paths, payload
