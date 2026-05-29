from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def distribution_summary(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    quantiles = values.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return pd.DataFrame(
        [
            {
                "value_column": value_column,
                "n": int(values.notna().sum()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0,
                "min": float(quantiles.loc[0]),
                "q05": float(quantiles.loc[0.05]),
                "q25": float(quantiles.loc[0.25]),
                "median": float(quantiles.loc[0.5]),
                "q75": float(quantiles.loc[0.75]),
                "q95": float(quantiles.loc[0.95]),
                "max": float(quantiles.loc[1.0]),
            }
        ]
    )


def group_centroid_table(group_assignment: pd.DataFrame, group_summary: pd.DataFrame) -> pd.DataFrame:
    centroids = (
        group_assignment.groupby(["sample_id", "slice_id", "group_id"], as_index=False)
        .agg(
            anchor_x=("anchor_x", "first"),
            anchor_y=("anchor_y", "first"),
            group_x=("x", "mean"),
            group_y=("y", "mean"),
            n_assignment_rows=("cellbin_id", "size"),
        )
    )
    keep = [
        "sample_id",
        "slice_id",
        "group_id",
        "total_lineage_count",
        "detected_feature_count",
        "feature_entropy",
        "fraction_member_cellbins_with_lineage_evidence",
    ]
    return centroids.merge(group_summary[keep], on=["sample_id", "slice_id", "group_id"], how="left")


def save_histogram(frame: pd.DataFrame, value_column: str, title: str, path_base: Path) -> list[Path]:
    values = pd.to_numeric(frame[value_column], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=40, color="#4C78A8", edgecolor="white", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(value_column)
    ax.set_ylabel("Groups")
    fig.tight_layout()
    outputs = []
    for suffix in [".png", ".pdf"]:
        out = path_base.with_suffix(suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        outputs.append(out)
    plt.close(fig)
    return outputs


def save_spatial_map(frame: pd.DataFrame, value_column: str, title: str, path_base: Path) -> list[Path]:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    scatter = ax.scatter(frame["anchor_x"], frame["anchor_y"], c=values, s=7, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("anchor_x")
    ax.set_ylabel("anchor_y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(scatter, ax=ax, label=value_column)
    fig.tight_layout()
    outputs = []
    for suffix in [".png", ".pdf"]:
        out = path_base.with_suffix(suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        outputs.append(out)
    plt.close(fig)
    return outputs
