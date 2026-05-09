#!/usr/bin/env python
"""M4E-03 figure/report QC patch.

This patch is intentionally read-only with respect to upstream M3/M4A/M4B/M4C
production artifacts. It reads existing M4E-03 refinement outputs and M4E
annotation tables, then writes corrected QC figures and report notes under M4E.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


ROOT = Path("/home/zhutao/scratch/nichefate")
M4E = ROOT / "m4e"
REFINEMENT = M4E / "endpoint_refinement"
NEIGHBORHOOD = M4E / "neighborhood_annotation"
REPORTS = M4E / "reports"
QC_FIGURES = REPORTS / "figures" / "endpoint_refinement_qc"

MAPPING_CSV = REFINEMENT / "refined_endpoint_mapping.csv"
EVIDENCE_CSV = REFINEMENT / "endpoint_refinement_evidence_table.csv"
LEIDEN_COUNTS_CSV = NEIGHBORHOOD / "endpoint_by_leiden_neigh_counts.csv"
LEIDEN_FRACTION_ENDPOINT_CSV = NEIGHBORHOOD / "endpoint_by_leiden_neigh_fraction_by_endpoint.csv"
LEIDEN_FRACTION_NEIGHBORHOOD_CSV = NEIGHBORHOOD / "endpoint_by_leiden_neigh_fraction_by_neighborhood.csv"
LEIDEN_ENRICHMENT_CSV = NEIGHBORHOOD / "endpoint_by_leiden_neigh_enrichment.csv"
NODE_NEIGHBORHOOD_PARQUET = NEIGHBORHOOD / "node_neighborhood_annotation.parquet"

BY_TIME_CSV = REFINEMENT / "m4c_v1_refined_endpoint_by_time.csv"
BY_SLICE_CSV = REFINEMENT / "m4c_v1_refined_endpoint_by_slice.csv"
BY_MOUSE_CSV = REFINEMENT / "m4c_v1_refined_endpoint_by_mouse.csv"

SHORT_LABEL_CSV = REPORTS / "refined_endpoint_short_label_mapping.csv"
QC_NOTE_MD = REPORTS / "m4e03_leiden_heatmap_qc_note.md"
INVENTORY_CSV = REPORTS / "m4e03_figure_qc_inventory.csv"
FREEZE_REPORT_MD = REPORTS / "m4c_v1_baseline_dynamic_niche_fate_freeze_report.md"


SHORT_LABELS = {
    0: "ME_SMC2",
    1: "MU_Stem",
    2: "Slice_SM",
    3: "SM_Mixed",
    4: "Rare_B",
    5: "MU_Mixed",
    6: "Rare_ME_Enteric_06",
    7: "ME_SMC1",
    8: "Rare_ME_Fibro",
    9: "Rare_ME_Enteric_09",
    10: "FOL_B",
    11: "MU_Colonocyte",
}

PALETTE = {
    0: "#7f7f7f",
    1: "#1b9e77",
    2: "#d95f02",
    3: "#7570b3",
    4: "#e7298a",
    5: "#66a61e",
    6: "#e6ab02",
    7: "#a6761d",
    8: "#1f78b4",
    9: "#b2df8a",
    10: "#fb9a99",
    11: "#6a3d9a",
}


def ensure_dirs() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    QC_FIGURES.mkdir(parents=True, exist_ok=True)


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required M4E-03 QC input(s): " + ", ".join(missing))


def read_inputs() -> dict[str, pd.DataFrame]:
    require_files(
        [
            MAPPING_CSV,
            EVIDENCE_CSV,
            LEIDEN_COUNTS_CSV,
            LEIDEN_FRACTION_ENDPOINT_CSV,
            LEIDEN_FRACTION_NEIGHBORHOOD_CSV,
            LEIDEN_ENRICHMENT_CSV,
            NODE_NEIGHBORHOOD_PARQUET,
            BY_TIME_CSV,
            BY_SLICE_CSV,
            BY_MOUSE_CSV,
        ]
    )
    return {
        "mapping": pd.read_csv(MAPPING_CSV),
        "evidence": pd.read_csv(EVIDENCE_CSV),
        "leiden_counts": pd.read_csv(LEIDEN_COUNTS_CSV),
        "leiden_fraction_endpoint": pd.read_csv(LEIDEN_FRACTION_ENDPOINT_CSV),
        "leiden_fraction_neighborhood": pd.read_csv(LEIDEN_FRACTION_NEIGHBORHOOD_CSV),
        "leiden_enrichment": pd.read_csv(LEIDEN_ENRICHMENT_CSV),
        "by_time": pd.read_csv(BY_TIME_CSV),
        "by_slice": pd.read_csv(BY_SLICE_CSV),
        "by_mouse": pd.read_csv(BY_MOUSE_CSV),
    }


def make_short_label_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    out = mapping.copy()
    out["raw_endpoint_id"] = out["raw_terminal_macrostate"].map(
        lambda value: f"terminal_macrostate_{int(value):02d}"
    )
    out["plot_short_label"] = out["raw_terminal_macrostate"].map(
        lambda value: SHORT_LABELS[int(value)]
    )
    out["plot_color"] = out["raw_terminal_macrostate"].map(lambda value: PALETTE[int(value)])
    columns = [
        "raw_terminal_macrostate",
        "raw_endpoint_id",
        "raw_terminal_macrostate_label",
        "plot_short_label",
        "plot_color",
        "refined_endpoint_id",
        "refined_endpoint_label",
        "refined_endpoint_category",
        "action",
        "label_validation_status",
        "confidence_tier_after_refinement",
    ]
    out = out[columns].sort_values("raw_terminal_macrostate")
    out.to_csv(SHORT_LABEL_CSV, index=False)
    return out


def add_plot_labels(table: pd.DataFrame, short_mapping: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "raw_terminal_macrostate",
        "plot_short_label",
        "refined_endpoint_id",
        "refined_endpoint_label",
        "confidence_tier_after_refinement",
    ]
    label_map = short_mapping[cols].rename(
        columns={"raw_terminal_macrostate": "candidate_endpoint"}
    )
    return table.merge(label_map, on="candidate_endpoint", how="left")


def pivot_heatmap(
    table: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    row_order: list[str],
) -> pd.DataFrame:
    pivot = table.pivot_table(
        index=row_col,
        columns=col_col,
        values=value_col,
        aggfunc="sum",
        fill_value=0.0,
    )
    pivot = pivot.reindex(row_order)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot.fillna(0.0)


def save_heatmap(
    pivot: pd.DataFrame,
    path: Path,
    title: str,
    colorbar_label: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: LogNorm | None = None,
    cmap: str = "viridis",
) -> None:
    fig_w = max(7.5, 0.55 * pivot.shape[1] + 5.5)
    fig_h = max(5.5, 0.38 * pivot.shape[0] + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    ax.set_title(title)
    ax.set_xlabel("Leiden neighborhood")
    ax.set_ylabel("Raw endpoint with refined short label")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in pivot.index], fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_leiden_heatmaps(
    inputs: dict[str, pd.DataFrame],
    short_mapping: pd.DataFrame,
) -> tuple[list[dict[str, str]], dict[str, float | str]]:
    row_order = short_mapping["plot_short_label"].tolist()
    counts = add_plot_labels(inputs["leiden_counts"], short_mapping)
    fraction = counts.copy()
    totals = fraction.groupby("plot_short_label")["n_nodes"].transform("sum")
    fraction["fraction_within_raw_endpoint"] = fraction["n_nodes"] / totals
    fraction_pivot = pivot_heatmap(
        fraction,
        "plot_short_label",
        "leiden_neigh",
        "fraction_within_raw_endpoint",
        row_order,
    )
    fraction_path = QC_FIGURES / "refined_endpoint_by_leiden_neigh_fraction_heatmap.png"
    save_heatmap(
        fraction_pivot,
        fraction_path,
        "Refined endpoint by Leiden neighborhood: fraction within raw endpoint",
        "fraction within raw endpoint",
        vmin=0.0,
        vmax=1.0,
    )

    enrichment = add_plot_labels(inputs["leiden_enrichment"], short_mapping)
    enrichment_pivot = pivot_heatmap(
        enrichment,
        "plot_short_label",
        "leiden_neigh",
        "enrichment_observed_over_expected",
        row_order,
    )
    enrichment_values = enrichment_pivot.to_numpy(dtype=float)
    positive = enrichment_values[enrichment_values > 0]
    norm = LogNorm(vmin=max(positive.min(), 1e-3), vmax=max(positive.max(), 1.0))
    enrichment_path = QC_FIGURES / "refined_endpoint_by_leiden_neigh_enrichment_heatmap.png"
    save_heatmap(
        enrichment_pivot.clip(lower=1e-3),
        enrichment_path,
        "Refined endpoint by Leiden neighborhood: relative enrichment",
        "relative enrichment (observed / expected)",
        norm=norm,
        cmap="magma",
    )

    original_like = add_plot_labels(inputs["leiden_fraction_endpoint"], short_mapping)
    original_like = original_like.merge(
        short_mapping[["raw_terminal_macrostate", "refined_endpoint_id"]].rename(
            columns={"raw_terminal_macrostate": "candidate_endpoint"}
        ),
        on="candidate_endpoint",
        how="left",
        suffixes=("", "_mapping"),
    )
    original_refined_id_col = "refined_endpoint_id_mapping"
    old_pivot = original_like.pivot_table(
        index=original_refined_id_col,
        columns="leiden_neigh",
        values="fraction_within_endpoint",
        aggfunc="sum",
        fill_value=0.0,
    )
    qc = {
        "old_like_max_value": float(old_pivot.to_numpy(dtype=float).max()),
        "corrected_fraction_max_value": float(fraction_pivot.to_numpy(dtype=float).max()),
        "enrichment_max_value": float(enrichment_pivot.to_numpy(dtype=float).max()),
        "fraction_source_table": str(LEIDEN_COUNTS_CSV),
        "enrichment_source_table": str(LEIDEN_ENRICHMENT_CSV),
    }
    inventory = [
        {
            "figure_path": str(fraction_path),
            "figure_type": "Leiden neighborhood true fraction heatmap",
            "source_table": str(LEIDEN_COUNTS_CSV),
            "corrected_from_original": "yes",
            "correction_reason": "Avoided summing per-raw-endpoint fractions across duplicate refined_endpoint_id values.",
            "notes": "Values are recomputed from existing counts and constrained to [0, 1].",
        },
        {
            "figure_path": str(enrichment_path),
            "figure_type": "Leiden neighborhood enrichment heatmap",
            "source_table": str(LEIDEN_ENRICHMENT_CSV),
            "corrected_from_original": "yes",
            "correction_reason": "Values can exceed 1 and are explicitly labeled as relative enrichment.",
            "notes": "No endpoint refinement or M4C probability recomputation; plotting-only correction.",
        },
    ]
    return inventory, qc


def load_node_table() -> pd.DataFrame:
    columns = [
        "global_node_index",
        "slice_id",
        "time_label",
        "is_final_time",
        "dominant_fate",
        "dominant_fate_probability",
        "normalized_plasticity_entropy",
        "x",
        "y",
    ]
    return pd.read_parquet(NODE_NEIGHBORHOOD_PARQUET, columns=columns)


def add_node_short_labels(node: pd.DataFrame, short_mapping: pd.DataFrame) -> pd.DataFrame:
    label_map = short_mapping[
        [
            "raw_terminal_macrostate",
            "plot_short_label",
            "plot_color",
            "refined_endpoint_id",
            "refined_endpoint_label",
            "confidence_tier_after_refinement",
        ]
    ].rename(columns={"raw_terminal_macrostate": "dominant_fate"})
    out = node.merge(label_map, on="dominant_fate", how="left")
    out["plot_short_label"] = out["plot_short_label"].fillna("Unmapped")
    out["plot_color"] = out["plot_color"].fillna("#cccccc")
    return out


def representative_slices(node: pd.DataFrame, final: bool, n: int = 2) -> list[str]:
    subset = node[node["is_final_time"] == final]
    return subset["slice_id"].value_counts().head(n).index.astype(str).tolist()


def stable_legend_labels(sub: pd.DataFrame, short_mapping: pd.DataFrame) -> pd.DataFrame:
    present = set(sub["plot_short_label"].dropna().astype(str))
    legend = short_mapping[short_mapping["plot_short_label"].isin(present)].copy()
    return legend.sort_values("raw_terminal_macrostate")


def make_endpoint_tissue_maps(node: pd.DataFrame, short_mapping: pd.DataFrame) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    d35 = node[node["is_final_time"] == True].copy()
    for slice_id in representative_slices(node, final=True, n=2):
        sub = d35[d35["slice_id"] == slice_id].copy()
        if len(sub) > 12000:
            sub = sub.sample(12000, random_state=13)
        fig, ax = plt.subplots(figsize=(9.5, 6.0))
        colors = sub["plot_color"].astype(str).tolist()
        ax.scatter(sub["x"], sub["y"], c=colors, s=2, linewidths=0, rasterized=True)
        ax.set_title(f"M4C-v1 dominant refined endpoint, {slice_id}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        legend = stable_legend_labels(sub, short_mapping)
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=row.plot_color,
                markersize=5,
                label=row.plot_short_label,
            )
            for row in legend.itertuples(index=False)
        ]
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=6,
            frameon=False,
            title="Endpoint",
            title_fontsize=7,
            borderaxespad=0,
        )
        safe_slice = str(slice_id).replace("/", "_")
        path = QC_FIGURES / f"m4c_v1_dominant_refined_endpoint_tissue_map_{safe_slice}.png"
        fig.tight_layout(rect=(0, 0, 0.78, 1))
        fig.savefig(path, dpi=220)
        plt.close(fig)
        inventory.append(
            {
                "figure_path": str(path),
                "figure_type": "Representative dominant refined endpoint tissue map",
                "source_table": str(NODE_NEIGHBORHOOD_PARQUET),
                "corrected_from_original": "yes",
                "correction_reason": "Legend moved outside plotting area and endpoint labels shortened.",
                "notes": "Representative D35 slice only; stable raw-endpoint color palette.",
            }
        )
    return inventory


def make_plasticity_tissue_maps(node: pd.DataFrame) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    nonfinal = node[node["is_final_time"] == False].copy()
    if nonfinal.empty:
        return inventory
    vmax = max(0.01, float(nonfinal["normalized_plasticity_entropy"].quantile(0.98)))
    for slice_id in representative_slices(node, final=False, n=2):
        sub = nonfinal[nonfinal["slice_id"] == slice_id].copy()
        if len(sub) > 12000:
            sub = sub.sample(12000, random_state=17)
        fig, ax = plt.subplots(figsize=(7.4, 6.0))
        sc = ax.scatter(
            sub["x"],
            sub["y"],
            c=sub["normalized_plasticity_entropy"],
            s=2,
            linewidths=0,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_title(f"M4C-v1 normalized fate entropy / plasticity, {slice_id}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label("normalized fate entropy / plasticity")
        safe_slice = str(slice_id).replace("/", "_")
        path = QC_FIGURES / f"m4c_v1_normalized_plasticity_tissue_map_{safe_slice}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        inventory.append(
            {
                "figure_path": str(path),
                "figure_type": "Representative normalized plasticity tissue map",
                "source_table": str(NODE_NEIGHBORHOOD_PARQUET),
                "corrected_from_original": "yes",
                "correction_reason": "Colorbar label clarified and non-final maps share one color scale.",
                "notes": "D35 plasticity remains endpoint-initialized/one-hot and is not biologically interpreted.",
            }
        )
    return inventory


def write_qc_note(qc: dict[str, float | str]) -> None:
    old_label_misleading = float(qc["old_like_max_value"]) > 1.0
    body = f"""# M4E-03 Leiden Heatmap QC Note

## Source Tables

- True fraction heatmap source: `{qc["fraction_source_table"]}`
- Enrichment heatmap source: `{qc["enrichment_source_table"]}`

## Finding

The old refined-endpoint Leiden heatmap label was misleading for the generated
plot if per-raw-endpoint `fraction_within_endpoint` values were summed after
collapsing duplicate `refined_endpoint_id` values. In that old-like aggregation,
the maximum plotted value is `{float(qc["old_like_max_value"]):.4f}`, which is
not a valid fraction. The main cause is that raw endpoints 06 and 09 both map to
`rare_me_enteric_smc`, so summing raw endpoint fractions can exceed 1.

Old label misleading: `{"yes" if old_label_misleading else "no"}`

## Corrected Figures

- `refined_endpoint_by_leiden_neigh_fraction_heatmap.png` recomputes a true
  fraction from existing count rows while preserving raw endpoint traceability
  with short labels. The maximum corrected fraction is
  `{float(qc["corrected_fraction_max_value"]):.4f}`.
- `refined_endpoint_by_leiden_neigh_enrichment_heatmap.png` uses the existing
  enrichment table and labels the colorbar as relative enrichment. The maximum
  enrichment value is `{float(qc["enrichment_max_value"]):.4f}`.

## Numerical Scope

No endpoint refinement, upstream data, or M4C fate probabilities were changed.
The patch only corrected plotting labels/layout and recomputed display-only
fractions from existing M4E count tables.
"""
    QC_NOTE_MD.write_text(body)


def append_freeze_report_qc_section() -> None:
    section = """## Figure QC Patch

M4C-v1 remains a pseudo-only baseline endpoint-attraction / fate-propagation map, not lineage-validated fate.

Corrected Leiden-neighborhood heatmap labels now distinguish true endpoint-neighborhood fractions from relative enrichment. Representative tissue maps now use short endpoint labels, a stable raw-endpoint color palette, and legends outside the tissue coordinate area. D35 plasticity remains endpoint-initialized / one-hot and should not be interpreted as true biological low plasticity.

No upstream data or M4C probabilities were recomputed. No pyGPCCA, M4D diagnostics, M3-v2, K_gpcca, M5/regulator, BranchSBM / Branched NicheFlow, barcode preprocessing, or downstream analysis was run.
"""
    if FREEZE_REPORT_MD.exists():
        body = FREEZE_REPORT_MD.read_text()
    else:
        body = "# M4C-v1 Baseline Dynamic Niche-Fate Freeze Report\n"
    marker = "## Figure QC Patch"
    if marker in body:
        body = body.split(marker)[0].rstrip() + "\n\n" + section
    else:
        body = body.rstrip() + "\n\n" + section
    FREEZE_REPORT_MD.write_text(body)


def main() -> None:
    ensure_dirs()
    inputs = read_inputs()
    short_mapping = make_short_label_mapping(inputs["mapping"])
    inventory, qc = make_leiden_heatmaps(inputs, short_mapping)
    node = add_node_short_labels(load_node_table(), short_mapping)
    inventory.extend(make_endpoint_tissue_maps(node, short_mapping))
    inventory.extend(make_plasticity_tissue_maps(node))
    pd.DataFrame(inventory).to_csv(INVENTORY_CSV, index=False)
    write_qc_note(qc)
    append_freeze_report_qc_section()
    print(f"qc_figures: {QC_FIGURES}")
    print(f"short_label_mapping: {SHORT_LABEL_CSV}")
    print(f"figure_inventory: {INVENTORY_CSV}")
    print(f"qc_note: {QC_NOTE_MD}")


if __name__ == "__main__":
    main()
