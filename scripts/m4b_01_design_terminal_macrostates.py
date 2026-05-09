#!/usr/bin/env python
"""Design M4B candidate terminal niche macrostates without fate computation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config


DEFAULT_CONFIG = "configs/m4b_markov_terminal_design.yaml"
METADATA_COLUMNS = [
    "slice_id",
    "slice_file",
    "time",
    "time_day",
    "mouse_id",
    "anchor_index",
    "anchor_cell_id",
    "cell_type_l1",
    "cell_type_l2",
    "cell_type_l3",
]
NO_DOWNSTREAM_FLAGS = {
    "no_gpcca": True,
    "no_fate_probability": True,
    "no_absorption_probability": True,
    "no_branched_nicheflow_training": True,
    "no_branchsbm_training": True,
    "no_m5": True,
    "no_regulator_analysis": True,
}
ROUTE_COMPATIBILITY_NOTE = (
    "M4C is Markov baseline v1 using final-time clustering targets. "
    "M4D is the standard GPCCA/CellRank-inspired Markov route. "
    "Branched NicheFlow / BranchSBM can use terminal macrostates as candidate endpoint/branch labels. "
    "Future barcode-aware M3 can replace or supplement pseudo-lineage evidence without changing the M4C fate interface."
)
STRUCTURAL_DIAGNOSTIC_NOTE = (
    "Incoming mass and incoming degree are structural diagnostics from M4A/M3 transition edges, "
    "not absorption probabilities or fate probabilities."
)


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


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def assert_no_ssd_path(path: Path, label: str) -> None:
    resolved = str(path.expanduser().resolve())
    if resolved == "/ssd" or resolved.startswith("/ssd/"):
        raise ValueError(f"Refusing to use /ssd for {label}: {path}")


def configured_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    inputs = {key: Path(value) for key, value in config.get("inputs", {}).items()}
    for key, path in {**paths, **inputs}.items():
        assert_no_ssd_path(path, key)
    return {**paths, **inputs}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_feature_columns(
    m2_schema: dict[str, Any],
    m3_feature_groups: dict[str, Any],
    configured_groups: list[str],
) -> tuple[list[str], pd.DataFrame]:
    numeric_columns = list(m2_schema.get("numeric_feature_columns", []))
    if not numeric_columns:
        raise ValueError("M2 schema has no numeric_feature_columns.")
    numeric_set = set(numeric_columns)
    groups = m3_feature_groups.get("feature_groups", {})
    selected: list[str] = []
    rows: list[dict[str, Any]] = []
    for group in configured_groups:
        if group not in groups:
            raise KeyError(f"Configured feature group is absent from M3 compatibility groups: {group}")
        group_columns = list(groups[group])
        missing = sorted(set(group_columns) - numeric_set)
        if missing:
            raise KeyError(f"Feature group {group} does not map cleanly to M2 schema columns: {missing[:10]}")
        for column in group_columns:
            if column not in selected:
                selected.append(column)
        rows.append(
            {
                "feature_group": group,
                "m3_group_columns": len(group_columns),
                "mapped_m2_columns": len(group_columns),
                "mapping_status": "mapped_to_m2_schema",
            }
        )
    if not selected:
        raise ValueError("No terminal clustering features were resolved.")
    return selected, pd.DataFrame(rows)


def infer_final_time(node_table: pd.DataFrame) -> tuple[float, str]:
    max_day = float(node_table["time_day"].astype(float).max())
    labels = sorted(
        node_table.loc[np.isclose(node_table["time_day"].astype(float), max_day), "time"].dropna().astype(str).unique()
    )
    if len(labels) != 1:
        raise ValueError(f"Expected one final time label for max time_day {max_day}, found {labels}.")
    return max_day, labels[0]


def select_terminal_nodes(node_table: pd.DataFrame) -> pd.DataFrame:
    final_day, final_time = infer_final_time(node_table)
    mask = np.isclose(node_table["time_day"].astype(float), final_day) & (
        node_table["time"].astype(str) == final_time
    )
    terminal = node_table.loc[mask].copy()
    if terminal.empty:
        raise ValueError("No terminal nodes were selected from final time.")
    return terminal.sort_values("global_node_index").reset_index(drop=True)


def m2_path_by_slice(m2_root: Path) -> dict[str, Path]:
    completed = m2_root / "completed_slices.csv"
    if not completed.exists():
        paths = sorted(m2_root.glob("*/m2_representation_*.parquet"))
        return {path.parent.name: path for path in paths}
    summary = pd.read_csv(completed)
    if not {"slice_id", "output_path"} <= set(summary.columns):
        raise KeyError("M2 completed_slices.csv must contain slice_id and output_path.")
    return {str(row["slice_id"]): Path(row["output_path"]) for row in summary.to_dict("records")}


def load_terminal_m2_rows(m2_root: Path, terminal_nodes: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    path_by_slice = m2_path_by_slice(m2_root)
    read_columns = list(dict.fromkeys(METADATA_COLUMNS + feature_columns))
    frames: list[pd.DataFrame] = []
    for slice_id in sorted(terminal_nodes["slice_id"].astype(str).unique()):
        path = path_by_slice.get(slice_id)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing M2 representation for terminal slice {slice_id}: {path}")
        frame = pd.read_parquet(path, columns=read_columns)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["anchor_id"] = data["slice_id"].astype(str) + "::" + data["anchor_index"].astype(str)
    if bool(data["anchor_id"].duplicated().any()):
        raise ValueError("Duplicate anchor IDs in final-time M2 rows.")
    merged = terminal_nodes.merge(data, on="anchor_id", how="left", suffixes=("_node", ""))
    if int(merged[feature_columns].isna().sum().sum()):
        raise ValueError("Terminal M2 feature matrix contains missing values after node join.")
    if int(merged["anchor_index"].isna().sum()):
        raise ValueError("Some terminal M4A nodes did not map to final-time M2 rows.")
    return merged.sort_values("global_node_index").reset_index(drop=True)


def robust_standardize_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    near_constant_iqr_threshold: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    values = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    med = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    iqr = q75 - q25
    valid = np.isfinite(iqr) & (iqr >= float(near_constant_iqr_threshold))
    safe_iqr = np.where(valid, iqr, 1.0)
    centered = values - med.astype(np.float32, copy=False)
    scaled = centered / safe_iqr.astype(np.float32, copy=False)
    scaled[:, ~valid] = 0.0
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    report = pd.DataFrame(
        {
            "feature": feature_columns,
            "median": med,
            "iqr": iqr,
            "used_for_clustering": valid,
            "near_constant": ~valid,
        }
    )
    return scaled, report


def run_candidate_clusterings(
    matrix: np.ndarray,
    candidate_ks: list[int],
    default_k: int,
    random_seed: int,
    severe_min_fraction: float,
) -> tuple[int, dict[int, np.ndarray], pd.DataFrame]:
    labels_by_k: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    n = int(matrix.shape[0])
    for k in candidate_ks:
        model = MiniBatchKMeans(
            n_clusters=int(k),
            random_state=int(random_seed),
            batch_size=4096,
            n_init=3,
            reassignment_ratio=0.01,
        )
        labels = model.fit_predict(matrix)
        labels_by_k[int(k)] = labels.astype(np.int32, copy=False)
        counts = np.bincount(labels, minlength=int(k))
        min_fraction = float(counts.min() / n) if n else 0.0
        rows.append(
            {
                "n_macrostates": int(k),
                "inertia": float(model.inertia_),
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
                "min_cluster_fraction": min_fraction,
                "max_cluster_fraction": float(counts.max() / n) if n else 0.0,
                "empty_clusters": int((counts == 0).sum()),
                "severe_imbalance": bool(min_fraction < severe_min_fraction),
            }
        )
    qc = pd.DataFrame(rows)
    passing = qc[(qc["empty_clusters"] == 0) & (~qc["severe_imbalance"])]
    if int(default_k) in set(passing["n_macrostates"].astype(int)):
        selected = int(default_k)
    elif len(passing):
        passing = passing.assign(distance=(passing["n_macrostates"].astype(int) - int(default_k)).abs())
        selected = int(passing.sort_values(["distance", "n_macrostates"]).iloc[0]["n_macrostates"])
    else:
        selected = int(default_k)
    qc["selected_default"] = qc["n_macrostates"].astype(int) == selected
    qc["selection_reason"] = np.where(
        qc["selected_default"],
        "default_k_selected" if selected == int(default_k) else "nearest_passing_k_selected",
        "diagnostic_alternative",
    )
    return selected, labels_by_k, qc


def categorical_entropy(values: pd.Series) -> float:
    probs = values.dropna().astype(str).value_counts(normalize=True).to_numpy(dtype=float)
    if len(probs) == 0:
        return 0.0
    return float(-(probs * np.log(np.clip(probs, 1e-300, None))).sum())


def dominant_label(values: pd.Series) -> tuple[str, float]:
    counts = values.fillna("NA").astype(str).value_counts()
    if counts.empty:
        return "NA", 0.0
    return str(counts.index[0]), float(counts.iloc[0] / counts.sum())


def add_incoming_diagnostics(assignments: pd.DataFrame, p_forward_path: Path) -> pd.DataFrame:
    matrix = sp.load_npz(p_forward_path).tocsr()
    incoming_degree = np.asarray((matrix != 0).sum(axis=0)).ravel()
    incoming_mass = np.asarray(matrix.sum(axis=0)).ravel()
    indices = assignments["global_node_index"].to_numpy(dtype=np.int64)
    out = assignments.copy()
    out["incoming_degree_structural"] = incoming_degree[indices].astype(np.int64, copy=False)
    out["incoming_mass_structural"] = incoming_mass[indices].astype(float, copy=False)
    return out


def build_assignments(
    terminal_data: pd.DataFrame,
    labels: np.ndarray,
    selected_k: int,
    feature_matrix: np.ndarray,
    centers: np.ndarray | None = None,
) -> pd.DataFrame:
    assignments = terminal_data[
        [
            "global_node_index",
            "anchor_id",
            "slice_id_node",
            "anchor_index",
            "anchor_cell_id",
            "time_node",
            "time_day_node",
            "mouse_id_node",
            "cell_type_l1",
            "cell_type_l2",
            "cell_type_l3",
        ]
    ].copy()
    assignments = assignments.rename(
        columns={
            "slice_id_node": "slice_id",
            "time_node": "time",
            "time_day_node": "time_day",
            "mouse_id_node": "mouse_id",
        }
    )
    assignments["terminal_macrostate_id"] = labels.astype(np.int32, copy=False)
    assignments["terminal_macrostate_label"] = [
        f"terminal_macrostate_{value:02d}" for value in assignments["terminal_macrostate_id"].to_numpy()
    ]
    assignments["selected_n_macrostates"] = int(selected_k)
    return assignments


def macrostate_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(assignments)
    for macro_id, group in assignments.groupby("terminal_macrostate_id", observed=True):
        row: dict[str, Any] = {
            "terminal_macrostate_id": int(macro_id),
            "terminal_macrostate_label": f"terminal_macrostate_{int(macro_id):02d}",
            "n_nodes": int(len(group)),
            "fraction_final_nodes": float(len(group) / total) if total else 0.0,
            "time": str(group["time"].iloc[0]),
            "time_day": float(group["time_day"].iloc[0]),
            "incoming_degree_sum_structural": int(group["incoming_degree_structural"].sum()),
            "incoming_degree_mean_structural": float(group["incoming_degree_structural"].mean()),
            "incoming_mass_sum_structural": float(group["incoming_mass_structural"].sum()),
            "incoming_mass_mean_structural": float(group["incoming_mass_structural"].mean()),
        }
        for column in ["cell_type_l1", "cell_type_l2", "cell_type_l3"]:
            label, fraction = dominant_label(group[column])
            row[f"dominant_{column}"] = label
            row[f"dominant_{column}_fraction"] = fraction
            row[f"{column}_entropy"] = categorical_entropy(group[column])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("terminal_macrostate_id").reset_index(drop=True)


def feature_summary(assignments: pd.DataFrame, terminal_data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    frame = pd.concat(
        [assignments[["terminal_macrostate_id"]].reset_index(drop=True), terminal_data[feature_columns].reset_index(drop=True)],
        axis=1,
    )
    means = frame.groupby("terminal_macrostate_id", observed=True)[feature_columns].mean()
    medians = frame.groupby("terminal_macrostate_id", observed=True)[feature_columns].median()
    rows: list[dict[str, Any]] = []
    for macro_id in means.index:
        for feature in feature_columns:
            rows.append(
                {
                    "terminal_macrostate_id": int(macro_id),
                    "terminal_macrostate_label": f"terminal_macrostate_{int(macro_id):02d}",
                    "feature": feature,
                    "mean": float(means.loc[macro_id, feature]),
                    "median": float(medians.loc[macro_id, feature]),
                }
            )
    return pd.DataFrame(rows)


def m4c_inputs_payload(
    paths: dict[str, Path],
    selected_k: int,
    final_time: str,
    final_time_day: float,
) -> dict[str, Any]:
    output_root = paths["output_root"]
    return {
        "schema_version": "m4c_fate_probability_inputs_v1",
        "recommended_p_object": str(paths["p_forward"]),
        "structural_absorbing_p_object": str(paths["p_absorbing"]),
        "terminal_macrostate_assignments": str(output_root / "terminal_states" / "terminal_macrostate_assignments.parquet"),
        "node_table": str(paths["node_table"]),
        "selected_n_terminal_macrostates": int(selected_k),
        "final_time": final_time,
        "final_time_day": float(final_time_day),
        "recommended_fate_computation": "time-layered backward propagation to terminal macrostate labels",
        "expected_outputs": [
            "per-node fate probability matrix F[n_nodes, n_terminal_macrostates]",
            "per-node plasticity score = entropy(F_i)",
            "dominant fate label",
            "confidence/top1 fate probability",
        ],
        "visualization_requirements": [
            "fate probability maps by time",
            "plasticity maps",
            "dominant fate maps",
            "terminal macrostate composition plots",
            "source-to-terminal fate flow heatmaps",
        ],
        "directionality_evidence_source": "pseudo-lineage/time-coupled transition",
        "barcode_compatibility_note": (
            "M4C should record the directionality evidence source as pseudo-lineage/time-coupled transition. "
            "Future barcode-aware M3 can replace or supplement transition evidence without changing the M4C fate probability interface."
        ),
        "route_compatibility_note": ROUTE_COMPATIBILITY_NOTE,
        **NO_DOWNSTREAM_FLAGS,
    }


def design_report(
    selected_k: int,
    final_time: str,
    final_time_day: float,
    terminal_count: int,
    feature_count: int,
    scaling_report: pd.DataFrame,
    candidate_qc: pd.DataFrame,
    summary: pd.DataFrame,
    figure_warnings: list[str],
) -> str:
    lines = [
        "# M4B-01 Terminal Macrostate Design",
        "",
        "This stage designs candidate terminal niche macrostates for later M4C fate probability computation.",
        "It does not run GPCCA, compute fate probabilities, compute absorption probabilities, train Branched NicheFlow / BranchSBM, run M5, or run regulator analysis.",
        "",
        "## Terminal Node Selection",
        f"- final time inferred from max time_day: {final_time} ({final_time_day:g})",
        f"- final-time nodes: {terminal_count}",
        f"- clustering features resolved from M2 schema: {feature_count}",
        f"- near-constant features handled safely: {int(scaling_report['near_constant'].sum())}",
        "",
        "## Candidate K Diagnostics",
    ]
    for row in candidate_qc.to_dict("records"):
        lines.append(
            "- "
            f"K={int(row['n_macrostates'])}: min_size={int(row['min_cluster_size'])}, "
            f"max_size={int(row['max_cluster_size'])}, min_fraction={row['min_cluster_fraction']:.6g}, "
            f"empty={int(row['empty_clusters'])}, severe_imbalance={bool(row['severe_imbalance'])}, "
            f"{row['selection_reason']}"
        )
    lines.extend(
        [
            "",
            f"Selected default terminal macrostate granularity: K={selected_k}",
            "",
            "## Terminal Macrostate Summary",
        ]
    )
    for row in summary.to_dict("records"):
        lines.append(
            "- "
            f"{row['terminal_macrostate_label']}: nodes={int(row['n_nodes'])}, "
            f"incoming_mass_structural={row['incoming_mass_sum_structural']:.6g}, "
            f"incoming_degree_structural={int(row['incoming_degree_sum_structural'])}, "
            f"dominant_l1={row['dominant_cell_type_l1']}"
        )
    lines.extend(
        [
            "",
            "## Diagnostic Semantics",
            f"- {STRUCTURAL_DIAGNOSTIC_NOTE}",
            f"- {ROUTE_COMPATIBILITY_NOTE}",
            "",
            "## Downstream Boundary",
            "- no GPCCA was run",
            "- no fate probability was computed",
            "- no absorption probability was computed",
            "- no Branched NicheFlow / BranchSBM training was run",
            "- no M5 was run",
            "- no regulator analysis was run",
        ]
    )
    if figure_warnings:
        lines.extend(["", "## Figure Warnings", *[f"- {warning}" for warning in figure_warnings]])
    return "\n".join(lines).rstrip() + "\n"


def handoff_report(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# M4C Fate Probability Handoff",
            "",
            "M4B has produced candidate terminal niche macrostates for later M4C computation.",
            "M4C should compute fate probabilities only after explicit approval.",
            "",
            "## Recommended Inputs",
            f"- P object: {payload['recommended_p_object']}",
            f"- structural absorbing P object: {payload['structural_absorbing_p_object']}",
            f"- terminal macrostate assignments: {payload['terminal_macrostate_assignments']}",
            f"- node table: {payload['node_table']}",
            f"- terminal macrostates: {payload['selected_n_terminal_macrostates']}",
            "",
            "## Recommended Computation",
            f"- {payload['recommended_fate_computation']}",
            "",
            "## Compatibility",
            f"- {payload['route_compatibility_note']}",
            f"- {payload['barcode_compatibility_note']}",
            "",
            "## Not Run",
            "- GPCCA was not run.",
            "- Fate probability was not computed.",
            "- Absorption probability was not computed.",
            "- Branched NicheFlow / BranchSBM was not trained.",
            "- M5 and regulator analysis were not run.",
            "",
        ]
    )


def make_figures(
    figures_dir: Path,
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    matrix: np.ndarray,
    figure_failure_is_warning: bool,
) -> list[str]:
    warnings: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir.mkdir(parents=True, exist_ok=True)
        labels = summary["terminal_macrostate_label"].astype(str)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, summary["n_nodes"])
        ax.set_title("Terminal macrostate size distribution")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(figures_dir / "m4b_terminal_macrostate_size_distribution.png", dpi=140)
        plt.close(fig)

        for column, name in [
            ("cell_type_l1", "m4b_terminal_macrostate_celltype_l1_composition.png"),
            ("cell_type_l2", "m4b_terminal_macrostate_celltype_l2_composition.png"),
        ]:
            comp = pd.crosstab(assignments["terminal_macrostate_label"], assignments[column], normalize="index")
            top_cols = comp.mean(axis=0).sort_values(ascending=False).head(12).index
            comp = comp[top_cols]
            fig, ax = plt.subplots(figsize=(10, 5))
            bottom = np.zeros(len(comp), dtype=float)
            for label in comp.columns:
                values = comp[label].to_numpy(dtype=float)
                ax.bar(comp.index.astype(str), values, bottom=bottom, label=str(label))
                bottom += values
            ax.set_title(f"Terminal macrostate {column} composition")
            ax.tick_params(axis="x", rotation=45)
            ax.legend(fontsize=6, ncol=2)
            fig.tight_layout()
            fig.savefig(figures_dir / name, dpi=140)
            plt.close(fig)

        pca = PCA(n_components=2, random_state=1)
        coords = pca.fit_transform(matrix)
        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=assignments["terminal_macrostate_id"].to_numpy(),
            s=1,
            cmap="tab20",
            alpha=0.7,
        )
        ax.set_title("Terminal macrostate PCA")
        fig.colorbar(scatter, ax=ax, fraction=0.04, label="macrostate")
        fig.tight_layout()
        fig.savefig(figures_dir / "m4b_terminal_macrostate_embedding_umap_or_pca.png", dpi=140)
        plt.close(fig)

        for column, name, title in [
            ("incoming_mass_sum_structural", "m4b_terminal_macrostate_incoming_mass.png", "Incoming mass"),
            ("incoming_degree_sum_structural", "m4b_terminal_macrostate_incoming_degree.png", "Incoming degree"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(labels, summary[column])
            ax.set_title(f"Terminal macrostate {title}")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(figures_dir / name, dpi=140)
            plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        if not figure_failure_is_warning:
            raise
        warnings.append(f"Figure generation failed after terminal design passed: {exc}")
    return warnings


def write_outputs(
    config: dict[str, Any],
    paths: dict[str, Path],
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    feature_summary_frame: pd.DataFrame,
    feature_mapping: pd.DataFrame,
    scaling_report: pd.DataFrame,
    candidate_qc: pd.DataFrame,
    selected_k: int,
    final_time: str,
    final_time_day: float,
    feature_count: int,
    figure_warnings: list[str],
) -> dict[str, Path]:
    terminal_dir = paths["output_root"] / "terminal_states"
    reports_dir = paths["reports_dir"]
    output_paths = {
        "assignments": terminal_dir / "terminal_macrostate_assignments.parquet",
        "summary": terminal_dir / "terminal_macrostate_summary.csv",
        "feature_summary": terminal_dir / "terminal_macrostate_feature_summary.csv",
        "feature_mapping": reports_dir / "m4b_terminal_feature_mapping.csv",
        "scaling_report": reports_dir / "m4b_terminal_feature_scaling_report.csv",
        "candidate_qc": reports_dir / "m4b_terminal_macrostate_candidate_qc.csv",
        "design_report": reports_dir / "m4b_terminal_macrostate_design_report.md",
        "design_summary": reports_dir / "m4b_terminal_macrostate_design_summary.json",
        "m4c_handoff_md": reports_dir / "m4c_fate_probability_handoff.md",
        "m4c_inputs_json": reports_dir / "m4c_fate_probability_inputs.json",
    }
    atomic_write_parquet(output_paths["assignments"], assignments)
    atomic_write_csv(output_paths["summary"], summary)
    atomic_write_csv(output_paths["feature_summary"], feature_summary_frame)
    atomic_write_csv(output_paths["feature_mapping"], feature_mapping)
    atomic_write_csv(output_paths["scaling_report"], scaling_report)
    atomic_write_csv(output_paths["candidate_qc"], candidate_qc)
    payload = m4c_inputs_payload(paths, selected_k, final_time, final_time_day)
    atomic_write_json(output_paths["m4c_inputs_json"], payload)
    atomic_write_text(output_paths["m4c_handoff_md"], handoff_report(payload))
    atomic_write_text(
        output_paths["design_report"],
        design_report(
            selected_k,
            final_time,
            final_time_day,
            len(assignments),
            feature_count,
            scaling_report,
            candidate_qc,
            summary,
            figure_warnings,
        ),
    )
    atomic_write_json(
        output_paths["design_summary"],
        {
            "schema_version": "m4b_terminal_macrostate_design_summary_v1",
            "generated_at_utc": utc_now_iso(),
            "selected_n_macrostates": int(selected_k),
            "candidate_n_macrostates": [int(x) for x in config["terminal_design"]["candidate_n_macrostates"]],
            "final_time": final_time,
            "final_time_day": final_time_day,
            "final_time_nodes": int(len(assignments)),
            "feature_count": int(feature_count),
            "near_constant_features": int(scaling_report["near_constant"].sum()),
            "terminal_macrostate_sizes": summary[["terminal_macrostate_id", "n_nodes"]].to_dict("records"),
            "structural_diagnostic_note": STRUCTURAL_DIAGNOSTIC_NOTE,
            "route_compatibility_note": ROUTE_COMPATIBILITY_NOTE,
            "figure_warnings": figure_warnings,
            "outputs": {key: str(value) for key, value in output_paths.items()},
            **NO_DOWNSTREAM_FLAGS,
        },
    )
    return output_paths


def run(args: argparse.Namespace) -> int:
    start = time.monotonic()
    config = load_config(args.config)
    paths = configured_paths(config)
    m2_schema = load_json(paths["m2_schema"])
    m3_groups = load_json(paths["m3_feature_groups"])
    feature_columns, feature_mapping = resolve_feature_columns(
        m2_schema,
        m3_groups,
        list(config["terminal_design"]["feature_groups"]),
    )
    node_table = pd.read_parquet(paths["node_table"])
    terminal_nodes = select_terminal_nodes(node_table)
    final_time_day, final_time = infer_final_time(node_table)
    terminal_data = load_terminal_m2_rows(paths["m2_by_slice_root"], terminal_nodes, feature_columns)
    matrix, scaling_report = robust_standardize_features(
        terminal_data,
        feature_columns,
        float(config["terminal_design"].get("near_constant_iqr_threshold", 1e-6)),
    )
    selected_k, labels_by_k, candidate_qc = run_candidate_clusterings(
        matrix,
        [int(value) for value in config["terminal_design"]["candidate_n_macrostates"]],
        int(config["terminal_design"]["default_n_macrostates"]),
        int(config["terminal_design"]["random_seed"]),
        float(config["terminal_design"].get("severe_imbalance_min_fraction", 0.001)),
    )
    assignments = build_assignments(terminal_data, labels_by_k[selected_k], selected_k, matrix)
    assignments = add_incoming_diagnostics(assignments, paths["p_forward"])
    summary = macrostate_summary(assignments)
    feature_summary_frame = feature_summary(assignments, terminal_data, feature_columns)
    figure_warnings: list[str] = []
    if bool(config["visualization"].get("make_figures", True)):
        figure_warnings = make_figures(
            paths["figures_dir"],
            assignments,
            summary,
            matrix,
            bool(config["visualization"].get("figure_failure_is_warning", True)),
        )
    output_paths = write_outputs(
        config,
        paths,
        assignments,
        summary,
        feature_summary_frame,
        feature_mapping,
        scaling_report,
        candidate_qc,
        selected_k,
        final_time,
        final_time_day,
        len(feature_columns),
        figure_warnings,
    )
    runtime = time.monotonic() - start
    print("M4B_01_TERMINAL_MACROSTATE_DESIGN_COMPLETE")
    print(f"GLOBAL_NODES_READ {len(node_table)}")
    print(f"FINAL_TIME {final_time}")
    print(f"FINAL_TIME_DAY {final_time_day:g}")
    print(f"FINAL_TIME_NODES {len(assignments)}")
    print(f"CANDIDATE_K_VALUES {','.join(str(v) for v in config['terminal_design']['candidate_n_macrostates'])}")
    print(f"SELECTED_K {selected_k}")
    print(f"FEATURES_USED {len(feature_columns)}")
    print(f"NEAR_CONSTANT_FEATURES {int(scaling_report['near_constant'].sum())}")
    print(f"RUNTIME_SECONDS {runtime:.3f}")
    print(f"DESIGN_REPORT {output_paths['design_report']}")
    print("NO_GPCCA True")
    print("NO_FATE_PROBABILITY True")
    print("NO_ABSORPTION_PROBABILITY True")
    print("NO_BRANCHED_NICHEFLOW_TRAINING True")
    print("NO_M5 True")
    print("NO_REGULATOR_ANALYSIS True")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
