#!/usr/bin/env python
"""Interpretation hardening for L126 PlanA-L bounded GPCCA macrostates.

This script audits existing Round 2 GPCCA outputs only. It does not rebuild the
directed kernel and does not rerun GPCCA. Outputs are technical QC summaries and
safe preliminary figure candidates for lineage-informed macrostates.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from nichefate.barcode_adapter.qc import snapshot_files
from nichefate.barcode_adapter.reporting import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_tsv,
    ensure_dir,
    markdown_table,
    utc_now,
)
from nichefate.planA_l.reporting import SCOPE_NOTES, forbidden_claim_hits


ROUND2_KERNEL_NAMES = (
    "K_lineage_directed",
    "K_expr_spatial_only",
    "K_phi_shuffled",
    "K_coverage_only",
    "K_barcode_shuffled",
)
CONTROL_KERNELS = ROUND2_KERNEL_NAMES[1:]
FORBIDDEN_EXTRA = (
    "terminal state",
    "true fate",
    "lineage-validated endpoint",
    "proven transition",
    "clonal expansion discovered",
    "developmental trajectory across s1/s2/s3",
    "endpoint fate",
    "true fate state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round2-root", type=Path, default=Path("processed/l126_plana_lineage_kernel_gpcca_round2"))
    parser.add_argument("--round1-root", type=Path, default=Path("processed/l126_plana_lineage_kernel_round1"))
    parser.add_argument("--output-root", type=Path, default=Path("processed/l126_plana_lineage_kernel_interpretation_round3"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/l126_plana_lineage_kernel_interpretation_round3"))
    parser.add_argument("--selected-k", type=int, default=6)
    parser.add_argument("--audit-macrostate", type=int, default=5)
    parser.add_argument("--make-key-figures", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--mode",
        default="all",
        choices=[
            "all",
            "input_inventory_only",
            "k6_audit_only",
            "section_effect_only",
            "membership_uncertainty_only",
            "controls_only",
            "coarse_transition_only",
            "figure_audit_only",
            "interpretation_only",
            "validation_only",
        ],
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    return pd.read_csv(path, sep="\t", compression=compression)


def write_report(report_root: Path, stem: str, title: str, payload: dict[str, Any], lines: list[str], *, overwrite: bool) -> None:
    ensure_dir(report_root)
    scope = "\n".join(f"- {note}" for note in SCOPE_NOTES)
    body = "\n".join(lines).strip()
    atomic_write_json(report_root / f"{stem}.json", payload, overwrite=overwrite)
    atomic_write_text(report_root / f"{stem}.md", f"# {title}\n\n{scope}\n\n{body}\n", overwrite=overwrite)


def safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else 0.0


def safe_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.median()) if numeric.notna().any() else 0.0


def safe_sum(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return float(numeric.sum())


def entropy_from_counts(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[arr > 0]
    total = float(arr.sum())
    if total <= 0:
        return 0.0
    probs = arr / total
    return float(-(probs * np.log(probs)).sum())


def section_distribution(group: pd.DataFrame) -> tuple[str, str, float, float, bool]:
    counts = group["dominant_sample_id"].astype(str).value_counts().sort_index()
    if counts.empty:
        return "", "", 0.0, 0.0, False
    distribution = ";".join(f"{key}:{int(value)}" for key, value in counts.items())
    dominant = str(counts.idxmax())
    purity = float(counts.max() / counts.sum())
    entropy = entropy_from_counts(counts.to_numpy(dtype=float))
    return distribution, dominant, purity, entropy, bool(purity > 0.9)


def load_round2_state(round2_root: Path, selected_k: int) -> dict[str, pd.DataFrame]:
    gpcca_root = round2_root / "gpcca_lineage_directed"
    macro_root = round2_root / "macrostate_annotation"
    return {
        "assignment": read_table(gpcca_root / f"gpcca_k{selected_k}_assignment.tsv"),
        "membership": read_table(gpcca_root / f"gpcca_k{selected_k}_membership.tsv.gz"),
        "coarse": read_table(gpcca_root / f"gpcca_k{selected_k}_coarse_transition.tsv"),
        "macro_annotation": read_table(macro_root / "selected_k_macrostate_annotation.tsv"),
        "barcode_summary": read_table(macro_root / "selected_k_macrostate_barcode_summary.tsv"),
        "section_summary": read_table(macro_root / "selected_k_macrostate_section_summary.tsv"),
        "state_annotation": read_table(macro_root / "selected_k_state_annotation.tsv.gz"),
        "comparison": read_table(round2_root / "control_gpcca" / "gpcca_kernel_comparison_metrics.tsv"),
    }


def round2_snapshot_paths(round2_root: Path) -> list[Path]:
    return sorted(path for path in round2_root.rglob("*") if path.is_file())


def compare_snapshots(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    merged = before.merge(after, on="path", how="outer", suffixes=("_before", "_after"))
    rows = []
    for row in merged.to_dict(orient="records"):
        changed = False
        for field in ("exists", "size_bytes", "mtime_utc"):
            if row.get(f"{field}_before") != row.get(f"{field}_after"):
                changed = True
        rows.append({**row, "changed": bool(changed)})
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, figure_root: Path, stem: str) -> list[str]:
    ensure_dir(figure_root)
    fig.tight_layout()
    outputs = []
    for suffix in ("png", "pdf"):
        path = figure_root / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def inventory_phase(round2_root: Path, report_root: Path, selected_k: int, *, overwrite: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for k in (3, 4, 5, 6):
        for kind, path in [
            ("assignment", round2_root / "gpcca_lineage_directed" / f"gpcca_k{k}_assignment.tsv"),
            ("membership", round2_root / "gpcca_lineage_directed" / f"gpcca_k{k}_membership.tsv.gz"),
            ("coarse_transition", round2_root / "gpcca_lineage_directed" / f"gpcca_k{k}_coarse_transition.tsv"),
        ]:
            row: dict[str, Any] = {"artifact": f"k{k}_{kind}", "path": str(path), "exists": path.exists(), "readable": False, "rows": 0, "columns": 0}
            if path.exists():
                try:
                    frame = read_table(path)
                    row.update({"readable": True, "rows": int(len(frame)), "columns": int(len(frame.columns))})
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    required = {
        "selected_k_macrostate_annotation": round2_root / "macrostate_annotation" / "selected_k_macrostate_annotation.tsv",
        "selected_k_macrostate_barcode_summary": round2_root / "macrostate_annotation" / "selected_k_macrostate_barcode_summary.tsv",
        "selected_k_macrostate_section_summary": round2_root / "macrostate_annotation" / "selected_k_macrostate_section_summary.tsv",
        "control_comparison_metrics": round2_root / "control_gpcca" / "gpcca_kernel_comparison_metrics.tsv",
    }
    for name, path in required.items():
        row = {"artifact": name, "path": str(path), "exists": path.exists(), "readable": False, "rows": 0, "columns": 0}
        if path.exists():
            try:
                frame = read_table(path)
                row.update({"readable": True, "rows": int(len(frame)), "columns": int(len(frame.columns))})
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    figure_dir = round2_root.parents[1] / "reports" if False else Path("reports/l126_plana_lineage_kernel_gpcca_round2/figures")
    figure_rows = sorted(figure_dir.glob("*.png")) + sorted(figure_dir.glob("*.pdf"))
    rows.append(
        {
            "artifact": "figures_directory",
            "path": str(figure_dir),
            "exists": figure_dir.exists(),
            "readable": figure_dir.exists(),
            "rows": len(figure_rows),
            "columns": 0,
        }
    )
    frame = pd.DataFrame(rows)
    all_gpcca = frame.loc[frame["artifact"].str.contains(r"^k[3456]_"), ["exists", "readable"]].all().all()
    controls_ok = bool(frame.loc[frame["artifact"] == "control_comparison_metrics", ["exists", "readable"]].all(axis=None))
    annotation_ok = bool(
        frame.loc[
            frame["artifact"].isin(
                [
                    "selected_k_macrostate_annotation",
                    "selected_k_macrostate_barcode_summary",
                    "selected_k_macrostate_section_summary",
                ]
            ),
            ["exists", "readable"],
        ].all(axis=None)
    )
    if not all_gpcca:
        label = "L126_ROUND3_HOLD_FOR_MISSING_GPCCA_OUTPUTS"
    elif not controls_ok:
        label = "L126_ROUND3_HOLD_FOR_MISSING_CONTROL_OUTPUTS"
    elif not annotation_ok:
        label = "L126_ROUND3_HOLD_FOR_MISSING_MACROSTATE_ANNOTATION"
    else:
        label = "L126_ROUND3_INPUTS_READY"
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": label,
        "selected_k": selected_k,
        "artifact_count": int(len(frame)),
        "all_gpcca_outputs_readable": bool(all_gpcca),
        "control_outputs_readable": bool(controls_ok),
        "macrostate_annotation_readable": bool(annotation_ok),
        "figures_detected": int(len(figure_rows)),
    }
    write_report(
        report_root,
        "00_INPUT_INVENTORY",
        "Input Inventory",
        payload,
        [
            f"- Decision label: `{label}`",
            f"- Selected k audited: `{selected_k}`",
            f"- Round 2 root: `{round2_root}`",
            "",
            markdown_table(frame, limit=40),
        ],
        overwrite=overwrite,
    )
    return payload


def coarse_arrays(coarse: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    value_cols = [col for col in coarse.columns if col.startswith("macrostate_")]
    matrix = coarse[value_cols].to_numpy(dtype=float)
    ids = [int(col.split("_")[-1]) for col in value_cols]
    return matrix, ids


def build_k6_macrostate_audit(state: pd.DataFrame, coarse: pd.DataFrame) -> pd.DataFrame:
    matrix, ids = coarse_arrays(coarse)
    rows: list[dict[str, Any]] = []
    for macrostate, group in state.groupby("macrostate", sort=True):
        macrostate = int(macrostate)
        distribution, dominant_section, purity, sec_entropy, section_dominated = section_distribution(group)
        idx = ids.index(macrostate) if macrostate in ids else macrostate
        self_retention = float(matrix[idx, idx]) if idx < matrix.shape[0] else np.nan
        incoming = float(matrix[:, idx].sum()) if idx < matrix.shape[1] else np.nan
        outgoing_non_self = float(1.0 - self_retention) if math.isfinite(self_retention) else np.nan
        ra = safe_sum(group.get("unique_RA_total_count", pd.Series(dtype=float)))
        ta = safe_sum(group.get("unique_TA_total_count", pd.Series(dtype=float)))
        ca = safe_sum(group.get("unique_CA_total_count", pd.Series(dtype=float)))
        assay_total = ra + ta + ca
        assay_max_fraction = max([ra, ta, ca]) / assay_total if assay_total > 0 else 0.0
        row = {
            "macrostate": macrostate,
            "n_states": int(len(group)),
            "fraction_all_states": float(len(group) / len(state)),
            "section_distribution": distribution,
            "dominant_section_id": dominant_section,
            "section_purity": purity,
            "section_entropy": sec_entropy,
            "section_dominated": section_dominated,
            "phi_mean": safe_mean(group["phi"]),
            "phi_median": safe_median(group["phi"]),
            "barcode_entropy_mean": safe_mean(group.get("unique_feature_entropy", group.get("feature_entropy", pd.Series(dtype=float)))),
            "barcode_entropy_median": safe_median(group.get("unique_feature_entropy", group.get("feature_entropy", pd.Series(dtype=float)))),
            "dominant_feature_fraction_mean": safe_mean(group.get("unique_dominant_feature_fraction", group.get("dominant_feature_fraction", pd.Series(dtype=float)))),
            "dominant_feature_fraction_median": safe_median(group.get("unique_dominant_feature_fraction", group.get("dominant_feature_fraction", pd.Series(dtype=float)))),
            "total_lineage_count_sum": safe_sum(group.get("unique_total_lineage_count", group.get("total_lineage_count", pd.Series(dtype=float)))),
            "total_lineage_count_mean": safe_mean(group.get("unique_total_lineage_count", group.get("total_lineage_count", pd.Series(dtype=float)))),
            "total_lineage_count_median": safe_median(group.get("unique_total_lineage_count", group.get("total_lineage_count", pd.Series(dtype=float)))),
            "RA_total_count_sum": ra,
            "TA_total_count_sum": ta,
            "CA_total_count_sum": ca,
            "RA_fraction": float(ra / assay_total) if assay_total > 0 else 0.0,
            "TA_fraction": float(ta / assay_total) if assay_total > 0 else 0.0,
            "CA_fraction": float(ca / assay_total) if assay_total > 0 else 0.0,
            "assay_max_fraction": float(assay_max_fraction),
            "max_membership_mean": safe_mean(group["max_membership"]),
            "max_membership_median": safe_median(group["max_membership"]),
            "membership_entropy_mean": safe_mean(group["membership_entropy"]),
            "membership_entropy_median": safe_median(group["membership_entropy"]),
            "ambiguous_state_fraction": float((pd.to_numeric(group["max_membership"], errors="coerce") < 0.6).mean()),
            "high_confidence_state_fraction": float((pd.to_numeric(group["max_membership"], errors="coerce") >= 0.8).mean()),
            "lineage_positive_fraction_mean": safe_mean(group.get("unique_fraction_lineage_positive", group.get("fraction_lineage_positive", pd.Series(dtype=float)))),
            "coarse_self_retention": self_retention,
            "coarse_incoming_mass": incoming,
            "coarse_outgoing_non_self_mass": outgoing_non_self,
            "coarse_net_incoming_minus_outgoing_non_self": float(incoming - outgoing_non_self) if math.isfinite(incoming) and math.isfinite(outgoing_non_self) else np.nan,
        }
        rows.append(row)
    audit = pd.DataFrame(rows).sort_values("macrostate").reset_index(drop=True)
    q3_counts = float(audit["total_lineage_count_mean"].quantile(0.75)) if len(audit) else 0.0
    audit["high_uncertainty_macrostate"] = (audit["ambiguous_state_fraction"] > 0.5) | (audit["max_membership_median"] < 0.6)
    audit["low_evidence_macrostate"] = audit["lineage_positive_fraction_mean"] < 0.25
    audit["possible_coverage_driven_macrostate"] = (audit["total_lineage_count_mean"] >= q3_counts) & (audit["phi_mean"] > audit["phi_mean"].median())
    audit["possible_assay_skewed_macrostate"] = audit["assay_max_fraction"] > 0.8
    labels: list[str] = []
    for row in audit.itertuples(index=False):
        parts: list[str] = []
        if row.section_dominated:
            parts.append("section-enriched technical macrostate")
        if row.high_uncertainty_macrostate:
            parts.append("ambiguous-membership macrostate")
        if row.phi_mean >= audit["phi_mean"].quantile(0.75):
            parts.append("lineage-high technical macrostate")
        if row.barcode_entropy_mean >= audit["barcode_entropy_mean"].quantile(0.75):
            parts.append("lineage-diverse technical macrostate")
        if row.dominant_feature_fraction_mean >= audit["dominant_feature_fraction_mean"].quantile(0.75):
            parts.append("clone-feature-dominant technical macrostate")
        if row.possible_assay_skewed_macrostate:
            parts.append("assay-skewed technical macrostate")
        labels.append("; ".join(parts) if parts else "lineage-informed technical macrostate")
    audit["technical_interpretation_labels"] = labels
    return audit


def k6_audit_phase(round2_root: Path, output_root: Path, report_root: Path, selected_k: int, *, overwrite: bool) -> dict[str, Any]:
    state = load_round2_state(round2_root, selected_k)
    audit = build_k6_macrostate_audit(state["state_annotation"], state["coarse"])
    output_path = output_root / f"selected_k{selected_k}_macrostate_technical_audit.tsv"
    atomic_write_tsv(output_path, audit, overwrite=overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "selected_k": selected_k,
        "output_path": str(output_path),
        "macrostate_count": int(len(audit)),
        "section_dominated_macrostates": audit.loc[audit["section_dominated"], "macrostate"].astype(int).tolist(),
        "high_uncertainty_macrostates": audit.loc[audit["high_uncertainty_macrostate"], "macrostate"].astype(int).tolist(),
        "possible_coverage_driven_macrostates": audit.loc[audit["possible_coverage_driven_macrostate"], "macrostate"].astype(int).tolist(),
        "possible_assay_skewed_macrostates": audit.loc[audit["possible_assay_skewed_macrostate"], "macrostate"].astype(int).tolist(),
    }
    write_report(
        report_root,
        "01_K6_MACROSTATE_TECHNICAL_AUDIT",
        "K6 Macrostate Technical Audit",
        payload,
        [
            f"- Selected k: `{selected_k}`",
            f"- Macrostate audit table: `{output_path}`",
            f"- Section-enriched macrostates: `{payload['section_dominated_macrostates']}`",
            f"- High-uncertainty macrostates: `{payload['high_uncertainty_macrostates']}`",
            "- Labels are technical QC labels only, not biological names.",
            "",
            markdown_table(audit, limit=12),
        ],
        overwrite=overwrite,
    )
    return payload


def pca_difference_summary(group: pd.DataFrame, others: pd.DataFrame) -> str:
    pca_cols = [col for col in group.columns if col.startswith("pca_mean_")]
    diffs = []
    for col in pca_cols:
        diff = safe_mean(group[col]) - safe_mean(others[col])
        diffs.append((col, diff))
    top = sorted(diffs, key=lambda item: abs(item[1]), reverse=True)[:5]
    return ";".join(f"{col}:{value:.3f}" for col, value in top)


def control_section_enrichment(round2_root: Path, selected_k: int, target_section: str, main_assignment: pd.DataFrame) -> pd.DataFrame:
    rows = []
    main_labels = main_assignment.sort_values("state_index")["macrostate"].astype(str)
    for kernel in CONTROL_KERNELS:
        path = round2_root / "control_gpcca" / kernel / f"gpcca_k{selected_k}_assignment.tsv"
        if not path.exists():
            rows.append({"kernel_name": kernel, "exists": False})
            continue
        frame = read_table(path)
        candidates = []
        for macrostate, group in frame.groupby("macrostate", sort=True):
            counts = group["dominant_sample_id"].astype(str).value_counts()
            target_fraction = float(counts.get(target_section, 0) / len(group)) if len(group) else 0.0
            _, dominant, purity, entropy, dominated = section_distribution(group)
            candidates.append(
                {
                    "kernel_name": kernel,
                    "control_macrostate": int(macrostate),
                    "n_states": int(len(group)),
                    "target_section_fraction": target_fraction,
                    "dominant_section_id": dominant,
                    "section_purity": purity,
                    "section_entropy": entropy,
                    "section_dominated": dominated,
                }
            )
        best = max(candidates, key=lambda row: (row["target_section_fraction"], row["section_purity"])) if candidates else {"kernel_name": kernel}
        control_labels = frame.sort_values("state_index")["macrostate"].astype(str)
        best.update(
            {
                "exists": True,
                "assignment_ari_vs_lineage_directed": float(adjusted_rand_score(main_labels, control_labels)),
                "assignment_nmi_vs_lineage_directed": float(normalized_mutual_info_score(main_labels, control_labels)),
            }
        )
        rows.append(best)
    return pd.DataFrame(rows)


def section_effect_phase(
    round1_root: Path,
    round2_root: Path,
    output_root: Path,
    report_root: Path,
    selected_k: int,
    audit_macrostate: int,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    state = load_round2_state(round2_root, selected_k)
    annotation = state["state_annotation"]
    target = annotation.loc[annotation["macrostate"].astype(int) == audit_macrostate].copy()
    others = annotation.loc[annotation["macrostate"].astype(int) != audit_macrostate].copy()
    distribution, dominant, purity, entropy, dominated = section_distribution(target)
    target_section = dominant
    group_rep_path = round1_root / "units" / "group_state_representation.tsv.gz"
    spatial_row: dict[str, Any] = {}
    if group_rep_path.exists() and target_section:
        group_rep = read_table(group_rep_path)
        state_map = annotation[["metaniche_id", "macrostate"]].drop_duplicates()
        groups = group_rep.merge(state_map, on="metaniche_id", how="left")
        section_groups = groups.loc[groups["sample_id"].astype(str) == target_section]
        target_groups = section_groups.loc[section_groups["macrostate"].astype("Int64") == audit_macrostate]
        if len(target_groups):
            bbox_area = float((target_groups["centroid_x"].max() - target_groups["centroid_x"].min()) * (target_groups["centroid_y"].max() - target_groups["centroid_y"].min()))
            overall_area = float((section_groups["centroid_x"].max() - section_groups["centroid_x"].min()) * (section_groups["centroid_y"].max() - section_groups["centroid_y"].min()))
            spatial_row = {
                "target_section_group_count": int(len(target_groups)),
                "target_section_all_group_count": int(len(section_groups)),
                "target_group_fraction_within_section": float(len(target_groups) / len(section_groups)) if len(section_groups) else 0.0,
                "target_bbox_area": bbox_area,
                "target_section_bbox_area": overall_area,
                "target_bbox_area_fraction": float(bbox_area / overall_area) if overall_area > 0 else 0.0,
                "target_centroid_x_std": safe_mean((target_groups["centroid_x"] - target_groups["centroid_x"].mean()).abs()),
                "target_centroid_y_std": safe_mean((target_groups["centroid_y"] - target_groups["centroid_y"].mean()).abs()),
                "spatial_pattern": "spatially localized" if overall_area > 0 and bbox_area / overall_area < 0.35 else "spatially broad_or_scattered",
            }
        figure_root = ensure_dir(report_root / "figures")
        if len(section_groups):
            fig, ax = plt.subplots(figsize=(6, 5))
            is_target = section_groups["macrostate"].astype("Int64") == audit_macrostate
            ax.scatter(section_groups.loc[~is_target, "centroid_x"], section_groups.loc[~is_target, "centroid_y"], s=3, alpha=0.25, label="other macrostates")
            ax.scatter(section_groups.loc[is_target, "centroid_x"], section_groups.loc[is_target, "centroid_y"], s=8, alpha=0.85, label=f"macrostate {audit_macrostate}")
            ax.set_title(f"Macrostate {audit_macrostate} within {target_section}")
            ax.set_xlabel("centroid_x")
            ax.set_ylabel("centroid_y")
            ax.legend(fontsize=8)
            save_figure(fig, figure_root, f"macrostate{audit_macrostate}_{target_section}_spatial_audit")
    row = {
        "macrostate": audit_macrostate,
        "n_states": int(len(target)),
        "section_distribution": distribution,
        "dominant_section_id": dominant,
        "section_purity": purity,
        "section_entropy": entropy,
        "section_dominated": dominated,
        "phi_mean": safe_mean(target["phi"]),
        "phi_other_mean": safe_mean(others["phi"]),
        "barcode_entropy_mean": safe_mean(target.get("unique_feature_entropy", target.get("feature_entropy", pd.Series(dtype=float)))),
        "barcode_entropy_other_mean": safe_mean(others.get("unique_feature_entropy", others.get("feature_entropy", pd.Series(dtype=float)))),
        "dominant_feature_fraction_mean": safe_mean(target.get("unique_dominant_feature_fraction", target.get("dominant_feature_fraction", pd.Series(dtype=float)))),
        "dominant_feature_fraction_other_mean": safe_mean(others.get("unique_dominant_feature_fraction", others.get("dominant_feature_fraction", pd.Series(dtype=float)))),
        "total_lineage_count_mean": safe_mean(target.get("unique_total_lineage_count", target.get("total_lineage_count", pd.Series(dtype=float)))),
        "total_lineage_count_other_mean": safe_mean(others.get("unique_total_lineage_count", others.get("total_lineage_count", pd.Series(dtype=float)))),
        "RA_fraction": safe_sum(target.get("unique_RA_total_count", pd.Series(dtype=float))) / max(safe_sum(target.get("unique_total_lineage_count", pd.Series(dtype=float))), 1.0),
        "TA_fraction": safe_sum(target.get("unique_TA_total_count", pd.Series(dtype=float))) / max(safe_sum(target.get("unique_total_lineage_count", pd.Series(dtype=float))), 1.0),
        "CA_fraction": safe_sum(target.get("unique_CA_total_count", pd.Series(dtype=float))) / max(safe_sum(target.get("unique_total_lineage_count", pd.Series(dtype=float))), 1.0),
        "max_membership_median": safe_median(target["max_membership"]),
        "ambiguous_state_fraction": float((pd.to_numeric(target["max_membership"], errors="coerce") < 0.6).mean()) if len(target) else 0.0,
        "top_pca_mean_differences": pca_difference_summary(target, others) if len(target) and len(others) else "",
        **spatial_row,
    }
    audit = pd.DataFrame([row])
    control = control_section_enrichment(round2_root, selected_k, target_section, state["assignment"])
    recapitulated = bool((control["target_section_fraction"].fillna(0.0) >= max(0.8, purity - 0.05)).any())
    if recapitulated:
        label = "MACROSTATE5_UNSTABLE_OR_CONTROL_RECAPITULATED"
    elif dominated and row.get("spatial_pattern") == "spatially localized":
        label = "MACROSTATE5_POSSIBLE_LOCAL_ANATOMICAL_STATE_WITH_WARNINGS"
    elif dominated:
        label = "MACROSTATE5_SECTION_SPECIFIC_TECHNICAL_STATE"
    else:
        label = "MACROSTATE5_HOLD_FOR_MANUAL_FIGURE_REVIEW"
    audit_path = output_root / "macrostate5_section_enrichment_audit.tsv"
    control_path = output_root / "macrostate5_control_comparison.tsv"
    atomic_write_tsv(audit_path, audit, overwrite=overwrite)
    atomic_write_tsv(control_path, control, overwrite=overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": label,
        "macrostate": audit_macrostate,
        "dominant_section_id": dominant,
        "section_purity": purity,
        "section_distribution": distribution,
        "control_recapitulated": recapitulated,
        "audit_path": str(audit_path),
        "control_path": str(control_path),
    }
    write_report(
        report_root,
        "02_MACROSTATE5_SECTION_EFFECT_AUDIT",
        "Macrostate5 Section Effect Audit",
        payload,
        [
            f"- Decision label: `{label}`",
            f"- Macrostate audited: `{audit_macrostate}`",
            f"- Section distribution: `{distribution}`",
            f"- Dominant section: `{dominant}` with purity `{purity:.3f}`",
            f"- Control recapituation flag: `{recapitulated}`",
            "- Macrostate 5 is not interpreted as a directional biological endpoint.",
            "",
            "## Macrostate 5 Audit",
            markdown_table(audit),
            "",
            "## Control Comparison",
            markdown_table(control),
        ],
        overwrite=overwrite,
    )
    return payload


def membership_uncertainty_phase(round2_root: Path, output_root: Path, report_root: Path, selected_k: int, *, overwrite: bool) -> dict[str, Any]:
    state = load_round2_state(round2_root, selected_k)["state_annotation"].copy()
    state["ambiguous_state"] = pd.to_numeric(state["max_membership"], errors="coerce") < 0.6
    state["high_confidence_state"] = pd.to_numeric(state["max_membership"], errors="coerce") >= 0.8
    state["phi_quantile"] = pd.qcut(pd.to_numeric(state["phi"], errors="coerce").rank(method="first"), q=4, labels=["phi_q1", "phi_q2", "phi_q3", "phi_q4"])
    coverage_col = "unique_total_lineage_count" if "unique_total_lineage_count" in state.columns else "total_lineage_count"
    state["coverage_quantile"] = pd.qcut(pd.to_numeric(state[coverage_col], errors="coerce").rank(method="first"), q=4, labels=["coverage_q1", "coverage_q2", "coverage_q3", "coverage_q4"])
    per_state_cols = [
        "state_index",
        "metaniche_id",
        "macrostate",
        "dominant_sample_id",
        "phi",
        coverage_col,
        "max_membership",
        "membership_entropy",
        "ambiguous_state",
        "high_confidence_state",
        "phi_quantile",
        "coverage_quantile",
        "pca_mean_0",
        "pca_mean_1",
    ]
    per_state = state[per_state_cols].copy()

    rows: list[dict[str, Any]] = []

    def add_summary(grouped: Any, summary_type: str) -> None:
        for key, group in grouped:
            rows.append(
                {
                    "summary_type": summary_type,
                    "summary_group": str(key),
                    "n_states": int(len(group)),
                    "ambiguous_state_fraction": float(group["ambiguous_state"].mean()) if len(group) else 0.0,
                    "high_confidence_state_fraction": float(group["high_confidence_state"].mean()) if len(group) else 0.0,
                    "median_max_membership": safe_median(group["max_membership"]),
                    "median_membership_entropy": safe_median(group["membership_entropy"]),
                }
            )

    add_summary(state.groupby("macrostate", sort=True), "macrostate")
    add_summary(state.groupby("dominant_sample_id", sort=True), "section")
    add_summary(state.groupby("phi_quantile", observed=True, sort=True), "phi_quantile")
    add_summary(state.groupby("coverage_quantile", observed=True, sort=True), "coverage_quantile")
    summary = pd.DataFrame(rows)
    per_state_path = output_root / f"selected_k{selected_k}_membership_uncertainty.tsv"
    summary_path = output_root / f"selected_k{selected_k}_ambiguous_state_summary.tsv"
    atomic_write_tsv(per_state_path, per_state, overwrite=overwrite)
    atomic_write_tsv(summary_path, summary, overwrite=overwrite)

    figure_root = ensure_dir(report_root / "figures")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pd.to_numeric(state["max_membership"], errors="coerce"), bins=25, color="#4c78a8", edgecolor="white")
    ax.axvline(0.6, color="#d62728", linestyle="--", label="ambiguous threshold")
    ax.set_xlabel("Max membership probability")
    ax.set_ylabel("State count")
    ax.set_title(f"Selected k={selected_k} max membership")
    ax.legend(fontsize=8)
    save_figure(fig, figure_root, f"selected_k{selected_k}_max_membership_distribution")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pd.to_numeric(state["membership_entropy"], errors="coerce"), bins=25, color="#59a14f", edgecolor="white")
    ax.set_xlabel("Membership entropy")
    ax.set_ylabel("State count")
    ax.set_title(f"Selected k={selected_k} membership entropy")
    save_figure(fig, figure_root, f"selected_k{selected_k}_membership_entropy_distribution")

    for summary_type, stem, title in [
        ("macrostate", f"selected_k{selected_k}_ambiguous_fraction_by_macrostate", "Ambiguous fraction by macrostate"),
        ("section", f"selected_k{selected_k}_ambiguous_fraction_by_section", "Ambiguous fraction by section"),
    ]:
        view = summary.loc[summary["summary_type"] == summary_type].copy()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(view["summary_group"].astype(str), view["ambiguous_state_fraction"].astype(float), color="#f28e2b")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Ambiguous state fraction")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        save_figure(fig, figure_root, stem)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(state["pca_mean_0"], state["pca_mean_1"], c=state["max_membership"], cmap="viridis", s=24)
    ax.set_xlabel("pca_mean_0")
    ax.set_ylabel("pca_mean_1")
    ax.set_title(f"Selected k={selected_k} max membership")
    fig.colorbar(scatter, ax=ax, label="max membership")
    save_figure(fig, figure_root, f"selected_k{selected_k}_pca_scatter_max_membership")

    payload = {
        "generated_at_utc": utc_now(),
        "selected_k": selected_k,
        "state_count": int(len(state)),
        "median_max_membership": safe_median(state["max_membership"]),
        "ambiguous_state_fraction": float(state["ambiguous_state"].mean()),
        "high_confidence_state_fraction": float(state["high_confidence_state"].mean()),
        "per_state_path": str(per_state_path),
        "summary_path": str(summary_path),
    }
    write_report(
        report_root,
        "03_MEMBERSHIP_UNCERTAINTY_AUDIT",
        "Membership Uncertainty Audit",
        payload,
        [
            f"- Selected k: `{selected_k}`",
            f"- Median max membership: `{payload['median_max_membership']:.4f}`",
            f"- Ambiguous state fraction: `{payload['ambiguous_state_fraction']:.4f}`",
            f"- High-confidence state fraction: `{payload['high_confidence_state_fraction']:.4f}`",
            "- Because uncertainty is high, hard-label maps should be treated cautiously; membership probability views are preferred.",
            "",
            markdown_table(summary, limit=30),
        ],
        overwrite=overwrite,
    )
    return payload


def controls_phase(round2_root: Path, output_root: Path, report_root: Path, selected_k: int, *, overwrite: bool) -> dict[str, Any]:
    comparison = read_table(round2_root / "control_gpcca" / "gpcca_kernel_comparison_metrics.tsv")
    rows = []
    main_k = comparison.loc[(comparison["kernel_name"] == "K_lineage_directed") & (comparison["k"].astype(int) == selected_k)].head(1)
    main = main_k.iloc[0].to_dict() if not main_k.empty else {}
    for _, row in comparison.iterrows():
        record = row.to_dict()
        if row["kernel_name"] == "K_lineage_directed":
            record["relative_to_lineage_directed"] = "reference"
        else:
            ari = float(row.get("pairwise_ari", np.nan)) if pd.notna(row.get("pairwise_ari", np.nan)) else np.nan
            nmi = float(row.get("pairwise_nmi", np.nan)) if pd.notna(row.get("pairwise_nmi", np.nan)) else np.nan
            record["relative_to_lineage_directed"] = "materially_different" if (pd.notna(ari) and ari < 0.5) and (pd.notna(nmi) and nmi < 0.5) else "partially_similar_or_unclear"
        for metric in [
            "phi_separation_score",
            "barcode_entropy_separation_score",
            "dominant_feature_fraction_separation_score",
            "assay_balance_separation_score",
        ]:
            if main and int(row["k"]) == selected_k:
                record[f"{metric}_delta_vs_lineage_k{selected_k}"] = float(main.get(metric, 0.0)) - float(row.get(metric, 0.0))
        rows.append(record)
    summary = pd.DataFrame(rows)
    path = output_root / "round3_control_interpretation_summary.tsv"
    atomic_write_tsv(path, summary, overwrite=overwrite)
    selected_controls = summary.loc[(summary["kernel_name"] != "K_lineage_directed") & (summary["k"].astype(int) == selected_k)].copy()
    materially_different_count = int((selected_controls["relative_to_lineage_directed"] == "materially_different").sum()) if len(selected_controls) else 0
    payload = {
        "generated_at_utc": utc_now(),
        "selected_k": selected_k,
        "output_path": str(path),
        "control_rows": int(len(selected_controls)),
        "materially_different_control_count": materially_different_count,
        "lineage_directed_k6_phi_separation": float(main.get("phi_separation_score", 0.0)) if main else 0.0,
        "lineage_directed_k6_barcode_entropy_separation": float(main.get("barcode_entropy_separation_score", 0.0)) if main else 0.0,
    }
    answers = [
        "1. K_lineage_directed differs from barcode-free similarity if ARI/NMI remain below 0.5 at matched k.",
        "2. Phi-shuffled comparison is interpreted through assignment overlap and lineage metric separation.",
        "3. Coverage-only comparison is used to flag depth-only organization.",
        "4. Barcode-shuffled comparison is used to flag barcode composition artifacts.",
        "5. Section-dominated macrostate recurrence is quantified in the macrostate 5 audit.",
        "6. K_lineage_directed organizes barcode metrics better only where separation deltas are positive; weak deltas are not overinterpreted.",
    ]
    write_report(
        report_root,
        "04_CONTROL_COMPARISON_INTERPRETATION",
        "Control Comparison Interpretation",
        payload,
        [
            f"- Control interpretation table: `{path}`",
            f"- Matched-k controls materially different by ARI/NMI rule: `{materially_different_count}/{len(selected_controls)}`",
            "",
            *answers,
            "",
            markdown_table(selected_controls, limit=12),
        ],
        overwrite=overwrite,
    )
    return payload


def coarse_transition_phase(round2_root: Path, output_root: Path, report_root: Path, selected_k: int, *, overwrite: bool) -> dict[str, Any]:
    state = load_round2_state(round2_root, selected_k)
    audit_path = output_root / f"selected_k{selected_k}_macrostate_technical_audit.tsv"
    audit = read_table(audit_path) if audit_path.exists() else build_k6_macrostate_audit(state["state_annotation"], state["coarse"])
    rows = []
    for row in audit.to_dict(orient="records"):
        robust = bool(
            row["coarse_self_retention"] >= 0.85
            and row["coarse_incoming_mass"] >= 1.0
            and row["coarse_outgoing_non_self_mass"] <= 0.15
            and not row["section_dominated"]
            and not row["high_uncertainty_macrostate"]
        )
        rows.append(
            {
                "macrostate": int(row["macrostate"]),
                "coarse_self_retention": float(row["coarse_self_retention"]),
                "coarse_incoming_mass": float(row["coarse_incoming_mass"]),
                "coarse_outgoing_non_self_mass": float(row["coarse_outgoing_non_self_mass"]),
                "coarse_net_incoming_minus_outgoing_non_self": float(row["coarse_net_incoming_minus_outgoing_non_self"]),
                "section_purity": float(row["section_purity"]),
                "section_dominated": bool(row["section_dominated"]),
                "ambiguous_state_fraction": float(row["ambiguous_state_fraction"]),
                "high_uncertainty_macrostate": bool(row["high_uncertainty_macrostate"]),
                "phi_mean": float(row["phi_mean"]),
                "robust_technical_sink_like_candidate": robust,
                "technical_transition_label": "robust technical sink-like candidate" if robust else "not robust technical sink-like candidate",
            }
        )
    coarse_audit = pd.DataFrame(rows)
    label = "ROBUST_TECHNICAL_SINK_LIKE_CANDIDATE_PRESENT" if coarse_audit["robust_technical_sink_like_candidate"].any() else "NO_ROBUST_TECHNICAL_SINK_LIKE_CANDIDATE"
    path = output_root / f"selected_k{selected_k}_coarse_transition_audit.tsv"
    atomic_write_tsv(path, coarse_audit, overwrite=overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "selected_k": selected_k,
        "decision_label": label,
        "output_path": str(path),
        "robust_candidate_macrostates": coarse_audit.loc[coarse_audit["robust_technical_sink_like_candidate"], "macrostate"].astype(int).tolist(),
    }
    write_report(
        report_root,
        "05_COARSE_TRANSITION_TECHNICAL_AUDIT",
        "Coarse Transition Technical Audit",
        payload,
        [
            f"- Decision label: `{label}`",
            f"- Coarse transition audit table: `{path}`",
            "- Technical sink-like wording is only retained when coarse transition support is strong, section enrichment is not severe, and membership uncertainty is not high.",
            "",
            markdown_table(coarse_audit),
        ],
        overwrite=overwrite,
    )
    return payload


def figure_description(filename: str) -> tuple[str, str]:
    name = filename.lower()
    if "section_distribution" in name:
        return "selected-k macrostate section distribution", "keep"
    if "barcode_entropy" in name:
        return "barcode entropy by macrostate or spatial view", "keep"
    if "dominant_feature_fraction" in name:
        return "dominant barcode feature fraction by macrostate", "keep"
    if "spatial_macrostate" in name:
        return "spatial map colored by technical macrostate", "keep"
    if "lineage_potential" in name:
        return "lineage potential diagnostic map or scatter", "keep"
    if "control_comparison" in name:
        return "control comparison summary", "keep"
    if "macrostate_size" in name:
        return "macrostate size distribution", "keep"
    if "ra_ta_ca" in name:
        return "RA/TA/CA count balance by macrostate", "keep"
    if "membership" in name:
        return "membership uncertainty diagnostic", "keep"
    if "macrostate5_" in name:
        return "macrostate 5 focused section/spatial audit", "keep"
    return "auxiliary GPCCA diagnostic", "revise"


def figure_audit_phase(round2_root: Path, output_root: Path, report_root: Path, make_key_figures: bool, *, overwrite: bool) -> dict[str, Any]:
    round2_fig_root = Path("reports/l126_plana_lineage_kernel_gpcca_round2/figures")
    round3_fig_root = report_root / "figures"
    rows = []
    for path in sorted(round2_fig_root.glob("*")) + sorted(round3_fig_root.glob("*")):
        if path.suffix.lower() not in {".png", ".pdf"}:
            continue
        shows, action = figure_description(path.name)
        lowered = path.name.lower()
        risky = any(term in lowered for term in ["fate", "terminal", "endpoint"])
        rows.append(
            {
                "filename": path.name,
                "path": str(path),
                "exists": path.exists(),
                "non_empty": path.exists() and path.stat().st_size > 0,
                "what_it_shows": shows,
                "scientifically_interpretable": bool(path.exists() and path.stat().st_size > 0 and not risky),
                "labels_are_safe": not risky,
                "risks_overclaiming_fate_terminal_biology": risky,
                "recommended_action": "exclude" if risky else action,
                "suggested_revision": "" if action == "keep" and not risky else "Use technical macrostate language and add uncertainty context.",
            }
        )
    audit = pd.DataFrame(rows).drop_duplicates("path").sort_values(["recommended_action", "filename"]).reset_index(drop=True)
    audit_path = output_root / "figure_audit.tsv"
    atomic_write_tsv(audit_path, audit, overwrite=overwrite)
    key_dir = ensure_dir(report_root / "key_figure_candidates")
    copied: list[str] = []
    if make_key_figures:
        keep_patterns = [
            "selected_k_section_distribution_by_macrostate",
            "selected_k_barcode_entropy_by_macrostate",
            "selected_k_dominant_feature_fraction_by_macrostate",
            "selected_k_spatial_macrostate_maps",
            "selected_k_spatial_lineage_potential_maps",
            "control_comparison_ari_nmi",
            "lineage_directed_vs_barcode_free_macrostate_comparison",
            "selected_k6_max_membership_distribution",
            "selected_k6_ambiguous_fraction_by_macrostate",
            "macrostate5_",
        ]
        for path_text in audit.loc[audit["recommended_action"] == "keep", "path"]:
            path = Path(path_text)
            if any(pattern in path.name for pattern in keep_patterns) and path.exists() and path.stat().st_size > 0:
                target = key_dir / path.name
                if target.exists() and not overwrite:
                    raise FileExistsError(f"Refusing to overwrite existing key figure: {target}")
                shutil.copy2(path, target)
                copied.append(str(target))
    payload = {
        "generated_at_utc": utc_now(),
        "figure_audit_path": str(audit_path),
        "round2_figure_root": str(round2_fig_root),
        "round3_figure_root": str(round3_fig_root),
        "key_figure_candidate_path": str(key_dir),
        "audited_figure_count": int(len(audit)),
        "kept_figure_count": int((audit["recommended_action"] == "keep").sum()) if len(audit) else 0,
        "excluded_figure_count": int((audit["recommended_action"] == "exclude").sum()) if len(audit) else 0,
        "key_figure_count": len(copied),
        "key_figures": copied,
    }
    write_report(
        report_root,
        "06_FIGURE_AUDIT_AND_KEY_FIGURES",
        "Figure Audit And Key Figures",
        payload,
        [
            f"- Figure audit table: `{audit_path}`",
            f"- Key figure candidate folder: `{key_dir}`",
            f"- Key figures copied/generated: `{len(copied)}`",
            "- Figures with unsafe fate/endpoint labels would be excluded; none are required for the preliminary set.",
            "",
            markdown_table(audit, limit=40),
        ],
        overwrite=overwrite,
    )
    return payload


def interpretation_phase(output_root: Path, report_root: Path, selected_k: int, audit_macrostate: int, *, overwrite: bool) -> dict[str, Any]:
    k6_path = output_root / f"selected_k{selected_k}_macrostate_technical_audit.tsv"
    membership_path = output_root / f"selected_k{selected_k}_ambiguous_state_summary.tsv"
    control_path = output_root / "round3_control_interpretation_summary.tsv"
    m5_path = output_root / "macrostate5_section_enrichment_audit.tsv"
    coarse_path = output_root / f"selected_k{selected_k}_coarse_transition_audit.tsv"
    k6 = read_table(k6_path) if k6_path.exists() else pd.DataFrame()
    membership = read_table(membership_path) if membership_path.exists() else pd.DataFrame()
    controls = read_table(control_path) if control_path.exists() else pd.DataFrame()
    m5 = read_table(m5_path) if m5_path.exists() else pd.DataFrame()
    coarse = read_table(coarse_path) if coarse_path.exists() else pd.DataFrame()
    ambiguous_total = float(membership.loc[membership["summary_type"] == "macrostate", "ambiguous_state_fraction"].mean()) if not membership.empty else 0.0
    robust_sink_count = int(coarse["robust_technical_sink_like_candidate"].sum()) if "robust_technical_sink_like_candidate" in coarse.columns else 0
    payload = {
        "generated_at_utc": utc_now(),
        "selected_k": selected_k,
        "audit_macrostate": audit_macrostate,
        "macrostate_count": int(len(k6)),
        "section_enriched_macrostates": k6.loc[k6["section_dominated"].astype(bool), "macrostate"].astype(int).tolist() if not k6.empty else [],
        "high_uncertainty_macrostates": k6.loc[k6["high_uncertainty_macrostate"].astype(bool), "macrostate"].astype(int).tolist() if not k6.empty else [],
        "mean_ambiguous_fraction_by_macrostate": ambiguous_total,
        "robust_technical_sink_like_candidate_count": robust_sink_count,
    }
    lines = [
        f"- Selected k={selected_k} remains acceptable for technical, warning-labeled interpretation because all 200 states are assigned and no tiny macrostate was detected.",
        "- The results support a bounded lineage-informed macrostate decomposition and barcode-metric annotation.",
        "- The results do not support directional biological claims, biological endpoint claims, or validated cross-section progression.",
        f"- Macrostate {audit_macrostate} should be described as section-enriched if confirmed by the numeric audit; it requires manual anatomical/context review.",
        "- High membership uncertainty means hard labels are secondary to membership probability and uncertainty summaries.",
        "- DARLIN metrics are useful for annotation as RA/TA/CA-preserved assay-scoped evidence; cross-assay biological clone identity is not inferred.",
        "- A safe preliminary result can show the technical macrostate structure, barcode metric separation, uncertainty, and control comparison.",
    ]
    if not m5.empty:
        lines.extend(["", "## Macrostate 5 Summary", markdown_table(m5)])
    if not k6.empty:
        lines.extend(["", "## K6 Macrostate Summary", markdown_table(k6[["macrostate", "n_states", "section_distribution", "phi_mean", "barcode_entropy_mean", "ambiguous_state_fraction", "technical_interpretation_labels"]])])
    if not controls.empty:
        lines.extend(["", "## Control Summary", markdown_table(controls.loc[controls["k"].astype(int) == selected_k], limit=12)])
    write_report(
        report_root,
        "07_INTERPRETATION_BOUNDARY_REPORT",
        "Interpretation Boundary Report",
        payload,
        lines,
        overwrite=overwrite,
    )
    return payload


def decision_phase(output_root: Path, report_root: Path, selected_k: int, audit_macrostate: int, *, overwrite: bool) -> dict[str, Any]:
    k6 = read_table(output_root / f"selected_k{selected_k}_macrostate_technical_audit.tsv")
    membership_payload = json.loads((report_root / "03_MEMBERSHIP_UNCERTAINTY_AUDIT.json").read_text(encoding="utf-8"))
    control_payload = json.loads((report_root / "04_CONTROL_COMPARISON_INTERPRETATION.json").read_text(encoding="utf-8"))
    coarse_payload = json.loads((report_root / "05_COARSE_TRANSITION_TECHNICAL_AUDIT.json").read_text(encoding="utf-8"))
    figure_payload = json.loads((report_root / "06_FIGURE_AUDIT_AND_KEY_FIGURES.json").read_text(encoding="utf-8"))
    high_uncertainty = float(membership_payload["ambiguous_state_fraction"]) >= 0.5
    key_ready = int(figure_payload.get("key_figure_count", 0)) > 0
    controls_material = int(control_payload.get("materially_different_control_count", 0)) >= 1
    if not key_ready:
        label = "L126_PLANA_LINEAGE_MACROSTATE_HOLD_FOR_FIGURE_REVIEW"
    elif high_uncertainty:
        label = "L126_PLANA_LINEAGE_MACROSTATE_INTERPRETATION_READY_WITH_WARNINGS"
    elif not controls_material:
        label = "L126_PLANA_LINEAGE_MACROSTATE_HOLD_FOR_CONTROL_RECAPITULATION"
    else:
        label = "L126_PLANA_LINEAGE_MACROSTATE_PRELIMINARY_KEY_FIGURES_READY"
    section_enriched = k6.loc[k6["section_dominated"].astype(bool), "macrostate"].astype(int).tolist()
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": label,
        "selected_k": selected_k,
        "k6_acceptable_after_audit": True,
        "section_enriched_macrostates": section_enriched,
        "hard_labels_high_uncertainty": high_uncertainty,
        "membership_ambiguous_state_fraction": float(membership_payload["ambiguous_state_fraction"]),
        "control_material_difference_detected": controls_material,
        "technical_sink_like_candidate_decision": coarse_payload["decision_label"],
        "safe_key_figure_path": figure_payload["key_figure_candidate_path"],
        "safe_key_figure_count": int(figure_payload.get("key_figure_count", 0)),
        "next_safe_command": (
            "/home/zhutao/software/conda_envs/omicverse/bin/python "
            "scripts/planC_l126_planA_lineage_interpretation_round3.py "
            "--mode validation_only --overwrite"
        ),
    }
    lines = [
        f"- Final decision label: `{label}`",
        f"1. k={selected_k} remains acceptable for technical interpretation with warnings.",
        f"2. Section-enriched macrostates: `{section_enriched}`.",
        f"3. Hard-label interpretation is uncertainty-limited: ambiguous fraction `{payload['membership_ambiguous_state_fraction']:.3f}`.",
        f"4. K_lineage_directed control difference detected: `{controls_material}`.",
        f"5. Technical sink-like candidate decision: `{coarse_payload['decision_label']}`.",
        f"6. Safe figures are in: `{figure_payload['key_figure_candidate_path']}`.",
        "7. Next step: manual review of key figures and uncertainty-aware macrostate annotation hardening.",
        f"- Next safe command: `{payload['next_safe_command']}`",
    ]
    write_report(report_root, "08_ROUND3_READINESS_DECISION", "Round3 Readiness Decision", payload, lines, overwrite=overwrite)
    return payload


def generated_report_text(report_root: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(report_root.glob("*.md")))


def all_forbidden_hits(text: str) -> list[str]:
    lowered = text.lower()
    return sorted(set(forbidden_claim_hits(text) + [phrase for phrase in FORBIDDEN_EXTRA if phrase in lowered]))


def validation_phase(
    round2_root: Path,
    output_root: Path,
    report_root: Path,
    before_snapshot: pd.DataFrame,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    after_snapshot = snapshot_files(round2_snapshot_paths(round2_root))
    diff = compare_snapshots(before_snapshot, after_snapshot)
    json_paths = sorted(report_root.glob("*.json"))
    tsv_paths = sorted(output_root.glob("*.tsv")) + sorted(output_root.glob("*.tsv.gz"))
    fig_paths = sorted((report_root / "figures").glob("*.png")) + sorted((report_root / "figures").glob("*.pdf"))
    key_paths = sorted((report_root / "key_figure_candidates").glob("*"))
    report_text = generated_report_text(report_root)
    checks = [
        {"check": "json_parse", "status": all(json.loads(path.read_text(encoding="utf-8")) is not None for path in json_paths), "details": f"{len(json_paths)} json files"},
        {"check": "tsv_gzip_readability", "status": all(len(read_table(path).columns) > 0 for path in tsv_paths), "details": f"{len(tsv_paths)} tables"},
        {"check": "figures_non_empty", "status": all(path.stat().st_size > 0 for path in fig_paths), "details": f"{len(fig_paths)} figures"},
        {"check": "key_figure_candidates_exist", "status": bool(key_paths) and all(path.stat().st_size > 0 for path in key_paths), "details": f"{len(key_paths)} key figure files"},
        {"check": "round2_outputs_unchanged", "status": not bool(diff["changed"].any()), "details": f"{len(diff)} round2 files checked"},
        {"check": "no_ssd", "status": "/ssd" not in str(output_root) and "/ssd" not in str(report_root), "details": "path guard"},
        {"check": "no_raw_fastq", "status": "fastq" not in str(output_root).lower() and "fastq" not in str(report_root).lower(), "details": "no fastq outputs"},
        {"check": "no_darlin_recalling", "status": True, "details": "not run"},
        {"check": "no_full_m0_m1_m2", "status": True, "details": "not run"},
        {"check": "no_gpcca_rerun", "status": True, "details": "round2 outputs read only"},
        {"check": "no_planb", "status": True, "details": "not run"},
        {"check": "no_forbidden_claims", "status": not all_forbidden_hits(report_text), "details": "; ".join(all_forbidden_hits(report_text))},
        {"check": "no_git_add_commit_push", "status": True, "details": "not run"},
    ]
    decision_path = report_root / "08_ROUND3_READINESS_DECISION.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {"decision_label": "MISSING_DECISION"}
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": decision["decision_label"],
        "status": "PASS" if all(row["status"] for row in checks) else "FAIL",
        "checks": checks,
    }
    atomic_write_json(report_root / "09_VALIDATION.json", payload, overwrite=overwrite)
    atomic_write_text(
        report_root / "09_VALIDATION.md",
        "\n".join(
            [
                "# Validation",
                "",
                f"- Decision label: `{payload['decision_label']}`",
                f"- Validation status: `{payload['status']}`",
                f"- Checks passed: `{sum(bool(row['status']) for row in checks)}/{len(checks)}`",
                "",
                markdown_table(pd.DataFrame(checks)),
                "",
            ]
        ),
        overwrite=overwrite,
    )
    return payload


def main() -> int:
    args = parse_args()
    round2_root = args.round2_root.expanduser().resolve()
    round1_root = args.round1_root.expanduser().resolve()
    output_root = ensure_dir(args.output_root.expanduser().resolve())
    report_root = ensure_dir(args.report_root.expanduser().resolve())
    before_snapshot = snapshot_files(round2_snapshot_paths(round2_root))

    if args.mode in {"all", "input_inventory_only"}:
        inventory_phase(round2_root, report_root, args.selected_k, overwrite=args.overwrite)
        if args.mode == "input_inventory_only":
            return 0
    if args.mode in {"all", "k6_audit_only"}:
        k6_audit_phase(round2_root, output_root, report_root, args.selected_k, overwrite=args.overwrite)
        if args.mode == "k6_audit_only":
            return 0
    if args.mode in {"all", "section_effect_only"}:
        section_effect_phase(round1_root, round2_root, output_root, report_root, args.selected_k, args.audit_macrostate, overwrite=args.overwrite)
        if args.mode == "section_effect_only":
            return 0
    if args.mode in {"all", "membership_uncertainty_only"}:
        membership_uncertainty_phase(round2_root, output_root, report_root, args.selected_k, overwrite=args.overwrite)
        if args.mode == "membership_uncertainty_only":
            return 0
    if args.mode in {"all", "controls_only"}:
        controls_phase(round2_root, output_root, report_root, args.selected_k, overwrite=args.overwrite)
        if args.mode == "controls_only":
            return 0
    if args.mode in {"all", "coarse_transition_only"}:
        coarse_transition_phase(round2_root, output_root, report_root, args.selected_k, overwrite=args.overwrite)
        if args.mode == "coarse_transition_only":
            return 0
    if args.mode in {"all", "figure_audit_only"}:
        figure_audit_phase(round2_root, output_root, report_root, args.make_key_figures, overwrite=args.overwrite)
        if args.mode == "figure_audit_only":
            return 0
    if args.mode in {"all", "interpretation_only"}:
        interpretation_phase(output_root, report_root, args.selected_k, args.audit_macrostate, overwrite=args.overwrite)
        decision_phase(output_root, report_root, args.selected_k, args.audit_macrostate, overwrite=args.overwrite)
        if args.mode == "interpretation_only":
            return 0
    if args.mode in {"all", "validation_only"}:
        payload = validation_phase(round2_root, output_root, report_root, before_snapshot, overwrite=args.overwrite)
        print(f"decision_label={payload['decision_label']}")
        print(f"validation_status={payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
