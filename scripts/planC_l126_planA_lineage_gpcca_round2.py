#!/usr/bin/env python
"""Run bounded GPCCA smoke for the L126 PlanA-L directed kernel.

This is a small technical GPCCA smoke over the already-built 200-state
lineage-informed directed kernel. It reads Round 1 outputs as immutable inputs,
audits control kernels before use, writes macrostate annotation/QC, and keeps
interpretation at the macrostate/reachability-like abstraction level.
"""

from __future__ import annotations

import argparse
import json
import os
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
import scipy.sparse as sp

from nichefate.barcode_adapter.qc import snapshot_files
from nichefate.barcode_adapter.reporting import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_tsv,
    atomic_write_tsv_gz,
    ensure_dir,
    markdown_table,
    utc_now,
)
from nichefate.planA_l.gpcca_round2 import (
    AMBIGUOUS_STATE_THRESHOLD,
    DEFAULT_MACROSTATE_GRID,
    _macrostate_barcode_summary,
    _macrostate_labels,
    _macrostate_section_summary,
    _macrostate_summary_payload,
    _macrostate_top_features,
    _read_table,
    _state_annotation_frame,
    _summary_for_macrostate,
    audit_gpcca_environment,
    build_kernel_comparison_metrics,
    run_gpcca_grid,
    select_technical_k,
    validate_round1_artifacts,
    validation_payload_for_round2,
)
from nichefate.planA_l.reporting import forbidden_claim_hits, write_report_pair
from nichefate.planA_k.gpcca_probe import safe_gpcca_runtime_dir


DEFAULT_INPUT_PACKET_ROOT = Path("/home/zhutao/scratch/nichefate/spatio_darlin_public_brain_hpc/input_packet")
DEFAULT_ROUND1_ROOT = Path("processed/l126_plana_lineage_kernel_round1")
DEFAULT_OUTPUT_ROOT = Path("processed/l126_plana_lineage_kernel_gpcca_round2")
DEFAULT_REPORT_ROOT = Path("reports/l126_plana_lineage_kernel_gpcca_round2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1-root", type=Path, default=DEFAULT_ROUND1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--input-packet-root", type=Path, default=DEFAULT_INPUT_PACKET_ROOT)
    parser.add_argument("--kernel-name", default="K_lineage_directed")
    parser.add_argument(
        "--control-kernels",
        default="K_expr_spatial_only,K_phi_shuffled,K_coverage_only,K_barcode_shuffled",
    )
    parser.add_argument("--n-macrostates", default="3,4,5,6")
    parser.add_argument("--selected-k", default="auto")
    parser.add_argument("--run-controls", action="store_true", default=False)
    parser.add_argument("--make-figures", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--mode",
        default="all",
        choices=[
            "all",
            "env_audit_only",
            "lineage_gpcca_only",
            "controls_only",
            "annotation_only",
            "figures_only",
            "validation_only",
        ],
    )
    return parser.parse_args()


def _parse_csv(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    return values or DEFAULT_MACROSTATE_GRID


def _load_state_metadata(round1_root: Path) -> pd.DataFrame:
    metadata = _read_table(round1_root / "kernel" / "state_metadata.tsv")
    lineage = _read_table(round1_root / "kernel" / "metaniche_lineage_potential.tsv")
    metadata = metadata.loc[:, ~metadata.columns.duplicated()].sort_values("state_index").reset_index(drop=True)
    lineage = lineage.loc[:, ~lineage.columns.duplicated()].sort_values("state_index").reset_index(drop=True)
    missing_cols = [col for col in lineage.columns if col not in metadata.columns]
    if missing_cols:
        metadata = metadata.merge(lineage[["state_index", *missing_cols]], on="state_index", how="left")
    return metadata


def _load_barcode_tables(round1_root: Path) -> dict[str, pd.DataFrame]:
    barcode_root = round1_root / "barcode"
    return {
        "unique": _read_table(barcode_root / "metaniche_barcode_annotation_unique_cellbin.tsv.gz"),
        "local": _read_table(barcode_root / "metaniche_barcode_annotation_local_context.tsv.gz"),
        "unique_top": _read_table(barcode_root / "metaniche_top_features_unique_cellbin.tsv.gz"),
        "local_top": _read_table(barcode_root / "metaniche_top_features_local_context.tsv.gz"),
    }


def _apply_safe_runtime_env() -> None:
    runtime = str(safe_gpcca_runtime_dir())
    for key in ["TMPDIR", "TEMP", "TMP", "XDG_RUNTIME_DIR"]:
        os.environ[key] = runtime
    for key in [
        "OMPI_MCA_orte_tmpdir_base",
        "OMPI_MCA_tmpdir_base",
        "PRTE_MCA_tmpdir_base",
        "PMIX_MCA_tmpdir_base",
        "PETSC_TMPDIR",
    ]:
        os.environ[key] = runtime


def _round1_snapshot_paths(round1_root: Path, kernel_name: str, control_kernels: tuple[str, ...]) -> list[Path]:
    paths = [
        round1_root / "kernel" / f"{kernel_name}.npz",
        round1_root / "kernel" / "state_metadata.tsv",
        round1_root / "kernel" / "metaniche_lineage_potential.tsv",
        round1_root / "barcode" / "metaniche_barcode_annotation_unique_cellbin.tsv.gz",
        round1_root / "barcode" / "metaniche_barcode_annotation_local_context.tsv.gz",
    ]
    paths.extend(round1_root / "controls" / f"{name}.npz" for name in control_kernels)
    return paths


def _read_coarse_table(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, sep="\t")
    value_cols = [col for col in frame.columns if col.startswith("macrostate_")]
    if not value_cols:
        return None
    return frame[value_cols].to_numpy(dtype=float)


def _write_phase_report(
    report_root: Path,
    stem: str,
    title: str,
    payload: dict[str, Any],
    lines: list[str],
    *,
    overwrite: bool,
) -> None:
    write_report_pair(report_root, stem, title, payload, "\n".join(lines), overwrite=overwrite)


def _save_figure(fig: plt.Figure, figure_root: Path, stem: str) -> None:
    ensure_dir(figure_root)
    fig.tight_layout()
    fig.savefig(figure_root / f"{stem}.png", dpi=180)
    fig.savefig(figure_root / f"{stem}.pdf")
    plt.close(fig)


def _plot_box_by_macrostate(frame: pd.DataFrame, value_col: str, selected_k: int, title: str, figure_root: Path, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    groups = [pd.to_numeric(group[value_col], errors="coerce").dropna().to_numpy() for _, group in frame.groupby("macrostate", sort=True)]
    labels = [str(key) for key in sorted(frame["macrostate"].unique())]
    if groups:
        ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_xlabel("Macrostate")
    ax.set_ylabel(value_col)
    ax.set_title(f"{title} (k={selected_k})")
    _save_figure(fig, figure_root, stem)


def make_round2_figures(
    output_root: Path,
    report_root: Path,
    selected_k: int | None,
    main_assignments: dict[int, pd.DataFrame],
    control_assignments: dict[str, dict[int, pd.DataFrame]],
    comparison: pd.DataFrame,
    state_annotation: pd.DataFrame | None,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    figure_root = ensure_dir(report_root / "figures")
    outputs: list[str] = []
    if main_assignments:
        fig, ax = plt.subplots(figsize=(7, 4))
        data = []
        labels = []
        for k, assignment in sorted(main_assignments.items()):
            data.append(assignment["macrostate"].value_counts().sort_index().to_numpy())
            labels.append(str(k))
        if data:
            ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_xlabel("k")
        ax.set_ylabel("States per macrostate")
        ax.set_title("Macrostate size distribution")
        _save_figure(fig, figure_root, "macrostate_size_distribution_k3_k4_k5_k6")
        outputs.extend(str(path) for path in sorted(figure_root.glob("macrostate_size_distribution_k3_k4_k5_k6.*")))

    if selected_k is not None and selected_k in main_assignments:
        assignment = main_assignments[selected_k].copy()
        x_col = "pca_mean_0" if "pca_mean_0" in assignment.columns else "centroid_x"
        y_col = "pca_mean_1" if "pca_mean_1" in assignment.columns else "centroid_y"

        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(assignment[x_col], assignment[y_col], c=assignment["macrostate"], s=20, cmap="tab20")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Selected k={selected_k} macrostate assignment")
        fig.colorbar(scatter, ax=ax, label="macrostate")
        _save_figure(fig, figure_root, "selected_k_pca_scatter_by_macrostate")

        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(assignment[x_col], assignment[y_col], c=pd.to_numeric(assignment["phi"], errors="coerce"), s=20, cmap="viridis")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Selected k={selected_k} lineage potential")
        fig.colorbar(scatter, ax=ax, label="phi")
        _save_figure(fig, figure_root, "selected_k_pca_scatter_by_lineage_potential")

        section_pivot = pd.crosstab(assignment["macrostate"], assignment["dominant_sample_id"])
        fig, ax = plt.subplots(figsize=(8, 4))
        bottom = np.zeros(len(section_pivot), dtype=float)
        xs = np.arange(len(section_pivot))
        for column in section_pivot.columns:
            values = section_pivot[column].to_numpy(dtype=float)
            ax.bar(xs, values, bottom=bottom, label=str(column))
            bottom += values
        ax.set_xticks(xs)
        ax.set_xticklabels([str(idx) for idx in section_pivot.index])
        ax.set_xlabel("Macrostate")
        ax.set_ylabel("State count")
        ax.set_title(f"Section distribution by macrostate (k={selected_k})")
        ax.legend(fontsize=8)
        _save_figure(fig, figure_root, "selected_k_section_distribution_by_macrostate")

        if state_annotation is not None and not state_annotation.empty:
            _plot_box_by_macrostate(
                state_annotation,
                "unique_feature_entropy",
                selected_k,
                "Unique-cellbin barcode entropy",
                figure_root,
                "selected_k_barcode_entropy_by_macrostate",
            )
            _plot_box_by_macrostate(
                state_annotation,
                "unique_dominant_feature_fraction",
                selected_k,
                "Unique-cellbin dominant feature fraction",
                figure_root,
                "selected_k_dominant_feature_fraction_by_macrostate",
            )
            assay = (
                state_annotation.groupby("macrostate")[["unique_RA_total_count", "unique_TA_total_count", "unique_CA_total_count"]]
                .sum()
                .sort_index()
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            bottom = np.zeros(len(assay), dtype=float)
            xs = np.arange(len(assay))
            for column in assay.columns:
                values = assay[column].to_numpy(dtype=float)
                ax.bar(xs, values, bottom=bottom, label=column.replace("unique_", "").replace("_total_count", ""))
                bottom += values
            ax.set_xticks(xs)
            ax.set_xticklabels([str(idx) for idx in assay.index])
            ax.set_xlabel("Macrostate")
            ax.set_ylabel("Unique-cellbin lineage count")
            ax.set_title(f"RA/TA/CA balance by macrostate (k={selected_k})")
            ax.legend(fontsize=8)
            _save_figure(fig, figure_root, "selected_k_ra_ta_ca_balance_by_macrostate")

        for color_col, stem, cmap in [
            ("macrostate", "selected_k_spatial_macrostate_maps", "tab20"),
            ("phi", "selected_k_spatial_lineage_potential_maps", "viridis"),
            ("feature_entropy", "selected_k_spatial_barcode_entropy_maps", "magma"),
        ]:
            if color_col not in assignment.columns:
                continue
            sections = sorted(assignment["dominant_sample_id"].astype(str).unique())
            fig, axes = plt.subplots(1, len(sections), figsize=(5 * len(sections), 4), squeeze=False)
            for ax, section in zip(axes.ravel(), sections):
                view = assignment.loc[assignment["dominant_sample_id"].astype(str) == section]
                sc = ax.scatter(
                    view["centroid_x"],
                    view["centroid_y"],
                    c=pd.to_numeric(view[color_col], errors="coerce") if color_col != "macrostate" else view[color_col],
                    s=18,
                    cmap=cmap,
                )
                ax.set_title(section)
                ax.set_xlabel("centroid_x")
                ax.set_ylabel("centroid_y")
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            _save_figure(fig, figure_root, stem)

    if not comparison.empty:
        valid_controls = comparison.loc[comparison["kernel_name"] != "K_lineage_directed"].copy()
        if not valid_controls.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = [f"{row.kernel_name}\nk={int(row.k)}" for row in valid_controls.itertuples(index=False)]
            xs = np.arange(len(valid_controls))
            ax.bar(xs - 0.18, pd.to_numeric(valid_controls["pairwise_ari"], errors="coerce").fillna(0), width=0.36, label="ARI")
            ax.bar(xs + 0.18, pd.to_numeric(valid_controls["pairwise_nmi"], errors="coerce").fillna(0), width=0.36, label="NMI")
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Agreement with lineage-directed")
            ax.set_title("Control macrostate agreement")
            ax.legend()
            _save_figure(fig, figure_root, "control_comparison_ari_nmi")

    if selected_k is not None and selected_k in main_assignments and "K_expr_spatial_only" in control_assignments:
        control = control_assignments["K_expr_spatial_only"].get(selected_k)
        if control is not None:
            main = main_assignments[selected_k].sort_values("state_index")
            ctrl = control.sort_values("state_index")
            table = pd.crosstab(main["macrostate"], ctrl["macrostate"])
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(table.to_numpy(dtype=float), cmap="Blues")
            ax.set_xlabel("Barcode-free macrostate")
            ax.set_ylabel("Lineage-directed macrostate")
            ax.set_title(f"Lineage-directed vs barcode-free assignments (k={selected_k})")
            fig.colorbar(im, ax=ax, label="state count")
            _save_figure(fig, figure_root, "lineage_directed_vs_barcode_free_macrostate_comparison")

    outputs.extend(str(path) for path in sorted(figure_root.glob("*.png")))
    outputs.extend(str(path) for path in sorted(figure_root.glob("*.pdf")))
    payload = {
        "generated_at_utc": utc_now(),
        "selected_k": selected_k,
        "figure_count": len(outputs),
        "figures": outputs,
    }
    _write_phase_report(
        report_root,
        "06_FIGURES",
        "Figures",
        payload,
        [
            f"- Figure directory: `{figure_root}`",
            f"- Figures written: {len(outputs)}",
            "- Figure labels use macrostate, lineage-informed kernel, and reachability-like wording only.",
        ],
        overwrite=overwrite,
    )
    return payload


def main() -> int:
    args = parse_args()
    _apply_safe_runtime_env()
    np.random.seed(args.seed)
    round1_root = args.round1_root.expanduser().resolve()
    output_root = ensure_dir(args.output_root.expanduser().resolve())
    report_root = ensure_dir(args.report_root.expanduser().resolve())
    control_kernels = _parse_csv(args.control_kernels)
    k_grid = _parse_ints(args.n_macrostates)
    qc_root = ensure_dir(output_root / "qc")

    source_before = snapshot_files([args.input_packet_root.expanduser().resolve()])
    round1_before = snapshot_files(_round1_snapshot_paths(round1_root, args.kernel_name, control_kernels))

    env_frame: pd.DataFrame | None = None
    env_payload: dict[str, Any] = {}
    readback_payload: dict[str, Any] = {}
    readback_checks: pd.DataFrame | None = None
    metadata = _load_state_metadata(round1_root)
    barcode_tables = _load_barcode_tables(round1_root)
    main_assignments: dict[int, pd.DataFrame] = {}
    main_memberships: dict[int, pd.DataFrame] = {}
    main_coarse_tables: dict[int, pd.DataFrame] = {}
    main_run_frame = pd.DataFrame()
    main_comparison = pd.DataFrame()
    control_assignments: dict[str, dict[int, pd.DataFrame]] = {}
    all_comparisons: list[pd.DataFrame] = []
    selected_k: int | None = None
    selection_payload: dict[str, Any] = {}
    state_annotation: pd.DataFrame | None = None
    annotation_payload: dict[str, Any] = {}
    figure_payload: dict[str, Any] = {}

    if args.mode in {"all", "env_audit_only"}:
        env_frame, env_payload = audit_gpcca_environment(("nichefate-gpcca", "omicverse"))
        atomic_write_tsv(qc_root / "environment_audit.tsv", env_frame, overwrite=args.overwrite)
        _write_phase_report(
            report_root,
            "00_ENVIRONMENT_AND_INPUT_AUDIT",
            "Environment And Input Audit",
            env_payload,
            [
                f"- Decision label: `{env_payload['decision_label']}`",
                f"- Selected pyGPCCA environment: `{env_payload.get('selected_environment')}`",
                "- No package installation or conda environment modification was performed.",
                "- Existing PlanA-K GPCCA helpers were reused for safe runtime settings and pyGPCCA fitting.",
                "",
                markdown_table(env_frame),
            ],
            overwrite=args.overwrite,
        )
        if args.mode == "env_audit_only":
            return 0 if env_payload.get("environment_check_passed") else 2

    if args.mode in {"all", "lineage_gpcca_only", "controls_only", "annotation_only", "figures_only", "validation_only"}:
        readback_checks, readback_payload = validate_round1_artifacts(round1_root, args.kernel_name, control_kernels)
        atomic_write_tsv(qc_root / "kernel_readback_checks.tsv", readback_checks, overwrite=args.overwrite)
        atomic_write_tsv(qc_root / "control_kernel_qc.tsv", pd.DataFrame(readback_payload["control_kernel_rows"]), overwrite=args.overwrite)
        _write_phase_report(
            report_root,
            "01_KERNEL_READBACK",
            "Kernel Readback",
            readback_payload,
            [
                f"- K_lineage_directed shape: `{tuple(readback_payload['kernel_shape'])}`",
                f"- Row-stochastic: `{readback_payload['kernel_row_stochastic']}`",
                f"- Negative entries: `{readback_payload['kernel_negative_entries']}`",
                f"- Zero rows: `{readback_payload['kernel_zero_rows']}`",
                f"- Barcode annotation join ready: `{readback_payload['barcode_join_ready']}`",
                "",
                markdown_table(readback_checks),
            ],
            overwrite=args.overwrite,
        )

    if args.mode in {"all", "lineage_gpcca_only"}:
        if env_payload and not env_payload.get("environment_check_passed", False):
            print("GPCCA environment is not ready; wrote audit report.")
            return 2
        if not readback_payload.get("readback_passed", False):
            print("Round 1 kernel readback failed; wrote readback report.")
            return 2
        matrix = sp.load_npz(round1_root / "kernel" / f"{args.kernel_name}.npz").tocsr().astype(float)
        main_run_frame, _, main_assignments, main_memberships, main_coarse_tables = run_gpcca_grid(
            matrix,
            metadata,
            k_grid,
            kernel_name=args.kernel_name,
            output_root=output_root,
            output_subdir="gpcca_lineage_directed",
            overwrite=args.overwrite,
        )
        atomic_write_tsv(output_root / "gpcca_lineage_directed" / "gpcca_lineage_directed_run_summary.tsv", main_run_frame, overwrite=args.overwrite)
        main_comparison = build_kernel_comparison_metrics(args.kernel_name, main_run_frame, main_assignments)
        all_comparisons.append(main_comparison)
        _write_phase_report(
            report_root,
            "02_LINEAGE_DIRECTED_GPCCA_SMOKE",
            "Lineage Directed GPCCA Smoke",
            {
                "generated_at_utc": utc_now(),
                "kernel_name": args.kernel_name,
                "n_macrostates": list(k_grid),
                "successful_k": [int(row.k) for row in main_run_frame.loc[main_run_frame["valid"].astype(bool)].itertuples(index=False)],
                "run_summary": main_run_frame.to_dict(orient="records"),
            },
            [
                f"- Kernel: `{args.kernel_name}`",
                f"- k values tested: `{list(k_grid)}`",
                "- Outputs are macrostate memberships and coarse transition diagnostics.",
                "- Scores are reported as macrostate membership or reachability-like quantities.",
                "",
                markdown_table(main_run_frame),
            ],
            overwrite=args.overwrite,
        )

    if args.mode in {"all", "controls_only"} and args.run_controls:
        if main_comparison.empty:
            if (output_root / "gpcca_lineage_directed" / "gpcca_lineage_directed_run_summary.tsv").exists():
                main_run_frame = pd.read_csv(output_root / "gpcca_lineage_directed" / "gpcca_lineage_directed_run_summary.tsv", sep="\t")
            for k in k_grid:
                assignment_path = output_root / "gpcca_lineage_directed" / f"gpcca_k{k}_assignment.tsv"
                if assignment_path.exists():
                    main_assignments[int(k)] = pd.read_csv(assignment_path, sep="\t")
            if not main_run_frame.empty:
                main_comparison = build_kernel_comparison_metrics(args.kernel_name, main_run_frame, main_assignments)
                all_comparisons.append(main_comparison)
        control_qc = pd.DataFrame(readback_payload.get("control_kernel_rows", []))
        control_rows: list[pd.DataFrame] = []
        for control_name in control_kernels:
            qc_row = control_qc.loc[control_qc["control_kernel"] == control_name]
            valid_control = bool(qc_row["valid"].iloc[0]) if not qc_row.empty else False
            if not valid_control:
                hold = pd.DataFrame(
                    [
                        {
                            "kernel_name": control_name,
                            "k": int(k),
                            "valid": False,
                            "n_macrostates": int(k),
                            "tiny_macrostate_count": None,
                            "section_nmi": None,
                            "section_ari": None,
                            "median_max_membership": None,
                            "ambiguous_state_fraction": None,
                            "phi_separation_score": None,
                            "barcode_entropy_separation_score": None,
                            "dominant_feature_fraction_separation_score": None,
                            "control_status": "CONTROL_KERNEL_HOLD_FOR_QC",
                        }
                        for k in k_grid
                    ]
                )
                control_rows.append(hold)
                continue
            control_matrix = sp.load_npz(round1_root / "controls" / f"{control_name}.npz").tocsr().astype(float)
            run_frame, _, assignments, _, _ = run_gpcca_grid(
                control_matrix,
                metadata,
                k_grid,
                kernel_name=control_name,
                output_root=output_root,
                output_subdir=f"control_gpcca/{control_name}",
                overwrite=args.overwrite,
            )
            atomic_write_tsv(output_root / "control_gpcca" / control_name / f"{control_name}_run_summary.tsv", run_frame, overwrite=args.overwrite)
            control_assignments[control_name] = assignments
            comparison = build_kernel_comparison_metrics(control_name, run_frame, assignments, reference_assignments=main_assignments)
            comparison["control_status"] = "PASS"
            control_rows.append(comparison)
        if control_rows:
            all_comparisons.extend(control_rows)
        comparison_frame = pd.concat(all_comparisons, ignore_index=True) if all_comparisons else pd.DataFrame()
        comparison_path = output_root / "control_gpcca" / "gpcca_kernel_comparison_metrics.tsv"
        atomic_write_tsv(comparison_path, comparison_frame, overwrite=args.overwrite)
        atomic_write_tsv(output_root / "control_gpcca" / "control_comparison_summary.tsv", comparison_frame, overwrite=args.overwrite)
        overlap_cols = [col for col in ["kernel_name", "k", "pairwise_ari", "pairwise_nmi", "same_assignment_fraction", "control_status"] if col in comparison_frame.columns]
        atomic_write_tsv(output_root / "control_gpcca" / "hard_assignment_overlap.tsv", comparison_frame[overlap_cols], overwrite=args.overwrite)
        section_cols = [col for col in ["kernel_name", "k", "section_nmi", "section_ari", "tiny_macrostate_count", "ambiguous_state_fraction"] if col in comparison_frame.columns]
        atomic_write_tsv(output_root / "control_gpcca" / "macrostate_section_purity_comparison.tsv", comparison_frame[section_cols], overwrite=args.overwrite)
        barcode_cols = [
            col
            for col in [
                "kernel_name",
                "k",
                "phi_separation_score",
                "barcode_entropy_separation_score",
                "dominant_feature_fraction_separation_score",
                "total_lineage_count_separation_score",
                "assay_balance_separation_score",
            ]
            if col in comparison_frame.columns
        ]
        atomic_write_tsv(output_root / "control_gpcca" / "macrostate_barcode_metric_comparison.tsv", comparison_frame[barcode_cols], overwrite=args.overwrite)
        _write_phase_report(
            report_root,
            "03_CONTROL_GPCCA_COMPARISON",
            "Control GPCCA Comparison",
            {
                "generated_at_utc": utc_now(),
                "control_kernels": list(control_kernels),
                "control_kernel_qc": readback_payload.get("control_kernel_rows", []),
                "comparison_path": str(comparison_path),
                "valid_control_rows": int(comparison_frame["valid"].fillna(False).astype(bool).sum()) if not comparison_frame.empty else 0,
                "controls_reduced": False,
            },
            [
                "- Control kernels were QC-gated before GPCCA.",
                "- Invalid control kernels are marked `CONTROL_KERNEL_HOLD_FOR_QC` and skipped.",
                f"- Comparison table: `{comparison_path}`",
                "",
                markdown_table(comparison_frame),
            ],
            overwrite=args.overwrite,
        )
    else:
        comparison_frame = pd.concat(all_comparisons, ignore_index=True) if all_comparisons else main_comparison

    if args.mode in {"all", "annotation_only", "figures_only", "validation_only"}:
        comparison_path = output_root / "control_gpcca" / "gpcca_kernel_comparison_metrics.tsv"
        if comparison_path.exists():
            comparison_frame = pd.read_csv(comparison_path, sep="\t")
        elif main_comparison.empty and (output_root / "gpcca_lineage_directed" / "gpcca_lineage_directed_run_summary.tsv").exists():
            main_run_frame = pd.read_csv(output_root / "gpcca_lineage_directed" / "gpcca_lineage_directed_run_summary.tsv", sep="\t")
            for k in k_grid:
                assignment_path = output_root / "gpcca_lineage_directed" / f"gpcca_k{k}_assignment.tsv"
                if assignment_path.exists():
                    main_assignments[int(k)] = pd.read_csv(assignment_path, sep="\t")
            comparison_frame = build_kernel_comparison_metrics(args.kernel_name, main_run_frame, main_assignments)
        main_only = comparison_frame.loc[comparison_frame["kernel_name"] == args.kernel_name].copy() if not comparison_frame.empty else main_comparison
        if args.selected_k == "auto":
            decision_label, selection_payload = select_technical_k(main_only)
            selected_k = selection_payload.get("selected_k")
        else:
            selected_k = int(args.selected_k)
            selected_row = main_only.loc[main_only["k"].astype(int) == selected_k].head(1)
            decision_label = f"L126_GPCCA_K{selected_k}_TECHNICAL_CANDIDATE" if not selected_row.empty and bool(selected_row["valid"].iloc[0]) else "L126_GPCCA_NO_TECHNICAL_CANDIDATE"
            selection_payload = {
                "decision_label": decision_label,
                "selected_k": selected_k if decision_label != "L126_GPCCA_NO_TECHNICAL_CANDIDATE" else None,
                "selected_metrics": selected_row.iloc[0].to_dict() if not selected_row.empty else {},
                "warnings": ["Manual selected-k override was used."],
            }
        candidate_label = f"L126_GPCCA_K{selected_k}_TECHNICAL_CANDIDATE" if selected_k is not None else "L126_GPCCA_NO_TECHNICAL_CANDIDATE"
        selection_payload["technical_candidate_label"] = candidate_label
        _write_phase_report(
            report_root,
            "04_TECHNICAL_K_SELECTION",
            "Technical K Selection",
            selection_payload,
            [
                f"- Candidate label: `{candidate_label}`",
                f"- Selected k: `{selected_k}`",
                "- Selection used tiny macrostate count, membership uncertainty, section association, and barcode metric separation.",
                "- This is a technical selection for hardening, not a biological model selection.",
                "",
                markdown_table(main_only),
            ],
            overwrite=args.overwrite,
        )

        if selected_k is not None:
            if selected_k not in main_assignments:
                assignment_path = output_root / "gpcca_lineage_directed" / f"gpcca_k{selected_k}_assignment.tsv"
                if assignment_path.exists():
                    main_assignments[selected_k] = pd.read_csv(assignment_path, sep="\t")
            selected_assignment = main_assignments[selected_k]
            state_annotation = _state_annotation_frame(
                selected_assignment,
                metadata,
                barcode_tables["unique"],
                barcode_tables["local"],
            )
            coarse = _read_coarse_table(output_root / "gpcca_lineage_directed" / f"gpcca_k{selected_k}_coarse_transition.tsv")
            macro_summary = _summary_for_macrostate(selected_assignment)
            macro_summary = _macrostate_labels(macro_summary, coarse)
            barcode_summary = _macrostate_barcode_summary(state_annotation)
            section_summary = _macrostate_section_summary(state_annotation)
            top_features = _macrostate_top_features(state_annotation, barcode_tables["unique_top"], barcode_tables["local_top"])
            macro_annotation = macro_summary.merge(barcode_summary, on=["macrostate", "n_states"], how="left")
            macro_annotation = macro_annotation.merge(
                section_summary.drop(columns=["section_distribution"], errors="ignore"),
                on="macrostate",
                how="left",
                suffixes=("", "_section"),
            )
            annotation_root = ensure_dir(output_root / "macrostate_annotation")
            atomic_write_tsv(annotation_root / "selected_k_macrostate_annotation.tsv", macro_annotation, overwrite=args.overwrite)
            atomic_write_tsv_gz(annotation_root / "selected_k_macrostate_top_features.tsv.gz", top_features, overwrite=args.overwrite)
            atomic_write_tsv(annotation_root / "selected_k_macrostate_barcode_summary.tsv", barcode_summary, overwrite=args.overwrite)
            atomic_write_tsv(annotation_root / "selected_k_macrostate_section_summary.tsv", section_summary, overwrite=args.overwrite)
            atomic_write_tsv_gz(annotation_root / "selected_k_state_annotation.tsv.gz", state_annotation, overwrite=args.overwrite)
            annotation_payload = {
                "generated_at_utc": utc_now(),
                "selected_k": int(selected_k),
                "macrostate_count": int(macro_annotation["macrostate"].nunique()),
                "state_count": int(len(state_annotation)),
                "barcode_summary_rows": int(len(barcode_summary)),
                "top_feature_rows": int(len(top_features)),
                "summary": _macrostate_summary_payload(state_annotation, macro_summary, section_summary, coarse),
                "outputs": {
                    "macrostate_annotation": str(annotation_root / "selected_k_macrostate_annotation.tsv"),
                    "top_features": str(annotation_root / "selected_k_macrostate_top_features.tsv.gz"),
                    "barcode_summary": str(annotation_root / "selected_k_macrostate_barcode_summary.tsv"),
                    "section_summary": str(annotation_root / "selected_k_macrostate_section_summary.tsv"),
                },
            }
            _write_phase_report(
                report_root,
                "05_MACROSTATE_BARCODE_ANNOTATION",
                "Macrostate Barcode Annotation",
                annotation_payload,
                [
                    f"- Selected k: `{selected_k}`",
                    "- Macrostate annotation includes section distribution, phi, barcode entropy, dominant feature fraction, total lineage count, and RA/TA/CA balance.",
                    "- Local-context and unique-cellbin barcode views are both retained.",
                    "- `technical sink-like candidate` is only assigned when coarse transition diagnostics support it and section artifact checks are not severe.",
                    "",
                    markdown_table(macro_annotation),
                ],
                overwrite=args.overwrite,
            )

    if args.mode in {"all", "figures_only"} and args.make_figures:
        if selected_k is None and (report_root / "04_TECHNICAL_K_SELECTION.json").exists():
            selected_payload = json.loads((report_root / "04_TECHNICAL_K_SELECTION.json").read_text(encoding="utf-8"))
            selected_k = selected_payload.get("selected_k")
        if selected_k is not None and selected_k not in main_assignments:
            for k in k_grid:
                assignment_path = output_root / "gpcca_lineage_directed" / f"gpcca_k{k}_assignment.tsv"
                if assignment_path.exists():
                    main_assignments[int(k)] = pd.read_csv(assignment_path, sep="\t")
        for control_name in control_kernels:
            control_assignments.setdefault(control_name, {})
            for k in k_grid:
                assignment_path = output_root / "control_gpcca" / control_name / f"gpcca_k{k}_assignment.tsv"
                if assignment_path.exists():
                    control_assignments[control_name][int(k)] = pd.read_csv(assignment_path, sep="\t")
        if state_annotation is None and selected_k is not None:
            state_path = output_root / "macrostate_annotation" / "selected_k_state_annotation.tsv.gz"
            if state_path.exists():
                state_annotation = pd.read_csv(state_path, sep="\t", compression="gzip")
        figure_payload = make_round2_figures(
            output_root,
            report_root,
            selected_k,
            main_assignments,
            control_assignments,
            comparison_frame if "comparison_frame" in locals() else pd.DataFrame(),
            state_annotation,
            overwrite=args.overwrite,
        )

    if args.mode in {"all", "validation_only"}:
        if not env_payload and (report_root / "00_ENVIRONMENT_AND_INPUT_AUDIT.json").exists():
            env_payload = json.loads((report_root / "00_ENVIRONMENT_AND_INPUT_AUDIT.json").read_text(encoding="utf-8"))
        if not selection_payload and (report_root / "04_TECHNICAL_K_SELECTION.json").exists():
            selection_payload = json.loads((report_root / "04_TECHNICAL_K_SELECTION.json").read_text(encoding="utf-8"))
            selected_k = selection_payload.get("selected_k")
        if not annotation_payload and (report_root / "05_MACROSTATE_BARCODE_ANNOTATION.json").exists():
            annotation_payload = json.loads((report_root / "05_MACROSTATE_BARCODE_ANNOTATION.json").read_text(encoding="utf-8"))
        if not figure_payload and (report_root / "06_FIGURES.json").exists():
            figure_payload = json.loads((report_root / "06_FIGURES.json").read_text(encoding="utf-8"))
        comparison_path = output_root / "control_gpcca" / "gpcca_kernel_comparison_metrics.tsv"
        comparison_for_decision = pd.read_csv(comparison_path, sep="\t") if comparison_path.exists() else pd.DataFrame()
        selected_metrics = selection_payload.get("selected_metrics", {})
        selected_section_dominated = bool(
            float(selected_metrics.get("section_nmi", 0.0) or 0.0) >= 0.75
            or float(selected_metrics.get("section_ari", 0.0) or 0.0) >= 0.75
            or float(selected_metrics.get("section_dominated_macrostate_fraction", 0.0) or 0.0) > 0.5
        )
        controls_valid = (
            not comparison_for_decision.empty
            and comparison_for_decision.loc[comparison_for_decision["kernel_name"] != args.kernel_name, "valid"].fillna(False).astype(bool).any()
        )
        warnings: list[str] = []
        if selected_section_dominated:
            warnings.append("Selected k has section dominance warning.")
        if selection_payload.get("warnings"):
            warnings.extend(str(item) for item in selection_payload.get("warnings", []))
        if not controls_valid:
            warnings.append("No valid control GPCCA rows were available.")
        if selected_k is None:
            final_label = "L126_PLANA_LINEAGE_GPCCA_NO_TECHNICAL_CANDIDATE"
        elif selected_section_dominated:
            final_label = "L126_PLANA_LINEAGE_GPCCA_READY_WITH_WARNINGS"
        elif selection_payload.get("warnings"):
            final_label = "L126_PLANA_LINEAGE_GPCCA_READY_WITH_WARNINGS"
        elif not controls_valid:
            final_label = "L126_PLANA_LINEAGE_GPCCA_READY_WITH_WARNINGS"
        else:
            final_label = "L126_PLANA_LINEAGE_GPCCA_SMOKE_READY"
        decision_payload = {
            "generated_at_utc": utc_now(),
            "decision_label": final_label,
            "pygpcca_environment": env_payload.get("selected_environment"),
            "k_values_tested": list(k_grid),
            "selected_k": selected_k,
            "controls_ran": bool(controls_valid),
            "warnings": warnings,
            "selection_payload": selection_payload,
            "annotation_payload": annotation_payload,
            "figure_payload": figure_payload,
            "next_safe_command": (
                "conda run --no-capture-output -n nichefate-gpcca python "
                "scripts/planC_l126_planA_lineage_gpcca_round2.py "
                "--mode annotation_only --selected-k auto --overwrite"
            ),
        }
        _write_phase_report(
            report_root,
            "07_GPCCA_READINESS_DECISION",
            "GPCCA Readiness Decision",
            decision_payload,
            [
                f"- Final decision label: `{final_label}`",
                f"- pyGPCCA environment: `{env_payload.get('selected_environment')}`",
                f"- k values tested: `{list(k_grid)}`",
                f"- Selected technical k: `{selected_k}`",
                f"- Controls ran: `{bool(controls_valid)}`",
                f"- Warnings: `{warnings}`",
                "- Direction is already encoded in the Round 1 lineage-informed directed kernel; section order is not used.",
                "- Outputs are macrostate decompositions and reachability-like abstractions only.",
                f"- Next safe command: `{decision_payload['next_safe_command']}`",
            ],
            overwrite=args.overwrite,
        )

        report_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(report_root.glob("*.md")))
        source_after = snapshot_files([args.input_packet_root.expanduser().resolve()])
        round1_after = snapshot_files(_round1_snapshot_paths(round1_root, args.kernel_name, control_kernels))
        validation_payload = validation_payload_for_round2(
            report_root,
            output_root,
            source_before,
            source_after,
            round1_before,
            round1_after,
            report_text,
            final_label,
            {
                "env": env_payload,
                "selection": selection_payload,
                "annotation": annotation_payload,
                "figures": figure_payload,
            },
        )
        if forbidden_claim_hits(report_text):
            validation_payload["status"] = "FAIL"
        atomic_write_json(report_root / "08_VALIDATION.json", validation_payload, overwrite=args.overwrite)
        atomic_write_text(
            report_root / "08_VALIDATION.md",
            "\n".join(
                [
                    "# Validation",
                    "",
                    f"- Decision label: `{final_label}`",
                    f"- Validation status: `{validation_payload['status']}`",
                    f"- Checks passed: {sum(bool(row['status']) for row in validation_payload['checks'])}/{len(validation_payload['checks'])}",
                    "",
                    markdown_table(pd.DataFrame(validation_payload["checks"])),
                    "",
                ]
            ),
            overwrite=args.overwrite,
        )
        print(f"decision_label={final_label}")
        print(f"selected_k={selected_k}")
        print(f"validation_status={validation_payload['status']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
