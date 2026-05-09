#!/usr/bin/env python
"""Review Markov/GPCCA feasibility after M4A assembly without running GPCCA."""

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
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from nichefate.io import load_config
from m4b_01_design_terminal_macrostates import (  # noqa: E402
    NO_DOWNSTREAM_FLAGS,
    ROUTE_COMPATIBILITY_NOTE,
    STRUCTURAL_DIAGNOSTIC_NOTE,
    assert_no_ssd_path,
    atomic_write_json,
    atomic_write_text,
    configured_paths,
    infer_final_time,
    load_json,
)


DEFAULT_CONFIG = "configs/m4b_markov_terminal_design.yaml"


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
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def load_m4a_schema(paths: dict[str, Path]) -> dict[str, Any]:
    schema_path = paths["m4a_root"] / "reports" / "m4a_transition_object_schema.json"
    return load_json(schema_path)


def validate_sparse_objects(paths: dict[str, Path], schema: dict[str, Any]) -> tuple[dict[str, sp.csr_matrix], dict[str, Any]]:
    matrices = {
        "p_forward": sp.load_npz(paths["p_forward"]).tocsr(),
        "p_absorbing": sp.load_npz(paths["p_absorbing"]).tocsr(),
        "w_raw": sp.load_npz(paths["w_raw"]).tocsr(),
        "w_mass_adjusted": sp.load_npz(paths["w_mass_adjusted"]).tocsr(),
    }
    expected_shape = tuple(int(x) for x in schema["matrix_qc"]["shape"])
    expected_nnz = {
        "p_forward": int(schema["matrix_qc"]["p_forward_nnz"]),
        "p_absorbing": int(schema["matrix_qc"]["p_absorbing_nnz"]),
        "w_raw": int(schema["matrix_qc"]["w_raw_nnz"]),
        "w_mass_adjusted": int(schema["matrix_qc"]["w_mass_adjusted_nnz"]),
    }
    qc: dict[str, Any] = {}
    for name, matrix in matrices.items():
        if matrix.shape != expected_shape:
            raise ValueError(f"{name} shape {matrix.shape} != expected {expected_shape}.")
        if int(matrix.nnz) != expected_nnz[name]:
            raise ValueError(f"{name} nnz {matrix.nnz} != expected {expected_nnz[name]}.")
        qc[f"{name}_shape"] = list(matrix.shape)
        qc[f"{name}_nnz"] = int(matrix.nnz)
    return matrices, qc


def absorbing_structure_qc(
    p_forward: sp.csr_matrix,
    p_absorbing: sp.csr_matrix,
    node_table: pd.DataFrame,
) -> dict[str, Any]:
    final_nodes = node_table.loc[node_table["is_final_time"], "global_node_index"].to_numpy(dtype=np.int64)
    non_final_nodes = node_table.loc[~node_table["is_final_time"], "global_node_index"].to_numpy(dtype=np.int64)
    forward_out_degree = np.diff(p_forward.indptr)
    absorbing_diag = p_absorbing.diagonal()
    final_diag = absorbing_diag[final_nodes]
    non_final_diag = absorbing_diag[non_final_nodes]
    qc = {
        "final_nodes": int(len(final_nodes)),
        "forward_final_out_degree_max": int(forward_out_degree[final_nodes].max()) if len(final_nodes) else 0,
        "absorbing_final_diag_min": float(final_diag.min()) if len(final_diag) else 0.0,
        "absorbing_final_diag_max": float(final_diag.max()) if len(final_diag) else 0.0,
        "non_final_absorbing_diag_nonzero": int((non_final_diag != 0).sum()),
    }
    if qc["forward_final_out_degree_max"] != 0:
        raise ValueError("Forward matrix has outgoing edges from final-time nodes.")
    if not np.allclose(final_diag, 1.0):
        raise ValueError("Absorbing matrix final-time diagonal is not exactly structural self-loop weight 1.")
    return qc


def sparse_memory_estimate(schema: dict[str, Any]) -> dict[str, Any]:
    n_nodes = int(schema["node_count"])
    nnz = int(schema["matrix_qc"]["p_absorbing_nnz"])
    csr_bytes_float32 = nnz * 4 + nnz * 4 + (n_nodes + 1) * 4
    dense_float64_bytes = n_nodes * n_nodes * 8
    eigvec_64_float64_bytes = n_nodes * 64 * 8
    return {
        "n_nodes": n_nodes,
        "p_absorbing_nnz": nnz,
        "estimated_single_csr_float32_gib": csr_bytes_float32 / (1024**3),
        "estimated_dense_float64_gib": dense_float64_bytes / (1024**3),
        "estimated_64_eigenvectors_float64_gib": eigvec_64_float64_bytes / (1024**3),
        "full_gpcca_risk": "high_for_immediate_full_matrix_review",
    }


def connected_component_review(matrix: sp.csr_matrix, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"status": "disabled"}
    try:
        n_components, labels = connected_components(matrix, directed=True, connection="weak", return_labels=True)
        counts = np.bincount(labels)
        return {
            "status": "completed",
            "weak_components": int(n_components),
            "largest_component_nodes": int(counts.max()) if len(counts) else 0,
            "smallest_component_nodes": int(counts.min()) if len(counts) else 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "warning_only_failed", "reason": str(exc)}


def feasibility_report(
    schema: dict[str, Any],
    final_time: str,
    final_time_day: float,
    matrix_qc: dict[str, Any],
    absorbing_qc: dict[str, Any],
    memory: dict[str, Any],
    components: dict[str, Any],
    terminal_design: dict[str, Any] | None,
    runtime_seconds: float,
) -> str:
    selected_k = terminal_design.get("selected_n_macrostates") if terminal_design else "unavailable"
    lines = [
        "# M4B Markov/GPCCA Feasibility Review",
        "",
        "This stage reviews feasibility only. It does not run GPCCA, compute fate probabilities, compute absorption probabilities, train Branched NicheFlow / BranchSBM, run M5, or run regulator analysis.",
        "",
        "## M4A Object Checks",
        f"- global nodes: {schema['node_count']}",
        f"- final time inferred from max time_day: {final_time} ({final_time_day:g})",
        f"- P_forward nnz: {matrix_qc['p_forward_nnz']}",
        f"- P_absorbing nnz: {matrix_qc['p_absorbing_nnz']}",
        f"- final-time nodes: {absorbing_qc['final_nodes']}",
        f"- forward final-time out-degree max: {absorbing_qc['forward_final_out_degree_max']}",
        f"- absorbing final diagonal min/max: {absorbing_qc['absorbing_final_diag_min']:.6g} / {absorbing_qc['absorbing_final_diag_max']:.6g}",
        "",
        "## Existing M4A Row-Sum QC",
        f"- forward non-final max error: {schema['row_sum_qc']['forward_nonfinal_row_sum_error']['max']:.6g}",
        f"- forward non-final p99 error: {schema['row_sum_qc']['forward_nonfinal_row_sum_error']['p99']:.6g}",
        f"- absorbing max error: {schema['row_sum_qc']['absorbing_all_row_sum_error']['max']:.6g}",
        "",
        "## Full GPCCA Risk Estimate",
        f"- estimated dense float64 matrix GiB: {memory['estimated_dense_float64_gib']:.3g}",
        f"- estimated 64 dense eigenvectors GiB: {memory['estimated_64_eigenvectors_float64_gib']:.3g}",
        f"- risk conclusion: {memory['full_gpcca_risk']}",
        "",
        "## Connected Components",
        f"- status: {components['status']}",
    ]
    if components.get("status") == "completed":
        lines.extend(
            [
                f"- weak components: {components['weak_components']}",
                f"- largest component nodes: {components['largest_component_nodes']}",
            ]
        )
    elif "reason" in components:
        lines.append(f"- warning: {components['reason']}")
    lines.extend(
        [
            "",
            "## Recommended Markov Route",
            f"- terminal macrostate design is available with selected K={selected_k}",
            "- next stage should compute M4C fate probabilities by time-layered backward propagation to terminal macrostate labels",
            "- optional GPCCA/coarse-grained macrostate review should be deferred until after M4C design review",
            "",
            "## Route Compatibility",
            f"- {ROUTE_COMPATIBILITY_NOTE}",
            f"- {STRUCTURAL_DIAGNOSTIC_NOTE}",
            "",
            "## Runtime",
            f"- feasibility review runtime seconds: {runtime_seconds:.3f}",
            "",
            "## Not Run",
            "- GPCCA was not run.",
            "- Fate probability was not computed.",
            "- Absorption probability was not computed.",
            "- Branched NicheFlow / BranchSBM was not trained.",
            "- M5 and regulator analysis were not run.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def make_dashboard(figures_dir: Path, schema: dict[str, Any], memory: dict[str, Any], components: dict[str, Any], warning_only: bool) -> list[str]:
    warnings: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].bar(["nodes", "edges"], [schema["node_count"], schema["matrix_qc"]["p_forward_nnz"]])
        axes[0, 0].set_title("M4A object scale")
        axes[0, 1].bar(
            ["dense", "eigvec64"],
            [memory["estimated_dense_float64_gib"], memory["estimated_64_eigenvectors_float64_gib"]],
        )
        axes[0, 1].set_title("Memory risk GiB")
        axes[1, 0].bar(
            ["forward", "absorbing"],
            [
                schema["row_sum_qc"]["forward_nonfinal_row_sum_error"]["max"],
                schema["row_sum_qc"]["absorbing_all_row_sum_error"]["max"],
            ],
        )
        axes[1, 0].set_title("Max row-sum error")
        axes[1, 1].bar(
            ["components"],
            [components.get("weak_components", 0) if components.get("status") == "completed" else 0],
        )
        axes[1, 1].set_title("Weak components")
        fig.tight_layout()
        fig.savefig(figures_dir / "m4b_markov_feasibility_dashboard.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        if not warning_only:
            raise
        warnings.append(f"Feasibility dashboard generation failed after checks passed: {exc}")
    return warnings


def run(args: argparse.Namespace) -> int:
    start = time.monotonic()
    config = load_config(args.config)
    paths = configured_paths(config)
    schema = load_m4a_schema(paths)
    node_table = pd.read_parquet(paths["node_table"], columns=["global_node_index", "time", "time_day", "is_final_time"])
    final_time_day, final_time = infer_final_time(node_table)
    matrices, matrix_qc = validate_sparse_objects(paths, schema)
    absorbing_qc = absorbing_structure_qc(matrices["p_forward"], matrices["p_absorbing"], node_table)
    memory = sparse_memory_estimate(schema)
    components = connected_component_review(
        matrices["p_forward"],
        bool(config["markov_feasibility"].get("check_connected_components", True)),
    )
    terminal_summary_path = paths["reports_dir"] / "m4b_terminal_macrostate_design_summary.json"
    terminal_design = load_json(terminal_summary_path) if terminal_summary_path.exists() else None
    figure_warnings = make_dashboard(
        paths["figures_dir"],
        schema,
        memory,
        components,
        bool(config["visualization"].get("figure_failure_is_warning", True)),
    )
    runtime = time.monotonic() - start
    report_path = paths["reports_dir"] / "m4b_markov_gpcca_feasibility_report.md"
    summary_path = paths["reports_dir"] / "m4b_markov_gpcca_feasibility_summary.json"
    atomic_write_text(
        report_path,
        feasibility_report(
            schema,
            final_time,
            final_time_day,
            matrix_qc,
            absorbing_qc,
            memory,
            components,
            terminal_design,
            runtime,
        ),
    )
    atomic_write_json(
        summary_path,
        {
            "schema_version": "m4b_markov_gpcca_feasibility_summary_v1",
            "generated_at_utc": utc_now_iso(),
            "final_time": final_time,
            "final_time_day": final_time_day,
            "matrix_qc": matrix_qc,
            "absorbing_structure_qc": absorbing_qc,
            "m4a_row_sum_qc": schema["row_sum_qc"],
            "memory_risk": memory,
            "connected_components": components,
            "terminal_design_selected_k": terminal_design.get("selected_n_macrostates") if terminal_design else None,
            "figure_warnings": figure_warnings,
            "recommendation": "M4C time-layered backward propagation before any optional full GPCCA review.",
            "route_compatibility_note": ROUTE_COMPATIBILITY_NOTE,
            **NO_DOWNSTREAM_FLAGS,
        },
    )
    print("M4B_02_MARKOV_GPCCA_FEASIBILITY_REVIEW_COMPLETE")
    print(f"GLOBAL_NODES {schema['node_count']}")
    print(f"FINAL_TIME {final_time}")
    print(f"FINAL_TIME_DAY {final_time_day:g}")
    print(f"P_FORWARD_NNZ {matrix_qc['p_forward_nnz']}")
    print(f"P_ABSORBING_NNZ {matrix_qc['p_absorbing_nnz']}")
    print(f"CONNECTED_COMPONENT_STATUS {components['status']}")
    if components.get("status") == "completed":
        print(f"WEAK_COMPONENTS {components['weak_components']}")
    print(f"REPORT {report_path}")
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
