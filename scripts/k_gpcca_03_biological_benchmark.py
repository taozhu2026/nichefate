#!/usr/bin/env python
"""K_gpcca-03 pilot biological benchmark and fate-feasibility review.

This script benchmarks existing K_gpcca-02 pilot artifacts. It does not
reconstruct K matrices, does not run full-resolution pyGPCCA, and does not use
custom GPCCA-like terminal/fate fallbacks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import k_gpcca_01_pilot_kernel_preflight as k01


ROOT = Path("/home/zhutao/scratch/nichefate")
DEFAULT_OUTPUT_ROOT = ROOT / "k_gpcca_pilot_benchmark"
K_PILOT_ROOT = ROOT / "k_gpcca_pilot"
TMPDIR = Path("/home/zhutao/tmp/k_gpcca")
PRIMARY_CANDIDATE = "pilot_v1_balanced"
SECONDARY_CANDIDATE = "pilot_v2_balanced"
DEFAULT_K_VALUES = [4, 6, 8, 10]
PYGPCCA_ENV = "nichefate-gpcca"

PROTECTED_ROOTS = [
    ROOT / "m3",
    ROOT / "m3_v2",
    ROOT / "m4a",
    ROOT / "m4a_v2",
    ROOT / "m4b",
    ROOT / "m4c",
    ROOT / "m4c_v2",
    ROOT / "planA_freeze",
    K_PILOT_ROOT,
]
FORBIDDEN_ROOTS = [
    ROOT / "m5",
    ROOT / "branchsbm",
    ROOT / "barcode",
    ROOT / "darlin",
    ROOT / "m4d",
    ROOT / "gpcca",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--k-values", default="4,6,8,10")
    parser.add_argument("--skip-v2", action="store_true")
    parser.add_argument("--timeout-seconds-per-k", type=int, default=3600)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def validate_output_root(output_root: Path) -> Path:
    output_root = resolved(output_root)
    k01.reject_ssd(output_root)
    for protected in PROTECTED_ROOTS:
        if k01.paths_overlap(output_root, protected):
            raise ValueError(f"Output root overlaps protected root {protected}: {output_root}")
    for forbidden in FORBIDDEN_ROOTS:
        if k01.paths_overlap(output_root, forbidden):
            raise ValueError(f"Output root overlaps forbidden root {forbidden}: {output_root}")
    return output_root


def output_paths(output_root: Path) -> dict[str, Path]:
    root = validate_output_root(output_root)
    return {
        "root": root,
        "reports": root / "reports",
        "figures": root / "reports" / "figures",
        "gpcca": root / "gpcca",
    }


def ensure_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    k01.reject_ssd(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(k01.json_safe(payload), indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    k01.reject_ssd(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.head(max_rows).copy()
    cols = [str(col) for col in shown.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for record in shown.astype(str).to_dict(orient="records"):
        lines.append("| " + " | ".join(record[col].replace("|", "\\|") for col in cols) + " |")
    if len(frame) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(frame)} rows.")
    return "\n".join(lines)


def input_paths() -> dict[str, Path]:
    return {
        "v1_kernel": K_PILOT_ROOT / "kernels" / "K_gpcca_pilot_pilot_v1_balanced.npz",
        "v1_node_table": K_PILOT_ROOT / "kernels" / "K_gpcca_pilot_pilot_v1_balanced_node_table.parquet",
        "v2_kernel": K_PILOT_ROOT / "kernels" / "K_gpcca_pilot_pilot_v2_balanced.npz",
        "v2_node_table": K_PILOT_ROOT / "kernels" / "K_gpcca_pilot_pilot_v2_balanced_node_table.parquet",
        "v1_k8_memberships": K_PILOT_ROOT / "gpcca" / "k_gpcca_pilot_gpcca_memberships_pilot_v1_balanced.parquet",
        "v1_k8_macrostates": K_PILOT_ROOT / "gpcca" / "k_gpcca_pilot_gpcca_macrostates_pilot_v1_balanced.csv",
        "v1_k8_coarse": K_PILOT_ROOT / "gpcca" / "k_gpcca_pilot_gpcca_coarse_transition_pilot_v1_balanced.csv",
        "k02_summary": K_PILOT_ROOT / "k_gpcca_02_summary.json",
        "k02_annotation": K_PILOT_ROOT / "reports" / "k_gpcca_02_macrostate_annotation_pilot_v1_balanced.csv",
    }


def validate_inputs(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, path in input_paths().items():
        rows.append(
            {
                "input_name": name,
                "path": str(path),
                "exists": path.exists(),
                "bytes": int(path.stat().st_size) if path.exists() else 0,
                "status": "PASS" if path.exists() and path.stat().st_size > 0 else "FAIL",
            }
        )
    rows.append(
        {
            "input_name": "custom_fallback_outputs",
            "path": str(K_PILOT_ROOT),
            "exists": False,
            "bytes": 0,
            "status": "PASS"
            if not any("fallback" in p.name.lower() for p in K_PILOT_ROOT.rglob("*"))
            else "FAIL",
        }
    )
    rows.append(
        {
            "input_name": "ssd_outputs",
            "path": str(K_PILOT_ROOT),
            "exists": False,
            "bytes": 0,
            "status": "PASS" if k01.count_ssd_outputs(K_PILOT_ROOT) == 0 else "FAIL",
        }
    )
    return pd.DataFrame(rows)


def parse_k_values(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("No k values provided")
    return values


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return bool(value)


def sensitivity_path(paths: dict[str, Path]) -> Path:
    return paths["root"] / "k_gpcca_03_macrostate_sensitivity_by_k.csv"


def load_sensitivity(paths: dict[str, Path]) -> pd.DataFrame:
    path = sensitivity_path(paths)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def upsert_sensitivity(paths: dict[str, Path], row: dict[str, Any]) -> None:
    existing = load_sensitivity(paths)
    if not existing.empty:
        existing = existing[
            ~(
                (existing["candidate_id"] == row["candidate_id"])
                & (existing["k"] == row["k"])
            )
        ]
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated = updated.sort_values(["candidate_id", "k"]).reset_index(drop=True)
    atomic_write_csv(sensitivity_path(paths), updated)


def copy_existing_k8(paths: dict[str, Path]) -> dict[str, Any]:
    src_macro = input_paths()["v1_k8_macrostates"]
    src_memberships = input_paths()["v1_k8_memberships"]
    src_coarse = input_paths()["v1_k8_coarse"]
    macro = pd.read_csv(src_macro)
    dest_macro = paths["gpcca"] / "k_gpcca03_pilot_v1_balanced_k8_macrostates.csv"
    dest_memberships = paths["gpcca"] / "k_gpcca03_pilot_v1_balanced_k8_memberships.parquet"
    dest_coarse = paths["gpcca"] / "k_gpcca03_pilot_v1_balanced_k8_coarse_transition.csv"
    macro.to_csv(dest_macro, index=False)
    pd.read_parquet(src_memberships).to_parquet(dest_memberships, index=False)
    pd.read_csv(src_coarse).to_csv(dest_coarse, index=False)
    return summarize_macrostate_result(
        macro,
        PRIMARY_CANDIDATE,
        8,
        True,
        "",
        "reused existing K02 k=8 output",
        0.0,
        dest_macro,
        dest_memberships,
        dest_coarse,
    )


def run_standard_pygpcca_for_k(
    paths: dict[str, Path],
    candidate_id: str,
    k_value: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    matrix_path = K_PILOT_ROOT / "kernels" / f"K_gpcca_pilot_{candidate_id}.npz"
    node_table_path = K_PILOT_ROOT / "kernels" / f"K_gpcca_pilot_{candidate_id}_node_table.parquet"
    macro_path = paths["gpcca"] / f"k_gpcca03_{candidate_id}_k{k_value}_macrostates.csv"
    memberships_path = paths["gpcca"] / f"k_gpcca03_{candidate_id}_k{k_value}_memberships.parquet"
    coarse_path = paths["gpcca"] / f"k_gpcca03_{candidate_id}_k{k_value}_coarse_transition.csv"
    TMPDIR.mkdir(parents=True, exist_ok=True)
    code = r"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pygpcca
from scipy import sparse

payload = json.loads(sys.argv[1])
start = time.perf_counter()
matrix = sparse.load_npz(payload["matrix_path"]).tocsr().astype(float)
row_sums = np.asarray(matrix.sum(axis=1)).ravel()
inv = np.zeros_like(row_sums, dtype=float)
mask = row_sums > 0
inv[mask] = 1.0 / row_sums[mask]
matrix = sparse.diags(inv).dot(matrix).tocsr()
node_table = pd.read_parquet(payload["node_table_path"])
result = {
    "candidate_id": payload["candidate_id"],
    "k": int(payload["k"]),
    "success": False,
    "error": "",
    "runtime_seconds": None,
    "macro_path": payload["macro_path"],
    "memberships_path": payload["memberships_path"],
    "coarse_path": payload["coarse_path"],
}
try:
    gpcca = pygpcca.GPCCA(matrix, z="LM", method="krylov")
    gpcca.optimize(int(payload["k"]))
    memberships = np.real_if_close(np.asarray(gpcca.memberships)).astype(float)
    memberships[memberships < 0] = 0.0
    rs = memberships.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    memberships = memberships / rs
    assignments = np.asarray(gpcca.macrostate_assignment).reshape(-1).astype(int)
    coarse = np.real_if_close(np.asarray(gpcca.coarse_grained_transition_matrix)).astype(float)
    entropy = -np.sum(np.where(memberships > 0, memberships * np.log(memberships), 0.0), axis=1)
    max_membership = memberships.max(axis=1)
    mdf = pd.DataFrame(memberships, columns=[f"membership_{i}" for i in range(memberships.shape[1])])
    mdf.insert(0, "global_node_index", node_table["global_node_index"].to_numpy())
    mdf.insert(1, "local_index", np.arange(len(mdf)))
    mdf.to_parquet(payload["memberships_path"], index=False)
    macro = node_table.copy()
    macro["macrostate"] = assignments
    macro["membership_entropy"] = entropy
    macro["max_membership"] = max_membership
    macro.to_csv(payload["macro_path"], index=False)
    pd.DataFrame(coarse).to_csv(payload["coarse_path"], index=False)
    result["success"] = True
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
result["runtime_seconds"] = time.perf_counter() - start
print(json.dumps(result))
"""
    payload = {
        "matrix_path": str(matrix_path),
        "node_table_path": str(node_table_path),
        "candidate_id": candidate_id,
        "k": int(k_value),
        "macro_path": str(macro_path),
        "memberships_path": str(memberships_path),
        "coarse_path": str(coarse_path),
    }
    env = os.environ.copy()
    for key in [
        "TMPDIR",
        "TMP",
        "TEMP",
        "OMPI_MCA_orte_tmpdir_base",
        "OMPI_MCA_prte_tmpdir_base",
        "PRTE_MCA_prte_tmpdir_base",
        "PMIX_MCA_pmix_tmpdir_base",
    ]:
        env[key] = str(TMPDIR)
    try:
        result = subprocess.run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                PYGPCCA_ENV,
                "python",
                "-c",
                code,
                json.dumps(payload),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        row = {
            "candidate_id": candidate_id,
            "k": int(k_value),
            "success": False,
            "error": f"TimeoutExpired: exceeded {timeout_seconds} seconds",
            "runtime_seconds": timeout_seconds,
            "macro_path": str(macro_path),
            "memberships_path": str(memberships_path),
            "coarse_path": str(coarse_path),
        }
        atomic_write_text(
            paths["reports"] / f"k_gpcca_03_{candidate_id}_k{k_value}_failure.md",
            f"# pyGPCCA k={k_value} Timeout\n\n{exc}\n",
        )
        return row
    if result.returncode != 0:
        return {
            "candidate_id": candidate_id,
            "k": int(k_value),
            "success": False,
            "error": f"SubprocessError: {result.stderr or result.stdout}",
            "runtime_seconds": np.nan,
            "macro_path": str(macro_path),
            "memberships_path": str(memberships_path),
            "coarse_path": str(coarse_path),
        }
    parsed = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    if parsed.get("success"):
        macro = pd.read_csv(macro_path)
        return summarize_macrostate_result(
            macro,
            candidate_id,
            k_value,
            True,
            "",
            "standard pyGPCCA",
            float(parsed.get("runtime_seconds", np.nan)),
            macro_path,
            memberships_path,
            coarse_path,
        )
    return {
        **parsed,
        "macrostate_count": 0,
        "macrostate_size_min": np.nan,
        "macrostate_size_max": np.nan,
        "largest_macrostate_fraction": np.nan,
        "membership_entropy_mean": np.nan,
        "membership_entropy_median": np.nan,
        "max_membership_mean": np.nan,
        "max_membership_median": np.nan,
        "source": "standard pyGPCCA",
    }


def summarize_macrostate_result(
    macro: pd.DataFrame,
    candidate_id: str,
    k_value: int,
    success: bool,
    error: str,
    source: str,
    runtime_seconds: float,
    macro_path: Path,
    memberships_path: Path,
    coarse_path: Path,
) -> dict[str, Any]:
    sizes = macro["macrostate"].value_counts().sort_index()
    return {
        "candidate_id": candidate_id,
        "k": int(k_value),
        "success": bool(success),
        "error": error,
        "source": source,
        "runtime_seconds": float(runtime_seconds),
        "macrostate_count": int(sizes.size),
        "macrostate_size_min": int(sizes.min()),
        "macrostate_size_max": int(sizes.max()),
        "largest_macrostate_fraction": float(sizes.max() / len(macro)),
        "smallest_macrostate_fraction": float(sizes.min() / len(macro)),
        "membership_entropy_mean": float(macro["membership_entropy"].mean()),
        "membership_entropy_median": float(macro["membership_entropy"].median()),
        "max_membership_mean": float(macro["max_membership"].mean()),
        "max_membership_median": float(macro["max_membership"].median()),
        "macro_path": str(macro_path),
        "memberships_path": str(memberships_path),
        "coarse_path": str(coarse_path),
    }


def run_k_sensitivity(paths: dict[str, Path], k_values: list[int], timeout_seconds: int) -> pd.DataFrame:
    for k_value in k_values:
        existing = load_sensitivity(paths)
        if (
            not existing.empty
            and ((existing["candidate_id"] == PRIMARY_CANDIDATE) & (existing["k"] == k_value)).any()
            and as_bool(existing[(existing["candidate_id"] == PRIMARY_CANDIDATE) & (existing["k"] == k_value)].iloc[0]["success"])
        ):
            continue
        if k_value == 8 and input_paths()["v1_k8_macrostates"].exists():
            row = copy_existing_k8(paths)
        else:
            row = run_standard_pygpcca_for_k(paths, PRIMARY_CANDIDATE, k_value, timeout_seconds)
        upsert_sensitivity(paths, row)
    return load_sensitivity(paths)


def maybe_run_v2(paths: dict[str, Path], skip_v2: bool, timeout_seconds: int) -> pd.DataFrame:
    sensitivity = load_sensitivity(paths)
    v1 = sensitivity[sensitivity["candidate_id"] == PRIMARY_CANDIDATE]
    if skip_v2:
        write_v2_report(paths, "not run: --skip-v2 was set", pd.DataFrame())
        return pd.DataFrame()
    if v1.empty or not v1["success"].map(as_bool).any():
        write_v2_report(paths, "not run: no successful v1 k-sensitivity result", pd.DataFrame())
        return pd.DataFrame()
    row = run_standard_pygpcca_for_k(paths, SECONDARY_CANDIDATE, 8, timeout_seconds)
    comparison = pd.DataFrame([row])
    atomic_write_csv(paths["root"] / "k_gpcca_03_v1_v2_gpcca_pilot_comparison.csv", comparison)
    write_v2_report(paths, "run: v1 sensitivity completed and at least one k succeeded", comparison)
    return comparison


def write_v2_report(paths: dict[str, Path], reason: str, comparison: pd.DataFrame) -> None:
    body = "# K_gpcca-03 v1/v2 GPCCA Pilot Comparison\n\n"
    body += f"Status: {reason}\n\n"
    if not comparison.empty:
        body += markdown_table(comparison)
    atomic_write_text(paths["reports"] / "k_gpcca_03_v1_v2_gpcca_pilot_comparison_report.md", body + "\n")


def annotation_metadata() -> pd.DataFrame:
    endpoint_path = ROOT / "m4e" / "neighborhood_annotation" / "node_neighborhood_annotation.parquet"
    columns = [
        "global_node_index",
        "leiden_neigh",
        "cadinu_neighborhood_label",
        "cell_type_l1",
        "cell_type_l3",
        "candidate_endpoint_label",
        "endpoint_biological_label",
        "endpoint_phenotype_class",
        "biological_confidence_tier",
    ]
    available = k01.parquet_metadata(endpoint_path)[1]
    meta = pd.read_parquet(endpoint_path, columns=[col for col in columns if col in available])
    v1_path = ROOT / "m4c" / "fate_probabilities" / "fate_probability_node_summary.parquet"
    if v1_path.exists():
        v1 = pd.read_parquet(v1_path, columns=["global_node_index", "dominant_fate_label", "dominant_fate_probability"])
        meta = meta.merge(v1.rename(columns={"dominant_fate_label": "p_fate_v1_dominant", "dominant_fate_probability": "p_fate_v1_probability"}), on="global_node_index", how="left")
    v2_path = ROOT / "m4c_v2" / "fate_probabilities" / "node_fate_summary_v2.parquet"
    if v2_path.exists():
        v2 = pd.read_parquet(v2_path, columns=["global_node_index", "dominant_endpoint_label", "dominant_refined_endpoint_label", "dominant_endpoint_probability"])
        meta = meta.merge(
            v2.rename(
                columns={
                    "dominant_endpoint_label": "p_fate_v2_dominant",
                    "dominant_refined_endpoint_label": "p_fate_v2_refined_dominant",
                    "dominant_endpoint_probability": "p_fate_v2_probability",
                }
            ),
            on="global_node_index",
            how="left",
        )
    return meta


def entropy_from_counts(counts: pd.Series) -> float:
    values = counts.to_numpy(dtype=float)
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values / total
    return float(-np.sum(np.where(probs > 0, probs * np.log(probs), 0.0)))


def summarize_annotation_for_macro(macro: pd.DataFrame, k_value: int, candidate_id: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    meta = annotation_metadata()
    data = macro.merge(meta, on="global_node_index", how="left", suffixes=("", "_m4e"))
    group_map = {
        "time": "time",
        "neighborhood": "leiden_neigh",
        "cell_type": "cell_type_l3",
        "endpoint": "endpoint_biological_label",
        "slice": "slice_id",
        "mouse": "mouse_id",
        "p_fate_v1": "p_fate_v1_dominant",
        "p_fate_v2": "p_fate_v2_dominant",
    }
    tables: dict[str, pd.DataFrame] = {}
    summary_rows = []
    for group_name, column in group_map.items():
        if column not in data.columns:
            continue
        counts = (
            data.groupby(["macrostate", column], dropna=False)
            .size()
            .reset_index(name="node_count")
            .rename(columns={column: "label"})
        )
        total_by_macro = data.groupby("macrostate").size().rename("macrostate_total").reset_index()
        background = data[column].fillna("NA").value_counts(normalize=True).to_dict()
        counts["label"] = counts["label"].fillna("NA").astype(str)
        counts = counts.merge(total_by_macro, on="macrostate", how="left")
        counts["fraction_within_macrostate"] = counts["node_count"] / counts["macrostate_total"]
        counts["background_fraction"] = counts["label"].map(lambda label: float(background.get(label, 0.0)))
        counts["enrichment_vs_background"] = counts["fraction_within_macrostate"] / counts["background_fraction"].replace(0, np.nan)
        counts.insert(0, "candidate_id", candidate_id)
        counts.insert(1, "k", int(k_value))
        counts.insert(2, "annotation_group", group_name)
        tables[group_name] = counts.sort_values(["macrostate", "node_count"], ascending=[True, False])
        for macrostate, sub in counts.groupby("macrostate"):
            top = sub.sort_values("node_count", ascending=False).iloc[0]
            summary_rows.append(
                {
                    "candidate_id": candidate_id,
                    "k": int(k_value),
                    "macrostate": int(macrostate),
                    "annotation_group": group_name,
                    "dominant_label": top["label"],
                    "dominant_fraction": float(top["fraction_within_macrostate"]),
                    "annotation_entropy": entropy_from_counts(sub["node_count"]),
                    "macrostate_total": int(top["macrostate_total"]),
                }
            )
    return pd.DataFrame(summary_rows), tables


def build_all_annotations(paths: dict[str, Path], sensitivity: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summaries = []
    combined_tables: dict[str, list[pd.DataFrame]] = {
        "time": [],
        "neighborhood": [],
        "cell_type": [],
        "endpoint": [],
        "slice": [],
        "mouse": [],
        "p_fate_v1": [],
        "p_fate_v2": [],
    }
    for row in sensitivity.itertuples(index=False):
        if not as_bool(row.success):
            continue
        macro_path = Path(row.macro_path)
        if not macro_path.exists():
            continue
        macro = pd.read_csv(macro_path)
        summary, tables = summarize_annotation_for_macro(macro, int(row.k), str(row.candidate_id))
        summaries.append(summary)
        for key, table in tables.items():
            combined_tables.setdefault(key, []).append(table)
    summary_frame = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    table_frames = {
        key: pd.concat(value, ignore_index=True) if value else pd.DataFrame()
        for key, value in combined_tables.items()
    }
    return summary_frame, table_frames


def artifact_flags(sensitivity: pd.DataFrame, annotation_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in sensitivity.itertuples(index=False):
        if not as_bool(row.success):
            rows.append(
                {
                    "candidate_id": row.candidate_id,
                    "k": int(row.k),
                    "artifact": "pygpcca_failure",
                    "status": "FAIL",
                    "metric": "success",
                    "value": 0,
                    "threshold": 1,
                    "detail": row.error,
                }
            )
            continue
        rows.append(
            {
                "candidate_id": row.candidate_id,
                "k": int(row.k),
                "artifact": "major_macrostate_imbalance",
                "status": "WARN" if float(row.largest_macrostate_fraction) > 0.5 else "PASS",
                "metric": "largest_macrostate_fraction",
                "value": float(row.largest_macrostate_fraction),
                "threshold": 0.5,
                "detail": "Requires strong biological annotation to justify when flagged.",
            }
        )
        rows.append(
            {
                "candidate_id": row.candidate_id,
                "k": int(row.k),
                "artifact": "tiny_macrostate",
                "status": "WARN" if float(row.smallest_macrostate_fraction) < 0.005 else "PASS",
                "metric": "smallest_macrostate_fraction",
                "value": float(row.smallest_macrostate_fraction),
                "threshold": 0.005,
                "detail": "Smallest macrostate is below 0.5% of pilot nodes when flagged.",
            }
        )
        for group_name, threshold in [("slice", 0.5), ("mouse", 0.6), ("neighborhood", 0.85), ("endpoint", 0.85), ("time", 0.95)]:
            sub = annotation_summary[
                (annotation_summary["candidate_id"] == row.candidate_id)
                & (annotation_summary["k"] == row.k)
                & (annotation_summary["annotation_group"] == group_name)
            ]
            if sub.empty:
                continue
            max_fraction = float(sub["dominant_fraction"].max())
            rows.append(
                {
                    "candidate_id": row.candidate_id,
                    "k": int(row.k),
                    "artifact": f"{group_name}_dominance",
                    "status": "WARN" if max_fraction > threshold else "PASS",
                    "metric": "max_dominant_fraction",
                    "value": max_fraction,
                    "threshold": threshold,
                    "detail": f"Dominant {group_name} fraction by macrostate.",
                }
            )
    return pd.DataFrame(rows)


def select_preferred_k(sensitivity: pd.DataFrame, flags: pd.DataFrame) -> dict[str, Any]:
    successful = sensitivity[(sensitivity["candidate_id"] == PRIMARY_CANDIDATE) & (sensitivity["success"].map(as_bool))].copy()
    if successful.empty:
        return {
            "decision_category": "pyGPCCA_not_stable_enough",
            "selected_k": None,
            "reason": "No successful primary-candidate k values.",
        }
    warn_counts = (
        flags[(flags["candidate_id"] == PRIMARY_CANDIDATE) & (flags["status"] == "WARN")]
        .groupby("k")
        .size()
        .to_dict()
    )
    successful["warn_count"] = successful["k"].map(lambda value: int(warn_counts.get(value, 0)))
    successful["score"] = (
        successful["warn_count"] * 10
        + successful["largest_macrostate_fraction"]
        + successful["smallest_macrostate_fraction"].rsub(0.005).clip(lower=0) * 100
        - successful["max_membership_mean"] * 0.1
    )
    selected = successful.sort_values(["score", "largest_macrostate_fraction", "k"]).iloc[0]
    category = "select_k_for_terminal_review" if int(selected["warn_count"]) == 0 else "need_more_k_sensitivity"
    return {
        "decision_category": category,
        "selected_k": int(selected["k"]),
        "reason": f"Selected by lowest artifact-weighted score; warn_count={int(selected['warn_count'])}.",
    }


def inspect_standard_api(paths: dict[str, Path]) -> dict[str, Any]:
    code = r"""
import inspect, json
out = {}
try:
    import pygpcca
    methods = [m for m in dir(pygpcca.GPCCA) if not m.startswith("_")]
    out["pygpcca_available"] = True
    out["pygpcca_version"] = getattr(pygpcca, "__version__", "unknown")
    out["pygpcca_methods"] = methods
    out["pygpcca_terminal_api_available"] = any("terminal" in m.lower() or "initial" in m.lower() for m in methods)
    out["pygpcca_fate_probability_api_available"] = any("fate" in m.lower() or "absorption" in m.lower() for m in methods)
except Exception as exc:
    out["pygpcca_available"] = False
    out["pygpcca_error"] = f"{type(exc).__name__}: {exc}"
try:
    import cellrank
    out["cellrank_available"] = True
    out["cellrank_version"] = getattr(cellrank, "__version__", "unknown")
    out["cellrank_estimators"] = [m for m in dir(cellrank.estimators) if not m.startswith("_")] if hasattr(cellrank, "estimators") else []
    out["cellrank_kernels"] = [m for m in dir(cellrank.kernels) if not m.startswith("_")] if hasattr(cellrank, "kernels") else []
    out["cellrank_precomputed_available"] = any("Precomputed" in m for m in out["cellrank_kernels"])
    estimator_methods = []
    for module_name in ["cellrank.estimators", "cellrank.estimators.terminal_states"]:
        try:
            module = __import__(module_name, fromlist=["*"])
            for attr in ["GPCCA", "CFLARE"]:
                if hasattr(module, attr):
                    cls = getattr(module, attr)
                    out[f"cellrank_{attr.lower()}_available"] = True
                    estimator_methods.extend([m for m in dir(cls) if not m.startswith("_")])
        except Exception:
            pass
    out["cellrank_estimator_methods"] = sorted(set(estimator_methods))
    out["cellrank_terminal_api_available"] = any("terminal" in m.lower() or "initial" in m.lower() for m in estimator_methods)
    out["cellrank_fate_probability_api_available"] = any("fate" in m.lower() or "absorption" in m.lower() for m in estimator_methods)
except Exception as exc:
    out["cellrank_available"] = False
    out["cellrank_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out))
"""
    env = os.environ.copy()
    for key in ["TMPDIR", "TMP", "TEMP"]:
        env[key] = str(TMPDIR)
    result = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", PYGPCCA_ENV, "python", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        return {
            "api_inspection_status": "FAIL",
            "error": result.stderr or result.stdout,
            "terminal_states_computed": False,
            "fate_probabilities_computed": False,
        }
    parsed = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    cellrank_standard_path = bool(parsed.get("cellrank_precomputed_available")) and (
        bool(parsed.get("cellrank_gpcca_available")) or bool(parsed.get("cellrank_cflare_available"))
    )
    terminal_available = bool(parsed.get("pygpcca_terminal_api_available")) or (
        cellrank_standard_path and bool(parsed.get("cellrank_terminal_api_available"))
    )
    fate_available = bool(parsed.get("pygpcca_fate_probability_api_available")) or (
        cellrank_standard_path and bool(parsed.get("cellrank_fate_probability_api_available"))
    )
    parsed.update(
        {
            "api_inspection_status": "PASS",
            "terminal_state_standard_api_available": terminal_available,
            "fate_probability_standard_api_available": fate_available,
            "terminal_states_computed": False,
            "fate_probabilities_computed": False,
            "cellrank_precomputed_gpcca_path_available": cellrank_standard_path,
            "terminal_fate_reason": "No terminal/fate computation performed in K_gpcca-03; pyGPCCA exposes macrostates but no verified direct terminal/fate API for this precomputed pilot path." if not terminal_available or not fate_available else "Standard pyGPCCA/CellRank-compatible API path detected; execution deferred to K_gpcca-04 integration.",
        }
    )
    return parsed


def p_fate_comparison(annotation_tables: dict[str, pd.DataFrame], selected_k: int | None) -> pd.DataFrame:
    rows = []
    for group in ["p_fate_v1", "p_fate_v2", "endpoint"]:
        table = annotation_tables.get(group, pd.DataFrame())
        if table.empty or selected_k is None:
            rows.append(
                {
                    "comparison": group,
                    "selected_k": selected_k,
                    "status": "NOT_COMPUTED",
                    "mean_dominant_fraction": np.nan,
                    "max_dominant_fraction": np.nan,
                    "interpretation": "not computed",
                }
            )
            continue
        sub = table[(table["candidate_id"] == PRIMARY_CANDIDATE) & (table["k"] == selected_k)]
        top = sub.sort_values("node_count", ascending=False).groupby("macrostate").head(1)
        rows.append(
            {
                "comparison": group,
                "selected_k": selected_k,
                "status": "PASS" if not top.empty else "NOT_COMPUTED",
                "mean_dominant_fraction": float(top["fraction_within_macrostate"].mean()) if not top.empty else np.nan,
                "max_dominant_fraction": float(top["fraction_within_macrostate"].max()) if not top.empty else np.nan,
                "interpretation": "K_gpcca compared for consistency and added structure; P_fate remains valid frozen baseline.",
            }
        )
    return pd.DataFrame(rows)


def make_figures(paths: dict[str, Path], sensitivity: pd.DataFrame, annotation_tables: dict[str, pd.DataFrame], selected_k: int | None) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    success = sensitivity[(sensitivity["candidate_id"] == PRIMARY_CANDIDATE) & (sensitivity["success"].map(as_bool))].copy()
    if not success.empty:
        fig = paths["figures"] / "k_gpcca_03_macrostate_size_range_by_k.png"
        plt.figure()
        plt.plot(success["k"], success["macrostate_size_min"], marker="o", label="min")
        plt.plot(success["k"], success["macrostate_size_max"], marker="o", label="max")
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("macrostate size")
        plt.tight_layout()
        plt.savefig(fig, dpi=150)
        plt.close()
        figures.append((fig, "macrostate size range by k"))
        for column, label in [("membership_entropy_mean", "entropy"), ("max_membership_mean", "max membership"), ("largest_macrostate_fraction", "largest fraction")]:
            fig = paths["figures"] / f"k_gpcca_03_{column}.png"
            plt.figure()
            plt.plot(success["k"], success[column], marker="o")
            plt.xlabel("k")
            plt.ylabel(label)
            plt.tight_layout()
            plt.savefig(fig, dpi=150)
            plt.close()
            figures.append((fig, label))
    for group in ["time", "neighborhood", "cell_type", "endpoint"]:
        table = annotation_tables.get(group, pd.DataFrame())
        if table.empty or selected_k is None:
            continue
        sub = table[(table["candidate_id"] == PRIMARY_CANDIDATE) & (table["k"] == selected_k)]
        if sub.empty:
            continue
        top_labels = sub.groupby("label")["node_count"].sum().sort_values(ascending=False).head(20).index
        heat = pd.pivot_table(
            sub[sub["label"].isin(top_labels)],
            index="macrostate",
            columns="label",
            values="fraction_within_macrostate",
            aggfunc="sum",
            fill_value=0,
        )
        fig = paths["figures"] / f"k_gpcca_03_macrostate_by_{group}.png"
        plt.figure(figsize=(max(6, len(heat.columns) * 0.35), 4))
        plt.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
        plt.xticks(range(len(heat.columns)), heat.columns, rotation=90)
        plt.yticks(range(len(heat.index)), heat.index)
        plt.colorbar(label="fraction")
        plt.tight_layout()
        plt.savefig(fig, dpi=150)
        plt.close()
        figures.append((fig, f"macrostate by {group}"))
    if selected_k is not None:
        coarse_path = paths["gpcca"] / f"k_gpcca03_{PRIMARY_CANDIDATE}_k{selected_k}_coarse_transition.csv"
        if coarse_path.exists():
            coarse = pd.read_csv(coarse_path)
            fig = paths["figures"] / "k_gpcca_03_coarse_transition_selected_k.png"
            plt.figure(figsize=(5, 4))
            plt.imshow(coarse.to_numpy(dtype=float), aspect="auto", cmap="magma")
            plt.colorbar(label="transition")
            plt.xlabel("target macrostate")
            plt.ylabel("source macrostate")
            plt.tight_layout()
            plt.savefig(fig, dpi=150)
            plt.close()
            figures.append((fig, "coarse transition heatmap for selected k"))
    return pd.DataFrame(
        [{"figure": str(path), "description": desc, "bytes": int(path.stat().st_size)} for path, desc in figures]
    )


def write_reports(
    paths: dict[str, Path],
    input_validation: pd.DataFrame,
    sensitivity: pd.DataFrame,
    annotation_summary: pd.DataFrame,
    annotation_tables: dict[str, pd.DataFrame],
    flags: pd.DataFrame,
    selected: dict[str, Any],
    api: dict[str, Any],
    pfate: pd.DataFrame,
    figures: pd.DataFrame,
    safety: dict[str, Any],
    v2_comparison: pd.DataFrame,
) -> None:
    atomic_write_text(paths["reports"] / "k_gpcca_03_input_validation_report.md", "# K_gpcca-03 Input Validation\n\n" + markdown_table(input_validation) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_03_macrostate_sensitivity_report.md", "# K_gpcca-03 Macrostate Sensitivity\n\n" + markdown_table(sensitivity, max_rows=40) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_03_biological_annotation_report.md", "# K_gpcca-03 Biological Annotation\n\n" + markdown_table(annotation_summary, max_rows=80) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_03_artifact_review.md", "# K_gpcca-03 Artifact Review\n\n" + markdown_table(flags, max_rows=80) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_03_selected_k_decision_report.md", "# K_gpcca-03 Selected k Decision\n\n" + json.dumps(k01.json_safe(selected), indent=2) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_03_terminal_classification_feasibility_report.md", "# K_gpcca-03 Terminal Classification Feasibility\n\n" + json.dumps(k01.json_safe(api), indent=2) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_03_pilot_fate_probability_feasibility_report.md", "# K_gpcca-03 Pilot Fate Probability Feasibility\n\n" + json.dumps(k01.json_safe(api), indent=2) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_03_vs_p_fate_branch_comparison.md", "# K_gpcca-03 vs P_fate Branch Comparison\n\nP_fate remains the frozen endpoint-anchored baseline/control. K_gpcca is evaluated for consistency and added macrostate structure, not replacement.\n\n" + markdown_table(pfate) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_03_visualization_qc_report.md", "# K_gpcca-03 Visualization QC\n\n" + markdown_table(figures) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_03_benchmark_decision_report.md", "# K_gpcca-03 Benchmark Decision\n\n" + f"Decision: `{decision_category(selected, flags, api)}`\n")
    atomic_write_text(paths["reports"] / "k_gpcca_03_next_step_recommendation.md", "# K_gpcca-03 Next Step Recommendation\n\n" + next_step(selected, flags, api) + "\n")


def decision_category(selected: dict[str, Any], flags: pd.DataFrame, api: dict[str, Any]) -> str:
    if selected.get("selected_k") is None:
        return "pause_k_gpcca_and_report_limitations"
    selected_flags = flags[
        (flags["candidate_id"] == PRIMARY_CANDIDATE)
        & (flags["k"] == selected["selected_k"])
        & (flags["status"] == "WARN")
    ]
    if not selected_flags.empty:
        return "revise_k_gpcca_kernel_weights"
    if api.get("terminal_state_standard_api_available") and api.get("fate_probability_standard_api_available"):
        return "proceed_to_k_gpcca_04_pilot_fate_probability"
    return "proceed_to_k_gpcca_04_terminal_state_standard_api_integration"


def next_step(selected: dict[str, Any], flags: pd.DataFrame, api: dict[str, Any]) -> str:
    category = decision_category(selected, flags, api)
    if category == "proceed_to_k_gpcca_04_pilot_fate_probability":
        return "K_gpcca-04 pilot terminal/fate probability computation through a standard API."
    if category == "proceed_to_k_gpcca_04_terminal_state_standard_api_integration":
        return "K_gpcca-04 terminal-state standard API integration before fate-probability computation."
    if category == "revise_k_gpcca_kernel_weights":
        return "Revise K_gpcca kernel weights to address macrostate imbalance/artifacts before terminal-state work."
    return "Pause K_gpcca and report limitations."


def run(output_root: Path, k_values: list[int], skip_v2: bool, timeout_seconds: int) -> dict[str, Any]:
    start = time.perf_counter()
    paths = output_paths(output_root)
    ensure_dirs(paths)
    protected_before = k01.snapshot(PROTECTED_ROOTS)
    forbidden_before = k01.snapshot(FORBIDDEN_ROOTS)
    input_validation = validate_inputs(paths)
    if set(input_validation["status"]) != {"PASS"}:
        atomic_write_csv(paths["root"] / "k_gpcca_03_input_validation_summary.csv", input_validation)
        raise RuntimeError("K_gpcca-03 input validation failed")
    atomic_write_csv(paths["root"] / "k_gpcca_03_input_validation_summary.csv", input_validation)

    sensitivity = run_k_sensitivity(paths, k_values, timeout_seconds)
    v2_comparison = maybe_run_v2(paths, skip_v2, timeout_seconds)
    if not v2_comparison.empty:
        for row in v2_comparison.to_dict(orient="records"):
            upsert_sensitivity(paths, row)
        sensitivity = load_sensitivity(paths)

    annotation_summary, annotation_tables = build_all_annotations(paths, sensitivity)
    atomic_write_csv(paths["root"] / "k_gpcca_03_macrostate_annotation_summary.csv", annotation_summary)
    output_map = {
        "time": "k_gpcca_03_macrostate_by_time.csv",
        "neighborhood": "k_gpcca_03_macrostate_by_neighborhood.csv",
        "cell_type": "k_gpcca_03_macrostate_by_cell_type.csv",
        "endpoint": "k_gpcca_03_macrostate_by_endpoint.csv",
        "slice": "k_gpcca_03_macrostate_by_slice.csv",
        "mouse": "k_gpcca_03_macrostate_by_mouse.csv",
        "p_fate_v1": "k_gpcca_03_macrostate_by_p_fate_v1.csv",
        "p_fate_v2": "k_gpcca_03_macrostate_by_p_fate_v2.csv",
    }
    for key, filename in output_map.items():
        atomic_write_csv(paths["root"] / filename, annotation_tables.get(key, pd.DataFrame()))
    flags = artifact_flags(sensitivity, annotation_summary)
    atomic_write_csv(paths["root"] / "k_gpcca_03_artifact_flags.csv", flags)
    selected = select_preferred_k(sensitivity, flags)
    atomic_write_json(paths["root"] / "k_gpcca_03_selected_k_summary.json", selected)
    api = inspect_standard_api(paths)
    pfate = p_fate_comparison(annotation_tables, selected.get("selected_k"))
    atomic_write_csv(paths["root"] / "k_gpcca_03_vs_p_fate_summary.csv", pfate)
    figures = make_figures(paths, sensitivity, annotation_tables, selected.get("selected_k"))
    atomic_write_csv(paths["reports"] / "k_gpcca_03_figure_inventory.csv", figures)
    protected_after = k01.snapshot(PROTECTED_ROOTS)
    forbidden_after = k01.snapshot(FORBIDDEN_ROOTS)
    safety = {
        "upstream_metadata_diff_count": len(k01.diff_snapshot(protected_before, protected_after)),
        "forbidden_downstream_diff_count": len(k01.diff_snapshot(forbidden_before, forbidden_after)),
        "ssd_output_count": k01.count_ssd_outputs(paths["root"]),
        "custom_fallback_used": False,
    }
    write_reports(paths, input_validation, sensitivity, annotation_summary, annotation_tables, flags, selected, api, pfate, figures, safety, v2_comparison)
    summary = {
        "stage": "K_gpcca-03",
        "status": "PASSED",
        "generated_at_utc": utc_now(),
        "runtime_seconds": time.perf_counter() - start,
        "output_root": paths["root"],
        "k_values_run": [int(v) for v in sensitivity[sensitivity["candidate_id"] == PRIMARY_CANDIDATE]["k"].tolist()],
        "k_values_succeeded": [int(v) for v in sensitivity[(sensitivity["candidate_id"] == PRIMARY_CANDIDATE) & (sensitivity["success"].map(as_bool))]["k"].tolist()],
        "k_values_failed": [int(v) for v in sensitivity[(sensitivity["candidate_id"] == PRIMARY_CANDIDATE) & (~sensitivity["success"].map(as_bool))]["k"].tolist()],
        "selected_k": selected.get("selected_k"),
        "selected_k_decision": selected.get("decision_category"),
        "terminal_state_standard_api_available": api.get("terminal_state_standard_api_available"),
        "fate_probability_standard_api_available": api.get("fate_probability_standard_api_available"),
        "terminal_states_computed": api.get("terminal_states_computed", False),
        "fate_probabilities_computed": api.get("fate_probabilities_computed", False),
        "final_decision_category": decision_category(selected, flags, api),
        "next_recommended_step": next_step(selected, flags, api),
        **safety,
    }
    atomic_write_json(paths["root"] / "k_gpcca_03_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run(
        args.output_root,
        parse_k_values(args.k_values),
        args.skip_v2,
        args.timeout_seconds_per_k,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "k_values_run": summary["k_values_run"],
                "k_values_succeeded": summary["k_values_succeeded"],
                "k_values_failed": summary["k_values_failed"],
                "selected_k": summary["selected_k"],
                "final_decision_category": summary["final_decision_category"],
                "upstream_metadata_diff_count": summary["upstream_metadata_diff_count"],
                "ssd_output_count": summary["ssd_output_count"],
                "custom_fallback_used": summary["custom_fallback_used"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
