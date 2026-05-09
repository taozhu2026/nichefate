#!/usr/bin/env python
"""K_gpcca-04 bounded kernel weight and feature-construction revision pilot.

This runner constructs only bounded K_gpcca revision pilot kernels and runs
standard pyGPCCA at k=10 for QC-passing candidates. It does not compute
terminal states, fate probabilities, CellRank production outputs, or custom
GPCCA-like fallback outputs. No custom GPCCA-like fallback is implemented.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import k_gpcca_01_pilot_kernel_preflight as k01
import k_gpcca_02_construct_and_run_pilot as k02
import k_gpcca_03_biological_benchmark as k03


ROOT = Path("/home/zhutao/scratch/nichefate")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "k_gpcca_revision.yaml"
TMPDIR = Path("/home/zhutao/tmp/k_gpcca")
PYGPCCA_ENV = "nichefate-gpcca"
BASELINE_CANDIDATE = "pilot_v1_balanced"
BASELINE_K = 10


@dataclass(frozen=True)
class RevisionCandidate:
    grid_id: str
    route: str
    cross_time_source: str
    alpha: float
    beta: float
    gamma: float
    delta: float
    within_time_k: int
    similarity_metric: str
    priority: str

    def as_k02(self) -> k02.Candidate:
        return k02.Candidate(
            grid_id=self.grid_id,
            route=self.route,
            cross_time_source=self.cross_time_source,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
            within_time_k=self.within_time_k,
            similarity_metric=self.similarity_metric,
            priority=self.priority,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--candidate-id", action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--timeout-seconds-per-candidate", type=int, default=3600)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config did not parse to a mapping: {path}")
    return config


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


def markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    return k03.markdown_table(frame, max_rows=max_rows)


def output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    root = k01.resolved(config["paths"]["output_root"])
    reports = k01.resolved(config["paths"].get("reports_dir", root / "reports"))
    k01.reject_ssd(root)
    k01.reject_ssd(reports)
    if not k01.is_relative_to(reports, root):
        raise ValueError(f"Reports directory must be under output root: {reports}")
    for protected in [k01.resolved(path) for path in config.get("protected_roots", [])]:
        if k01.paths_overlap(root, protected):
            raise ValueError(f"Output root overlaps protected root {protected}: {root}")
    for forbidden in [k01.resolved(path) for path in config.get("forbidden_downstream_roots", [])]:
        if k01.paths_overlap(root, forbidden):
            raise ValueError(f"Output root overlaps forbidden root {forbidden}: {root}")
    paths = {
        "root": root,
        "kernels": root / "kernels",
        "gpcca": root / "gpcca",
        "reports": reports,
        "figures": reports / "figures",
    }
    for path in paths.values():
        k01.reject_ssd(path)
    return paths


def ensure_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def load_candidates(config: dict[str, Any], candidate_ids: list[str] | None = None) -> list[RevisionCandidate]:
    rows = config.get("revision_candidates", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("revision_candidates must be a non-empty list")
    requested = set(candidate_ids or [])
    candidates: list[RevisionCandidate] = []
    for row in rows:
        candidate = RevisionCandidate(
            grid_id=str(row["grid_id"]),
            route=str(row.get("route", "full_resolution_subset")),
            cross_time_source=str(row["cross_time_source"]),
            alpha=float(row["alpha"]),
            beta=float(row["beta"]),
            gamma=float(row["gamma"]),
            delta=float(row.get("delta", 0.0)),
            within_time_k=int(row["within_time_k"]),
            similarity_metric=str(row.get("similarity_metric", "cosine")),
            priority=str(row.get("priority", "bounded_revision")),
        )
        if requested and candidate.grid_id not in requested:
            continue
        if candidate.route != "full_resolution_subset":
            raise ValueError(f"Non-pilot route is not executable in K04: {candidate.grid_id}")
        if candidate.cross_time_source not in {"M3-v1", "M3-v2"}:
            raise ValueError(f"Unsupported cross-time source: {candidate.cross_time_source}")
        if candidate.delta != 0:
            raise ValueError(f"Barcode/delta candidates are not executable in K04: {candidate.grid_id}")
        candidates.append(candidate)
    if requested and requested - {candidate.grid_id for candidate in candidates}:
        raise ValueError(f"Unknown candidate ids: {sorted(requested - {candidate.grid_id for candidate in candidates})}")
    if len(candidates) > 5:
        raise ValueError("K04 is bounded: refusing to run more than five candidates")
    return candidates


def kernel_paths(paths: dict[str, Path], candidate_id: str) -> dict[str, Path]:
    return {
        "matrix": paths["kernels"] / f"K_gpcca_revision_{candidate_id}.npz",
        "node_table": paths["kernels"] / f"K_gpcca_revision_node_table_{candidate_id}.parquet",
        "qc_csv": paths["reports"] / f"k_gpcca_04_kernel_qc_{candidate_id}.csv",
        "report_md": paths["reports"] / f"k_gpcca_04_kernel_report_{candidate_id}.md",
    }


def gpcca_paths(paths: dict[str, Path], candidate_id: str, k_value: int) -> dict[str, Path]:
    return {
        "macro": paths["gpcca"] / f"k_gpcca_04_pygpcca_macrostates_{candidate_id}_k{k_value}.csv",
        "memberships": paths["gpcca"] / f"k_gpcca_04_pygpcca_memberships_{candidate_id}_k{k_value}.parquet",
        "coarse": paths["gpcca"] / f"k_gpcca_04_pygpcca_coarse_transition_{candidate_id}_k{k_value}.csv",
        "report": paths["reports"] / f"k_gpcca_04_pygpcca_report_{candidate_id}_k{k_value}.md",
    }


def classify_feature(column: str, denylist: list[str]) -> tuple[str, bool, str]:
    lowered = column.lower()
    if any(
        lowered == token
        or lowered.startswith(f"{token}_")
        or lowered.endswith(f"_{token}")
        for token in denylist
    ):
        return "excluded_metadata_like", True, "matches metadata denylist"
    if "__ct_" in column:
        return "composition", False, "cell-type composition or entropy"
    if "__emb_mean_pc" in column or "__emb_var_pc" in column:
        return "molecular_state", False, "embedding mean/variance PC summary"
    if any(token in column for token in ["__n_neighbors", "__mean_neighbor_distance", "__pseudo_local_density", "__local_topology_"]):
        return "spatial_topology", False, "neighborhood count/distance/density/topology"
    return "other_numeric_diagnostics", False, "numeric feature not matched to primary groups"


def feature_audit(config: dict[str, Any]) -> pd.DataFrame:
    columns = k02.m2_feature_columns(config)
    denylist = [str(item).lower() for item in config["feature_processing"].get("metadata_denylist", [])]
    rows = []
    for column in columns:
        group, excluded, reason = classify_feature(column, denylist)
        rows.append(
            {
                "feature_column": column,
                "feature_group": group,
                "excluded": bool(excluded),
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def group_indices(audit: pd.DataFrame) -> dict[str, np.ndarray]:
    usable = audit[~audit["excluded"]].copy()
    groups: dict[str, np.ndarray] = {}
    for group_name, frame in usable.groupby("feature_group", sort=True):
        groups[group_name] = frame.index.to_numpy(dtype=np.int64)
    return groups


def group_balanced_embedding(
    features: np.ndarray,
    groups: dict[str, np.ndarray],
    n_components: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    from sklearn.decomposition import TruncatedSVD

    blocks: list[np.ndarray] = []
    rows = []
    for group_name in sorted(groups):
        indices = groups[group_name]
        if len(indices) == 0:
            continue
        block = features[:, indices].astype(np.float32, copy=True)
        means = block.mean(axis=0, dtype=np.float64).astype(np.float32)
        stds = block.std(axis=0, dtype=np.float64).astype(np.float32)
        stds[stds == 0] = 1.0
        block -= means
        block /= stds
        block /= np.sqrt(float(block.shape[1]))
        blocks.append(block)
        rows.append(
            {
                "feature_group": group_name,
                "column_count": int(block.shape[1]),
                "scaling": "zscore_then_divide_by_sqrt_group_size",
                "finite_values": bool(np.isfinite(block).all()),
            }
        )
    if not blocks:
        raise ValueError("No usable feature groups remain after filtering")
    balanced = np.concatenate(blocks, axis=1)
    if not np.isfinite(balanced).all():
        raise ValueError("Nonfinite values after group-balanced scaling")
    if balanced.shape[1] <= n_components or balanced.shape[0] <= n_components + 1:
        return balanced.astype(np.float32, copy=False), pd.DataFrame(rows)
    svd = TruncatedSVD(
        n_components=min(n_components, balanced.shape[1] - 1, balanced.shape[0] - 1),
        random_state=0,
        algorithm="randomized",
    )
    embedding = svd.fit_transform(balanced).astype(np.float32, copy=False)
    variance = float(np.sum(svd.explained_variance_ratio_))
    scale_report = pd.DataFrame(rows)
    scale_report["svd_components"] = int(embedding.shape[1])
    scale_report["svd_explained_variance_ratio_sum"] = variance
    return embedding, scale_report


def write_feature_audit(paths: dict[str, Path], audit: pd.DataFrame, scale_report: pd.DataFrame) -> None:
    summary = (
        audit.groupby(["feature_group", "excluded"], dropna=False)
        .size()
        .reset_index(name="column_count")
    )
    atomic_write_csv(paths["root"] / "k_gpcca_04_feature_group_audit.csv", audit)
    body = [
        "# K_gpcca-04 Feature Audit",
        "",
        "Current K02 construction used all M2 numeric features, global z-score scaling, TruncatedSVD-50, and cosine same-time kNN.",
        "K04 uses group-balanced z-score scaling before TruncatedSVD-50.",
        "",
        "## Feature Groups",
        "",
        markdown_table(summary),
        "",
        "## Group Scaling",
        "",
        markdown_table(scale_report),
        "",
        "Direct metadata columns are excluded by denylist if present. The current M2 numeric schema contains no direct slice/mouse/time IDs, but topology/count features can still correlate with sample structure and are monitored downstream.",
    ]
    atomic_write_text(paths["reports"] / "k_gpcca_04_feature_audit_report.md", "\n".join(body) + "\n")


def write_kernel_report(path: Path, qc: dict[str, Any]) -> None:
    lines = [
        f"# K_gpcca-04 Kernel Report: {qc['candidate_id']}",
        "",
        f"Generated: {utc_now()}",
        "",
    ]
    for key, value in qc.items():
        lines.append(f"- `{key}`: {value}")
    atomic_write_text(path, "\n".join(lines) + "\n")


def construct_candidate_kernel(
    paths: dict[str, Path],
    selected: pd.DataFrame,
    within_graphs: dict[int, sparse.csr_matrix],
    cross_graphs: dict[str, sparse.csr_matrix],
    candidate: RevisionCandidate,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    outputs = kernel_paths(paths, candidate.grid_id)
    if outputs["matrix"].exists() and outputs["qc_csv"].exists() and resume and not overwrite:
        qc = pd.read_csv(outputs["qc_csv"]).iloc[0].to_dict()
        return {"candidate_id": candidate.grid_id, "qc": qc, "paths": outputs}
    if outputs["matrix"].exists() and not overwrite and not resume:
        raise FileExistsError(f"Kernel already exists; use --resume or --overwrite: {outputs['matrix']}")
    within = within_graphs[candidate.within_time_k]
    cross = cross_graphs[candidate.cross_time_source]
    kernel, masses = k02.combine_components(within, cross, candidate.as_k02())
    sparse.save_npz(outputs["matrix"], kernel, compressed=True)
    selected.to_parquet(outputs["node_table"], index=False)
    qc = k02.kernel_qc(kernel, within, cross, selected, candidate.as_k02(), masses, outputs["matrix"])
    qc["feature_processing_mode"] = "group_balanced_zscore_svd50"
    atomic_write_csv(outputs["qc_csv"], pd.DataFrame([qc]))
    write_kernel_report(outputs["report_md"], qc)
    return {"candidate_id": candidate.grid_id, "qc": qc, "paths": outputs}


def kernel_qc_allows_gpcca(qc: dict[str, Any]) -> bool:
    return k03.as_bool(qc.get("kernel_qc_pass", False))


def run_standard_pygpcca(
    paths: dict[str, Path],
    candidate_id: str,
    k_value: int,
    timeout_seconds: int,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    kernel = kernel_paths(paths, candidate_id)["matrix"]
    node_table = kernel_paths(paths, candidate_id)["node_table"]
    outputs = gpcca_paths(paths, candidate_id, k_value)
    if outputs["macro"].exists() and outputs["memberships"].exists() and outputs["coarse"].exists() and resume and not overwrite:
        macro = pd.read_csv(outputs["macro"], low_memory=False)
        return k03.summarize_macrostate_result(
            macro,
            candidate_id,
            k_value,
            True,
            "",
            "standard pyGPCCA reused existing K04 output",
            0.0,
            outputs["macro"],
            outputs["memberships"],
            outputs["coarse"],
        )
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
matrix = sparse.load_npz(payload["matrix"]).tocsr().astype(float)
row_sums = np.asarray(matrix.sum(axis=1)).ravel()
inv = np.zeros_like(row_sums, dtype=float)
mask = row_sums > 0
inv[mask] = 1.0 / row_sums[mask]
matrix = sparse.diags(inv).dot(matrix).tocsr()
node_table = pd.read_parquet(payload["node_table"])
result = {
    "candidate_id": payload["candidate_id"],
    "k": int(payload["k"]),
    "success": False,
    "error": "",
    "runtime_seconds": None,
}
try:
    gpcca = pygpcca.GPCCA(matrix, z="LM", method="krylov")
    gpcca.optimize(int(payload["k"]))
    memberships = np.real_if_close(np.asarray(gpcca.memberships)).astype(float)
    memberships[memberships < 0] = 0.0
    row_sums = memberships.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    memberships = memberships / row_sums
    assignments = np.asarray(gpcca.macrostate_assignment).reshape(-1).astype(int)
    coarse = np.real_if_close(np.asarray(gpcca.coarse_grained_transition_matrix)).astype(float)
    entropy = -np.sum(np.where(memberships > 0, memberships * np.log(memberships), 0.0), axis=1)
    max_membership = memberships.max(axis=1)
    mdf = pd.DataFrame(memberships, columns=[f"membership_{i}" for i in range(memberships.shape[1])])
    mdf.insert(0, "global_node_index", node_table["global_node_index"].to_numpy())
    mdf.insert(1, "local_index", np.arange(len(mdf)))
    mdf.to_parquet(payload["memberships"], index=False)
    macro = node_table.copy()
    macro["macrostate"] = assignments
    macro["membership_entropy"] = entropy
    macro["max_membership"] = max_membership
    macro.to_csv(payload["macro"], index=False)
    pd.DataFrame(coarse).to_csv(payload["coarse"], index=False)
    result["success"] = True
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
result["runtime_seconds"] = time.perf_counter() - start
print(json.dumps(result))
"""
    payload = {
        "matrix": str(kernel),
        "node_table": str(node_table),
        "candidate_id": candidate_id,
        "k": int(k_value),
        "macro": str(outputs["macro"]),
        "memberships": str(outputs["memberships"]),
        "coarse": str(outputs["coarse"]),
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
            ["conda", "run", "--no-capture-output", "-n", PYGPCCA_ENV, "python", "-c", code, json.dumps(payload)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        row = {
            "candidate_id": candidate_id,
            "k": int(k_value),
            "success": False,
            "error": f"TimeoutExpired: exceeded {timeout_seconds} seconds",
            "runtime_seconds": float(timeout_seconds),
            "macrostate_count": 0,
            "macrostate_size_min": np.nan,
            "macrostate_size_max": np.nan,
            "largest_macrostate_fraction": np.nan,
            "smallest_macrostate_fraction": np.nan,
            "membership_entropy_mean": np.nan,
            "membership_entropy_median": np.nan,
            "max_membership_mean": np.nan,
            "max_membership_median": np.nan,
            "macro_path": str(outputs["macro"]),
            "memberships_path": str(outputs["memberships"]),
            "coarse_path": str(outputs["coarse"]),
        }
        atomic_write_text(outputs["report"], "# K_gpcca-04 pyGPCCA Timeout\n\n" + json.dumps(k01.json_safe(row), indent=2) + "\n")
        return row
    if result.returncode != 0:
        row = {
            "candidate_id": candidate_id,
            "k": int(k_value),
            "success": False,
            "error": f"SubprocessError: {result.stderr or result.stdout}",
            "runtime_seconds": np.nan,
            "macrostate_count": 0,
            "macrostate_size_min": np.nan,
            "macrostate_size_max": np.nan,
            "largest_macrostate_fraction": np.nan,
            "smallest_macrostate_fraction": np.nan,
            "membership_entropy_mean": np.nan,
            "membership_entropy_median": np.nan,
            "max_membership_mean": np.nan,
            "max_membership_median": np.nan,
            "macro_path": str(outputs["macro"]),
            "memberships_path": str(outputs["memberships"]),
            "coarse_path": str(outputs["coarse"]),
        }
        atomic_write_text(outputs["report"], "# K_gpcca-04 pyGPCCA Failure\n\n" + json.dumps(k01.json_safe(row), indent=2) + "\n")
        return row
    parsed = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    if parsed.get("success"):
        macro = pd.read_csv(outputs["macro"], low_memory=False)
        row = k03.summarize_macrostate_result(
            macro,
            candidate_id,
            k_value,
            True,
            "",
            "standard pyGPCCA",
            float(parsed.get("runtime_seconds", np.nan)),
            outputs["macro"],
            outputs["memberships"],
            outputs["coarse"],
        )
    else:
        row = {
            **parsed,
            "macrostate_count": 0,
            "macrostate_size_min": np.nan,
            "macrostate_size_max": np.nan,
            "largest_macrostate_fraction": np.nan,
            "smallest_macrostate_fraction": np.nan,
            "membership_entropy_mean": np.nan,
            "membership_entropy_median": np.nan,
            "max_membership_mean": np.nan,
            "max_membership_median": np.nan,
            "macro_path": str(outputs["macro"]),
            "memberships_path": str(outputs["memberships"]),
            "coarse_path": str(outputs["coarse"]),
        }
    atomic_write_text(outputs["report"], "# K_gpcca-04 pyGPCCA Report\n\n" + json.dumps(k01.json_safe(row), indent=2) + "\n")
    return row


def load_k03_baseline(config: dict[str, Any]) -> tuple[dict[str, Any], int]:
    root = k01.resolved(config["paths"]["k03_benchmark_root"])
    sensitivity = pd.read_csv(root / "k_gpcca_03_macrostate_sensitivity_by_k.csv")
    row = sensitivity[(sensitivity["candidate_id"] == BASELINE_CANDIDATE) & (sensitivity["k"] == BASELINE_K)]
    if row.empty:
        raise FileNotFoundError("Missing K03 k=10 baseline sensitivity row")
    flags = pd.read_csv(root / "k_gpcca_03_artifact_flags.csv")
    warn_count = int(
        len(
            flags[
                (flags["candidate_id"] == BASELINE_CANDIDATE)
                & (flags["k"] == BASELINE_K)
                & (flags["status"] == "WARN")
            ]
        )
    )
    baseline = row.iloc[0].to_dict()
    baseline["warn_count"] = warn_count
    return baseline, warn_count


def annotate_revision(paths: dict[str, Path], gpcca_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    summary, tables = k03.build_all_annotations(paths, gpcca_rows)
    atomic_write_csv(paths["root"] / "k_gpcca_04_macrostate_annotation_summary.csv", summary)
    for key, table in tables.items():
        atomic_write_csv(paths["root"] / f"k_gpcca_04_macrostate_by_{key}.csv", table)
    flags = k03.artifact_flags(gpcca_rows, summary)
    atomic_write_csv(paths["root"] / "k_gpcca_04_artifact_flags.csv", flags)
    return summary, tables, flags


def build_comparison(
    baseline: dict[str, Any],
    baseline_warn_count: int,
    candidates: list[RevisionCandidate],
    kernel_qc: pd.DataFrame,
    gpcca: pd.DataFrame,
    flags: pd.DataFrame,
    acceptance: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        qc_row = kernel_qc[kernel_qc["candidate_id"] == candidate.grid_id]
        gpcca_row = gpcca[gpcca["candidate_id"] == candidate.grid_id]
        warn_count = int(len(flags[(flags["candidate_id"] == candidate.grid_id) & (flags["status"] == "WARN")]))
        record = {
            "candidate_id": candidate.grid_id,
            "alpha": candidate.alpha,
            "beta": candidate.beta,
            "gamma": candidate.gamma,
            "within_time_k": candidate.within_time_k,
            "cross_time_source": candidate.cross_time_source,
            "feature_processing_mode": "group_balanced_zscore_svd50",
            "kernel_qc_pass": False,
            "pygpcca_success": False,
            "warn_count": warn_count,
            "baseline_warn_count": baseline_warn_count,
            "warn_count_delta": warn_count - baseline_warn_count,
            "selection_eligible": False,
        }
        if not qc_row.empty:
            for key in [
                "selected_node_count",
                "nnz",
                "row_sum_max_error",
                "weak_component_count",
                "largest_weak_component_fraction",
                "within_time_mass_fraction",
                "cross_time_mass_fraction",
                "self_loop_mass_fraction",
                "kernel_qc_pass",
            ]:
                record[key] = qc_row.iloc[0].get(key)
        if not gpcca_row.empty:
            for key in [
                "success",
                "macrostate_size_min",
                "macrostate_size_max",
                "largest_macrostate_fraction",
                "smallest_macrostate_fraction",
                "membership_entropy_mean",
                "membership_entropy_median",
                "max_membership_mean",
                "max_membership_median",
            ]:
                record[f"pygpcca_{key}" if key == "success" else key] = gpcca_row.iloc[0].get(key)
            record["pygpcca_success"] = k03.as_bool(gpcca_row.iloc[0].get("success"))
            record["largest_fraction_delta_vs_k03"] = float(gpcca_row.iloc[0]["largest_macrostate_fraction"]) - float(baseline["largest_macrostate_fraction"])
            record["smallest_fraction_delta_vs_k03"] = float(gpcca_row.iloc[0]["smallest_macrostate_fraction"]) - float(baseline["smallest_macrostate_fraction"])
            record["max_membership_delta_vs_k03"] = float(gpcca_row.iloc[0]["max_membership_mean"]) - float(baseline["max_membership_mean"])
        record["selection_eligible"] = bool(
            k03.as_bool(record.get("kernel_qc_pass"))
            and k03.as_bool(record.get("pygpcca_success"))
            and float(record.get("largest_macrostate_fraction", np.inf)) < float(acceptance["largest_macrostate_fraction_lt"])
            and float(record.get("smallest_macrostate_fraction", -np.inf)) >= float(acceptance["smallest_macrostate_fraction_gte"])
            and float(record.get("max_membership_mean", -np.inf)) > float(acceptance["mean_max_membership_gt"])
            and warn_count < baseline_warn_count
        )
        rows.append(record)
    return pd.DataFrame(rows)


def select_revision(comparison: pd.DataFrame) -> dict[str, Any]:
    eligible = comparison[comparison["selection_eligible"].map(k03.as_bool)].copy()
    if not eligible.empty:
        eligible["score"] = (
            eligible["warn_count"] * 10
            + eligible["largest_macrostate_fraction"]
            - eligible["max_membership_mean"] * 0.1
        )
        selected = eligible.sort_values(["score", "warn_count", "largest_macrostate_fraction"]).iloc[0]
        return {
            "selected_candidate": selected["candidate_id"],
            "decision_category": "revised_kernel_selected_for_terminal_review",
            "reason": "Candidate satisfies bounded K04 acceptance criteria.",
        }
    successful = comparison[comparison["pygpcca_success"].map(k03.as_bool)].copy()
    if successful.empty:
        return {
            "selected_candidate": None,
            "decision_category": "pause_k_gpcca_report_limitations",
            "reason": "No revised candidate produced successful standard pyGPCCA output.",
        }
    improves_any = bool(
        (
            (successful["warn_count"] < successful["baseline_warn_count"])
            | (successful["largest_fraction_delta_vs_k03"] < 0)
            | (successful["smallest_fraction_delta_vs_k03"] > 0)
        ).any()
    )
    return {
        "selected_candidate": None,
        "decision_category": "need_feature_processing_redesign" if improves_any else "no_revision_better_keep_k03_baseline",
        "reason": "No revised candidate satisfies all acceptance criteria; terminal/fate work remains blocked.",
    }


def p_fate_overlap(tables: dict[str, pd.DataFrame], candidates: list[RevisionCandidate], k_value: int) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        for group in ["p_fate_v1", "p_fate_v2", "endpoint"]:
            table = tables.get(group, pd.DataFrame())
            sub = table[(table["candidate_id"] == candidate.grid_id) & (table["k"] == k_value)] if not table.empty else pd.DataFrame()
            top = sub.sort_values("node_count", ascending=False).groupby("macrostate").head(1) if not sub.empty else pd.DataFrame()
            rows.append(
                {
                    "candidate_id": candidate.grid_id,
                    "comparison": group,
                    "mean_dominant_fraction": float(top["fraction_within_macrostate"].mean()) if not top.empty else np.nan,
                    "max_dominant_fraction": float(top["fraction_within_macrostate"].max()) if not top.empty else np.nan,
                    "interpretation": "K_gpcca revision compared for consistency and added structure; P_fate remains frozen baseline.",
                }
            )
    return pd.DataFrame(rows)


def make_figures(paths: dict[str, Path], comparison: pd.DataFrame) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    if not comparison.empty:
        for column, ylabel in [
            ("largest_macrostate_fraction", "largest macrostate fraction"),
            ("smallest_macrostate_fraction", "smallest macrostate fraction"),
            ("warn_count", "artifact warning count"),
            ("max_membership_mean", "mean max membership"),
        ]:
            if column not in comparison.columns:
                continue
            fig = paths["figures"] / f"k_gpcca_04_{column}.png"
            plt.figure(figsize=(8, 4))
            plt.bar(comparison["candidate_id"], comparison[column])
            plt.xticks(rotation=30, ha="right")
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(fig, dpi=150)
            plt.close()
            figures.append({"figure": str(fig), "description": ylabel, "bytes": int(fig.stat().st_size)})
    inventory = pd.DataFrame(figures)
    atomic_write_csv(paths["reports"] / "k_gpcca_04_figure_inventory.csv", inventory)
    return inventory


def write_reports(
    paths: dict[str, Path],
    kernel_qc: pd.DataFrame,
    gpcca: pd.DataFrame,
    annotation_summary: pd.DataFrame,
    flags: pd.DataFrame,
    pfate: pd.DataFrame,
    comparison: pd.DataFrame,
    decision: dict[str, Any],
    safety: dict[str, Any],
) -> None:
    atomic_write_text(paths["reports"] / "k_gpcca_04_revision_comparison_report.md", "# K_gpcca-04 Revision Comparison\n\n" + markdown_table(comparison, 80) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_04_biological_annotation_report.md", "# K_gpcca-04 Biological Annotation\n\n" + markdown_table(annotation_summary, 80) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_04_artifact_review.md", "# K_gpcca-04 Artifact Review\n\n" + markdown_table(flags, 100) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_04_vs_p_fate_comparison.md", "# K_gpcca-04 vs P_fate\n\nP_fate remains the frozen endpoint-anchored baseline/control. K_gpcca revisions are evaluated for consistency and added macrostate structure, not replacement.\n\n" + markdown_table(pfate, 80) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_04_revision_decision_report.md", "# K_gpcca-04 Revision Decision\n\n" + json.dumps(k01.json_safe(decision), indent=2) + "\n")
    next_step = next_step_text(decision)
    atomic_write_text(paths["reports"] / "k_gpcca_04_next_step_recommendation.md", "# K_gpcca-04 Next Step Recommendation\n\n" + next_step + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_04_kernel_qc_summary.md", "# K_gpcca-04 Kernel QC Summary\n\n" + markdown_table(kernel_qc, 80) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_04_pygpcca_summary.md", "# K_gpcca-04 pyGPCCA Summary\n\n" + markdown_table(gpcca, 80) + "\n")
    atomic_write_text(paths["reports"] / "k_gpcca_04_safety_report.md", "# K_gpcca-04 Safety Report\n\n" + json.dumps(k01.json_safe(safety), indent=2) + "\n")


def next_step_text(decision: dict[str, Any]) -> str:
    if decision["decision_category"] == "revised_kernel_selected_for_terminal_review":
        return f"K_gpcca-05 standard terminal-state / fate-probability pilot using selected revised kernel `{decision['selected_candidate']}`."
    if decision["decision_category"] == "no_revision_better_keep_k03_baseline":
        return "Keep K03 baseline for limited pilot interpretation or pause K_gpcca before terminal/fate computation."
    if decision["decision_category"] == "need_feature_processing_redesign":
        return "Redesign K_gpcca feature processing or evaluate supernode strategy before terminal/fate computation."
    return "Pause K_gpcca and report standard pyGPCCA revision limitations."


def run(
    config_path: Path,
    candidate_ids: list[str] | None,
    max_nodes: int | None,
    resume: bool,
    overwrite: bool,
    timeout_seconds: int,
    stop_on_error: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    config = load_config(config_path)
    paths = output_dirs(config)
    ensure_dirs(paths)
    candidates = load_candidates(config, candidate_ids)
    protected = [k01.resolved(path) for path in config.get("protected_roots", [])]
    forbidden = [k01.resolved(path) for path in config.get("forbidden_downstream_roots", [])]
    protected_before = k01.snapshot(protected)
    forbidden_before = k01.snapshot(forbidden)

    max_node_count = int(max_nodes or config["pilot"]["target_max_nodes"])
    selected, time_points, time_pairs = k02.prepare_selected_nodes(config, max_node_count)
    audit = feature_audit(config)
    features = k02.read_selected_m2_features(config, selected)
    embedding, scale_report = group_balanced_embedding(
        features,
        group_indices(audit),
        int(config["feature_processing"].get("svd_components", 50)),
    )
    write_feature_audit(paths, audit, scale_report)

    within_graphs = {}
    for within_k in sorted({candidate.within_time_k for candidate in candidates}):
        within_graphs[within_k] = k02.build_within_time_graph(selected, embedding, within_k, "cosine")
    cross_graphs = {}
    for source in sorted({candidate.cross_time_source for candidate in candidates}):
        representative = next(candidate for candidate in candidates if candidate.cross_time_source == source)
        cross_graphs[source] = k02.build_cross_time_graph(config, selected, representative.as_k02(), time_pairs)

    kernel_results = []
    gpcca_rows = []
    for candidate in candidates:
        try:
            result = construct_candidate_kernel(paths, selected, within_graphs, cross_graphs, candidate, resume, overwrite)
            kernel_results.append(result["qc"])
            if kernel_qc_allows_gpcca(result["qc"]):
                gpcca_rows.append(
                    run_standard_pygpcca(
                        paths,
                        candidate.grid_id,
                        int(config["acceptance"]["k"]),
                        timeout_seconds,
                        resume,
                        overwrite,
                    )
                )
            else:
                gpcca_rows.append(
                    {
                        "candidate_id": candidate.grid_id,
                        "k": int(config["acceptance"]["k"]),
                        "success": False,
                        "error": "kernel QC failed; pyGPCCA not run",
                        "runtime_seconds": np.nan,
                        "macrostate_count": 0,
                        "macrostate_size_min": np.nan,
                        "macrostate_size_max": np.nan,
                        "largest_macrostate_fraction": np.nan,
                        "smallest_macrostate_fraction": np.nan,
                        "membership_entropy_mean": np.nan,
                        "membership_entropy_median": np.nan,
                        "max_membership_mean": np.nan,
                        "max_membership_median": np.nan,
                        "macro_path": "",
                        "memberships_path": "",
                        "coarse_path": "",
                    }
                )
        except Exception:
            if stop_on_error:
                raise
            raise

    kernel_qc = pd.DataFrame(kernel_results)
    gpcca = pd.DataFrame(gpcca_rows)
    atomic_write_csv(paths["root"] / "k_gpcca_04_kernel_qc_summary.csv", kernel_qc)
    atomic_write_csv(paths["root"] / "k_gpcca_04_pygpcca_candidate_summary.csv", gpcca)
    annotation_summary, annotation_tables, flags = annotate_revision(paths, gpcca)
    pfate = p_fate_overlap(annotation_tables, candidates, int(config["acceptance"]["k"]))
    atomic_write_csv(paths["root"] / "k_gpcca_04_vs_p_fate_summary.csv", pfate)
    baseline, baseline_warn_count = load_k03_baseline(config)
    comparison = build_comparison(baseline, baseline_warn_count, candidates, kernel_qc, gpcca, flags, config["acceptance"])
    atomic_write_csv(paths["root"] / "k_gpcca_04_revision_candidate_comparison.csv", comparison)
    decision = select_revision(comparison)
    figures = make_figures(paths, comparison)

    protected_after = k01.snapshot(protected)
    forbidden_after = k01.snapshot(forbidden)
    safety = {
        "upstream_metadata_diff_count": len(k01.diff_snapshot(protected_before, protected_after)),
        "forbidden_downstream_diff_count": len(k01.diff_snapshot(forbidden_before, forbidden_after)),
        "ssd_output_count": k01.count_ssd_outputs(paths["root"]),
        "custom_fallback_used": False,
        "terminal_states_computed": False,
        "fate_probabilities_computed": False,
        "cellrank_executed": False,
    }
    write_reports(paths, kernel_qc, gpcca, annotation_summary, flags, pfate, comparison, decision, safety)
    summary = {
        "stage": "K_gpcca-04",
        "status": "PASSED",
        "generated_at_utc": utc_now(),
        "runtime_seconds": time.perf_counter() - start,
        "output_root": str(paths["root"]),
        "candidate_count": len(candidates),
        "candidates": [candidate.grid_id for candidate in candidates],
        "time_points": time_points,
        "time_pairs": time_pairs,
        "selected_node_count": int(len(selected)),
        "feature_processing_mode": "group_balanced_zscore_svd50",
        "k_tested": int(config["acceptance"]["k"]),
        "pygpcca_succeeded_candidates": gpcca[gpcca["success"].map(k03.as_bool)]["candidate_id"].tolist() if not gpcca.empty else [],
        "selected_candidate": decision["selected_candidate"],
        "final_decision_category": decision["decision_category"],
        "next_recommended_step": next_step_text(decision),
        "figure_count": int(len(figures)),
        **safety,
    }
    atomic_write_json(paths["root"] / "k_gpcca_04_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run(
        args.config,
        args.candidate_id,
        args.max_nodes,
        args.resume,
        args.overwrite,
        args.timeout_seconds_per_candidate,
        args.stop_on_error,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "candidates": summary["candidates"],
                "selected_candidate": summary["selected_candidate"],
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
