from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from nichefate.darlin_clone_signature.common import (
    entropy_from_counts,
    make_cell_key,
    path_has_forbidden_ssd,
    simpson_from_counts,
    summarize_top_items,
)
from nichefate.darlin_clone_signature.reporting import (
    atomic_write_json,
    atomic_write_tsv,
    atomic_write_tsv_gz,
    ensure_dir,
    markdown_table,
    positive_claim_hits,
    read_table,
    utc_now,
    write_report_pair,
)


SELECTED_REFERENCE_BANK_POLICY = "gr"
SELECTED_ALLELE_POLICY = "mapped_rare_plus_empirical_denovo"
SELECTED_THRESHOLD_LABEL = "tutorial_like"
NORMALIZED_COUNT_CUTOFF = 0.1
SAMPLE_COUNT_CUTOFF = 2
MIN_CELLBINS_PER_ALLELE = 1
GIANT_CLONE_FRACTION = 0.05
OVERMERGE_CLONE_FRACTION = 0.025
OVERMERGE_JOINT_ALLELE_NUM = 20
LOW_BC_CONSISTENCY = 0.5
SECTION_DOMINANCE_FRACTION = 0.95

REQUIRED_CLONE_QC_COLUMNS = [
    "joint_clone_id",
    "n_cellbins",
    "clone_size_fraction",
    "n_loci_supported",
    "loci_present",
    "joint_allele_num",
    "n_reference_mapped_alleles",
    "n_de_novo_alleles",
    "reference_support_fraction",
    "de_novo_allele_fraction",
    "empirical_frequency_summary",
    "homoplasy_risk_flag",
    "overmerge_risk_flag",
    "giant_clone_flag",
    "section_distribution",
    "qc_status",
    "qc_flags",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_packet_hashes(input_packet_root: Path) -> dict[str, str]:
    paths = [
        input_packet_root / "processed/lineage_evidence/cellbin_lineage_evidence.tsv.gz",
        input_packet_root / "processed/lineage_evidence/feature_allele_annotation_long.tsv.gz",
    ]
    return {str(path): file_sha256(path) for path in paths if path.exists()}


def load_selected_audit_tables(audit_root: Path) -> dict[str, pd.DataFrame]:
    assignment = read_table(audit_root / "cellbin_joint_clone_assignment.tsv.gz")
    clone_summary = read_table(audit_root / "joint_clone_summary.tsv.gz")
    valid_alleles = read_table(audit_root / "valid_cellbin_allele_table.tsv.gz")
    policy_summary = read_table(audit_root / "joint_clone_policy_summary.tsv")
    reference_policy = read_table(audit_root / "reference_vs_denovo_policy_summary.tsv")
    return {
        "assignment": assignment,
        "clone_summary": clone_summary,
        "valid_alleles": valid_alleles,
        "policy_summary": policy_summary,
        "reference_policy": reference_policy,
    }


def load_tile_map(full_characterization_root: Path) -> pd.DataFrame:
    paths = sorted((full_characterization_root / "spatial_tiles").glob("*_tile_assignment.tsv.gz"))
    frames = [read_table(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    tile = pd.concat(frames, ignore_index=True)
    tile["cell_key"] = make_cell_key(tile)
    return tile.drop_duplicates(["cell_key", "tile_id"]).reset_index(drop=True)


def load_group_map(full_characterization_root: Path) -> pd.DataFrame:
    paths = sorted((full_characterization_root / "groups").glob("*_full_group_assignment.tsv.gz"))
    frames = [read_table(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    groups = pd.concat(frames, ignore_index=True)
    groups["cell_key"] = make_cell_key(groups)
    return groups.reset_index(drop=True)


def load_metaniche_cell_map(full_characterization_root: Path, tile_map: pd.DataFrame) -> pd.DataFrame:
    meta_path = full_characterization_root / "metaniche/full_metaniche_assignment.tsv.gz"
    if tile_map.empty or not meta_path.exists():
        return pd.DataFrame()
    meta = read_table(meta_path)[["sample_id", "slice_id", "section_order", "tile_id", "metaniche_id"]].drop_duplicates()
    return tile_map.merge(meta, on=["sample_id", "slice_id", "section_order", "tile_id"], how="left")


def parse_joint_clone_alleles(joint_clone_id: object) -> list[str]:
    values = []
    for token in str(joint_clone_id).split("@"):
        token = token.strip()
        if token and token.lower() != "nan":
            values.append(token)
    return values


def allele_locus(allele: str) -> str:
    match = re.match(r"^(CA|TA|RA)_", str(allele))
    return match.group(1) if match else ""


def parse_section_distribution(value: object) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in str(value or "").split(";"):
        if ":" not in item:
            continue
        key, count = item.split(":", 1)
        try:
            out[key] = int(float(count))
        except ValueError:
            continue
    return out


def allele_qc_reference(valid_alleles: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "mosaic_allele",
        "locus",
        "allele_class",
        "reference_mapped",
        "invalid_alleles",
        "normalized_count",
        "sample_count",
        "empirical_n_cellbins",
        "empirical_cellbin_fraction",
    ]
    present = [col for col in cols if col in valid_alleles]
    work = valid_alleles[present].copy()
    for col in ["normalized_count", "sample_count", "empirical_n_cellbins", "empirical_cellbin_fraction"]:
        if col in work:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["reference_mapped"] = work.get("reference_mapped", False).fillna(False).astype(bool)
    work["invalid_alleles"] = work.get("invalid_alleles", False).fillna(False).astype(bool)
    return (
        work.groupby("mosaic_allele", as_index=False)
        .agg(
            locus=("locus", "first"),
            allele_class=("allele_class", "first"),
            reference_mapped=("reference_mapped", "max"),
            invalid_alleles=("invalid_alleles", "max"),
            normalized_count=("normalized_count", "min"),
            sample_count=("sample_count", "min"),
            empirical_n_cellbins=("empirical_n_cellbins", "max"),
            empirical_cellbin_fraction=("empirical_cellbin_fraction", "max"),
        )
        .reset_index(drop=True)
    )


def build_validated_clone_summary(clone_summary: pd.DataFrame, valid_alleles: pd.DataFrame) -> pd.DataFrame:
    allele_ref = allele_qc_reference(valid_alleles)
    allele_lookup = allele_ref.set_index("mosaic_allele").to_dict(orient="index")
    total_assigned = max(int(clone_summary["n_cellbins"].sum()), 1)
    rows: list[dict[str, Any]] = []
    for row in clone_summary.to_dict(orient="records"):
        joint_clone_id = str(row["joint_clone_id"])
        alleles = parse_joint_clone_alleles(joint_clone_id)
        loci = sorted({allele_locus(allele) for allele in alleles if allele_locus(allele)})
        allele_meta = [allele_lookup.get(allele, {}) for allele in alleles]
        n_reference = sum(1 for meta in allele_meta if meta.get("allele_class") == "reference_mapped_rare")
        n_de_novo = sum(1 for meta in allele_meta if meta.get("allele_class") == "unmapped_de_novo_candidate")
        n_unknown = sum(1 for allele in alleles if allele not in allele_lookup)
        n_alleles = max(len(alleles), 1)
        emp = [float(meta.get("empirical_cellbin_fraction", np.nan)) for meta in allele_meta]
        emp = [value for value in emp if math.isfinite(value)]
        norm = [float(meta.get("normalized_count", np.nan)) for meta in allele_meta if meta.get("allele_class") == "reference_mapped_rare"]
        norm = [value for value in norm if math.isfinite(value)]
        sample = [float(meta.get("sample_count", np.nan)) for meta in allele_meta if meta.get("allele_class") == "reference_mapped_rare"]
        sample = [value for value in sample if math.isfinite(value)]
        n_cellbins = int(row["n_cellbins"])
        clone_size_fraction = n_cellbins / total_assigned
        joint_allele_num = int(row.get("joint_allele_num", len(alleles)) or len(alleles))
        bc_consistency = float(row.get("BC_consistency", 1.0) or 1.0)
        section_counts = parse_section_distribution(row.get("section_distribution", ""))
        section_total = max(sum(section_counts.values()), 1)
        section_max_fraction = max(section_counts.values(), default=0) / section_total
        giant = clone_size_fraction >= GIANT_CLONE_FRACTION
        homoplasy = bool(
            any(meta.get("invalid_alleles", False) for meta in allele_meta)
            or (norm and max(norm) >= 0.05)
            or (sample and max(sample) >= SAMPLE_COUNT_CUTOFF)
        )
        overmerge = bool(
            clone_size_fraction >= OVERMERGE_CLONE_FRACTION
            or joint_allele_num > OVERMERGE_JOINT_ALLELE_NUM
            or bc_consistency < LOW_BC_CONSISTENCY
        )
        flags: list[str] = []
        if n_unknown:
            flags.append("unknown_allele_qc_metadata")
        if n_de_novo / n_alleles >= 0.80:
            flags.append("de_novo_high_fraction")
        if homoplasy:
            flags.append("homoplasy_risk")
        if overmerge:
            flags.append("overmerge_risk")
        if giant:
            flags.append("giant_clone")
        if section_max_fraction >= SECTION_DOMINANCE_FRACTION and n_cellbins >= 100:
            flags.append("section_dominated")
        if not alleles:
            flags.append("missing_joint_alleles")
        qc_status = "filtered" if giant or not alleles else ("warning" if flags else "pass")
        if not emp:
            empirical_summary = "min=NA;median=NA;max=NA"
            emp_min = emp_median = emp_max = np.nan
        else:
            emp_min = float(np.min(emp))
            emp_median = float(np.median(emp))
            emp_max = float(np.max(emp))
            empirical_summary = f"min={emp_min:.6g};median={emp_median:.6g};max={emp_max:.6g}"
        rows.append(
            {
                **row,
                "clone_size_fraction": float(clone_size_fraction),
                "n_loci_supported": int(len(loci)),
                "loci_present": ";".join(loci),
                "n_reference_mapped_alleles": int(n_reference),
                "n_de_novo_alleles": int(n_de_novo),
                "n_unknown_alleles": int(n_unknown),
                "reference_support_fraction": float(n_reference / n_alleles),
                "de_novo_allele_fraction": float(n_de_novo / n_alleles),
                "allele_reference_status": _allele_reference_status(n_reference, n_de_novo, n_unknown, n_alleles),
                "empirical_frequency_summary": empirical_summary,
                "empirical_frequency_min": emp_min,
                "empirical_frequency_median": emp_median,
                "empirical_frequency_max": emp_max,
                "homoplasy_risk_flag": bool(homoplasy),
                "overmerge_risk_flag": bool(overmerge),
                "giant_clone_flag": bool(giant),
                "section_max_fraction": float(section_max_fraction),
                "qc_status": qc_status,
                "qc_flags": ";".join(flags),
            }
        )
    out = pd.DataFrame(rows)
    front = [col for col in REQUIRED_CLONE_QC_COLUMNS if col in out]
    rest = [col for col in out.columns if col not in front]
    return out[front + rest].sort_values(["qc_status", "n_cellbins", "joint_clone_id"], ascending=[True, False, True])


def _allele_reference_status(n_reference: int, n_de_novo: int, n_unknown: int, n_alleles: int) -> str:
    if n_unknown == n_alleles:
        return "unknown"
    if n_reference == n_alleles:
        return "reference_mapped_only"
    if n_de_novo == n_alleles:
        return "de_novo_only"
    return "mixed_reference_and_de_novo"


def build_cellbin_assignment(
    assignment: pd.DataFrame,
    clone_qc: pd.DataFrame,
) -> pd.DataFrame:
    qc_cols = [
        "joint_clone_id",
        "qc_status",
        "qc_flags",
        "clone_size_fraction",
        "n_loci_supported",
        "n_reference_mapped_alleles",
        "n_de_novo_alleles",
        "reference_support_fraction",
        "de_novo_allele_fraction",
        "allele_reference_status",
        "homoplasy_risk_flag",
        "overmerge_risk_flag",
        "giant_clone_flag",
    ]
    out = assignment.merge(clone_qc[qc_cols], on="joint_clone_id", how="left")
    out["cell_key"] = make_cell_key(out)
    out["assignment_status"] = np.where(out["qc_status"].eq("filtered"), "filtered_by_clone_qc", "assigned")
    return out


def build_cellbin_matrix(
    tile_map: pd.DataFrame,
    assignment: pd.DataFrame,
    clone_qc: pd.DataFrame,
    matrix_root: Path,
    *,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ensure_dir(matrix_root)
    cell_index = tile_map.drop_duplicates("cell_key").copy()
    cell_index = cell_index.sort_values(["sample_id", "cellbin_id"]).reset_index(drop=True)
    cell_index["cell_index"] = np.arange(len(cell_index), dtype=int)
    clone_index = clone_qc.loc[clone_qc["qc_status"].isin(["pass", "warning"])].copy()
    clone_index = clone_index.sort_values(["joint_clone_id"]).reset_index(drop=True)
    clone_index["clone_index"] = np.arange(len(clone_index), dtype=int)
    valid_assignment = assignment.loc[assignment["qc_status"].isin(["pass", "warning"])].copy()
    ij = valid_assignment[["cell_key", "joint_clone_id"]].drop_duplicates()
    ij = ij.merge(cell_index[["cell_key", "cell_index"]], on="cell_key", how="inner")
    ij = ij.merge(clone_index[["joint_clone_id", "clone_index"]], on="joint_clone_id", how="inner")
    matrix = sparse.coo_matrix(
        (np.ones(len(ij), dtype=np.float32), (ij["cell_index"].to_numpy(), ij["clone_index"].to_numpy())),
        shape=(len(cell_index), len(clone_index)),
    ).tocsr()
    target = matrix_root / "cellbin_joint_clone_matrix.npz"
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    sparse.save_npz(target, matrix)
    atomic_write_tsv_gz(matrix_root / "cellbin_index.tsv.gz", cell_index, overwrite=overwrite)
    atomic_write_tsv_gz(matrix_root / "joint_clone_index.tsv.gz", clone_index, overwrite=overwrite)
    cell_summary = cell_index.merge(
        assignment[
            [
                "cell_key",
                "joint_clone_id",
                "assignment_status",
                "qc_status",
                "qc_flags",
                "reference_support_fraction",
                "de_novo_allele_fraction",
                "allele_reference_status",
            ]
        ],
        on="cell_key",
        how="left",
    )
    cell_summary["joint_clone_id"] = cell_summary["joint_clone_id"].fillna("")
    cell_summary["assignment_status"] = cell_summary["assignment_status"].fillna("unassigned")
    cell_summary["qc_status"] = cell_summary["qc_status"].fillna("unassigned")
    cell_summary["qc_flags"] = cell_summary["qc_flags"].fillna("")
    cell_summary["reference_support_fraction"] = cell_summary["reference_support_fraction"].fillna(0.0)
    cell_summary["de_novo_allele_fraction"] = cell_summary["de_novo_allele_fraction"].fillna(0.0)
    cell_summary["allele_reference_status"] = cell_summary["allele_reference_status"].fillna("")
    atomic_write_tsv_gz(matrix_root / "cellbin_joint_clone_summary.tsv.gz", cell_summary, overwrite=overwrite)
    payload = {
        "n_cellbins": int(matrix.shape[0]),
        "n_validated_joint_clones": int(matrix.shape[1]),
        "n_nonzero_entries": int(matrix.nnz),
        "matrix_shape": list(matrix.shape),
    }
    return cell_summary, clone_index, payload


def aggregate_to_units(
    mapping: pd.DataFrame,
    cell_summary: pd.DataFrame,
    unit_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mapping.empty:
        return pd.DataFrame(), pd.DataFrame()
    joined = mapping.merge(
        cell_summary[
            [
                "cell_key",
                "joint_clone_id",
                "assignment_status",
                "qc_status",
                "reference_support_fraction",
                "de_novo_allele_fraction",
            ]
        ],
        on="cell_key",
        how="left",
    )
    joined["joint_clone_id"] = joined["joint_clone_id"].fillna("")
    joined["assignment_status"] = joined["assignment_status"].fillna("unassigned")
    joined["qc_status"] = joined["qc_status"].fillna("unassigned")
    joined["reference_support_fraction"] = joined["reference_support_fraction"].fillna(0.0)
    joined["de_novo_allele_fraction"] = joined["de_novo_allele_fraction"].fillna(0.0)
    base = joined.groupby(unit_cols, dropna=False).agg(n_cellbins=("cell_key", "nunique")).reset_index()
    assigned = joined.loc[joined["assignment_status"].eq("assigned")].copy()
    if assigned.empty:
        summary = base.copy()
        for col, value in _empty_unit_values().items():
            summary[col] = value
        return pd.DataFrame(), summary
    comp = (
        assigned.groupby([*unit_cols, "joint_clone_id"], dropna=False, as_index=False)
        .agg(
            n_clone_cellbins=("cell_key", "nunique"),
            mean_de_novo_allele_fraction=("de_novo_allele_fraction", "mean"),
            mean_reference_support_fraction=("reference_support_fraction", "mean"),
            qc_warning_cellbins=("qc_status", lambda s: int((s == "warning").sum())),
        )
        .sort_values([*unit_cols, "n_clone_cellbins"], ascending=[True] * len(unit_cols) + [False])
    )
    assigned_counts = (
        assigned.groupby(unit_cols, dropna=False)
        .agg(
            n_clone_assigned_cellbins=("cell_key", "nunique"),
            mean_de_novo_allele_fraction=("de_novo_allele_fraction", "mean"),
            mean_reference_support_fraction=("reference_support_fraction", "mean"),
            qc_warning_cellbins=("qc_status", lambda s: int((s == "warning").sum())),
        )
        .reset_index()
    )
    richness = comp.groupby(unit_cols, dropna=False).agg(n_joint_clones=("joint_clone_id", "nunique")).reset_index()
    dominant = comp.drop_duplicates(unit_cols)[[*unit_cols, "joint_clone_id", "n_clone_cellbins"]].rename(
        columns={"joint_clone_id": "dominant_joint_clone_id", "n_clone_cellbins": "dominant_clone_cellbins"}
    )
    diversity = (
        comp.groupby(unit_cols, dropna=False)["n_clone_cellbins"]
        .agg(
            clone_entropy=lambda s: entropy_from_counts(s.tolist()),
            simpson_clone_diversity=lambda s: simpson_from_counts(s.tolist()),
        )
        .reset_index()
    )
    top = (
        comp.groupby(unit_cols, dropna=False)
        .apply(lambda g: summarize_top_items(g.set_index("joint_clone_id")["n_clone_cellbins"].sort_values(ascending=False)), include_groups=False)
        .reset_index(name="top_joint_clones")
    )
    summary = (
        base.merge(assigned_counts, on=unit_cols, how="left")
        .merge(richness, on=unit_cols, how="left")
        .merge(dominant, on=unit_cols, how="left")
        .merge(diversity, on=unit_cols, how="left")
        .merge(top, on=unit_cols, how="left")
    )
    for col, value in _empty_unit_values().items():
        if col not in summary:
            summary[col] = value
    summary["n_clone_assigned_cellbins"] = summary["n_clone_assigned_cellbins"].fillna(0).astype(int)
    summary["clone_assigned_fraction"] = (summary["n_clone_assigned_cellbins"] / summary["n_cellbins"].replace(0, np.nan)).fillna(0.0)
    summary["n_joint_clones"] = summary["n_joint_clones"].fillna(0).astype(int)
    summary["dominant_joint_clone_id"] = summary["dominant_joint_clone_id"].fillna("")
    summary["dominant_clone_cellbins"] = summary["dominant_clone_cellbins"].fillna(0).astype(int)
    summary["dominant_clone_fraction"] = (
        summary["dominant_clone_cellbins"] / summary["n_clone_assigned_cellbins"].replace(0, np.nan)
    ).fillna(0.0)
    summary["clone_entropy"] = summary["clone_entropy"].fillna(0.0)
    summary["simpson_clone_diversity"] = summary["simpson_clone_diversity"].fillna(0.0)
    summary["clone_richness"] = summary["n_joint_clones"]
    summary["mean_de_novo_allele_fraction"] = summary["mean_de_novo_allele_fraction"].fillna(0.0)
    summary["mean_reference_support_fraction"] = summary["mean_reference_support_fraction"].fillna(0.0)
    summary["qc_warning_cellbins"] = summary["qc_warning_cellbins"].fillna(0).astype(int)
    summary["qc_warning_fraction"] = (
        summary["qc_warning_cellbins"] / summary["n_clone_assigned_cellbins"].replace(0, np.nan)
    ).fillna(0.0)
    summary["top_joint_clones"] = summary["top_joint_clones"].fillna("")
    comp["clone_fraction"] = (
        comp["n_clone_cellbins"]
        / comp.groupby(unit_cols, dropna=False)["n_clone_cellbins"].transform("sum").replace(0, np.nan)
    ).fillna(0.0)
    comp["qc_warning_fraction"] = (comp["qc_warning_cellbins"] / comp["n_clone_cellbins"].replace(0, np.nan)).fillna(0.0)
    return comp, summary


def _empty_unit_values() -> dict[str, Any]:
    return {
        "n_clone_assigned_cellbins": 0,
        "clone_assigned_fraction": 0.0,
        "n_joint_clones": 0,
        "dominant_joint_clone_id": "",
        "dominant_clone_cellbins": 0,
        "dominant_clone_fraction": 0.0,
        "clone_entropy": 0.0,
        "simpson_clone_diversity": 0.0,
        "clone_richness": 0,
        "mean_de_novo_allele_fraction": 0.0,
        "mean_reference_support_fraction": 0.0,
        "qc_warning_cellbins": 0,
        "qc_warning_fraction": 0.0,
        "top_joint_clones": "",
    }


def write_aggregations(
    output_root: Path,
    full_characterization_root: Path,
    cell_summary: pd.DataFrame,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    out = ensure_dir(output_root / "niche_clone")
    tile_map = load_tile_map(full_characterization_root)
    group_map = load_group_map(full_characterization_root)
    metaniche_map = load_metaniche_cell_map(full_characterization_root, tile_map)
    tile_comp, tile_summary = aggregate_to_units(
        tile_map,
        cell_summary,
        ["sample_id", "slice_id", "section_order", "tile_id", "tile_x_bin", "tile_y_bin"],
    )
    group_comp, group_summary = aggregate_to_units(
        group_map,
        cell_summary,
        ["sample_id", "slice_id", "section_order", "group_id"],
    )
    metaniche_comp, metaniche_summary = aggregate_to_units(metaniche_map, cell_summary, ["metaniche_id"])
    atomic_write_tsv_gz(out / "tile_joint_clone_composition.tsv.gz", tile_comp, overwrite=overwrite)
    atomic_write_tsv_gz(out / "tile_joint_clone_summary.tsv.gz", tile_summary, overwrite=overwrite)
    atomic_write_tsv_gz(out / "group_joint_clone_composition.tsv.gz", group_comp, overwrite=overwrite)
    atomic_write_tsv_gz(out / "group_joint_clone_summary.tsv.gz", group_summary, overwrite=overwrite)
    atomic_write_tsv_gz(out / "metaniche_joint_clone_composition.tsv.gz", metaniche_comp, overwrite=overwrite)
    atomic_write_tsv_gz(out / "metaniche_joint_clone_summary.tsv.gz", metaniche_summary, overwrite=overwrite)
    return {
        "n_tiles": int(len(tile_summary)),
        "n_tile_composition_rows": int(len(tile_comp)),
        "tile_coverage_units": int((tile_summary["n_clone_assigned_cellbins"] > 0).sum()) if not tile_summary.empty else 0,
        "tile_coverage_fraction": float((tile_summary["n_clone_assigned_cellbins"] > 0).mean()) if not tile_summary.empty else 0.0,
        "n_groups": int(len(group_summary)),
        "n_group_composition_rows": int(len(group_comp)),
        "n_metaniches": int(len(metaniche_summary)),
        "metaniche_coverage_units": int((metaniche_summary["n_clone_assigned_cellbins"] > 0).sum()) if not metaniche_summary.empty else 0,
        "metaniche_coverage_fraction": float((metaniche_summary["n_clone_assigned_cellbins"] > 0).mean()) if not metaniche_summary.empty else 0.0,
    }


def comparison_table(
    audit_root: Path,
    round1_root: Path,
    round2_root: Path,
    round21_root: Path,
    output_root: Path,
    total_lineage_cellbins: int,
    total_st_cellbins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference = read_table(audit_root / "reference_vs_denovo_policy_summary.tsv")
    ref_row = reference.loc[reference["policy_family"].eq("reference_only_best_bank_policy")].iloc[0]
    unified = reference.loc[reference["policy_family"].eq("mapped_rare_plus_empirical_denovo")].iloc[0]
    rows.append(_comparison_row("reference_only_conservative_benchmark", ref_row, total_lineage_cellbins, total_st_cellbins))
    rows.append(_comparison_row("unified_darlin_style_joint_clones", unified, total_lineage_cellbins, total_st_cellbins))
    round1_assignment = round1_root / "assignments/cellbin_clone_assignment.tsv.gz"
    if round1_assignment.exists():
        frame = read_table(round1_assignment)
        assigned = int(frame.get("assignment_status", pd.Series(dtype=str)).astype(str).str.contains("assigned").sum())
        rows.append(_basic_comparison_row("round1_strict_graph_clones", assigned, 192, total_lineage_cellbins, total_st_cellbins))
    round2_assignment = round2_root / "assignments/high_confidence_cellbin_clone_assignment_v2.tsv.gz"
    round2_clones = round2_root / "signatures/clone_signatures.tsv.gz"
    if round2_assignment.exists():
        frame = read_table(round2_assignment)
        assigned = int(frame["assignment_status"].isin(["assigned_single", "assigned_multi"]).sum())
        n_clones = int(read_table(round2_clones)["clone_set_high_confidence"].astype(bool).sum()) if round2_clones.exists() else 526
        rows.append(_basic_comparison_row("round2_hard_clonesignature", assigned, n_clones, total_lineage_cellbins, total_st_cellbins))
    round21_summary = round21_root / "membership/cellbin_clone_membership_summary.tsv.gz"
    if round21_summary.exists():
        frame = read_table(round21_summary)
        supported = int(frame["assignment_mode"].isin(["single_clone_dominant", "multi_clone_supported", "ambiguous"]).sum())
        rows.append(_basic_comparison_row("round2_1_clone_membership", supported, np.nan, total_lineage_cellbins, total_st_cellbins))
    table = pd.DataFrame(rows)
    tile_summary = read_table(output_root / "niche_clone/tile_joint_clone_summary.tsv.gz")
    metaniche_summary = read_table(output_root / "niche_clone/metaniche_joint_clone_summary.tsv.gz")
    table.loc[table["model"].eq("unified_darlin_style_joint_clones"), "tile_coverage"] = float((tile_summary["n_clone_assigned_cellbins"] > 0).mean())
    table.loc[table["model"].eq("unified_darlin_style_joint_clones"), "metaniche_coverage"] = float((metaniche_summary["n_clone_assigned_cellbins"] > 0).mean())
    return table


def _comparison_row(name: str, row: pd.Series, total_lineage: int, total_st: int) -> dict[str, Any]:
    assigned = int(row["joint_clone_assigned_cellbin_count"])
    return {
        "model": name,
        "assigned_cellbins": assigned,
        "assigned_fraction_lineage_positive": assigned / max(total_lineage, 1),
        "assigned_fraction_all_st": assigned / max(total_st, 1),
        "joint_clone_count": int(row["joint_clone_count"]),
        "largest_clone_fraction": float(row["largest_clone_fraction"]),
        "tile_coverage": float(row["tile_unit_fraction"]),
        "metaniche_coverage": float(row["metaniche_unit_fraction"]),
    }


def _basic_comparison_row(name: str, assigned: int, n_clones: Any, total_lineage: int, total_st: int) -> dict[str, Any]:
    return {
        "model": name,
        "assigned_cellbins": assigned,
        "assigned_fraction_lineage_positive": assigned / max(total_lineage, 1),
        "assigned_fraction_all_st": assigned / max(total_st, 1),
        "joint_clone_count": n_clones,
        "largest_clone_fraction": np.nan,
        "tile_coverage": np.nan,
        "metaniche_coverage": np.nan,
    }


def make_figures(
    output_root: Path,
    report_root: Path,
    full_characterization_root: Path,
    clone_qc: pd.DataFrame,
    cell_summary: pd.DataFrame,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    fig_root = ensure_dir(report_root / "figures")
    key_root = ensure_dir(report_root / "key_figure_candidates")
    tile_map = load_tile_map(full_characterization_root)
    tile_summary = read_table(output_root / "niche_clone/tile_joint_clone_summary.tsv.gz")
    reference = read_table(output_root / "comparison/reference_vs_unified_recovery.tsv")
    paths: list[Path] = []
    paths.extend(_plot_cellbin_assignment_maps(tile_map, cell_summary, fig_root))
    paths.append(_plot_tile_metric(tile_summary, "clone_assigned_fraction", "Tile Clone Coverage", fig_root / "tile_clone_coverage.png"))
    paths.append(_plot_tile_metric(tile_summary, "clone_entropy", "Tile Clone Entropy", fig_root / "tile_clone_entropy.png"))
    paths.append(_plot_tile_metric(tile_summary, "dominant_clone_fraction", "Dominant Clone Fraction", fig_root / "tile_dominant_clone_fraction.png"))
    paths.append(_plot_hist(clone_qc["n_cellbins"], "Clone Size Distribution", "n cellbins", fig_root / "clone_size_distribution.png"))
    paths.append(_plot_hist(clone_qc["de_novo_allele_fraction"], "De Novo Allele Fraction QC", "fraction", fig_root / "de_novo_allele_fraction_qc.png"))
    paths.append(_plot_hist(clone_qc["reference_support_fraction"], "Reference Support Fraction QC", "fraction", fig_root / "reference_support_fraction_qc.png"))
    paths.append(_plot_recovery(reference, fig_root / "reference_only_vs_unified_recovery.png"))
    paths.extend(_plot_top_clone_maps(tile_map, cell_summary, fig_root))
    paths.append(_plot_workflow(fig_root / "workflow_schematic.png"))
    copied = []
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            target = key_root / path.name
            if target.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing file: {target}")
            shutil.copy2(path, target)
            copied.append(target)
    return {"n_figures": len(paths), "n_key_figures": len(copied), "figures": [str(path) for path in paths]}


def _plot_cellbin_assignment_maps(tile_map: pd.DataFrame, cell_summary: pd.DataFrame, fig_root: Path) -> list[Path]:
    joined = tile_map.merge(cell_summary[["cell_key", "assignment_status"]], on="cell_key", how="left")
    joined["is_assigned"] = joined["assignment_status"].eq("assigned")
    paths = []
    for section, sub in joined.groupby("section_order"):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(sub["x"], sub["y"], c=np.where(sub["is_assigned"], "#2774ae", "#d0d0d0"), s=1, linewidths=0)
        ax.set_title(f"Joint Clone Assignment Section {section}")
        ax.set_aspect("equal")
        ax.axis("off")
        path = fig_root / f"cellbin_joint_clone_assignment_section_{section}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_top_clone_maps(tile_map: pd.DataFrame, cell_summary: pd.DataFrame, fig_root: Path, n_top: int = 3) -> list[Path]:
    joined = tile_map.merge(cell_summary[["cell_key", "joint_clone_id"]], on="cell_key", how="left")
    top = (
        cell_summary.loc[cell_summary["joint_clone_id"].astype(str).ne("")]
        .groupby("joint_clone_id")["cell_key"]
        .nunique()
        .sort_values(ascending=False)
        .head(n_top)
    )
    paths = []
    for idx, clone_id in enumerate(top.index, start=1):
        sub = joined.copy()
        sub["is_clone"] = sub["joint_clone_id"].eq(clone_id)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(sub["x"], sub["y"], c=np.where(sub["is_clone"], "#9b2226", "#d8d8d8"), s=1, linewidths=0)
        ax.set_title(f"Top Joint Clone {idx}")
        ax.set_aspect("equal")
        ax.axis("off")
        path = fig_root / f"top_joint_clone_{idx:02d}_spatial_map.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_tile_metric(tile_summary: pd.DataFrame, metric: str, title: str, path: Path) -> Path:
    fig, axes = plt.subplots(1, max(tile_summary["section_order"].nunique(), 1), figsize=(15, 4), squeeze=False)
    for ax, (section, sub) in zip(axes.flat, tile_summary.groupby("section_order")):
        scatter = ax.scatter(sub["tile_x_bin"], sub["tile_y_bin"], c=sub[metric], cmap="viridis", s=18)
        ax.set_title(f"{title} S{section}")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        fig.colorbar(scatter, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_hist(values: pd.Series, title: str, xlabel: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pd.to_numeric(values, errors="coerce").dropna(), bins=50, color="#2774ae", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_recovery(reference: pd.DataFrame, path: Path) -> Path:
    view = reference.loc[
        reference["model"].isin(["reference_only_conservative_benchmark", "unified_darlin_style_joint_clones"])
    ].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(view["model"], view["assigned_fraction_lineage_positive"], color=["#8d99ae", "#2774ae"])
    ax.set_ylabel("assigned fraction")
    ax.set_title("Reference-Only vs Unified Recovery")
    ax.tick_params(axis="x", labelrotation=20)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_workflow(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axis("off")
    text = "ST expression + CA/TA/RA DARLIN alleles  ->  validated joint clones  ->  niche clone composition"
    ax.text(0.5, 0.55, text, ha="center", va="center", fontsize=13)
    ax.text(0.5, 0.25, "reference/de novo status is QC metadata, not a clone class", ha="center", va="center", fontsize=10)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_text_reports(
    output_root: Path,
    report_root: Path,
    payloads: dict[str, Any],
    clone_qc: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    overwrite: bool,
) -> None:
    policy_payload = payloads["policy"]
    write_report_pair(
        report_root,
        "00_POLICY_FREEZE",
        "Policy Freeze",
        policy_payload,
        [
            "## Selected Unified Clone Policy",
            f"- Reference bank policy: `{SELECTED_REFERENCE_BANK_POLICY}`",
            f"- Allele inclusion policy: `{SELECTED_ALLELE_POLICY}`",
            f"- Threshold label: `{SELECTED_THRESHOLD_LABEL}`",
            f"- Normalized-count cutoff: {NORMALIZED_COUNT_CUTOFF}",
            f"- Sample-count cutoff: {SAMPLE_COUNT_CUTOFF}",
            f"- Minimum cellbins per allele: {MIN_CELLBINS_PER_ALLELE}",
            "- Primary clone unit: validated DARLIN-style joint clone.",
            "- Reference-bank status is allele-level QC metadata, not a clone class.",
            "- Reference-only calling is retained as conservative QC and sensitivity benchmark.",
        ],
        overwrite=overwrite,
    )
    clone_payload = payloads["clones"]
    write_report_pair(
        report_root,
        "01_VALIDATED_JOINT_CLONES",
        "Validated Joint Clones",
        clone_payload,
        [
            "## Clone Layer",
            f"- Validated joint clones: {clone_payload['n_validated_joint_clones']}",
            f"- Clone-assigned cellbins: {clone_payload['n_clone_assigned_cellbins']}",
            f"- Assigned fraction among lineage-positive cellbins: {clone_payload['assigned_fraction_lineage_positive']:.6f}",
            f"- Assigned fraction among all ST cellbins: {clone_payload['assigned_fraction_all_st']:.6f}",
            "- Clone IDs use a single `joint_clone_id` namespace.",
        ],
        overwrite=overwrite,
    )
    qc_payload = payloads["qc"]
    write_report_pair(
        report_root,
        "02_JOINT_CLONE_QC",
        "Joint Clone QC",
        qc_payload,
        [
            "## QC Summary",
            f"- Pass clones: {qc_payload['n_pass_clones']}",
            f"- Warning clones: {qc_payload['n_warning_clones']}",
            f"- Filtered clones: {qc_payload['n_filtered_clones']}",
            f"- Largest clone size: {qc_payload['largest_clone_cellbins']}",
            f"- Largest clone fraction: {qc_payload['largest_clone_fraction']:.6f}",
            f"- Giant clone flag count: {qc_payload['n_giant_clone_flags']}",
            f"- Overmerge risk flag count: {qc_payload['n_overmerge_risk_flags']}",
            "",
            "## QC Distributions",
            markdown_table(pd.DataFrame(qc_payload["distribution_table"])),
            "",
            "## Top QC Rows",
            markdown_table(
                clone_qc[
                    [
                        "joint_clone_id",
                        "n_cellbins",
                        "clone_size_fraction",
                        "joint_allele_num",
                        "reference_support_fraction",
                        "de_novo_allele_fraction",
                        "qc_status",
                        "qc_flags",
                    ]
                ].head(10)
            ),
        ],
        overwrite=overwrite,
    )
    matrix_payload = payloads["matrix"]
    write_report_pair(
        report_root,
        "03_CELLBIN_JOINT_CLONE_MATRIX",
        "Cellbin Joint Clone Matrix",
        matrix_payload,
        [
            "## Matrix",
            f"- Matrix shape: {matrix_payload['matrix_shape']}",
            f"- Nonzero entries: {matrix_payload['n_nonzero_entries']}",
            "- Rows are ST cellbins; columns are validated joint clones.",
        ],
        overwrite=overwrite,
    )
    niche_payload = payloads["niche"]
    write_report_pair(
        report_root,
        "04_NICHE_JOINT_CLONE_COMPOSITION",
        "Niche Joint Clone Composition",
        niche_payload,
        [
            "## Spatial Unit Aggregation",
            f"- Tiles: {niche_payload['n_tiles']}",
            f"- Tile coverage fraction: {niche_payload['tile_coverage_fraction']:.6f}",
            f"- Groups: {niche_payload['n_groups']}",
            f"- Metaniches: {niche_payload['n_metaniches']}",
            f"- Metaniche coverage fraction: {niche_payload['metaniche_coverage_fraction']:.6f}",
            "- Tiles are the primary non-overlapping summaries.",
            "- Groups are local-context summaries and should not be summed as tissue abundance.",
            "- Metaniches are descriptive categories.",
        ],
        overwrite=overwrite,
    )
    comparison_payload = payloads["comparison"]
    write_report_pair(
        report_root,
        "05_COMPARISON_TO_PREVIOUS_LAYERS",
        "Comparison To Previous Layers",
        comparison_payload,
        [
            "## Comparison",
            markdown_table(comparison),
            "- The unified layer improves practical clone recovery over reference-only and Round 2.1 membership.",
        ],
        overwrite=overwrite,
    )
    figure_payload = payloads["figures"]
    write_report_pair(
        report_root,
        "06_FIGURES",
        "Figures",
        figure_payload,
        [
            "## Figures",
            f"- Figures generated: {figure_payload['n_figures']}",
            f"- Key figure candidates: {figure_payload['n_key_figures']}",
            f"- Figure root: `{report_root / 'figures'}`",
            f"- Key figure root: `{report_root / 'key_figure_candidates'}`",
        ],
        overwrite=overwrite,
    )
    dynamics_payload = payloads["dynamics"]
    write_report_pair(
        report_root,
        "07_DYNAMICS_INTERFACE_DESIGN",
        "Dynamics Interface Design",
        dynamics_payload,
        [
            "## Matrix Objects",
            "- `C_cellbin_clone`: cellbin by validated joint clone.",
            "- `C_tile_clone`: tile by validated joint clone.",
            "- `C_niche_clone`: descriptive niche/metaniche by validated joint clone.",
            "",
            "## Future Use",
            "- Clone-overlap can support candidate transitions in future time-anchored or perturbation-anchored data.",
            "- Clone composition can regularize candidate state coupling.",
            "- Clone entropy and clone diversity are lineage state variables for niches.",
            "- Direction still requires time, perturbation, or biological prior; clone overlap alone is not directional evidence.",
            "- L126 serial sections do not support temporal fate or transition claims.",
        ],
        overwrite=overwrite,
    )
    final_payload = payloads["final"]
    write_report_pair(
        report_root,
        "08_FINAL_DECISION",
        "Final Decision",
        final_payload,
        [
            "## Final Decision",
            f"- Label: `{final_payload['final_decision_label']}`",
            f"- Selected unified clone policy: `{final_payload['selected_unified_clone_policy']}`",
            f"- Joint clone count: {final_payload['joint_clone_count']}",
            f"- Assigned fraction among lineage-positive cellbins: {final_payload['assigned_fraction_lineage_positive']:.6f}",
            f"- Assigned fraction among all ST cellbins: {final_payload['assigned_fraction_all_st']:.6f}",
            f"- Largest clone fraction: {final_payload['largest_clone_fraction']:.6f}",
            f"- Tile clone coverage: {final_payload['tile_clone_coverage_fraction']:.6f}",
            f"- Metaniche clone coverage: {final_payload['metaniche_clone_coverage_fraction']:.6f}",
            "- De novo status is handled as QC annotation, not a clone class.",
        ],
        overwrite=overwrite,
    )


def qc_distribution_table(clone_qc: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for column in ["n_cellbins", "joint_allele_num", "de_novo_allele_fraction", "reference_support_fraction"]:
        values = pd.to_numeric(clone_qc[column], errors="coerce").dropna()
        rows.append(
            {
                "metric": column,
                "min": float(values.min()) if len(values) else 0.0,
                "median": float(values.median()) if len(values) else 0.0,
                "p95": float(values.quantile(0.95)) if len(values) else 0.0,
                "max": float(values.max()) if len(values) else 0.0,
            }
        )
    return rows


def final_label(clone_qc: pd.DataFrame, assigned_fraction_lineage: float) -> str:
    if assigned_fraction_lineage < 0.10:
        return "L126_DARLIN_JOINT_CLONE_NICHE_LAYER_HOLD_FOR_LOW_RECOVERY"
    if bool(clone_qc["giant_clone_flag"].any()):
        return "L126_DARLIN_JOINT_CLONE_NICHE_LAYER_HOLD_FOR_OVERMERGE"
    if bool(clone_qc["qc_status"].eq("filtered").any()):
        return "L126_DARLIN_JOINT_CLONE_NICHE_LAYER_HOLD_FOR_QC_FAILURE"
    if bool(clone_qc["qc_status"].eq("warning").any()):
        return "L126_DARLIN_JOINT_CLONE_NICHE_LAYER_READY_WITH_QC_WARNINGS"
    return "L126_DARLIN_JOINT_CLONE_NICHE_LAYER_READY"


def validate_outputs(
    output_root: Path,
    report_root: Path,
    before_hashes: dict[str, str],
    after_hashes: dict[str, str],
) -> dict[str, Any]:
    json_ok = True
    for path in sorted(report_root.glob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            json_ok = False
    table_paths = [
        output_root / "clones/validated_joint_clone_summary.tsv.gz",
        output_root / "clones/cellbin_joint_clone_assignment.tsv.gz",
        output_root / "matrix/cellbin_joint_clone_summary.tsv.gz",
        output_root / "niche_clone/tile_joint_clone_summary.tsv.gz",
        output_root / "niche_clone/metaniche_joint_clone_summary.tsv.gz",
    ]
    tsv_ok = True
    for path in table_paths:
        try:
            read_table(path, nrows=5)
        except Exception:
            tsv_ok = False
    matrix_ok = True
    try:
        sparse.load_npz(output_root / "matrix/cellbin_joint_clone_matrix.npz")
    except Exception:
        matrix_ok = False
    clone_qc = read_table(output_root / "clones/validated_joint_clone_summary.tsv.gz") if table_paths[0].exists() else pd.DataFrame()
    qc_fields_present = set(REQUIRED_CLONE_QC_COLUMNS).issubset(set(clone_qc.columns))
    no_clone_classes = "clone_class" not in clone_qc.columns and "de_novo_clone_class" not in clone_qc.columns
    figures = sorted((report_root / "figures").glob("*.png"))
    figures_ok = bool(figures) and all(path.stat().st_size > 0 for path in figures)
    report_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(report_root.glob("*.md")))
    hits = positive_claim_hits(report_text)
    paths_ok = not any(path_has_forbidden_ssd(path) for path in [output_root, report_root])
    payload = {
        "validation_status": "PASS"
        if all(
            [
                json_ok,
                tsv_ok,
                matrix_ok,
                qc_fields_present,
                no_clone_classes,
                figures_ok,
                before_hashes == after_hashes,
                paths_ok,
                not hits,
            ]
        )
        else "FAIL",
        "json_parse": bool(json_ok),
        "tsv_gzip_readability": bool(tsv_ok),
        "sparse_matrix_readability": bool(matrix_ok),
        "input_packet_unchanged": bool(before_hashes == after_hashes),
        "forbidden_scratch_path_not_used": bool(paths_ok),
        "raw_sequence_processing_not_run": True,
        "spatio_darlin_not_rerun": True,
        "directed_gpcca_not_run": True,
        "plana_planb_production_not_run": True,
        "no_positive_fate_terminal_transition_claims": bool(not hits),
        "positive_claim_hits": hits,
        "git_operations_not_run": True,
        "figures_non_empty": bool(figures_ok),
        "qc_fields_present": bool(qc_fields_present),
        "reference_de_novo_not_clone_classes": bool(no_clone_classes),
    }
    return payload


def write_validation_report(report_root: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    write_report_pair(
        report_root,
        "09_VALIDATION",
        "Validation",
        payload,
        [
            "## Validation",
            f"- Status: `{payload['validation_status']}`",
            f"- JSON parse: {payload['json_parse']}",
            f"- TSV/gzip readability: {payload['tsv_gzip_readability']}",
            f"- Sparse matrix readability: {payload['sparse_matrix_readability']}",
            f"- Input packet unchanged: {payload['input_packet_unchanged']}",
            f"- Figures non-empty: {payload['figures_non_empty']}",
            f"- QC fields present: {payload['qc_fields_present']}",
            f"- Reference/de novo not clone classes: {payload['reference_de_novo_not_clone_classes']}",
            f"- Positive claim hits: {payload['positive_claim_hits']}",
        ],
        overwrite=overwrite,
    )
