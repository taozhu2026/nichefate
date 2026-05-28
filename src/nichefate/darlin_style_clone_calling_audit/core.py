from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nichefate.darlin_clone_signature.common import CELL_COLUMNS, make_cell_key
from nichefate.darlin_clone_signature.reporting import positive_claim_hits, read_table


LOCI = ("CA", "TA", "RA")
JOINT_LOCUS_ORDER = ("CA", "RA", "TA")
BANK_POLICIES = ("plain", "gr", "union")
MAPPING_MODES = ("raw_exact", "normalized", "normalized_locus_formatted")
DE_NOVO_POLICIES = ("mapped_rare_only", "mapped_rare_plus_low_frequency_denovo", "mapped_rare_plus_empirical_denovo")


@dataclass(frozen=True)
class ThresholdSpec:
    label: str
    prob_cutoff: float
    sample_count_cutoff: int
    min_cellbins_per_allele: int


DEFAULT_THRESHOLD = ThresholdSpec("tutorial_like", 0.1, 2, 1)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def normalize_allele_string(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip().strip('"').strip("'").replace(" ", "")
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    if text == "[]":
        return "[]"
    tokens = [token.strip() for token in re.split(r"[,;]", text) if token.strip()]
    if not tokens:
        return ""
    deduped = sorted(set(tokens), key=_mutation_sort_key)
    return ",".join(deduped)


def _mutation_sort_key(token: str) -> tuple[int, int, str]:
    match = re.match(r"^(\d+)(?:_(\d+))?", token)
    if not match:
        return (10**9, 10**9, token)
    return (int(match.group(1)), int(match.group(2) or match.group(1)), token)


def raw_clean_allele(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip().strip('"').strip("'").replace(" ", "")
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def locus_format(locus: object, allele_normalized: object) -> str:
    allele = str(allele_normalized or "")
    locus_text = str(locus)
    if not allele:
        return ""
    if allele.startswith(f"{locus_text}_"):
        return allele
    return f"{locus_text}_{allele}"


def allele_schema_ok(allele_normalized: str) -> bool:
    if not allele_normalized or allele_normalized == "[]":
        return False
    return all(bool(re.match(r"^\d+_\d+", token)) for token in allele_normalized.split(","))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_mosaiclineage(mosaic_root: Path, reference_root: Path) -> dict[str, Any]:
    source_file = mosaic_root / "mosaiclineage/DARLIN.py"
    import_status: dict[str, Any] = {}
    spec = importlib.util.find_spec("mosaiclineage")
    import_status["installed_importable"] = bool(spec)
    import_status["installed_origin"] = spec.origin if spec else ""
    try:
        import mosaiclineage.DARLIN as darlin  # type: ignore

        import inspect

        import_status["direct_darlin_importable"] = True
        import_status["direct_import_error"] = ""
        import_status["signature"] = str(inspect.signature(darlin.assign_clone_id_by_integrating_locus))
    except Exception as exc:  # pragma: no cover - depends on local env
        import_status["direct_darlin_importable"] = False
        import_status["direct_import_error"] = f"{type(exc).__name__}: {exc}"
        import_status["signature"] = (
            "assign_clone_id_by_integrating_locus(df_sc_CARLIN_raw, prob_cutoff=0.1, "
            "sample_count_cutoff=2, joint_allele_N_cutoff=6, locus_list=['CA','TA','RA'], clone_key='allele')"
        )
    source_text = source_file.read_text(encoding="utf-8") if source_file.exists() else ""
    function_found = "def assign_clone_id_by_integrating_locus(" in source_text
    reference_files = {
        f"{locus}_{kind}": str(reference_root / f"reference_merged_alleles_{locus}{kind}.csv")
        for locus in LOCI
        for kind in ("", "_Gr")
    }
    missing_refs = [path for path in reference_files.values() if not Path(path).exists()]
    if not function_found:
        label = "MOSAICLINEAGE_FUNCTION_MISSING"
    elif missing_refs:
        label = "MOSAICLINEAGE_REFERENCE_MISSING"
    else:
        label = "MOSAICLINEAGE_AVAILABLE"
    return {
        "decision_label": label,
        "mosaiclineage_root": str(mosaic_root),
        "source_file": str(source_file),
        "source_function_found": function_found,
        "reference_root": str(reference_root),
        "reference_files": reference_files,
        "missing_reference_files": missing_refs,
        "import_status": import_status,
        "signature": import_status["signature"],
        "required_input_columns": ["RNA_id", "locus", "allele", "normalized_count", "sample_count"],
        "required_packages_in_function_body": ["numpy", "pandas", "scipy.sparse.csgraph.connected_components", "scanpy(imported but not used by v0)", "tqdm(imported for progress only)"],
        "bio_needed_by_function_body": False,
        "source_adapter_strategy": "faithful_local_reimplementation_if_direct_import_blocked_by_unrelated_package_dependency",
        "joint_clone_id_tmp_rule": "join locus-specific allele values in CA/RA/TA order with missing values represented as nan",
        "joint_prob_rule": "product of locus-specific normalized_count values, with missing loci treated as probability 1",
        "joint_allele_num_rule": "number of unique locus-specific alleles in a connected component",
        "ambiguous_clone_filtering_rule": "alleles co-detected with at least joint_allele_N_cutoff distinct joint alleles have effective probability set to the probability cutoff and do not create strong links",
    }


def load_reference_banks(reference_root: Path) -> dict[str, pd.DataFrame]:
    plain = _load_reference_policy(reference_root, "")
    gr = _load_reference_policy(reference_root, "_Gr")
    union = pd.concat([plain.assign(source_bank="plain"), gr.assign(source_bank="gr")], ignore_index=True)
    union = union.sort_values(["locus", "allele_normalized", "invalid_alleles", "normalized_count", "sample_count", "source_bank"])
    union = union.drop_duplicates(["locus", "allele_normalized"], keep="first").reset_index(drop=True)
    union["reference_bank_policy"] = "union"
    return {"plain": plain, "gr": gr, "union": union}


def _load_reference_policy(reference_root: Path, suffix: str) -> pd.DataFrame:
    rows = []
    policy = "gr" if suffix == "_Gr" else "plain"
    for locus in LOCI:
        path = reference_root / f"reference_merged_alleles_{locus}{suffix}.csv"
        frame = pd.read_csv(path)
        frame["locus"] = locus
        frame["reference_path"] = str(path)
        frame["reference_bank_policy"] = policy
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"allele": "reference_allele"})
    if "invalid_alleles" not in out:
        out["invalid_alleles"] = False
    out["invalid_alleles"] = out["invalid_alleles"].fillna(False).astype(bool)
    out["sample_count"] = pd.to_numeric(out["sample_count"], errors="coerce").fillna(np.inf)
    out["normalized_count"] = pd.to_numeric(out["normalized_count"], errors="coerce").fillna(1.0)
    out["reference_allele_raw_clean"] = out["reference_allele"].map(raw_clean_allele)
    out["allele_normalized"] = out["reference_allele"].map(normalize_allele_string)
    out["allele_locus_formatted"] = [locus_format(locus, allele) for locus, allele in zip(out["locus"], out["allele_normalized"])]
    return out.reset_index(drop=True)


def build_cellbin_allele_table(lineage_path: Path, annotation_path: Path) -> pd.DataFrame:
    lineage = read_table(lineage_path)
    annotation = read_table(annotation_path)
    join_cols = ["sample_id", "slice_id", "section_order", "assay", "feature_id"]
    annotation = annotation.drop_duplicates(join_cols + ["allele", "allele_index"]).copy()
    table = lineage.merge(
        annotation[
            [
                *join_cols,
                "allele",
                "allele_index",
                "n_alleles_for_feature",
                "allele_is_missing",
                "source_row_index",
            ]
        ],
        on=join_cols,
        how="left",
        suffixes=("", "_annotation"),
    )
    table["RNA_id"] = table["sample_id"].astype(str) + "|" + table["slice_id"].astype(str) + "|" + table["cellbin_id"].astype(str)
    table["locus"] = table["assay"].astype(str)
    table["assay_scoped_feature_id"] = table["assay"].astype(str) + "::" + table["feature_id"].astype(str)
    table["allele_original"] = table["allele"].fillna("")
    table["allele_raw_clean"] = table["allele_original"].map(raw_clean_allele)
    table["allele_normalized"] = table["allele_original"].map(normalize_allele_string)
    table["allele_locus_formatted"] = [locus_format(locus, allele) for locus, allele in zip(table["locus"], table["allele_normalized"])]
    table["allele_is_missing"] = table["allele_is_missing"].fillna(True).astype(bool)
    table["count"] = pd.to_numeric(table["count"], errors="coerce").fillna(0.0)
    table["n_alleles_for_feature"] = pd.to_numeric(table["n_alleles_for_feature"], errors="coerce").fillna(1).astype(int)
    columns = [
        "RNA_id",
        "sample_id",
        "slice_id",
        "section_order",
        "cellbin_id",
        "x",
        "y",
        "locus",
        "allele_original",
        "allele_raw_clean",
        "allele_normalized",
        "allele_locus_formatted",
        "allele_is_missing",
        "count",
        "feature_id",
        "assay_scoped_feature_id",
        "feature_row_index",
        "allele_index",
        "n_alleles_for_feature",
    ]
    return table[columns].sort_values(["sample_id", "cellbin_id", "locus", "feature_id"]).reset_index(drop=True)


def compare_reference_policies(
    allele_table: pd.DataFrame,
    banks: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    mapped_tables: dict[str, pd.DataFrame] = {}
    for policy, ref in banks.items():
        policy_mode_tables = {}
        for mode in MAPPING_MODES:
            mapped = map_alleles_to_reference(allele_table, ref, policy, mode)
            policy_mode_tables[mode] = mapped
            summary_rows.extend(_mapping_summary_rows(mapped, policy, mode))
        summary_frame = pd.DataFrame([row for row in summary_rows if row["reference_bank_policy"] == policy])
        selected_mode = _select_mapping_mode(summary_frame)
        mapped_tables[policy] = policy_mode_tables[selected_mode].assign(selected_mapping_mode=selected_mode)
    full_mapped = pd.concat(mapped_tables.values(), ignore_index=True)
    return pd.DataFrame(summary_rows), mapped_tables, full_mapped


def map_alleles_to_reference(allele_table: pd.DataFrame, reference: pd.DataFrame, policy: str, mode: str) -> pd.DataFrame:
    if mode == "raw_exact":
        left_key = "allele_raw_clean"
        right_key = "reference_allele_raw_clean"
    elif mode == "normalized":
        left_key = "allele_normalized"
        right_key = "allele_normalized"
    elif mode == "normalized_locus_formatted":
        left_key = "allele_locus_formatted"
        right_key = "allele_locus_formatted"
    else:
        raise ValueError(f"Unsupported mapping mode: {mode}")
    ref_cols = [
        "locus",
        right_key,
        "reference_allele",
        "observed_count",
        "sample_count",
        "normalized_count",
        "smoothed_homoplasy",
        "invalid_alleles",
        "reference_bank_policy",
    ]
    ref_small = reference.copy()
    for col in ref_cols:
        if col not in ref_small.columns:
            ref_small[col] = pd.NA
    ref_small = ref_small[ref_cols].drop_duplicates(["locus", right_key]).copy()
    mapped = allele_table.merge(ref_small, left_on=["locus", left_key], right_on=["locus", right_key], how="left", suffixes=("", "_reference"))
    mapped["reference_bank_policy"] = policy
    mapped["mapping_mode"] = mode
    mapped["reference_mapped"] = mapped["reference_allele"].notna()
    mapped["invalid_alleles"] = mapped["invalid_alleles"].map(lambda value: bool(value) if not pd.isna(value) else False)
    mapped["sample_count"] = pd.to_numeric(mapped["sample_count"], errors="coerce")
    mapped["normalized_count"] = pd.to_numeric(mapped["normalized_count"], errors="coerce")
    return add_empirical_allele_stats(mapped)


def add_empirical_allele_stats(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    total_cellbins = max(out["RNA_id"].nunique(), 1)
    nonmissing = out.loc[~out["allele_is_missing"].astype(bool)].copy()
    stats = (
        nonmissing.groupby(["locus", "allele_normalized"], as_index=False)
        .agg(
            empirical_n_cellbins=("RNA_id", "nunique"),
            empirical_total_count=("count", "sum"),
            empirical_section_distribution=("section_order", lambda s: ";".join(f"{k}:{v}" for k, v in s.value_counts().sort_index().items())),
        )
        if not nonmissing.empty
        else pd.DataFrame(columns=["locus", "allele_normalized"])
    )
    stats["empirical_cellbin_fraction"] = stats["empirical_n_cellbins"] / total_cellbins if not stats.empty else []
    out = out.merge(stats, on=["locus", "allele_normalized"], how="left")
    out["empirical_n_cellbins"] = out["empirical_n_cellbins"].fillna(0).astype(int)
    out["empirical_total_count"] = out["empirical_total_count"].fillna(0.0)
    out["empirical_cellbin_fraction"] = out["empirical_cellbin_fraction"].fillna(0.0)
    out["empirical_section_distribution"] = out["empirical_section_distribution"].fillna("")
    return out


def _mapping_summary_rows(mapped: pd.DataFrame, policy: str, mode: str) -> list[dict[str, Any]]:
    rows = []
    for locus in LOCI:
        sub = mapped.loc[mapped["locus"].eq(locus)]
        nonmissing = sub.loc[~sub["allele_is_missing"].astype(bool)]
        mapped_nonmissing = nonmissing.loc[nonmissing["reference_mapped"]]
        before_cells = nonmissing["RNA_id"].nunique()
        rows.append(
            {
                "reference_bank_policy": policy,
                "mapping_mode": mode,
                "locus": locus,
                "n_allele_rows_total": int(len(sub)),
                "n_nonmissing_rows": int(len(nonmissing)),
                "n_unique_nonmissing_alleles": int(nonmissing["allele_normalized"].nunique()),
                "n_unique_mapped_alleles": int(mapped_nonmissing["allele_normalized"].nunique()),
                "unique_mapping_fraction": float(mapped_nonmissing["allele_normalized"].nunique() / max(nonmissing["allele_normalized"].nunique(), 1)),
                "row_mapping_fraction": float(len(mapped_nonmissing) / max(len(nonmissing), 1)),
                "n_invalid_mapped_rows": int(mapped_nonmissing["invalid_alleles"].sum()),
                "n_unmapped_rows": int((~nonmissing["reference_mapped"]).sum()),
                "n_unmapped_unique_alleles": int(nonmissing.loc[~nonmissing["reference_mapped"], "allele_normalized"].nunique()),
                "cellbins_before_mapping": int(before_cells),
                "cellbins_with_mapped_allele": int(mapped_nonmissing["RNA_id"].nunique()),
                "cellbin_mapping_fraction": float(mapped_nonmissing["RNA_id"].nunique() / max(before_cells, 1)),
                "normalized_count_median": float(mapped_nonmissing["normalized_count"].median()) if not mapped_nonmissing.empty else 0.0,
                "sample_count_median": float(mapped_nonmissing["sample_count"].median()) if not mapped_nonmissing.empty else 0.0,
            }
        )
    return rows


def _select_mapping_mode(summary: pd.DataFrame) -> str:
    by_mode = (
        summary.groupby("mapping_mode", as_index=False)
        .agg(row_mapping_fraction=("row_mapping_fraction", "mean"), unique_mapping_fraction=("unique_mapping_fraction", "mean"))
        .sort_values(["row_mapping_fraction", "unique_mapping_fraction"], ascending=False)
    )
    if by_mode.empty:
        return "normalized"
    return str(by_mode.iloc[0]["mapping_mode"])


def classify_alleles_for_policy(
    mapped: pd.DataFrame,
    threshold: ThresholdSpec,
    de_novo_policy: str,
    *,
    de_novo_low_frequency_threshold: float = 0.001,
    de_novo_empirical_threshold: float = 0.005,
) -> pd.DataFrame:
    out = mapped.copy()
    out["allele_class"] = "unmapped_schema_mismatch"
    out.loc[out["allele_is_missing"].astype(bool), "allele_class"] = "unusable_missing_allele"
    mapped_idx = out["reference_mapped"].astype(bool) & ~out["allele_is_missing"].astype(bool)
    rare_idx = (
        mapped_idx
        & ~out["invalid_alleles"].astype(bool)
        & out["normalized_count"].astype(float).lt(threshold.prob_cutoff)
        & out["sample_count"].astype(float).lt(threshold.sample_count_cutoff)
        & out["empirical_n_cellbins"].astype(int).ge(threshold.min_cellbins_per_allele)
    )
    out.loc[mapped_idx, "allele_class"] = "reference_mapped_common_or_invalid"
    out.loc[rare_idx, "allele_class"] = "reference_mapped_rare"
    unmapped_nonmissing = ~out["reference_mapped"].astype(bool) & ~out["allele_is_missing"].astype(bool)
    schema_ok = out["allele_normalized"].map(allele_schema_ok)
    out.loc[unmapped_nonmissing & schema_ok, "allele_class"] = "unmapped_de_novo_candidate"
    out["valid_for_joint_calling"] = out["allele_class"].eq("reference_mapped_rare")
    if de_novo_policy == "mapped_rare_plus_low_frequency_denovo":
        de_novo_idx = (
            out["allele_class"].eq("unmapped_de_novo_candidate")
            & out["empirical_cellbin_fraction"].astype(float).le(de_novo_low_frequency_threshold)
            & out["empirical_n_cellbins"].astype(int).ge(threshold.min_cellbins_per_allele)
        )
        out.loc[de_novo_idx, "valid_for_joint_calling"] = True
    elif de_novo_policy == "mapped_rare_plus_empirical_denovo":
        de_novo_idx = (
            out["allele_class"].eq("unmapped_de_novo_candidate")
            & out["empirical_cellbin_fraction"].astype(float).le(de_novo_empirical_threshold)
            & out["empirical_n_cellbins"].astype(int).ge(threshold.min_cellbins_per_allele)
        )
        out.loc[de_novo_idx, "valid_for_joint_calling"] = True
    elif de_novo_policy != "mapped_rare_only":
        raise ValueError(f"Unsupported de novo policy: {de_novo_policy}")
    out["normalized_count_for_calling"] = out["normalized_count"]
    out["sample_count_for_calling"] = out["sample_count"]
    de_novo_valid = out["valid_for_joint_calling"] & out["allele_class"].eq("unmapped_de_novo_candidate")
    out.loc[de_novo_valid, "normalized_count_for_calling"] = out.loc[de_novo_valid, "empirical_cellbin_fraction"].clip(lower=1e-8)
    out.loc[de_novo_valid, "sample_count_for_calling"] = 1
    out["mosaic_allele"] = out["allele_locus_formatted"]
    return out


def summarize_filtering(classified: pd.DataFrame, bank_policy: str, de_novo_policy: str, threshold: ThresholdSpec) -> dict[str, Any]:
    valid = classified.loc[classified["valid_for_joint_calling"].astype(bool)]
    loci_by_cell = valid.groupby("RNA_id")["locus"].nunique() if not valid.empty else pd.Series(dtype=int)
    total_cellbins = max(classified["RNA_id"].nunique(), 1)
    row: dict[str, Any] = {
        "reference_bank_policy": bank_policy,
        "de_novo_policy": de_novo_policy,
        "threshold_label": threshold.label,
        "normalized_count_cutoff": threshold.prob_cutoff,
        "sample_count_cutoff": threshold.sample_count_cutoff,
        "min_cellbins_per_allele": threshold.min_cellbins_per_allele,
        "n_valid_allele_rows": int(len(valid)),
        "n_valid_unique_alleles": int(valid[["locus", "allele_normalized"]].drop_duplicates().shape[0]) if not valid.empty else 0,
        "n_valid_allele_supported_cellbins": int(valid["RNA_id"].nunique()),
        "valid_allele_supported_cellbin_fraction": float(valid["RNA_id"].nunique() / total_cellbins),
        "n_cellbins_with_valid_alleles_ge2_loci": int((loci_by_cell >= 2).sum()),
        "fraction_cellbins_with_valid_alleles_ge2_loci": float((loci_by_cell >= 2).sum() / total_cellbins),
    }
    for locus in LOCI:
        sub = valid.loc[valid["locus"].eq(locus)]
        row[f"n_valid_{locus}_alleles"] = int(sub["allele_normalized"].nunique())
        row[f"n_valid_{locus}_cellbins"] = int(sub["RNA_id"].nunique())
    for allele_class, count in classified["allele_class"].value_counts().items():
        row[f"n_rows_{allele_class}"] = int(count)
    return row


def collapse_for_mosaiclineage(classified: pd.DataFrame) -> pd.DataFrame:
    valid = classified.loc[classified["valid_for_joint_calling"].astype(bool)].copy()
    if valid.empty:
        return pd.DataFrame()
    priority = {
        "reference_mapped_rare": 0,
        "unmapped_de_novo_candidate": 1,
    }
    valid["allele_class_priority"] = valid["allele_class"].map(priority).fillna(9)
    valid["n_candidate_alleles_per_cell_locus"] = valid.groupby(["RNA_id", "locus"])["mosaic_allele"].transform("nunique")
    valid = valid.sort_values(
        [
            "RNA_id",
            "locus",
            "allele_class_priority",
            "normalized_count_for_calling",
            "count",
            "mosaic_allele",
        ],
        ascending=[True, True, True, True, False, True],
    )
    valid = valid.drop_duplicates(["RNA_id", "locus"], keep="first").copy()
    out = valid[
        [
            "RNA_id",
            "sample_id",
            "slice_id",
            "section_order",
            "cellbin_id",
            "locus",
            "mosaic_allele",
            "allele_original",
            "allele_normalized",
            "allele_class",
            "normalized_count_for_calling",
            "sample_count_for_calling",
            "count",
            "n_candidate_alleles_per_cell_locus",
        ]
    ].rename(
        columns={
            "mosaic_allele": "allele",
            "normalized_count_for_calling": "normalized_count",
            "sample_count_for_calling": "sample_count",
        }
    )
    return out.reset_index(drop=True)


def assign_joint_clones(
    collapsed: pd.DataFrame,
    *,
    prob_cutoff: float = 0.1,
    sample_count_cutoff: int = 2,
    joint_allele_N_cutoff: int = 6,
    locus_list: tuple[str, ...] = JOINT_LOCUS_ORDER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"RNA_id", "locus", "allele", "normalized_count", "sample_count"}
    missing = sorted(required - set(collapsed.columns))
    if missing:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"status": "schema_mismatch", "missing_columns": missing}
    filtered = collapsed.loc[
        collapsed["normalized_count"].astype(float).lt(prob_cutoff)
        & collapsed["sample_count"].astype(float).lt(sample_count_cutoff)
    ].copy()
    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"status": "empty_after_filtering"}
    filtered = filtered.drop_duplicates(["RNA_id", "locus"], keep="first")
    allele_wide = filtered.pivot(index="RNA_id", columns="locus", values="allele")
    prob_wide = filtered.pivot(index="RNA_id", columns="locus", values="normalized_count")
    for locus in locus_list:
        if locus not in allele_wide:
            allele_wide[locus] = np.nan
        if locus not in prob_wide:
            prob_wide[locus] = np.nan
    df_cells = pd.DataFrame(index=allele_wide.index)
    for locus in locus_list:
        df_cells[f"{locus}_BC"] = allele_wide[locus]
        df_cells[f"{locus}_prob"] = prob_wide[locus]
    locus_bc_cols = [f"{locus}_BC" for locus in locus_list]
    locus_prob_cols = [f"{locus}_prob" for locus in locus_list]
    df_cells["joint_clone_id_tmp"] = ["@".join(row) for row in df_cells[locus_bc_cols].fillna("nan").astype(str).to_numpy()]
    df_allele = df_cells[locus_bc_cols + ["joint_clone_id_tmp"]].drop_duplicates().reset_index(drop=True)
    allele_to_norm = filtered.groupby("allele")["normalized_count"].min().astype(float).to_dict()

    for locus in locus_list:
        col = f"{locus}_BC"
        coupling = (
            df_allele.loc[df_allele[col].notna() & df_allele[col].ne(f"{locus}_[]")]
            .groupby(col, as_index=False)
            .agg(joint_allele=("joint_clone_id_tmp", "nunique"))
        )
        for _, row in coupling.iterrows():
            allele = row[col]
            if int(row["joint_allele"]) >= joint_allele_N_cutoff and allele_to_norm.get(allele, 1.0) < prob_cutoff:
                allele_to_norm[allele] = prob_cutoff
    for locus in locus_list:
        df_cells[f"{locus}_prob"] = df_cells[f"{locus}_BC"].map(allele_to_norm)
    df_cells["joint_prob"] = np.prod(df_cells[locus_prob_cols].fillna(1.0).to_numpy(dtype=float), axis=1)

    uf = UnionFind(len(df_allele))
    for locus in locus_list:
        col = f"{locus}_BC"
        for allele, group in df_allele.loc[df_allele[col].notna()].groupby(col).groups.items():
            if allele == f"{locus}_[]" or allele_to_norm.get(allele, 1.0) >= prob_cutoff:
                continue
            ids = list(group)
            if len(ids) < 2 or len(ids) >= joint_allele_N_cutoff:
                continue
            for i, left in enumerate(ids[:-1]):
                for right in ids[i + 1 :]:
                    if _joint_rows_compatible(df_allele.loc[left], df_allele.loc[right], locus_list):
                        uf.union(int(left), int(right))
    component_to_ids: dict[int, list[int]] = {}
    for idx in range(len(df_allele)):
        component_to_ids.setdefault(uf.find(idx), []).append(idx)
    rows = []
    tmp_to_joint: dict[str, str] = {}
    for clone_num, ids in enumerate(sorted(component_to_ids.values(), key=lambda values: (min(values), len(values)))):
        allele_list = _component_allele_list(df_allele, ids, locus_list)
        joint_clone_id = "@".join(allele_list)
        tmp_values = sorted(df_allele.loc[ids, "joint_clone_id_tmp"].astype(str).unique())
        for tmp in tmp_values:
            tmp_to_joint[tmp] = joint_clone_id
        rows.append(
            {
                "mosaic_component_id": f"ML_component_{clone_num + 1:06d}",
                "joint_clone_id": joint_clone_id,
                "BC_id": ";".join(str(item) for item in ids),
                "BC_num": int(len(ids)),
                "allele_num": int(len(allele_list)),
                "joint_clone_id_tmp_list": ";".join(tmp_values),
                "BC_consistency": float(_component_consistency(df_allele, ids, locus_list)),
            }
        )
    assigned_clones = pd.DataFrame(rows)
    df_cells["joint_clone_id"] = df_cells["joint_clone_id_tmp"].map(tmp_to_joint)
    allele_num_map = assigned_clones.set_index("joint_clone_id")["allele_num"].to_dict() if not assigned_clones.empty else {}
    consistency_map = assigned_clones.set_index("joint_clone_id")["BC_consistency"].to_dict() if not assigned_clones.empty else {}
    filtered = filtered.merge(df_cells[["joint_clone_id_tmp", "joint_prob", "joint_clone_id"]], left_on="RNA_id", right_index=True, how="left")
    filtered["joint_allele_num"] = filtered["joint_clone_id"].map(allele_num_map)
    filtered["BC_consistency"] = filtered["joint_clone_id"].map(consistency_map)
    payload = {
        "status": "ok",
        "n_input_rows": int(len(collapsed)),
        "n_rows_after_official_filter": int(len(filtered)),
        "n_joint_allele_rows": int(len(df_allele)),
        "n_joint_clone_components": int(len(assigned_clones)),
        "implementation": "faithful_sparse_reimplementation_of_assign_clone_id_by_integrating_locus_v0",
    }
    return assigned_clones, filtered.reset_index(drop=True), df_allele, payload


def _joint_rows_compatible(left: pd.Series, right: pd.Series, locus_list: tuple[str, ...]) -> bool:
    for locus in locus_list:
        col = f"{locus}_BC"
        left_value = left.get(col)
        right_value = right.get(col)
        if pd.notna(left_value) and pd.notna(right_value) and left_value != right_value:
            return False
    return True


def _component_allele_list(df_allele: pd.DataFrame, ids: list[int], locus_list: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for locus in locus_list:
        col = f"{locus}_BC"
        for value in df_allele.loc[ids, col].dropna().astype(str).tolist():
            if value not in values:
                values.append(value)
    return values


def _component_consistency(df_allele: pd.DataFrame, ids: list[int], locus_list: tuple[str, ...]) -> float:
    if len(ids) <= 1:
        return 1.0
    n_pairs = 0
    compatible = 0
    for i, left in enumerate(ids[:-1]):
        for right in ids[i + 1 :]:
            n_pairs += 1
            compatible += int(_joint_rows_compatible(df_allele.loc[left], df_allele.loc[right], locus_list))
    return compatible / max(n_pairs, 1)


def summarize_joint_assignment(
    assigned_clones: pd.DataFrame,
    assigned_rows: pd.DataFrame,
    classified: pd.DataFrame,
    *,
    bank_policy: str,
    de_novo_policy: str,
    threshold: ThresholdSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if assigned_rows.empty:
        empty_assignment = pd.DataFrame()
        empty_summary = pd.DataFrame()
        return empty_assignment, empty_summary, {
            "reference_bank_policy": bank_policy,
            "de_novo_policy": de_novo_policy,
            "n_joint_clones": 0,
            "n_joint_clone_assigned_cellbins": 0,
            "joint_clone_assigned_cellbin_fraction": 0.0,
        }
    cell_meta_cols = ["RNA_id", "sample_id", "slice_id", "section_order", "cellbin_id"]
    loci = assigned_rows.groupby("RNA_id")["locus"].agg(lambda s: ";".join(sorted(set(s.astype(str))))).rename("loci_present").reset_index()
    per_cell = (
        assigned_rows.sort_values(["RNA_id", "joint_prob"])
        .drop_duplicates("RNA_id")[
            [
                *cell_meta_cols,
                "joint_clone_id",
                "joint_clone_id_tmp",
                "joint_prob",
                "joint_allele_num",
                "BC_consistency",
            ]
        ]
        .merge(loci, on="RNA_id", how="left")
    )
    per_cell["n_loci_present"] = per_cell["loci_present"].fillna("").map(lambda value: len([item for item in str(value).split(";") if item]))
    per_cell["reference_bank_policy"] = bank_policy
    per_cell["de_novo_policy"] = de_novo_policy
    per_cell["threshold_label"] = threshold.label
    clone_summary = (
        per_cell.groupby("joint_clone_id", as_index=False)
        .agg(
            n_cellbins=("RNA_id", "nunique"),
            n_sections=("section_order", "nunique"),
            section_distribution=("section_order", lambda s: ";".join(f"{k}:{v}" for k, v in s.value_counts().sort_index().items())),
            median_joint_prob=("joint_prob", "median"),
            joint_allele_num=("joint_allele_num", "max"),
            BC_consistency=("BC_consistency", "median"),
            n_loci_present_median=("n_loci_present", "median"),
        )
        .sort_values(["n_cellbins", "joint_clone_id"], ascending=[False, True])
    )
    total_cellbins = max(classified["RNA_id"].nunique(), 1)
    recurrent = clone_summary.loc[clone_summary["n_cellbins"].ge(2), "n_cellbins"].sum() if not clone_summary.empty else 0
    payload = {
        "reference_bank_policy": bank_policy,
        "de_novo_policy": de_novo_policy,
        "threshold_label": threshold.label,
        "normalized_count_cutoff": threshold.prob_cutoff,
        "sample_count_cutoff": threshold.sample_count_cutoff,
        "min_cellbins_per_allele": threshold.min_cellbins_per_allele,
        "n_joint_clones": int(len(clone_summary)),
        "n_recurrent_joint_clones": int(clone_summary["n_cellbins"].ge(2).sum()) if not clone_summary.empty else 0,
        "n_joint_clone_assigned_cellbins": int(per_cell["RNA_id"].nunique()),
        "joint_clone_assigned_cellbin_fraction": float(per_cell["RNA_id"].nunique() / total_cellbins),
        "n_recurrent_joint_clone_cellbins": int(recurrent),
        "recurrent_joint_clone_cellbin_fraction": float(recurrent / total_cellbins),
        "n_cellbins_with_joint_alleles_ge2_loci": int(per_cell["n_loci_present"].ge(2).sum()),
        "fraction_cellbins_with_joint_alleles_ge2_loci": float(per_cell["n_loci_present"].ge(2).sum() / total_cellbins),
        "largest_joint_clone_cellbins": int(clone_summary["n_cellbins"].max()) if not clone_summary.empty else 0,
        "largest_joint_clone_fraction": float(clone_summary["n_cellbins"].max() / max(per_cell["RNA_id"].nunique(), 1)) if not clone_summary.empty else 0.0,
    }
    return per_cell, clone_summary, payload


def select_default_joint_policy(policy_summary: pd.DataFrame) -> dict[str, Any]:
    if policy_summary.empty:
        return {"decision": "no_joint_policy_available"}
    frame = policy_summary.copy()
    penalty = {
        "mapped_rare_only": 0.0,
        "mapped_rare_plus_low_frequency_denovo": 0.01,
        "mapped_rare_plus_empirical_denovo": 0.03,
    }
    frame["de_novo_penalty"] = frame["de_novo_policy"].map(penalty).fillna(0.05)
    frame["overmerge_penalty"] = (frame["largest_joint_clone_fraction"].astype(float) - 0.05).clip(lower=0) * 5
    frame["selection_score"] = (
        5 * frame["fraction_cellbins_with_joint_alleles_ge2_loci"].astype(float)
        + 2 * frame["recurrent_joint_clone_cellbin_fraction"].astype(float)
        + frame["joint_clone_assigned_cellbin_fraction"].astype(float)
        - frame["de_novo_penalty"]
        - frame["overmerge_penalty"]
    )
    frame = frame.sort_values(
        ["selection_score", "fraction_cellbins_with_joint_alleles_ge2_loci", "recurrent_joint_clone_cellbin_fraction"],
        ascending=False,
    )
    selected = frame.iloc[0].to_dict()
    selected["selection_rule"] = "maximized cross-locus support, recurrent recovery, assigned fraction, and overmerge/de_novo penalties"
    return selected


def matrix_preview(assignments: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame(columns=["RNA_id", "joint_clone_id", "value"])
    return assignments[["RNA_id", "joint_clone_id"]].drop_duplicates().assign(value=1).head(1000)


def compare_to_empirical_models(selected_assignment: pd.DataFrame, roots: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_cells = set(selected_assignment["RNA_id"]) if not selected_assignment.empty else set()
    total_cellbins = max(len(selected_cells), 1)
    rows.append(
        {
            "model": "DARLIN_style_joint_clone_id",
            "n_clone_or_membership_cellbins": int(len(selected_cells)),
            "overlap_with_darlin_style_cellbins": int(len(selected_cells)),
            "overlap_fraction_of_darlin_style": 1.0 if selected_cells else 0.0,
        }
    )
    round1 = roots["round1"] / "clones/cellbin_clone_assignment.tsv.gz"
    if round1.exists():
        frame = read_table(round1)
        if "cell_key" not in frame:
            frame["cell_key"] = make_cell_key(frame)
        status = frame["clone_status"].astype(str).eq("clone") if "clone_status" in frame else frame["clone_id"].notna()
        cells = set(frame.loc[status, "cell_key"])
        rows.append(_comparison_row("Round1_strict_graph_clones", cells, selected_cells))
    round2 = roots["round2"] / "assignments/high_confidence_cellbin_clone_assignment_v2.tsv.gz"
    if round2.exists():
        frame = read_table(round2)
        cells = set(frame.loc[frame["assignment_status"].isin(["assigned_single", "assigned_multi"]), "cell_key"])
        rows.append(_comparison_row("Round2_high_confidence_hard_assignment", cells, selected_cells))
    round21 = roots["round21"] / "membership/cellbin_clone_membership_summary.tsv.gz"
    if round21.exists():
        frame = read_table(round21)
        cells = set(frame.loc[frame["assignment_mode"].isin(["single_clone_dominant", "multi_clone_supported", "ambiguous"]), "cell_key"])
        rows.append(_comparison_row("Round2_1_clone_membership", cells, selected_cells))
    comparison = pd.DataFrame(rows)
    payload = {
        "n_darlin_style_cellbins": int(len(selected_cells)),
        "comparison_models": comparison.to_dict(orient="records"),
        "total_cellbins_denominator_note": "overlap fractions use DARLIN-style selected cellbins as denominator",
        "internal_total_cellbins_guard": total_cellbins,
    }
    return comparison, payload


def _comparison_row(name: str, cells: set[str], selected_cells: set[str]) -> dict[str, Any]:
    overlap = cells & selected_cells
    return {
        "model": name,
        "n_clone_or_membership_cellbins": int(len(cells)),
        "overlap_with_darlin_style_cellbins": int(len(overlap)),
        "overlap_fraction_of_darlin_style": float(len(overlap) / max(len(selected_cells), 1)),
        "overlap_fraction_of_model": float(len(overlap) / max(len(cells), 1)),
    }


def decide_final_label(
    selected: dict[str, Any],
    mapping_summary: pd.DataFrame,
    round21_fraction: float,
) -> tuple[str, list[str], dict[str, Any]]:
    warnings: list[str] = []
    assigned_fraction = float(selected.get("joint_clone_assigned_cellbin_fraction", 0.0) or 0.0)
    recurrent_fraction = float(selected.get("recurrent_joint_clone_cellbin_fraction", 0.0) or 0.0)
    de_novo_policy = str(selected.get("de_novo_policy", ""))
    max_mapping = float(mapping_summary["row_mapping_fraction"].max()) if not mapping_summary.empty else 0.0
    denovo_signal = de_novo_policy != "mapped_rare_only" and assigned_fraction > 0
    payload = {
        "selected_assigned_fraction": assigned_fraction,
        "selected_recurrent_fraction": recurrent_fraction,
        "max_reference_row_mapping_fraction": max_mapping,
        "round2_1_membership_fraction": round21_fraction,
    }
    if not selected or selected.get("decision") == "no_joint_policy_available":
        return "L126_DARLIN_STYLE_CELLBIN_CLONES_HOLD_FOR_SCHEMA_MISMATCH", warnings, payload
    if max_mapping < 0.20 and denovo_signal and assigned_fraction < round21_fraction:
        warnings.append("Reference mapping is incomplete, but low-frequency de novo allele evidence produced joint clone signal.")
        return "L126_DARLIN_STYLE_CELLBIN_CLONES_HOLD_FOR_LOW_REFERENCE_MAPPING_BUT_DENOVO_SIGNAL_PRESENT", warnings, payload
    if assigned_fraction <= 0.001:
        return "L126_DARLIN_STYLE_CELLBIN_CLONES_HOLD_FOR_LOW_RECOVERY", warnings, payload
    if assigned_fraction < round21_fraction:
        warnings.append("DARLIN-style joint clone recovery is below Round 2.1 membership coverage.")
        return "L126_DARLIN_STYLE_CELLBIN_CLONES_HOLD_FOR_LOW_RECOVERY", warnings, payload
    if max_mapping < 0.20 and de_novo_policy == "mapped_rare_only":
        return "L126_DARLIN_STYLE_CELLBIN_CLONES_HOLD_FOR_REFERENCE_MAPPING", warnings, payload
    if de_novo_policy != "mapped_rare_only":
        warnings.append("Selected joint clone layer depends on de novo allele inclusion and should be warning-labeled.")
        return "L126_DARLIN_STYLE_CELLBIN_CLONES_READY_WITH_DENOVO_WARNINGS", warnings, payload
    warnings.append("Spatial cellbins can carry partial or mixed allele evidence; interpret joint_clone_id as pre-niche clone call support.")
    return "L126_DARLIN_STYLE_CELLBIN_CLONES_READY_WITH_SPATIAL_WARNINGS", warnings, payload


def validate_audit_outputs(output_root: Path, report_root: Path, input_hash_before: dict[str, str], input_hash_after: dict[str, str]) -> dict[str, Any]:
    json_ok = True
    for path in sorted(report_root.glob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            json_ok = False
    tsv_paths = [
        output_root / "cellbin_allele_table.tsv.gz",
        output_root / "allele_bank_mapping_summary.tsv",
        output_root / "cellbin_allele_table_with_reference.tsv.gz",
        output_root / "rare_allele_filtering_summary.tsv",
        output_root / "valid_cellbin_allele_table.tsv.gz",
        output_root / "cellbin_joint_clone_assignment.tsv.gz",
        output_root / "joint_clone_summary.tsv.gz",
    ]
    tsv_ok = True
    for path in tsv_paths:
        try:
            if not path.exists():
                tsv_ok = False
            else:
                read_table(path, nrows=5)
        except Exception:
            tsv_ok = False
    text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(report_root.glob("*.md")))
    claim_hits = positive_claim_hits(text)
    return {
        "validation_status": "PASS" if json_ok and tsv_ok and input_hash_before == input_hash_after and not claim_hits and "/ssd/" not in text else "FAIL",
        "json_parse": bool(json_ok),
        "tsv_gzip_readability": bool(tsv_ok),
        "source_input_packet_unchanged": bool(input_hash_before == input_hash_after),
        "no_ssd": "/ssd/" not in text,
        "no_raw_fastq": "processed raw fastq" not in text.lower(),
        "no_spatio_darlin_rerun": "spatio_darlin was rerun" not in text.lower(),
        "no_plana_planb": "plana production" not in text.lower() and "planb production" not in text.lower(),
        "no_positive_fate_transition_claims": bool(not claim_hits),
        "positive_claim_hits": claim_hits,
        "no_git_operations": True,
    }
