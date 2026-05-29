from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .input_contract import EXPECTED_ASSAYS, PRIMARY_JOIN_KEY


def sha256_path(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(
        microsecond=0
    ).isoformat()


def snapshot_files(paths: list[str | Path], *, include_sha256: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        row: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            "mtime_utc": _mtime_utc(path) if path.exists() else "",
        }
        if include_sha256 and path.exists() and path.is_file():
            row["sha256"] = sha256_path(path)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_file_snapshots(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    merged = before.merge(after, on="path", how="outer", suffixes=("_before", "_after"))
    checks = []
    for row in merged.to_dict(orient="records"):
        changed = False
        for field in ["exists", "size_bytes", "mtime_utc", "sha256"]:
            before_value = row.get(f"{field}_before")
            after_value = row.get(f"{field}_after")
            if before_value != after_value:
                changed = True
        checks.append({**row, "changed": bool(changed)})
    return pd.DataFrame(checks)


def read_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, sep="\t")
    required = {"path", "size_bytes", "sha256"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    return manifest


def verify_manifest(packet_root: str | Path, manifest_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(packet_root)
    manifest = read_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    for row in manifest.to_dict(orient="records"):
        relative = str(row["path"])
        path = root / relative
        exists = path.exists()
        size = int(path.stat().st_size) if exists else 0
        observed_sha = sha256_path(path) if exists else ""
        rows.append(
            {
                "path": relative,
                "exists": exists,
                "expected_size_bytes": int(row["size_bytes"]),
                "observed_size_bytes": size,
                "size_ok": exists and size == int(row["size_bytes"]),
                "expected_sha256": str(row["sha256"]),
                "observed_sha256": observed_sha,
                "sha256_ok": exists and observed_sha == str(row["sha256"]),
                "mtime_utc": _mtime_utc(path) if exists else "",
            }
        )
    frame = pd.DataFrame(rows)
    payload = {
        "manifest_path": str(manifest_path),
        "packet_root": str(root),
        "manifest_rows": int(len(frame)),
        "all_files_exist": bool(frame["exists"].all()),
        "all_sizes_match": bool(frame["size_ok"].all()),
        "all_sha256_match": bool(frame["sha256_ok"].all()),
    }
    payload["validation_passed"] = all(
        [payload["all_files_exist"], payload["all_sizes_match"], payload["all_sha256_match"]]
    )
    return frame, payload


def validate_cellbin_lineage_join(
    cellbins: pd.DataFrame,
    lineage_evidence: pd.DataFrame,
    expected_samples: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate primary-key joins and coordinate provenance."""

    key_cols = list(PRIMARY_JOIN_KEY)
    cellbin_keys = cellbins[key_cols + ["x", "y"]].drop_duplicates(key_cols)
    evidence = lineage_evidence.copy()
    evidence["count"] = pd.to_numeric(evidence["count"], errors="raise")
    evidence_keys = evidence[key_cols + ["x", "y"]].drop_duplicates(key_cols)
    joined = evidence_keys.merge(cellbin_keys, on=key_cols, how="left", suffixes=("_evidence", "_h5ad"))
    joined["key_in_h5ad"] = joined["x_h5ad"].notna() & joined["y_h5ad"].notna()
    joined["xy_match"] = (
        joined["key_in_h5ad"]
        & (joined["x_evidence"].astype(float) == joined["x_h5ad"].astype(float))
        & (joined["y_evidence"].astype(float) == joined["y_h5ad"].astype(float))
    )
    evidence_coord_unique = (
        evidence.groupby(key_cols)[["x", "y"]].nunique().reset_index()
    )
    inconsistent_coordinates = int(
        ((evidence_coord_unique["x"] > 1) | (evidence_coord_unique["y"] > 1)).sum()
    )
    sample_rows = []
    for sample_id in sorted(set(expected_samples) | set(cellbins["sample_id"].astype(str))):
        cellbin_sample = cellbin_keys.loc[cellbin_keys["sample_id"].astype(str) == sample_id]
        evidence_sample = evidence_keys.loc[evidence_keys["sample_id"].astype(str) == sample_id]
        shared = evidence_sample.merge(cellbin_sample[key_cols], on=key_cols, how="inner")
        sample_joined = joined.loc[joined["sample_id"].astype(str) == sample_id]
        sample_rows.append(
            {
                "sample_id": sample_id,
                "h5ad_cellbin_count": int(len(cellbin_sample)),
                "evidence_positive_cellbin_count": int(len(evidence_sample)),
                "shared_positive_cellbin_count": int(len(shared)),
                "all_positive_evidence_keys_in_h5ad": bool(sample_joined["key_in_h5ad"].all())
                if not sample_joined.empty
                else True,
                "xy_match_fraction": float(sample_joined["xy_match"].mean())
                if not sample_joined.empty
                else 1.0,
            }
        )
    frame = pd.DataFrame(sample_rows)
    unexpected_samples = sorted(
        (set(cellbins["sample_id"].astype(str)) | set(evidence["sample_id"].astype(str)))
        - set(expected_samples)
    )
    payload = {
        "primary_join_key": list(PRIMARY_JOIN_KEY),
        "coordinate_role": "validation_and_spatial_provenance_only",
        "unexpected_samples": unexpected_samples,
        "positive_evidence_keys_missing_h5ad": int((~joined["key_in_h5ad"]).sum()),
        "coordinate_mismatch_count": int((~joined["xy_match"]).sum()),
        "inconsistent_evidence_coordinate_keys": inconsistent_coordinates,
        "join_validation_passed": bool(
            not unexpected_samples
            and (~joined["key_in_h5ad"]).sum() == 0
            and (~joined["xy_match"]).sum() == 0
            and inconsistent_coordinates == 0
        ),
    }
    return frame, payload


def build_cellbin_assay_qc(lineage_evidence: pd.DataFrame) -> pd.DataFrame:
    evidence = lineage_evidence.copy()
    evidence["count"] = pd.to_numeric(evidence["count"], errors="raise")
    evidence["assay_feature_id"] = evidence["assay"].astype(str) + "::" + evidence["feature_id"].astype(str)
    qc = (
        evidence.groupby(["sample_id", "slice_id", "section_order", "assay"], as_index=False)
        .agg(
            evidence_row_count=("count", "size"),
            total_count=("count", "sum"),
            positive_cellbin_count=("cellbin_id", "nunique"),
            detected_feature_count=("assay_feature_id", "nunique"),
        )
        .sort_values(["sample_id", "assay"])
    )
    return qc


def audit_allele_annotation(
    lineage_evidence: pd.DataFrame,
    allele_annotation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit allele annotations without expanding primary evidence counts."""

    join_cols = ["sample_id", "slice_id", "section_order", "assay", "feature_id"]
    primary = (
        lineage_evidence.groupby(join_cols, as_index=False)
        .agg(
            primary_evidence_row_count=("count", "size"),
            primary_lineage_total_count=("count", "sum"),
        )
    )
    annotation = (
        allele_annotation.groupby(join_cols, as_index=False)
        .agg(
            clone_id=("clone_id", "first"),
            annotation_expanded_row_count=("allele", "size"),
            n_alleles_for_feature=("n_alleles_for_feature", "max"),
            allele_missing_count=("allele_is_missing", lambda s: int(s.astype(str).str.lower().eq("true").sum()))
            if "allele_is_missing" in allele_annotation.columns
            else ("allele", "size"),
        )
    )
    audit = annotation.merge(primary, on=join_cols, how="left")
    audit["primary_evidence_row_count"] = audit["primary_evidence_row_count"].fillna(0).astype(int)
    audit["primary_lineage_total_count"] = audit["primary_lineage_total_count"].fillna(0.0)
    audit["annotation_only"] = True
    audit["counts_are_independent_evidence"] = False
    observed_primary_total = float(lineage_evidence["count"].sum())
    audited_primary_total = float(audit["primary_lineage_total_count"].sum())
    misused_expanded_total = float(
        (audit["primary_lineage_total_count"] * audit["annotation_expanded_row_count"]).sum()
    )
    payload = {
        "primary_count_total": observed_primary_total,
        "audit_feature_level_primary_count_total": audited_primary_total,
        "allele_expanded_count_total_if_misused": misused_expanded_total,
        "allele_annotation_rows": int(len(allele_annotation)),
        "audit_rows": int(len(audit)),
        "max_annotation_rows_per_feature": int(audit["annotation_expanded_row_count"].max())
        if not audit.empty
        else 0,
        "non_inflation_passed": bool(abs(observed_primary_total - audited_primary_total) < 1e-6),
        "assays_preserved": sorted(lineage_evidence["assay"].dropna().astype(str).unique().tolist())
        == sorted(EXPECTED_ASSAYS),
    }
    return audit, payload
