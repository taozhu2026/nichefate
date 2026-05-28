from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_ASSAYS: tuple[str, ...] = ("RA", "TA", "CA")
PRIMARY_JOIN_KEY: tuple[str, ...] = ("sample_id", "slice_id", "cellbin_id")
GROUP_ASSIGNMENT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "slice_id",
    "cellbin_id",
    "group_id",
)
GROUP_ASSIGNMENT_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "group_type",
    "anchor_id",
    "niche_id",
    "metaniche_id",
)
REQUIRED_H5AD_OBS_FIELDS: tuple[str, ...] = (
    "cellbin_id",
    "sample_id",
    "slice_id",
    "x",
    "y",
)
REQUIRED_H5AD_LAYERS: tuple[str, ...] = ("counts",)
REQUIRED_H5AD_OBSM: tuple[str, ...] = ("spatial",)
LINEAGE_EVIDENCE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "slice_id",
    "section_order",
    "assay",
    "cellbin_id",
    "x",
    "y",
    "feature_id",
    "clone_id",
    "count",
)
ALLELE_ANNOTATION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "slice_id",
    "section_order",
    "assay",
    "feature_id",
    "clone_id",
    "allele",
    "n_alleles_for_feature",
)
CELLBIN_SUMMARY_REQUIRED_FIELDS: tuple[str, ...] = (
    "sample_id",
    "slice_id",
    "section_order",
    "cellbin_id",
    "x",
    "y",
    "total_lineage_count",
    "detected_feature_count",
    "detected_assay_count",
    "RA_total_count",
    "TA_total_count",
    "CA_total_count",
    "RA_detected_feature_count",
    "TA_detected_feature_count",
    "CA_detected_feature_count",
    "dominant_assay",
    "dominant_feature_id",
    "dominant_feature_count",
    "dominant_feature_fraction",
    "feature_entropy",
    "simpson_diversity",
    "evidence_present",
)


@dataclass(frozen=True)
class BarcodeInputContract:
    path: Path
    raw: dict[str, Any]
    packet_archive: Path | None
    packet_root: Path | None
    sample_list: tuple[str, ...]
    excluded_samples: tuple[str, ...]
    assay_list: tuple[str, ...]
    primary_join_key: tuple[str, ...]
    expected_h5ad_files: tuple[str, ...]
    primary_evidence_file: str
    allele_annotation_file: str
    expected_reports: tuple[str, ...]


def _as_tuple(data: dict[str, Any], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = data.get(key, default)
    if value is None:
        return default
    return tuple(str(item) for item in value)


def _path_or_none(value: Any) -> Path | None:
    if not value:
        return None
    return Path(str(value)).expanduser()


def _expected_h5ad_files(data: dict[str, Any]) -> tuple[str, ...]:
    files = data.get("expected_h5ad_files")
    if files:
        return tuple(str(path) for path in files)
    h5ad_paths = data.get("h5ad_paths", [])
    return tuple("processed/h5ad/" + Path(str(path)).name for path in h5ad_paths)


def load_barcode_input_contract(path: str | Path) -> BarcodeInputContract:
    """Load a local or transferred BarcodeEvidenceAdapter contract."""

    contract_path = Path(path).expanduser().resolve()
    data = json.loads(contract_path.read_text(encoding="utf-8"))
    evidence_files = data.get("lineage_evidence_files", {})
    lineage_paths = [str(item) for item in data.get("lineage_evidence_paths", [])]
    primary_file = evidence_files.get("primary")
    allele_file = evidence_files.get("allele_annotation")
    if primary_file is None:
        primary_file = next(
            (
                "processed/lineage_evidence/" + Path(path).name
                for path in lineage_paths
                if Path(path).name == "cellbin_lineage_evidence.tsv.gz"
            ),
            "processed/lineage_evidence/cellbin_lineage_evidence.tsv.gz",
        )
    if allele_file is None:
        allele_file = next(
            (
                "processed/lineage_evidence/" + Path(path).name
                for path in lineage_paths
                if Path(path).name == "feature_allele_annotation_long.tsv.gz"
            ),
            "processed/lineage_evidence/feature_allele_annotation_long.tsv.gz",
        )

    contract = BarcodeInputContract(
        path=contract_path,
        raw=data,
        packet_archive=_path_or_none(data.get("packet_archive")),
        packet_root=_path_or_none(data.get("packet_root")),
        sample_list=_as_tuple(data, "sample_list", ()),
        excluded_samples=_as_tuple(data, "excluded_samples", ()),
        assay_list=_as_tuple(data, "assay_list", EXPECTED_ASSAYS),
        primary_join_key=_as_tuple(data, "primary_join_key", PRIMARY_JOIN_KEY),
        expected_h5ad_files=_expected_h5ad_files(data),
        primary_evidence_file=str(primary_file),
        allele_annotation_file=str(allele_file),
        expected_reports=_as_tuple(data, "expected_reports", ()),
    )
    if contract.primary_join_key != PRIMARY_JOIN_KEY:
        raise ValueError(
            "Round 1 primary join key must be sample_id + slice_id + cellbin_id"
        )
    return contract


def draft_contract_payload(contract: BarcodeInputContract) -> dict[str, Any]:
    """Return the implementation-level contract used by the Round 1 reports."""

    return {
        "assay_list": list(contract.assay_list),
        "primary_join_key": list(PRIMARY_JOIN_KEY),
        "coordinate_usage": "validation_and_spatial_provenance_only",
        "sample_list": list(contract.sample_list),
        "excluded_samples": list(contract.excluded_samples),
        "required_h5ad": {
            "obs": list(REQUIRED_H5AD_OBS_FIELDS),
            "obsm": list(REQUIRED_H5AD_OBSM),
            "layers": list(REQUIRED_H5AD_LAYERS),
        },
        "lineage_evidence_required_columns": list(LINEAGE_EVIDENCE_REQUIRED_COLUMNS),
        "allele_annotation_required_columns": list(ALLELE_ANNOTATION_REQUIRED_COLUMNS),
        "cellbin_summary_required_fields": list(CELLBIN_SUMMARY_REQUIRED_FIELDS),
        "group_assignment_contract": {
            "required_columns": list(GROUP_ASSIGNMENT_REQUIRED_COLUMNS),
            "optional_columns": list(GROUP_ASSIGNMENT_OPTIONAL_COLUMNS),
        },
        "allele_annotation_rule": (
            "annotation only; one-to-many allele rows must not multiply primary "
            "feature/clone-level counts"
        ),
        "section_interpretation": "L126_Brain_s1/s2/s3 are serial sections, not timepoints",
        "round1_exclusions": {
            "L0927_Brain": "excluded because processed lineage evidence is absent"
        },
    }
