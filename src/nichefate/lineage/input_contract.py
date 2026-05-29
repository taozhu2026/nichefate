from __future__ import annotations

from nichefate.barcode_adapter.input_contract import (
    ALLELE_ANNOTATION_REQUIRED_COLUMNS,
    CELLBIN_SUMMARY_REQUIRED_FIELDS,
    EXPECTED_ASSAYS,
    GROUP_ASSIGNMENT_OPTIONAL_COLUMNS,
    GROUP_ASSIGNMENT_REQUIRED_COLUMNS,
    LINEAGE_EVIDENCE_REQUIRED_COLUMNS,
    PRIMARY_JOIN_KEY,
    REQUIRED_H5AD_LAYERS,
    REQUIRED_H5AD_OBSM,
    REQUIRED_H5AD_OBS_FIELDS,
    BarcodeInputContract as LineageInputContract,
    draft_contract_payload,
    load_barcode_input_contract,
)

draft_lineage_input_contract_payload = draft_contract_payload
load_lineage_input_contract = load_barcode_input_contract

__all__ = [
    "ALLELE_ANNOTATION_REQUIRED_COLUMNS",
    "CELLBIN_SUMMARY_REQUIRED_FIELDS",
    "EXPECTED_ASSAYS",
    "GROUP_ASSIGNMENT_OPTIONAL_COLUMNS",
    "GROUP_ASSIGNMENT_REQUIRED_COLUMNS",
    "LINEAGE_EVIDENCE_REQUIRED_COLUMNS",
    "LineageInputContract",
    "PRIMARY_JOIN_KEY",
    "REQUIRED_H5AD_LAYERS",
    "REQUIRED_H5AD_OBSM",
    "REQUIRED_H5AD_OBS_FIELDS",
    "draft_lineage_input_contract_payload",
    "load_lineage_input_contract",
]
