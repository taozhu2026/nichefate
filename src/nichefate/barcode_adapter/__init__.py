"""Barcode-aware input adapter utilities for NicheFate."""

from .aggregation import (
    aggregate_lineage_to_groups,
    compute_lineage_diversity_metrics,
    summarize_cellbin_lineage_evidence,
)
from .input_contract import (
    BarcodeInputContract,
    load_barcode_input_contract,
)
from .loaders import (
    PacketPaths,
    load_cellbin_lineage_evidence,
    load_feature_allele_annotation,
    load_l126_h5ad_packet,
    prepare_packet_root,
)
from .l126_schema import (
    h5ad_path_for_sample,
    load_l126_cellbin_table,
    validate_l126_h5ad_schema,
)
from .spatial_neighborhood import (
    GROUP_TYPE,
    build_spatial_neighborhood_groups,
    group_membership_multiplicity,
    spatially_stratified_subset,
)
from .qc import (
    audit_allele_annotation,
    compare_file_snapshots,
    snapshot_files,
    validate_cellbin_lineage_join,
    verify_manifest,
)

__all__ = [
    "BarcodeInputContract",
    "PacketPaths",
    "aggregate_lineage_to_groups",
    "audit_allele_annotation",
    "compare_file_snapshots",
    "compute_lineage_diversity_metrics",
    "load_barcode_input_contract",
    "load_cellbin_lineage_evidence",
    "load_feature_allele_annotation",
    "load_l126_h5ad_packet",
    "load_l126_cellbin_table",
    "h5ad_path_for_sample",
    "validate_l126_h5ad_schema",
    "prepare_packet_root",
    "snapshot_files",
    "GROUP_TYPE",
    "build_spatial_neighborhood_groups",
    "group_membership_multiplicity",
    "spatially_stratified_subset",
    "summarize_cellbin_lineage_evidence",
    "validate_cellbin_lineage_join",
    "verify_manifest",
]
