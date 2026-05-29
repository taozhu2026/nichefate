# Lineage Evidence Adapter

The lineage evidence adapter is the first barcode-supported evidence module in
the lineage-aware NicheFate mode. It converts processed lineage evidence into
canonical cellbin-level tables used by downstream clone calling.

## Responsibilities

- Read complete lineage evidence rather than a truncated top-feature summary
- Preserve assay-scoped features as `assay::feature_id`
- Preserve CA/TA/RA separation
- Track allele annotation as QC metadata

## Output Tables

- cellbin-feature evidence
- feature frequency reference
- cellbin barcode complexity
- allele annotation audit tables

The adapter does not construct niches by itself. It attaches barcode evidence
to the same cellbin identities used by the shared spatial-niche substrate.
