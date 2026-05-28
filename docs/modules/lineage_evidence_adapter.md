# Lineage Evidence Adapter

NF-L1 converts processed lineage evidence into canonical cellbin-level tables
used by downstream clone calling.

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
