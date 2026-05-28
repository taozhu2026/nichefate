# Freeze Summary

## Result

- Final algorithm decision label: `NICHEFATE_LINEAGE_AWARE_BASELINE_V1_READY_WITH_QC_WARNINGS`
- Benchmark label: `L126_SPATIODARLIN_BENCHMARK_READY_WITH_QC_WARNINGS`
- Branch: `freeze/lineage-aware-nichefate-v1`
- Code freeze commit hash: `44001ee9e677aa4fda899a52c7352e0c4ffc1212`

## Public Modules

- NF-L0: lineage input contract
- NF-L1: lineage evidence adapter
- NF-L2: DARLIN-style joint clone calling
- NF-L3: clone x niche integration
- NF-L4: lineage-aware spatial niche characterization
- NF-L5: dynamics interface design

## Unified Clone Definition

Primary clone unit: validated DARLIN-style joint clone.
Reference/de novo status is allele-level QC metadata only.

## Included

- Generic lineage and DARLIN facades
- Benchmark-backed implementation code
- Documentation and benchmark summaries
- Config schemas and benchmark config
- Focused tests
- Small validation and audit reports

## Excluded

- Raw FASTQ and packet inputs
- Processed outputs
- Scratch outputs
- Large matrices and tables
- Future-dynamics claims from serial sections

## Validation And Staging

- Validation status: `PASS`
- Staging audit status: `PASS`

## Remaining Limitations

- L126 is the first benchmark only.
- Serial sections do not justify future-dynamics claims.
- Reference-only calling remains a conservative benchmark.
- QC warnings remain part of the frozen result set.

## Next Recommended Step

Review the freeze branch, then decide whether to tag
`nichefate-lineage-aware-v1`.
