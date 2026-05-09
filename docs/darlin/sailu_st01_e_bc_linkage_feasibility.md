# Sailu-ST-01 E-to-BC Linkage Feasibility

## E FASTQ Header Probe

Representative E FASTQ headers were sampled from the first 20 records per lane/read only. Example:

`@2211180001:S:00000000000000:1:000016:R001:C003 1:L:0`

The sampled headers expose:

- lane field: present
- numeric field after lane: present, possibly FOV/tile-like but not documented locally
- row/column-like fields: present as `R###` and `C###`
- read pair flag: present as `1` or `2`
- nucleotide sample index: not observed; E `RunInfo.xml` also reports `IndexLen=0`

## Linkage Decision

The E headers contain row/column-like fields that might eventually link to a BC spatial grid. However, the linkage is not contract-ready:

- E `RunInfo.xml` reports `RowMax=4` and `ColumnMax=110`.
- BC `RunInfo.xml` reports `RowMax=14` and `ColumnMax=130`.
- BC HTML reports are under `Res/Lane02/report/` and are a subset of observed row/column reports.
- No local file states how E header fields map to BC `ListTable.csv`, BC FOV reports, or final spot IDs.

Current status: E-to-BC linkage is possible in concept but not locally confirmed. It must remain blocked pending a Sailu/Salus contract or implementation that defines the mapping.
