# Sailu-ST-03 BC/E Linkage Smoke Review

## Scope

This review compares tiny parser candidate windows from E FASTQs against previously observed BC structural hints for `Sailu_0313_19A_mixed_ST`.

## BC Structural Hints Used

- `SalusCallFile/ListTable.csv`: 2 rows x 1821 numeric fields.
- BC `RunInfo.xml`: `RowMax=14`, `ColumnMax=130`, so `14 x 130 = 1820` grid-like positions.
- BC report filenames expose `R###C###` FOV-like structure.

## E FASTQ Parser Findings

Best R1 barcode-like candidate: `CAND_R1_ALT_8_18` with `357368` unique candidate sequences across the bounded smoke.

Best UMI-like stress-test candidate: `CAND_R1_UMI_15_30` with `369007` unique candidate sequences across the bounded smoke.

R2 control candidate: `CAND_R2_ALT_1_20` showed extractable structure but is not selected as the best barcode candidate because ST-02 classified R2 as `mixed_or_unknown`.

## Linkage Decision

BC/E linkage status: `unresolved_no_direct_bc_e_linkage`.

The candidate E sequence windows can be parsed consistently, but no local file maps those sequence windows to BC `ListTable.csv`, BC `R###C###` reports, or a coordinate whitelist. Unique candidate counts do not by themselves establish a BC spatial relationship.

Do not reinterpret `ListTable.csv` as a final coordinate table in ST-03.
