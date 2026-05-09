# Sailu-ST-01 BC Called-Table Contract

## Sample Scope

- `sample_id`: `Sailu_0313_19A_mixed_ST`
- `tissue_status`: `mixed_unresolved`
- Brain/Pancreas split is not required before preliminary raw ST processing.
- This audit does not label the sample as `Brain_E1_ST` or E1-only.

## Inspected BC Evidence

- BC folder: `/data/zhutao/nichefate_data/202603131741_Pro019_A_SE30_1_0313_19A_BC`
- Main called-table candidate: `SalusCallFile/ListTable.csv`
- Runtime metadata: `LogFiles/RunInfo.xml`, `LogFiles/AInfo.txt`
- QC/report evidence: `Res/Lane02/report/R###C###.html`, `LogFiles/ASalusCall.csv`, `LogFiles/APerformance.csv`, `LogFiles/AError.txt`

## Called-Table Findings

`SalusCallFile/ListTable.csv` exists and is lightweight enough to inspect directly. It has 2 rows and 1821 numeric fields per row, with no header row. The value count is consistent with `14 * 130 + 1`, matching `RunInfo.xml` fields `RowMax=14` and `ColumnMax=130` plus one leading value. The rows are monotonically increasing numeric lists and look more like lane-wise FOV/tile offset vectors than a biological spot table.

No explicit barcode ID, spot ID, x/y column, sample name, tissue label, Brain/Pancreas label, E1/E2 label, or gene-expression field was found in `ListTable.csv`.

## Runtime Spatial Hints

`RunInfo.xml` reports `SeqMode=CustomSeq`, `CycleNumShow=30`, `LaneNum=2`, `RowMax=14`, `ColumnMax=130`, `IndexLen=0`, and `Prefix=0313_19A_BC`.

`LogFiles/AInfo.txt` reports `mBasePara.imgPara.nRowPerChl=14`, `mBasePara.imgPara.nColPerChl=130`, `mBasePara.nStepX=429.6`, and `mBasePara.nStepY=709.8`. These fields are coordinate-like imaging geometry, but they do not by themselves define a confirmed M0 coordinate table because origin, orientation, unit interpretation, spot identity, and E-read linkage are not documented locally.

## Contract Decision

The BC folder contains a called spatial-structure candidate, not a ready M0 spot/coordinate contract. A pseudo-coordinate table may be possible later from row/column grid fields and step sizes, but only after the Salus/Sailu contract confirms how `ListTable.csv`, FOV reports, and E FASTQ header fields map to spatial spots.

Current status: `bc_called_table_found = true`, `coordinate_like_fields_found = true`, `coordinate_table_ready = false`.
