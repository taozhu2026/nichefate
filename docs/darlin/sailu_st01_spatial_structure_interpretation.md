# Sailu-ST-01 Spatial Structure Interpretation

## Candidate Classes

- `SalusCallFile/ListTable.csv`: `tile_or_fov_offset_table` / `coordinate_like_candidate`
- `LogFiles/RunInfo.xml`: `image_tile_layout`
- `LogFiles/AInfo.txt`: `tile_step_geometry_candidate`
- `Res/Lane02/report/R###C###.html`: `image_tile_layout` plus FOV QC visualization
- `LogFiles/ASalusCall.csv`, `LogFiles/APerformance.csv`, `LogFiles/AError.txt`: QC/runtime logs

## Interpretation

The strongest local spatial evidence is the agreement between:

- `RunInfo.xml`: `RowMax=14`, `ColumnMax=130`
- `AInfo.txt`: `nStepX=429.6`, `nStepY=709.8`
- `ListTable.csv`: 1821 numeric fields per row, consistent with `14 * 130 + 1`
- FOV HTML names: `R###C###.html`, with observed rows 1-14 and columns 3-130 in the inspected reports

This supports a spatial-grid hypothesis, but not a complete M0 coordinate table. The local files do not explain whether the two `ListTable.csv` rows are lanes, channels, offsets into basecall output, or another Salus-specific structure. The local files also do not provide a spot ID schema or a direct mapping from expression FASTQ records to BC spatial positions.

## M0 Coordinate Readiness

No confirmed coordinate table is ready for M0. A future coordinate table could be derived only if provider documentation or Salus code confirms:

- the semantics of `ListTable.csv` rows and values;
- whether `R###C###` is the spot/FOV grid identifier;
- how `nStepX` and `nStepY` convert row/column to coordinates;
- the coordinate origin, orientation, and unit;
- how E FASTQ headers link to BC rows/columns/FOVs.
