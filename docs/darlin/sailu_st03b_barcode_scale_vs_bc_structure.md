# Sailu-ST-03B Barcode Scale vs BC Structure

## BC Hints Used As Context Only

- `ListTable.csv`: 2 rows x 1821 fields.
- BC grid hint: `RowMax=14`, `ColumnMax=130`, so `14 x 130 = 1820`.
- `R###C###` appears as FOV-like structure in BC reports.

These are not coordinate truth and are not a barcode whitelist.

## Best Non-Overlapping Structure

- Structure: `NONOVERLAP_R1_BC1_8_UMI9_20`
- Barcode window: `R1:1-8`
- UMI window: `R1:9-20`
- Barcode unique count: `58904`
- Scale plausibility: `inflated_but_possible_with_whitelist_error_correction`

## Interpretation

Exact parsed candidate counts remain much larger than the BC grid hint unless barcode correction or a whitelist is applied. A whitelist and coordinate map are still required before any spot x gene matrix or M0 input can be produced.
