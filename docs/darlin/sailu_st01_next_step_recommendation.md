# Sailu-ST-01 Next Step Recommendation

## Exact Next Step

Ask Sailu/provider or locate Salus processing documentation/code that answers these contract questions:

1. What do the two rows and 1821 fields in `SalusCallFile/ListTable.csv` represent?
2. Does `ListTable.csv` map FOVs, tiles, byte offsets, spatial barcodes, or spot coordinates?
3. How should `RunInfo.xml` fields `RowMax=14`, `ColumnMax=130`, `nStepX=429.6`, and `nStepY=709.8` be converted into spatial coordinates?
4. How do E FASTQ header fields `lane:numeric:R###:C###` link to BC `R###C###` reports or `ListTable.csv` positions?
5. What is the UMI rule and expression read structure for `ZHY_XBB_E_PE100_100`?
6. Which raw ST processing script should be used to generate expression matrix, spot metadata, coordinate table, and optional `.h5ad`?

Do not start Sailu-ST-02 until the BC coordinate contract and E-to-BC linkage are defined well enough to build a tiny mixed-ST smoke command.
