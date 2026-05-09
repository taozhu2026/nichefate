# Sailu-ST-01 Mixed ST Smoke Feasibility

## Decision

`Sailu_0313_19A_mixed_ST` is conceptually allowed as an exploratory mixed ST input. Brain/Pancreas demultiplexing is not a prerequisite for preliminary raw ST processing.

However, a Sailu-ST-02 tiny smoke run is not ready yet.

Decision: `blocked_pending_provider_salus_contract`

## Blocking Gaps

- BC called-table semantics are not defined.
- No confirmed barcode/spot ID column was found.
- No confirmed coordinate table was found.
- E-to-BC linkage is not defined.
- UMI rule is unknown.
- Raw ST processing implementation is not identified.
- Reference genome/transcriptome and gene annotation choices are not recorded.
- Matrix writer, coordinate writer, AnnData writer, and M0 metadata writer are not available for this sample.

## Future Smoke Preconditions

Before Sailu-ST-02, obtain or identify:

- Salus/Sailu documentation for `ListTable.csv` and `R###C###` FOV reports;
- a parser contract connecting BC spatial structure to E PE100 reads;
- UMI/read-structure rules;
- reference and gene annotation paths;
- a raw ST processing script or command that emits an expression matrix plus spatial metadata.

Only after those contracts are available should ST-02 prepare a tiny read subset and smoke-run plan under `/home/zhutao/scratch/nichefate/darlin_st_sailu_mixed/`.
