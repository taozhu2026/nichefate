# DARLIN-02A Corrected CA/TA/RA Locus Mapping

Timestamp: 2026-05-09T12:51:43Z

Branch: `darlin-onboarding`

## Correction

The prior DARLIN-02A draft mapping `RA=Tigre / TA=Rosa` is deprecated.

Status label:

`deprecated_wrong_mapping_from_prior_plan`

Official DARLIN/CARLIN terminology to use from this point:

- `CA` / `CC` = `Col1a1` / `cCARLIN`
- `TA` / `TC` = `Tigre`
- `RA` / `RC` = `Rosa` / `Rosa26`

## Corrected Meiji E1 Interpretation

- `Brain_E1_all_TA` is the Tigre candidate track.
- `Brain_E1_all_RA` is the Rosa candidate track.
- `Brain_E1_all_CA` is absent/deferred for Meiji E1.

## Official Local Evidence

Inspected read-only:

- `/home/zhutao/projects/darlin_cell_repro/code/snakemake_DARLIN/README.md`
- `/home/zhutao/projects/darlin_cell_repro/code/Custom_CARLIN/README.md`
- `/home/zhutao/projects/darlin_cell_repro/code/Custom_CARLIN/README_SW.md`
- `/home/zhutao/projects/darlin_cell_repro/code/Custom_CARLIN/cfg/parse_config_file.m`
- `/home/zhutao/projects/darlin_cell_repro/code/snakemake_DARLIN/darlin/help_functions.py`
- `/home/zhutao/projects/darlin_cell_repro/code/Custom_CARLIN/switch_template.m`
- `/home/zhutao/projects/darlin_cell_repro/code/MosaicLineage`
- `/data/zhutao/darlin_data/config.yaml`

Observed support:

- Custom_CARLIN README states the extended CARLIN arrays are `CC`, `TC`, and `RC` for `Col1a1`, `Tigre`, and `Rosa26`.
- Snakemake DARLIN README describes support for CA, TA, and RA loci.
- Official template options include `cCARLIN`, `Tigre`, `Tigre_2022`, `Tigre_2022_v2`, `Rosa`, and `Rosa_v2`.
- Official cfg files include `BulkRNA_Tigre*`, `BulkDNA_Tigre*`, `BulkRNA_Rosa*`, and `BulkDNA_Rosa*`.
- MosaicLineage contains reference allele files for `CA`, `TA`, and `RA`.

## Remaining Barcode Configuration Decisions

The official family mapping is corrected and no longer blocks the barcode track. DARLIN-01C is still blocked because these execution-specific values remain unresolved:

- `TA/Tigre`: exact template variant, e.g. `Tigre_2022` versus `Tigre_2022_v2`.
- `RA/Rosa`: exact template variant, e.g. `Rosa` versus `Rosa_v2`.
- Exact `cfg_type` for each sample, including DNA/RNA and UMI length variant.
- `read_cutoff_UMI_override` and any allele/QC thresholds.
- Safe staging root and symlink naming approval.
- MATLAB/Custom_CARLIN dependency policy.
- Whether Snakemake parse-time `rsync`/template-copy behavior is acceptable in the dry-run staging root.

## Safety

No Snakemake, DARLIN preprocessing, MATLAB, PEAR/FastQC/MultiQC, symlink creation, data movement, raw FASTQ modification, `/ssd` write, or Git commit was performed in this node.
