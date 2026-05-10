# DARLIN-02B Protocol-Based Config Resolution

Timestamp: 2026-05-10T03:24:10Z

Branch: `darlin-onboarding`

## Scope

This update applies only to the Meiji E1 RA/TA DARLIN barcode preprocessing track. Sailu-ST work is paused and not modified here.

## Authoritative Protocol Choices

The Nature Protocols DARLIN parameter table provided by the user is now the authoritative source for the current Meiji RA/TA cfg_type/template choices.

- `Brain_E1_all_TA` maps to `TA / TC`, official family `Tigre`, `cfg_type = BulkRNA_Tigre_14UMI`, `template = Tigre_2022_v2`, `read_cutoff_UMI_override = [3, 10]`.
- `Brain_E1_all_RA` maps to `RA / RC`, official family `Rosa / Rosa26`, `cfg_type = BulkRNA_Rosa_14UMI`, `template = Rosa_v2`, `read_cutoff_UMI_override = [3, 10]`.

## CA Reference

If `CA / CC` appears later, the protocol table points to Col1a1 / cCARLIN with `cfg_type = BulkRNA_12UMI`, `template = cCARLIN`, and `read_cutoff_UMI_override = [3, 10]`. Current Meiji E1 CA remains absent/deferred and is not part of the RA/TA 01C plan.

## Impact On DARLIN-02B

The previous candidate matrix remains an audit trail. The execution-facing choice for current RA/TA is now protocol-resolved. The cfg_type/template/cutoff blocker is resolved for `Brain_E1_all_TA` and `Brain_E1_all_RA`.

Remaining blockers are engineering/runtime blockers: safe scratch staging, raw FASTQ symlink naming, dependency environment activation, MATLAB/Custom_CARLIN policy, and Snakemake parse-time side-effect review.
