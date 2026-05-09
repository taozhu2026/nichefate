# DARLIN-02B Best-Supported Candidate Review

## Decision

No final production config is selected in DARLIN-02B.

The corrected locus families are resolved:

- `Brain_E1_all_TA` → Tigre.
- `Brain_E1_all_RA` → Rosa.

But exact official dry-run config values remain unresolved.

## Brain_E1_all_TA / Tigre

Best-supported template status:

`unresolved_between_Tigre_2022_and_Tigre_2022_v2`

Evidence:

- README lists `Tigre_2022_v2` as a short-primer TC candidate.
- README lists `Tigre_2022` as a long-primer candidate.
- Legacy `/data/zhutao/darlin_data/config.yaml` uses `template: Tigre_2022_v2` and `cfg_type: BulkRNA_Tigre_14UMI`.

Limit:

The legacy config is not Meiji E1-specific and cannot decide the Meiji TA production config.

## Brain_E1_all_RA / Rosa

Best-supported template status:

`unresolved_between_Rosa_and_Rosa_v2`

Evidence:

- README lists `Rosa_v2` as a short-primer RC candidate.
- README lists `Rosa` as a long-primer candidate.
- `parse_config_file.m` supports Rosa DNA/RNA 12UMI and 14UMI variants.

Limit:

No Meiji E1-specific historical Rosa config was found in the inspected local evidence.

## Required Review Before DARLIN-01C

Final pre-execution review is required for:

- exact template per sample
- exact `cfg_type` per sample
- read/UMI cutoff
- separate-run layout for TA and RA
- staging root
- MATLAB and side-effect policy
