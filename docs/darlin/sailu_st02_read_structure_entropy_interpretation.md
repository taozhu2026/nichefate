# Sailu-ST-02 Read-Structure Entropy Interpretation

## Scope

This is bounded FASTQ read-structure discovery for `Sailu_0313_19A_mixed_ST` with `tissue_status = mixed_unresolved`. Brain/Pancreas split is not required before preliminary raw ST processing.

## Sampling

The first 10,000 read pairs were sampled from each of four E_PE100_100 lane pairs. No full FASTQ scan was performed.

## Entropy Summary

- R1 aggregate mean per-position entropy: `1.640` bits.
- R1 high-diversity position fraction: `38.75%`.
- R1 low-diversity/motif position fraction: `18.00%`.
- R2 aggregate mean per-position entropy: `1.947` bits.
- R2 high-diversity position fraction: `76.75%`.
- R2 low-diversity/motif position fraction: `0.00%`.

## Interpretation Rules Used

- High entropy across most positions supports transcript-like sequence, but does not prove gene alignment readiness.
- Short high-diversity leading windows can be barcode/UMI-like only if there is additional evidence such as a whitelist, flanking motif, or provider read-structure contract.
- Low-diversity positions can indicate constant/linker motifs, but no biological role is assigned without chemistry documentation.

## Current Interpretation

R1 role: `likely_barcode_umi_read` with `low` confidence.

R2 role: `mixed_or_unknown` with `low` confidence.

Candidate barcode/UMI region found: `true`.

The entropy profiles alone are not sufficient to produce an expression matrix or coordinate table for M0.
