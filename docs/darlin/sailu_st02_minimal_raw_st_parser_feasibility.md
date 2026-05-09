# Sailu-ST-02 Minimal Raw ST Parser Feasibility

## Parser Feasibility Decision

Raw FASTQ-based mixed ST preprocessing status: `possible_but_unconfirmed`.

## Candidate Evidence

- Candidate spatial barcode region found: `true`.
- Candidate UMI region found: `true`.
- Candidate transcript read found: `true`.
- R1 role inference: `likely_barcode_umi_read`.
- R2 role inference: `mixed_or_unknown`.

## Requirements For A Tiny Parser

A minimal parser would need:

1. a confirmed read/position rule for spot barcode extraction;
2. a confirmed read/position rule for UMI extraction;
3. a barcode whitelist or accepted correction policy if the spatial barcode is not directly inferable;
4. a coordinate or spatial map linking spot barcodes to positions;
5. a transcript read assignment rule;
6. a reference genome/transcriptome and gene annotation;
7. an aligner or pseudoaligner;
8. a matrix writer and M0-compatible metadata writer.

## Still Impossible Without Documentation

The current FASTQ statistics cannot define final spot coordinates, barcode correction, UMI deduplication, gene counting, or Brain/Pancreas labels. Those require Sailu/Salus chemistry documentation, a whitelist/coordinate map, and a validated processing implementation.
