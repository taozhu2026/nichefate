# DARLIN-01A E1 Integration Matching Contract

Generated: 2026-05-09T03:57:32Z

## Current Status

- E1 route is provider-confirmed at route level.
- Barcode component can be planned for Meiji E1 RA/TA.
- Sailu ST remains a shared-folder route-level candidate until the E1/E2 split key is found.
- Exact machine-readable matching key between processed Meiji barcode outputs and Sailu ST metadata has not been found.

Barcode-informed NicheFate cannot start until official DARLIN preprocessing outputs and ST metadata expose a reviewed matching key.

## Possible Matching Levels

- Sample-level only: useful for route tracking, not sufficient for cell/spot-level barcode-informed transition evidence.
- Library-level: may support library-consistency checks if library IDs are available.
- Spot-level: suitable only if barcode evidence can be assigned to spatial spots or anchors.
- Cell-level: suitable only if barcode cell IDs can be matched to expression cell IDs.
- Spatial barcode-level: suitable only if Sailu spatial barcodes can be linked directly to DARLIN barcode evidence.

## Required Decision Before Adapter Work

Before BarcodeEvidenceAdapter or TransitionEvidence[barcode] work begins, the project must decide which matching level is biologically valid for this dataset and provide the corresponding key in both processed barcode output and ST metadata.

If only sample-level matching is available, NicheFate should remain pseudo-only for fate inference and use barcode evidence only for sample-level validation or another explicitly approved method.
