# Lineage Insertion Points

Lineage-aware NicheFate adds barcode-supported evidence to the shared
NicheFate substrate. It does not replace NicheEncoder or the M0-M2.5 workflow.

## Where Lineage Enters

1. M0-M2.5 can run without lineage evidence.
2. E1 adapts processed lineage evidence to the same cellbin identity used by
   the shared substrate.
3. E2 calls validated DARLIN-style joint clones before clone x niche
   integration.
4. E3 aggregates joint clone composition into cellbins, tiles, groups, and
   metaniches.
5. E4 can expose clone composition as additional representation channels for
   future PlanA or PlanB designs.

## Relationship To NicheEncoder

NicheEncoder remains the public M2 module for building niche representation
matrices from spatial transcriptomics-derived niche features. Lineage evidence
can add feature blocks or clone composition variables after barcode evidence is
validated, but it does not replace the encoder.

## Downstream Use

Clone x niche composition can feed:

- QC-aware niche characterization,
- lineage-aware representation channels,
- future PlanA design interfaces,
- future PlanB branch/clone-aware designs.

Clone evidence is a state variable. Directionality still requires time,
perturbation, or another valid prior.
