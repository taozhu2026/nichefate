# DARLIN-00D Provider-Confirmed E1 Route

Generated: 2026-05-09T03:47:54Z

DARLIN-00D updates the onboarding state after provider confirmation. This node is documentation and audit only. No DARLIN preprocessing, Snakemake run, ST processing, NicheFate run, GPCCA/CellRank run, data movement, symlink creation, raw-file rename, `/ssd` write, or Git commit was performed.

## Confirmed First-Round Route

- `selected_first_route`: `Brain_031319A_E1_RA_TA_ST`
- `route_status`: `provider_confirmed_for_first_round_development`
- ST / expression component: `Sailu_Brain_031319A_E1_ST`
- Barcode / DARLIN component: `Meiji_Brain_031319A_E1_RA_TA`
- Missing barcode component: `E1_CA`
- Deferred components: `Sailu_Brain_B1_CA_TA_RA_barcode_only`, `Pancreas_E2_ST`, public/legacy references

## Interpretation

The first onboarding route is now Brain `031319A-E1`: Meiji E1 RA/TA barcode/DARLIN reads plus Sailu Brain E1 ST data. This supersedes the DARLIN-00C status of `E1_matched_pending_provider_confirmation`.

The confirmation is route-level confirmation. It resolves the first-round subset choice, but it does not by itself provide a machine-readable cell, spot, well, library, or lane key linking Meiji RA/TA barcode data to Sailu ST output. That exact key must still be extracted from provider documents or confirmed in DARLIN-01A before barcode evidence can enter NicheFate integration.

## Deferred Routes

- B1 remains separate from E1. Sailu Brain B1 has barcode data only and no confirmed ST/expression counterpart, so it must not be combined with E1 ST in first-round barcode+ST NicheFate onboarding.
- B1 may later be used for barcode-only official preprocessing smoke testing, or as a separate matched route if provider-confirmed B1 ST/expression data becomes available.
- Pancreas E2 is deferred because it is a different tissue and chip position from the Brain E1 first route.
- E1 CA is absent in the current first-round route and should be treated as missing/deferred until updated barcode data becomes available.

## DARLIN-01A Gate

DARLIN-01A may construct an E1 sample sheet and dry-run plan only after this route state is reviewed. The first sample sheet should use Meiji E1 RA/TA for barcode preprocessing and Sailu E1 ST for the ST processing plan. It should not include B1, Pancreas E2, or E1 CA unless new provider information is supplied.
