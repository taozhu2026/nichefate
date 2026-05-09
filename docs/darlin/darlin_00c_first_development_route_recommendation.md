# DARLIN-00C First Development Route Recommendation

Generated: 2026-05-09T03:20:35Z

## Recommendation

- Selected route recommendation: `E1_matched_pending_provider_confirmation`
- Route confidence: `moderate`

## Decision Logic

- If `Brain 031319A-E1` Meiji RA/TA and `Brain 031319A-E1` Sailu ST are both present and chip-position matching is credible, use `E1_matched` as the first route, pending provider confirmation of the matching key.
- If E1 matching remains uncertain but `B1` CA/TA/RA stays clear, use `B1_barcode_only` only as an official preprocessing smoke route, not as barcode-informed NicheFate.
- Do not use `B1 + E1` as a matched route unless the provider explicitly confirms a B1/E1 relationship.

## Current Outcome

- Local presence checks support `E1_matched` as the leading development route because rows 1, 2, and 3 all resolve locally and share `Brain` plus chip position `031319A-E1` in the manual table.
- This remains a pending-confirmation route because the local files do not provide the actual cross-company matching key.
