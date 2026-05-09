# DARLIN-00C Data Relationship Resolution

Generated: 2026-05-09T03:20:35Z

## Relationship Graph

- `Brain 031319A-E1 Meiji RA/TA` resolves to paired RA and TA FASTQs under `/data/zhutao/nichefate_data/MeiJi`.
- `Brain 031319A-E1 Sailu ST` and `Pancreas 031319A-E2 Sailu ST` both point to the same shared Sailu folder `202604271751_2211180001_B_1_ZHY_XBB_E_PE100_100`.
- `Brain 031319A-B1 Sailu CA/TA/RA` all point to the same shared Sailu folder `202604271754_C2302270016_A_1_ZHY_XBB_R_PE100_250`, with distinction supplied by manual index metadata rather than filename-level demultiplexing.

## Hypothesis Classification

| hypothesis | classification | rationale | key risk |
| --- | --- | --- | --- |
| `E1_matched` | recommended pending provider confirmation | Same tissue and chip position (`Brain`, `031319A-E1`) across Meiji RA/TA and Sailu ST. | Cross-company match is plausible but not yet verified by provider metadata. |
| `B1_barcode_only` | valid barcode-only candidate | Same folder, same tissue, same chip position, explicit manual CA/TA/RA indices. | No confirmed B1 ST/expression counterpart. |
| `B1_to_E1_forced_match` | not recommended | Same Sailu provenance is weaker than chip-position consistency. | `031319A-B1` and `031319A-E1` are different chip positions. |
| `Pancreas_E2_deferred` | deferred | Different tissue and chip position from the Brain-first routes. | Shared Sailu folder likely reflects sequencing batch co-delivery, not biological matching. |

## Resolution

- DARLIN-00C should explicitly replace the earlier default bias toward `B1` as the first matched route.
- The most credible first matched hypothesis is `E1_matched`, but only as a pending-provider-confirmation route.
- `B1` remains useful as a preprocessing smoke path only if E1 matching cannot be confirmed.
