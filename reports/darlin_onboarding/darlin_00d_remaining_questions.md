# DARLIN-00D Remaining Questions

Generated: 2026-05-09T03:47:54Z

## Required Before DARLIN-01A Execution

- What exact machine-readable key should link Meiji E1 RA/TA barcode samples to Sailu Brain E1 ST output: sample ID, chip position, well, library ID, lane, section ID, or another provider field?
- Is there an original provider sample sheet or demultiplexing sheet outside the currently inspected local files?
- For Sailu Brain E1 ST, which user ST script is the official first entry point, and what are its expected input and output paths?
- For Meiji E1 RA/TA, what official DARLIN `cfg_type`, template/locus, read cutoff, and sample naming convention should be used?
- Do RA and TA denote biological groups, technical sampling regions, library types, or DARLIN capture categories for E1?
- If E1 CA arrives later, should it join the same E1 route as a third barcode component, or should it trigger a separate route update node?

## Not Blocking DARLIN-00D

- Provider has confirmed the first-round route at the route level.
- B1 is not used for first-round barcode+ST onboarding.
- Pancreas E2 is deferred.
- No preprocessing or ST processing was run in this node.
