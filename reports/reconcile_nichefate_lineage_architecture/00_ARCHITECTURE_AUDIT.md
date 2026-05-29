# Architecture Audit

## Result

- Decision label: `NICHEFATE_ARCHITECTURE_RECONCILIATION_READY`
- Starting branch: `freeze/lineage-aware-nichefate-v1`
- Target branch: `docs/reconcile-nichefate-lineage-architecture`

## Audit Answers

1. Does the current README make lineage-aware baseline look like the whole
   project?

   Yes. The prior README title was `NicheFate Lineage-Aware Baseline v1` and
   presented lineage modules as the public entry point. It did not clearly
   surface NicheFate as a broader spatial niche dynamics framework.

2. Are M0-M2.5 and PlanA-ST clearly visible?

   No. The M0-M2.5 and PlanA-ST documentation existed, but it was not prominent
   in the top-level README or documentation index.

3. Are ST-only and lineage-aware modes presented as parallel evidence regimes?

   No. The prior public docs emphasized the lineage-aware freeze and did not
   clearly describe ST-only / lineage-free and ST + lineage-aware modes as
   parallel regimes of one framework.

4. Are L126-specific names leaking into generic module names?

   Partially. Generic package names were already present under
   `src/nichefate/lineage/` and `src/nichefate/darlin/`, but public docs still
   centered the L126 lineage freeze. L126-specific `planC_l126_*` scripts remain
   as provenance and compatibility entrypoints.

5. Which docs need repair?

   The primary repairs are `README.md`, `docs/index.md`,
   `docs/pipeline_module_index.md`, the lineage module docs under
   `docs/modules/`, and the L126 benchmark page. A new
   `docs/nichefate_architecture.md` document is needed to make the integrated
   architecture explicit.

## Repair Strategy

- Present one NicheFate framework with shared M0-M2.5 substrate modules.
- Present E0 and E1-E3 as parallel evidence regimes.
- Present PlanA and PlanB as dynamics engines that consume the shared substrate
  plus available evidence.
- Keep ST-only baseline prominent and not legacy.
- Keep L126 as the lineage-aware benchmark, not a method name.
