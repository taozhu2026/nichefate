# README And Docs Audit

Decision: `CLEANUP_REQUIRED_AND_COMPLETED`

## Findings

| Area | Finding | Action |
|---|---|---|
| README | README still used PlanA-ST-only as the main public workflow name and retained lab-log sections. | Rewritten around NicheFate ST-only workflow v1 and functional modules. |
| README | README included long server/raw-data/pipeline command sections that obscured the frozen baseline. | Removed from the GitHub-facing entry point. |
| README | README did not give module-level input/output expectations. | Added a concise functional workflow table. |
| Docs index | No tracked `docs/index.md` existed on `origin/main`. | Added a tracked docs index as the current entry point. |
| Pipeline module index | Legacy M-stage framing was still prominent. | Added a current-entry note and positioned the file as provenance. |
| Production module doc | PlanA naming remained public-facing in title and overview. | Reworded as NicheFate ST-only workflow v1 while retaining legacy provenance. |
| Project status checkpoint | ReviewPack/P_fate checkpoint text could be mistaken for current status. | Marked as superseded / legacy. |
| Quickstart | ReviewPack quickstart could be mistaken for current onboarding. | Marked as superseded / legacy. |
| Reproducibility guide | ReviewPack command framing could be mistaken for current workflow instructions. | Marked as superseded / legacy. |
| DARLIN/barcode | Current docs needed to state future-stage status without lineage claims. | README/docs now state DARLIN/barcode integration is future work. |

## Current Entry Points After Cleanup

- `README.md`
- `docs/index.md`
- `docs/planA_st_only_v1_production_modules.md`
- `docs/pipeline_module_index.md`
- `reports/planA_st_only_v1_index/00_PLAN_A_ST_ONLY_V1_INDEX.md`

## Scope Boundary

No numerical code, raw data, scratch outputs, production matrices, DARLIN data,
or figure binaries were modified.
