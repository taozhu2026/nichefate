# nichefate project context

This project builds a generalizable spatial niche-fate modeling framework.

Current status:
- M0 completed.
- M1 full by-slice anchor-centered multi-scale niche construction completed.
- M2 full by-slice niche representation completed.
- M3-v2 production planning generated.
- Official M3-v2 mode: constrained_v1prior_sharpening.
- Locked parameters:
  - lambda = 1.0
  - tau_scale = 0.5
  - top_k = 10
  - G_barcode = 1

Key paths:
- Project code: /home/zhutao/projects/nichefate
- M2 outputs: /home/zhutao/scratch/nichefate/m2/by_slice
- M2 summary: /home/zhutao/scratch/nichefate/m2/reports/m2_full_by_slice_summary.{csv,md}
- M2 schema: /home/zhutao/scratch/nichefate/m2/reports/m2_full_feature_schema.json
- M3-v2 production plan: /home/zhutao/scratch/nichefate/m3_v2_production_plan

Design constraints:
- Do not hard-code Moffitt/Cadinu/DSS/colon-specific logic.
- Keep M1/M2/M3/M4/M5 as general spatial transcriptomics algorithms.
- Barcode evidence should be integrated after official/lab-standard DARLIN preprocessing.
- DARLIN barcode raw reads should not be preprocessed from scratch by Codex unless explicitly requested.
