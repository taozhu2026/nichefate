# Quickstart

Generated for ReviewPack-02 on 2026-05-08.

This quickstart is conservative. It is for inspecting the checkpoint and running
lightweight checks, not for launching full production analyses.

## Repository Layout

```text
nichefate/
  configs/                 lightweight pipeline and reproducibility configs
  docs/                    review, installation, quickstart, and packaging docs
  envs/                    draft conda environment YAMLs
  reports/review_pack/     machine-readable review-pack summaries
  scripts/                 pipeline entry points
  src/                     reusable package modules
  tests/                   lightweight test coverage
```

Large external outputs live outside the repository, mainly under
`/home/zhutao/scratch/nichefate` and `/data/zhutao/datasets`.

## Inspect ReviewPack Docs

Start with:

```bash
sed -n '1,160p' docs/project_status_checkpoint.md
sed -n '1,160p' docs/pipeline_module_index.md
sed -n '1,160p' docs/reproducibility_guide.md
```

ReviewPack audit files:

```bash
python -m json.tool reports/review_pack/reviewpack_01_audit_summary.json
python -m json.tool reports/review_pack/reviewpack_02_packaging_summary.json
```

## Run Lightweight Tests

After creating a suitable environment, run only lightweight tests by default:

```bash
conda activate nichefate-core
python -m pytest -q tests
```

Do not run production scripts as part of quickstart validation.

## Inspect The Module Manifest

```bash
python - <<'PY'
from pathlib import Path
import yaml

manifest = yaml.safe_load(Path("configs/reproducibility/reviewpack_01_module_manifest.yaml").read_text())
for module in manifest["modules"]:
    print(module["legacy_stage_name"], module["status"], "safe_to_run=", module["safe_to_run_in_reviewpack"])
PY
```

All heavy pipeline modules should be treated as inspect-only during ReviewPack.

## Avoid Accidental Heavy Runs

Do not run commands that start:

- M0/M1/M2/M3/M4/M5 production scripts
- GPCCA, pyGPCCA, or CellRank
- terminal-state design
- fate-probability propagation
- P_fate propagation
- BranchSBM
- barcode or DARLIN preprocessing

## Local External Output Roots

- `/home/zhutao/scratch/nichefate/m0`
- `/home/zhutao/scratch/nichefate/m1`
- `/home/zhutao/scratch/nichefate/m2`
- `/home/zhutao/scratch/nichefate/m3`
- `/home/zhutao/scratch/nichefate/m3_v2`
- `/home/zhutao/scratch/nichefate/m4a`
- `/home/zhutao/scratch/nichefate/m4c`
- `/home/zhutao/scratch/nichefate/m4e`
- `/home/zhutao/scratch/nichefate/planA_freeze`
- `/home/zhutao/scratch/nichefate/k_gpcca_revision`

## Future Toy Example

A small toy dataset and fixture-based smoke test can be added later under
`examples/` or `tests/fixtures/`. No toy example is included in ReviewPack-02.
