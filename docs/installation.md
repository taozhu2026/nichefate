# Installation

Generated for ReviewPack-02 on 2026-05-08.

These instructions prepare a future clean checkout. The environment YAML files
are drafts, not final locked files, until regression validation is complete.

## Recommended Clone Location

Use a source-code location separate from raw data and scratch outputs:

```bash
mkdir -p /home/zhutao/projects
cd /home/zhutao/projects
# git clone <future-github-url> nichefate
cd nichefate
```

Large data and scientific outputs are not stored in GitHub. Keep raw data under
external data roots such as `/data/zhutao/datasets` and active outputs under
scratch/work roots such as `/home/zhutao/scratch/nichefate`.

## Create `nichefate-core`

```bash
conda env create -f envs/nichefate-core.yml
conda activate nichefate-core
```

`nichefate-core` is the recommended future name for the main pipeline
environment. The historical `omicverse` environment was used during early
execution and does not define the project identity.

## Create `nichefate-gpcca`

```bash
conda env create -f envs/nichefate-gpcca.yml
conda activate nichefate-gpcca
```

This environment is for isolated pyGPCCA/CellRank validation only. Standard
pyGPCCA sparse-matrix runs should use `method="krylov"` where configured.

Before any heavy GPCCA validation, place temporary files away from `/ssd`:

```bash
export TMPDIR=/home/zhutao/tmp/k_gpcca
export TMP=/home/zhutao/tmp/k_gpcca
export TEMP=/home/zhutao/tmp/k_gpcca
```

ReviewPack-02 does not run pyGPCCA, CellRank, or GPCCA.

## Create `nichefate-dev`

```bash
conda env create -f envs/nichefate-dev.yml
conda activate nichefate-dev
```

Use this environment for tests, formatting, notebooks, documentation, and
packaging work.

## Notes

- Do not modify the historical `omicverse` environment as part of packaging.
- Do not write large raw data, scratch outputs, or generated matrices into the
  repository.
- Treat the YAML files as drafts until full regression validation is completed.
