# GitHub Packaging Plan

Generated for ReviewPack-01 on 2026-05-08.

This plan prepares the workspace for a future GitHub repository. ReviewPack-01
does not initialize Git, commit, move data, or upload outputs.

## Include In GitHub

- `README.md`
- `AGENTS.md`
- `pyproject.toml`
- `requirements.txt`
- `environment.yml`
- `src/`
- `scripts/`
- `configs/`
- `docs/`
- `tests/`

## Exclude From GitHub

- Raw data and external datasets under `/data/zhutao/datasets`
- Scratch outputs under `/home/zhutao/scratch/nichefate`
- Work outputs under `/data/zhutao/work/nichefate`
- `/ssd` paths
- Large or generated scientific files: `*.h5ad`, `*.h5`, `*.hdf5`, `*.zarr/`,
  `*.loom`, `*.npz`, `*.parquet`, large `*.csv`, large `*.tsv`, figures, logs,
  caches, and temporary directories
- Conda environments and local editor metadata

Scratch reports can be cited by path in documentation, but should not be
uploaded wholesale.

## `.gitignore` Recommendations

The current `.gitignore` already excludes Python caches, local environments,
logs, temporary directories, and common large scientific formats. ReviewPack-02
should refine it by:

- Keeping source/config/docs/tests tracked.
- Keeping generated reports and inventories tracked only when they are small
  and intentionally curated.
- Adding explicit scratch/data root patterns if a future Git repo is initialized
  from a parent directory.
- Considering exceptions for small metadata examples only after deliberate
  curation.

## README Layout Recommendation

Future README structure:

1. Project purpose and current checkpoint.
2. Pipeline overview with production module names.
3. Environment setup.
4. Reproducibility entry points.
5. Data availability and large-output policy.
6. Current review status and limitations.
7. Citation/license placeholders if applicable.

ReviewPack-01 only adds a short checkpoint link section and does not rewrite
scientific claims.

## Data Availability Notes

- The MERFISH colitis source data are external and should be referenced through
  the original data provider, not redistributed through this repository.
- Existing local raw data root: `/data/zhutao/datasets/merfish_colitis_moffitt_2024/raw`.
- Existing active scratch output root: `/home/zhutao/scratch/nichefate`.
- Reviewers should receive paths, checksums, and artifact manifests rather than
  GitHub-hosted large outputs.

## Recommended First Commit Structure

When Git is initialized in a later task, use small, reviewable commits:

1. `chore: initialize nichefate source checkpoint`
2. `docs: add reviewpack checkpoint documentation`
3. `chore: add reproducibility manifest and audit inventory`
4. `docs: refine README and packaging notes`

Do not initialize Git or commit in ReviewPack-01.

## Recommended Tags

- `v0.1-pfate-freeze`
- `v0.2-kgpcca-pilot`
- `v0.3-reviewpack-checkpoint`
- `v0.4-darlin-ready`

