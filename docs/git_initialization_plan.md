# Git Initialization Plan

Generated for ReviewPack-02 on 2026-05-08.

This is a plan only. ReviewPack-02 does not run `git init`, create commits, or
push to GitHub.

## Preflight Check

```bash
cd /home/zhutao/projects/nichefate
test ! -d .git
find /ssd -maxdepth 4 \( -iname '*nichefate*' -o -iname '*gpcca*' \)
ps -eo pid,ppid,etime,cmd | rg -i 'k_gpcca|pygpcca|cellrank|terminal|fate|branchsbm|darlin|barcode'
```

If a real analysis process is active, stop the Git initialization workflow and
investigate before proceeding.

## Review `.gitignore`

Confirm the ignore policy excludes raw data, scratch outputs, heavy scientific
artifacts, local generated results, logs, caches, temporary files, notebook
checkpoints, and large matrix/table formats.

Never add:

- `scratch/`, `data/`, `outputs/`, `results/`, `logs/`, `tmp/`
- `.h5ad`, `.npz`, `.npy`, `.parquet`, `.fastq*`, `.bam`, `.cram`
- local conda environments or external data roots

## Initialize And Inspect

```bash
git init
git status --short
```

Do not stage everything blindly. Inspect the file list first.

## First Add Strategy

Recommended first commits:

1. Initialize repository metadata and documentation.
2. Add reproducibility configs and env drafts.
3. Add pipeline scripts/configs/tests.
4. Add legacy mapping and review checkpoint docs.

Suggested first commit message:

```text
chore: initialize nichefate review checkpoint
```

## Tag Plan

Recommended tags after the relevant commits exist:

- `v0.1-pfate-freeze`
- `v0.2-kgpcca-pilot`
- `v0.3-reviewpack-checkpoint`
- `v0.4-darlin-ready`

## GitHub Remote Placeholder

```bash
git remote add origin <future-github-url>
git remote -v
```

Do not push until `.gitignore`, staged files, and large-output exclusions have
been reviewed.
