# Server Layout

`nichefate` uses separate roots for code, raw data, and temporary outputs.

## Active Paths

- Code root: `/home/zhutao/projects/nichefate`
- Raw dataset root: `/data/zhutao/datasets/merfish_colitis_moffitt_2024/raw`
- External reference root: `/data/zhutao/datasets/merfish_colitis_moffitt_2024/external`
- Temporary M0 output root: `/data/zhutao/work/nichefate/m0`
- Cache root: `/data/zhutao/work/nichefate/cache`
- Temporary-file root: `/data/zhutao/work/nichefate/tmp`

## Disabled Future Path

- Intended high-I/O M0 root: `/ssd/zhutao/nichefate/m0`

`/ssd` is currently full and should not be used for active outputs. Keep M0
working outputs under `/data` until `/ssd` has sufficient free space.

## Storage Rules

- Keep source code, configs, docs, tests, and lightweight notebooks under
  `/home/zhutao/projects/nichefate`.
- Keep raw datasets under `/data/zhutao/datasets`.
- Keep temporary M0 outputs under `/data/zhutao/work/nichefate` for now.
- Do not write large files to `/`, container overlay paths, or the code root.
- Do not create active `/ssd` outputs until the storage issue is resolved.
- M0 v1 should be run with `conda run -n omicverse ...`; do not create a new
  environment or modify the existing one unless that is approved separately.
