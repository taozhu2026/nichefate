# .gitignore and Data Exclusion Audit

- Generated: 2026-05-16T02:05:26.193095+00:00
- All required patterns present: `True`

| pattern | present |
| --- | --- |
| `scratch/` | True |
| `data/` | True |
| `raw/` | True |
| `processed/` | True |
| `*.h5ad` | True |
| `*.fastq` | True |
| `*.fastq.gz` | True |
| `*.fq.gz` | True |
| `*.parquet` | True |
| `*.npz` | True |
| `__pycache__/` | True |
| `.pytest_cache/` | True |
| `*.log` | True |
| `*.out` | True |
| `*.err` | True |
| `*.tmp` | True |
| `ssd/` | True |
| `darlin_evidence/` | True |
| `barcode_evidence/` | True |

Absolute /home/zhutao/scratch and /ssd outputs are outside the repository; repo-local scratch/ and ssd/ are ignored and staging audit checks for forbidden absolute path strings.

No source code, configs, tests, docs, or freeze reports are ignored by the added patterns.
