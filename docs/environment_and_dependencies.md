# Environment And Dependencies

Generated for ReviewPack-01 on 2026-05-08.

## Current Environments

| Environment | Path | Python | Role |
|---|---|---:|---|
| `omicverse` | `/home/zhutao/software/conda_envs/omicverse` | 3.10.14 | Historical main execution environment for current scripts and tests. |
| `nichefate-gpcca` | `/home/zhutao/software/conda_envs/nichefate-gpcca` | 3.12.13 | Isolated GPCCA validation environment for pyGPCCA/CellRank interface checks. |

The `omicverse` environment name is historical and should not define the
project identity. NicheFate is the project identity. OmicVerse is present as an
installed package/dependency in the historical environment; this checkpoint does
not claim that NicheFate depends on OmicVerse as a framework beyond actual
package availability and usage in existing environment checks.

## Critical Package Versions

Current `omicverse` environment:

| Package | Version |
|---|---:|
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scipy | 1.11.4 |
| scikit-learn | 1.7.2 |
| scanpy | 1.11.5 |
| anndata | 0.11.4 |
| omicverse | 1.7.9 |
| pyarrow | 23.0.1 |
| networkx | 3.4.2 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| pytest | 9.0.2 |
| PyYAML | 6.0.3 |

Current `nichefate-gpcca` environment:

| Package | Version |
|---|---:|
| numpy | 2.4.3 |
| pandas | 2.3.3 |
| scipy | 1.17.1 |
| pygpcca | 1.0.4 |
| cellrank | 2.2.0 |
| scanpy | 1.12.1 |
| anndata | 0.12.11 |

## GPCCA Backend Note

The standard pyGPCCA validation path uses `method="krylov"` where configured.
ReviewPack-01 does not run pyGPCCA, CellRank, or any GPCCA computation.

## Future Environment Recommendation

- `nichefate-core`: main reproducible pipeline environment, targeting Python 3.10.
- `nichefate-gpcca`: isolated GPCCA/CellRank backend validation environment.
- `nichefate-dev`: development, tests, linters, notebooks, and packaging tools.

## Temporary Directory Recommendation

For future heavy runs, set temporary directories away from `/ssd` unless storage
has been explicitly validated:

```bash
export TMPDIR=/home/zhutao/tmp/nichefate
export TMP=/home/zhutao/tmp/nichefate
export TEMP=/home/zhutao/tmp/nichefate
```

Do not write ReviewPack outputs to `/ssd`.

