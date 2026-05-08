# M0 Design

M0 is the first implementation milestone for the `nichefate` project. Its goal
is to prepare time-anchored MERFISH colitis AnnData objects for later
NicheFate modules:

- M1 anchor-centered niche construction
- M2 niche representation
- M3 pseudo-lineage coupling
- M4 terminal niche macrostate definition
- M5 fate probability inference

M0 does not implement M1 and does not claim true lineage. The Moffitt colitis
MERFISH dataset has no lineage barcode, so downstream work is pseudo-lineage
inference only.

## Input Policy

Expected raw files:

- `adata.h5ad`
- `adata_day35.h5ad`
- `README.md`

Optional:

- `ligand_receptor_pair_masterlist.csv`

Dryad lists `adata.h5ad` as 17.96 GB and `adata_day35.h5ad` as 1.51 GB, so the
required core download is about 19.5 GB plus tiny README/LR files. The complete
Dryad dataset is listed at 108.70 GB and should not be downloaded for M0.

Avoid for the first M0 workflow:

- `X.csv`
- `X_raw.csv`
- transcript-level RNA metadata CSVs
- all supplementary CSV sets

These avoided files are large and mostly redundant for the initial h5ad-based
workflow. Do not download the full Dryad dataset unless explicitly requested.

## Planned Pipeline Stages

1. `m0_00_check_environment.py`: validate paths, package availability, and file policy.
2. `m0_01_inspect_raw_anndata.py`: inspect raw h5ad structure.
3. `m0_02_build_m0_anndata.py`: build a canonical M0 AnnData object.
4. `m0_03_compute_embeddings.py`: compute or validate embeddings.
5. `m0_04_build_spatial_graphs.py`: construct spatial neighborhood graphs.
6. `m0_05_export_m0_objects.py`: export reusable M0 artifacts.
7. `m0_06_make_qc_report.py`: generate QC summaries and reports.

M0 v1 uses `scanpy`, `anndata`, `sklearn`, `scipy`, and related core packages
from the existing `omicverse` conda environment. It intentionally does not
depend on `squidpy`, `spatialdata`, or `harmonypy`.
