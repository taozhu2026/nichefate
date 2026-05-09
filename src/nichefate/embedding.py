"""Embedding helpers for M0."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def compute_pca_m0(
    adata: Any,
    n_comps: int = 50,
    scale: bool = True,
    variance_csv: str | Path | None = None,
) -> Any:
    """Compute PCA for M0 and store it in ``obsm['X_pca_m0']``."""

    import pandas as pd
    import scanpy as sc

    max_comps = max(1, min(int(n_comps), adata.n_obs - 1, adata.n_vars - 1))
    working = adata.copy()
    if scale:
        sc.pp.scale(working, max_value=10)
    sc.tl.pca(working, n_comps=max_comps, svd_solver="arpack")
    adata.obsm["X_pca_m0"] = working.obsm["X_pca"].copy()
    variance_ratio = working.uns["pca"]["variance_ratio"]
    adata.uns["pca_m0"] = {
        "variance": working.uns["pca"]["variance"].tolist(),
        "variance_ratio": variance_ratio.tolist(),
        "n_comps": max_comps,
        "scaled": bool(scale),
    }
    if variance_csv is not None:
        output_path = Path(variance_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "component": list(range(1, len(variance_ratio) + 1)),
                "variance_ratio": variance_ratio,
            }
        ).to_csv(output_path, index=False)
    return adata


def compute_embeddings(_adata: object, *, method: str) -> object:
    """Compute or validate embeddings for an AnnData object."""

    if method.lower() == "harmony":
        raise NotImplementedError("Harmony is disabled for M0 v1.")
    raise NotImplementedError(f"Embedding method is not implemented yet: {method}")
