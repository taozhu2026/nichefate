"""Quality-control summaries and report writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_anndata(adata: Any) -> dict[str, object]:
    """Summarize an AnnData object for QC reporting."""

    return {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_columns": list(adata.obs.columns),
        "var_names_count": int(len(adata.var_names)),
        "layers": list(adata.layers.keys()),
        "obsm_keys": list(adata.obsm.keys()),
        "uns_keys": list(adata.uns.keys()),
        "x_type": type(adata.X).__name__,
    }


def compute_obs_summary(adata: Any) -> dict[str, dict[str, int]]:
    """Compute value-count summaries for standard M0 obs fields."""

    summaries: dict[str, dict[str, int]] = {}
    for field in (
        "time",
        "time_day",
        "dataset_part",
        "mouse_id",
        "slice_id",
        "cell_type_l1",
        "cell_type_l2",
        "cell_type_l3",
    ):
        if field in adata.obs:
            counts = adata.obs[field].value_counts(dropna=False).sort_index()
            summaries[field] = {str(key): int(value) for key, value in counts.items()}
    return summaries


def write_json_report(obj: object, path: str | Path) -> Path:
    """Write a JSON report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=str)
    return output_path


def write_markdown_report(lines: list[str], path: str | Path) -> Path:
    """Write a Markdown report from preformatted lines."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path
