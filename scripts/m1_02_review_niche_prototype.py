#!/usr/bin/env python
"""Review bounded M1 niche prototype outputs and write QC reports."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from nichefate.io import load_config
from nichefate.niche_qc import (
    composition_columns,
    composition_sum_qc,
    dominant_composition,
    estimate_full_m1_storage,
    load_neighbor_metadata,
    neighbor_raw_bytes_from_metadata,
    summarize_distribution,
    summarize_feature_integrity,
    validate_neighbor_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m1_niche_construction.yaml")
    return parser.parse_args()


def _paths(config: dict) -> dict[str, Path]:
    return {
        key: Path(value)
        for key, value in config["paths"].items()
        if isinstance(value, str)
    }


def _format_number(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(float(value)):
            return f"{float(value):.6g}"
        return str(value)
    text = str(value)
    return text if len(text) <= 90 else text[:87] + "..."


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    max_rows: int = 20,
) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    view = frame.copy()
    if columns is not None:
        view = view[[column for column in columns if column in view.columns]]
    if len(view) > max_rows:
        view = view.head(max_rows)
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_format_number(row[column]) for column in headers) + " |")
    return lines


def _format_bytes(num_bytes: float | int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _read_slice_sizes(m0_by_slice_dir: Path) -> pd.DataFrame:
    try:
        import anndata as ad
    except ImportError:
        return pd.DataFrame(columns=["slice_file", "slice_id", "time", "n_obs"])

    rows = []
    for path in sorted(m0_by_slice_dir.glob("*.m0.h5ad")):
        data = ad.read_h5ad(path, backed="r")
        try:
            rows.append(
                {
                    "slice_file": path.name,
                    "slice_path": str(path),
                    "slice_id": str(data.obs["slice_id"].iloc[0])
                    if "slice_id" in data.obs
                    else path.stem,
                    "time": str(data.obs["time"].iloc[0]) if "time" in data.obs else "",
                    "n_obs": int(data.n_obs),
                }
            )
        finally:
            if hasattr(data, "file"):
                data.file.close()
    return pd.DataFrame(rows)


def _slice_n_obs_map(slice_sizes: pd.DataFrame) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for _, row in slice_sizes.iterrows():
        mapping[str(row["slice_file"])] = int(row["n_obs"])
        mapping[str(row["slice_id"])] = int(row["n_obs"])
    return mapping


def _prototype_wall_seconds(report_path: Path) -> float | None:
    if not report_path.exists():
        return None
    match = re.search(r"Wall seconds:\s*([0-9.]+)", report_path.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else None


def _section_frame(section: str, frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "section", section)
    return output


def _write_qc_csv(path: Path, frames: list[tuple[str, pd.DataFrame]]) -> None:
    output_frames = []
    for section, frame in frames:
        if not frame.empty:
            output_frames.append(_section_frame(section, frame))
    if output_frames:
        pd.concat(output_frames, ignore_index=True, sort=False).to_csv(path, index=False)
    else:
        pd.DataFrame({"section": []}).to_csv(path, index=False)


def _write_optional_figures(features: pd.DataFrame, figures_dir: Path) -> tuple[list[Path], str | None]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        return [], str(exc)

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    scales = list(features["scale"].drop_duplicates()) if "scale" in features else []

    def boxplot(column: str, title: str, filename: str) -> None:
        if column not in features or not scales:
            return
        data = [
            pd.to_numeric(features.loc[features["scale"] == scale, column], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
            for scale in scales
        ]
        if not any(len(values) for values in data):
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        try:
            ax.boxplot(data, tick_labels=scales, showfliers=False)
        except TypeError:  # pragma: no cover - compatibility with older matplotlib
            ax.boxplot(data, labels=scales, showfliers=False)
        ax.set_title(title)
        ax.set_xlabel("scale")
        ax.set_ylabel(column)
        fig.tight_layout()
        path = figures_dir / filename
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figure_paths.append(path)

    boxplot("n_neighbors", "Prototype n_neighbors by scale", "m1_qc_n_neighbors_by_scale.png")
    boxplot(
        "mean_neighbor_distance",
        "Prototype mean neighbor distance by scale",
        "m1_qc_mean_neighbor_distance_by_scale.png",
    )

    entropy_columns = [column for column in ("ct_l1_entropy", "ct_l2_entropy", "ct_l3_entropy") if column in features]
    if entropy_columns and scales:
        entropy_means = (
            features.groupby("scale", observed=True)[entropy_columns]
            .mean(numeric_only=True)
            .reindex(scales)
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        entropy_means.plot(kind="bar", ax=ax)
        ax.set_title("Prototype entropy means by scale")
        ax.set_xlabel("scale")
        ax.set_ylabel("entropy")
        fig.tight_layout()
        path = figures_dir / "m1_qc_entropy_by_scale.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figure_paths.append(path)

    return figure_paths, None


def _choose_pilot_slice(
    slice_sizes: pd.DataFrame,
    prototype_slice_files: set[str],
    prototype_times: set[str],
) -> pd.Series | None:
    if slice_sizes.empty:
        return None
    candidates = slice_sizes.loc[~slice_sizes["slice_file"].isin(prototype_slice_files)].copy()
    if candidates.empty:
        candidates = slice_sizes.copy()
    if "time" in candidates and prototype_times:
        non_prototype_times = candidates.loc[~candidates["time"].astype(str).isin(prototype_times)]
        if not non_prototype_times.empty:
            candidates = non_prototype_times
    candidates = candidates.sort_values(["n_obs", "slice_file"])
    medium = candidates.loc[candidates["n_obs"] >= 15000]
    if not medium.empty:
        return medium.iloc[0]
    return candidates.iloc[min(len(candidates) - 1, max(0, len(candidates) // 2))]


def _scan_terms(path: Path, terms: tuple[str, ...]) -> list[str]:
    if not path.exists():
        return []
    hits = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if any(term in line for term in terms):
            hits.append(f"{path.relative_to(PROJECT_ROOT)}:{line_no}: {line.strip()}")
    return hits


def _write_feature_qc_report(
    path: Path,
    features: pd.DataFrame,
    feature_path: Path,
    neighbor_path: Path,
    integrity: pd.DataFrame,
    composition_qc: pd.DataFrame,
    composition_counts: pd.DataFrame,
    dominant: pd.DataFrame,
    distributions: dict[str, pd.DataFrame],
    neighbor_validation: pd.DataFrame,
    npz_keys: list[str],
    figure_paths: list[Path],
    figure_error: str | None,
) -> None:
    scale_count = int(features["scale"].nunique()) if "scale" in features else 0
    anchor_key = ["slice_id", "anchor_index"] if "slice_id" in features else ["slice_file", "anchor_index"]
    unique_anchors = int(features[anchor_key].drop_duplicates().shape[0])
    duplicate_rows = int(features.duplicated(anchor_key + ["scale"]).sum())
    scale_rows_per_anchor = features.groupby(anchor_key, observed=True)["scale"].nunique()
    anchors_with_exact_scales = int((scale_rows_per_anchor == scale_count).sum())
    anchors_without_exact_scales = int((scale_rows_per_anchor != scale_count).sum())
    rows_per_slice = (
        features.groupby("slice_file", observed=True).size().reset_index(name="rows")
        if "slice_file" in features
        else pd.DataFrame()
    )
    rows_per_scale = (
        features.groupby("scale", observed=True).size().reset_index(name="rows")
        if "scale" in features
        else pd.DataFrame()
    )
    d35_note = []
    if "time" in features and "ct_l2__na" in features:
        day35 = features["time"].astype(str) == "D35"
        if bool(day35.any()):
            d35_note.append(
                "Day35 Tier2 fallback: rows labeled D35 have mean `ct_l2__na` "
                f"{features.loc[day35, 'ct_l2__na'].mean():.3f}. "
                "This is expected from source metadata and should not be read as biological signal."
            )

    lines = [
        "# M1 Prototype Feature QC",
        "",
        "This report reviews the bounded M1 prototype only. Full M1 was not run.",
        "",
        "## Inputs",
        "",
        f"- Feature table: `{feature_path}`",
        f"- Neighbor index: `{neighbor_path}`",
        f"- NPZ keys: {len(npz_keys)} total",
        "",
        "## Shape And Anchor Integrity",
        "",
        f"- Feature table shape: {features.shape[0]} rows x {features.shape[1]} columns",
        f"- Unique anchors: {unique_anchors}",
        f"- Slices: {features['slice_file'].nunique() if 'slice_file' in features else 0}",
        f"- Scales: {scale_count}",
        f"- Duplicated anchor/scale rows: {duplicate_rows}",
        f"- Anchors with exactly {scale_count} scale rows: {anchors_with_exact_scales}",
        f"- Anchors without exactly {scale_count} scale rows: {anchors_without_exact_scales}",
        "",
        "### Rows Per Slice",
        "",
        *_markdown_table(rows_per_slice),
        "",
        "### Rows Per Scale",
        "",
        *_markdown_table(rows_per_scale),
        "",
        "## Missing And Infinite Values By Feature Group",
        "",
        *_markdown_table(
            integrity,
            columns=[
                "feature_group",
                "n_columns",
                "missing_values",
                "missing_fraction",
                "infinite_values",
                "infinite_fraction",
                "columns_with_missing",
                "columns_with_infinite",
            ],
            max_rows=30,
        ),
        "",
        "## Distribution Checks",
        "",
    ]
    for name, frame in distributions.items():
        lines.extend([f"### {name}", ""])
        lines.extend(_markdown_table(frame, max_rows=30))
        lines.append("")

    lines.extend(
        [
            "## Composition QC",
            "",
            "### Composition Column Counts",
            "",
            *_markdown_table(composition_counts),
            "",
            "### Composition Row Sums",
            "",
            *_markdown_table(composition_qc, max_rows=10),
            "",
            "### Dominant Composition Labels By Scale",
            "",
            *_markdown_table(dominant, max_rows=45),
            "",
            "Cell-type composition biological sanity checks should prioritize `cell_type_l1` and `cell_type_l3`.",
            "",
        ]
    )
    if d35_note:
        lines.extend(["## Day35 Tier2 NA Handling", "", *[f"- {note}" for note in d35_note], ""])

    lines.extend(
        [
            "## Neighbor Index QC",
            "",
            f"- Metadata entries: {len(neighbor_validation)}",
            f"- All entries passed: {bool(neighbor_validation['ok'].all()) if not neighbor_validation.empty else False}",
            "- Anchor inclusion behavior: `include_anchor` is enabled by config and validated per entry.",
            "",
            *_markdown_table(
                neighbor_validation,
                columns=[
                    "entry",
                    "slice_file",
                    "scale",
                    "n_anchors",
                    "indptr_length",
                    "neighbor_indices_length",
                    "within_slice_bounds",
                    "avg_neighbors_npz",
                    "avg_neighbors_feature",
                    "avg_neighbors_match",
                    "anchor_inclusion_ok",
                    "ok",
                    "errors",
                ],
                max_rows=20,
            ),
            "",
            "## Figures",
            "",
        ]
    )
    if figure_paths:
        lines.extend([f"- `{path}`" for path in figure_paths])
    elif figure_error:
        lines.append(f"- Figure generation skipped: {figure_error}")
    else:
        lines.append("- No figures generated.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_storage_plan(
    path: Path,
    config: dict,
    feature_path: Path,
    neighbor_path: Path,
    features: pd.DataFrame,
    slice_sizes: pd.DataFrame,
    metadata: list[dict[str, object]],
    prototype_wall_seconds: float | None,
) -> str:
    scales = list(config["niche"]["scales"])
    full_anchors = int(slice_sizes["n_obs"].sum()) if not slice_sizes.empty else 1_439_542
    n_slices = int(len(slice_sizes)) if not slice_sizes.empty else None
    avg_neighbors = (
        features.groupby("scale", observed=True)["n_neighbors"].mean().to_dict()
        if "scale" in features and "n_neighbors" in features
        else {}
    )
    raw_neighbor = neighbor_raw_bytes_from_metadata(metadata)
    raw_neighbor_bytes = int(raw_neighbor["raw_bytes"].sum()) if not raw_neighbor.empty else 0
    neighbor_ratio = (
        neighbor_path.stat().st_size / raw_neighbor_bytes
        if raw_neighbor_bytes
        else 0.35
    )
    estimate = estimate_full_m1_storage(
        full_anchors=full_anchors,
        scales=scales,
        prototype_rows=len(features),
        prototype_feature_bytes=feature_path.stat().st_size,
        avg_neighbors_by_scale=avg_neighbors,
        n_slices=n_slices,
        neighbor_compression_ratio=neighbor_ratio,
    )
    try:
        import pyarrow  # noqa: F401

        pyarrow_status = "available in the active environment"
    except ImportError:
        pyarrow_status = "not importable in the active environment"

    prototype_anchors = (
        features[["slice_file", "anchor_index"]].drop_duplicates().shape[0]
        if {"slice_file", "anchor_index"}.issubset(features.columns)
        else 0
    )
    runtime_seconds = (
        prototype_wall_seconds * full_anchors / prototype_anchors
        if prototype_wall_seconds and prototype_anchors
        else None
    )
    prototype_slice_files = set(features["slice_file"].astype(str)) if "slice_file" in features else set()
    prototype_times = set(features["time"].astype(str)) if "time" in features else set()
    pilot = _choose_pilot_slice(slice_sizes, prototype_slice_files, prototype_times)
    pilot_command = ""
    pilot_lines = ["No candidate slice available from M0 by-slice outputs."]
    if pilot is not None:
        pilot_rows = int(pilot["n_obs"]) * len(scales)
        bytes_per_row = float(estimate["bytes_per_feature_row_csv"])
        pilot_feature_csv = int(bytes_per_row * pilot_rows) if np.isfinite(bytes_per_row) else 0
        pilot_neighbor_raw = 0
        for scale in scales:
            avg = float(avg_neighbors.get(scale, 0.0))
            pilot_neighbor_raw += int((pilot["n_obs"] + pilot["n_obs"] + 1 + pilot["n_obs"] * avg) * 8)
        pilot_neighbor_npz = int(pilot_neighbor_raw * neighbor_ratio)
        pilot_runtime = (
            prototype_wall_seconds * int(pilot["n_obs"]) / prototype_anchors
            if prototype_wall_seconds and prototype_anchors
            else None
        )
        pilot_command = (
            "conda run --no-capture-output -n omicverse python "
            "scripts/m1_03_build_niche_full.py "
            "--config configs/m1_niche_construction.yaml "
            f"--slice-file {pilot['slice_path']} "
            "--output-dir /home/zhutao/scratch/nichefate/m1/pilot_full_slice "
            "--force"
        )
        pilot_lines = [
            f"- Candidate slice: `{pilot['slice_file']}`",
            f"- Cells: {int(pilot['n_obs'])}",
            f"- Expected feature rows: {pilot_rows}",
            f"- Estimated feature CSV size: {_format_bytes(pilot_feature_csv)}",
            f"- Estimated compressed neighbor NPZ size: {_format_bytes(pilot_neighbor_npz)}",
            f"- Expected runtime: {pilot_runtime:.1f} seconds plus I/O, if prototype scaling is linear"
            if pilot_runtime
            else "- Expected runtime: not estimated",
            "- The command below is for the later full-slice execution stage; it was not run here.",
            "",
            "```bash",
            pilot_command,
            "```",
        ]

    estimate_frame = pd.DataFrame(
        [
            {"metric": key, "value": value, "human": _format_bytes(value) if key.endswith("_bytes") else value}
            for key, value in estimate.items()
        ]
    )
    avg_frame = pd.DataFrame(
        [{"scale": scale, "avg_neighbors": value} for scale, value in avg_neighbors.items()]
    )
    lines = [
        "# Full M1 Storage And Chunking Plan",
        "",
        "This is a planning report only. Full M1 and the full-slice pilot were not run.",
        "",
        "## Full Dataset Estimate",
        "",
        f"- Full anchors: {full_anchors}",
        f"- Scales: {len(scales)}",
        f"- Expected full feature rows: {estimate['full_feature_rows']}",
        f"- Prototype CSV: `{feature_path}` ({_format_bytes(feature_path.stat().st_size)})",
        f"- Prototype neighbor NPZ: `{neighbor_path}` ({_format_bytes(neighbor_path.stat().st_size)})",
        f"- Parquet support check: {pyarrow_status}",
        "",
        "### Average Neighbors Used For Estimate",
        "",
        *_markdown_table(avg_frame),
        "",
        "### Storage Estimate",
        "",
        *_markdown_table(estimate_frame, max_rows=30),
        "",
        "## Recommended Full M1 Strategy",
        "",
        "- Write per-slice feature tables instead of one global CSV.",
        "- Prefer parquet if pyarrow remains stable in the active environment; otherwise use compressed CSV fallback.",
        "- Write one compressed NPZ neighbor index per slice.",
        "- Write one global summary CSV assembled from per-slice QC summaries.",
        "- Make full execution resumable: skip valid completed slice outputs unless `--force` is set.",
        "- Write `failed_slices.txt` for any slice-level failure.",
        "- Do not materialize full point clouds.",
        "- Do not construct cross-slice niches.",
        "",
        "## Runtime And Memory Risks",
        "",
    ]
    if runtime_seconds:
        lines.append(
            f"- Rough linear runtime from prototype: {_format_number(runtime_seconds / 60.0)} minutes plus I/O."
        )
    else:
        lines.append("- Runtime could not be estimated from the prototype report.")
    lines.extend(
        [
            "- Main memory risk is concatenating all 4.3M feature rows with wide composition and embedding columns.",
            "- Per-slice writes bound memory and make restart behavior straightforward.",
            "- Neighbor index storage is dominated by `radius_x8`; keep neighbor indices per-slice to avoid large global ragged arrays.",
            "",
            "## Later Full-Slice Pilot Recommendation",
            "",
            *pilot_lines,
            "",
            "Validation checks after the later pilot:",
            "",
            "- Feature rows equal `n_cells x 3`.",
            "- Neighbor NPZ has one entry per configured scale.",
            "- `indptr[-1]` matches neighbor array length for every scale.",
            "- Average neighbor counts match the feature table.",
            "- Composition row sums are close to 1 for each cell-type level.",
            "- Missing and infinite value summaries remain limited to expected metadata fallback cases.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return pilot_command


def _write_generalizability_review(path: Path) -> None:
    terms = ("DSS", "D35", "Day35", "Moffitt", "Cadinu", "Sample_type")
    niche_hits = _scan_terms(PROJECT_ROOT / "src/nichefate/niche.py", terms)
    prototype_hits = _scan_terms(PROJECT_ROOT / "scripts/m1_01_build_niche_prototype.py", terms)
    lines = [
        "# M1 Generalizability Review",
        "",
        "This review separates dataset adapter behavior from the NicheFate method core.",
        "",
        "## A. Dataset Adapter Layer",
        "",
        "- Moffitt/Cadinu-specific file layout belongs in M0 loading and verification scripts.",
        "- Moffitt/Cadinu-specific obs field names belong in M0 metadata mapping.",
        "- DSS-specific time labels belong in configuration and M0 mapping.",
        "- Day35 Tier2 fallback belongs in M0/reporting notes and must not drive M1 feature algorithms.",
        "",
        "Dataset-specific code paths currently include:",
        "",
        "- `src/nichefate/metadata.py`: source obs mapping, `Sample_type` handling, and Day35 fallback.",
        "- `scripts/m0_*`: raw file inspection, M0 construction, and dataset QC reporting.",
        "- `scripts/m1_01_build_niche_prototype.py`: one prototype report note for D35 `cell_type_l2` NA fallback.",
        "",
        "## B. NicheFate Method Core",
        "",
        "The general method core is:",
        "",
        "- per-slice graph input",
        "- configurable graph keys",
        "- anchor-centered niche extraction",
        "- multi-scale neighbor index",
        "- configurable cell type keys",
        "- embedding summary",
        "- spatial summary",
        "- topology summary",
        "- feature table construction",
        "",
        "General M1 core functions include:",
        "",
        "- `get_graph_neighbors()`",
        "- `compute_neighbor_index()`",
        "- `compute_celltype_composition()`",
        "- `compute_shannon_entropy_from_composition()`",
        "- `compute_embedding_summary()`",
        "- `compute_spatial_summary()`",
        "- `compute_topology_summary()`",
        "- `build_basic_niche_feature_table()`",
        "- `write_neighbor_index_npz()`",
        "- `write_niche_feature_table_parquet_or_csv()`",
        "",
        "## Code Review Findings",
        "",
        f"- `src/nichefate/niche.py` dataset-specific string hits: {len(niche_hits)}",
    ]
    if niche_hits:
        lines.extend([f"  - `{hit}`" for hit in niche_hits])
    else:
        lines.append("  - No DSS, D35, Day35, Moffitt, Cadinu, or Sample_type strings found.")
    lines.extend(
        [
            f"- `scripts/m1_01_build_niche_prototype.py` dataset-specific string hits: {len(prototype_hits)}",
        ]
    )
    lines.extend([f"  - `{hit}`" for hit in prototype_hits] if prototype_hits else ["  - None."])
    lines.extend(
        [
            "- `scripts/m1_01_build_niche_prototype.py` is mostly config-driven for graph keys, cell type keys, embedding key, and spatial key.",
            "- The prototype script's D35 note is reporting-layer behavior, not method-core behavior.",
            "- The new `src/nichefate/niche_qc.py` helpers are schema-oriented and do not require Moffitt/Cadinu fields.",
            "",
            "## Porting Requirements For Another ST Dataset",
            "",
            "To run on Xenium kidney IRI, Stereo-seq liver injury, Visium HD intestinal regeneration, or another MERFISH dataset, change the adapter/config layer to provide:",
            "",
            "- per-slice M0 objects with the standard NicheFate obs schema",
            "- `slice_id` and time ordering fields",
            "- configured spatial and molecular embedding keys",
            "- configured cell type annotation keys",
            "- configured per-slice graph keys and optional topology/ablation graphs",
            "",
            "No M1 core algorithm change should be needed if those schema contracts are satisfied.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    start = time.monotonic()
    args = parse_args()
    config = load_config(args.config)
    paths = _paths(config)
    prototype_dir = paths["m1_output_dir"] / "prototype"
    reports_dir = paths["reports_dir"]
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    feature_path = prototype_dir / "niche_features_prototype.csv"
    neighbor_path = prototype_dir / "neighbor_index_prototype.npz"
    feature_qc_md = reports_dir / "m1_prototype_feature_qc.md"
    feature_qc_csv = reports_dir / "m1_prototype_feature_qc.csv"
    storage_md = reports_dir / "m1_full_m1_storage_plan.md"
    general_md = reports_dir / "m1_generalizability_review.md"

    features = pd.read_csv(feature_path, low_memory=False)
    slice_sizes = _read_slice_sizes(paths["m0_by_slice_dir"])
    slice_n_obs = _slice_n_obs_map(slice_sizes)
    metadata = load_neighbor_metadata(neighbor_path)
    with np.load(neighbor_path, allow_pickle=False) as loaded:
        npz_keys = list(loaded.files)

    expected_entries = (
        int(features["slice_file"].nunique()) * len(config["niche"]["scales"])
        if "slice_file" in features
        else None
    )
    integrity = summarize_feature_integrity(features)
    composition_qc = composition_sum_qc(features)
    composition_counts = pd.DataFrame(
        [
            {"composition_level": level, "n_columns": len(composition_columns(features, level))}
            for level in ("cell_type_l1", "cell_type_l2", "cell_type_l3")
        ]
    )
    dominant = pd.concat(
        [
            dominant_composition(features, level, by=("scale",), top_n=8)
            for level in ("cell_type_l1", "cell_type_l2", "cell_type_l3")
        ],
        ignore_index=True,
    )
    distributions = {
        "n_neighbors by scale": summarize_distribution(features, "n_neighbors", by=("scale",)),
        "n_neighbors by scale and slice": summarize_distribution(
            features, "n_neighbors", by=("scale", "slice_file")
        ),
        "ct_l1 entropy by scale and slice": summarize_distribution(
            features, "ct_l1_entropy", by=("scale", "slice_file")
        ),
        "ct_l2 entropy by scale and slice": summarize_distribution(
            features, "ct_l2_entropy", by=("scale", "slice_file")
        ),
        "ct_l3 entropy by scale and slice": summarize_distribution(
            features, "ct_l3_entropy", by=("scale", "slice_file")
        ),
        "mean_neighbor_distance by scale": summarize_distribution(
            features, "mean_neighbor_distance", by=("scale",)
        ),
        "pseudo_local_density by scale": summarize_distribution(
            features, "pseudo_local_density", by=("scale",)
        ),
        "topology degree by scale": summarize_distribution(
            features, "local_topology_degree", by=("scale",)
        ),
    }
    neighbor_validation = validate_neighbor_npz(
        neighbor_path,
        feature_table=features,
        slice_n_obs=slice_n_obs,
        expected_entries=expected_entries,
        include_anchor=bool(config["niche"]["include_anchor"]),
    )
    figure_paths, figure_error = _write_optional_figures(features, figures_dir)

    _write_feature_qc_report(
        feature_qc_md,
        features,
        feature_path,
        neighbor_path,
        integrity,
        composition_qc,
        composition_counts,
        dominant,
        distributions,
        neighbor_validation,
        npz_keys,
        figure_paths,
        figure_error,
    )
    _write_qc_csv(
        feature_qc_csv,
        [
            ("feature_integrity", integrity),
            ("composition_counts", composition_counts),
            ("composition_sum_qc", composition_qc),
            ("dominant_composition", dominant),
            ("neighbor_validation", neighbor_validation),
            *[(name, frame) for name, frame in distributions.items()],
        ],
    )
    pilot_command = _write_storage_plan(
        storage_md,
        config,
        feature_path,
        neighbor_path,
        features,
        slice_sizes,
        metadata,
        _prototype_wall_seconds(reports_dir / "m1_prototype_report.md"),
    )
    _write_generalizability_review(general_md)

    elapsed = time.monotonic() - start
    print(f"Wrote feature QC report: {feature_qc_md}")
    print(f"Wrote feature QC CSV: {feature_qc_csv}")
    print(f"Wrote storage plan: {storage_md}")
    print(f"Wrote generalizability review: {general_md}")
    if figure_paths:
        for path in figure_paths:
            print(f"Wrote figure: {path}")
    elif figure_error:
        print(f"Figure generation skipped: {figure_error}")
    print(f"Recommended later pilot command: {pilot_command}")
    print(f"WALL_SECONDS {elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
