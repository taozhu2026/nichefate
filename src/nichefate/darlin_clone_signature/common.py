from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_ASSAYS = ("CA", "TA", "RA")
CELL_KEY_COLUMNS = ["sample_id", "slice_id", "cellbin_id"]
CELL_COLUMNS = ["sample_id", "slice_id", "section_order", "cellbin_id"]
CLONE_CLASSES = (
    "cross_locus_clone",
    "single_locus_recurrent_clone",
    "multi_feature_single_locus_clone",
)
NON_CLONE_STATUSES = ("ambiguous", "filtered", "unassigned")


@dataclass(frozen=True)
class CloneSignatureParams:
    rare_threshold: float = 0.001
    low_frequency_threshold: float = 0.005
    min_single_feature_cellbins: int = 2
    min_feature_cooccurrence_cellbins: int = 2
    min_cross_locus_support_cellbins: int = 2
    min_multifeature_support_cellbins: int = 2
    max_bridge_dependency_score: float = 0.50
    bridge_filter_mode: str = "p99"
    high_complexity_pair_warning: int = 10_000_000
    max_signature_component_features: int = 100
    topk_scores: int = 5
    membership_ratio: float = 0.75
    shared_support_overlap_ambiguity: float = 0.50
    random_seed: int = 126


def make_cell_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["sample_id"].astype(str)
        + "|"
        + frame["slice_id"].astype(str)
        + "|"
        + frame["cellbin_id"].astype(str)
    )


def assay_scoped_feature(frame: pd.DataFrame) -> pd.Series:
    return frame["assay"].astype(str) + "::" + frame["feature_id"].astype(str)


def split_assay_scoped_feature(value: str) -> tuple[str, str]:
    assay, feature_id = str(value).split("::", 1)
    return assay, feature_id


def entropy_from_counts(counts: Iterable[float]) -> float:
    values = np.asarray([float(value) for value in counts if float(value) > 0], dtype=float)
    if values.size <= 1:
        return 0.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def simpson_from_counts(counts: Iterable[float]) -> float:
    values = np.asarray([float(value) for value in counts if float(value) > 0], dtype=float)
    if values.size <= 1:
        return 0.0
    probabilities = values / values.sum()
    return float(1.0 - np.square(probabilities).sum())


def compact_distribution(frame: pd.DataFrame, key: str, count_col: str = "cell_key", limit: int = 8) -> str:
    if frame.empty or key not in frame:
        return ""
    counts = frame.groupby(key, dropna=False)[count_col].nunique().sort_values(ascending=False)
    items = [f"{idx}:{int(value)}" for idx, value in counts.head(limit).items()]
    if len(counts) > limit:
        items.append(f"other:{int(counts.iloc[limit:].sum())}")
    return ";".join(items)


def summarize_top_items(counts: pd.Series, limit: int = 5) -> str:
    if counts.empty:
        return ""
    items = [f"{idx}:{float(value):.4g}" for idx, value in counts.head(limit).items()]
    if len(counts) > limit:
        items.append(f"other:{float(counts.iloc[limit:].sum()):.4g}")
    return ";".join(items)


def path_has_forbidden_ssd(path: str | Path) -> bool:
    return Path(path).expanduser().as_posix().startswith("/ssd/")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isfinite(result):
        return result
    return default
