#!/usr/bin/env python
"""Inspect optional ANN backend availability for M3 without reading M2 data."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config
from nichefate.transition import (
    ANN_BACKENDS,
    REQUIRED_NEIGHBOR_BACKENDS,
    inspect_candidate_neighbor_backend,
)


BACKEND_ORDER = ["sklearn_exact", "numpy_chunked", "faiss", "hnswlib", "pynndescent"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_transition_kernel.yaml")
    return parser.parse_args()


def _reports_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["reports_dir"])


def _assert_no_ssd(config: dict[str, Any]) -> None:
    if bool(config["paths"].get("use_ssd", False)):
        raise RuntimeError("Refusing M3-08 inspection while paths.use_ssd is true.")
    for value in config.get("paths", {}).values():
        if isinstance(value, str) and value.startswith("/ssd"):
            raise RuntimeError(f"Refusing to use /ssd path in M3-08 inspection: {value}")


def inspect_backends(metric: str) -> pd.DataFrame:
    rows = []
    for backend in BACKEND_ORDER:
        status = inspect_candidate_neighbor_backend(
            backend,
            metric=metric,
            run_toy_check=True,
        )
        row = asdict(status)
        row["required"] = backend in REQUIRED_NEIGHBOR_BACKENDS
        row["ann_backend"] = backend in ANN_BACKENDS
        row["check_scope"] = "toy_in_memory_only"
        rows.append(row)
    return pd.DataFrame(rows)


def write_backend_report(path: Path, availability: pd.DataFrame) -> None:
    usable_ann = availability[
        (availability["ann_backend"].astype(bool)) & (availability["usable"].astype(bool))
    ]["backend"].astype(str).tolist()
    lines = [
        "# M3 ANN Backend Availability",
        "",
        "This inspection is toy-level only. It reads no M2 representation data, creates no M3 edge shards, submits no jobs, and does not alter the conda environment.",
        "",
        "## Summary",
        f"- Required backends usable: {bool(availability[availability['required']]['usable'].all())}",
        f"- Usable optional ANN backends: {', '.join(usable_ann) if usable_ann else 'none'}",
        "- `sklearn_exact` remains the validation backend for sampled/small work.",
        "- Optional ANN backends must be validated before any large full time-pair construction.",
        "",
        "## Backend Status",
        "",
        "| backend | package | required | importable | usable | reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in availability.to_dict("records"):
        lines.append(
            "| "
            f"{row['backend']} | "
            f"{row.get('package') or ''} | "
            f"{bool(row['required'])} | "
            f"{bool(row['importable'])} | "
            f"{bool(row['usable'])} | "
            f"{row['reason']} |"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    _assert_no_ssd(config)
    reports_dir = _reports_dir(config)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metric = config["candidate_edges"].get("retrieval_metric", "euclidean")
    availability = inspect_backends(metric)
    csv_path = reports_dir / "m3_ann_backend_availability.csv"
    md_path = reports_dir / "m3_ann_backend_availability.md"
    availability.to_csv(csv_path, index=False)
    write_backend_report(md_path, availability)
    usable_ann = availability[
        (availability["ann_backend"].astype(bool)) & (availability["usable"].astype(bool))
    ]["backend"].astype(str).tolist()
    print(f"Wrote ANN backend availability CSV: {csv_path}")
    print(f"Wrote ANN backend availability report: {md_path}")
    print(f"USABLE_ANN_BACKENDS {','.join(usable_ann) if usable_ann else 'none'}")
    print("TOY_CHECK_ONLY True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
