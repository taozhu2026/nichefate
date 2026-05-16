#!/usr/bin/env python
"""Compute QC for sampled M2.5 metaniche pilot outputs.

This script is read-only with respect to pilot source outputs. It writes small
QC reports and the final M2.5 pilot summary under the report directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.planA_k_gpcca import (
    METANICHE_REPORT_ROOT,
    atomic_write_json,
    atomic_write_text,
    atomic_write_tsv,
    compute_metaniche_qc_from_outputs,
    ensure_dir,
    git_status_short,
    metaniche_final_summary_markdown,
    metaniche_final_summary_payload,
    metaniche_qc_markdown,
    next_sparse_k_pilot_design,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=METANICHE_REPORT_ROOT)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def read_json_if_present(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv_if_present(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t")


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    qc_frame, qc_payload = compute_metaniche_qc_from_outputs(output_dir)
    design_text, design_payload = next_sparse_k_pilot_design(qc_payload)

    atomic_write_text(
        output_dir / "05_metaniche_qc.md",
        metaniche_qc_markdown(qc_frame, qc_payload),
        overwrite=args.overwrite,
    )
    atomic_write_tsv(output_dir / "05_metaniche_qc.tsv", qc_frame, overwrite=args.overwrite)
    atomic_write_json(output_dir / "05_metaniche_qc.json", qc_payload, overwrite=args.overwrite)

    atomic_write_text(
        output_dir / "06_next_sparse_k_pilot_design.md",
        design_text,
        overwrite=args.overwrite,
    )
    atomic_write_json(
        output_dir / "06_next_sparse_k_pilot_design.json",
        design_payload,
        overwrite=args.overwrite,
    )

    inventory_json = read_json_if_present(output_dir / "01_m2_inventory.json")
    feature_audit = read_tsv_if_present(output_dir / "02_feature_group_audit.tsv")
    pilot_payload = read_json_if_present(output_dir / "04_pilot_run_summary.json")
    summary_payload = metaniche_final_summary_payload(
        output_dir=output_dir,
        inventory_summary=inventory_json.get("summary"),
        feature_audit=feature_audit,
        pilot_payload=pilot_payload,
        qc_payload=qc_payload,
        git_status_after=git_status_short(),
    )
    atomic_write_text(
        output_dir / "00_M2_5_METANICHE_PILOT_SUMMARY.md",
        metaniche_final_summary_markdown(summary_payload),
        overwrite=args.overwrite,
    )
    atomic_write_json(
        output_dir / "00_M2_5_METANICHE_PILOT_SUMMARY.json",
        summary_payload,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
