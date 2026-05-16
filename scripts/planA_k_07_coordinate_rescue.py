#!/usr/bin/env python
"""Rescue M2.5 pilot coordinates from M1 and compute spatial QC.

Default mode is dry-run for the coordinate preview. Inventory and audit reports
are always lightweight and inspect-only. Use `--no-dry-run` to write the pilot
coordinate preview under reports/planA_k_metaniche_hardening/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.planA_k_gpcca import (
    M1_BY_SLICE_ROOT,
    METANICHE_HARDENING_ROOT,
    PILOT_OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_text,
    atomic_write_tsv,
    audit_coordinate_join_keys,
    collect_metaniche_hardening_preflight_payload,
    compute_spatial_compactness_qc,
    coordinate_join_preview_markdown,
    coordinate_source_inventory_markdown,
    dataframe_to_markdown,
    discover_coordinate_sources,
    ensure_dir,
    join_key_audit_markdown,
    run_coordinate_join_preview,
    spatial_compactness_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=PILOT_OUTPUT_ROOT)
    parser.add_argument("--m1-root", type=Path, default=M1_BY_SLICE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=METANICHE_HARDENING_ROOT)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    preflight_inventory, preflight_payload = collect_metaniche_hardening_preflight_payload()
    atomic_write_text(
        output_dir / "00_preflight.md",
        "# Metaniche Hardening Preflight\n\n"
        + f"- Dry run: {args.dry_run}\n"
        + f"- Git branch: {preflight_payload['environment']['git_branch']}\n"
        + f"- Git status entries: {len(preflight_payload['environment']['git_status_short'])}\n\n"
        + dataframe_to_markdown(preflight_inventory),
        overwrite=args.overwrite,
    )
    atomic_write_json(output_dir / "00_preflight.json", preflight_payload, overwrite=args.overwrite)

    coord_frame, coord_summary = discover_coordinate_sources(
        pilot_root=args.pilot_root,
        m1_root=args.m1_root,
    )
    atomic_write_text(
        output_dir / "01_coordinate_source_inventory.md",
        coordinate_source_inventory_markdown(coord_frame, coord_summary),
        overwrite=args.overwrite,
    )
    atomic_write_tsv(output_dir / "01_coordinate_source_inventory.tsv", coord_frame, overwrite=args.overwrite)
    atomic_write_json(
        output_dir / "01_coordinate_source_inventory.json",
        {"summary": coord_summary, "rows": coord_frame.to_dict(orient="records")},
        overwrite=args.overwrite,
    )

    join_frame, join_summary = audit_coordinate_join_keys(
        pilot_root=args.pilot_root,
        m1_root=args.m1_root,
    )
    atomic_write_text(
        output_dir / "02_join_key_audit.md",
        join_key_audit_markdown(join_frame, join_summary),
        overwrite=args.overwrite,
    )
    atomic_write_tsv(output_dir / "02_join_key_audit.tsv", join_frame, overwrite=args.overwrite)
    atomic_write_json(
        output_dir / "02_join_key_audit.json",
        {"summary": join_summary, "rows": join_frame.to_dict(orient="records")},
        overwrite=args.overwrite,
    )

    preview_payload = run_coordinate_join_preview(
        output_dir=output_dir,
        pilot_root=args.pilot_root,
        m1_root=args.m1_root,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    atomic_write_text(
        output_dir / "03_coordinate_join_preview.md",
        coordinate_join_preview_markdown(preview_payload),
        overwrite=args.overwrite,
    )
    atomic_write_json(
        output_dir / "03_coordinate_join_preview.json",
        preview_payload,
        overwrite=args.overwrite,
    )

    spatial_frame, spatial_payload = compute_spatial_compactness_qc(output_dir)
    atomic_write_text(
        output_dir / "04_spatial_compactness_qc.md",
        spatial_compactness_markdown(spatial_frame, spatial_payload),
        overwrite=args.overwrite,
    )
    atomic_write_tsv(output_dir / "04_spatial_compactness_qc.tsv", spatial_frame, overwrite=args.overwrite)
    atomic_write_json(output_dir / "04_spatial_compactness_qc.json", spatial_payload, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
