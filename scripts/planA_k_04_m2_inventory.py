#!/usr/bin/env python
"""Index M2 by-slice outputs and audit feature groups for the M2.5 pilot.

This script is inspect-only. It reads existing M2 metadata, Parquet schemas,
and one metadata row per file, then writes lightweight reports under the repo
report tree. It does not process DARLIN data, modify scratch outputs, or load
the full M2 dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.planA_k_gpcca import (
    DOC_ROOT,
    M2_BY_SLICE_ROOT,
    M2_SCHEMA_PATH,
    METANICHE_REPORT_ROOT,
    atomic_write_json,
    atomic_write_text,
    atomic_write_tsv,
    classify_m2_feature_groups,
    collect_metaniche_preflight_payload,
    dataframe_to_markdown,
    discover_m2_inventory,
    ensure_dir,
    feature_group_audit_markdown,
    load_m2_feature_schema,
    m2_inventory_markdown,
    metaniche_pilot_protocol_text,
    select_m2_feature_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-root", type=Path, default=M2_BY_SLICE_ROOT)
    parser.add_argument("--schema-path", type=Path, default=M2_SCHEMA_PATH)
    parser.add_argument("--output-dir", type=Path, default=METANICHE_REPORT_ROOT)
    parser.add_argument("--doc-root", type=Path, default=DOC_ROOT)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    doc_root = ensure_dir(args.doc_root)

    preflight_inventory, preflight_payload = collect_metaniche_preflight_payload()
    inventory, inventory_summary = discover_m2_inventory(
        m2_root=args.m2_root,
        schema_path=args.schema_path,
    )
    schema = load_m2_feature_schema(args.schema_path)
    feature_audit = classify_m2_feature_groups(schema)
    safe_columns = select_m2_feature_columns(schema, feature_mode="safe")
    protocol_text = metaniche_pilot_protocol_text()

    atomic_write_text(
        output_dir / "00_preflight.md",
        "# M2.5 Metaniche Pilot Preflight\n\n"
        + f"- Dry run: {args.dry_run}\n"
        + f"- Git branch: {preflight_payload['environment']['git_branch']}\n"
        + f"- Git status entries: {len(preflight_payload['environment']['git_status_short'])}\n\n"
        + dataframe_to_markdown(preflight_inventory),
        overwrite=args.overwrite,
    )
    atomic_write_json(output_dir / "00_preflight.json", preflight_payload, overwrite=args.overwrite)

    atomic_write_text(
        output_dir / "01_m2_inventory.md",
        m2_inventory_markdown(inventory, inventory_summary),
        overwrite=args.overwrite,
    )
    atomic_write_tsv(output_dir / "01_m2_inventory.tsv", inventory, overwrite=args.overwrite)
    atomic_write_json(
        output_dir / "01_m2_inventory.json",
        {
            "summary": inventory_summary,
            "rows": inventory.to_dict(orient="records"),
        },
        overwrite=args.overwrite,
    )

    atomic_write_text(
        output_dir / "02_feature_group_audit.md",
        feature_group_audit_markdown(feature_audit, schema),
        overwrite=args.overwrite,
    )
    atomic_write_tsv(output_dir / "02_feature_group_audit.tsv", feature_audit, overwrite=args.overwrite)
    atomic_write_json(
        output_dir / "02_feature_group_audit.json",
        {
            "schema_path": str(args.schema_path),
            "feature_mode_default": "safe",
            "safe_feature_column_count": len(safe_columns),
            "safe_feature_column_examples": safe_columns[:20],
            "rows": feature_audit.to_dict(orient="records"),
        },
        overwrite=args.overwrite,
    )

    atomic_write_text(
        doc_root / "05_m2_5_metaniche_pilot_protocol.md",
        protocol_text,
        overwrite=args.overwrite,
    )
    atomic_write_text(
        output_dir / "03_metaniche_pilot_protocol.md",
        protocol_text,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
