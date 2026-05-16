#!/usr/bin/env python
"""Design and optionally run a bounded per-slice stratified metaniche pilot.

The default is dry-run. A real stratified pilot must be requested with
`--no-dry-run`; limits remain capped at four slices and 20,000 anchors total.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.planA_k_gpcca import (
    M1_BY_SLICE_ROOT,
    M2_BY_SLICE_ROOT,
    M2_SCHEMA_PATH,
    METANICHE_HARDENING_ROOT,
    atomic_write_json,
    atomic_write_text,
    atomic_write_tsv,
    compare_original_and_stratified_pilots,
    compute_spatial_compactness_qc,
    ensure_dir,
    rare_state_audit_markdown,
    rare_state_preservation_audit,
    run_stratified_metaniche_pilot,
    run_coordinate_join_preview,
    spatial_compactness_markdown,
    stratified_pilot_design_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-root", type=Path, default=M2_BY_SLICE_ROOT)
    parser.add_argument("--m1-root", type=Path, default=M1_BY_SLICE_ROOT)
    parser.add_argument("--schema-path", type=Path, default=M2_SCHEMA_PATH)
    parser.add_argument("--output-dir", type=Path, default=METANICHE_HARDENING_ROOT)
    parser.add_argument("--max-slices", type=int, default=4)
    parser.add_argument("--max-anchors-per-slice", type=int, default=5000)
    parser.add_argument(
        "--feature-mode",
        choices=["safe", "embedding_only", "composition_only", "all_safe"],
        default="safe",
    )
    parser.add_argument("--n-components", type=int, default=30)
    parser.add_argument("--n-clusters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    design_text, design_payload = stratified_pilot_design_payload()
    atomic_write_text(output_dir / "06_stratified_pilot_design.md", design_text, overwrite=args.overwrite)
    atomic_write_json(output_dir / "06_stratified_pilot_design.json", design_payload, overwrite=args.overwrite)

    pilot_payload = run_stratified_metaniche_pilot(
        output_dir=output_dir,
        m2_root=args.m2_root,
        schema_path=args.schema_path,
        max_slices=args.max_slices,
        max_anchors_per_slice=args.max_anchors_per_slice,
        feature_mode=args.feature_mode,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        seed=args.seed,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    if not args.dry_run and pilot_payload.get("stratified_pilot_run"):
        stratified_root = output_dir / "stratified_pilot_outputs"
        run_coordinate_join_preview(
            output_dir=stratified_root,
            pilot_root=stratified_root,
            m1_root=args.m1_root,
            overwrite=args.overwrite,
            dry_run=False,
        )
        spatial_frame, spatial_payload = compute_spatial_compactness_qc(stratified_root)
        atomic_write_text(
            stratified_root / "spatial_compactness_qc.md",
            spatial_compactness_markdown(spatial_frame, spatial_payload),
            overwrite=args.overwrite,
        )
        atomic_write_tsv(
            stratified_root / "spatial_compactness_qc.tsv",
            spatial_frame,
            overwrite=args.overwrite,
        )
        atomic_write_json(
            stratified_root / "spatial_compactness_qc.json",
            spatial_payload,
            overwrite=args.overwrite,
        )
        rare_frame, rare_payload = rare_state_preservation_audit(pilot_root=stratified_root)
        atomic_write_text(
            stratified_root / "rare_state_preservation_audit.md",
            rare_state_audit_markdown(rare_frame, rare_payload),
            overwrite=args.overwrite,
        )
        atomic_write_tsv(
            stratified_root / "rare_state_preservation_audit.tsv",
            rare_frame,
            overwrite=args.overwrite,
        )
        atomic_write_json(
            stratified_root / "rare_state_preservation_audit.json",
            rare_payload,
            overwrite=args.overwrite,
        )
    comparison_text, comparison_payload = compare_original_and_stratified_pilots(output_dir)
    comparison_payload["pilot_run_payload"] = pilot_payload
    if args.dry_run:
        comparison_text += (
            "\n\nDry run only. Recommended run command: `"
            + pilot_payload.get("recommended_command", "not available")
            + "`\n"
        )
    atomic_write_text(
        output_dir / "07_stratified_pilot_summary.md",
        comparison_text,
        overwrite=args.overwrite,
    )
    atomic_write_json(
        output_dir / "07_stratified_pilot_summary.json",
        comparison_payload,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
