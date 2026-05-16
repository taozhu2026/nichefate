#!/usr/bin/env python
"""Run a bounded, sampled M2.5 metaniche coarsening pilot.

The default is dry-run mode. A real pilot must be requested with
`--no-dry-run`; even then it is capped to at most four slices, 5,000 anchors
per slice, 20,000 total anchors, 30 PCA components, and 500 metaniches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.planA_k_gpcca import (
    M2_BY_SLICE_ROOT,
    M2_SCHEMA_PATH,
    METANICHE_REPORT_ROOT,
    atomic_write_json,
    atomic_write_text,
    discover_m2_inventory,
    ensure_dir,
    load_m2_feature_schema,
    pilot_run_summary_markdown,
    run_sampled_metaniche_pilot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-root", type=Path, default=M2_BY_SLICE_ROOT)
    parser.add_argument("--schema-path", type=Path, default=M2_SCHEMA_PATH)
    parser.add_argument("--output-dir", type=Path, default=METANICHE_REPORT_ROOT)
    parser.add_argument("--max-slices", type=int, default=4)
    parser.add_argument("--max-anchors-per-slice", type=int, default=5000)
    parser.add_argument(
        "--feature-mode",
        choices=["safe", "embedding_only", "composition_only", "all_safe"],
        default="safe",
    )
    parser.add_argument("--n-components", type=int, default=30)
    parser.add_argument("--n-clusters", type=int, default=200)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument(
        "--cluster-method",
        choices=["kmeans", "leiden"],
        default="kmeans",
        help="Use kmeans for a capped reproducible pilot; leiden is optional.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    inventory, _ = discover_m2_inventory(m2_root=args.m2_root, schema_path=args.schema_path)
    schema = load_m2_feature_schema(args.schema_path)
    payload = run_sampled_metaniche_pilot(
        inventory=inventory,
        schema=schema,
        output_dir=output_dir,
        max_slices=args.max_slices,
        max_anchors_per_slice=args.max_anchors_per_slice,
        feature_mode=args.feature_mode,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        cluster_method=args.cluster_method,
        resolution=args.resolution,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        seed=args.seed,
    )
    atomic_write_json(output_dir / "04_pilot_run_summary.json", payload, overwrite=args.overwrite)
    atomic_write_text(
        output_dir / "04_pilot_run_summary.md",
        pilot_run_summary_markdown(payload),
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
