#!/usr/bin/env python
"""Run or preflight full PlanA-K M2.5 metaniche production.

Default mode is dry-run and report-only. Full production requires
``--no-dry-run`` and writes only under the approved PlanA-K production scratch
root. Smoke-test mode is bounded and must not use the production output root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.planA_k_gpcca import (  # noqa: E402
    DEFAULT_TMP_ROOT,
    FULL_M2_5_PRODUCTION_ROOT,
    M1_BY_SLICE_ROOT,
    M2_BY_SLICE_ROOT,
    FullM25Params,
    run_full_m2_5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=FULL_M2_5_PRODUCTION_ROOT)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--m1-root", type=Path, default=M1_BY_SLICE_ROOT)
    parser.add_argument("--m2-root", type=Path, default=M2_BY_SLICE_ROOT)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-test", action="store_true", default=False)
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--max-anchors-per-slice", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--n-pca-components", type=int, default=30)
    parser.add_argument("--target-mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--min-metaniches-per-slice", type=int, default=50)
    parser.add_argument("--max-metaniches-per-slice", type=int, default=150)
    parser.add_argument("--base-metaniches-per-slice", type=int, default=100)
    parser.add_argument("--tmp-dir", type=Path, default=DEFAULT_TMP_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = FullM25Params(
        feature_lock=args.feature_lock,
        output_root=args.output_root,
        seed=args.seed,
        m1_root=args.m1_root,
        m2_root=args.m2_root,
        dry_run=args.dry_run,
        smoke_test=args.smoke_test,
        max_slices=args.max_slices,
        max_anchors_per_slice=args.max_anchors_per_slice,
        overwrite=args.overwrite,
        resume=args.resume,
        n_pca_components=args.n_pca_components,
        target_mode=args.target_mode,
        min_metaniches_per_slice=args.min_metaniches_per_slice,
        max_metaniches_per_slice=args.max_metaniches_per_slice,
        base_metaniches_per_slice=args.base_metaniches_per_slice,
        tmp_dir=args.tmp_dir,
    )
    payload = run_full_m2_5(params)
    if payload.get("blockers"):
        print(f"status=blocked blockers={payload['blockers']}")
    else:
        print(f"status={payload.get('status', 'dry_run_ok')}")
    print(f"output_root={payload.get('output_root', args.output_root)}")
    print(f"production_executed={payload.get('production_executed', False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
