#!/usr/bin/env python
"""Generate PlanA-K full production preflight reports and draft configs.

This script is dry-run by default. It writes lightweight metadata reports under
reports/planA_k_production_preflight/ and draft configs under configs/planA_k/.
It does not run full M2.5, GPCCA, Slurm, DARLIN, or BranchSBM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.planA_k_gpcca import (  # noqa: E402
    PRODUCTION_PREFLIGHT_ROOT,
    PROJECT_ROOT as DEFAULT_PROJECT_ROOT,
    write_production_preflight_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PRODUCTION_PREFLIGHT_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_PROJECT_ROOT / "configs" / "planA_k")
    parser.add_argument("--bounded-join-sample-rows", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = write_production_preflight_outputs(
        output_dir=args.output_dir,
        config_dir=args.config_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        bounded_join_sample_rows=args.bounded_join_sample_rows,
    )
    print(f"decision_label={payload['decision_label']}")
    print(f"next_safe_command={payload['next_safe_command']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
