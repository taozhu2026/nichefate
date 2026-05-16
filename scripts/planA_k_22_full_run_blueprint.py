#!/usr/bin/env python
"""Write the non-executed PlanA-K full production run blueprint.

This script is dry-run by default. It only writes the command blueprint under
reports/planA_k_production_preflight/ and never launches production work.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.planA_k_gpcca import (  # noqa: E402
    DECISION_LABELS,
    PRODUCTION_PREFLIGHT_ROOT,
    write_full_run_blueprint_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PRODUCTION_PREFLIGHT_ROOT)
    parser.add_argument(
        "--decision-label",
        choices=sorted(DECISION_LABELS),
        default="DIRECT_FULL_RUN_READY_WITH_RESOURCE_CAUTION",
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        raise SystemExit("Refusing --no-dry-run: this blueprint script must not execute production work.")
    payload = write_full_run_blueprint_outputs(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        decision_label=args.decision_label,
    )
    print(f"decision_label={payload['decision_label']}")
    print(f"next_safe_command={payload['next_safe_command']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
