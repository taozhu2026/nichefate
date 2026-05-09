#!/usr/bin/env python
"""Generate M4V-01 visualizations for M4C baseline and M4D GPCCA outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nichefate.io import load_config  # noqa: E402
from nichefate.m4v_markov_visualization import run_m4v_01  # noqa: E402


DEFAULT_CONFIG = "configs/m4d_markov_macrostate_visualization.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true", help="Reuse valid existing visualization outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of existing M4V-01 outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    result = run_m4v_01(config, resume=bool(args.resume), overwrite=bool(args.overwrite))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
