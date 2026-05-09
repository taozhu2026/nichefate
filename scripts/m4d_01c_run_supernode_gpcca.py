#!/usr/bin/env python
"""Run M4D-01c standard pyGPCCA on a supernode Markov chain."""

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
from nichefate.m4d_standard_gpcca import run_m4d_01c  # noqa: E402


DEFAULT_CONFIG = "configs/m4d_markov_macrostate_visualization.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true", help="Reuse existing outputs only when they pass QC.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of existing M4D-01c outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    result = run_m4d_01c(config, resume=bool(args.resume), overwrite=bool(args.overwrite))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
