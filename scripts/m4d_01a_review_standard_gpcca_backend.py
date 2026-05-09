#!/usr/bin/env python
"""Write M4D-01a standard GPCCA backend plan and recommendation reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config
from nichefate.m4d_supernode import (
    configured_paths,
    inspect_backend_availability,
    m4d_output_paths,
    selected_backend_label,
    write_standard_gpcca_review_reports,
)


DEFAULT_CONFIG = "configs/m4d_markov_macrostate_visualization.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = configured_paths(config)
    outputs = m4d_output_paths(paths)

    backend = inspect_backend_availability()
    selected_backend, result_label, true_gpcca_backend_available = selected_backend_label(backend)
    write_standard_gpcca_review_reports(outputs, backend)
    print(
        "M4D-01a standard GPCCA backend review complete: "
        f"{selected_backend} ({result_label}); "
        f"true_gpcca_backend_available={bool(true_gpcca_backend_available)}"
    )


if __name__ == "__main__":
    main()
