#!/usr/bin/env python
"""Build the generic lineage evidence adapter using the benchmark-backed flow."""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().with_name("planC_l126_barcode_adapter_round1.py")),
        run_name="__main__",
    )
