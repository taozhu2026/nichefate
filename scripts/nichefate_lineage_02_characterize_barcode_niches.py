#!/usr/bin/env python
"""Characterize barcode-supported niches using the benchmark-backed flow."""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().with_name("planC_l126_full_barcode_niche_characterization.py")),
        run_name="__main__",
    )
