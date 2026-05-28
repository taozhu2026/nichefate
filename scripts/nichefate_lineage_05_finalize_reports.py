#!/usr/bin/env python
"""Finalize the lineage-aware benchmark reports."""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().with_name("planC_l126_full_characterization_finalize.py")),
        run_name="__main__",
    )
