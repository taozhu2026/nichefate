#!/usr/bin/env python
"""Integrate validated joint clones into spatial niche summaries."""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().with_name("planC_l126_darlin_joint_clone_niche_v1.py")),
        run_name="__main__",
    )
