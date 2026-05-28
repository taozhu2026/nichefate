#!/usr/bin/env python
"""Call validated DARLIN-style joint clones using the benchmark-backed flow."""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().with_name("planC_l126_darlin_joint_clone_niche_v1.py")),
        run_name="__main__",
    )
